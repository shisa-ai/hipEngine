"""Correctness fixtures for the selected GGUF Q5_K/Q6_K pack8 GEMV decode (P9.B2).

Mirrors ``test_gguf_q4_k_selected_dual_pack8_gemv_decode.py`` but for the
single-output selected GEMV decode used by the qwen35moe down projection.
Tests both Q5_K and Q6_K under the same compact-MoE ABI (``x`` compact slab,
``expert_start_compact[E+1]``, raw rank-3 expert weights). Coverage spans
single/multi-expert layouts, uneven row counts, empty experts at every
position, and non-multiple-of-16 / non-multiple-of-256 out widths (only the
P9.B2 GEMV constraint ``out_features % 8 == 0`` applies; the WMMA prefill's
multiple-of-16 constraint does not).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_selected_pack8_gemv import (
    build_gguf_k_selected_pack8_gemv,
    gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out,
    gguf_q5_k_selected_pack8_gemv_decode_compact_fp16_fp16_out,
    gguf_q6_k_selected_pack8_gemv_decode_compact_bf16_bf16_out,
    gguf_q6_k_selected_pack8_gemv_decode_compact_fp16_fp16_out,
    plan_gguf_k_selected_pack8_gemv_build,
    register_gguf_k_selected_pack8_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q5_k_weight, make_q6_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def k_selected_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_k_selected_pack8_gemv(load=True)


# ---------------------------------------------------------------------------
# No-GPU surface.
# ---------------------------------------------------------------------------


def test_p9_b2_registry_keys_resolve() -> None:
    register_gguf_k_selected_pack8_gemv_kernels()
    for quant in ("gguf_q5_k", "gguf_q6_k"):
        for variant in (
            "selected_pack8_gemv_decode_compact_bf16_bf16_out",
            "selected_pack8_gemv_decode_compact_fp16_fp16_out",
            "selected_pack8_gemv_decode_bf16_bf16_out",
            "selected_pack8_gemv_decode_fp16_fp16_out",
        ):
            fn = resolve(backend="hip_gfx1100", layer="moe_linear", quant=quant, variant=variant)
            assert fn is not None, f"missing registry entry: {quant} / {variant}"


def test_p9_b2_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_k_selected_pack8_gemv_build()
    assert plan.output_path.name == "gguf_k_selected_pack8_gemv.so"


def test_p9_b2_wrapper_validates_args() -> None:
    with pytest.raises(ValueError, match="compact_rows must be positive"):
        gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 0, 256, 16, 1)
    with pytest.raises(ValueError, match="in_features must be divisible by GGUF Q5_K/Q6_K block size 256"):
        gguf_q6_k_selected_pack8_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 1, 255, 16, 1)
    with pytest.raises(ValueError, match=r"out_features must be a multiple of 8 \(pack8 lane\)"):
        gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 1, 256, 17, 1)


# ---------------------------------------------------------------------------
# Correctness vs CPU oracle.
# ---------------------------------------------------------------------------


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _stack_experts(builder, out_features: int, in_features: int, num_experts: int, seed: int):
    base = builder(out_features, in_features)
    return np.stack([np.roll(base, shift=e + seed, axis=0) for e in range(num_experts)], axis=0)


def _expected_single(
    x_ref: np.ndarray,
    expert_start: np.ndarray,
    qw: np.ndarray,
    out_features: int,
    qtype_enum: GGMLQuantizationType,
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    out = np.zeros((compact_rows, out_features), dtype=np.float32)
    for e in range(len(expert_start) - 1):
        s, sl = int(expert_start[e]), int(expert_start[e + 1])
        if sl == s:
            continue
        out[s:sl] = gguf_quant_gemv(x_ref[s:sl], qw[e], qtype_enum)
    return out


def _run_single(
    fn,
    x_dev: np.ndarray,
    expert_start: np.ndarray,
    qw: np.ndarray,
    out_features: int,
    out_dtype: np.dtype,
    library,
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    es_buf = malloc(expert_start.nbytes)
    copy_host_to_device(es_buf, host_array_ptr(expert_start), expert_start.nbytes)
    w_buf = malloc(qw.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(qw), qw.nbytes)
    out_arr = np.zeros((compact_rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr, es_buf.ptr, w_buf.ptr, out_buf.ptr,
            compact_rows, in_features, out_features, qw.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, es_buf, w_buf, out_buf):
            free(b)


_TOL = dict(atol=1.0e-3, rtol=1.0e-2)


_EXPERT_LAYOUTS = [
    pytest.param([8], id="single-expert"),
    pytest.param([1], id="single-row"),
    pytest.param([3, 5], id="two-uneven"),
    pytest.param([0, 8], id="empty-start"),
    pytest.param([4, 0, 4], id="empty-middle"),
    pytest.param([8, 0], id="empty-tail"),
    pytest.param([1, 1, 1, 1, 1, 1, 1, 1], id="qwen35moe-top_k=8"),
]


_QUANT_CASES = [
    pytest.param(
        "Q5_K",
        make_q5_k_weight,
        gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out,
        gguf_q5_k_selected_pack8_gemv_decode_compact_fp16_fp16_out,
        GGMLQuantizationType.Q5_K,
        id="Q5_K",
    ),
    pytest.param(
        "Q6_K",
        make_q6_k_weight,
        gguf_q6_k_selected_pack8_gemv_decode_compact_bf16_bf16_out,
        gguf_q6_k_selected_pack8_gemv_decode_compact_fp16_fp16_out,
        GGMLQuantizationType.Q6_K,
        id="Q6_K",
    ),
]


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("counts", _EXPERT_LAYOUTS)
@pytest.mark.parametrize(
    "in_features,out_features",
    [
        (256, 8),     # tile-boundary
        (256, 256),
        (512, 512),
        (1024, 2048),
        (2048, 4096),
        (4096, 256),
    ],
)
@pytest.mark.parametrize("name,builder,fn_bf16,_fn_fp16,qtype_enum", _QUANT_CASES)
def test_p9_b2_bf16_bf16_compact_matches_cpu_oracle(
    name: str,
    builder,
    fn_bf16,
    _fn_fp16,
    qtype_enum,
    counts: list[int],
    in_features: int,
    out_features: int,
    k_selected_library,
) -> None:
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(num_experts * 401 + in_features + out_features + hash(name) % 100)
    qw = _stack_experts(builder, out_features, in_features, num_experts, seed=7)
    x = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_single(fn_bf16, x_bf16, expert_start, qw, out_features, np.uint16, k_selected_library)
    actual_f32 = _bf16_u16_to_f32(actual)
    expected = _expected_single(x_ref, expert_start, qw, out_features, qtype_enum)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(actual_f32, expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("counts", _EXPERT_LAYOUTS[:4])
@pytest.mark.parametrize(
    "in_features,out_features",
    [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("name,builder,_fn_bf16,fn_fp16,qtype_enum", _QUANT_CASES)
def test_p9_b2_fp16_fp16_compact_matches_cpu_oracle(
    name: str,
    builder,
    _fn_bf16,
    fn_fp16,
    qtype_enum,
    counts: list[int],
    in_features: int,
    out_features: int,
    k_selected_library,
) -> None:
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(num_experts * 503 + in_features * 3 + out_features + hash(name) % 100)
    qw = _stack_experts(builder, out_features, in_features, num_experts, seed=11)
    x_f16 = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float16)
    x_ref = x_f16.astype(np.float32)
    actual = _run_single(fn_fp16, x_f16, expert_start, qw, out_features, np.float16, k_selected_library)
    expected = _expected_single(x_ref, expert_start, qw, out_features, qtype_enum)
    expected_f16 = expected.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(actual.astype(np.float32), expected_f16, **_TOL)
