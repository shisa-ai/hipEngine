"""Correctness fixtures for the selected GGUF Q4_K dual pack8 GEMV decode (P9.B1).

Covers the compact-MoE ABI introduced in P8.4:

* ``x[compact_rows, in_features]``
* ``expert_start_compact[num_experts + 1]``
* raw Q4_K expert weights ``[num_experts, out_features, row_bytes]``
* row-major concatenated output ``[compact_rows, out_features_a + out_features_b]``

The CPU oracle assembles per-expert ``gguf_quant_gemv(..., Q4_K)`` calls in
compact-row order, matching the kernel's expert lookup via the linear scan
over ``expert_start_compact``. Coverage spans single/multi-expert layouts,
uneven row counts, empty experts at the start/middle/tail, the all-empty
case (skipped: the kernel is never launched with ``compact_rows == 0``;
exercised at the wrapper level instead), and tile-boundary out widths.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_pack8_gemv import (
    build_gguf_q4_k_selected_pack8_gemv,
    gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out,
    gguf_q4_k_selected_dual_pack8_gemv_decode_compact_fp16_fp16_out,
    plan_gguf_q4_k_selected_pack8_gemv_build,
    register_gguf_q4_k_selected_pack8_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def q4_k_selected_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_selected_pack8_gemv(load=True)


# ---------------------------------------------------------------------------
# No-GPU surface.
# ---------------------------------------------------------------------------


def test_p9_b1_registry_keys_resolve() -> None:
    register_gguf_q4_k_selected_pack8_gemv_kernels()
    for variant in (
        "selected_dual_pack8_gemv_decode_compact_bf16_bf16_out",
        "selected_dual_pack8_gemv_decode_compact_fp16_fp16_out",
        "selected_dual_pack8_gemv_decode_bf16_bf16_out",
        "selected_dual_pack8_gemv_decode_fp16_fp16_out",
    ):
        fn = resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant=variant,
        )
        assert fn is not None, f"missing registry entry: {variant}"


def test_p9_b1_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_q4_k_selected_pack8_gemv_build()
    assert plan.output_path.name == "gguf_q4_k_selected_pack8_gemv.so"


def test_p9_b1_wrapper_validates_args() -> None:
    with pytest.raises(ValueError, match="compact_rows must be positive"):
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out(
            0, 0, 0, 0, 0, 0, 256, 16, 16, 1,
        )
    with pytest.raises(ValueError, match="in_features must be divisible by GGUF Q4_K block size 256"):
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out(
            0, 0, 0, 0, 0, 1, 255, 16, 16, 1,
        )
    with pytest.raises(ValueError, match=r"out_features_a must be a multiple of 8 \(pack8 lane\)"):
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out(
            0, 0, 0, 0, 0, 1, 256, 17, 16, 1,
        )
    with pytest.raises(ValueError, match=r"out_features_b must be a multiple of 8 \(pack8 lane\)"):
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out(
            0, 0, 0, 0, 0, 1, 256, 16, 17, 1,
        )
    with pytest.raises(ValueError, match="num_experts must be positive"):
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out(
            0, 0, 0, 0, 0, 1, 256, 16, 16, 0,
        )


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


def _build_expert_weights(out_features: int, in_features: int, num_experts: int, seed: int) -> np.ndarray:
    """Build a rank-3 ``[num_experts, out_features, row_bytes]`` Q4_K tensor.

    Each expert's bytes are a rolled copy of a base block tensor so the
    selected kernel must read the right expert slice (a degenerate "all
    experts identical" tensor would mask bugs).
    """

    base = make_q4_k_weight(out_features, in_features)
    return np.stack([np.roll(base, shift=e + seed, axis=0) for e in range(num_experts)], axis=0)


def _expected_dual(
    x_ref: np.ndarray,
    expert_start: np.ndarray,
    qa: np.ndarray,
    qb: np.ndarray,
    out_features_a: int,
    out_features_b: int,
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    out = np.zeros((compact_rows, out_features_a + out_features_b), dtype=np.float32)
    for e in range(len(expert_start) - 1):
        s, sl = int(expert_start[e]), int(expert_start[e + 1])
        if sl == s:
            continue
        out[s:sl, :out_features_a] = gguf_quant_gemv(x_ref[s:sl], qa[e], GGMLQuantizationType.Q4_K)
        out[s:sl, out_features_a:] = gguf_quant_gemv(x_ref[s:sl], qb[e], GGMLQuantizationType.Q4_K)
    return out


def _run_dual(
    fn,
    x_dev: np.ndarray,
    expert_start: np.ndarray,
    qa: np.ndarray,
    qb: np.ndarray,
    out_features_a: int,
    out_features_b: int,
    out_dtype: np.dtype,
    library,
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    es_buf = malloc(expert_start.nbytes)
    copy_host_to_device(es_buf, host_array_ptr(expert_start), expert_start.nbytes)
    qa_buf = malloc(qa.nbytes)
    copy_host_to_device(qa_buf, host_array_ptr(qa), qa.nbytes)
    qb_buf = malloc(qb.nbytes)
    copy_host_to_device(qb_buf, host_array_ptr(qb), qb.nbytes)
    out_arr = np.zeros((compact_rows, out_features_a + out_features_b), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            es_buf.ptr,
            qa_buf.ptr,
            qb_buf.ptr,
            out_buf.ptr,
            compact_rows,
            in_features,
            out_features_a,
            out_features_b,
            qa.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, es_buf, qa_buf, qb_buf, out_buf):
            free(b)


_TOL = dict(atol=1.0e-3, rtol=1.0e-2)


_EXPERT_LAYOUTS = [
    pytest.param([8], id="single-expert-rows=8"),
    pytest.param([1], id="single-row-rows=1"),
    pytest.param([3, 5], id="two-expert-uneven"),
    pytest.param([0, 8], id="empty-expert-start"),
    pytest.param([4, 0, 4], id="empty-expert-middle"),
    pytest.param([8, 0], id="empty-expert-tail"),
    pytest.param([2, 0, 0, 3, 0, 5], id="six-expert-uneven-empties"),
    pytest.param([1, 1, 1, 1, 1, 1, 1, 1], id="qwen35moe-top_k=8-decode"),
]


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("counts", _EXPERT_LAYOUTS)
@pytest.mark.parametrize(
    "in_features,out_features_a,out_features_b",
    [
        (256, 16, 16),
        (512, 256, 256),
        (1024, 512, 2048),
        (2048, 2048, 4096),
    ],
)
def test_p9_b1_bf16_bf16_compact_matches_cpu_oracle(
    counts: list[int],
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    q4_k_selected_library,
) -> None:
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(num_experts * 101 + in_features + out_features_a + out_features_b + compact_rows * 7)
    qa = _build_expert_weights(out_features_a, in_features, num_experts, seed=1)
    qb = _build_expert_weights(out_features_b, in_features, num_experts, seed=2)
    x = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_dual(
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out,
        x_bf16, expert_start, qa, qb, out_features_a, out_features_b, np.uint16, q4_k_selected_library,
    )
    actual_f32 = _bf16_u16_to_f32(actual)
    expected = _expected_dual(x_ref, expert_start, qa, qb, out_features_a, out_features_b)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(actual_f32, expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("counts", _EXPERT_LAYOUTS[:5])
@pytest.mark.parametrize(
    "in_features,out_features_a,out_features_b",
    [(512, 256, 256), (1024, 1024, 1024)],
)
def test_p9_b1_fp16_fp16_compact_matches_cpu_oracle(
    counts: list[int],
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    q4_k_selected_library,
) -> None:
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(num_experts * 211 + in_features + out_features_a + out_features_b)
    qa = _build_expert_weights(out_features_a, in_features, num_experts, seed=3)
    qb = _build_expert_weights(out_features_b, in_features, num_experts, seed=5)
    x_f16 = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float16)
    x_ref = x_f16.astype(np.float32)
    actual = _run_dual(
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_fp16_fp16_out,
        x_f16, expert_start, qa, qb, out_features_a, out_features_b, np.float16, q4_k_selected_library,
    )
    expected = _expected_dual(x_ref, expert_start, qa, qb, out_features_a, out_features_b)
    expected_f16 = expected.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(actual.astype(np.float32), expected_f16, **_TOL)
