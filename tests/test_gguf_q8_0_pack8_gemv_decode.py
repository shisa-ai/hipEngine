"""Correctness fixtures for the dense GGUF Q8_0 pack8 GEMV decode (P9.B3).

Covers both the single-output kernel (drop-in replacement for the legacy
``gguf_k_pack8_prefill_out_kernel<...,8>`` at decode shapes) and the fused
gate+up dual kernel used by the qwen35moe shared-expert decode bundle.

Each instantiation is validated against
``kernels/cpu_reference/ops.py::gguf_quant_gemv`` on synthetic
``make_q8_0_weight``-generated blocks across realistic Qwen3.5-family shapes
(``in_features`` in {32, 256, 512, 1024, 2048, 4096}; ``out_features`` in
{8, 256, 512, 2048, 4096}). Tolerances are ``atol=5e-4, rtol=5e-3`` per
``docs/GGUF.md`` P9 (Q8_0 has the smallest block size of any GGUF K-quant,
so the kernel math hits BF16 output-rounding before it hits dequant drift).
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv import (
    build_gguf_q8_0_pack8_gemv,
    gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out,
    gguf_q8_0_pack8_dual_gate_up_gemv_decode_fp16_fp16_out,
    gguf_q8_0_pack8_gemv_decode_bf16_bf16_out,
    gguf_q8_0_pack8_gemv_decode_fp16_fp16_out,
    plan_gguf_q8_0_pack8_gemv_build,
    register_gguf_q8_0_pack8_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q8_0_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def q8_0_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q8_0_pack8_gemv(load=True)


# ---------------------------------------------------------------------------
# No-GPU surface.
# ---------------------------------------------------------------------------


def test_p9_b3_registry_keys_resolve() -> None:
    register_gguf_q8_0_pack8_gemv_kernels()
    for variant in (
        "pack8_gemv_decode_bf16_bf16_out",
        "pack8_gemv_decode_fp16_fp16_out",
        "pack8_dual_gate_up_gemv_decode_bf16_bf16_out",
        "pack8_dual_gate_up_gemv_decode_fp16_fp16_out",
    ):
        fn = resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0",
            variant=variant,
        )
        assert fn is not None, f"missing registry entry: {variant}"


def test_p9_b3_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_q8_0_pack8_gemv_build()
    assert plan.output_path.name == "gguf_q8_0_pack8_gemv.so"
    assert plan.sources[0].name == "gguf_q8_0_pack8_gemv.hip"


def test_p9_b3_single_wrapper_validates_args() -> None:
    with pytest.raises(ValueError, match="in_features must be divisible by GGUF Q8_0 block size 32"):
        gguf_q8_0_pack8_gemv_decode_bf16_bf16_out(0, 0, 0, 1, 31, 16)
    with pytest.raises(ValueError, match=r"out_features must be a multiple of 8 \(pack8 lane\)"):
        gguf_q8_0_pack8_gemv_decode_bf16_bf16_out(0, 0, 0, 1, 32, 7)
    with pytest.raises(ValueError, match="rows must be positive"):
        gguf_q8_0_pack8_gemv_decode_bf16_bf16_out(0, 0, 0, 0, 32, 8)


def test_p9_b3_dual_wrapper_validates_args() -> None:
    with pytest.raises(ValueError, match="must be multiples of 8"):
        gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out(0, 0, 0, 0, 1, 32, 16, 7)


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


def _run_single(fn, x, qweight, rows, in_features, out_features, out_dtype, library):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    w_buf = malloc(qweight.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(qweight), qweight.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(x_buf.ptr, w_buf.ptr, out_buf.ptr, rows, in_features, out_features, library=library)
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, w_buf, out_buf):
            free(b)


def _run_dual(fn, x, qa, qb, rows, in_features, oa, ob, out_dtype, library):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    a_buf = malloc(qa.nbytes)
    copy_host_to_device(a_buf, host_array_ptr(qa), qa.nbytes)
    b_buf = malloc(qb.nbytes)
    copy_host_to_device(b_buf, host_array_ptr(qb), qb.nbytes)
    out_arr = np.zeros((rows, oa + ob), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(x_buf.ptr, a_buf.ptr, b_buf.ptr, out_buf.ptr, rows, in_features, oa, ob, library=library)
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, a_buf, b_buf, out_buf):
            free(b)


_TOL = dict(atol=5.0e-4, rtol=5.0e-3)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 32, 8),    # smallest valid shape
        (1, 256, 256),
        (1, 512, 512),
        (1, 1024, 2048),
        (1, 2048, 4096),
        (1, 4096, 4096),
        (4, 512, 256),
        (8, 1024, 512),
    ],
)
def test_p9_b3_single_bf16_bf16_matches_cpu_oracle(rows, in_features, out_features, q8_0_library) -> None:
    rng = np.random.default_rng(rows * 7 + in_features + out_features * 3)
    qweight = make_q8_0_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_single(
        gguf_q8_0_pack8_gemv_decode_bf16_bf16_out,
        x_bf16, qweight, rows, in_features, out_features, np.uint16, q8_0_library,
    )
    actual_f32 = _bf16_u16_to_f32(actual)
    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q8_0)
    # The kernel writes BF16; round the f32 reference through BF16 first.
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(actual_f32, expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 512),
        (1, 2048, 2048),
        (4, 1024, 4096),
    ],
)
def test_p9_b3_single_fp16_fp16_matches_cpu_oracle(rows, in_features, out_features, q8_0_library) -> None:
    rng = np.random.default_rng(rows + in_features * 5 + out_features)
    qweight = make_q8_0_weight(out_features, in_features)
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)
    x_ref = x_f16.astype(np.float32)
    actual = _run_single(
        gguf_q8_0_pack8_gemv_decode_fp16_fp16_out,
        x_f16, qweight, rows, in_features, out_features, np.float16, q8_0_library,
    )
    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q8_0)
    expected_f16 = expected.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(actual.astype(np.float32), expected_f16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features_a,out_features_b",
    [
        (1, 256, 8, 8),    # smallest dual
        (1, 512, 256, 512),
        (1, 2048, 1024, 2048),
        (4, 1024, 256, 256),
    ],
)
def test_p9_b3_dual_bf16_bf16_matches_cpu_oracle(
    rows, in_features, out_features_a, out_features_b, q8_0_library,
) -> None:
    rng = np.random.default_rng(rows * 13 + in_features + out_features_a + out_features_b * 2)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_dual(
        gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out,
        x_bf16, qa, qb, rows, in_features, out_features_a, out_features_b, np.uint16, q8_0_library,
    )
    actual_f32 = _bf16_u16_to_f32(actual)
    expected = np.zeros_like(actual_f32)
    expected[:, :out_features_a] = gguf_quant_gemv(x_ref, qa, GGMLQuantizationType.Q8_0)
    expected[:, out_features_a:] = gguf_quant_gemv(x_ref, qb, GGMLQuantizationType.Q8_0)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(actual_f32, expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features_a,out_features_b",
    [
        (1, 512, 256, 256),
        (4, 2048, 1024, 1024),
    ],
)
def test_p9_b3_dual_fp16_fp16_matches_cpu_oracle(
    rows, in_features, out_features_a, out_features_b, q8_0_library,
) -> None:
    rng = np.random.default_rng(rows * 23 + in_features + out_features_a)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)
    x_ref = x_f16.astype(np.float32)
    actual = _run_dual(
        gguf_q8_0_pack8_dual_gate_up_gemv_decode_fp16_fp16_out,
        x_f16, qa, qb, rows, in_features, out_features_a, out_features_b, np.float16, q8_0_library,
    )
    expected = np.zeros((rows, out_features_a + out_features_b), dtype=np.float32)
    expected[:, :out_features_a] = gguf_quant_gemv(x_ref, qa, GGMLQuantizationType.Q8_0)
    expected[:, out_features_a:] = gguf_quant_gemv(x_ref, qb, GGMLQuantizationType.Q8_0)
    expected_f16 = expected.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(actual.astype(np.float32), expected_f16, **_TOL)
