"""Correctness fixtures for the dense GGUF Q4_K pack8 GEMV decode (P9.B4).

Covers four ``(scalar_in_t, scalar_out_t)`` instantiations:

* BF16/BF16 and FP16/FP16 -- attention QKV/O surfaces in qwen35moe.
* BF16/F32 and FP16/F32 -- lm-head logits projection (F32 output feeds the
  sampler).

Each instantiation is validated against
``kernels/cpu_reference/ops.py::gguf_quant_gemv`` on synthetic
``make_q4_k_weight``-generated blocks across realistic Qwen3.5-family shapes
(``in_features`` in {256, 512, 1024, 2048, 4096}; ``out_features`` in {16,
256, 512, 2048, 4096}). Tolerances are ``atol=1e-3, rtol=1e-2`` for the
half-precision output variants and tightened to ``atol=5e-3, rtol=5e-3`` for
the F32 lm-head case to make the bit-exact regime explicit.
"""

from __future__ import annotations

import ctypes
import importlib

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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_pack8_gemv import (
    build_gguf_q4_k_pack8_gemv,
    gguf_q4_k_pack8_gemv_decode_bf16_bf16_out,
    gguf_q4_k_pack8_gemv_decode_bf16_f32_out,
    gguf_q4_k_pack8_gemv_decode_fp16_f32_out,
    gguf_q4_k_pack8_gemv_decode_fp16_fp16_out,
    plan_gguf_q4_k_pack8_gemv_build,
    register_gguf_q4_k_pack8_gemv_kernels,
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
def q4_k_dense_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_pack8_gemv(load=True)


# ---------------------------------------------------------------------------
# No-GPU registry and wrapper-contract surface.
# ---------------------------------------------------------------------------


def test_p9_b4_registry_keys_resolve() -> None:
    register_gguf_q4_k_pack8_gemv_kernels()
    for variant in (
        "pack8_gemv_decode_bf16_bf16_out",
        "pack8_gemv_decode_fp16_fp16_out",
        "pack8_gemv_decode_bf16_f32_out",
        "pack8_gemv_decode_fp16_f32_out",
    ):
        fn = resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant=variant,
        )
        assert fn is not None, f"missing registry entry: {variant}"


def test_p9_b4_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_q4_k_pack8_gemv_build()
    assert plan.output_path.name == "gguf_q4_k_pack8_gemv.so"
    assert plan.sources[0].name == "gguf_q4_k_pack8_gemv.hip"


@pytest.mark.parametrize(
    "fn,rows,in_features,out_features,expected_error",
    [
        (gguf_q4_k_pack8_gemv_decode_bf16_bf16_out, 0, 256, 256, "rows must be positive"),
        (
            gguf_q4_k_pack8_gemv_decode_bf16_bf16_out,
            1,
            255,
            256,
            "in_features must be divisible by GGUF Q4_K block size 256",
        ),
        (
            gguf_q4_k_pack8_gemv_decode_bf16_bf16_out,
            1,
            256,
            17,
            r"out_features must be a multiple of 8 \(pack8 lane\)",
        ),
        (gguf_q4_k_pack8_gemv_decode_fp16_f32_out, 1, 256, 0, "out_features must be positive"),
    ],
)
def test_p9_b4_wrapper_validates_args(
    fn,
    rows: int,
    in_features: int,
    out_features: int,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        fn(0, 0, 0, rows, in_features, out_features)


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


def _run_q4_k_dense(
    fn,
    x_dev: np.ndarray,
    qweight: np.ndarray,
    rows: int,
    in_features: int,
    out_features: int,
    out_dtype: np.dtype,
    library,
):
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
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


_HALF_TOL = dict(atol=1.0e-3, rtol=1.0e-2)
_F32_TOL = dict(atol=5.0e-3, rtol=5.0e-3)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 256),
        (1, 512, 512),
        (1, 1024, 2048),
        (1, 2048, 4096),
        (1, 4096, 4096),
        (1, 256, 16),  # tile-boundary out_features
        (4, 512, 512),
        (8, 1024, 2048),
    ],
)
def test_p9_b4_bf16_bf16_matches_cpu_oracle(
    rows: int,
    in_features: int,
    out_features: int,
    q4_k_dense_library,
) -> None:
    rng = np.random.default_rng(rows * 31 + in_features + out_features)
    qweight = make_q4_k_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_q4_k_dense(
        gguf_q4_k_pack8_gemv_decode_bf16_bf16_out,
        x_bf16,
        qweight,
        rows,
        in_features,
        out_features,
        np.uint16,
        q4_k_dense_library,
    )
    actual_f32 = _bf16_u16_to_f32(actual)
    expected_f32 = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q4_K)
    # The kernel writes BF16; round the float reference through BF16 to make
    # the comparison about kernel math (and not the output dtype itself).
    expected_bf16_f32 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected_f32))
    np.testing.assert_allclose(actual_f32, expected_bf16_f32, **_HALF_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 256),
        (1, 1024, 2048),
        (1, 4096, 4096),
        (4, 2048, 512),
    ],
)
def test_p9_b4_fp16_fp16_matches_cpu_oracle(
    rows: int,
    in_features: int,
    out_features: int,
    q4_k_dense_library,
) -> None:
    rng = np.random.default_rng(rows + in_features * 7 + out_features)
    qweight = make_q4_k_weight(out_features, in_features)
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)
    x_ref = x_f16.astype(np.float32)
    actual = _run_q4_k_dense(
        gguf_q4_k_pack8_gemv_decode_fp16_fp16_out,
        x_f16,
        qweight,
        rows,
        in_features,
        out_features,
        np.float16,
        q4_k_dense_library,
    )
    expected_f32 = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q4_K)
    expected_f16 = expected_f32.astype(np.float16)
    np.testing.assert_allclose(actual.astype(np.float32), expected_f16.astype(np.float32), **_HALF_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 2048),     # tiny lm-head
        (1, 2048, 32_768),  # qwen35moe-ish lm-head vocab subset
        (4, 2048, 4096),    # multi-row F32 path
    ],
)
def test_p9_b4_bf16_f32_lm_head_matches_cpu_oracle(
    rows: int,
    in_features: int,
    out_features: int,
    q4_k_dense_library,
) -> None:
    rng = np.random.default_rng(rows * 11 + in_features + out_features)
    qweight = make_q4_k_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_q4_k_dense(
        gguf_q4_k_pack8_gemv_decode_bf16_f32_out,
        x_bf16,
        qweight,
        rows,
        in_features,
        out_features,
        np.float32,
        q4_k_dense_library,
    )
    expected_f32 = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q4_K)
    np.testing.assert_allclose(actual, expected_f32, **_F32_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 2048),
        (1, 2048, 4096),
    ],
)
def test_p9_b4_fp16_f32_lm_head_matches_cpu_oracle(
    rows: int,
    in_features: int,
    out_features: int,
    q4_k_dense_library,
) -> None:
    rng = np.random.default_rng(rows * 17 + in_features + out_features * 3)
    qweight = make_q4_k_weight(out_features, in_features)
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)
    x_ref = x_f16.astype(np.float32)
    actual = _run_q4_k_dense(
        gguf_q4_k_pack8_gemv_decode_fp16_f32_out,
        x_f16,
        qweight,
        rows,
        in_features,
        out_features,
        np.float32,
        q4_k_dense_library,
    )
    expected_f32 = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q4_K)
    np.testing.assert_allclose(actual, expected_f32, **_F32_TOL)
