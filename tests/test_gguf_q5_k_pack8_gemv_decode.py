"""Exact dense GGUF Q5_K pack8 decode coverage for Laguna Q2 XL."""

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
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q5_k_pack8_gemv_bf16_bf16_out,
    gguf_q5_k_pack8_gemv_bf16_f32_out,
    gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
    gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from tests._gguf_synthetic_weights import make_q5_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _f32_to_bf16_u16(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    rounded = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
    rounded[np.isnan(f32)] = 0x7FC0
    return rounded.reshape(f32.shape)


def _run(
    fn,
    x: np.ndarray,
    qweight: np.ndarray,
    *,
    out_dtype: np.dtype,
    library,
) -> np.ndarray:
    rows, in_features = x.shape
    out_features = qweight.shape[0]
    x_buf = malloc(x.nbytes)
    qweight_buf = malloc(qweight.nbytes)
    out = np.empty((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out.nbytes)
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        copy_host_to_device(qweight_buf, host_array_ptr(qweight), qweight.nbytes)
        fn(
            x_buf.ptr,
            qweight_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        return out
    finally:
        for buffer in (out_buf, qweight_buf, x_buf):
            free(buffer)


def test_q5_k_pack8_decode_registry_keys_resolve() -> None:
    register_gguf_k_gemv_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k",
        variant="pack8_gemv_decode_bf16_bf16_out",
    ) is gguf_q5_k_pack8_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k",
        variant="pack8_gemv_decode_bf16_f32_out",
    ) is gguf_q5_k_pack8_gemv_decode_bf16_f32_out


def test_q5_k_pack8_decode_wrapper_rejects_unaligned_shape() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out(
            1,
            2,
            3,
            rows=1,
            in_features=256,
            out_features=7,
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("out_features,in_features", [(48, 3072), (72, 3072), (1024, 3072)])
def test_q5_k_pack8_decode_is_bit_exact_to_existing_raw_pack8(
    out_features: int,
    in_features: int,
) -> None:
    rng = np.random.default_rng(20260724 + out_features)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(1, in_features)))
    qweight = make_q5_k_weight(out_features, in_features)
    library = build_gguf_k_gemv(load=True)

    baseline_bf16 = _run(
        gguf_q5_k_pack8_gemv_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    actual_bf16 = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    baseline_f32 = _run(
        gguf_q5_k_pack8_gemv_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )
    actual_f32 = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )

    np.testing.assert_array_equal(actual_bf16, baseline_bf16)
    np.testing.assert_array_equal(actual_f32, baseline_f32)
