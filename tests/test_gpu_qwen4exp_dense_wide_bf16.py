"""Guarded HIP numerical/range checks for the opt-in BF16 dense-wide probe."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _f32_to_bf16(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _dequantize_weight(qweight: np.ndarray, in_features: int) -> np.ndarray:
    blocks = qweight.reshape(qweight.shape[0], in_features // 32, 34)
    scales = np.ascontiguousarray(blocks[:, :, :2]).view(np.float16).reshape(
        blocks.shape[:2]
    ).astype(np.float32)
    codes = blocks[:, :, 2:].view(np.int8).astype(np.float32)
    return (scales[:, :, None] * codes).reshape(qweight.shape[0], in_features)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_bf16_probe_matches_bf16_operand_reference() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    rows, in_features, out_features = 257, 256, 128
    rng = np.random.default_rng(20261006)
    x_f32 = rng.standard_normal((rows, in_features), dtype=np.float32)
    x_bf16 = _f32_to_bf16(x_f32)
    qweight = make_q8_0_weight(out_features, in_features)
    weight = _dequantize_weight(qweight, in_features)
    weight_bf16 = _bf16_to_f32(_f32_to_bf16(weight))
    expected = _bf16_to_f32(x_bf16).astype(np.float64) @ weight_bf16.astype(np.float64).T
    out = np.zeros((rows, out_features), dtype=np.float32)

    runtime = get_hip_runtime()
    library = wide.build_gguf_q8_0_dense_wide(load=True)
    buffers = []
    try:
        dx = malloc(x_bf16.nbytes)
        dw = malloc(qweight.nbytes)
        dout = malloc(out.nbytes)
        buffers.extend((dx, dw, dout))
        copy_host_to_device(dx, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(dw, host_array_ptr(qweight), runtime=runtime)
        wide.gguf_q8_0_dense_wide256_bf16in_f32_f32_out(
            dx.ptr,
            dw.ptr,
            dout.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), dout, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    scale = float(np.max(np.abs(expected)))
    error = float(np.max(np.abs(out.astype(np.float64) - expected)))
    assert error <= 2e-3 * scale, (error, scale)
    assert np.isfinite(out).all()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_bf16_probe_accepts_bf16_range_that_is_outside_f16() -> None:
    """The candidate's declared range advantage is exercised, not inferred."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    rows, in_features, out_features = 256, 256, 128
    x_f32 = np.zeros((rows, in_features), dtype=np.float32)
    x_f32[0, ::2] = np.float32(2.0**100)
    x_f32[0, 1::2] = np.float32(-(2.0**100))
    x_bf16 = _f32_to_bf16(x_f32)
    qweight = make_q8_0_weight(out_features, in_features)
    out = np.zeros((rows, out_features), dtype=np.float32)

    runtime = get_hip_runtime()
    library = wide.build_gguf_q8_0_dense_wide(load=True)
    buffers = []
    try:
        dx = malloc(x_bf16.nbytes)
        dw = malloc(qweight.nbytes)
        dout = malloc(out.nbytes)
        buffers.extend((dx, dw, dout))
        copy_host_to_device(dx, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(dw, host_array_ptr(qweight), runtime=runtime)
        wide.gguf_q8_0_dense_wide256_bf16in_f32_f32_out(
            dx.ptr, dw.ptr, dout.ptr, rows, in_features, out_features,
            library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), dout, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    assert np.isfinite(out).all()
