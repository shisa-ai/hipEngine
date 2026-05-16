from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    build_gguf_ops,
    gguf_add_rmsnorm_bf16_f32_weight,
    gguf_bf16_add,
    gguf_gate_repeat_value_bf16,
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_ops_bf16_add_and_f32_weight_rmsnorm() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_ops(load=True)
    x_f32 = np.asarray([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    y_f32 = np.asarray([[0.5, 0.25, -0.75, 1.25]], dtype=np.float32)
    weight = np.asarray([1.0, 0.5, -1.0, 2.0], dtype=np.float32)
    x = float_array_to_bf16_bits(x_f32)
    y = float_array_to_bf16_bits(y_f32)
    add_out = np.empty_like(x)
    norm_out = np.empty_like(x)
    add_norm = np.empty_like(x)
    residual = np.empty_like(x)
    bufs = []
    try:
        dx = _dev(x, runtime, bufs)
        dy = _dev(y, runtime, bufs)
        dw = _dev(weight, runtime, bufs)
        dadd = malloc(add_out.nbytes, runtime=runtime)
        dnorm = malloc(norm_out.nbytes, runtime=runtime)
        dadd_norm = malloc(add_norm.nbytes, runtime=runtime)
        dres = malloc(residual.nbytes, runtime=runtime)
        bufs.extend((dadd, dnorm, dadd_norm, dres))
        gguf_bf16_add(dx.ptr, dy.ptr, dadd.ptr, x.size, library=library, runtime=runtime)
        gguf_rmsnorm_bf16_f32_weight(
            dx.ptr, dw.ptr, dnorm.ptr, 1, x.shape[1], 1.0e-6, library=library, runtime=runtime
        )
        gguf_add_rmsnorm_bf16_f32_weight(
            dx.ptr,
            dy.ptr,
            dw.ptr,
            dadd_norm.ptr,
            dres.ptr,
            1,
            x.shape[1],
            1.0e-6,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(add_out), dadd, runtime=runtime)
        copy_device_to_host(host_array_ptr(norm_out), dnorm, runtime=runtime)
        copy_device_to_host(host_array_ptr(add_norm), dadd_norm, runtime=runtime)
        copy_device_to_host(host_array_ptr(residual), dres, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    expected_add = bf16_to_float32(float_array_to_bf16_bits(x_f32 + y_f32))
    np.testing.assert_array_equal(bf16_to_float32(add_out), expected_add)
    np.testing.assert_allclose(
        bf16_to_float32(norm_out),
        bf16_to_float32(float_array_to_bf16_bits(_rmsnorm(x_f32, weight))),
    )
    expected_residual = bf16_to_float32(float_array_to_bf16_bits(x_f32 + y_f32))
    np.testing.assert_array_equal(bf16_to_float32(residual), expected_residual)
    np.testing.assert_allclose(
        bf16_to_float32(add_norm),
        bf16_to_float32(float_array_to_bf16_bits(_rmsnorm(x_f32 + y_f32, weight))),
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_ops_gate_repeat_value() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_ops(load=True)
    gate_f32 = np.asarray([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    value_f32 = np.asarray([2.0, -3.0], dtype=np.float32)
    gate = float_array_to_bf16_bits(gate_f32)
    value = float_array_to_bf16_bits(value_f32)
    out = np.empty((4,), dtype=np.uint16)
    bufs = []
    try:
        dg = _dev(gate, runtime, bufs)
        dv = _dev(value, runtime, bufs)
        do = malloc(out.nbytes, runtime=runtime)
        bufs.append(do)
        gguf_gate_repeat_value_bf16(dg.ptr, dv.ptr, do.ptr, 4, 2, 1, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), do, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
    expected = np.asarray(
        [
            _sigmoid(gate_f32[0]) * value_f32[0],
            _sigmoid(gate_f32[1]) * value_f32[0],
            _sigmoid(gate_f32[2]) * value_f32[1],
            _sigmoid(gate_f32[3]) * value_f32[1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(bf16_to_float32(out), bf16_to_float32(float_array_to_bf16_bits(expected)))


def _dev(array: np.ndarray, runtime, bufs: list):
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes, runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(contiguous), runtime=runtime)
    return buf


def _rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    inv_rms = 1.0 / np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + 1.0e-6)
    return x * inv_rms * weight


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))
