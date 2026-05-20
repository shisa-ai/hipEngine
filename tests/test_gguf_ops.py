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
    gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight,
    gguf_qwen35_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight,
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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_ops_qwen35_f32_weight_head_rmsnorm_rope() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_ops(load=True)
    query = np.asarray([[1.0, -2.0, 0.5, 3.0], [-1.5, 0.25, 2.5, -0.75]], dtype=np.float32)
    key = np.asarray([[0.75, -1.25, 1.5, -2.0]], dtype=np.float32)
    q_weight = np.asarray([0.1, -0.2, 0.05, 0.3], dtype=np.float32)
    k_weight = np.asarray([-0.15, 0.25, -0.05, 0.2], dtype=np.float32)
    cos, sin = _rope_tables(max_positions=3, rotary_dim=4, base=10000.0)
    position = np.asarray([2], dtype=np.int64)
    query_out = np.empty_like(query)
    key_out = np.empty_like(key)
    query_out_bf16_key = np.empty_like(query)
    key_out_bf16_key = np.empty_like(key)
    key_bf16 = float_array_to_bf16_bits(key)
    bufs = []
    try:
        dq = _dev(query, runtime, bufs)
        dk = _dev(key, runtime, bufs)
        dk_bf16 = _dev(key_bf16, runtime, bufs)
        dqw = _dev(q_weight, runtime, bufs)
        dkw = _dev(k_weight, runtime, bufs)
        dcos = _dev(cos, runtime, bufs)
        dsin = _dev(sin, runtime, bufs)
        dpos = _dev(position, runtime, bufs)
        dqo = malloc(query_out.nbytes, runtime=runtime)
        dko = malloc(key_out.nbytes, runtime=runtime)
        dqo_bf16_key = malloc(query_out_bf16_key.nbytes, runtime=runtime)
        dko_bf16_key = malloc(key_out_bf16_key.nbytes, runtime=runtime)
        bufs.extend((dqo, dko, dqo_bf16_key, dko_bf16_key))
        gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight(
            dq.ptr,
            dk.ptr,
            dqw.ptr,
            dkw.ptr,
            dcos.ptr,
            dsin.ptr,
            dpos.ptr,
            dqo.ptr,
            dko.ptr,
            1.0e-6,
            2,
            1,
            4,
            4,
            3,
            library=library,
            runtime=runtime,
        )
        gguf_qwen35_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight(
            dq.ptr,
            dk_bf16.ptr,
            dqw.ptr,
            dkw.ptr,
            dcos.ptr,
            dsin.ptr,
            dpos.ptr,
            dqo_bf16_key.ptr,
            dko_bf16_key.ptr,
            1.0e-6,
            2,
            1,
            4,
            4,
            3,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(query_out), dqo, runtime=runtime)
        copy_device_to_host(host_array_ptr(key_out), dko, runtime=runtime)
        copy_device_to_host(host_array_ptr(query_out_bf16_key), dqo_bf16_key, runtime=runtime)
        copy_device_to_host(host_array_ptr(key_out_bf16_key), dko_bf16_key, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    expected_query = _apply_rope(_rmsnorm_offset(query, q_weight), cos[2], sin[2], 4)
    expected_key = _apply_rope(_rmsnorm_offset(key, k_weight), cos[2], sin[2], 4)
    expected_key_bf16_input = _apply_rope(
        _rmsnorm_offset(bf16_to_float32(key_bf16), k_weight), cos[2], sin[2], 4
    )
    np.testing.assert_allclose(query_out, expected_query, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(key_out, expected_key, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(query_out_bf16_key, expected_query, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(key_out_bf16_key, expected_key_bf16_input, rtol=1e-6, atol=1e-6)


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


def _rmsnorm_offset(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    inv_rms = 1.0 / np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + 1.0e-6)
    return x.astype(np.float32) * inv_rms * (1.0 + weight.astype(np.float32))


def _apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, rotary_dim: int) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    half = rotary_dim // 2
    first = out[..., :half].copy()
    second = out[..., half:rotary_dim].copy()
    out[..., :half] = first * cos[:half] - second * sin[:half]
    out[..., half:rotary_dim] = second * cos[half:rotary_dim] + first * sin[half:rotary_dim]
    return out


def _rope_tables(*, max_positions: int, rotary_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(base), -2.0 * dims / np.float32(rotary_dim))
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32, copy=False)
    sin = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)
