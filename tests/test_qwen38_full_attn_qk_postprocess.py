from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.convert import bf16_to_f32, build_cast
from hipengine.kernels.hip_gfx1100.fused import gguf_ops
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import (
    build_qwen35_rotary,
    qwen35_split_qgate_bf16,
)
from hipengine.kernels.registry import resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

_LAYER = "split_qgate+head_rmsnorm+partial_rotary"
_QUANT = "gguf_f32_weight"
_VARIANT = "qwen35_position_qk_bf16_f32"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _candidate():
    fn = getattr(
        gguf_ops,
        "gguf_qwen35_split_qgate_head_rmsnorm_partial_rotary_position_qk_bf16_f32_weight",
        None,
    )
    assert callable(fn), "Qwen3.8 fused Q/K postprocess wrapper is not implemented"
    return fn


def _upload(runtime, buffers, value: np.ndarray):
    array = np.ascontiguousarray(value)
    buffer = malloc(array.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read(runtime, buffer, shape, dtype):
    value = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(value), buffer, runtime=runtime)
    return value


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(bits.shape).copy()


def _cpu_head_norm_rope(
    values: np.ndarray,
    weight: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    eps: float,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    inv_rms = 1.0 / np.sqrt(
        np.mean(source * source, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(eps)
    )
    normalized = source * inv_rms * weight[None, :]
    result = normalized.copy()
    half = normalized.shape[-1] // 2
    result[:, :half] = (
        normalized[:, :half] * cos[:half]
        - normalized[:, half:] * sin[:half]
    )
    result[:, half:] = (
        normalized[:, half:] * cos[half:]
        + normalized[:, :half] * sin[half:]
    )
    return result


def _rope_tables(max_positions: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    freq = 1.0 / (
        10000.0
        ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / float(rotary_dim))
    )
    angles = positions * freq[None, :]
    return (
        np.concatenate((np.cos(angles), np.cos(angles)), axis=1).astype(np.float32),
        np.concatenate((np.sin(angles), np.sin(angles)), axis=1).astype(np.float32),
    )


def test_qk_postprocess_wrapper_is_exposed() -> None:
    assert callable(_candidate())


def test_qk_postprocess_registers_and_routes_only_qualified_gfx1151_shape() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    candidate = _candidate()
    assert resolve(
        backend="hip_gfx1151",
        layer=_LAYER,
        quant=_QUANT,
        variant=_VARIANT,
    ) is candidate

    config = SimpleNamespace(head_count=24, head_count_kv=4, key_length=256)
    qualified = SimpleNamespace(
        backend="hip_gfx1151",
        weights=SimpleNamespace(config=config),
    )
    miss_backend = SimpleNamespace(
        backend="hip_gfx1100",
        weights=SimpleNamespace(config=config),
    )
    miss_shape = SimpleNamespace(
        backend="hip_gfx1151",
        weights=SimpleNamespace(
            config=SimpleNamespace(head_count=8, head_count_kv=2, key_length=256)
        ),
    )
    method = Qwen35GGUFFullStackRunner._full_attn_qk_postprocess_fn
    assert method(qualified) is candidate
    assert method(miss_backend) is None
    assert method(miss_shape) is None


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qk_postprocess_is_bit_exact_and_passes_cpu_kl_top1() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    ops_library = gguf_ops.build_gguf_ops(load=True)
    rotary_library = build_qwen35_rotary(load=True)
    cast_library = build_cast(load=True)
    rng = np.random.default_rng(0x5138A)
    q_heads = 4
    kv_heads = 2
    head_dim = 256
    rotary_dim = 256
    max_positions = 8
    position = np.asarray([5], dtype=np.int64)
    q_projected = float_array_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(q_heads, 2 * head_dim)).astype(np.float32)
    )
    key_bf16 = float_array_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(kv_heads, head_dim)).astype(np.float32)
    )
    q_weight = rng.normal(1.0, 0.1, size=(head_dim,)).astype(np.float32)
    k_weight = rng.normal(1.0, 0.1, size=(head_dim,)).astype(np.float32)
    cos, sin = _rope_tables(max_positions, rotary_dim)
    q_shape = (q_heads, head_dim)
    k_shape = (kv_heads, head_dim)
    buffers = []
    try:
        dq = _upload(runtime, buffers, q_projected)
        dk = _upload(runtime, buffers, key_bf16)
        dqw = _upload(runtime, buffers, q_weight)
        dkw = _upload(runtime, buffers, k_weight)
        dcos = _upload(runtime, buffers, cos)
        dsin = _upload(runtime, buffers, sin)
        dpos = _upload(runtime, buffers, position)
        allocations = [
            malloc(np.prod(q_shape) * 4, runtime=runtime),
            malloc(np.prod(k_shape) * 4, runtime=runtime),
            malloc(np.prod(q_shape) * 2, runtime=runtime),
            malloc(np.prod(q_shape) * 4, runtime=runtime),
            malloc(np.prod(k_shape) * 4, runtime=runtime),
            malloc(np.prod(q_shape) * 2, runtime=runtime),
            malloc(np.prod(q_shape) * 4, runtime=runtime),
            malloc(np.prod(k_shape) * 4, runtime=runtime),
        ]
        buffers.extend(allocations)
        (
            query_raw,
            key_raw,
            gate_control,
            query_control,
            key_control,
            gate_candidate,
            query_candidate,
            key_candidate,
        ) = allocations
        qwen35_split_qgate_bf16(
            dq.ptr,
            query_raw.ptr,
            gate_control.ptr,
            1,
            q_heads,
            head_dim,
            library=rotary_library,
            runtime=runtime,
        )
        bf16_to_f32(
            dk.ptr,
            key_raw.ptr,
            kv_heads * head_dim,
            library=cast_library,
            runtime=runtime,
        )
        gguf_ops.gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight(
            query_raw.ptr,
            key_raw.ptr,
            dqw.ptr,
            dkw.ptr,
            dcos.ptr,
            dsin.ptr,
            dpos.ptr,
            query_control.ptr,
            key_control.ptr,
            1.0e-6,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            max_positions,
            library=ops_library,
            runtime=runtime,
        )
        _candidate()(
            dq.ptr,
            dk.ptr,
            dqw.ptr,
            dkw.ptr,
            dcos.ptr,
            dsin.ptr,
            dpos.ptr,
            query_candidate.ptr,
            key_candidate.ptr,
            gate_candidate.ptr,
            1.0e-6,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            max_positions,
            library=ops_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        q_control = _read(runtime, query_control, q_shape, np.float32)
        k_control = _read(runtime, key_control, k_shape, np.float32)
        g_control = _read(runtime, gate_control, q_shape, np.uint16)
        q_actual = _read(runtime, query_candidate, q_shape, np.float32)
        k_actual = _read(runtime, key_candidate, k_shape, np.float32)
        g_actual = _read(runtime, gate_candidate, q_shape, np.uint16)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(q_actual.view(np.uint32), q_control.view(np.uint32))
    np.testing.assert_array_equal(k_actual.view(np.uint32), k_control.view(np.uint32))
    np.testing.assert_array_equal(g_actual, g_control)
    np.testing.assert_array_equal(
        g_actual,
        q_projected[:, head_dim:],
    )
    expected_q = _cpu_head_norm_rope(
        _bf16_to_f32(q_projected[:, :head_dim]),
        q_weight,
        cos[int(position[0])],
        sin[int(position[0])],
        1.0e-6,
    )
    expected_k = _cpu_head_norm_rope(
        _bf16_to_f32(key_bf16),
        k_weight,
        cos[int(position[0])],
        sin[int(position[0])],
        1.0e-6,
    )
    np.testing.assert_allclose(q_actual, expected_q, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(k_actual, expected_k, rtol=2e-6, atol=2e-6)
    logits = np.concatenate((q_actual.ravel(), k_actual.ravel())).astype(np.float64)
    reference = np.concatenate((expected_q.ravel(), expected_k.ravel())).astype(np.float64)
    logits -= np.max(logits)
    reference -= np.max(reference)
    p = np.exp(reference) / np.sum(np.exp(reference))
    q = np.exp(logits) / np.sum(np.exp(logits))
    kl = float(np.sum(p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300))))
    assert kl <= 0.05
    assert int(np.argmax(logits)) == int(np.argmax(reference))
