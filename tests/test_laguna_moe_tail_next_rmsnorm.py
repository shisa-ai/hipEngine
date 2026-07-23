"""Exact RED/GREEN gate for Laguna aggregate MoE-tail plus next RMSNorm."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import laguna_aggregate_moe_tail_next_rmsnorm
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


requires_hip = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


@pytest.fixture(scope="module")
def _libs():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import build_paro_combine

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return (
            build_paro_combine(load=True, compiler_version=compiler_version),
            build_gguf_ops(load=True, compiler_version=compiler_version),
        )


def _upload(runtime, buffers, array: np.ndarray):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    array = np.ascontiguousarray(array)
    buffer = malloc(max(4, array.nbytes), runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _allocate(runtime, buffers, nbytes: int):
    from hipengine.core.memory import malloc

    buffer = malloc(max(4, nbytes), runtime=runtime)
    buffers.append(buffer)
    return buffer


def _download(runtime, buffer, shape) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    out = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(out), buffer, runtime=runtime)
    return out


def _free_all(runtime, buffers) -> None:
    from hipengine.core.memory import free

    for buffer in reversed(buffers):
        free(buffer, runtime=runtime)


def _bf16_fixture(hidden: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0xD900 + hidden)
    routed = float_array_to_bf16_bits(rng.standard_normal(hidden).astype(np.float32) * 0.7)
    shared = float_array_to_bf16_bits(rng.standard_normal(hidden).astype(np.float32) * 0.7)
    post_attention = float_array_to_bf16_bits(
        rng.standard_normal(hidden).astype(np.float32) * 0.7
    )
    norm_weight = rng.uniform(0.25, 1.75, size=hidden).astype(np.float32)
    return routed, shared, post_attention, norm_weight


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64) - float(np.max(values))
    probs = np.exp(shifted)
    return probs / np.sum(probs)


@pytest.mark.parametrize("hidden", [17, 3_072])
def test_laguna_aggregate_moe_tail_cpu_reference_preserves_two_bf16_boundaries(
    hidden: int,
) -> None:
    routed_bits, shared_bits, post_bits, norm_weight = _bf16_fixture(hidden)
    routed = bf16_to_float32(routed_bits)
    shared = bf16_to_float32(shared_bits)
    post = bf16_to_float32(post_bits)

    hidden_out, norm_out = laguna_aggregate_moe_tail_next_rmsnorm(
        routed,
        shared,
        post,
        norm_weight,
    )

    first_rounded = bf16_to_float32(float_array_to_bf16_bits(routed + shared))
    expected_hidden_bits = float_array_to_bf16_bits(post + first_rounded)
    collapsed_hidden_bits = float_array_to_bf16_bits(post + routed + shared)
    np.testing.assert_array_equal(float_array_to_bf16_bits(hidden_out), expected_hidden_bits)
    assert np.any(expected_hidden_bits != collapsed_hidden_bits)
    assert norm_out.shape == (hidden,)


@requires_hip
@pytest.mark.parametrize("hidden", [17, 3_072])
def test_laguna_aggregate_moe_tail_gpu_is_byte_exact_to_three_kernel_fallback(
    _libs,
    _runtime,
    hidden: int,
) -> None:
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
        gguf_bf16_add,
        gguf_rmsnorm_bf16_f32_weight,
    )
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out,
    )

    combine_lib, gguf_lib = _libs
    routed_bits, shared_bits, post_bits, norm_weight = _bf16_fixture(hidden)
    expected_hidden, expected_norm = laguna_aggregate_moe_tail_next_rmsnorm(
        bf16_to_float32(routed_bits),
        bf16_to_float32(shared_bits),
        bf16_to_float32(post_bits),
        norm_weight,
    )
    expected_hidden_bits = float_array_to_bf16_bits(expected_hidden)
    expected_norm_bits = float_array_to_bf16_bits(expected_norm)

    buffers = []
    try:
        routed_d = _upload(_runtime, buffers, routed_bits)
        shared_d = _upload(_runtime, buffers, shared_bits)
        post_d = _upload(_runtime, buffers, post_bits)
        weight_d = _upload(_runtime, buffers, norm_weight)
        nbytes = hidden * np.dtype(np.uint16).itemsize
        first_add_d = _allocate(_runtime, buffers, nbytes)
        fallback_hidden_d = _allocate(_runtime, buffers, nbytes)
        fallback_norm_d = _allocate(_runtime, buffers, nbytes)
        fused_hidden_d = _allocate(_runtime, buffers, nbytes)
        fused_norm_d = _allocate(_runtime, buffers, nbytes)

        gguf_bf16_add(
            routed_d.ptr,
            shared_d.ptr,
            first_add_d.ptr,
            hidden,
            library=gguf_lib,
            runtime=_runtime,
        )
        gguf_bf16_add(
            post_d.ptr,
            first_add_d.ptr,
            fallback_hidden_d.ptr,
            hidden,
            library=gguf_lib,
            runtime=_runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            fallback_hidden_d.ptr,
            weight_d.ptr,
            fallback_norm_d.ptr,
            1,
            hidden,
            1e-6,
            library=gguf_lib,
            runtime=_runtime,
        )
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out(
            routed_d.ptr,
            shared_d.ptr,
            post_d.ptr,
            weight_d.ptr,
            fused_norm_d.ptr,
            fused_hidden_d.ptr,
            hidden,
            library=combine_lib,
            runtime=_runtime,
        )
        _runtime.stream_synchronize(0)

        fallback_hidden = _download(_runtime, fallback_hidden_d, (hidden,))
        fallback_norm = _download(_runtime, fallback_norm_d, (hidden,))
        fused_hidden = _download(_runtime, fused_hidden_d, (hidden,))
        fused_norm = _download(_runtime, fused_norm_d, (hidden,))
    finally:
        _free_all(_runtime, buffers)

    np.testing.assert_array_equal(fused_hidden, fallback_hidden)
    np.testing.assert_array_equal(fused_norm, fallback_norm)
    np.testing.assert_array_equal(fused_hidden, expected_hidden_bits)
    np.testing.assert_array_equal(fused_norm, expected_norm_bits)

    expected_logits = bf16_to_float32(expected_norm_bits)
    actual_logits = bf16_to_float32(fused_norm)
    expected_probs = _softmax(expected_logits)
    actual_probs = _softmax(actual_logits)
    kl = float(np.sum(expected_probs * np.log(expected_probs / actual_probs)))
    assert kl <= 0.05
    assert int(np.argmax(actual_logits)) == int(np.argmax(expected_logits))


@requires_hip
def test_laguna_aggregate_moe_tail_wrapper_rejects_empty_hidden(_libs, _runtime) -> None:
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out,
    )

    with pytest.raises(ValueError, match="features must be positive"):
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out(
            1,
            2,
            3,
            4,
            5,
            6,
            0,
            library=_libs[0],
            runtime=_runtime,
        )
