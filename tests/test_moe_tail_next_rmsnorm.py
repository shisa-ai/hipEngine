"""Exact GPU gate for MoE-tail plus next-layer RMSNorm fusion."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import moe_tail_next_rmsnorm
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


@pytest.fixture(scope="module")
def _libs():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import build_paro_combine
    from hipengine.kernels.hip_gfx1100.norm.rmsnorm import build_qwen35_rmsnorm

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return (
            build_paro_combine(load=True, compiler_version=compiler_version),
            build_gguf_ops(load=True, compiler_version=compiler_version),
            build_qwen35_rmsnorm(load=True, compiler_version=compiler_version),
        )


def _encode(values: np.ndarray, activation_dtype: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if activation_dtype == "bf16":
        return float_array_to_bf16_bits(values)
    return np.ascontiguousarray(values, dtype=np.float16).view(np.uint16)


def _decode(bits: np.ndarray, activation_dtype: str) -> np.ndarray:
    if activation_dtype == "bf16":
        return bf16_to_float32(bits)
    return np.ascontiguousarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)


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


def test_cpu_reference_preserves_selected_and_combined_rounding_boundaries() -> None:
    selected = np.asarray([[[0.5, -1.0], [0.5, -1.0]]], dtype=np.float32)
    routing = np.asarray([[1.0, 1.0]], dtype=np.float32)
    residual = np.asarray([[3.0, -1.0]], dtype=np.float32)
    shared = np.asarray([[2.0, 4.0]], dtype=np.float32)
    gate = np.asarray([[0.0, 99.0]], dtype=np.float32)
    weight = np.ones((2,), dtype=np.float32)

    rounded, normed = moe_tail_next_rmsnorm(
        selected,
        shared,
        gate,
        residual,
        weight,
        routing_weights=routing,
        activation_dtype="bf16",
    )

    np.testing.assert_array_equal(rounded, np.asarray([[5.0, -1.0]], dtype=np.float32))
    expected_scale = np.float32(1.0 / np.sqrt(np.mean(rounded * rounded, axis=-1, keepdims=True) + 1e-6))
    expected_norm = bf16_to_float32(float_array_to_bf16_bits(rounded * expected_scale))
    np.testing.assert_array_equal(normed, expected_norm)


@pytest.mark.parametrize("tokens,hidden", [(1, 17), (3, 2048)])
@pytest.mark.parametrize("weighted", [False, True], ids=["aggregate", "slot_weighted"])
@pytest.mark.parametrize("flavor", ["gguf_bf16", "paro_bf16", "paro_fp16"])
def test_fused_moe_tail_next_rmsnorm_matches_unfused(
    _libs,
    _runtime,
    flavor: str,
    weighted: bool,
    tokens: int,
    hidden: int,
) -> None:
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
        shared_gate_combine_residual_batch_out_bf16,
        shared_gate_combine_residual_batch_out_fp16,
        shared_gate_combine_residual_rmsnorm_gguf_bf16_out,
        shared_gate_combine_residual_rmsnorm_paro_bf16_out,
        shared_gate_combine_residual_rmsnorm_paro_fp16_out,
        weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
        weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w,
        weighted_sum_shared_gate_combine_residual_rmsnorm_gguf_bf16_out,
        weighted_sum_shared_gate_combine_residual_rmsnorm_paro_bf16_out,
        weighted_sum_shared_gate_combine_residual_rmsnorm_paro_fp16_out,
    )
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import gguf_rmsnorm_bf16_f32_weight
    from hipengine.kernels.hip_gfx1100.norm.rmsnorm import paro_rmsnorm_out_bf16, paro_rmsnorm_out_fp16

    combine_lib, gguf_lib, norm_lib = _libs
    activation_dtype = "fp16" if flavor == "paro_fp16" else "bf16"
    rng = np.random.default_rng(0x21A0 + tokens * 37 + hidden + 11 * weighted + len(flavor))
    top_k = 8
    gate_stride = 3
    eps = 1e-6
    selected_shape = (tokens, top_k, hidden) if weighted else (tokens, hidden)
    selected_bits = _encode(rng.standard_normal(selected_shape).astype(np.float32) * 0.2, activation_dtype)
    shared_bits = _encode(rng.standard_normal((tokens, hidden)).astype(np.float32) * 0.2, activation_dtype)
    residual_bits = _encode(rng.standard_normal((tokens, hidden)).astype(np.float32) * 0.2, activation_dtype)
    routing = rng.uniform(-0.5, 0.75, size=(tokens, top_k)).astype(np.float32)
    gate_logits = rng.uniform(-4.0, 4.0, size=(tokens, gate_stride)).astype(np.float32)
    if flavor == "gguf_bf16":
        norm_weight = rng.uniform(0.5, 1.5, size=hidden).astype(np.float32)
    else:
        norm_weight = _encode(rng.uniform(0.5, 1.5, size=hidden).astype(np.float32), activation_dtype)

    buffers = []
    try:
        selected_d = _upload(_runtime, buffers, selected_bits)
        shared_d = _upload(_runtime, buffers, shared_bits)
        residual_d = _upload(_runtime, buffers, residual_bits)
        routing_d = _upload(_runtime, buffers, routing)
        gate_d = _upload(_runtime, buffers, gate_logits)
        norm_weight_d = _upload(_runtime, buffers, norm_weight)
        nbytes = tokens * hidden * np.dtype(np.uint16).itemsize
        ref_residual_d = _allocate(_runtime, buffers, nbytes)
        ref_norm_d = _allocate(_runtime, buffers, nbytes)
        fused_residual_d = _allocate(_runtime, buffers, nbytes)
        fused_norm_d = _allocate(_runtime, buffers, nbytes)

        if activation_dtype == "bf16":
            if weighted:
                weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(
                    selected_d.ptr,
                    routing_d.ptr,
                    shared_d.ptr,
                    gate_d.ptr,
                    residual_d.ptr,
                    ref_residual_d.ptr,
                    tokens,
                    top_k,
                    hidden,
                    gate_stride,
                    library=combine_lib,
                    runtime=_runtime,
                )
            else:
                shared_gate_combine_residual_batch_out_bf16(
                    selected_d.ptr,
                    shared_d.ptr,
                    gate_d.ptr,
                    residual_d.ptr,
                    ref_residual_d.ptr,
                    tokens,
                    hidden,
                    gate_stride,
                    library=combine_lib,
                    runtime=_runtime,
                )
        elif weighted:
            weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w(
                selected_d.ptr,
                routing_d.ptr,
                shared_d.ptr,
                gate_d.ptr,
                residual_d.ptr,
                ref_residual_d.ptr,
                tokens,
                top_k,
                hidden,
                gate_stride,
                library=combine_lib,
                runtime=_runtime,
            )
        else:
            shared_gate_combine_residual_batch_out_fp16(
                selected_d.ptr,
                shared_d.ptr,
                gate_d.ptr,
                residual_d.ptr,
                ref_residual_d.ptr,
                tokens,
                hidden,
                gate_stride,
                library=combine_lib,
                runtime=_runtime,
            )

        if flavor == "gguf_bf16":
            gguf_rmsnorm_bf16_f32_weight(
                ref_residual_d.ptr,
                norm_weight_d.ptr,
                ref_norm_d.ptr,
                tokens,
                hidden,
                eps,
                library=gguf_lib,
                runtime=_runtime,
            )
            fused = (
                weighted_sum_shared_gate_combine_residual_rmsnorm_gguf_bf16_out
                if weighted
                else shared_gate_combine_residual_rmsnorm_gguf_bf16_out
            )
        elif flavor == "paro_bf16":
            paro_rmsnorm_out_bf16(
                ref_residual_d.ptr,
                norm_weight_d.ptr,
                ref_norm_d.ptr,
                tokens,
                hidden,
                eps,
                library=norm_lib,
                runtime=_runtime,
            )
            fused = (
                weighted_sum_shared_gate_combine_residual_rmsnorm_paro_bf16_out
                if weighted
                else shared_gate_combine_residual_rmsnorm_paro_bf16_out
            )
        else:
            paro_rmsnorm_out_fp16(
                ref_residual_d.ptr,
                norm_weight_d.ptr,
                ref_norm_d.ptr,
                tokens,
                hidden,
                eps,
                library=norm_lib,
                runtime=_runtime,
            )
            fused = (
                weighted_sum_shared_gate_combine_residual_rmsnorm_paro_fp16_out
                if weighted
                else shared_gate_combine_residual_rmsnorm_paro_fp16_out
            )

        common = (
            selected_d.ptr,
            *((routing_d.ptr,) if weighted else ()),
            shared_d.ptr,
            gate_d.ptr,
            residual_d.ptr,
            norm_weight_d.ptr,
            fused_norm_d.ptr,
            fused_residual_d.ptr,
            tokens,
            *((top_k,) if weighted else ()),
            hidden,
            gate_stride,
        )
        fused(*common, eps=eps, library=combine_lib, runtime=_runtime)
        _runtime.stream_synchronize(0)

        ref_residual = _download(_runtime, ref_residual_d, (tokens, hidden))
        ref_norm = _download(_runtime, ref_norm_d, (tokens, hidden))
        fused_residual = _download(_runtime, fused_residual_d, (tokens, hidden))
        fused_norm = _download(_runtime, fused_norm_d, (tokens, hidden))
    finally:
        _free_all(_runtime, buffers)

    np.testing.assert_array_equal(fused_residual, ref_residual)
    np.testing.assert_array_equal(fused_norm, ref_norm)

    selected_cpu = _decode(selected_bits, activation_dtype)
    shared_cpu = _decode(shared_bits, activation_dtype)
    residual_cpu = _decode(residual_bits, activation_dtype)
    weight_cpu = norm_weight if flavor == "gguf_bf16" else _decode(norm_weight, activation_dtype)
    cpu_residual, cpu_norm = moe_tail_next_rmsnorm(
        selected_cpu,
        shared_cpu,
        gate_logits,
        residual_cpu,
        weight_cpu,
        routing_weights=routing if weighted else None,
        eps=eps,
        activation_dtype=activation_dtype,
    )
    atol = 0.04 if activation_dtype == "bf16" else 0.008
    np.testing.assert_allclose(_decode(fused_residual, activation_dtype), cpu_residual, rtol=0.02, atol=atol)
    np.testing.assert_allclose(_decode(fused_norm, activation_dtype), cpu_norm, rtol=0.02, atol=atol)
