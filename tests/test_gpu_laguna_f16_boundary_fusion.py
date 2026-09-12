"""Exactness gates for Laguna's fused BF16-to-FP16 projection boundaries."""

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
from hipengine.kernels.hip_gfx1100.convert.cast import (
    bf16_to_fp16,
    build_cast,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    build_gguf_ops,
    gguf_rmsnorm_bf16_f32_weight,
    gguf_rmsnorm_bf16_f32_weight_out_fp16_via_bf16,
)
from hipengine.kernels.hip_gfx1100.fused.laguna_attention import (
    build_laguna_attention,
    laguna_softplus_head_gate_f32_bf16_out,
    laguna_softplus_head_gate_f32_fp16_via_bf16_out,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray):
    host = np.ascontiguousarray(values)
    device = malloc(host.nbytes, runtime=runtime)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")
def test_laguna_f16_boundary_fusions_match_separate_cast_bits() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    gguf_library = build_gguf_ops(load=True)
    gate_library = build_laguna_attention(load=True)
    cast_library = build_cast(load=True)
    rng = np.random.default_rng(20260726)
    rows, hidden = 3, 256
    heads, head_dim = 6, 32
    source = _bf16_bits(
        rng.normal(0.0, 0.8, size=(rows, hidden)).astype(np.float32)
    )
    weight = rng.normal(1.0, 0.1, size=(hidden,)).astype(np.float32)
    context = rng.normal(
        0.0, 0.3, size=(rows, heads, head_dim)
    ).astype(np.float32)
    gate = rng.normal(0.0, 3.0, size=(rows, heads)).astype(np.float32)

    buffers = []
    try:
        source_dev = _upload(runtime, source)
        weight_dev = _upload(runtime, weight)
        context_dev = _upload(runtime, context)
        gate_dev = _upload(runtime, gate)
        rms_bf16_dev = malloc(source.nbytes, runtime=runtime)
        rms_baseline_dev = malloc(source.nbytes, runtime=runtime)
        rms_fused_dev = malloc(source.nbytes, runtime=runtime)
        gate_nbytes = context.size * np.dtype(np.uint16).itemsize
        gate_bf16_dev = malloc(gate_nbytes, runtime=runtime)
        gate_baseline_dev = malloc(gate_nbytes, runtime=runtime)
        gate_fused_dev = malloc(gate_nbytes, runtime=runtime)
        buffers.extend(
            (
                source_dev,
                weight_dev,
                context_dev,
                gate_dev,
                rms_bf16_dev,
                rms_baseline_dev,
                rms_fused_dev,
                gate_bf16_dev,
                gate_baseline_dev,
                gate_fused_dev,
            )
        )

        gguf_rmsnorm_bf16_f32_weight(
            source_dev.ptr,
            weight_dev.ptr,
            rms_bf16_dev.ptr,
            rows,
            hidden,
            1.0e-6,
            library=gguf_library,
            runtime=runtime,
        )
        bf16_to_fp16(
            rms_bf16_dev.ptr,
            rms_baseline_dev.ptr,
            source.size,
            library=cast_library,
            runtime=runtime,
        )
        gguf_rmsnorm_bf16_f32_weight_out_fp16_via_bf16(
            source_dev.ptr,
            weight_dev.ptr,
            rms_fused_dev.ptr,
            rows,
            hidden,
            1.0e-6,
            library=gguf_library,
            runtime=runtime,
        )

        laguna_softplus_head_gate_f32_bf16_out(
            context_dev.ptr,
            gate_dev.ptr,
            gate_bf16_dev.ptr,
            rows,
            heads,
            head_dim,
            library=gate_library,
            runtime=runtime,
        )
        bf16_to_fp16(
            gate_bf16_dev.ptr,
            gate_baseline_dev.ptr,
            context.size,
            library=cast_library,
            runtime=runtime,
        )
        laguna_softplus_head_gate_f32_fp16_via_bf16_out(
            context_dev.ptr,
            gate_dev.ptr,
            gate_fused_dev.ptr,
            rows,
            heads,
            head_dim,
            library=gate_library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        rms_baseline = np.empty_like(source)
        rms_fused = np.empty_like(source)
        gate_baseline = np.empty(context.shape, dtype=np.uint16)
        gate_fused = np.empty_like(gate_baseline)
        copy_device_to_host(
            host_array_ptr(rms_baseline),
            rms_baseline_dev,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(rms_fused),
            rms_fused_dev,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(gate_baseline),
            gate_baseline_dev,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(gate_fused),
            gate_fused_dev,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    assert np.array_equal(rms_fused, rms_baseline)
    assert np.array_equal(gate_fused, gate_baseline)
