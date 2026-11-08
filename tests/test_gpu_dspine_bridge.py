"""GPU RED: DSpine bridge (elementwise kernels + dense projections) == CPU oracle."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP = True
except OSError:
    _HIP = False

pytestmark = pytest.mark.skipif(not _HIP, reason="ROCm runtime not available")


def _f32_to_bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_to_f32(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def test_gpu_bridge_matches_cpu_oracle():
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.dspine import DSpineBridgeWeights, bridge_write
    from hipengine.kernels.hip_gfx1100.speculative.dspine import (
        dspine_rms_normalize, dspine_codes, dspine_gate_message, dspine_residual)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_dense_bf16_to_f32

    rt = get_hip_runtime()
    T, H, C, R, V = 8, 64, 32, 16, 32
    eps = 1e-6
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((T, H)).astype(np.float32) * 0.5
    anchor = 7
    bridge = DSpineBridgeWeights(
        codebook=rng.standard_normal((V, C)).astype(np.float32) * 0.3,
        early_read_weights=rng.standard_normal((1, C, H)).astype(np.float32) * 0.1,
        read_weight=rng.standard_normal((C, H)).astype(np.float32) * 0.1,
        feature_write_weights=rng.standard_normal((2, H, R)).astype(np.float32) * 0.1,
        gate_weight=rng.standard_normal((R, C)).astype(np.float32) * 0.1,
        gate_bias=rng.standard_normal(R).astype(np.float32) * 0.1,
        predecessor_gate_weight=rng.standard_normal((R, C)).astype(np.float32) * 0.1,
        prev_weight=rng.standard_normal((R, C)).astype(np.float32) * 0.1,
    )
    next_norm = np.ones(H, dtype=np.float32)
    _, expected = bridge_write(raw, np.array([anchor] + [0]*(T-1)), 0, 2, bridge, next_norm, eps)

    def up(x):
        b = malloc(x.nbytes, runtime=rt)
        copy_host_array_to_device(b, np.ascontiguousarray(x), runtime=rt)
        return b

    raw_b = up(_f32_to_bf16(raw))
    Rl_b = up(_f32_to_bf16(bridge.early_read_weights[0]))
    packed_w = np.concatenate([bridge.gate_weight, bridge.predecessor_gate_weight, bridge.prev_weight], axis=0)
    packed_b = up(_f32_to_bf16(packed_w))
    Wl_b = up(_f32_to_bf16(bridge.feature_write_weights[0]))
    cb_b = up(_f32_to_bf16(bridge.codebook))
    bias_b = up(_f32_to_bf16(bridge.gate_bias))
    norm_b = up(_f32_to_bf16(raw))  # placeholder
    normalized_b = malloc(T * H * 2, runtime=rt)
    rms_b = malloc(T * 4, runtime=rt)
    projected_b = malloc(T * C * 4, runtime=rt)
    codes_b = malloc(T * C * 2, runtime=rt)
    packedout_b = malloc(T * 3 * R * 4, runtime=rt)
    message_b = malloc(T * R * 2, runtime=rt)
    delta_b = malloc(T * H * 4, runtime=rt)
    out_b = malloc(T * H * 2, runtime=rt)
    try:
        dspine_rms_normalize(raw_b.ptr, normalized_b.ptr, rms_b.ptr, T, H, eps, runtime=rt)
        dflash_dense_bf16_to_f32(normalized_b.ptr, Rl_b.ptr, projected_b.ptr, T, H, C, runtime=rt)
        dspine_codes(projected_b.ptr, cb_b.ptr, codes_b.ptr, anchor, T, C, eps, np.sqrt(C), runtime=rt)
        dflash_dense_bf16_to_f32(codes_b.ptr, packed_b.ptr, packedout_b.ptr, T, C, 3 * R, runtime=rt)
        dspine_gate_message(packedout_b.ptr, bias_b.ptr, message_b.ptr, T, R, runtime=rt)
        dflash_dense_bf16_to_f32(message_b.ptr, Wl_b.ptr, delta_b.ptr, T, R, H, runtime=rt)
        dspine_residual(raw_b.ptr, rms_b.ptr, delta_b.ptr, out_b.ptr, T, H, runtime=rt)
        rt.device_synchronize()
        got = np.empty(T * H, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(got), out_b, runtime=rt)
        got = _bf16_to_f32(got).reshape(T, H)
    finally:
        for b in (raw_b, Rl_b, packed_b, Wl_b, cb_b, bias_b, norm_b, normalized_b, rms_b,
                  projected_b, codes_b, packedout_b, message_b, delta_b, out_b):
            free(b, runtime=rt)
    assert np.allclose(got, expected, atol=2e-2), f"max diff {np.max(np.abs(got-expected))}"