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
from hipengine.kernels.hip_gfx1100.linear_attn import gdn as gdn_module
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.fixture(scope="module", autouse=True)
def _build_for_detected_target(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment

    with hip_target_arch_environment(hip_test_target_arch):
        yield


def _grouped_cpu_reference(
    conv_out: np.ndarray,
    gate_bits: np.ndarray,
    beta_bits: np.ndarray,
    norm_weight: np.ndarray,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> np.ndarray:
    key_dim = num_k_heads * head_k_dim
    gate = bf16_to_float32(gate_bits).reshape(num_v_heads, head_v_dim)
    beta_raw = bf16_to_float32(beta_bits).reshape(num_v_heads)
    heads_per_k = num_v_heads // num_k_heads
    out = np.empty((num_v_heads, head_v_dim), dtype=np.float32)
    for v_head in range(num_v_heads):
        k_head = v_head // heads_per_k
        q = conv_out[k_head * head_k_dim : (k_head + 1) * head_k_dim]
        k = conv_out[
            key_dim + k_head * head_k_dim : key_dim + (k_head + 1) * head_k_dim
        ]
        value = conv_out[
            2 * key_dim + v_head * head_v_dim : 2 * key_dim + (v_head + 1) * head_v_dim
        ]
        q_norm = q / max(float(np.linalg.norm(q)), 1.0e-6)
        q_norm = q_norm * np.float32(1.0 / np.sqrt(float(head_k_dim)))
        k_norm = k / max(float(np.linalg.norm(k)), 1.0e-6)
        beta = np.float32(1.0 / (1.0 + np.exp(-float(beta_raw[v_head]))))
        raw = np.sum(q_norm * k_norm, dtype=np.float32) * beta * value
        inv_rms = np.float32(
            1.0 / np.sqrt(float(np.mean(raw * raw, dtype=np.float32)) + 1.0e-6)
        )
        silu_gate = gate[v_head] / (1.0 + np.exp(-gate[v_head]))
        out[v_head] = raw * inv_rms * norm_weight * silu_gate
    return out


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_paro_grouped_head_gdn_matches_transformers_repeat_interleave() -> None:
    """Canonical HF/PARO heads are grouped; GGUF-only weights are tiled."""

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    build_grouped = getattr(gdn_module, "build_qwen35_linear_attn_gdn_grouped_heads", None)
    assert callable(build_grouped), "canonical PARO grouped-head GDN build is not implemented"
    library = build_grouped(load=True)

    num_k_heads = 2
    num_v_heads = 4
    head_k_dim = 4
    head_v_dim = 4
    key_dim = num_k_heads * head_k_dim
    q = np.asarray([[1.0, 2.0, 3.0, 4.0], [1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    k = np.asarray([[4.0, 3.0, 2.0, 1.0], [-4.0, 3.0, -2.0, 1.0]], dtype=np.float32)
    value = np.asarray(
        [[0.5, -0.25, 0.75, -1.0], [1.0, 0.5, -0.5, -1.0],
         [-0.75, 0.25, 1.25, -0.5], [0.25, -1.25, 0.5, 0.75]],
        dtype=np.float32,
    )
    conv_out = np.ascontiguousarray(np.concatenate((q.reshape(-1), k.reshape(-1), value.reshape(-1))))
    assert conv_out.size == 2 * key_dim + num_v_heads * head_v_dim
    gate_bits = float_array_to_bf16_bits(
        np.linspace(-0.75, 1.25, num_v_heads * head_v_dim, dtype=np.float32)
    )
    alpha_bits = float_array_to_bf16_bits(np.zeros(num_v_heads, dtype=np.float32))
    beta_bits = float_array_to_bf16_bits(np.asarray([-0.5, 0.25, 0.75, -0.25], dtype=np.float32))
    dt_bias = np.zeros(num_v_heads, dtype=np.float32)
    a_log = np.zeros(num_v_heads, dtype=np.float32)
    norm_weight = np.asarray([0.75, 1.0, 1.25, 1.5], dtype=np.float32)
    recurrent_state = np.zeros((num_v_heads, head_k_dim, head_v_dim), dtype=np.float32)
    actual = np.empty((num_v_heads, head_v_dim), dtype=np.float32)
    expected = _grouped_cpu_reference(
        conv_out,
        gate_bits,
        beta_bits,
        norm_weight,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )

    buffers = []

    def device(array: np.ndarray):
        allocation = malloc(array.nbytes, runtime=runtime)
        buffers.append(allocation)
        copy_host_to_device(
            allocation,
            host_array_ptr(np.ascontiguousarray(array)),
            runtime=runtime,
        )
        return allocation

    try:
        dconv = device(conv_out)
        dgate = device(gate_bits)
        dalpha = device(alpha_bits)
        dbeta = device(beta_bits)
        ddt = device(dt_bias)
        dalog = device(a_log)
        dnorm = device(norm_weight)
        dstate = device(recurrent_state)
        dout = malloc(actual.nbytes, runtime=runtime)
        buffers.append(dout)
        gdn_module.qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
            dconv.ptr,
            dgate.ptr,
            dalpha.ptr,
            dbeta.ptr,
            ddt.ptr,
            dalog.ptr,
            dnorm.ptr,
            dstate.ptr,
            dout.ptr,
            1.0e-6,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), dout, runtime=runtime)
    finally:
        for allocation in reversed(buffers):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)
