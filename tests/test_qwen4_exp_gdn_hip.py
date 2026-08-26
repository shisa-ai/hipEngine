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
from hipengine.kernels.cpu_reference import gdn_prefill_recurrent_segments
from hipengine.kernels.cpu_reference.qwen4_exp import sigmoid_gated_rmsnorm


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_qwen4_exp_gdn_build_and_registry_contract() -> None:
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        plan_qwen4_exp_gdn_build,
        qwen4_exp_gdn_decode_f32,
        register_qwen4_exp_gdn_kernels,
    )
    from hipengine.kernels.registry import resolve

    artifact = plan_qwen4_exp_gdn_build()
    assert artifact.output_path.name == "qwen4_exp_gdn.so"
    register_qwen4_exp_gdn_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_recurrence_norm_gate",
            quant="f32_state",
            variant="qwen4exp_sigmoid_strict",
        )
        is qwen4_exp_gdn_decode_f32
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_gdn_decode_matches_cpu_at_production_geometry() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        build_qwen4_exp_gdn,
        qwen4_exp_gdn_decode_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gdn(load=True)
    rng = np.random.default_rng(4035)
    k_heads, v_heads, head_dim = 16, 48, 128
    value_dim = 128
    q_raw = rng.normal(0.0, 0.05, size=(k_heads, head_dim)).astype(np.float32)
    k_raw = rng.normal(0.0, 0.05, size=(k_heads, head_dim)).astype(np.float32)
    # HF Qwen4Exp l2norm adds eps to the sum. A zero head must stay finite,
    # rather than evaluating 0 * rsqrt(0) in the fused decode kernel.
    q_raw[0] = 0.0
    k_raw[0] = 0.0
    value = rng.normal(0.0, 0.05, size=(v_heads, value_dim)).astype(np.float32)
    conv = np.concatenate((q_raw.reshape(-1), k_raw.reshape(-1), value.reshape(-1)))
    gate = rng.normal(0.0, 0.5, size=(v_heads, value_dim)).astype(np.float32)
    alpha = rng.normal(-0.2, 0.1, size=(v_heads,)).astype(np.float32)
    beta_logits = rng.normal(0.0, 0.2, size=(v_heads,)).astype(np.float32)
    dt_bias = rng.normal(-1.0, 0.1, size=(v_heads,)).astype(np.float32)
    a_log = rng.normal(-0.5, 0.1, size=(v_heads,)).astype(np.float32)
    norm = rng.normal(1.0, 0.05, size=(value_dim,)).astype(np.float32)
    state = rng.normal(
        0.0,
        0.01,
        size=(v_heads, head_dim, value_dim),
    ).astype(np.float32)

    mapping = np.arange(v_heads) % k_heads
    query = q_raw[mapping]
    key = k_raw[mapping]
    query /= np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + np.float32(1e-6))
    query /= np.sqrt(np.float32(head_dim))
    key /= np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + np.float32(1e-6))
    beta = 1.0 / (1.0 + np.exp(-beta_logits))
    decay = np.exp(-np.exp(a_log) * np.log1p(np.exp(alpha + dt_bias)))
    core, expected_state = gdn_prefill_recurrent_segments(
        query[None],
        key[None],
        value[None],
        beta[None],
        decay[None],
        state[None],
        [0, 1],
        [0],
    )
    expected = sigmoid_gated_rmsnorm(core, norm, gate[None])[0]

    allocations = []
    try:
        d_conv = _upload(conv, runtime, allocations)
        d_gate = _upload(gate, runtime, allocations)
        d_alpha = _upload(alpha, runtime, allocations)
        d_beta = _upload(beta_logits, runtime, allocations)
        d_dt = _upload(dt_bias, runtime, allocations)
        d_a = _upload(a_log, runtime, allocations)
        d_norm = _upload(norm, runtime, allocations)
        d_state = _upload(state, runtime, allocations)
        d_output = _alloc(expected.shape, np.float32, runtime, allocations)
        qwen4_exp_gdn_decode_f32(
            d_conv.ptr,
            d_gate.ptr,
            d_alpha.ptr,
            d_beta.ptr,
            d_dt.ptr,
            d_a.ptr,
            d_norm.ptr,
            d_state.ptr,
            d_output.ptr,
            k_heads,
            v_heads,
            head_dim,
            value_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_output, expected.shape, np.float32, runtime)
        actual_state = _download(d_state, state.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(actual_state, expected_state[0], rtol=5e-5, atol=5e-5)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape, dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
