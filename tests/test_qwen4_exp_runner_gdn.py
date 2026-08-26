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
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGDNScratch,
    run_qwen4_exp_gdn_token_mixer,
)
from tests.test_qwen4_exp_runner_gr import _dense_f32_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_runner_gdn_token_mixer_matches_reduced_cpu_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(3504)
    hidden, k_heads, v_heads, head_dim, kernel_size = 8, 2, 4, 4, 4
    qkv_width = 2 * k_heads * head_dim + v_heads * head_dim
    core_width = v_heads * head_dim
    mixed = rng.normal(0.0, 0.1, size=(1, hidden)).astype(np.float32)
    arrays = {
        "attn_qkv": rng.normal(0.0, 0.15, size=(qkv_width, hidden)).astype(np.float32),
        "attn_gate": rng.normal(0.0, 0.15, size=(core_width, hidden)).astype(np.float32),
        "ssm_alpha": rng.normal(0.0, 0.15, size=(v_heads, hidden)).astype(np.float32),
        "ssm_beta": rng.normal(0.0, 0.15, size=(v_heads, hidden)).astype(np.float32),
        "ssm_out": rng.normal(0.0, 0.15, size=(hidden, core_width)).astype(np.float32),
    }
    conv_weight = rng.normal(0.0, 0.1, size=(qkv_width, kernel_size)).astype(np.float32)
    dt_bias = rng.normal(-1.0, 0.1, size=v_heads).astype(np.float32)
    a_log = rng.normal(-0.5, 0.1, size=v_heads).astype(np.float32)
    norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    conv_state = rng.normal(0.0, 0.05, size=(qkv_width, kernel_size)).astype(np.float32)
    matrix_state = rng.normal(
        0.0,
        0.01,
        size=(v_heads, head_dim, head_dim),
    ).astype(np.float32)

    qkv = mixed @ arrays["attn_qkv"].T
    gate = (mixed @ arrays["attn_gate"].T).reshape(v_heads, head_dim)
    alpha = (mixed @ arrays["ssm_alpha"].T).reshape(v_heads)
    beta_logits = (mixed @ arrays["ssm_beta"].T).reshape(v_heads)
    shifted = np.concatenate((conv_state[:, 1:], qkv.T), axis=1)
    conv_raw = np.sum(shifted * conv_weight, axis=1, dtype=np.float32)
    conv = conv_raw / (1.0 + np.exp(-conv_raw))
    q_raw = conv[: k_heads * head_dim].reshape(k_heads, head_dim)
    k_raw = conv[k_heads * head_dim : 2 * k_heads * head_dim].reshape(k_heads, head_dim)
    value = conv[2 * k_heads * head_dim :].reshape(v_heads, head_dim)
    mapping = np.arange(v_heads) % k_heads
    query = q_raw[mapping]
    key = k_raw[mapping]
    query /= np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + np.float32(1e-6))
    query /= np.sqrt(np.float32(head_dim))
    key /= np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + np.float32(1e-6))
    beta = 1.0 / (1.0 + np.exp(-beta_logits))
    decay = np.exp(-np.exp(a_log) * np.log1p(np.exp(alpha + dt_bias)))
    core, next_matrix = gdn_prefill_recurrent_segments(
        query[None], key[None], value[None], beta[None], decay[None],
        matrix_state[None], [0, 1], [0],
    )
    gated = sigmoid_gated_rmsnorm(core, norm, gate[None])[0].reshape(1, core_width)
    expected = gated @ arrays["ssm_out"].T

    allocations = []
    scratch = None
    try:
        d_mixed = _upload(mixed, runtime, allocations)
        weights = {
            name: _dense_f32_weight(name, array, runtime, allocations)
            for name, array in arrays.items()
        }
        d_conv_weight = _upload(conv_weight, runtime, allocations)
        d_dt = _upload(dt_bias, runtime, allocations)
        d_a = _upload(a_log, runtime, allocations)
        d_norm = _upload(norm, runtime, allocations)
        d_conv_state = _upload(conv_state, runtime, allocations)
        d_matrix = _upload(matrix_state, runtime, allocations)
        scratch = Qwen4ExpGDNScratch.allocate(
            rows=1,
            qkv_width=qkv_width,
            core_width=core_width,
            scalar_width=v_heads,
            hidden=hidden,
            runtime=runtime,
        )
        output = run_qwen4_exp_gdn_token_mixer(
            d_mixed.ptr,
            weights,
            conv_weight_ptr=d_conv_weight.ptr,
            dt_bias_ptr=d_dt.ptr,
            a_log_ptr=d_a.ptr,
            norm_weight_ptr=d_norm.ptr,
            conv_state_ptr=d_conv_state.ptr,
            recurrent_state_ptr=d_matrix.ptr,
            scratch=scratch,
            rows=1,
            hidden=hidden,
            num_k_heads=k_heads,
            num_v_heads=v_heads,
            head_dim=head_dim,
            conv_kernel=kernel_size,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(output, expected.shape, np.float32, runtime)
        actual_conv_state = _download(d_conv_state, conv_state.shape, np.float32, runtime)
        actual_matrix = _download(d_matrix, matrix_state.shape, np.float32, runtime)
    finally:
        if scratch is not None:
            scratch.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=8e-5, atol=8e-5)
    np.testing.assert_allclose(actual_conv_state, shifted, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_matrix, next_matrix[0], rtol=8e-5, atol=8e-5)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
