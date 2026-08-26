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
from hipengine.kernels.cpu_reference.qwen4_exp import (
    qsa_interleaved_rope,
    qsa_sparse_gqa_attention,
)
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpDenseAttentionState,
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAScratch,
    run_qwen4_exp_dense_qsa_token_mixer,
)
from tests.test_qwen4_exp_runner_gr import _dense_f32_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _rmsnorm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-6) * weight


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_dense_qsa_runner_matches_cpu_through_three_decode_positions() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4308)
    hidden, query_heads, kv_heads, head_dim = 8, 2, 1, 4
    rotary_dim, theta = 4, 100.0
    q_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    arrays = {
        "attn_q": rng.normal(0.0, 0.1, size=(q_width * 2, hidden)).astype(np.float32),
        "attn_k": rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
        "attn_v": rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
        "attn_output": rng.normal(0.0, 0.1, size=(hidden, q_width)).astype(np.float32),
    }
    q_norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    k_norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    mixed_rows = rng.normal(0.0, 0.2, size=(3, hidden)).astype(np.float32)

    allocations = []
    state = scratch = None
    try:
        weights = Qwen4ExpQSAMixerDeviceWeights(
            projections={
                name: _dense_f32_weight(name, array, runtime, allocations)
                for name, array in arrays.items()
            },
            q_norm_weight_ptr=_upload(q_norm, runtime, allocations).ptr,
            k_norm_weight_ptr=_upload(k_norm, runtime, allocations).ptr,
        )
        state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=4,
            block_size=256,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        scratch = Qwen4ExpQSAScratch.allocate(
            rows=1,
            hidden=hidden,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        cpu_keys = []
        cpu_values = []
        for position, mixed in enumerate(mixed_rows):
            qfull = (arrays["attn_q"] @ mixed).reshape(query_heads, 2, head_dim)
            query = _rmsnorm(qfull[:, 0], q_norm)
            query = qsa_interleaved_rope(
                query[None], positions=[position], rotary_dim=rotary_dim, theta=theta
            )[0]
            key = _rmsnorm((arrays["attn_k"] @ mixed).reshape(kv_heads, head_dim), k_norm)
            key = qsa_interleaved_rope(
                key[None], positions=[position], rotary_dim=rotary_dim, theta=theta
            )[0]
            value = (arrays["attn_v"] @ mixed).reshape(kv_heads, head_dim)
            cpu_keys.append(key)
            cpu_values.append(value)
            context = qsa_sparse_gqa_attention(
                query[None],
                np.asarray(cpu_keys),
                np.asarray(cpu_values),
                query_positions=[position],
                key_positions=np.arange(position + 1),
                selected_positions=(np.arange(position + 1),),
            )[0]
            gated = context * (1.0 / (1.0 + np.exp(-qfull[:, 1])))
            expected = (arrays["attn_output"] @ gated.reshape(-1)).astype(np.float32)

            d_mixed = _upload(mixed[None], runtime, allocations)
            output = run_qwen4_exp_dense_qsa_token_mixer(
                d_mixed.ptr,
                weights,
                attention_state=state,
                scratch=scratch,
                position=position,
                rows=1,
                hidden=hidden,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                theta=theta,
                runtime=runtime,
            )
            runtime.device_synchronize()
            actual = _download(output, (hidden,), np.float32, runtime)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    finally:
        if scratch is not None:
            scratch.close()
        if state is not None:
            state.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


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
