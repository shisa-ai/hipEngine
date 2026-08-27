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
    qsa_index_scores,
    qsa_interleaved_rope,
    qsa_prepare_index_keys,
    qsa_select_positions,
    qsa_sparse_gqa_attention,
)
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpDenseAttentionState,
    Qwen4ExpQSAIndexDeviceState,
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAPrefillMetadata,
    Qwen4ExpQSAScratch,
    run_qwen4_exp_dense_qsa_token_mixer,
    run_qwen4_exp_qsa_prefill_token_mixer,
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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_qsa_runner_switches_to_native_sparse_selection_above_budget() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4310)
    hidden, query_heads, kv_heads, head_dim = 8, 2, 1, 4
    index_heads, index_dim, ratio, block_budget = 2, 4, 2, 2
    q_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    arrays = {
        "attn_q": rng.normal(0.0, 0.1, size=(q_width * 2, hidden)).astype(np.float32),
        "attn_k": rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
        "attn_v": rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
        "attn_output": rng.normal(0.0, 0.1, size=(hidden, q_width)).astype(np.float32),
        "index_q": rng.normal(0.0, 0.1, size=(index_heads * index_dim, hidden)).astype(np.float32),
        "index_k": rng.normal(0.0, 0.1, size=(index_dim, hidden)).astype(np.float32),
    }
    q_norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    k_norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    index_q_norm = rng.normal(1.0, 0.05, size=index_dim).astype(np.float32)
    index_k_norm = rng.normal(1.0, 0.05, size=index_dim).astype(np.float32)
    mixed_rows = rng.normal(0.0, 0.2, size=(6, hidden)).astype(np.float32)

    allocations = []
    state = index_state = scratch = None
    try:
        weights = Qwen4ExpQSAMixerDeviceWeights(
            projections={
                name: _dense_f32_weight(name, array, runtime, allocations)
                for name, array in arrays.items()
            },
            q_norm_weight_ptr=_upload(q_norm, runtime, allocations).ptr,
            k_norm_weight_ptr=_upload(k_norm, runtime, allocations).ptr,
            index_q_norm_weight_ptr=_upload(index_q_norm, runtime, allocations).ptr,
            index_k_norm_weight_ptr=_upload(index_k_norm, runtime, allocations).ptr,
        )
        state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=8,
            block_size=256,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        index_state = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=state,
            index_heads=index_heads,
            index_dim=index_dim,
            compression_ratio=ratio,
            block_budget=block_budget,
            runtime=runtime,
        )
        scratch = Qwen4ExpQSAScratch.allocate(
            rows=1,
            hidden=hidden,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            index_heads=index_heads,
            index_dim=index_dim,
            runtime=runtime,
        )
        cpu_keys: list[np.ndarray] = []
        cpu_values: list[np.ndarray] = []
        raw_index_keys: list[np.ndarray] = []
        for position, mixed in enumerate(mixed_rows):
            qfull = (arrays["attn_q"] @ mixed).reshape(query_heads, 2, head_dim)
            query = qsa_interleaved_rope(
                _rmsnorm(qfull[:, 0], q_norm)[None],
                positions=[position],
                rotary_dim=4,
                theta=100.0,
            )[0]
            key = qsa_interleaved_rope(
                _rmsnorm((arrays["attn_k"] @ mixed).reshape(kv_heads, head_dim), k_norm)[None],
                positions=[position],
                rotary_dim=4,
                theta=100.0,
            )[0]
            value = (arrays["attn_v"] @ mixed).reshape(kv_heads, head_dim)
            cpu_keys.append(key)
            cpu_values.append(value)
            raw_index_keys.append(arrays["index_k"] @ mixed)
            if position + 1 <= block_budget * ratio + ratio - 1:
                selected = np.arange(position + 1, dtype=np.int64)
            else:
                index_query = _rmsnorm(
                    (arrays["index_q"] @ mixed).reshape(index_heads, index_dim),
                    index_q_norm,
                )
                index_query = qsa_interleaved_rope(
                    index_query[None], positions=[position], rotary_dim=4, theta=100.0
                )[0]
                pooled = qsa_prepare_index_keys(
                    np.asarray(raw_index_keys),
                    np.arange(position + 1),
                    index_k_norm,
                    compression_ratio=ratio,
                    rotary_dim=4,
                    theta=100.0,
                )
                selection = qsa_select_positions(
                    qsa_index_scores(index_query[None], pooled.keys),
                    pooled.block_starts,
                    query_positions=[position],
                    available_positions=np.arange(position + 1),
                    compression_ratio=ratio,
                    block_budget=block_budget,
                )
                selected = selection.selected_positions[0]
            context = qsa_sparse_gqa_attention(
                query[None],
                np.asarray(cpu_keys),
                np.asarray(cpu_values),
                query_positions=[position],
                key_positions=np.arange(position + 1),
                selected_positions=(selected,),
            )[0]
            gated = context * (1.0 / (1.0 + np.exp(-qfull[:, 1])))
            expected = arrays["attn_output"] @ gated.reshape(-1)

            d_mixed = _upload(mixed[None], runtime, allocations)
            output = run_qwen4_exp_dense_qsa_token_mixer(
                d_mixed.ptr,
                weights,
                attention_state=state,
                index_state=index_state,
                scratch=scratch,
                position=position,
                rows=1,
                hidden=hidden,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rotary_dim=4,
                theta=100.0,
                index_heads=index_heads,
                index_dim=index_dim,
                index_rotary_dim=4,
                runtime=runtime,
            )
            runtime.device_synchronize()
            actual = _download(output, (hidden,), np.float32, runtime)
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
        np.testing.assert_array_equal(index_state.selected_positions_host[: selected.size], selected)
    finally:
        if scratch is not None:
            scratch.close()
        if index_state is not None:
            index_state.close()
        if state is not None:
            state.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_qsa_prefill_mixer_matches_serial_across_sparse_boundary() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4320)
    rows, hidden, query_heads, kv_heads, head_dim = 6, 8, 2, 1, 4
    index_heads, index_dim, ratio, block_budget = 2, 4, 2, 2
    q_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    arrays = {
        "attn_q": rng.normal(0.0, 0.1, size=(q_width * 2, hidden)).astype(np.float32),
        "attn_k": rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
        "attn_v": rng.normal(0.0, 0.1, size=(kv_width, hidden)).astype(np.float32),
        "attn_output": rng.normal(0.0, 0.1, size=(hidden, q_width)).astype(np.float32),
        "index_q": rng.normal(0.0, 0.1, size=(index_heads * index_dim, hidden)).astype(np.float32),
        "index_k": rng.normal(0.0, 0.1, size=(index_dim, hidden)).astype(np.float32),
    }
    q_norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    k_norm = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    index_q_norm = rng.normal(1.0, 0.05, size=index_dim).astype(np.float32)
    index_k_norm = rng.normal(1.0, 0.05, size=index_dim).astype(np.float32)
    mixed_rows = rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)

    allocations = []
    serial_state = bulk_state = serial_index = bulk_index = None
    serial_scratch = bulk_scratch = metadata = None
    try:
        weights = Qwen4ExpQSAMixerDeviceWeights(
            projections={
                name: _dense_f32_weight(name, array, runtime, allocations)
                for name, array in arrays.items()
            },
            q_norm_weight_ptr=_upload(q_norm, runtime, allocations).ptr,
            k_norm_weight_ptr=_upload(k_norm, runtime, allocations).ptr,
            index_q_norm_weight_ptr=_upload(index_q_norm, runtime, allocations).ptr,
            index_k_norm_weight_ptr=_upload(index_k_norm, runtime, allocations).ptr,
        )
        serial_state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=8,
            block_size=256,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        bulk_state = Qwen4ExpDenseAttentionState.allocate(
            max_positions=8,
            block_size=256,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        serial_index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=serial_state,
            index_heads=index_heads,
            index_dim=index_dim,
            compression_ratio=ratio,
            block_budget=block_budget,
            runtime=runtime,
        )
        bulk_index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=bulk_state,
            index_heads=index_heads,
            index_dim=index_dim,
            compression_ratio=ratio,
            block_budget=block_budget,
            runtime=runtime,
        )
        serial_scratch = Qwen4ExpQSAScratch.allocate(
            rows=1,
            hidden=hidden,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            index_heads=index_heads,
            index_dim=index_dim,
            runtime=runtime,
        )
        bulk_scratch = Qwen4ExpQSAScratch.allocate(
            rows=rows,
            hidden=hidden,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            index_heads=index_heads,
            index_dim=index_dim,
            runtime=runtime,
        )
        metadata = Qwen4ExpQSAPrefillMetadata.allocate(
            bulk_state,
            rows=rows,
            selection_capacity=block_budget * ratio + ratio - 1,
        )
        d_mixed = _upload(mixed_rows, runtime, allocations)
        serial_outputs = []
        serial_selected = None
        for position in range(rows):
            output = run_qwen4_exp_dense_qsa_token_mixer(
                d_mixed.ptr + position * hidden * 4,
                weights,
                attention_state=serial_state,
                index_state=serial_index,
                scratch=serial_scratch,
                position=position,
                rows=1,
                hidden=hidden,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rotary_dim=4,
                theta=100.0,
                index_heads=index_heads,
                index_dim=index_dim,
                index_rotary_dim=4,
                runtime=runtime,
            )
            runtime.device_synchronize()
            serial_outputs.append(_download(output, (hidden,), np.float32, runtime))
            if position == rows - 1 and position + 1 > serial_index.dense_equivalent_limit:
                count = block_budget * ratio
                serial_selected = serial_index.selected_positions_host[:count].copy()
        bulk_output = run_qwen4_exp_qsa_prefill_token_mixer(
            d_mixed.ptr,
            weights,
            attention_state=bulk_state,
            index_state=bulk_index,
            scratch=bulk_scratch,
            metadata=metadata,
            start_position=0,
            rows=rows,
            hidden=hidden,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rotary_dim=4,
            theta=100.0,
            index_heads=index_heads,
            index_dim=index_dim,
            index_rotary_dim=4,
            runtime=runtime,
        )
        runtime.device_synchronize()
        bulk_outputs = _download(bulk_output, (rows, hidden), np.float32, runtime)
        serial_key = _download(
            serial_state.key_cache,
            (serial_state.max_positions, kv_heads, head_dim),
            np.uint16,
            runtime,
        )
        bulk_key = _download(
            bulk_state.key_cache,
            (bulk_state.max_positions, kv_heads, head_dim),
            np.uint16,
            runtime,
        )
        serial_value = _download(
            serial_state.value_cache,
            (serial_state.max_positions, kv_heads, head_dim),
            np.uint16,
            runtime,
        )
        bulk_value = _download(
            bulk_state.value_cache,
            (bulk_state.max_positions, kv_heads, head_dim),
            np.uint16,
            runtime,
        )
        serial_raw = _download(
            serial_index.raw_keys,
            (serial_index.capacity, index_dim),
            np.float32,
            runtime,
        )
        bulk_raw = _download(
            bulk_index.raw_keys,
            (bulk_index.capacity, index_dim),
            np.float32,
            runtime,
        )
    finally:
        if metadata is not None:
            metadata.close()
        if bulk_scratch is not None:
            bulk_scratch.close()
        if serial_scratch is not None:
            serial_scratch.close()
        if bulk_index is not None:
            bulk_index.close()
        if serial_index is not None:
            serial_index.close()
        if bulk_state is not None:
            bulk_state.close()
        if serial_state is not None:
            serial_state.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(
        bulk_outputs, np.asarray(serial_outputs), rtol=3e-4, atol=3e-4
    )
    np.testing.assert_array_equal(bulk_key, serial_key)
    np.testing.assert_array_equal(bulk_value, serial_value)
    np.testing.assert_array_equal(bulk_raw, serial_raw)
    assert bulk_index.count == serial_index.count == rows
    if serial_selected is not None:
        np.testing.assert_array_equal(
            metadata.selected_positions_host[-1, : serial_selected.size], serial_selected
        )


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, host.nbytes, runtime=runtime)
    return host
