from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference.qwen4_exp import qsa_sparse_gqa_attention
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_sparse_paged_gqa_matches_original_kv_cpu_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4608)
    capacity, block_size = 512, 256
    q_heads, kv_heads, head_dim = 24, 2, 256
    query = rng.normal(0.0, 0.1, size=(q_heads, head_dim)).astype(np.float32)
    key = rng.normal(0.0, 0.1, size=(capacity, kv_heads, head_dim)).astype(np.float32)
    value = rng.normal(0.0, 0.1, size=(capacity, kv_heads, head_dim)).astype(np.float32)
    key_bits = float_array_to_bf16_bits(key)
    value_bits = float_array_to_bf16_bits(value)
    key_bf16 = bf16_to_float32(key_bits)
    value_bf16 = bf16_to_float32(value_bits)
    selected = np.array([0, 3, 255, 256, 300, 400, 511], dtype=np.int64)
    expected = qsa_sparse_gqa_attention(
        query[None],
        key_bf16,
        value_bf16,
        query_positions=[511],
        key_positions=np.arange(capacity),
        selected_positions=(selected,),
    )[0]
    block_table = np.array([1, 0], dtype=np.int32)
    key_physical = np.concatenate((key_bits[256:], key_bits[:256]), axis=0)
    value_physical = np.concatenate((value_bits[256:], value_bits[:256]), axis=0)
    live = np.array([capacity], dtype=np.int64)

    allocations = []
    try:
        d_query = _upload(query, runtime, allocations)
        d_key = _upload(key_physical, runtime, allocations)
        d_value = _upload(value_physical, runtime, allocations)
        d_selected = _upload(selected, runtime, allocations)
        d_table = _upload(block_table, runtime, allocations)
        d_live = _upload(live, runtime, allocations)
        d_output = _alloc(expected.shape, np.float32, runtime, allocations)
        device = Device("hip", 0)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(d_table.ptr, block_table.shape, DType.INT32, device),
            live_counts=Tensor.from_handle(d_live.ptr, live.shape, DType.INT64, device),
            max_live_count=capacity,
            storage_dtype=DType.BF16,
        )
        qwen4_exp_qsa_sparse_attention_paged_bf16_f32(
            d_query.ptr,
            d_key.ptr,
            d_value.ptr,
            d_selected.ptr,
            d_output.ptr,
            spans,
            selected_count=selected.size,
            block_size=block_size,
            query_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_output, expected.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_sparse_ordered_three_pass_is_bit_exact_to_strict() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
        qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4621)
    capacity, block_size = 4_096, 256
    q_heads, kv_heads, head_dim = 24, 2, 256
    selected_count = 2_051
    query = rng.normal(0.0, 0.1, size=(q_heads, head_dim)).astype(np.float32)
    key_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(capacity, kv_heads, head_dim)).astype(np.float32)
    )
    value_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(capacity, kv_heads, head_dim)).astype(np.float32)
    )
    selected = np.sort(
        rng.choice(capacity, size=selected_count, replace=False).astype(np.int64)
    )
    block_table = np.arange(capacity // block_size, dtype=np.int32)[::-1].copy()
    live = np.array([capacity], dtype=np.int64)

    allocations = []
    try:
        d_query = _upload(query, runtime, allocations)
        d_key = _upload(key_bits.reshape(-1, kv_heads, head_dim), runtime, allocations)
        d_value = _upload(value_bits.reshape(-1, kv_heads, head_dim), runtime, allocations)
        d_selected = _upload(selected, runtime, allocations)
        d_table = _upload(block_table, runtime, allocations)
        d_live = _upload(live, runtime, allocations)
        d_strict = _alloc(query.shape, np.float32, runtime, allocations)
        d_candidate = _alloc(query.shape, np.float32, runtime, allocations)
        d_scores = _alloc((q_heads, selected_count), np.float32, runtime, allocations)
        d_coefficients = _alloc(
            (2, q_heads, selected_count), np.float32, runtime, allocations
        )
        device = Device("hip", 0)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                d_table.ptr, block_table.shape, DType.INT32, device
            ),
            live_counts=Tensor.from_handle(
                d_live.ptr, live.shape, DType.INT64, device
            ),
            max_live_count=capacity,
            storage_dtype=DType.BF16,
        )
        common = dict(
            selected_count=selected_count,
            block_size=block_size,
            query_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        qwen4_exp_qsa_sparse_attention_paged_bf16_f32(
            d_query.ptr,
            d_key.ptr,
            d_value.ptr,
            d_selected.ptr,
            d_strict.ptr,
            spans,
            **common,
        )
        qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32(
            d_query.ptr,
            d_key.ptr,
            d_value.ptr,
            d_selected.ptr,
            d_scores.ptr,
            d_coefficients.ptr,
            d_candidate.ptr,
            spans,
            **common,
        )
        runtime.device_synchronize()
        strict = _download(d_strict, query.shape, np.float32, runtime)
        candidate = _download(d_candidate, query.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(candidate.view(np.uint32), strict.view(np.uint32))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_sparse_wave32_h128_matches_strict_production_envelope() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
        qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4613)
    capacity, block_size = 4_096, 256
    q_heads, kv_heads, head_dim = 24, 2, 128
    query = rng.normal(0.0, 0.1, size=(q_heads, head_dim)).astype(np.float32)
    key_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(capacity, kv_heads, head_dim)).astype(np.float32)
    )
    value_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(capacity, kv_heads, head_dim)).astype(np.float32)
    )
    selected = np.sort(
        rng.choice(capacity, size=2_048, replace=False).astype(np.int64)
    )
    block_table = np.arange(capacity // block_size, dtype=np.int32)
    live = np.array([capacity], dtype=np.int64)

    allocations = []
    try:
        d_query = _upload(query, runtime, allocations)
        d_key = _upload(key_bits, runtime, allocations)
        d_value = _upload(value_bits, runtime, allocations)
        d_selected = _upload(selected, runtime, allocations)
        d_table = _upload(block_table, runtime, allocations)
        d_live = _upload(live, runtime, allocations)
        d_strict = _alloc(query.shape, np.float32, runtime, allocations)
        d_wave32 = _alloc(query.shape, np.float32, runtime, allocations)
        device = Device("hip", 0)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                d_table.ptr, block_table.shape, DType.INT32, device
            ),
            live_counts=Tensor.from_handle(
                d_live.ptr, live.shape, DType.INT64, device
            ),
            max_live_count=capacity,
            storage_dtype=DType.BF16,
        )
        common = dict(
            selected_count=selected.size,
            block_size=block_size,
            query_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        qwen4_exp_qsa_sparse_attention_paged_bf16_f32(
            d_query.ptr,
            d_key.ptr,
            d_value.ptr,
            d_selected.ptr,
            d_strict.ptr,
            spans,
            **common,
        )
        qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32(
            d_query.ptr,
            d_key.ptr,
            d_value.ptr,
            d_selected.ptr,
            d_wave32.ptr,
            spans,
            **common,
        )
        runtime.device_synchronize()
        strict = _download(d_strict, query.shape, np.float32, runtime)
        wave32 = _download(d_wave32, query.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(wave32, strict, rtol=2e-6, atol=2e-8)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_sparse_paged_rows_write_and_attend_variable_selections() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
        qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    )
    from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
        qwen35_write_paged_kv_f32_batch_spans,
    )
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4618)
    rows, capacity, block_size = 3, 512, 256
    q_heads, kv_heads, head_dim = 4, 2, 8
    positions = np.array([2, 257, 510], dtype=np.int64)
    query = rng.normal(0.0, 0.1, size=(rows, q_heads, head_dim)).astype(np.float32)
    key_rows = rng.normal(0.0, 0.1, size=(rows, kv_heads, head_dim)).astype(np.float32)
    value_rows = rng.normal(0.0, 0.1, size=(rows, kv_heads, head_dim)).astype(np.float32)
    selections = (
        np.array([0, 2], dtype=np.int64),
        np.array([1, 256, 257], dtype=np.int64),
        np.array([0, 255, 256, 400, 510], dtype=np.int64),
    )
    counts = np.asarray([values.size for values in selections], dtype=np.int32)
    selected = np.full((rows, int(counts.max())), -1, dtype=np.int64)
    for row, values in enumerate(selections):
        selected[row, : values.size] = values
    logical_key_bits = np.zeros((capacity, kv_heads, head_dim), dtype=np.uint16)
    logical_value_bits = np.zeros_like(logical_key_bits)
    logical_key_bits[positions] = float_array_to_bf16_bits(key_rows)
    logical_value_bits[positions] = float_array_to_bf16_bits(value_rows)
    logical_key = bf16_to_float32(logical_key_bits)
    logical_value = bf16_to_float32(logical_value_bits)
    expected = qsa_sparse_gqa_attention(
        query,
        logical_key,
        logical_value,
        query_positions=positions,
        key_positions=np.arange(capacity),
        selected_positions=selections,
    )
    dense_selections = tuple(np.arange(position + 1) for position in positions)
    expected_dense = qsa_sparse_gqa_attention(
        query,
        logical_key,
        logical_value,
        query_positions=positions,
        key_positions=np.arange(capacity),
        selected_positions=dense_selections,
    )
    block_table = np.array([1, 0], dtype=np.int32)
    row_tables = np.tile(block_table, (rows, 1))
    physical_key_bits = np.zeros_like(logical_key_bits)
    physical_value_bits = np.zeros_like(logical_value_bits)

    allocations = []
    try:
        d_query = _upload(query, runtime, allocations)
        d_key_rows = _upload(key_rows, runtime, allocations)
        d_value_rows = _upload(value_rows, runtime, allocations)
        d_key = _upload(physical_key_bits, runtime, allocations)
        d_value = _upload(physical_value_bits, runtime, allocations)
        d_selected = _upload(selected, runtime, allocations)
        d_counts = _upload(counts, runtime, allocations)
        d_tables = _upload(row_tables, runtime, allocations)
        d_positions = _upload(positions, runtime, allocations)
        contexts = positions + 1
        d_contexts = _upload(contexts, runtime, allocations)
        d_output = _alloc(expected.shape, np.float32, runtime, allocations)
        d_dense_output = _alloc(expected_dense.shape, np.float32, runtime, allocations)
        d_dense_fixed = _alloc(expected_dense.shape, np.float32, runtime, allocations)
        device = Device("hip", 0)
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                d_tables.ptr, row_tables.shape, DType.INT32, device
            ),
            live_counts=Tensor.from_handle(
                d_positions.ptr, positions.shape, DType.INT64, device
            ),
            max_live_count=capacity - 1,
            storage_dtype=DType.BF16,
        )
        qwen35_write_paged_kv_f32_batch_spans(
            d_key_rows.ptr,
            d_value_rows.ptr,
            d_key.ptr,
            d_value.ptr,
            spans,
            rows,
            block_size,
            kv_heads,
            head_dim,
            runtime=runtime,
        )
        qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32(
            d_query.ptr,
            d_key.ptr,
            d_value.ptr,
            d_selected.ptr,
            d_counts.ptr,
            d_output.ptr,
            spans,
            rows=rows,
            selected_stride=selected.shape[1],
            block_size=block_size,
            query_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            runtime=runtime,
        )
        dense_spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                d_tables.ptr, row_tables.shape, DType.INT32, device
            ),
            live_counts=Tensor.from_handle(
                d_contexts.ptr, contexts.shape, DType.INT64, device
            ),
            max_live_count=capacity,
            storage_dtype=DType.BF16,
        )
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
            d_query.ptr,
            d_key.ptr,
            d_value.ptr,
            d_dense_output.ptr,
            dense_spans,
            rows,
            capacity,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            head_dim ** -0.5,
            runtime=runtime,
        )
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans(
            d_query.ptr,
            d_key.ptr,
            d_value.ptr,
            d_dense_fixed.ptr,
            dense_spans,
            rows,
            capacity,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            head_dim ** -0.5,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_output, expected.shape, np.float32, runtime)
        actual_dense = _download(
            d_dense_output, expected_dense.shape, np.float32, runtime
        )
        actual_dense_fixed = _download(
            d_dense_fixed, expected_dense.shape, np.float32, runtime
        )
        actual_key = _download(d_key, physical_key_bits.shape, np.uint16, runtime)
        actual_value = _download(d_value, physical_value_bits.shape, np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    for logical_block, physical_block in enumerate(block_table):
        start = logical_block * block_size
        stop = start + block_size
        np.testing.assert_array_equal(actual_key[physical_block * block_size:(physical_block + 1) * block_size], logical_key_bits[start:stop])
        np.testing.assert_array_equal(actual_value[physical_block * block_size:(physical_block + 1) * block_size], logical_value_bits[start:stop])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_dense, expected_dense, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        actual_dense_fixed.view(np.uint32), actual_dense.view(np.uint32)
    )


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
