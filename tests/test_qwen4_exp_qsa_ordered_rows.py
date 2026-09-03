"""PF-2: tests for an ordered multi-row (chunked prefill) QSA variant.

Test 1 establishes the oracle envelope on the unmodified path: the current
strict rows kernel must match the CPU reference within the same tolerance used
by the existing rows-kernel test in test_qwen4_exp_qsa_sparse_hip.py.
Test 2 is the RED test for the new registered ordered rows variant; it fails
until qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_rows_f32 is
implemented. Both are HIP-availability guarded.
"""

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


def _build_fixture(rng):
    rows, capacity, block_size = 3, 512, 256
    q_heads, kv_heads, head_dim = 4, 2, 8
    positions = np.array([2, 257, 510][:rows], dtype=np.int64)
    query = rng.normal(0.0, 0.1, size=(rows, q_heads, head_dim)).astype(np.float32)
    key_rows = rng.normal(0.0, 0.1, size=(rows, kv_heads, head_dim)).astype(np.float32)
    value_rows = rng.normal(0.0, 0.1, size=(rows, kv_heads, head_dim)).astype(np.float32)
    selections = tuple(
        np.sort(
            rng.choice(
                positions[i] + 1, size=min(3, positions[i] + 1), replace=False
            ).astype(np.int64)
        )
        for i in range(rows)
    )
    counts = np.asarray([s.size for s in selections], dtype=np.int32)
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
    block_table = np.array([1, 0], dtype=np.int32)
    row_tables = np.tile(block_table, (rows, 1))
    physical_key_bits = np.zeros_like(logical_key_bits)
    physical_value_bits = np.zeros_like(logical_value_bits)
    return dict(
        rows=rows, capacity=capacity, block_size=block_size,
        q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
        positions=positions, query=query, key_rows=key_rows,
        value_rows=value_rows, selected=selected, counts=counts,
        physical_key_bits=physical_key_bits,
        physical_value_bits=physical_value_bits,
        expected=expected, row_tables=row_tables,
    )


def _write_kv_and_spans(fx, runtime, allocations):
    device = Device("hip", 0)
    d_key_rows = _upload(fx["key_rows"], runtime, allocations)
    d_value_rows = _upload(fx["value_rows"], runtime, allocations)
    d_key = _upload(fx["physical_key_bits"], runtime, allocations)
    d_value = _upload(fx["physical_value_bits"], runtime, allocations)
    d_tables = _upload(fx["row_tables"], runtime, allocations)
    d_positions = _upload(fx["positions"], runtime, allocations)
    spans = KVLiveSpans.paged_uniform(
        block_table=Tensor.from_handle(
            d_tables.ptr, fx["row_tables"].shape, DType.INT32, device
        ),
        live_counts=Tensor.from_handle(
            d_positions.ptr, fx["positions"].shape, DType.INT64, device
        ),
        max_live_count=fx["capacity"] - 1,
        storage_dtype=DType.BF16,
    )
    from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
        qwen35_write_paged_kv_f32_batch_spans,
    )

    qwen35_write_paged_kv_f32_batch_spans(
        d_key_rows.ptr, d_value_rows.ptr, d_key.ptr, d_value.ptr,
        spans, fx["rows"], fx["block_size"], fx["kv_heads"], fx["head_dim"],
        runtime=runtime,
    )
    return d_key, d_value, spans


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_strict_rows_kernel_matches_cpu_oracle_on_unmodified_path() -> None:
    """Oracle baseline on the unmodified path: strict rows kernel vs CPU reference."""
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4618)
    fx = _build_fixture(rng)
    allocations = []
    try:
        d_key, d_value, spans = _write_kv_and_spans(fx, runtime, allocations)
        d_query = _upload(fx["query"], runtime, allocations)
        d_selected = _upload(fx["selected"], runtime, allocations)
        d_counts = _upload(fx["counts"], runtime, allocations)
        out_shape = fx["expected"].shape
        d_output = _alloc(out_shape, np.float32, runtime, allocations)
        qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32(
            d_query.ptr, d_key.ptr, d_value.ptr, d_selected.ptr, d_counts.ptr,
            d_output.ptr, spans,
            rows=fx["rows"], selected_stride=fx["selected"].shape[1],
            block_size=fx["block_size"], query_heads=fx["q_heads"],
            kv_heads=fx["kv_heads"], head_dim=fx["head_dim"], runtime=runtime,
        )
        runtime.device_synchronize()
        out = _download(d_output, out_shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    # CPU oracle agreement within tolerance (same envelope as the existing
    # rows-kernel test in test_qwen4_exp_qsa_sparse_hip.py).
    np.testing.assert_allclose(out, fx["expected"], rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ordered_rows_variant_is_bit_exact_to_strict_rows() -> None:
    """RED test: ordered multi-row variant must be bit-exact vs strict rows kernel."""
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention import qwen4_exp_qsa as qsa_module

    wrapper = getattr(
        qsa_module,
        "qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_rows_f32",
        None,
    )
    if wrapper is None:
        pytest.fail(
            "PF-2 RED: qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_rows_f32 "
            "is not implemented yet"
        )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4622)
    fx = _build_fixture(rng)
    allocations = []
    try:
        d_key, d_value, spans = _write_kv_and_spans(fx, runtime, allocations)
        d_query = _upload(fx["query"], runtime, allocations)
        d_selected = _upload(fx["selected"], runtime, allocations)
        d_counts = _upload(fx["counts"], runtime, allocations)
        out_shape = fx["expected"].shape
        d_output_ordered = _alloc(out_shape, np.float32, runtime, allocations)
        d_output_strict = _alloc(out_shape, np.float32, runtime, allocations)

        wrapper(
            d_query.ptr, d_key.ptr, d_value.ptr, d_selected.ptr, d_counts.ptr,
            d_output_ordered.ptr, spans,
            rows=fx["rows"], selected_stride=fx["selected"].shape[1],
            block_size=fx["block_size"], query_heads=fx["q_heads"],
            kv_heads=fx["kv_heads"], head_dim=fx["head_dim"], runtime=runtime,
        )
        qsa_module.qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32(
            d_query.ptr, d_key.ptr, d_value.ptr, d_selected.ptr, d_counts.ptr,
            d_output_strict.ptr, spans,
            rows=fx["rows"], selected_stride=fx["selected"].shape[1],
            block_size=fx["block_size"], query_heads=fx["q_heads"],
            kv_heads=fx["kv_heads"], head_dim=fx["head_dim"], runtime=runtime,
        )
        runtime.device_synchronize()
        out_ordered = _download(d_output_ordered, out_shape, np.float32, runtime)
        out_strict = _download(d_output_strict, out_shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    np.testing.assert_array_equal(
        out_ordered.view(np.uint32), out_strict.view(np.uint32)
    )
