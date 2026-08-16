from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference import (
    paged_attn_decode_int8_per_token_head,
    quantize_kv_int8_per_token_head,
)
from hipengine.kernels.hip_gfx1100.attention import (
    build_qwen35_paged_attn_decode,
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_batch_strided_spans,
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans,
)
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
from hipengine.loading.materialize import float_array_to_bf16_bits


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    expanded = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return expanded.view(np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-source))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _compiler_version() -> str:
    path = Path("/tmp/hipengine-hipcc-version.txt")
    if not path.is_file():
        pytest.skip("cached HIP compiler-version file is unavailable")
    return path.read_text()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.parametrize(
    ("rows", "live_counts"),
    (
        (1, (8193,)),
        (2, (255, 257)),
        (4, (1, 256, 257, 1025)),
        (8, (1, 2, 255, 256, 257, 513, 1023, 0)),
    ),
    ids=("c1-8k-page-tail", "c2-page-boundary", "c4-ragged", "c8-sparse"),
)
def test_qwen38_int8_batch_attention_matches_cpu_and_independent_c1(
    rows: int,
    live_counts: tuple[int, ...],
) -> None:
    runtime = get_hip_runtime()
    library = build_qwen35_paged_attn_decode(
        load=True,
        compiler_version=_compiler_version(),
        require_cached=True,
    )
    block_size = 256
    num_q_heads, num_kv_heads, head_dim = 24, 4, 256
    q_width = num_q_heads * head_dim
    max_context = max(live_counts)
    num_splits = max(1, (max_context + block_size - 1) // block_size)
    blocks_per_row = num_splits
    scale = head_dim**-0.5
    rng = np.random.default_rng(0x38C200 + rows)

    block_table = np.full((rows, blocks_per_row), -1, dtype=np.int32)
    next_block = 3
    logical_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = []
    for row, context in enumerate(live_counts):
        if context <= 0:
            logical_rows.append(None)
            continue
        owned_blocks = (context + block_size - 1) // block_size
        row_blocks = np.arange(next_block, next_block + owned_blocks, dtype=np.int32)
        next_block += owned_blocks
        if row % 2:
            row_blocks = row_blocks[::-1].copy()
        block_table[row, :owned_blocks] = row_blocks
        block_table[row, owned_blocks:] = row_blocks[-1]
        logical_key = rng.normal(
            0.0,
            0.2,
            size=(context, num_kv_heads, head_dim),
        ).astype(np.float32)
        logical_value = rng.normal(
            0.0,
            0.2,
            size=(context, num_kv_heads, head_dim),
        ).astype(np.float32)
        logical_key[0, 0] = 0.0
        logical_value[-1, -1] = 0.0
        logical_rows.append(
            quantize_kv_int8_per_token_head(
                logical_key,
                logical_value,
                scale_dtype=np.float32,
            )
        )

    physical_blocks = max(rows * blocks_per_row, next_block)
    cache_shape = (physical_blocks, block_size, num_kv_heads, head_dim)
    scale_shape = cache_shape[:-1]
    key_cache = np.zeros(cache_shape, dtype=np.int8)
    value_cache = np.zeros_like(key_cache)
    k_scale = np.zeros(scale_shape, dtype=np.float32)
    v_scale = np.zeros_like(k_scale)
    for row, item in enumerate(logical_rows):
        if item is None:
            continue
        qkey, qvalue, key_scales, value_scales = item
        for token in range(live_counts[row]):
            physical_block = int(block_table[row, token // block_size])
            block_offset = token % block_size
            key_cache[physical_block, block_offset] = qkey[token]
            value_cache[physical_block, block_offset] = qvalue[token]
            k_scale[physical_block, block_offset] = key_scales[token]
            v_scale[physical_block, block_offset] = value_scales[token]

    query_row_stride = q_width + 7
    query_storage = np.zeros((rows * query_row_stride,), dtype=np.float32)
    query = rng.normal(0.0, 0.2, size=(rows, num_q_heads, head_dim)).astype(np.float32)
    for row in range(rows):
        query_storage[row * query_row_stride : row * query_row_stride + q_width] = query[row].reshape(-1)

    gate_head_stride = head_dim + 3
    gate_row_stride = num_q_heads * gate_head_stride + 11
    gate_storage = np.zeros((rows * gate_row_stride,), dtype=np.uint16)
    gate_f32 = rng.normal(0.0, 0.3, size=(rows, num_q_heads, head_dim)).astype(np.float32)
    gate_bits = float_array_to_bf16_bits(gate_f32)
    for row in range(rows):
        for head in range(num_q_heads):
            start = row * gate_row_stride + head * gate_head_stride
            gate_storage[start : start + head_dim] = gate_bits[row, head]

    out_head_stride = head_dim + 5
    out_row_stride = num_q_heads * out_head_stride + 13
    out_storage = np.full((rows * out_row_stride,), 0x7FC1, dtype=np.uint16)
    partial_out = np.zeros((rows, num_q_heads, num_splits, head_dim), dtype=np.float32)
    partial_m = np.zeros((rows, num_q_heads, num_splits), dtype=np.float32)
    partial_l = np.zeros_like(partial_m)
    counts = np.asarray(live_counts, dtype=np.int64)
    device = Device("hip", 0)
    buffers = []

    def to_device(array: np.ndarray):
        host = np.ascontiguousarray(array)
        buffer = malloc(host.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(host), host.nbytes, runtime=runtime)
        return buffer

    try:
        query_b = to_device(query_storage)
        key_b = to_device(key_cache)
        value_b = to_device(value_cache)
        k_scale_b = to_device(k_scale)
        v_scale_b = to_device(v_scale)
        gate_b = to_device(gate_storage)
        out_b = to_device(out_storage)
        partial_out_b = to_device(partial_out)
        partial_m_b = to_device(partial_m)
        partial_l_b = to_device(partial_l)
        table_b = to_device(block_table)
        counts_b = to_device(counts)
        metadata = KVScaleMetadata(
            k_scale=Tensor.from_handle(k_scale_b.ptr, k_scale.shape, DType.FP32, device),
            v_scale=Tensor.from_handle(v_scale_b.ptr, v_scale.shape, DType.FP32, device),
            scale_dtype=DType.FP32,
            granularity="per_token_head",
        )
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(table_b.ptr, block_table.shape, DType.INT32, device),
            live_counts=Tensor.from_handle(counts_b.ptr, counts.shape, DType.INT64, device),
            max_live_count=max_context,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=metadata,
        )
        qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_batch_strided_spans(
            query_b.ptr,
            key_b.ptr,
            value_b.ptr,
            k_scale_b.ptr,
            v_scale_b.ptr,
            gate_b.ptr,
            out_b.ptr,
            partial_out_b.ptr,
            partial_m_b.ptr,
            partial_l_b.ptr,
            spans,
            rows,
            block_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            query_row_stride,
            gate_row_stride,
            gate_head_stride,
            1,
            out_row_stride,
            out_head_stride,
            1,
            scale,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_storage), out_b, out_storage.nbytes, runtime=runtime)

        batch_bits = np.empty((rows, num_q_heads, head_dim), dtype=np.uint16)
        written = np.zeros_like(out_storage, dtype=np.bool_)
        for row in range(rows):
            for head in range(num_q_heads):
                start = row * out_row_stride + head * out_head_stride
                batch_bits[row, head] = out_storage[start : start + head_dim]
                written[start : start + head_dim] = True
        assert np.all(out_storage[~written] == np.uint16(0x7FC1))

        c1_bits = np.empty_like(batch_bits)
        c1_out = np.zeros((num_q_heads, head_dim), dtype=np.uint16)
        c1_partial_out = np.zeros((num_q_heads, num_splits, head_dim), dtype=np.float32)
        c1_partial_m = np.zeros((num_q_heads, num_splits), dtype=np.float32)
        c1_partial_l = np.zeros_like(c1_partial_m)
        c1_out_b = to_device(c1_out)
        c1_partial_out_b = to_device(c1_partial_out)
        c1_partial_m_b = to_device(c1_partial_m)
        c1_partial_l_b = to_device(c1_partial_l)
        for row in range(rows):
            row_spans = KVLiveSpans.paged_uniform(
                block_table=Tensor.from_handle(
                    table_b.ptr + row * blocks_per_row * np.dtype(np.int32).itemsize,
                    (blocks_per_row,),
                    DType.INT32,
                    device,
                ),
                live_counts=Tensor.from_handle(
                    counts_b.ptr + row * np.dtype(np.int64).itemsize,
                    (1,),
                    DType.INT64,
                    device,
                ),
                max_live_count=max_context,
                storage_dtype=DType.INT8_PER_TOKEN_HEAD,
                scale_metadata=metadata,
            )
            qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans(
                query_b.ptr + row * query_row_stride * np.dtype(np.float32).itemsize,
                key_b.ptr,
                value_b.ptr,
                k_scale_b.ptr,
                v_scale_b.ptr,
                gate_b.ptr + row * gate_row_stride * np.dtype(np.uint16).itemsize,
                c1_out_b.ptr,
                c1_partial_out_b.ptr,
                c1_partial_m_b.ptr,
                c1_partial_l_b.ptr,
                row_spans,
                block_size,
                num_splits,
                block_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                gate_head_stride,
                1,
                scale,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(c1_out), c1_out_b, c1_out.nbytes, runtime=runtime)
            c1_bits[row] = c1_out
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(batch_bits, c1_bits)
    if live_counts[-1] == 0:
        assert np.all(batch_bits[-1] == 0)

    candidate_rows = []
    oracle_rows = []
    for row, context in enumerate(live_counts):
        if context <= 0:
            continue
        attention = paged_attn_decode_int8_per_token_head(
            query[row],
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            np.asarray([context], dtype=np.int64),
            block_table=block_table[row],
            block_size=block_size,
            scale=scale,
            output_dtype=np.float32,
        )
        gate = _bf16_bits_to_f32(gate_bits[row])
        gated = attention * _sigmoid(gate)
        oracle_bits = float_array_to_bf16_bits(gated)
        candidate_rows.append(_bf16_bits_to_f32(batch_bits[row]).reshape(-1))
        oracle_rows.append(_bf16_bits_to_f32(oracle_bits).reshape(-1))

    candidate = np.stack(candidate_rows)
    oracle = np.stack(oracle_rows)
    np.testing.assert_allclose(candidate, oracle, rtol=0.02, atol=0.004)
    projection_rng = np.random.default_rng(0x38C2F17)
    projection = projection_rng.normal(0.0, 0.02, size=(q_width, 257)).astype(np.float32)
    candidate_logits = candidate @ projection
    oracle_logits = oracle @ projection
    candidate_probs = _softmax(candidate_logits)
    oracle_probs = _softmax(oracle_logits)
    kl = np.sum(
        oracle_probs * np.log(np.maximum(oracle_probs, 1e-20) / np.maximum(candidate_probs, 1e-20)),
        axis=-1,
    )
    top1 = np.mean(np.argmax(candidate_logits, axis=-1) == np.argmax(oracle_logits, axis=-1))
    assert float(np.max(kl)) <= 0.05
    assert float(top1) >= 0.90
