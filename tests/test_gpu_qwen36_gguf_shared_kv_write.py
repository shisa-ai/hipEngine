"""Exact shared-cache BF16 KV append for dense Qwen3.6 verifier rows."""

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
from hipengine.kernels.hip_gfx1100.attention import paged_kv_write
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.materialize import float_array_to_bf16_bits

_CANDIDATE_NAME = "qwen35_write_paged_kv_mixed_value_bf16_shared_batch_spans"
_BLOCK_SIZE = 256
_KV_HEADS = 4
_HEAD_DIM = 256


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _candidate():
    return getattr(paged_kv_write, _CANDIDATE_NAME, None)


def _upload(buffers: list, array: np.ndarray, *, runtime):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(
        buffer,
        host_array_ptr(contiguous),
        contiguous.nbytes,
        runtime=runtime,
    )
    return buffer


def _download(buffer, shape: tuple[int, ...], *, runtime) -> np.ndarray:
    out = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes, runtime=runtime)
    return out


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 4))
def test_shared_batch_kv_write_matches_scalar_and_cpu_cache_oracles(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    candidate = _candidate()
    assert callable(candidate), "shared-cache batch KV wrapper must be admitted"
    runtime = get_hip_runtime()
    library = paged_kv_write.build_qwen35_paged_kv_write(load=True)
    width = _KV_HEADS * _HEAD_DIM
    capacity = 2 * _BLOCK_SIZE
    cache_shape = (capacity, _KV_HEADS, _HEAD_DIM)
    rng = np.random.default_rng(0x36_5A00 + rows)
    key = rng.normal(0.0, 0.4, size=(rows, width)).astype(np.float32)
    value = float_array_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(rows, width)).astype(np.float32)
    )
    block_table = np.asarray([1, 0], dtype=np.int32)
    positions = np.arange(255, 255 + rows, dtype=np.int64)
    sentinel = np.full(cache_shape, 0xA55A, dtype=np.uint16)
    expected_key = sentinel.copy()
    expected_value = sentinel.copy()
    key_bits = float_array_to_bf16_bits(key)
    for row, position in enumerate(positions):
        logical_block, block_offset = divmod(int(position), _BLOCK_SIZE)
        physical_token = int(block_table[logical_block]) * _BLOCK_SIZE + block_offset
        expected_key[physical_token] = key_bits[row].reshape(_KV_HEADS, _HEAD_DIM)
        expected_value[physical_token] = value[row].reshape(_KV_HEADS, _HEAD_DIM)

    buffers: list = []
    try:
        key_d = _upload(buffers, key, runtime=runtime)
        value_d = _upload(buffers, value, runtime=runtime)
        table_d = _upload(buffers, block_table, runtime=runtime)
        positions_d = _upload(buffers, positions, runtime=runtime)
        control_key_d = _upload(buffers, sentinel, runtime=runtime)
        control_value_d = _upload(buffers, sentinel, runtime=runtime)
        candidate_key_d = _upload(buffers, sentinel, runtime=runtime)
        candidate_value_d = _upload(buffers, sentinel, runtime=runtime)

        table = _tensor(table_d.ptr, block_table.shape, DType.INT32)
        position_rows = _tensor(positions_d.ptr, positions.shape, DType.INT64)
        shared_spans = KVLiveSpans.paged_uniform(
            block_table=table,
            live_counts=position_rows,
            max_live_count=int(positions[-1]),
            storage_dtype=DType.BF16,
            row_positions=position_rows,
            span_role="verify_chain",
        )
        for row in range(rows):
            position = _tensor(
                positions_d.ptr + row * DType.INT64.itemsize,
                (1,),
                DType.INT64,
            )
            scalar_spans = KVLiveSpans.paged_uniform(
                block_table=table,
                live_counts=position,
                max_live_count=int(positions[row]),
                storage_dtype=DType.BF16,
                row_positions=position,
                span_role="verify_chain",
            )
            paged_kv_write.qwen35_write_paged_kv_mixed_value_bf16_spans(
                key_d.ptr + row * width * DType.FP32.itemsize,
                value_d.ptr + row * width * DType.BF16.itemsize,
                control_key_d.ptr,
                control_value_d.ptr,
                scalar_spans,
                _BLOCK_SIZE,
                _KV_HEADS,
                _HEAD_DIM,
                library=library,
                runtime=runtime,
            )
        candidate(
            key_d.ptr,
            value_d.ptr,
            candidate_key_d.ptr,
            candidate_value_d.ptr,
            shared_spans,
            rows,
            _BLOCK_SIZE,
            _KV_HEADS,
            _HEAD_DIM,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        control_key = _download(control_key_d, cache_shape, runtime=runtime)
        control_value = _download(control_value_d, cache_shape, runtime=runtime)
        candidate_key = _download(candidate_key_d, cache_shape, runtime=runtime)
        candidate_value = _download(candidate_value_d, cache_shape, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(candidate_key, control_key)
    np.testing.assert_array_equal(candidate_value, control_value)
    np.testing.assert_array_equal(candidate_key, expected_key)
    np.testing.assert_array_equal(candidate_value, expected_value)
