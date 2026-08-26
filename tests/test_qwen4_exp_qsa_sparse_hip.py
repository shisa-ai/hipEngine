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
