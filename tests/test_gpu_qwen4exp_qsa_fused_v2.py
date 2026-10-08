"""RED/GREEN gate: the fused QSA scores+coefficients/values kernel is bit-identical to the v2 pair."""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free, malloc
from hipengine.core.tensor import DType
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    build_qwen4_exp_qsa,
    qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_v2_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_fused_v2_f32,
)


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not hip_available(), reason="HIP runtime not available")


def _uniform_spans(tokens, block_size, runtime):
    from hipengine.core.device import Device
    from hipengine.core.memory import copy_host_to_device, host_array_ptr
    from hipengine.core.tensor import Tensor

    base_offsets = ((np.arange(tokens, dtype=np.int32) // block_size) * block_size)
    dev = malloc(base_offsets.nbytes, runtime=runtime)
    copy_host_to_device(dev, host_array_ptr(base_offsets), base_offsets.nbytes)
    t = Tensor(int(dev.ptr), (int(tokens),), DType.INT32, Device("hip", 0))
    spans = KVLiveSpans.paged_uniform(
        block_table=t, live_counts=t,
        max_live_count=int(tokens), storage_dtype=DType.BF16,
    )
    return spans, dev


@pytest.mark.parametrize("q_heads,kv_heads,selected,head_dim", [
    (16, 8, 64, 256),
    (24, 6, 512, 256),
    (32, 8, 2048, 256),
])
def test_fused_qsa_v2_is_bit_identical(q_heads, kv_heads, selected, head_dim):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(20261008)
    from hipengine.core.memory import copy_host_to_device, copy_device_to_host, host_array_ptr

    def upload(arr, dtype):
        a = np.ascontiguousarray(arr, dtype=dtype)
        dev = malloc(a.nbytes, runtime=runtime)
        copy_host_to_device(dev, host_array_ptr(a), a.nbytes)
        return dev

    def download(dev, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(out), dev, out.nbytes, runtime=runtime)
        return out

    allocations = []
    try:
        query = upload(rng.normal(0, 0.5, (q_heads, head_dim)).astype(np.float32), np.float32)
        key = upload(rng.integers(0, 65535, (tokens := selected * 1, kv_heads, head_dim)).astype(np.uint16), np.uint16)
        value = upload(rng.integers(0, 65535, (tokens, kv_heads, head_dim)).astype(np.uint16), np.uint16)
        positions = upload(np.arange(selected, dtype=np.int64), np.int64)
        block_size = 128
        spans, base_dev = _uniform_spans(tokens, block_size, runtime)
        allocations += [query, key, value, positions, base_dev]

        out_p = malloc(q_heads * head_dim * 4, runtime=runtime)
        out_f = malloc(q_heads * head_dim * 4, runtime=runtime)
        scores_p = malloc(q_heads * selected * 4, runtime=runtime)
        scores_f = malloc(q_heads * selected * 4, runtime=runtime)
        coef_p = malloc(2 * q_heads * selected * 4, runtime=runtime)
        coef_f = malloc(2 * q_heads * selected * 4, runtime=runtime)
        allocations += [out_p, out_f, scores_p, scores_f, coef_p, coef_f]

        qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_v2_f32(
            query.ptr, key.ptr, value.ptr, positions.ptr,
            scores_p.ptr, coef_p.ptr, out_p.ptr, spans,
            selected_count=selected, block_size=block_size,
            query_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
            runtime=runtime,
        )
        qwen4_exp_qsa_sparse_attention_paged_bf16_fused_v2_f32(
            query.ptr, key.ptr, value.ptr, positions.ptr,
            scores_f.ptr, coef_f.ptr, out_f.ptr, spans,
            selected_count=selected, block_size=block_size,
            query_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
            runtime=runtime,
        )
        runtime.device_synchronize()

        np.testing.assert_array_equal(
            download(out_p, (q_heads, head_dim), np.float32).view(np.uint32),
            download(out_f, (q_heads, head_dim), np.float32).view(np.uint32),
        )
        np.testing.assert_array_equal(
            download(scores_p, (q_heads * selected,), np.float32).view(np.uint32),
            download(scores_f, (q_heads * selected,), np.float32).view(np.uint32),
        )
        np.testing.assert_array_equal(
            download(coef_p, (2 * q_heads * selected,), np.float32).view(np.uint32),
            download(coef_f, (2 * q_heads * selected,), np.float32).view(np.uint32),
        )
    finally:
        for d in reversed(allocations):
            free(d, runtime=runtime)
