"""GPU correctness for the one-split batched GQA direct-gate decode kernel."""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest


def _has_gfx1100() -> bool:
    try:
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")


def _to_bf16_bits(x_f32: np.ndarray) -> np.ndarray:
    bits = x_f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16_bits(x_u16: np.ndarray) -> np.ndarray:
    return (x_u16.astype(np.uint32) << 16).view(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x).astype(np.float32))


def _cpu_reference(
    query: np.ndarray,
    key_bf16: np.ndarray,
    value_bf16: np.ndarray,
    gate_fp16: np.ndarray,
    live_counts: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    rows, num_q_heads, head_dim = query.shape
    num_kv_heads = key_bf16.shape[2]
    q_per_kv = num_q_heads // num_kv_heads
    key = _from_bf16_bits(key_bf16).astype(np.float32)
    value = _from_bf16_bits(value_bf16).astype(np.float32)
    gate = gate_fp16.astype(np.float32)
    out = np.empty((rows, num_q_heads, head_dim), dtype=np.float16)
    for row in range(rows):
        context_len = int(live_counts[row])
        for q_head in range(num_q_heads):
            kv_head = q_head // q_per_kv
            scores = (key[row, :context_len, kv_head, :] @ query[row, q_head, :]) * np.float32(scale)
            scores = scores.astype(np.float32)
            probs = np.exp(scores - np.max(scores)).astype(np.float32)
            probs = probs / np.sum(probs, dtype=np.float32)
            acc = probs @ value[row, :context_len, kv_head, :]
            out[row, q_head, :] = (acc * _sigmoid(gate[row, q_head, :])).astype(np.float16)
    return out


@pytest.fixture(scope="module")
def _attention_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_qwen35_paged_attn_decode(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array)
    buf = malloc(max(arr.nbytes, 4), runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes, runtime=runtime)
    return buf


def _alloc(runtime, bufs, nbytes: int):
    from hipengine.core.memory import malloc

    buf = malloc(max(nbytes, 4), runtime=runtime)
    bufs.append(buf)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes, runtime=runtime)
    return arr


def _free_all(runtime, bufs) -> None:
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


def _spans(block_table_buf, live_counts_buf, *, rows: int, max_live_count: int):
    from hipengine.core.device import Device
    from hipengine.core.tensor import Tensor
    from hipengine.kvcache import KVLiveSpans

    device = Device("hip", 0)
    return KVLiveSpans.paged_uniform(
        block_table=Tensor.from_handle(block_table_buf.ptr, (rows,), "int32", device),
        live_counts=Tensor.from_handle(live_counts_buf.ptr, (rows,), "int64", device),
        max_live_count=max_live_count,
        storage_dtype="bf16",
    )


def test_qwen35_decode_batched_direct_gate_matches_split_reduce(_attention_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.attention import (
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans,
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans,
    )

    rows = 4
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 256
    block_size = 256
    chunk_size = 64
    num_splits = 1
    scale = head_dim ** -0.5
    rng = np.random.default_rng(0xD1EC7A7E)
    query = (rng.standard_normal((rows, num_q_heads, head_dim), dtype=np.float32) * 0.05).astype(np.float32)
    key = _to_bf16_bits(
        (rng.standard_normal((rows, block_size, num_kv_heads, head_dim), dtype=np.float32) * 0.05).astype(np.float32)
    )
    value = _to_bf16_bits(
        (rng.standard_normal((rows, block_size, num_kv_heads, head_dim), dtype=np.float32) * 0.05).astype(np.float32)
    )
    gate = (rng.standard_normal((rows, num_q_heads, head_dim), dtype=np.float32) * 0.05).astype(np.float16)
    live_counts = np.array([7, 13, 29, 31], dtype=np.int64)
    block_table = np.arange(rows, dtype=np.int32)

    bufs = []
    try:
        query_buf = _upload(_runtime, bufs, query)
        key_buf = _upload(_runtime, bufs, key)
        value_buf = _upload(_runtime, bufs, value)
        gate_buf = _upload(_runtime, bufs, gate)
        block_table_buf = _upload(_runtime, bufs, block_table)
        live_counts_buf = _upload(_runtime, bufs, live_counts)
        out_split = _alloc(_runtime, bufs, rows * num_q_heads * head_dim * np.dtype(np.float16).itemsize)
        out_direct = _alloc(_runtime, bufs, rows * num_q_heads * head_dim * np.dtype(np.float16).itemsize)
        partial_out = _alloc(_runtime, bufs, rows * num_q_heads * num_splits * head_dim * np.dtype(np.float32).itemsize)
        partial_m = _alloc(_runtime, bufs, rows * num_q_heads * num_splits * np.dtype(np.float32).itemsize)
        partial_l = _alloc(_runtime, bufs, rows * num_q_heads * num_splits * np.dtype(np.float32).itemsize)
        spans = _spans(block_table_buf, live_counts_buf, rows=rows, max_live_count=int(live_counts.max()))

        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans(
            query_buf.ptr,
            key_buf.ptr,
            value_buf.ptr,
            gate_buf.ptr,
            out_split.ptr,
            partial_out.ptr,
            partial_m.ptr,
            partial_l.ptr,
            spans,
            rows,
            chunk_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            1,
            scale,
            library=_attention_lib,
            runtime=_runtime,
        )
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans(
            query_buf.ptr,
            key_buf.ptr,
            value_buf.ptr,
            gate_buf.ptr,
            out_direct.ptr,
            spans,
            rows,
            chunk_size,
            num_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            1,
            scale,
            library=_attention_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()

        split = _download(_runtime, out_split, (rows, num_q_heads, head_dim), np.float16)
        direct = _download(_runtime, out_direct, (rows, num_q_heads, head_dim), np.float16)
        cpu = _cpu_reference(query, key, value, gate, live_counts, scale=scale)
        np.testing.assert_array_equal(direct.view(np.uint16), split.view(np.uint16))
        np.testing.assert_allclose(direct.astype(np.float32), cpu.astype(np.float32), atol=3e-4, rtol=3e-3)
    finally:
        _free_all(_runtime, bufs)
