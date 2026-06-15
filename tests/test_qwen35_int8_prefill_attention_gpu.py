from __future__ import annotations

import ctypes
import os
import pathlib

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


@pytest.fixture(scope="module")
def _attention_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_qwen35_paged_attn_decode(load=True, compiler_version=compiler_version)


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array)
    buf = malloc(max(arr.nbytes, 4), runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes, runtime=runtime)
    return buf


def _alloc(runtime, bufs, nbytes: int):
    from hipengine.core.memory import malloc

    buf = malloc(max(int(nbytes), 4), runtime=runtime)
    bufs.append(buf)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes, runtime=runtime)
    return arr


def _free(runtime, bufs) -> None:
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


def _f32_to_bf16_f32(arr: np.ndarray) -> np.ndarray:
    bits = np.asarray(arr, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _reference_int8_prefill(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    k_scale: np.ndarray,
    v_scale: np.ndarray,
    gate: np.ndarray,
    block_table: np.ndarray,
    context_counts: np.ndarray,
    row_positions: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    rows, num_q_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    q_per_kv = num_q_heads // num_kv_heads
    block_size = key_cache.shape[1]
    out = np.empty((rows, num_q_heads, head_dim), dtype=np.float32)
    query_bf16 = _f32_to_bf16_f32(query)
    for row in range(rows):
        visible = min(int(context_counts[row]), int(row_positions[row]) + 1)
        for q_head in range(num_q_heads):
            kv_head = q_head // q_per_kv
            scores = []
            values = []
            for token in range(visible):
                logical_block = token // block_size
                block_offset = token - logical_block * block_size
                physical_block = int(block_table[row, logical_block])
                key = key_cache[physical_block, block_offset, kv_head].astype(np.float32) * float(
                    k_scale[physical_block, block_offset, kv_head]
                )
                value = value_cache[physical_block, block_offset, kv_head].astype(np.float32) * float(
                    v_scale[physical_block, block_offset, kv_head]
                )
                scores.append(float(np.dot(query_bf16[row, q_head], key) * scale))
                values.append(value)
            score_arr = np.asarray(scores, dtype=np.float32)
            weights = np.exp(score_arr - np.max(score_arr))
            weights /= np.sum(weights)
            attn = np.sum(np.asarray(values, dtype=np.float32) * weights[:, None], axis=0)
            attn_bf16 = _f32_to_bf16_f32(attn)
            gate_f32 = gate[row, q_head].astype(np.float32)
            out[row, q_head] = attn_bf16 * (1.0 / (1.0 + np.exp(-gate_f32)))
    return out.astype(np.float16).reshape(rows, num_q_heads * head_dim)


def test_qwen35_int8_prefill_attention_matches_numpy_reference(_runtime, _attention_lib) -> None:
    from hipengine.core.device import Device
    from hipengine.core.dtype import DType
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans
    from hipengine.kvcache import KVLiveSpans, KVScaleMetadata

    rng = np.random.default_rng(0x8835)
    rows = 5
    block_size = 4
    blocks = 2
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 16
    scale = head_dim ** -0.5

    query = (rng.standard_normal((rows, num_q_heads, head_dim)) * 0.15).astype(np.float32)
    key_cache = rng.integers(-12, 13, size=(blocks, block_size, num_kv_heads, head_dim), dtype=np.int8)
    value_cache = rng.integers(-10, 11, size=(blocks, block_size, num_kv_heads, head_dim), dtype=np.int8)
    k_scale = rng.uniform(0.008, 0.025, size=(blocks, block_size, num_kv_heads)).astype(np.float32)
    v_scale = rng.uniform(0.006, 0.02, size=(blocks, block_size, num_kv_heads)).astype(np.float32)
    gate = (rng.standard_normal((rows, num_q_heads, head_dim)) * 0.25).astype(np.float16)
    block_table = np.tile(np.arange(blocks, dtype=np.int32), (rows, 1))
    context_counts = np.full((rows,), rows, dtype=np.int64)
    row_positions = np.arange(rows, dtype=np.int64)
    expected = _reference_int8_prefill(
        query,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        gate,
        block_table,
        context_counts,
        row_positions,
        scale=scale,
    )

    bufs = []
    try:
        query_dev = _upload(_runtime, bufs, query)
        key_dev = _upload(_runtime, bufs, key_cache)
        value_dev = _upload(_runtime, bufs, value_cache)
        k_scale_dev = _upload(_runtime, bufs, k_scale)
        v_scale_dev = _upload(_runtime, bufs, v_scale)
        gate_dev = _upload(_runtime, bufs, gate)
        table_dev = _upload(_runtime, bufs, block_table)
        counts_dev = _upload(_runtime, bufs, context_counts)
        positions_dev = _upload(_runtime, bufs, row_positions)
        out_dev = _alloc(_runtime, bufs, expected.nbytes)

        device = Device("hip", 0)
        metadata = KVScaleMetadata(
            k_scale=Tensor.from_handle(k_scale_dev.ptr, k_scale.shape, DType.FP32, device),
            v_scale=Tensor.from_handle(v_scale_dev.ptr, v_scale.shape, DType.FP32, device),
            scale_dtype=DType.FP32,
        )
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(table_dev.ptr, block_table.shape, DType.INT32, device),
            live_counts=Tensor.from_handle(counts_dev.ptr, context_counts.shape, DType.INT64, device),
            max_live_count=rows,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            row_positions=Tensor.from_handle(positions_dev.ptr, row_positions.shape, DType.INT64, device),
            span_role="prefill",
            scale_metadata=metadata,
        )

        qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(
            query_dev.ptr,
            key_dev.ptr,
            value_dev.ptr,
            k_scale_dev.ptr,
            v_scale_dev.ptr,
            gate_dev.ptr,
            out_dev.ptr,
            spans,
            rows,
            rows,
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
        actual = _download(_runtime, out_dev, expected.shape, np.float16)
    finally:
        _free(_runtime, bufs)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), rtol=2.0e-2, atol=2.0e-3)
