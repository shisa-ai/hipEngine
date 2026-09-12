from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.dispatch import (
    PagedAttnDecodeKind,
    PagedAttnPrefillKind,
    PagedKVWriteKind,
    plan_paged_attn_decode,
    plan_paged_attn_prefill,
    plan_paged_kv_write,
)
from hipengine.kernels.cpu_reference import (
    dequantize_kv_int8_hadamard_group32,
    paged_attn_decode_int8_hadamard_group32,
    quantize_kv_int8_hadamard_group32,
    write_paged_kv_int8_hadamard_group32,
)
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata, resolve_kv_policy
from hipengine.kernels.hip_gfx1100.attention import (
    build_qwen35_paged_attn_decode,
    build_qwen35_paged_kv_write,
    qwen35_paged_attn_decode_int8_hadamard_group32_gqa_splitk_spans,
    qwen35_write_paged_kv_int8_hadamard_group32_prompt_spans,
)
from hipengine.kernels.registry import KernelKey


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str | DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _hadamard_spans() -> KVLiveSpans:
    metadata = KVScaleMetadata(
        k_scale=_tensor(0x3000, (2, 256, 2, 8), DType.FP16),
        v_scale=_tensor(0x4000, (2, 256, 2, 8), DType.FP16),
        scale_dtype=DType.FP16,
        granularity="hadamard_group32",
    )
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2,), DType.INT32),
        live_counts=_tensor(0x2000, (1,), DType.INT64),
        max_live_count=257,
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        scale_metadata=metadata,
    )


def test_tail4_hadamard_policy_selects_six_bf16_then_four_int8_layers() -> None:
    resolved = resolve_kv_policy("tail4_hadamard_group32", scale_dtype="fp16")
    policy = resolved.create_policy()

    assert resolved.storage_layout == "tail4_hadamard_group32"
    assert resolved.quantized_tail_layers == 4
    assert resolved.scale_granularity == "hadamard_group32"
    assert [policy.full_attention_storage_dtype(index, 10) for index in range(10)] == [
        *([DType.BF16] * 6),
        *([DType.INT8_PER_TOKEN_HEAD] * 4),
    ]
    assert [policy.full_attention_scale_granularity(index, 10) for index in range(10)] == [
        *([None] * 6),
        *(["hadamard_group32"] * 4),
    ]


def test_hadamard_group32_cpu_codec_matches_hand_transform_and_page_attention() -> None:
    base = np.arange(1, 33, dtype=np.float32)
    key = np.stack((base, -base), axis=0).reshape(2, 1, 32)
    value = np.stack((base[::-1], base), axis=0).reshape(2, 1, 32)

    qk, qv, k_scale, v_scale = quantize_kv_int8_hadamard_group32(key, value, scale_dtype=np.float16)
    key_deq, value_deq = dequantize_kv_int8_hadamard_group32(qk, qv, k_scale, v_scale)

    assert qk.dtype == np.int8 and qv.dtype == np.int8
    assert k_scale.shape == (2, 1, 1)
    assert v_scale.shape == (2, 1, 1)
    assert np.max(np.abs(key_deq - key)) < 0.35
    assert np.max(np.abs(value_deq - value)) < 0.35

    cache = write_paged_kv_int8_hadamard_group32(
        key,
        value,
        positions=np.asarray([0, 1], dtype=np.int64),
        block_table=np.asarray([1, 0], dtype=np.int32),
        block_size=1,
        scale_dtype=np.float16,
    )
    query = np.ones((1, 32), dtype=np.float32)
    candidate = paged_attn_decode_int8_hadamard_group32(
        query,
        *cache,
        live_counts=np.asarray([2], dtype=np.int64),
        block_table=np.asarray([1, 0], dtype=np.int32),
        block_size=1,
        scale=32**-0.5,
    )
    logits = np.einsum("hd,thd->ht", query, key_deq, optimize=True) * np.float32(32**-0.5)
    weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    expected = np.einsum("ht,thd->hd", weights, value_deq, optimize=True)

    assert np.allclose(candidate, expected, atol=2e-5, rtol=2e-5)


def test_hadamard_group32_dispatch_is_metadata_and_registry_keyed() -> None:
    spans = _hadamard_spans()

    assert plan_paged_kv_write(
        spans,
        kind=PagedKVWriteKind.DECODE,
        source_dtype=DType.FP32,
    ).key("hip_gfx1100") == KernelKey(
        "hip_gfx1100",
        "paged_kv_write",
        "int8_hadamard_group32",
        "hadamard_group32_spans",
    )
    assert plan_paged_kv_write(
        spans,
        kind=PagedKVWriteKind.PROMPT,
        source_dtype=DType.FP32,
    ).key("hip_gfx1100") == KernelKey(
        "hip_gfx1100",
        "paged_kv_write",
        "int8_hadamard_group32",
        "hadamard_group32_prompt_spans",
    )
    assert plan_paged_attn_decode(
        spans,
        kind=PagedAttnDecodeKind.GQA_SPLITK_GATE_BF16,
    ).key("hip_gfx1100") == KernelKey(
        "hip_gfx1100",
        "paged_attn_decode",
        "int8_hadamard_group32",
        "hadamard_group32_gqa_splitk_gate_bf16_spans",
    )
    assert plan_paged_attn_prefill(
        spans,
        kind=PagedAttnPrefillKind.GQA_GATE_FP16,
    ).key("hip_gfx1100") == KernelKey(
        "hip_gfx1100",
        "paged_attn_prefill",
        "int8_hadamard_group32",
        "hadamard_group32_gqa_gate_fp16_spans",
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_hadamard_group32_gfx1100_writer_and_decode_match_cpu_reference() -> None:
    runtime = get_hip_runtime()
    write_library = build_qwen35_paged_kv_write(load=True)
    decode_library = build_qwen35_paged_attn_decode(load=True)
    rng = np.random.default_rng(20260713)
    rows, block_size, kv_heads, head_dim = 4, 256, 2, 256
    q_heads, chunk_size, num_splits = 16, 8, 1
    key = rng.normal(0.0, 0.4, size=(rows, kv_heads, head_dim)).astype(np.float32)
    value = rng.normal(0.0, 0.4, size=(rows, kv_heads, head_dim)).astype(np.float32)
    query = rng.normal(0.0, 0.4, size=(q_heads, head_dim)).astype(np.float32)
    prompt_table = np.zeros((rows,), dtype=np.int32)
    decode_table = np.zeros((1,), dtype=np.int32)
    positions = np.arange(rows, dtype=np.int64)
    context = np.asarray([rows], dtype=np.int64)
    cache_shape = (1, block_size, kv_heads, head_dim)
    scale_shape = (1, block_size, kv_heads, head_dim // 32)
    key_cache = np.zeros(cache_shape, dtype=np.int8)
    value_cache = np.zeros(cache_shape, dtype=np.int8)
    k_scale = np.zeros(scale_shape, dtype=np.float16)
    v_scale = np.zeros(scale_shape, dtype=np.float16)
    out = np.zeros((q_heads, head_dim), dtype=np.float32)
    partial_out = np.zeros((q_heads, num_splits, head_dim), dtype=np.float32)
    partial_m = np.zeros((q_heads, num_splits), dtype=np.float32)
    partial_l = np.zeros((q_heads, num_splits), dtype=np.float32)
    buffers = []

    def upload(array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
        return buffer

    try:
        key_buf = upload(key)
        value_buf = upload(value)
        query_buf = upload(query)
        prompt_table_buf = upload(prompt_table)
        decode_table_buf = upload(decode_table)
        positions_buf = upload(positions)
        context_buf = upload(context)
        key_cache_buf = upload(key_cache)
        value_cache_buf = upload(value_cache)
        k_scale_buf = upload(k_scale)
        v_scale_buf = upload(v_scale)
        out_buf = upload(out)
        partial_out_buf = upload(partial_out)
        partial_m_buf = upload(partial_m)
        partial_l_buf = upload(partial_l)
        prompt_metadata = KVScaleMetadata(
            k_scale=_tensor(k_scale_buf.ptr, scale_shape, DType.FP16),
            v_scale=_tensor(v_scale_buf.ptr, scale_shape, DType.FP16),
            scale_dtype=DType.FP16,
            granularity="hadamard_group32",
        )
        prompt_spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(prompt_table_buf.ptr, prompt_table.shape, DType.INT32),
            live_counts=_tensor(positions_buf.ptr, positions.shape, DType.INT64),
            max_live_count=rows - 1,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            span_role="prefill",
            scale_metadata=prompt_metadata,
        )
        qwen35_write_paged_kv_int8_hadamard_group32_prompt_spans(
            key_buf.ptr,
            value_buf.ptr,
            key_cache_buf.ptr,
            value_cache_buf.ptr,
            k_scale_buf.ptr,
            v_scale_buf.ptr,
            prompt_spans,
            rows,
            block_size,
            kv_heads,
            head_dim,
            library=write_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in (
            (key_cache, key_cache_buf),
            (value_cache, value_cache_buf),
            (k_scale, k_scale_buf),
            (v_scale, v_scale_buf),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)

        expected_cache = write_paged_kv_int8_hadamard_group32(
            key,
            value,
            positions,
            decode_table,
            block_size=block_size,
            scale_dtype=np.float16,
        )
        np.testing.assert_array_equal(key_cache, expected_cache[0])
        np.testing.assert_array_equal(value_cache, expected_cache[1])
        np.testing.assert_array_equal(k_scale, expected_cache[2])
        np.testing.assert_array_equal(v_scale, expected_cache[3])

        decode_metadata = KVScaleMetadata(
            k_scale=_tensor(k_scale_buf.ptr, scale_shape, DType.FP16),
            v_scale=_tensor(v_scale_buf.ptr, scale_shape, DType.FP16),
            scale_dtype=DType.FP16,
            granularity="hadamard_group32",
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(decode_table_buf.ptr, decode_table.shape, DType.INT32),
            live_counts=_tensor(context_buf.ptr, context.shape, DType.INT64),
            max_live_count=rows,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=decode_metadata,
        )
        qwen35_paged_attn_decode_int8_hadamard_group32_gqa_splitk_spans(
            query_buf.ptr,
            key_cache_buf.ptr,
            value_cache_buf.ptr,
            k_scale_buf.ptr,
            v_scale_buf.ptr,
            out_buf.ptr,
            partial_out_buf.ptr,
            partial_m_buf.ptr,
            partial_l_buf.ptr,
            decode_spans,
            chunk_size,
            num_splits,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            head_dim**-0.5,
            library=decode_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    expected = paged_attn_decode_int8_hadamard_group32(
        query,
        *expected_cache,
        live_counts=context,
        block_table=decode_table,
        block_size=block_size,
        scale=head_dim**-0.5,
    )
    np.testing.assert_allclose(out, expected, atol=2e-4, rtol=2e-4)
