from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import (
    PagedAttnDecodeKind,
    PagedAttnPrefillKind,
    PagedKVWriteKind,
    plan_paged_attn_decode,
    plan_paged_attn_prefill,
    plan_paged_kv_write,
    resolve_paged_attn_decode,
    resolve_paged_attn_prefill,
    resolve_paged_kv_write,
)
from hipengine.kernels.hip_gfx1100.attention import (
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans,
    qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
    qwen35_write_paged_kv_int8_per_token_head_batch_spans,
    qwen35_write_paged_kv_int8_per_token_head_prompt_spans,
    qwen35_write_paged_kv_int8_per_token_head_spans,
    register_qwen35_paged_attn_decode_kernels,
    register_qwen35_paged_kv_write_kernels,
)
from hipengine.kernels.registry import (
    DuplicateKernelError,
    KernelKey,
    MissingKernelError,
    clear_registry_for_tests,
)
from hipengine.kvcache import FixedPagedKVPolicy, KVLiveSpans, KVScaleMetadata


def setup_function() -> None:
    clear_registry_for_tests()


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _bf16_policy_spans() -> KVLiveSpans:
    policy = FixedPagedKVPolicy(storage_dtype="bf16")
    policy.register(
        7,
        block_table=_tensor(0x1000, (1,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        max_live_count=2,
        capacity_tokens=256,
    )
    return policy.batch_spans([7])


def _int8_policy_spans() -> KVLiveSpans:
    metadata = KVScaleMetadata(
        k_scale=_tensor(0x3000, (1, 256, 2), "fp16"),
        v_scale=_tensor(0x4000, (1, 256, 2), "fp16"),
        scale_dtype="fp16",
    )
    policy = FixedPagedKVPolicy(storage_dtype="int8_per_token_head")
    policy.register(
        7,
        block_table=_tensor(0x1000, (1,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        max_live_count=2,
        capacity_tokens=256,
        scale_metadata=metadata,
    )
    return policy.batch_spans([7])


def test_paged_kv_write_resolution_uses_int8_policy_metadata() -> None:
    spans = _int8_policy_spans()
    register_qwen35_paged_kv_write_kernels()

    decode = plan_paged_kv_write(spans, kind=PagedKVWriteKind.DECODE, source_dtype="fp32")
    prompt = plan_paged_kv_write(spans, kind="prompt", source_dtype="fp32")
    batch = plan_paged_kv_write(spans, kind="batch", source_dtype="fp32")

    assert decode.key("hip_gfx1100") == KernelKey(
        "hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_spans"
    )
    assert prompt.key("hip_gfx1100") == KernelKey(
        "hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_prompt_spans"
    )
    assert batch.key("hip_gfx1100") == KernelKey(
        "hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_batch_spans"
    )
    assert (
        resolve_paged_kv_write(backend="hip_gfx1100", spans=spans, kind="decode", source_dtype="fp32")
        is qwen35_write_paged_kv_int8_per_token_head_spans
    )
    assert (
        resolve_paged_kv_write(backend="hip_gfx1100", spans=spans, kind="prompt", source_dtype="fp32")
        is qwen35_write_paged_kv_int8_per_token_head_prompt_spans
    )
    assert (
        resolve_paged_kv_write(backend="hip_gfx1100", spans=spans, kind="batch", source_dtype="fp32")
        is qwen35_write_paged_kv_int8_per_token_head_batch_spans
    )


def test_paged_attn_decode_resolution_uses_storage_aware_keys() -> None:
    int8_spans = _int8_policy_spans()
    bf16_spans = _bf16_policy_spans()
    register_qwen35_paged_attn_decode_kernels()

    int8 = plan_paged_attn_decode(
        int8_spans,
        kind=PagedAttnDecodeKind.GQA_SPLITK_GATE_FP16,
        model_quant="w4_paro",
    )
    bf16 = plan_paged_attn_decode(
        bf16_spans,
        kind=PagedAttnDecodeKind.GQA_SPLITK_GATE_FP16,
        model_quant="w4_paro",
    )

    assert int8.key("hip_gfx1100") == KernelKey(
        "hip_gfx1100",
        "paged_attn_decode",
        "int8_per_token_head",
        "per_token_head_gqa_splitk_gate_fp16_spans",
    )
    assert bf16.key("hip_gfx1100") == KernelKey(
        "hip_gfx1100",
        "paged_attn_decode",
        "w4_paro",
        "bf16_split_k_gqa_gate_fp16_spans",
    )
    assert (
        resolve_paged_attn_decode(
            backend="hip_gfx1100",
            spans=int8_spans,
            kind="gqa_splitk_gate_fp16",
            model_quant="w4_paro",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans
    )
    assert (
        resolve_paged_attn_decode(
            backend="hip_gfx1100",
            spans=bf16_spans,
            kind="gqa_splitk_gate_fp16",
            model_quant="w4_paro",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans
    )


def test_paged_attn_prefill_resolution_uses_storage_aware_keys() -> None:
    int8_spans = _int8_policy_spans()
    bf16_spans = _bf16_policy_spans()
    register_qwen35_paged_attn_decode_kernels()

    int8 = plan_paged_attn_prefill(
        int8_spans,
        kind=PagedAttnPrefillKind.GQA_GATE_FP16,
        model_quant="w4_paro",
    )
    bf16 = plan_paged_attn_prefill(
        bf16_spans,
        kind="gqa_gate_fp16",
        model_quant="w4_paro",
    )

    assert int8.key("hip_gfx1100") == KernelKey(
        "hip_gfx1100",
        "paged_attn_prefill",
        "int8_per_token_head",
        "per_token_head_gqa_gate_fp16_spans",
    )
    assert bf16.key("hip_gfx1100") == KernelKey(
        "hip_gfx1100",
        "paged_attn_prefill",
        "w4_paro",
        "bf16_gqa_gate_fp16_spans",
    )
    assert (
        resolve_paged_attn_prefill(
            backend="hip_gfx1100",
            spans=int8_spans,
            kind="gqa_gate_fp16",
            model_quant="w4_paro",
        )
        is qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans
    )
    assert (
        resolve_paged_attn_prefill(
            backend="hip_gfx1100",
            spans=bf16_spans,
            kind="gqa_gate_fp16",
            model_quant="w4_paro",
        )
        is qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans
    )


def test_storage_aware_dispatch_preserves_missing_and_duplicate_errors() -> None:
    spans = _int8_policy_spans()

    with pytest.raises(MissingKernelError):
        resolve_paged_attn_decode(
            backend="hip_gfx1100",
            spans=spans,
            kind="gqa_splitk_gate_fp16",
            model_quant="w4_paro",
        )

    register_qwen35_paged_attn_decode_kernels(replace=False)
    with pytest.raises(DuplicateKernelError):
        register_qwen35_paged_attn_decode_kernels(replace=False)
