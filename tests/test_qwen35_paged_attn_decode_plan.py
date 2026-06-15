from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    plan_qwen35_paged_attn_decode_build,
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans,
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans,
    qwen35_paged_attn_decode_int8_gqa_splitk_spans,
    qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans,
    qwen35_full_attn_decode_context_bf16,
    qwen35_full_attn_gate_mul_bf16,
    qwen35_full_attn_gate_mul_fp16,
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_f32_spans,
    qwen35_paged_full_attn_decode_split_k_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans,
    qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans,
    register_qwen35_paged_attn_decode_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata


def setup_function() -> None:
    clear_registry_for_tests()


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _spans(*, storage_dtype: str = "bf16", live_dtype: str = "int64") -> KVLiveSpans:
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (1,), "int32"),
        live_counts=_tensor(0x2000, (1,), live_dtype),
        max_live_count=2,
        storage_dtype=storage_dtype,
    )


def _int8_spans(
    *,
    block_table_shape: tuple[int, ...] = (1,),
    scale_shape: tuple[int, ...] = (1, 256, 2),
    live_dtype: str = "int64",
    scale_dtype: str = "fp16",
) -> KVLiveSpans:
    metadata = KVScaleMetadata(
        k_scale=_tensor(0x3000, scale_shape, scale_dtype),
        v_scale=_tensor(0x4000, scale_shape, scale_dtype),
        scale_dtype=scale_dtype,
    )
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, block_table_shape, "int32"),
        live_counts=_tensor(0x2000, (1,), live_dtype),
        max_live_count=2,
        storage_dtype="int8_per_token_head",
        scale_metadata=metadata,
    )


def test_qwen35_paged_attn_decode_registers_span_variant() -> None:
    register_qwen35_paged_attn_decode_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_gate_mul",
            quant="w4_paro",
            variant="bf16",
        )
        is qwen35_full_attn_gate_mul_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_gate_mul",
            quant="w4_paro",
            variant="fp16",
        )
        is qwen35_full_attn_gate_mul_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_decode",
            quant="w4_paro",
            variant="bf16_context",
        )
        is qwen35_full_attn_decode_context_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_context_spans",
        )
        is qwen35_paged_full_attn_decode_context_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_context_batch_spans",
        )
        is qwen35_paged_full_attn_decode_context_bf16_batch_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gate_f32_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gate_f32_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_warp_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_warp_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_warp_gate_bf16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_warp_gate_fp16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_bf16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_fp16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_fp16_batch_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gqa_gate_fp16_batch_direct_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gate_bf16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_split_k_gate_fp16_spans",
        )
        is qwen35_paged_full_attn_decode_split_k_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="gqa_splitk_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="gqa_splitk_gate_bf16_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="gqa_splitk_gate_fp16_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="per_token_head_gqa_splitk_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="per_token_head_gqa_splitk_gate_bf16_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_decode",
            quant="int8_per_token_head",
            variant="per_token_head_gqa_splitk_gate_fp16_spans",
        )
        is qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="qwen35_causal_gqa_gate_fp16",
        )
        is qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_prefill",
            quant="w4_paro",
            variant="bf16_gqa_gate_fp16_spans",
        )
        is qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paged_attn_prefill",
            quant="int8_per_token_head",
            variant="per_token_head_gqa_gate_fp16_spans",
        )
        is qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="qwen35_tree_gqa_gate_fp16",
        )
        is qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="qwen35_varlen_causal_gqa_gate_fp16",
        )
        is qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans
    )


def test_qwen35_paged_attn_decode_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_paged_attn_decode_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 paged attn decode test version",
    )

    assert artifact.family == "qwen35_paged_attn_decode"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_paged_attn_decode.so"
    assert artifact.compiler_version == "hipcc qwen35 paged attn decode test version"
    assert any(str(path).endswith("paged_attn_decode.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_paged_attn_decode_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="total must be positive"):
        qwen35_full_attn_gate_mul_bf16(0, 0, 0, 0)
    with pytest.raises(ValueError, match="total must be positive"):
        qwen35_full_attn_gate_mul_fp16(0, 0, 0, 0)
    with pytest.raises(ValueError, match="max_context_len must be positive"):
        qwen35_full_attn_decode_context_bf16(0, 0, 0, 0, 0, 0, 2, 1, 4, 1.0)
    with pytest.raises(ValueError, match="num_q_heads must be divisible"):
        qwen35_full_attn_decode_context_bf16(0, 0, 0, 0, 0, 2, 3, 2, 4, 1.0)
    with pytest.raises(ValueError, match="block_size=256"):
        qwen35_paged_full_attn_decode_context_bf16_spans(0, 0, 0, 0, _spans(), 2, 4, 2, 1, 4, 1.0)
    with pytest.raises(ValueError, match="int64 live_counts"):
        qwen35_paged_full_attn_decode_context_bf16_spans(
            0, 0, 0, 0, _spans(live_dtype="int32"), 2, 256, 2, 1, 4, 1.0
        )
    with pytest.raises(ValueError, match="bf16 storage"):
        qwen35_paged_full_attn_decode_context_bf16_spans(
            0, 0, 0, 0, _spans(storage_dtype="fp16"), 2, 256, 2, 1, 4, 1.0
        )
    with pytest.raises(ValueError, match="num_q_heads must be divisible"):
        qwen35_paged_full_attn_decode_context_bf16_spans(0, 0, 0, 0, _spans(), 2, 256, 3, 2, 4, 1.0)
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
            0, 0, 0, 0, _spans(), 0, 2, 256, 2, 1, 4, 1.0
        )
    with pytest.raises(ValueError, match="live_counts"):
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
            0,
            0,
            0,
            0,
            KVLiveSpans.paged_uniform(
                block_table=_tensor(0x1000, (2,), "int32"),
                live_counts=_tensor(0x2000, (1,), "int64"),
                max_live_count=2,
                storage_dtype="bf16",
            ),
            2,
            2,
            256,
            2,
            1,
            4,
            1.0,
        )
    with pytest.raises(ValueError, match="head_dim divisible by 8"):
        qwen35_paged_full_attn_decode_split_k_bf16_spans(
            0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 2, 1, 4, 1.0
        )
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 1, 2, 256, 2, 1, 8, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0x4000, 0, 0, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="tree_committed_count"):
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0x4000, -1, 1, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="ancestor_mask_ptr"):
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0, 0, 1, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="segments"):
        qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans(
            0, 0, 0, 0, 0, _spans(), 0, 0, 1, 0, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_full_attn_decode_split_k_gate_f32_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 2, 1, 8, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_full_attn_decode_split_k_gate_fp16_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 2, 1, 8, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="Qwen3.5 GQA"):
        qwen35_paged_full_attn_decode_split_k_warp_bf16_spans(
            0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 8, 1, 256, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 2, 2, 256, 16, 2, 256, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="rows"):
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 0, 2, 2, 256, 16, 2, 256, 256, 1, 1.0
        )
    with pytest.raises(ValueError, match="Qwen3.5 GQA"):
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans(
            0, 0, 0, 0, 0, 0, 0, 0, _spans(), 1, 2, 2, 256, 8, 1, 256, 256, 1, 1.0
        )
    with pytest.raises(ValueError, match="num_splits=1"):
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans(
            0, 0, 0, 0, 0, _spans(), 1, 2, 2, 256, 16, 2, 256, 256, 1, 1.0
        )
    int8_spans = _int8_spans()
    with pytest.raises(ValueError, match="int8_per_token_head storage"):
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, _spans(), 256, 1, 256, 16, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="int64 live_counts"):
        bad_live = _int8_spans(live_dtype="int32")
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, bad_live.scale_metadata.k_scale.ptr, bad_live.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, bad_live, 256, 1, 256, 16, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="Qwen3.5 INT8 GQA"):
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, int8_spans, 256, 1, 256, 8, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="k_scale_ptr"):
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr + 1, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, int8_spans, 256, 1, 256, 16, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="scale tensor shape"):
        bad_scale_shape = _int8_spans(scale_shape=(1, 128, 2))
        qwen35_paged_attn_decode_int8_gqa_splitk_spans(
            0, 0, 0, bad_scale_shape.scale_metadata.k_scale.ptr, bad_scale_shape.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, bad_scale_shape, 256, 1, 256, 16, 2, 256, 1.0
        )
    with pytest.raises(ValueError, match="gate_stride1"):
        qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, 0, 0, 0, int8_spans, 256, 1, 256, 16, 2, 256, 0, 1, 1.0
        )
    with pytest.raises(ValueError, match="int8_per_token_head storage"):
        qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, _spans(), 1, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
    with pytest.raises(ValueError, match="head_dim <= 256"):
        qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, int8_spans, 1, 2, 256, 2, 1, 512, 512, 1, 1.0
        )
    with pytest.raises(ValueError, match="k_scale_ptr"):
        qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(
            0, 0, 0, int8_spans.scale_metadata.k_scale.ptr + 1, int8_spans.scale_metadata.v_scale.ptr,
            0, 0, int8_spans, 1, 2, 256, 2, 1, 8, 8, 1, 1.0
        )
