"""Bulk full-attention execution for GGUF prefill rows.

This module owns the attention half of one bulk prefill layer: attention norm,
Q/K/V projection, q/gate split, head norm and partial RoPE, KV write, the
direct-INT8/AOTriton/native attention dispatch, gate multiply and the attention
output projection. The caller owns the FFN half, the scratch allocations, KV
planes, split-growth buffers, DMS capture sink and AOTriton stream bridge.

All policy and selected operations arrive through explicit records. Nothing
here reads a runner, a session, or module globals for ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.kernels.hip_gfx1100.attention import (
    aotriton_attn_fwd_compact_varlen,
    aotriton_attn_fwd_v3_compact_varlen,
)
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import (
    tensor1 as aotriton_tensor1,
    tensor2 as aotriton_tensor2,
    tensor4 as aotriton_tensor4,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_full_attn_gate_mul_bf16_to_bf16,
    qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_flash_spans,
    qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_spans,
    qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_wmma_spans,
    qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans,
    qwen35_write_paged_kv_mixed_value_bf16_prompt_spans,
)
from hipengine.kernels.hip_gfx1100.convert import (
    bf16_to_f32,
    bf16_to_fp16,
    f32_to_bf16,
    fp16_to_bf16,
)
from hipengine.kernels.hip_gfx1100.fused import (
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import qwen35_split_qgate_bf16
from hipengine.runtime.gguf_linear import (
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_triple,
)
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_layer_attn import AttentionGeometry, StageRecorder


class PrefillStreamBridge(Protocol):
    """The caller-owned isolated AOTriton queue for one dispatch."""

    stream: int

    def wait_for_inputs(self, runtime: HipRuntime, stream: int) -> None: ...

    def release_output(self, runtime: HipRuntime, stream: int) -> None: ...


@dataclass(frozen=True)
class BulkFullAttentionViews:
    """Activation, KV and AOTriton buffers for one bulk prefill layer.

    Every buffer is owned by the caller's scratch/session. This record only
    binds the views the attention body reads and writes.
    """

    norm: object
    full_q: object
    full_k: object
    full_v: object
    full_query_raw: object
    full_key_raw: object
    full_query: object
    full_key: object
    full_query_bf16: object | None
    full_gate: object
    full_gated: object
    full_attn_bf16: object | None
    attn_out: object
    positions_tensor: object
    key_cache: object | None
    value_cache: object | None
    retained_key_cache: object | None
    retained_value_cache: object | None
    append_spans: object
    prefill_spans: object
    retained_append_spans: object | None
    block_size: int
    max_positions: int
    int8_kv_value_bf16: bool
    cu_q: object | None
    cu_k: object | None
    softmax_lse: object | None
    atomic: object | None
    head_major_key_cache: object | None
    head_major_value_cache: object | None
    head_major_kv_capacity: int
    head_major_kv_dense_prefix: bool
    full_attn_split_batch_rows: int


@dataclass(frozen=True)
class BulkFullAttentionSplitView:
    """Current split-K buffers after the caller's growth callback."""

    partial: object
    m: object
    l: object
    batch_rows: int
    count: int


@dataclass(frozen=True)
class BulkFullAttentionSelectors:
    """Lazily resolved libraries and selected operations for one call."""

    cast_library: Callable[[], object]
    paged_kv_write_library: Callable[[], object]
    paged_attn_decode_library: Callable[[], object]
    aotriton_prefill_library: Callable[[], object]
    int8_kv_prompt_write_fn: Callable[[object], Callable[..., None]]
    try_launch_dense_q8_single_dp4a: Callable[..., bool]
    native_prefill_fn: Callable[[], Callable[..., None]]
    head_major_buffers: Callable[..., tuple[object, object, int] | None]
    paged_kv_copy_resolve: Callable[..., Callable[..., None] | None]
    ensure_split_rows: Callable[[int], BulkFullAttentionSplitView]


@dataclass(frozen=True)
class BulkFullAttentionPolicy:
    """Backend and kernel-choice predicates resolved by the caller."""

    aotriton_prefill_allowed: Callable[[str], bool]
    aotriton_prefill_mode: Callable[[int, int, int], str]
    int8_prefill_kernel: Callable[..., str]


def run_full_attention_prefill_attn_rows(
    layer_id: int,
    layer,
    hidden_ptr: int,
    views: BulkFullAttentionViews,
    *,
    cfg,
    geometry: AttentionGeometry,
    backend: str,
    runtime: HipRuntime,
    rows: int,
    start: int,
    selectors: BulkFullAttentionSelectors,
    policy: BulkFullAttentionPolicy,
    cos_table_ptr: int,
    sin_table_ptr: int,
    max_positions: int,
    stream: int = 0,
    allow_aotriton: bool = True,
    aotriton_min_tokens: int | None = None,
    paged_max_context_len: int | None = None,
    aotriton_bridge: PrefillStreamBridge | None = None,
    capture_qk: Callable[[], None] | None = None,
    stage_prefix: str = "prefill_full_attn",
    gpu_stage_recorder: StageRecorder | None = None,
) -> bool:
    """Run the attention half of one bulk prefill layer."""

    cast_library = selectors.cast_library()
    kv_write_library = selectors.paged_kv_write_library()
    gguf_rmsnorm_bf16_f32_weight(
        hidden_ptr,
        layer.weight("attn_norm").allocation().tensor.ptr,
        views.norm.ptr,
        rows=rows,
        hidden_size=geometry.hidden_size,
        eps=cfg.rms_norm_eps,
        stream=stream,
        runtime=runtime,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_attn_norm")
    if not launch_gguf_linear_triple(
        layer.weight("attn_q"),
        layer.weight("attn_k"),
        layer.weight("attn_v"),
        views.norm.ptr,
        views.full_q.ptr,
        views.full_k.ptr,
        views.full_v.ptr,
        rows=rows,
        in_features=geometry.hidden_size,
        out_features=2 * geometry.q_width,
        out_features_b=geometry.kv_width,
        out_features_c=geometry.kv_width,
        stream=stream,
        runtime=runtime,
    ):
        launch_gguf_linear(
            layer.weight("attn_q"),
            views.norm.ptr,
            views.full_q.ptr,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features=2 * geometry.q_width,
            stream=stream,
            runtime=runtime,
        )
        if not launch_gguf_linear_pair(
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            views.norm.ptr,
            views.full_k.ptr,
            views.full_v.ptr,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features=geometry.kv_width,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("attn_k"),
                views.norm.ptr,
                views.full_k.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=geometry.kv_width,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("attn_v"),
                views.norm.ptr,
                views.full_v.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=geometry.kv_width,
                stream=stream,
                runtime=runtime,
            )
    qwen35_split_qgate_bf16(
        views.full_q.ptr,
        views.full_query_raw.ptr,
        views.full_gate.ptr,
        rows,
        cfg.head_count,
        cfg.key_length,
        stream=stream,
        runtime=runtime,
    )
    bf16_to_f32(
        views.full_k.ptr,
        views.full_key_raw.ptr,
        rows * geometry.kv_width,
        stream=stream,
        library=cast_library,
        runtime=runtime,
    )
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
        views.full_query_raw.ptr,
        views.full_key_raw.ptr,
        layer.weight("attn_q_norm").allocation().tensor.ptr,
        layer.weight("attn_k_norm").allocation().tensor.ptr,
        cos_table_ptr,
        sin_table_ptr,
        views.positions_tensor.ptr,
        views.full_query.ptr,
        views.full_key.ptr,
        cfg.rms_norm_eps,
        rows,
        cfg.head_count,
        cfg.head_count_kv,
        cfg.key_length,
        cfg.rope_dimension_count,
        max_positions,
        stream=stream,
        runtime=runtime,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_qkv_head_norm_rope")
    if capture_qk is not None:
        capture_qk()
    if views.key_cache is None or views.value_cache is None:
        raise RuntimeError(
            "GGUF full-attention prefill requires cache-backed key/value buffers; "
            "resident bulk prefill should provide either retained BF16 caches or "
            "a BF16 oracle cache for INT8 retention"
        )
    append_metadata = views.append_spans.scale_metadata
    direct_hadamard_int8 = (
        views.append_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
        and append_metadata is not None
        and append_metadata.granularity == "hadamard_group32"
    )
    direct_per_token_int8 = (
        views.append_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
        and append_metadata is not None
        and append_metadata.granularity == "per_token_head"
        and views.prefill_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
    )
    direct_int8_prefill = direct_hadamard_int8 or direct_per_token_int8
    if direct_int8_prefill:
        if (
            views.prefill_spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD
            or views.prefill_spans.scale_metadata is not append_metadata
        ):
            raise RuntimeError(
                "GGUF direct INT8 prefill requires matching append/attention metadata"
            )
        bf16_to_f32(
            views.full_v.ptr,
            views.full_key_raw.ptr,
            rows * geometry.kv_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        int8_prompt_write_fn = selectors.int8_kv_prompt_write_fn(append_metadata)
        int8_prompt_write_fn(
            views.full_key.ptr,
            views.full_key_raw.ptr,
            views.key_cache.ptr,
            views.value_cache.ptr,
            append_metadata.k_scale.ptr,
            append_metadata.v_scale.ptr,
            views.append_spans,
            rows,
            views.block_size,
            cfg.head_count_kv,
            cfg.key_length,
            stream=stream,
            library=kv_write_library,
            runtime=runtime,
        )
    else:
        qwen35_write_paged_kv_mixed_value_bf16_prompt_spans(
            views.full_key.ptr,
            views.full_v.ptr,
            views.key_cache.ptr,
            views.value_cache.ptr,
            views.append_spans,
            rows,
            views.block_size,
            cfg.head_count_kv,
            cfg.key_length,
            stream=stream,
            library=kv_write_library,
            runtime=runtime,
        )
        if views.retained_key_cache is not None or views.retained_value_cache is not None:
            if views.retained_key_cache is None or views.retained_value_cache is None:
                raise RuntimeError("GGUF INT8 retained prefill requires both key and value caches")
            retained_spans = views.retained_append_spans
            if retained_spans is None or retained_spans.scale_metadata is None:
                raise RuntimeError(
                    "GGUF INT8 retained prefill requires append spans with scale metadata"
                )
            bf16_to_f32(
                views.full_v.ptr,
                views.full_key_raw.ptr,
                rows * geometry.kv_width,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            if views.int8_kv_value_bf16:
                qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans(
                    views.full_key.ptr,
                    views.full_key_raw.ptr,
                    views.retained_key_cache.ptr,
                    views.retained_value_cache.ptr,
                    retained_spans.scale_metadata.k_scale.ptr,
                    retained_spans,
                    rows,
                    views.block_size,
                    cfg.head_count_kv,
                    cfg.key_length,
                    stream=stream,
                    library=kv_write_library,
                    runtime=runtime,
                )
            else:
                int8_prompt_write_fn = selectors.int8_kv_prompt_write_fn(
                    retained_spans.scale_metadata
                )
                int8_prompt_write_fn(
                    views.full_key.ptr,
                    views.full_key_raw.ptr,
                    views.retained_key_cache.ptr,
                    views.retained_value_cache.ptr,
                    retained_spans.scale_metadata.k_scale.ptr,
                    retained_spans.scale_metadata.v_scale.ptr,
                    retained_spans,
                    rows,
                    views.block_size,
                    cfg.head_count_kv,
                    cfg.key_length,
                    stream=stream,
                    library=kv_write_library,
                    runtime=runtime,
                )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_kv_write")
    threshold = int(
        PrefillConfig().attn_aotriton_min_tokens
        if aotriton_min_tokens is None
        else aotriton_min_tokens
    )
    if threshold < 0:
        raise ValueError("aotriton_min_tokens must be non-negative")
    use_aotriton = bool(
        allow_aotriton
        and not direct_int8_prefill
        and threshold > 0
        and rows >= threshold
        and policy.aotriton_prefill_allowed(backend)
    )
    paged_attn_library = selectors.paged_attn_decode_library()
    end = start + rows
    paged_context_len = end if paged_max_context_len is None else int(paged_max_context_len)
    if paged_context_len <= 0 or paged_context_len > int(views.max_positions):
        raise ValueError(
            "paged_max_context_len must be positive and within the prefill scratch capacity"
        )

    if direct_int8_prefill:
        # Direct INT8 leaves consume an FP16 gate; GGUF projections and qgate
        # splitting produce BF16. This row-local buffer is dead after attention,
        # so adapt it in place without new workspace.
        bf16_to_fp16(
            views.full_gate.ptr,
            views.full_gate.ptr,
            rows * cfg.head_count * cfg.key_length,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
    if direct_hadamard_int8:
        qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans(
            views.full_query.ptr,
            views.key_cache.ptr,
            views.value_cache.ptr,
            append_metadata.k_scale.ptr,
            append_metadata.v_scale.ptr,
            views.full_gate.ptr,
            views.full_gated.ptr,
            views.prefill_spans,
            rows,
            end,
            views.block_size,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.key_length,
            1,
            cfg.key_length ** -0.5,
            stream=stream,
            library=paged_attn_library,
            runtime=runtime,
        )
    elif direct_per_token_int8:
        # GQA-grouped flash kernel by default (cooperative LDS tile staging,
        # online softmax in registers); the sequential online-softmax kernel
        # stays selectable for A/B and as the numerics reference.
        kernel_choice = policy.int8_prefill_kernel(
            geometry=(cfg.head_count, cfg.head_count_kv, cfg.key_length),
            scale_dtype=append_metadata.scale_dtype,
        )
        if kernel_choice == "sequential":
            qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_spans(
                views.full_query.ptr,
                views.key_cache.ptr,
                views.value_cache.ptr,
                append_metadata.k_scale.ptr,
                append_metadata.v_scale.ptr,
                views.full_gate.ptr,
                views.full_gated.ptr,
                views.prefill_spans,
                rows,
                end,
                views.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        elif kernel_choice == "wmma":
            qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_wmma_spans(
                views.full_query.ptr,
                views.key_cache.ptr,
                views.value_cache.ptr,
                append_metadata.k_scale.ptr,
                append_metadata.v_scale.ptr,
                views.full_gate.ptr,
                views.full_gated.ptr,
                views.prefill_spans,
                rows,
                end,
                views.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        else:
            qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_flash_spans(
                views.full_query.ptr,
                views.key_cache.ptr,
                views.value_cache.ptr,
                append_metadata.k_scale.ptr,
                append_metadata.v_scale.ptr,
                views.full_gate.ptr,
                views.full_gated.ptr,
                views.prefill_spans,
                rows,
                end,
                views.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
    elif use_aotriton:
        aotriton_library = selectors.aotriton_prefill_library()

        # Keep pre/post work on the caller stream. Only the high-scratch
        # AOTriton dispatch moves to the event-linked isolated queue.
        aotriton_stream = stream

        # Convert FP32 query to BF16 for AOTriton
        f32_to_bf16(
            views.full_query.ptr,
            views.full_query_bf16.ptr,
            rows * geometry.q_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )

        kv_key_ptr = views.key_cache.ptr
        kv_value_ptr = views.value_cache.ptr
        kv_strides = (geometry.kv_width * end, cfg.key_length, geometry.kv_width, 1)
        head_major_buffers = selectors.head_major_buffers(
            views,
            context_len=end,
        )
        if head_major_buffers is not None:
            head_major_key, head_major_value, head_major_capacity = head_major_buffers
            copy_variant = (
                "head_major_dense_prefix_spans"
                if views.head_major_kv_dense_prefix
                else "head_major_spans"
            )
            copy_head_major = selectors.paged_kv_copy_resolve(
                backend=backend,
                layer="paged_kv_copy",
                quant="bf16",
                variant=copy_variant,
                missing="none",
            )
            if copy_head_major is not None:
                copy_head_major(
                    views.key_cache.ptr,
                    views.value_cache.ptr,
                    head_major_key.ptr,
                    head_major_value.ptr,
                    views.prefill_spans,
                    end,
                    head_major_capacity,
                    views.block_size,
                    cfg.head_count_kv,
                    cfg.key_length,
                    stream=stream,
                    library=kv_write_library,
                    runtime=runtime,
                )
                kv_key_ptr = head_major_key.ptr
                kv_value_ptr = head_major_value.ptr
                kv_strides = (
                    geometry.kv_width * head_major_capacity,
                    cfg.key_length * head_major_capacity,
                    cfg.key_length,
                    1,
                )

        k_tensor = aotriton_tensor4(
            kv_key_ptr,
            (1, cfg.head_count_kv, end, cfg.key_length),
            kv_strides,
            DType.BF16,
        )
        v_tensor = aotriton_tensor4(
            kv_value_ptr,
            (1, cfg.head_count_kv, end, cfg.key_length),
            kv_strides,
            DType.BF16,
        )

        q_tensor = aotriton_tensor4(
            views.full_query_bf16.ptr,
            (1, cfg.head_count, rows, cfg.key_length),
            (
                cfg.head_count * cfg.key_length * rows,
                cfg.key_length,
                cfg.head_count * cfg.key_length,
                1,
            ),
            DType.BF16,
        )
        cu_q_tensor = aotriton_tensor1(views.cu_q.ptr, (2,), (1,), DType.INT32)
        cu_k_tensor = aotriton_tensor1(views.cu_k.ptr, (2,), (1,), DType.INT32)
        lse_tensor = aotriton_tensor2(
            views.softmax_lse.ptr,
            (cfg.head_count, rows),
            (rows, 1),
            DType.FP32,
        )
        out_tensor = aotriton_tensor4(
            views.full_attn_bf16.ptr,
            (1, cfg.head_count, rows, cfg.key_length),
            (
                cfg.head_count * cfg.key_length * rows,
                cfg.key_length,
                cfg.head_count * cfg.key_length,
                1,
            ),
            DType.BF16,
        )
        if aotriton_bridge is not None:
            aotriton_bridge.wait_for_inputs(runtime, stream)
            aotriton_stream = aotriton_bridge.stream
        aotriton_mode = policy.aotriton_prefill_mode(start, rows, end)
        if aotriton_mode == "v2":
            aotriton_attn_fwd_compact_varlen(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_q_tensor,
                cu_k_tensor,
                lse_tensor,
                out_tensor,
                max_seqlen_q=rows,
                max_seqlen_k=end,
                sm_scale=cfg.key_length ** -0.5,
                is_causal=True,
                stream=aotriton_stream,
                library=aotriton_library,
                runtime=runtime,
            )
        else:
            aotriton_attn_fwd_v3_compact_varlen(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_q_tensor,
                cu_k_tensor,
                lse_tensor,
                out_tensor,
                persistent_atomic_counter_ptr=views.atomic.ptr,
                max_seqlen_q=rows,
                max_seqlen_k=end,
                sm_scale=cfg.key_length ** -0.5,
                is_causal=True,
                stream=aotriton_stream,
                library=aotriton_library,
                runtime=runtime,
            )

        if aotriton_bridge is not None:
            aotriton_bridge.release_output(runtime, stream)
        qwen35_full_attn_gate_mul_bf16_to_bf16(
            views.full_attn_bf16.ptr,
            views.full_gate.ptr,
            views.full_gated.ptr,
            rows * cfg.head_count * cfg.key_length,
            stream=stream,
            library=paged_attn_library,
            runtime=runtime,
        )
    else:
        # Native fallback path: grow the demand-driven split-K partials
        # to the prefill batch-row count before the launch.
        split_view = selectors.ensure_split_rows(views.full_attn_split_batch_rows)
        selectors.native_prefill_fn()(
            views.full_query.ptr,
            views.key_cache.ptr,
            views.value_cache.ptr,
            views.full_gate.ptr,
            views.full_gated.ptr,
            views.prefill_spans,
            rows,
            paged_context_len,
            views.block_size,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.key_length,
            1,
            cfg.key_length ** -0.5,
            split_partial_out_ptr=split_view.partial.ptr,
            split_partial_m_ptr=split_view.m.ptr,
            split_partial_l_ptr=split_view.l.ptr,
            split_batch_rows=split_view.batch_rows,
            split_count=split_view.count,
            stream=stream,
            library=paged_attn_library,
            runtime=runtime,
        )
    if direct_hadamard_int8:
        # The Hadamard leaf also returns FP16, unlike the per-token
        # BF16-output leaves and the following GGUF projection consumer.
        fp16_to_bf16(
            views.full_gated.ptr,
            views.full_gated.ptr,
            rows * cfg.head_count * cfg.key_length,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_core_gate")
    used_aotriton = use_aotriton
    if not selectors.try_launch_dense_q8_single_dp4a(
        layer.weight("attn_output"),
        views.full_gated.ptr,
        views.attn_out.ptr,
        rows=rows,
        in_features=geometry.q_width,
        out_features=geometry.hidden_size,
        stream=stream,
        runtime=runtime,
    ):
        launch_gguf_linear(
            layer.weight("attn_output"),
            views.full_gated.ptr,
            views.attn_out.ptr,
            rows=rows,
            in_features=geometry.q_width,
            out_features=geometry.hidden_size,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_output")
    return used_aotriton
