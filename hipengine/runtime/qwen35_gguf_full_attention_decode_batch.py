"""Decode-batch full-attention execution for GGUF rows.

This module owns the attention half of one packed-AR decode-batch layer:
attention norm, Q/K/V projection, q/gate split, head norm and partial RoPE, KV
write, direct-INT8/short-context/long-context attention and the gate multiply.
The caller owns the output projection, F32 output composition, FFN call, split
workspace, KV planes and stage timing dict.

All buffers and selected operations arrive through explicit records. Nothing
here reads a runner, a session, or module globals for ownership.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from hipengine.core.hip import HipRuntime
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_full_attn_gate_mul_bf16,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans,
    qwen35_write_paged_kv_mixed_value_bf16_prompt_spans,
)
from hipengine.kernels.hip_gfx1100.convert import bf16_to_f32
from hipengine.kernels.hip_gfx1100.fused import (
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
)
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import qwen35_split_qgate_bf16
from hipengine.runtime.gguf_linear import (
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_triple,
)
from hipengine.runtime.qwen35_gguf_layer_attn import AttentionGeometry
from hipengine.runtime.qwen35_gguf_timing import _mark_sync_stage


@dataclass(frozen=True)
class DecodeBatchAttentionViews:
    """Activation and KV buffers for one decode-batch attention layer."""

    norm: object
    post_norm_f32: object | None
    full_q: object
    full_k: object
    full_v: object
    full_query_raw: object
    full_key_raw: object
    full_query: object
    full_key: object
    full_gate: object
    full_gated: object
    positions_tensor: object
    cos_table: object
    sin_table: object
    key_cache: object
    value_cache: object
    append_spans: object
    prefill_spans: object
    retained_key_cache: object | None
    retained_value_cache: object | None
    retained_append_spans: object | None
    retained_decode_spans: object | None
    retained_decode_kernel: Callable[..., None] | None
    int8_kv_value_bf16: bool
    block_size: int
    max_positions: int


@dataclass(frozen=True)
class DecodeBatchSplitWorkspace:
    """Current caller-owned split-K buffers for one attention dispatch."""

    rows: int
    num_splits: int
    partial_out: object
    partial_m: object
    partial_l: object


@dataclass(frozen=True)
class DecodeBatchAttentionSelectors:
    """Lazily resolved libraries and selected operations for one call."""

    cast_library: Callable[[], object]
    paged_kv_write_library: Callable[[], object]
    paged_attn_decode_library: Callable[[], object]
    run_attention_norm_rows: Callable[..., int | None]
    try_launch_dense_q8_triple_dp4a_f32: Callable[..., bool]
    try_launch_dense_q8_triple_dp4a: Callable[..., bool]
    int8_kv_prompt_write_fn: Callable[[object], Callable[..., None]]
    paged_attn_context_batch: Callable[[], Callable[..., None] | None]
    packed_ar_attention_batch_kernel: Callable[[], Callable[..., None] | None]
    split_workspace: Callable[[int, int], DecodeBatchSplitWorkspace | None]
    # Owner-resolved Q8T16 broad-rowtile policy for this call's row width.
    # ``None`` keeps the ambient/session resolver.
    q8_t16_rowtile_all: bool | None = None


@dataclass(frozen=True)
class DecodeBatchAttentionResult:
    """The attention route plus the stage clock for the caller's next mark."""

    attention_route: str
    kv_write_only: bool
    t_stage: float


def run_full_attention_decode_batch_attn_rows(
    layer,
    hidden_ptr: int,
    views: DecodeBatchAttentionViews,
    *,
    cfg,
    geometry: AttentionGeometry,
    runtime: HipRuntime,
    rows: int,
    start: int,
    selectors: DecodeBatchAttentionSelectors,
    stream: int = 0,
    hidden_f32_ptr: int | None = None,
    stage_timings: dict[str, float] | None = None,
    sync_stage_timings: bool = False,
    stage_prefix: str = "target_block_full_attn",
    kv_write_only: bool = False,
) -> DecodeBatchAttentionResult:
    """Run the attention half of one decode-batch layer."""

    rows = int(rows)
    end = int(start) + rows
    max_context_len = int(getattr(views.prefill_spans, "max_live_count", end))
    cast_library = selectors.cast_library()
    kv_write_library = selectors.paged_kv_write_library()
    paged_attn_library = selectors.paged_attn_decode_library()
    sync_stages = bool(sync_stage_timings and stage_timings is not None)
    t_stage = time.perf_counter() if sync_stages else 0.0
    attn_norm_f32_ptr = selectors.run_attention_norm_rows(
        hidden_ptr=hidden_ptr,
        hidden_f32_ptr=hidden_f32_ptr,
        weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
        out_ptr=views.norm.ptr,
        out_f32_ptr=(
            views.post_norm_f32.ptr if views.post_norm_f32 is not None else None
        ),
        rows=rows,
        eps=cfg.rms_norm_eps,
        stream=stream,
        runtime=runtime,
    )
    if not (
        attn_norm_f32_ptr is not None
        and selectors.try_launch_dense_q8_triple_dp4a_f32(
            layer.weight("attn_q"),
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            int(attn_norm_f32_ptr),
            views.full_q.ptr,
            views.full_k.ptr,
            views.full_v.ptr,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features_a=2 * geometry.q_width,
            out_features_b=geometry.kv_width,
            out_features_c=geometry.kv_width,
            stream=stream,
            runtime=runtime,
        )
    ) and not selectors.try_launch_dense_q8_triple_dp4a(
        layer.weight("attn_q"),
        layer.weight("attn_k"),
        layer.weight("attn_v"),
        views.norm.ptr,
        views.full_q.ptr,
        views.full_k.ptr,
        views.full_v.ptr,
        rows=rows,
        in_features=geometry.hidden_size,
        out_features_a=2 * geometry.q_width,
        out_features_b=geometry.kv_width,
        out_features_c=geometry.kv_width,
        stream=stream,
        runtime=runtime,
    ) and not launch_gguf_linear_triple(
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
        rowtile_all=selectors.q8_t16_rowtile_all,
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
            q8_t16_rowtile_all=selectors.q8_t16_rowtile_all,
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
            q8_t16_rowtile_all=selectors.q8_t16_rowtile_all,
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
                q8_t16_rowtile_all=selectors.q8_t16_rowtile_all,
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
                q8_t16_rowtile_all=selectors.q8_t16_rowtile_all,
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
    t_stage = _mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_norm_qkv_split",
        t_stage,
    )
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
        views.full_query_raw.ptr,
        views.full_key_raw.ptr,
        layer.weight("attn_q_norm").allocation().tensor.ptr,
        layer.weight("attn_k_norm").allocation().tensor.ptr,
        views.cos_table.ptr,
        views.sin_table.ptr,
        views.positions_tensor.ptr,
        views.full_query.ptr,
        views.full_key.ptr,
        cfg.rms_norm_eps,
        rows,
        cfg.head_count,
        cfg.head_count_kv,
        cfg.key_length,
        cfg.rope_dimension_count,
        views.max_positions,
        stream=stream,
        runtime=runtime,
    )
    t_stage = _mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_head_norm_rope",
        t_stage,
    )
    retained_spans = views.retained_append_spans
    retained_decode_spans = views.retained_decode_spans
    retained_decode_kernel = views.retained_decode_kernel
    direct_retained_batch = bool(
        rows > 1
        and retained_spans is not None
        and retained_decode_spans is not None
        and callable(retained_decode_kernel)
    )
    if retained_spans is None:
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
    else:
        metadata = retained_spans.scale_metadata
        retained_key_cache = views.retained_key_cache
        retained_value_cache = views.retained_value_cache
        if metadata is None or retained_key_cache is None or retained_value_cache is None:
            raise RuntimeError(
                "packed INT8 KV write requires retained payload and scale metadata"
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
                retained_key_cache.ptr,
                retained_value_cache.ptr,
                metadata.k_scale.ptr,
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
            selectors.int8_kv_prompt_write_fn(metadata)(
                views.full_key.ptr,
                views.full_key_raw.ptr,
                retained_key_cache.ptr,
                retained_value_cache.ptr,
                metadata.k_scale.ptr,
                metadata.v_scale.ptr,
                retained_spans,
                rows,
                views.block_size,
                cfg.head_count_kv,
                cfg.key_length,
                stream=stream,
                library=kv_write_library,
                runtime=runtime,
            )
        if not direct_retained_batch:
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
    t_stage = _mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_kv_write",
        t_stage,
    )
    if kv_write_only:
        return DecodeBatchAttentionResult("kv_write_only", True, t_stage)
    attention_route = "kv_live_spans_batch"
    if direct_retained_batch:
        assert retained_decode_spans is not None
        metadata = retained_decode_spans.scale_metadata
        retained_key_cache = views.retained_key_cache
        retained_value_cache = views.retained_value_cache
        num_splits = (max_context_len + int(views.block_size) - 1) // int(
            views.block_size
        )
        if (
            metadata is None
            or retained_key_cache is None
            or retained_value_cache is None
        ):
            raise RuntimeError(
                "direct INT8 batch attention requires payload and scale planes"
            )
        split_workspace = selectors.split_workspace(rows, num_splits)
        if (
            split_workspace is None
            or int(split_workspace.rows) < rows
            or int(split_workspace.num_splits) < num_splits
        ):
            raise NotImplementedError(
                "direct INT8 batch attention requires a row-sized split-K workspace"
            )
        retained_decode_kernel(
            views.full_query.ptr,
            retained_key_cache.ptr,
            retained_value_cache.ptr,
            metadata.k_scale.ptr,
            metadata.v_scale.ptr,
            views.full_gate.ptr,
            views.full_gated.ptr,
            split_workspace.partial_out.ptr,
            split_workspace.partial_m.ptr,
            split_workspace.partial_l.ptr,
            retained_decode_spans,
            rows,
            int(views.block_size),
            num_splits,
            int(views.block_size),
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            geometry.q_width,
            geometry.q_width,
            cfg.key_length,
            1,
            geometry.q_width,
            cfg.key_length,
            1,
            cfg.key_length ** -0.5,
            stream=stream,
            library=paged_attn_library,
            runtime=runtime,
        )
        attention_route = "kv_live_spans_int8_batch"
    elif max_context_len < 1024:
        context_batch = selectors.paged_attn_context_batch()
        if not callable(context_batch):  # pragma: no cover - registry resolve is fail-closed
            raise RuntimeError("paged context-batch attention kernel is unavailable")
        context_batch(
            views.full_query.ptr,
            views.key_cache.ptr,
            views.value_cache.ptr,
            views.full_query_raw.ptr,
            views.prefill_spans,
            rows,
            max_context_len,
            views.block_size,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.key_length ** -0.5,
            stream=stream,
            library=paged_attn_library,
            runtime=runtime,
        )
        qwen35_full_attn_gate_mul_bf16(
            views.full_query_raw.ptr,
            views.full_gate.ptr,
            views.full_gated.ptr,
            rows * geometry.q_width,
            stream=stream,
            library=paged_attn_library,
            runtime=runtime,
        )
    else:
        num_splits = (max_context_len + int(views.block_size) - 1) // int(
            views.block_size
        )
        split_workspace = selectors.split_workspace(rows, num_splits)
        if (
            split_workspace is None
            or int(split_workspace.rows) < rows
            or int(split_workspace.num_splits) < num_splits
        ):
            raise NotImplementedError(
                "long-context packed AR decode requires a row-sized split-K workspace"
            )
        split_batch = selectors.packed_ar_attention_batch_kernel()
        if not callable(split_batch):
            raise NotImplementedError(
                "long-context packed AR BF16 batch attention kernel is unavailable"
            )
        split_batch(
            views.full_query.ptr,
            views.key_cache.ptr,
            views.value_cache.ptr,
            views.full_gate.ptr,
            views.full_gated.ptr,
            split_workspace.partial_out.ptr,
            split_workspace.partial_m.ptr,
            split_workspace.partial_l.ptr,
            views.prefill_spans,
            rows,
            int(views.block_size),
            num_splits,
            int(views.block_size),
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
    t_stage = _mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_paged_attn",
        t_stage,
    )
    return DecodeBatchAttentionResult(attention_route, False, t_stage)
