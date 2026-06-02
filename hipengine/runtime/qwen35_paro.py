"""Qwen3.5/PARO runtime-state scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Sequence

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.rocblas import rocblas_gemm_ex_rowmajor_nt_fp16_compute_f32
from hipengine.core.tensor import Tensor
from hipengine.dispatch import (
    PagedAttnDecodeKind,
    PagedKVWriteKind,
    resolve_paged_attn_decode,
    resolve_paged_kv_write,
)
from hipengine.kernels.hip_gfx1100.attention import (
    aotriton_attn_fwd_v3_compact_varlen,
    aotriton_gate_mul_bf16_to_fp16,
    qwen35_full_attn_decode_context_bf16,
    qwen35_full_attn_gate_mul_bf16,
    qwen35_full_attn_gate_mul_fp16,
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import tensor1 as aotriton_tensor1
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import tensor2 as aotriton_tensor2
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import tensor4 as aotriton_tensor4
from hipengine.kernels.hip_gfx1100.convert import bf16_to_f32, f32_to_bf16, f32_to_fp16, fp16_to_bf16, fp16_to_f32
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    shared_gate_combine_residual_batch_out_bf16,
    shared_gate_combine_residual_batch_out_fp16,
    weighted_lanes_sum_out_bf16_f32w,
    weighted_lanes_sum_out_fp16_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w,
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_out_fp16_f32w,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_dual_out_bf16,
    silu_mul_dual_out_fp16,
    silu_mul_dual_rotate_out_bf16,
    silu_mul_dual_rotate_out_fp16,
    silu_mul_separate_out_bf16,
    silu_mul_separate_out_fp16,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    dense_dual_gemv_out_bf16,
    dense_dual_gemv_out_fp16,
    dense_gemv_out_bf16,
    dense_gemv_out_fp16,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_bf16,
    qwen35_linear_attn_conv_decode_fp16,
    qwen35_linear_attn_conv_prefill_f32,
    qwen35_linear_attn_conv_prefill_fp16,
    qwen35_linear_attn_conv_prefill_segments_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_recurrent_segments_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_gdn_prefill_rmsnorm_gate_fp16,
    qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16,
    qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16,
    qwen35_linear_attn_prefill_prepare_f32_bf16,
    qwen35_linear_attn_prefill_prepare_f32_fp16,
)
from hipengine.kernels.hip_gfx1100.norm import (
    paro_add_rmsnorm_out_bf16,
    paro_add_rmsnorm_out_fp16,
    paro_rmsnorm_out_bf16,
    paro_rmsnorm_out_fp16,
)
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import (
    w8a16_linear_bf16_lowp_out,
    w8a16_linear_fp16_lowp_out,
    w8a16_shared_down_combine_residual_fp16,
    w8a16_shared_down_combine_residual_fp16_token_tiled,
    w8a16_shared_gate_sigmoid_fp32,
    w8a16_shared_gate_up_silu_fp16,
    w8a16_shared_gate_up_silu_fp16_token_tiled,
)
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    qwen35_moe_group_count,
    qwen35_moe_group_prefix,
    qwen35_moe_group_scatter_gather_lowp,
    qwen35_moe_wmma_tile_map,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_topk_shared_coop_out_bf16,
    qwen35_router_topk_shared_coop_out_fp16,
    qwen35_router_topk_shared_out_bf16,
    qwen35_router_topk_shared_out_fp16,
    qwen35_router_topk_shared_sigmoid_out_fp16,
)
from hipengine.kernels.hip_gfx1100.quant.paro_marlin_k import gemv_paro_marlin_k_fma_fp16, marlin_k_default_threads
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
    awq_fusedw4_prefill_dual_fp16,
    awq_fusedw4_prefill_fp16,
    awq_fusedw4_prefill_strided_fp16,
    gemv_awq_dual_pack8_transposed_bf16,
    gemv_awq_dual_pack8_transposed_fp16,
    gemv_awq_dual_pack8_transposed_rotate_staged_bf16,
    gemv_awq_dual_pack8_transposed_rotate_staged_fp16,
    gemv_awq_pack8_strided_bf16,
    gemv_awq_pack8_strided_fp16,
    gemv_awq_pack8_transposed_bf16,
    gemv_awq_pack8_transposed_fp16,
    gemv_awq_selected_dual_pack8_transposed_bf16,
    gemv_awq_selected_dual_pack8_transposed_fp16,
    gemv_awq_selected_pack8_transposed_bf16,
    gemv_awq_selected_pack8_transposed_fp16,
)
from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import (
    paro_rotate1_bf16,
    paro_rotate1_bf16_gate_fp16,
    paro_rotate1_fp16,
    paro_rotate2_bf16,
    paro_rotate2_fp16,
    paro_rotate3_bf16,
    paro_rotate3_fp16,
)
from hipengine.kernels.hip_gfx1100.wmma import (
    gemm_awq_selected_dual_pack8_wmma_compact_bf16,
    gemm_awq_selected_dual_pack8_wmma_compact_fp16,
    gemm_awq_selected_pack8_wmma_compact_bf16,
    gemm_awq_selected_pack8_wmma_compact_fp16,
)
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import (
    qwen35_head_rmsnorm_partial_rotary_position_f32_bf16,
    qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16,
    qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32,
    qwen35_split_qgate_bf16,
    qwen35_split_qgate_fp16,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.qwen35_paro import Qwen35ParoLayerDeviceWeights, normalize_qwen35_weight_name
from hipengine.runtime.workspace import RuntimeWorkspace


_PAGED_KV_REGISTRY_BACKEND = "hip_gfx1100"


def qwen35_grouped_moe_lane_rows(tokens: int, top_k: int) -> tuple[int, ...]:
    """Return the token row for each token-major routed MoE lane."""

    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("tokens must be a positive int")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive int")
    return tuple(lane // top_k for lane in range(tokens * top_k))


def qwen35_grouped_moe_sorted_token_rows(
    sorted_lanes: Sequence[int],
    *,
    tokens: int,
    top_k: int,
) -> tuple[int, ...]:
    """Return token rows in grouped-MoE sorted-lane order."""

    lane_rows = qwen35_grouped_moe_lane_rows(tokens, top_k)
    total_lanes = len(lane_rows)
    if len(sorted_lanes) != total_lanes:
        raise ValueError("sorted_lanes length must match tokens * top_k")
    rows: list[int] = []
    seen: set[int] = set()
    for lane in sorted_lanes:
        if not isinstance(lane, int) or isinstance(lane, bool) or lane < 0 or lane >= total_lanes:
            raise ValueError("sorted_lanes entries must be unique lane ints in range")
        if lane in seen:
            raise ValueError("sorted_lanes entries must be unique lane ints in range")
        seen.add(lane)
        rows.append(lane_rows[lane])
    return tuple(rows)


def qwen35_grouped_moe_lane_to_sorted_row(
    sorted_lanes: Sequence[int],
    *,
    tokens: int,
    top_k: int,
) -> tuple[int, ...]:
    """Mirror the grouped-MoE combine kernel's lane-to-sorted-row inverse map."""

    total_lanes = len(qwen35_grouped_moe_lane_rows(tokens, top_k))
    if len(sorted_lanes) != total_lanes:
        raise ValueError("sorted_lanes length must match tokens * top_k")
    lane_to_row = [-1] * total_lanes
    for sorted_row, lane in enumerate(sorted_lanes):
        if not isinstance(lane, int) or isinstance(lane, bool) or lane < 0 or lane >= total_lanes:
            raise ValueError("sorted_lanes entries must be unique lane ints in range")
        if lane_to_row[lane] != -1:
            raise ValueError("sorted_lanes entries must be unique lane ints in range")
        lane_to_row[lane] = sorted_row
    return tuple(lane_to_row)


def _qwen35_grouped_moe_route_shape(selected_experts: Sequence[Sequence[int]]) -> tuple[int, int]:
    tokens = len(selected_experts)
    if tokens <= 0:
        raise ValueError("selected_experts must contain at least one token row")
    top_k = len(selected_experts[0])
    if top_k <= 0:
        raise ValueError("selected_experts rows must contain at least one expert")
    if any(len(row) != top_k for row in selected_experts):
        raise ValueError("selected_experts rows must have a consistent top_k")
    return tokens, top_k


def qwen35_grouped_moe_expert_lane_groups(
    selected_experts: Sequence[Sequence[int]],
    *,
    num_experts: int,
) -> tuple[tuple[int, ...], ...]:
    """Group token-major routed lanes by expert for the compact MoE path."""

    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be a positive int")
    tokens, top_k = _qwen35_grouped_moe_route_shape(selected_experts)
    groups: list[list[int]] = [[] for _ in range(num_experts)]
    for token_row, row in enumerate(selected_experts):
        for expert_rank, expert in enumerate(row):
            if not isinstance(expert, int) or isinstance(expert, bool) or expert < 0 or expert >= num_experts:
                raise ValueError("selected_experts entries must be expert ints in range")
            groups[expert].append(token_row * top_k + expert_rank)
    expected_total_lanes = tokens * top_k
    if sum(len(group) for group in groups) != expected_total_lanes:
        raise ValueError("selected_experts lane grouping did not cover all routed lanes")
    return tuple(tuple(group) for group in groups)


def qwen35_grouped_moe_expert_starts(
    selected_experts: Sequence[Sequence[int]],
    *,
    num_experts: int,
) -> tuple[int, ...]:
    """Return compact grouped-MoE expert-start offsets for token-major lanes."""

    groups = qwen35_grouped_moe_expert_lane_groups(selected_experts, num_experts=num_experts)
    starts = [0]
    for group in groups:
        starts.append(starts[-1] + len(group))
    return tuple(starts)


def qwen35_grouped_moe_sorted_lanes_from_selected_experts(
    selected_experts: Sequence[Sequence[int]],
    *,
    num_experts: int,
) -> tuple[int, ...]:
    """Return token-major lane ids in compact grouped-MoE expert order."""

    groups = qwen35_grouped_moe_expert_lane_groups(selected_experts, num_experts=num_experts)
    return tuple(lane for group in groups for lane in group)


def qwen35_grouped_moe_sorted_routing_weights(
    routing_weights: Sequence[Sequence[float]],
    sorted_lanes: Sequence[int],
    *,
    tokens: int,
    top_k: int,
) -> tuple[float, ...]:
    """Return routing weights in grouped-MoE sorted-lane order."""

    if len(routing_weights) != tokens or any(len(row) != top_k for row in routing_weights):
        raise ValueError("routing_weights shape must match tokens * top_k")
    lane_to_row_rank = tuple((lane // top_k, lane % top_k) for lane in range(tokens * top_k))
    lane_to_sorted_row = qwen35_grouped_moe_lane_to_sorted_row(sorted_lanes, tokens=tokens, top_k=top_k)
    sorted_weights = [0.0] * len(sorted_lanes)
    for lane, sorted_row in enumerate(lane_to_sorted_row):
        token_row, expert_rank = lane_to_row_rank[lane]
        weight = routing_weights[token_row][expert_rank]
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            raise ValueError("routing_weights entries must be numeric")
        sorted_weights[sorted_row] = float(weight)
    return tuple(sorted_weights)


def qwen35_grouped_moe_weighted_token_sums(
    sorted_values: Sequence[Sequence[float]],
    sorted_weights: Sequence[float],
    sorted_lanes: Sequence[int],
    *,
    tokens: int,
    top_k: int,
) -> tuple[tuple[float, ...], ...]:
    """Mirror grouped-MoE weighted selected-branch accumulation on CPU."""

    total_lanes = tokens * top_k
    if len(sorted_values) != total_lanes:
        raise ValueError("sorted_values length must match tokens * top_k")
    if len(sorted_weights) != total_lanes:
        raise ValueError("sorted_weights length must match tokens * top_k")
    feature_size = len(sorted_values[0]) if sorted_values else 0
    if feature_size <= 0 or any(len(row) != feature_size for row in sorted_values):
        raise ValueError("sorted_values rows must have a consistent non-empty feature size")
    lane_to_sorted_row = qwen35_grouped_moe_lane_to_sorted_row(sorted_lanes, tokens=tokens, top_k=top_k)
    out: list[tuple[float, ...]] = []
    for token in range(tokens):
        features: list[float] = []
        for col in range(feature_size):
            acc = 0.0
            for expert_rank in range(top_k):
                lane = token * top_k + expert_rank
                sorted_row = lane_to_sorted_row[lane]
                value = sorted_values[sorted_row][col]
                weight = sorted_weights[sorted_row]
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError("sorted_values entries must be numeric")
                if isinstance(weight, bool) or not isinstance(weight, int | float):
                    raise ValueError("sorted_weights entries must be numeric")
                acc += float(value) * float(weight)
            features.append(acc)
        out.append(tuple(features))
    return tuple(out)


@dataclass(frozen=True)
class Qwen35ParoAttentionScratch:
    attn_input: Tensor
    q_rot: Tensor
    k_rot: Tensor
    v_rot: Tensor
    rotate_fuse_barrier: Tensor
    q_proj_key: Tensor
    q_proj: Tensor
    key_bf16: Tensor
    query_raw: Tensor
    key_raw: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    kv_proj: Tensor | None
    gate: Tensor
    partial_out: Tensor
    partial_m: Tensor
    partial_l: Tensor
    attn_out: Tensor
    gated_attn: Tensor
    o_rot: Tensor
    o_proj: Tensor


@dataclass(frozen=True)
class Qwen35ParoLinearAttentionScratch:
    attn_input: Tensor
    qkv_rot: Tensor
    z_rot: Tensor
    rotate_fuse_barrier: Tensor
    qkv_z: Tensor
    qkv: Tensor
    z: Tensor
    qkv_f32: Tensor
    ab: Tensor
    a: Tensor
    b: Tensor
    conv_out: Tensor
    prefill_query: Tensor
    prefill_key: Tensor
    prefill_value: Tensor
    prefill_beta: Tensor
    prefill_decay: Tensor
    recurrent_out: Tensor
    recurrent_bf16: Tensor
    out_rot: Tensor
    out_proj: Tensor


@dataclass(frozen=True)
class Qwen35ParoMoeScratch:
    normed: Tensor
    residual: Tensor
    gate_up_input: Tensor
    router_logits: Tensor
    routing_weights: Tensor
    selected_experts: Tensor
    gate_up: Tensor
    down_input: Tensor
    down_out: Tensor
    shared_gate_input: Tensor
    shared_up_input: Tensor
    shared_gate_out: Tensor
    shared_up_out: Tensor
    shared_up: Tensor
    shared_intermediate: Tensor
    shared_down_input: Tensor
    shared_out: Tensor
    moe_out: Tensor


@dataclass(frozen=True)
class Qwen35ParoDenseMlpScratch:
    """Scratch for dense Qwen3.5 PARO MLP gate/up/down projections."""

    normed: Tensor
    residual: Tensor
    shared_gate_input: Tensor
    shared_up_input: Tensor
    shared_gate_out: Tensor
    shared_up_out: Tensor
    shared_up: Tensor
    shared_intermediate: Tensor
    shared_down_input: Tensor
    shared_out: Tensor
    shared_zero: Tensor
    gate_logits: Tensor
    moe_out: Tensor


@dataclass(frozen=True)
class Qwen35ParoGroupedMoeScratch:
    normed: Tensor
    residual: Tensor
    router_logits: Tensor
    routing_weights: Tensor
    selected_experts: Tensor
    counts: Tensor
    padded_counts: Tensor
    expert_start: Tensor
    total_padded: Tensor
    scatter_offsets: Tensor
    sorted_lanes: Tensor
    sorted_experts: Tensor
    sorted_weights: Tensor
    lane_to_row: Tensor
    wmma_expert_start: Tensor
    tile_expert: Tensor
    wmma_total: Tensor
    packed_hidden: Tensor
    packed_gate_up_input: Tensor
    gate_up: Tensor
    down_input: Tensor
    down_out: Tensor
    selected_out: Tensor
    shared_gate_input: Tensor
    shared_up_input: Tensor
    shared_gate_out: Tensor
    shared_up_out: Tensor
    shared_up: Tensor
    shared_intermediate: Tensor
    shared_down_input: Tensor
    shared_out: Tensor
    moe_out: Tensor


class Qwen35ParoDecodeState:
    """Minimal one-token decode state for a materialized Qwen3.5/PARO layer.

    This object intentionally does not encode backend conditionals. It owns only
    normalized device weights plus named scratch buffers. Kernel selection still
    flows through the registry/wrappers added in the gfx1100 backend tree.
    """

    def __init__(
        self,
        *,
        layer_weights: Qwen35ParoLayerDeviceWeights,
        workspace: RuntimeWorkspace | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        self.layer_weights = layer_weights
        self.runtime = runtime
        self.workspace = workspace or RuntimeWorkspace(runtime=runtime)
        self._rotate_fuse_ready: set[int] = set()

    @property
    def config(self):
        return self.layer_weights.config

    def tensor(self, name: str) -> Tensor:
        return self.layer_weights.tensor(name)

    def has_tensor(self, name: str) -> bool:
        return normalize_qwen35_weight_name(name) in self.layer_weights.weights.tensors

    def _shared_expert_is_legacy_w8a16(self) -> bool:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        return self.has_tensor(f"{prefix}.gate_up_weight_w8a16")

    def _shared_expert_is_packed_paro_w4(self) -> bool:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        return self.has_tensor(f"{prefix}.gate_proj.qweight_pack8_decode")

    def reserve_full_attention_scratch(
        self,
        *,
        tokens: int = 1,
        num_splits: int = 1,
        activation_dtype: str | DType = DType.BF16,
        gated_dtype: str | DType | None = None,
        query_dtype: str | DType = DType.FP32,
    ) -> Qwen35ParoAttentionScratch:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if num_splits <= 0:
            raise ValueError("num_splits must be positive")
        cfg = self.config
        q_width = cfg.num_attention_heads * cfg.head_dim
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        lowp = DType.parse(activation_dtype)
        if lowp not in {DType.BF16, DType.FP16}:
            raise ValueError("activation_dtype must be bf16 or fp16")
        gated = lowp if gated_dtype is None else DType.parse(gated_dtype)
        if gated not in {DType.BF16, DType.FP16, DType.FP32}:
            raise ValueError("gated_dtype must be bf16, fp16, or fp32")
        query_out_dtype = DType.parse(query_dtype)
        if query_out_dtype not in {DType.BF16, DType.FP32}:
            raise ValueError("query_dtype must be bf16 or fp32")
        q_proj_key = self.workspace.reserve_tensor("attn.q_proj_key", (tokens, 2 * q_width + kv_width), lowp)
        q_proj = Tensor.from_handle(q_proj_key.ptr, (tokens, 2 * q_width), lowp, q_proj_key.device)
        key_bf16 = Tensor.from_handle(
            q_proj_key.ptr + tokens * 2 * q_width * lowp.itemsize,
            (tokens, kv_width),
            lowp,
            q_proj_key.device,
        )
        kv_proj = None
        if _full_attn_kv_pack8_fused_enabled():
            kv_proj = self.workspace.reserve_tensor("attn.kv_proj", (tokens, 2 * kv_width), lowp)
            key_bf16 = Tensor.from_handle(kv_proj.ptr, (tokens, kv_width), lowp, kv_proj.device)
            value = Tensor.from_handle(
                kv_proj.ptr + tokens * kv_width * lowp.itemsize,
                (tokens, cfg.num_key_value_heads, cfg.head_dim),
                lowp,
                kv_proj.device,
            )
        else:
            value = self.workspace.reserve_tensor("attn.value", (tokens, cfg.num_key_value_heads, cfg.head_dim), lowp)
        return Qwen35ParoAttentionScratch(
            attn_input=self.workspace.reserve_tensor("attn.input", (tokens, cfg.hidden_size), lowp),
            q_rot=self.workspace.reserve_tensor("attn.q_rot", (tokens, cfg.hidden_size), lowp),
            k_rot=self.workspace.reserve_tensor("attn.k_rot", (tokens, cfg.hidden_size), lowp),
            v_rot=self.workspace.reserve_tensor("attn.v_rot", (tokens, cfg.hidden_size), lowp),
            rotate_fuse_barrier=self.workspace.reserve_tensor("attn.rotate_fuse_barrier", (2,), DType.INT32),
            q_proj_key=q_proj_key,
            q_proj=q_proj,
            key_bf16=key_bf16,
            query_raw=self.workspace.reserve_tensor("attn.query_raw", (tokens, cfg.num_attention_heads, cfg.head_dim), DType.FP32),
            key_raw=self.workspace.reserve_tensor("attn.key_raw", (tokens, cfg.num_key_value_heads, cfg.head_dim), DType.FP32),
            query=self.workspace.reserve_tensor("attn.query", (tokens, cfg.num_attention_heads, cfg.head_dim), query_out_dtype),
            key=self.workspace.reserve_tensor("attn.key", (tokens, cfg.num_key_value_heads, cfg.head_dim), DType.FP32),
            value=value,
            kv_proj=kv_proj,
            gate=self.workspace.reserve_tensor("attn.gate", (tokens, cfg.num_attention_heads, cfg.head_dim), gated),
            partial_out=self.workspace.reserve_tensor(
                "attn.partial_out",
                (cfg.num_attention_heads, num_splits, cfg.head_dim),
                DType.FP32,
            ),
            partial_m=self.workspace.reserve_tensor("attn.partial_m", (cfg.num_attention_heads, num_splits), DType.FP32),
            partial_l=self.workspace.reserve_tensor("attn.partial_l", (cfg.num_attention_heads, num_splits), DType.FP32),
            attn_out=self.workspace.reserve_tensor("attn.out", (cfg.num_attention_heads, cfg.head_dim), DType.FP32),
            gated_attn=self.workspace.reserve_tensor("attn.gated", (tokens, q_width), gated),
            o_rot=self.workspace.reserve_tensor("attn.o_rot", (tokens, q_width), lowp),
            o_proj=self.workspace.reserve_tensor("attn.o_proj", (tokens, cfg.hidden_size), lowp),
        )

    def project_pack8_bf16(
        self,
        x: Tensor,
        out: Tensor,
        *,
        weight_prefix: str,
        rows: int = 1,
        in_features: int | None = None,
        group_size: int = 128,
        threads: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = normalize_qwen35_weight_name(weight_prefix)
        qzeros = self.tensor(f"{prefix}.qzeros")
        scales = self.tensor(f"{prefix}.scales")
        width = x.shape[-1] if in_features is None else in_features
        awq_library = _library_for(library, "awq")
        if self.has_tensor(f"{prefix}.qweight"):
            qweight = self.tensor(f"{prefix}.qweight")
            if not qweight.shape:
                raise ValueError(f"{prefix}.qweight must have at least one dimension")
            gemv_awq_pack8_strided_bf16(
                x.ptr,
                qweight.ptr,
                qzeros.ptr,
                scales.ptr,
                out.ptr,
                rows,
                width,
                _out_packed_from_strided_qweight(qweight),
                group_size,
                threads=threads,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
        else:
            qweight = self.tensor(f"{prefix}.qweight_pack8_decode")
            gemv_awq_pack8_transposed_bf16(
                x.ptr,
                qweight.ptr,
                qzeros.ptr,
                scales.ptr,
                out.ptr,
                rows,
                width,
                _out_packed_from_generic_transposed_qweight(qweight),
                group_size,
                threads=threads,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
        return out

    def project_pack8_fp16(
        self,
        x: Tensor,
        out: Tensor,
        *,
        weight_prefix: str,
        rows: int = 1,
        in_features: int | None = None,
        group_size: int = 128,
        threads: int = 128,
        force_gemv: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = normalize_qwen35_weight_name(weight_prefix)
        qzeros = self.tensor(f"{prefix}.qzeros")
        scales = self.tensor(f"{prefix}.scales")
        width = x.shape[-1] if in_features is None else in_features
        awq_library = _library_for(library, "awq")
        if rows == 1 and self.has_tensor(f"{prefix}.qweight_mk"):
            qweight_mk = self.tensor(f"{prefix}.qweight_mk")
            out_packed = _out_packed_from_marlin_qweight(qweight_mk)
            gemv_paro_marlin_k_fma_fp16(
                x.ptr,
                qweight_mk.ptr,
                self.tensor(f"{prefix}.qzeros_mk").ptr,
                self.tensor(f"{prefix}.scales_mk").ptr,
                out.ptr,
                rows,
                width,
                out_packed,
                group_size,
                threads=marlin_k_default_threads(width, out_packed * 8),
                stream=stream,
                library=_library_for(library, "marlin_k"),
                runtime=self.runtime,
            )
        elif self.has_tensor(f"{prefix}.qweight"):
            qweight = self.tensor(f"{prefix}.qweight")
            if not qweight.shape:
                raise ValueError(f"{prefix}.qweight must have at least one dimension")
            out_packed = _out_packed_from_strided_qweight(qweight)
            if rows > 1 and not force_gemv and group_size % 16 == 0 and width % group_size == 0:
                awq_fusedw4_prefill_strided_fp16(
                    x.ptr,
                    qweight.ptr,
                    qzeros.ptr,
                    scales.ptr,
                    out.ptr,
                    rows,
                    width,
                    out_packed,
                    group_size,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
            else:
                gemv_awq_pack8_strided_fp16(
                    x.ptr,
                    qweight.ptr,
                    qzeros.ptr,
                    scales.ptr,
                    out.ptr,
                    rows,
                    width,
                    out_packed,
                    group_size,
                    threads=threads,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
        else:
            qweight = self.tensor(f"{prefix}.qweight_pack8_decode")
            out_packed = _out_packed_from_generic_transposed_qweight(qweight)
            if rows > 1 and not force_gemv and group_size % 16 == 0 and width % group_size == 0:
                awq_fusedw4_prefill_fp16(
                    x.ptr,
                    qweight.ptr,
                    qzeros.ptr,
                    scales.ptr,
                    out.ptr,
                    rows,
                    width,
                    out_packed,
                    group_size,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
            else:
                gemv_awq_pack8_transposed_fp16(
                    x.ptr,
                    qweight.ptr,
                    qzeros.ptr,
                    scales.ptr,
                    out.ptr,
                    rows,
                    width,
                    out_packed,
                    group_size,
                    threads=threads,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
        return out

    def rotate_full_attention_inputs_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn"
        q = f"{prefix}.q_proj"
        k = f"{prefix}.k_proj"
        v = f"{prefix}.v_proj"
        q_pairs = self.tensor(f"{q}.pairs")
        k_pairs = self.tensor(f"{k}.pairs")
        v_pairs = self.tensor(f"{v}.pairs")
        if (
            tokens == 1
            and hidden.ptr == scratch.attn_input.ptr
            and _rotate_dual_pack8_fused_enabled()
            and not _full_attn_kv_pack8_fused_enabled()
        ):
            self._rotate_fuse_ready.add(scratch.rotate_fuse_barrier.ptr)
            paro_rotate1_bf16(
                hidden.ptr,
                scratch.v_rot.ptr,
                v_pairs.ptr,
                self.tensor(f"{v}.theta").ptr,
                self.tensor(f"{v}.channel_scales").ptr,
                tokens,
                self.config.hidden_size,
                group_size,
                _rotation_krot(v_pairs),
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
            return scratch.q_rot, scratch.k_rot, scratch.v_rot
        self._rotate_fuse_ready.discard(scratch.rotate_fuse_barrier.ptr)
        paro_rotate3_bf16(
            hidden.ptr,
            scratch.q_rot.ptr,
            scratch.k_rot.ptr,
            scratch.v_rot.ptr,
            q_pairs.ptr,
            k_pairs.ptr,
            v_pairs.ptr,
            self.tensor(f"{q}.theta").ptr,
            self.tensor(f"{k}.theta").ptr,
            self.tensor(f"{v}.theta").ptr,
            self.tensor(f"{q}.channel_scales").ptr,
            self.tensor(f"{k}.channel_scales").ptr,
            self.tensor(f"{v}.channel_scales").ptr,
            tokens,
            self.config.hidden_size,
            group_size,
            _rotation_krot(q_pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        return scratch.q_rot, scratch.k_rot, scratch.v_rot

    def project_full_attention_qkv_bf16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn"
        q = f"{prefix}.q_proj"
        k = f"{prefix}.k_proj"
        v = f"{prefix}.v_proj"
        q_qweight = self.tensor(f"{q}.qweight_pack8_decode")
        k_qweight = self.tensor(f"{k}.qweight_pack8_decode")
        q_out_packed = _out_packed_from_generic_transposed_qweight(q_qweight)
        k_out_packed = _out_packed_from_generic_transposed_qweight(k_qweight)
        awq_library = _library_for(library, "awq")
        kv_fused = False
        if tokens == 1:
            use_rotate_fused = scratch.rotate_fuse_barrier.ptr in self._rotate_fuse_ready
            if scratch.kv_proj is not None and not use_rotate_fused:
                v_qweight = self.tensor(f"{v}.qweight_pack8_decode")
                v_out_packed = _out_packed_from_generic_transposed_qweight(v_qweight)
                self.project_pack8_bf16(
                    scratch.q_rot,
                    scratch.q_proj,
                    weight_prefix=q,
                    rows=tokens,
                    group_size=group_size,
                    library=library,
                    stream=stream,
                )
                gemv_awq_dual_pack8_transposed_bf16(
                    scratch.k_rot.ptr,
                    scratch.v_rot.ptr,
                    k_qweight.ptr,
                    self.tensor(f"{k}.qzeros").ptr,
                    self.tensor(f"{k}.scales").ptr,
                    v_qweight.ptr,
                    self.tensor(f"{v}.qzeros").ptr,
                    self.tensor(f"{v}.scales").ptr,
                    scratch.key_bf16.ptr,
                    tokens,
                    scratch.k_rot.shape[-1],
                    k_out_packed,
                    v_out_packed,
                    group_size,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
                kv_fused = True
            elif use_rotate_fused:
                gemv_awq_dual_pack8_transposed_rotate_staged_bf16(
                    scratch.attn_input.ptr,
                    scratch.q_rot.ptr,
                    scratch.k_rot.ptr,
                    self.tensor(f"{q}.pairs").ptr,
                    self.tensor(f"{k}.pairs").ptr,
                    self.tensor(f"{q}.theta").ptr,
                    self.tensor(f"{k}.theta").ptr,
                    self.tensor(f"{q}.channel_scales").ptr,
                    self.tensor(f"{k}.channel_scales").ptr,
                    q_qweight.ptr,
                    self.tensor(f"{q}.qzeros").ptr,
                    self.tensor(f"{q}.scales").ptr,
                    k_qweight.ptr,
                    self.tensor(f"{k}.qzeros").ptr,
                    self.tensor(f"{k}.scales").ptr,
                    scratch.q_proj_key.ptr,
                    scratch.rotate_fuse_barrier.ptr,
                    tokens,
                    scratch.q_rot.shape[-1],
                    q_out_packed,
                    k_out_packed,
                    group_size,
                    _rotation_krot(self.tensor(f"{q}.pairs")),
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
                self._rotate_fuse_ready.discard(scratch.rotate_fuse_barrier.ptr)
            else:
                gemv_awq_dual_pack8_transposed_bf16(
                    scratch.q_rot.ptr,
                    scratch.k_rot.ptr,
                    q_qweight.ptr,
                    self.tensor(f"{q}.qzeros").ptr,
                    self.tensor(f"{q}.scales").ptr,
                    k_qweight.ptr,
                    self.tensor(f"{k}.qzeros").ptr,
                    self.tensor(f"{k}.scales").ptr,
                    scratch.q_proj_key.ptr,
                    tokens,
                    scratch.q_rot.shape[-1],
                    q_out_packed,
                    k_out_packed,
                    group_size,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
        else:
            gemv_awq_pack8_transposed_bf16(
                scratch.q_rot.ptr,
                q_qweight.ptr,
                self.tensor(f"{q}.qzeros").ptr,
                self.tensor(f"{q}.scales").ptr,
                scratch.q_proj.ptr,
                tokens,
                scratch.q_rot.shape[-1],
                q_out_packed,
                group_size,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
            gemv_awq_pack8_transposed_bf16(
                scratch.k_rot.ptr,
                k_qweight.ptr,
                self.tensor(f"{k}.qzeros").ptr,
                self.tensor(f"{k}.scales").ptr,
                scratch.key_bf16.ptr,
                tokens,
                scratch.k_rot.shape[-1],
                k_out_packed,
                group_size,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
        if not kv_fused:
            self.project_pack8_bf16(
                scratch.v_rot,
                scratch.value,
                weight_prefix=f"{prefix}.v_proj",
                rows=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return scratch.q_proj, scratch.key_bf16, scratch.value

    def prepare_full_attention_qkv_bf16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        cos_table: Tensor,
        sin_table: Tensor,
        position: Tensor,
        max_positions: int,
        tokens: int = 1,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        qwen35_split_qgate_bf16(
            scratch.q_proj.ptr,
            scratch.query_raw.ptr,
            scratch.gate.ptr,
            tokens,
            cfg.num_attention_heads,
            cfg.head_dim,
            stream=stream,
            library=_library_for(library, "qwen_rotary"),
            runtime=self.runtime,
        )
        bf16_to_f32(
            scratch.key_bf16.ptr,
            scratch.key_raw.ptr,
            tokens * kv_width,
            stream=stream,
            library=_library_for(library, "cast"),
            runtime=self.runtime,
        )
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn"
        qwen_rotary_library = _library_for(library, "qwen_rotary")
        if tokens == 1:
            qwen35_head_rmsnorm_partial_rotary_position_f32_bf16(
                scratch.query_raw.ptr,
                scratch.key_raw.ptr,
                self.tensor(f"{prefix}.q_norm.weight").ptr,
                self.tensor(f"{prefix}.k_norm.weight").ptr,
                cos_table.ptr,
                sin_table.ptr,
                position.ptr,
                scratch.query.ptr,
                scratch.key.ptr,
                self.config.rms_norm_eps,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rotary_dim or cfg.head_dim,
                max_positions,
                stream=stream,
                library=qwen_rotary_library,
                runtime=self.runtime,
            )
        else:
            qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16(
                scratch.query_raw.ptr,
                scratch.key_raw.ptr,
                self.tensor(f"{prefix}.q_norm.weight").ptr,
                self.tensor(f"{prefix}.k_norm.weight").ptr,
                cos_table.ptr,
                sin_table.ptr,
                position.ptr,
                scratch.query.ptr,
                scratch.key.ptr,
                self.config.rms_norm_eps,
                tokens,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rotary_dim or cfg.head_dim,
                max_positions,
                stream=stream,
                library=qwen_rotary_library,
                runtime=self.runtime,
            )
        return scratch.query, scratch.key, scratch.value, scratch.gate

    def project_full_attention_o_bf16(
        self,
        gated_attn: Tensor,
        scratch: Qwen35ParoAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn.o_proj"
        q_width = self.config.num_attention_heads * self.config.head_dim
        pairs = self.tensor(f"{prefix}.pairs")
        paro_rotate1_bf16(
            gated_attn.ptr,
            scratch.o_rot.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.theta").ptr,
            self.tensor(f"{prefix}.channel_scales").ptr,
            tokens,
            q_width,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        self.project_pack8_bf16(
            scratch.o_rot,
            scratch.o_proj,
            weight_prefix=prefix,
            rows=tokens,
            in_features=q_width,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        return scratch.o_proj

    def reserve_linear_attention_scratch(
        self,
        *,
        tokens: int = 1,
        activation_dtype: str | DType = DType.BF16,
    ) -> Qwen35ParoLinearAttentionScratch:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        lowp = DType.parse(activation_dtype)
        if lowp not in {DType.BF16, DType.FP16}:
            raise ValueError("activation_dtype must be bf16 or fp16")
        cfg = self.config
        qkv_width = _linear_qkv_width(cfg)
        z_width = _linear_value_width(cfg)
        qkv_z = self.workspace.reserve_tensor("linear_attn.qkv_z", (tokens, qkv_width + z_width), lowp)
        qkv = Tensor.from_handle(qkv_z.ptr, (tokens, qkv_width), lowp, qkv_z.device)
        z = Tensor.from_handle(qkv_z.ptr + tokens * qkv_width * lowp.itemsize, (tokens, z_width), lowp, qkv_z.device)
        ab = self.workspace.reserve_tensor("linear_attn.ab", (tokens, 2 * cfg.linear_num_value_heads), lowp)
        a = Tensor.from_handle(ab.ptr, (tokens, cfg.linear_num_value_heads), lowp, ab.device)
        b = Tensor.from_handle(
            ab.ptr + tokens * cfg.linear_num_value_heads * lowp.itemsize,
            (tokens, cfg.linear_num_value_heads),
            lowp,
            ab.device,
        )
        return Qwen35ParoLinearAttentionScratch(
            attn_input=self.workspace.reserve_tensor("linear_attn.attn_input", (tokens, cfg.hidden_size), lowp),
            qkv_rot=self.workspace.reserve_tensor("linear_attn.qkv_rot", (tokens, cfg.hidden_size), lowp),
            z_rot=self.workspace.reserve_tensor("linear_attn.z_rot", (tokens, cfg.hidden_size), lowp),
            rotate_fuse_barrier=self.workspace.reserve_tensor("linear_attn.rotate_fuse_barrier", (2,), DType.INT32),
            qkv_z=qkv_z,
            qkv=qkv,
            z=z,
            qkv_f32=self.workspace.reserve_tensor("linear_attn.qkv_f32", (tokens, qkv_width), DType.FP32),
            ab=ab,
            a=a,
            b=b,
            conv_out=self.workspace.reserve_tensor("linear_attn.conv_out", (tokens, qkv_width), DType.FP32),
            prefill_query=self.workspace.reserve_tensor(
                "linear_attn.prefill_query",
                (tokens, cfg.linear_num_value_heads, cfg.linear_key_head_dim),
                DType.FP32,
            ),
            prefill_key=self.workspace.reserve_tensor(
                "linear_attn.prefill_key",
                (tokens, cfg.linear_num_value_heads, cfg.linear_key_head_dim),
                DType.FP32,
            ),
            prefill_value=self.workspace.reserve_tensor(
                "linear_attn.prefill_value",
                (tokens, cfg.linear_num_value_heads, cfg.linear_value_head_dim),
                DType.FP32,
            ),
            prefill_beta=self.workspace.reserve_tensor("linear_attn.prefill_beta", (tokens, cfg.linear_num_value_heads), DType.FP32),
            prefill_decay=self.workspace.reserve_tensor("linear_attn.prefill_decay", (tokens, cfg.linear_num_value_heads), DType.FP32),
            recurrent_out=self.workspace.reserve_tensor("linear_attn.recurrent_out", (tokens, z_width), DType.FP32),
            recurrent_bf16=self.workspace.reserve_tensor("linear_attn.recurrent_bf16", (tokens, z_width), lowp),
            out_rot=self.workspace.reserve_tensor("linear_attn.out_rot", (tokens, z_width), lowp),
            out_proj=self.workspace.reserve_tensor("linear_attn.out_proj", (tokens, cfg.hidden_size), lowp),
        )

    def rotate_linear_attention_inputs_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv = f"{prefix}.in_proj_qkv"
        z = f"{prefix}.in_proj_z"
        pairs_qkv = self.tensor(f"{qkv}.pairs")
        pairs_z = self.tensor(f"{z}.pairs")
        theta_qkv = self.tensor(f"{qkv}.theta")
        theta_z = self.tensor(f"{z}.theta")
        scales_qkv = self.tensor(f"{qkv}.channel_scales")
        scales_z = self.tensor(f"{z}.channel_scales")
        if tokens == 1 and hidden.ptr == scratch.attn_input.ptr and _rotate_dual_pack8_fused_enabled():
            self._rotate_fuse_ready.add(scratch.rotate_fuse_barrier.ptr)
            return scratch.qkv_rot, scratch.z_rot
        self._rotate_fuse_ready.discard(scratch.rotate_fuse_barrier.ptr)
        paro_rotate2_bf16(
            hidden.ptr,
            scratch.qkv_rot.ptr,
            scratch.z_rot.ptr,
            pairs_qkv.ptr,
            pairs_z.ptr,
            theta_qkv.ptr,
            theta_z.ptr,
            scales_qkv.ptr,
            scales_z.ptr,
            tokens,
            self.config.hidden_size,
            group_size,
            _rotation_krot(pairs_qkv),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        return scratch.qkv_rot, scratch.z_rot

    def project_linear_attention_qkv_z_bf16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv = f"{prefix}.in_proj_qkv"
        z = f"{prefix}.in_proj_z"
        qkv_qweight = self.tensor(f"{qkv}.qweight_pack8_decode")
        z_qweight = self.tensor(f"{z}.qweight_pack8_decode")
        qkv_out_packed = _out_packed_from_generic_transposed_qweight(qkv_qweight)
        z_out_packed = _out_packed_from_generic_transposed_qweight(z_qweight)
        if tokens == 1:
            awq_library = _library_for(library, "awq")
            use_rotate_fused = scratch.rotate_fuse_barrier.ptr in self._rotate_fuse_ready
            if use_rotate_fused:
                gemv_awq_dual_pack8_transposed_rotate_staged_bf16(
                    scratch.attn_input.ptr,
                    scratch.qkv_rot.ptr,
                    scratch.z_rot.ptr,
                    self.tensor(f"{qkv}.pairs").ptr,
                    self.tensor(f"{z}.pairs").ptr,
                    self.tensor(f"{qkv}.theta").ptr,
                    self.tensor(f"{z}.theta").ptr,
                    self.tensor(f"{qkv}.channel_scales").ptr,
                    self.tensor(f"{z}.channel_scales").ptr,
                    qkv_qweight.ptr,
                    self.tensor(f"{qkv}.qzeros").ptr,
                    self.tensor(f"{qkv}.scales").ptr,
                    z_qweight.ptr,
                    self.tensor(f"{z}.qzeros").ptr,
                    self.tensor(f"{z}.scales").ptr,
                    scratch.qkv_z.ptr,
                    scratch.rotate_fuse_barrier.ptr,
                    tokens,
                    scratch.qkv_rot.shape[-1],
                    qkv_out_packed,
                    z_out_packed,
                    group_size,
                    _rotation_krot(self.tensor(f"{qkv}.pairs")),
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
                self._rotate_fuse_ready.discard(scratch.rotate_fuse_barrier.ptr)
            else:
                gemv_awq_dual_pack8_transposed_bf16(
                    scratch.qkv_rot.ptr,
                    scratch.z_rot.ptr,
                    qkv_qweight.ptr,
                    self.tensor(f"{qkv}.qzeros").ptr,
                    self.tensor(f"{qkv}.scales").ptr,
                    z_qweight.ptr,
                    self.tensor(f"{z}.qzeros").ptr,
                    self.tensor(f"{z}.scales").ptr,
                    scratch.qkv_z.ptr,
                    tokens,
                    scratch.qkv_rot.shape[-1],
                    qkv_out_packed,
                    z_out_packed,
                    group_size,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
        else:
            # The dual GEMV writes row-major [qkv,z] per token.  Native
            # prefill conv/GDN consumes contiguous [tokens,qkv] and [tokens,z]
            # streams, so split multi-token prefill into two projections.
            awq_library = _library_for(library, "awq")
            gemv_awq_pack8_transposed_bf16(
                scratch.qkv_rot.ptr,
                qkv_qweight.ptr,
                self.tensor(f"{qkv}.qzeros").ptr,
                self.tensor(f"{qkv}.scales").ptr,
                scratch.qkv.ptr,
                tokens,
                scratch.qkv_rot.shape[-1],
                qkv_out_packed,
                group_size,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
            gemv_awq_pack8_transposed_bf16(
                scratch.z_rot.ptr,
                z_qweight.ptr,
                self.tensor(f"{z}.qzeros").ptr,
                self.tensor(f"{z}.scales").ptr,
                scratch.z.ptr,
                tokens,
                scratch.z_rot.shape[-1],
                z_out_packed,
                group_size,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
        return scratch.qkv, scratch.z

    def project_linear_attention_ab_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int = 1,
        threads: int = 64,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        a_weight = self.tensor(f"{prefix}.in_proj_a.weight")
        b_weight = self.tensor(f"{prefix}.in_proj_b.weight")
        dense_library = _library_for(library, "dense")
        if tokens == 1:
            dense_dual_gemv_out_bf16(
                hidden.ptr,
                a_weight.ptr,
                b_weight.ptr,
                scratch.ab.ptr,
                tokens,
                self.config.hidden_size,
                self.config.linear_num_value_heads,
                self.config.linear_num_value_heads,
                threads=threads,
                stream=stream,
                library=dense_library,
                runtime=self.runtime,
            )
        else:
            # The dual GEMV writes row-major [a,b] per token.  Native prefill
            # GDN consumes contiguous [tokens,a] and [tokens,b] streams.
            dense_gemv_out_bf16(
                hidden.ptr,
                a_weight.ptr,
                scratch.a.ptr,
                tokens,
                self.config.hidden_size,
                self.config.linear_num_value_heads,
                threads=threads,
                stream=stream,
                library=dense_library,
                runtime=self.runtime,
            )
            dense_gemv_out_bf16(
                hidden.ptr,
                b_weight.ptr,
                scratch.b.ptr,
                tokens,
                self.config.hidden_size,
                self.config.linear_num_value_heads,
                threads=threads,
                stream=stream,
                library=dense_library,
                runtime=self.runtime,
            )
        return scratch.a, scratch.b

    def run_linear_attention_conv_gdn_bf16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        conv_weight = self.tensor(f"{prefix}.conv1d.weight")
        dt_bias = self.tensor(f"{prefix}.dt_bias")
        a_log = self.tensor(f"{prefix}.A_log")
        norm_weight = self.tensor(f"{prefix}.norm.weight")
        qwen35_linear_attn_conv_decode_bf16(
            scratch.qkv.ptr,
            conv_state.ptr,
            conv_weight.ptr,
            scratch.conv_out.ptr,
            _linear_qkv_width(self.config),
            self.config.linear_conv_kernel_dim,
            stream=stream,
            library=_library_for(library, "linear_conv"),
            runtime=self.runtime,
        )
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
            scratch.conv_out.ptr,
            scratch.z.ptr,
            scratch.a.ptr,
            scratch.b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            norm_weight.ptr,
            recurrent_state.ptr,
            scratch.recurrent_out.ptr,
            self.config.rms_norm_eps if eps is None else eps,
            self.config.linear_num_key_heads,
            self.config.linear_num_value_heads,
            self.config.linear_key_head_dim,
            self.config.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        return scratch.recurrent_out

    def run_linear_attention_prefill_conv_gdn_bf16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        tokens: int,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run native batched linear-attention prefill conv + recurrent GDN."""

        cfg = self.config
        if tokens < cfg.linear_conv_kernel_dim:
            raise ValueError("native linear-attention prefill requires tokens >= linear_conv_kernel_dim")
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv_width = _linear_qkv_width(cfg)
        z_width = _linear_value_width(cfg)
        conv_weight = self.tensor(f"{prefix}.conv1d.weight")
        dt_bias = self.tensor(f"{prefix}.dt_bias")
        a_log = self.tensor(f"{prefix}.A_log")
        norm_weight = self.tensor(f"{prefix}.norm.weight")
        bf16_to_f32(
            scratch.qkv.ptr,
            scratch.qkv_f32.ptr,
            tokens * qkv_width,
            stream=stream,
            library=_library_for(library, "cast"),
            runtime=self.runtime,
        )
        qwen35_linear_attn_conv_prefill_f32(
            scratch.qkv_f32.ptr,
            conv_state.ptr,
            conv_weight.ptr,
            scratch.conv_out.ptr,
            tokens,
            qkv_width,
            cfg.linear_conv_kernel_dim,
            stream=stream,
            library=_library_for(library, "linear_conv"),
            runtime=self.runtime,
        )
        qwen35_linear_attn_prefill_prepare_f32_bf16(
            scratch.conv_out.ptr,
            scratch.a.ptr,
            scratch.b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            scratch.prefill_query.ptr,
            scratch.prefill_key.ptr,
            scratch.prefill_value.ptr,
            scratch.prefill_beta.ptr,
            scratch.prefill_decay.ptr,
            tokens,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        qwen35_gdn_prefill_recurrent_k2_f32(
            scratch.prefill_query.ptr,
            scratch.prefill_key.ptr,
            scratch.prefill_value.ptr,
            scratch.prefill_beta.ptr,
            scratch.prefill_decay.ptr,
            recurrent_state.ptr,
            scratch.recurrent_out.ptr,
            tokens,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        qwen35_gdn_prefill_rmsnorm_gate_bf16(
            scratch.recurrent_out.ptr,
            scratch.z.ptr,
            norm_weight.ptr,
            scratch.recurrent_bf16.ptr,
            cfg.rms_norm_eps if eps is None else eps,
            tokens,
            cfg.linear_num_value_heads,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        if scratch.recurrent_bf16.shape[-1] != z_width:
            raise ValueError("linear-attention recurrent scratch width mismatch")
        return scratch.recurrent_bf16

    def project_linear_attention_prefill_out_bf16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Rotate and project native prefill linear-attention BF16 hidden outputs."""

        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn.out_proj"
        width = scratch.recurrent_bf16.shape[-1]
        pairs = self.tensor(f"{prefix}.pairs")
        theta = self.tensor(f"{prefix}.theta")
        scales = self.tensor(f"{prefix}.channel_scales")
        paro_rotate1_bf16(
            scratch.recurrent_bf16.ptr,
            scratch.out_rot.ptr,
            pairs.ptr,
            theta.ptr,
            scales.ptr,
            tokens,
            width,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        self.project_pack8_bf16(
            scratch.out_rot,
            scratch.out_proj,
            weight_prefix=prefix,
            rows=tokens,
            in_features=width,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        return scratch.out_proj

    def project_linear_attention_out_bf16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Cast, rotate, and project the FP32 GDN output through linear_attn.out_proj."""

        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn.out_proj"
        width = scratch.recurrent_out.shape[-1]
        f32_to_bf16(
            scratch.recurrent_out.ptr,
            scratch.recurrent_bf16.ptr,
            tokens * width,
            stream=stream,
            library=_library_for(library, "cast"),
            runtime=self.runtime,
        )
        pairs = self.tensor(f"{prefix}.pairs")
        theta = self.tensor(f"{prefix}.theta")
        scales = self.tensor(f"{prefix}.channel_scales")
        paro_rotate1_bf16(
            scratch.recurrent_bf16.ptr,
            scratch.out_rot.ptr,
            pairs.ptr,
            theta.ptr,
            scales.ptr,
            tokens,
            width,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        self.project_pack8_bf16(
            scratch.out_rot,
            scratch.out_proj,
            weight_prefix=prefix,
            rows=tokens,
            in_features=width,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        return scratch.out_proj

    def run_linear_attention_out_proj_bf16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if tokens != 1:
            raise ValueError("linear-attention out-proj orchestrator currently requires tokens=1")
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens)
        self.run_linear_attention_state_bf16(
            hidden,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            scratch=scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        return self.project_linear_attention_out_bf16(
            scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_linear_attention_prefill_state_bf16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run native batched linear-attention prefill state path through RMSNorm+gate."""

        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens)
        self.rotate_linear_attention_inputs_bf16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_qkv_z_bf16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_ab_bf16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        return self.run_linear_attention_prefill_conv_gdn_bf16(
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            tokens=tokens,
            library=library,
            stream=stream,
        )

    def run_linear_attention_prefill_out_proj_bf16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run native batched linear-attention prefill through out_proj."""

        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens)
        self.run_linear_attention_prefill_state_bf16(
            hidden,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            scratch=scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        return self.project_linear_attention_prefill_out_bf16(
            scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_linear_attention_state_bf16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if tokens != 1:
            raise ValueError("linear-attention state orchestrator currently requires tokens=1")
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens)
        self.rotate_linear_attention_inputs_bf16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_qkv_z_bf16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_ab_bf16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        return self.run_linear_attention_conv_gdn_bf16(
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            library=library,
            stream=stream,
        )

    def input_rmsnorm_bf16(
        self,
        hidden: Tensor,
        out: Tensor,
        *,
        tokens: int = 1,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        weight = self.tensor(f"layers.{self.layer_weights.layer_id}.input_layernorm.weight")
        paro_rmsnorm_out_bf16(
            hidden.ptr,
            weight.ptr,
            out.ptr,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps if eps is None else eps,
            stream=stream,
            library=_library_for(library, "norm"),
            runtime=self.runtime,
        )
        return out

    def post_attention_add_rmsnorm_bf16(
        self,
        hidden: Tensor,
        attn_out: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        weight = self.tensor(f"layers.{self.layer_weights.layer_id}.post_attention_layernorm.weight")
        paro_add_rmsnorm_out_bf16(
            hidden.ptr,
            attn_out.ptr,
            weight.ptr,
            scratch.normed.ptr,
            scratch.residual.ptr,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps if eps is None else eps,
            stream=stream,
            library=_library_for(library, "norm"),
            runtime=self.runtime,
        )
        return scratch.normed, scratch.residual

    def run_linear_attention_moe_c1_layer_bf16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        linear_scratch: Qwen35ParoLinearAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        linear_scratch = linear_scratch or self.reserve_linear_attention_scratch(tokens=tokens)
        if tokens == 1:
            moe_scratch = moe_scratch or self.reserve_moe_c1_scratch(tokens=tokens)
        elif not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
            moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens)
        self.input_rmsnorm_bf16(hidden, linear_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        if tokens == 1:
            attn_out = self.run_linear_attention_out_proj_bf16(
                linear_scratch.attn_input,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                scratch=linear_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        else:
            attn_out = self.run_linear_attention_prefill_out_proj_bf16(
                linear_scratch.attn_input,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                scratch=linear_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        mlp_input, residual = self.post_attention_add_rmsnorm_bf16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        if tokens == 1:
            return self.run_moe_c1_bf16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_grouped_compact_bf16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def _full_attention_value_for_kv_write(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        spans: KVLiveSpans,
        rows: int,
        library=None,
        stream: int = 0,
    ) -> tuple[DType, Tensor]:
        if spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD:
            return scratch.value.dtype, scratch.value
        if scratch.value.dtype is DType.FP32:
            return DType.FP32, scratch.value
        value_f32 = scratch.key_raw
        if value_f32.dtype is not DType.FP32 or value_f32.shape != scratch.value.shape:
            raise ValueError("INT8 KV append expects an FP32 key_raw scratch matching value shape")
        count = int(rows) * self.config.num_key_value_heads * self.config.head_dim
        cast_library = _library_for(library, "cast")
        if scratch.value.dtype is DType.FP16:
            fp16_to_f32(
                scratch.value.ptr,
                value_f32.ptr,
                count,
                stream=stream,
                library=cast_library,
                runtime=self.runtime,
            )
        elif scratch.value.dtype is DType.BF16:
            bf16_to_f32(
                scratch.value.ptr,
                value_f32.ptr,
                count,
                stream=stream,
                library=cast_library,
                runtime=self.runtime,
            )
        else:
            raise ValueError("INT8 KV append value scratch must be fp16, bf16, or fp32")
        return DType.FP32, value_f32

    def _append_full_attention_kv_resolved(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        kind: PagedKVWriteKind,
        rows: int = 1,
        block_size: int = 256,
        library=None,
        stream: int = 0,
    ) -> None:
        source_dtype, value_source = self._full_attention_value_for_kv_write(
            scratch,
            spans=spans,
            rows=rows,
            library=library,
            stream=stream,
        )
        write_fn = resolve_paged_kv_write(
            backend=_PAGED_KV_REGISTRY_BACKEND,
            spans=spans,
            kind=kind,
            source_dtype=source_dtype,
        )
        if spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            if key_cache.dtype is not DType.INT8 or value_cache.dtype is not DType.INT8:
                raise ValueError("INT8 KV append requires INT8 key/value cache tensors")
            metadata = spans.scale_metadata
            if metadata is None:
                raise ValueError("INT8 KV append requires scale metadata")
            args = [
                scratch.key.ptr,
                value_source.ptr,
                key_cache.ptr,
                value_cache.ptr,
                metadata.k_scale.ptr,
                metadata.v_scale.ptr,
                spans,
            ]
        else:
            args = [scratch.key.ptr, value_source.ptr, key_cache.ptr, value_cache.ptr, spans]
        if kind is PagedKVWriteKind.DECODE:
            args.extend([block_size, self.config.num_key_value_heads, self.config.head_dim])
        else:
            args.extend([rows, block_size, self.config.num_key_value_heads, self.config.head_dim])
        write_fn(
            *args,
            stream=stream,
            library=_library_for(library, "kv"),
            runtime=self.runtime,
        )

    def append_full_attention_kv(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        block_size: int = 256,
        library=None,
        stream: int = 0,
    ) -> None:
        self._append_full_attention_kv_resolved(
            scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=spans,
            kind=PagedKVWriteKind.DECODE,
            rows=1,
            block_size=block_size,
            library=library,
            stream=stream,
        )

    def decode_full_attention_context_gate_bf16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        gate_tensor = scratch.gate if gate is None else gate
        if spans.max_live_count < 1024:
            qwen35_full_attn_decode_context_bf16(
                scratch.query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                scratch.attn_out.ptr,
                spans.live_counts.ptr,
                spans.max_live_count,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                (self.config.head_dim ** -0.5) if scale is None else scale,
                stream=stream,
                library=_library_for(library, "attention"),
                runtime=self.runtime,
            )
        else:
            qwen35_paged_full_attn_decode_context_bf16_spans(
                scratch.query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                scratch.attn_out.ptr,
                spans,
                spans.max_live_count,
                block_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                (self.config.head_dim ** -0.5) if scale is None else scale,
                stream=stream,
                library=_library_for(library, "attention"),
                runtime=self.runtime,
            )
        qwen35_full_attn_gate_mul_bf16(
            scratch.attn_out.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            self.config.num_attention_heads * self.config.head_dim,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def _decode_full_attention_int8_gqa_gate(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        chunk_size: int,
        num_splits: int,
        kind: PagedAttnDecodeKind,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if key_cache.dtype is not DType.INT8 or value_cache.dtype is not DType.INT8:
            raise ValueError("INT8 paged attention decode requires INT8 key/value cache tensors")
        metadata = spans.scale_metadata
        if metadata is None:
            raise ValueError("INT8 paged attention decode requires scale metadata")
        gate_tensor = scratch.gate if gate is None else gate
        decode_fn = resolve_paged_attn_decode(
            backend=_PAGED_KV_REGISTRY_BACKEND,
            spans=spans,
            kind=kind,
        )
        decode_fn(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            metadata.k_scale.ptr,
            metadata.v_scale.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            scratch.partial_out.ptr,
            scratch.partial_m.ptr,
            scratch.partial_l.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            gate_tensor.shape[-1],
            1,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def decode_full_attention_gqa_gate_bf16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        chunk_size: int,
        num_splits: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            return self._decode_full_attention_int8_gqa_gate(
                scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=spans,
                chunk_size=chunk_size,
                num_splits=num_splits,
                kind=PagedAttnDecodeKind.GQA_SPLITK_GATE_BF16,
                gate=gate,
                block_size=block_size,
                scale=scale,
                library=library,
                stream=stream,
            )
        gate_tensor = scratch.gate if gate is None else gate
        qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            scratch.partial_out.ptr,
            scratch.partial_m.ptr,
            scratch.partial_l.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            gate_tensor.shape[-1],
            1,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def decode_full_attention_split_gate_bf16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        chunk_size: int,
        num_splits: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            return self._decode_full_attention_int8_gqa_gate(
                scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=spans,
                chunk_size=chunk_size,
                num_splits=num_splits,
                kind=PagedAttnDecodeKind.GQA_SPLITK_GATE_BF16,
                gate=gate,
                block_size=block_size,
                scale=scale,
                library=library,
                stream=stream,
            )
        gate_tensor = scratch.gate if gate is None else gate
        decode_fn = _full_attention_split_gate_bf16_fn(
            self.config,
            block_size=block_size,
            num_splits=num_splits,
            max_live_count=spans.max_live_count,
        )
        decode_fn(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            scratch.partial_out.ptr,
            scratch.partial_m.ptr,
            scratch.partial_l.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            gate_tensor.shape[-1],
            1,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def run_full_attention_moe_c1_layer_bf16(
        self,
        hidden: Tensor,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        append_spans: KVLiveSpans,
        decode_spans: KVLiveSpans,
        cos_table: Tensor,
        sin_table: Tensor,
        position: Tensor,
        max_positions: int,
        attention_scratch: Qwen35ParoAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        block_size: int = 256,
        chunk_size: int = 256,
        num_splits: int = 1,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if tokens != 1:
            raise ValueError("full-attention+MoE c=1 layer orchestrator currently requires tokens=1")
        attention_scratch = attention_scratch or self.reserve_full_attention_scratch(tokens=tokens, num_splits=num_splits)
        moe_scratch = moe_scratch or self.reserve_moe_c1_scratch(tokens=tokens)
        self.input_rmsnorm_bf16(hidden, attention_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        self.rotate_full_attention_inputs_bf16(
            attention_scratch.attn_input,
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        self.project_full_attention_qkv_bf16(
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        _query, _key, _value, gate = self.prepare_full_attention_qkv_bf16(
            attention_scratch,
            cos_table=cos_table,
            sin_table=sin_table,
            position=position,
            max_positions=max_positions,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        self.append_full_attention_kv(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=append_spans,
            block_size=block_size,
            library=library,
            stream=stream,
        )
        if not _requires_full_attention_split_decode(decode_spans):
            gated = self.decode_full_attention_context_gate_bf16(
                attention_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=decode_spans,
                gate=gate,
                block_size=block_size,
                library=library,
                stream=stream,
            )
        else:
            gated = self.decode_full_attention_split_gate_bf16(
                attention_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=decode_spans,
                chunk_size=chunk_size,
                num_splits=num_splits,
                gate=gate,
                block_size=block_size,
                library=library,
                stream=stream,
            )
        attn_out = self.project_full_attention_o_bf16(
            gated,
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        mlp_input, residual = self.post_attention_add_rmsnorm_bf16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        return self.run_moe_c1_bf16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def rotate_full_attention_inputs_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn"
        q = f"{prefix}.q_proj"
        k = f"{prefix}.k_proj"
        v = f"{prefix}.v_proj"
        q_pairs = self.tensor(f"{q}.pairs")
        k_pairs = self.tensor(f"{k}.pairs")
        v_pairs = self.tensor(f"{v}.pairs")
        if (
            tokens == 1
            and hidden.ptr == scratch.attn_input.ptr
            and _rotate_dual_pack8_fused_enabled()
            and not _full_attn_kv_pack8_fused_enabled()
        ):
            self._rotate_fuse_ready.add(scratch.rotate_fuse_barrier.ptr)
            paro_rotate1_fp16(
                hidden.ptr,
                scratch.v_rot.ptr,
                v_pairs.ptr,
                self.tensor(f"{v}.theta").ptr,
                self.tensor(f"{v}.channel_scales").ptr,
                tokens,
                self.config.hidden_size,
                group_size,
                _rotation_krot(v_pairs),
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
            return scratch.q_rot, scratch.k_rot, scratch.v_rot
        self._rotate_fuse_ready.discard(scratch.rotate_fuse_barrier.ptr)
        paro_rotate3_fp16(
            hidden.ptr,
            scratch.q_rot.ptr,
            scratch.k_rot.ptr,
            scratch.v_rot.ptr,
            q_pairs.ptr,
            k_pairs.ptr,
            v_pairs.ptr,
            self.tensor(f"{q}.theta").ptr,
            self.tensor(f"{k}.theta").ptr,
            self.tensor(f"{v}.theta").ptr,
            self.tensor(f"{q}.channel_scales").ptr,
            self.tensor(f"{k}.channel_scales").ptr,
            self.tensor(f"{v}.channel_scales").ptr,
            tokens,
            self.config.hidden_size,
            group_size,
            _rotation_krot(q_pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        return scratch.q_rot, scratch.k_rot, scratch.v_rot

    def project_full_attention_qkv_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        producer_trace: Callable[[str, Tensor], None] | None = None,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn"
        q = f"{prefix}.q_proj"
        k = f"{prefix}.k_proj"
        v = f"{prefix}.v_proj"
        q_qweight = self.tensor(f"{q}.qweight_pack8_decode")
        k_qweight = self.tensor(f"{k}.qweight_pack8_decode")
        q_out_packed = _out_packed_from_generic_transposed_qweight(q_qweight)
        k_out_packed = _out_packed_from_generic_transposed_qweight(k_qweight)
        awq_library = _library_for(library, "awq")
        kv_fused = False
        if tokens == 1:
            use_rotate_fused = scratch.rotate_fuse_barrier.ptr in self._rotate_fuse_ready
            if scratch.kv_proj is not None and not use_rotate_fused:
                v_qweight = self.tensor(f"{v}.qweight_pack8_decode")
                v_out_packed = _out_packed_from_generic_transposed_qweight(v_qweight)
                self.project_pack8_fp16(
                    scratch.q_rot,
                    scratch.q_proj,
                    weight_prefix=q,
                    rows=tokens,
                    group_size=group_size,
                    library=library,
                    stream=stream,
                )
                gemv_awq_dual_pack8_transposed_fp16(
                    scratch.k_rot.ptr,
                    scratch.v_rot.ptr,
                    k_qweight.ptr,
                    self.tensor(f"{k}.qzeros").ptr,
                    self.tensor(f"{k}.scales").ptr,
                    v_qweight.ptr,
                    self.tensor(f"{v}.qzeros").ptr,
                    self.tensor(f"{v}.scales").ptr,
                    scratch.key_bf16.ptr,
                    tokens,
                    scratch.k_rot.shape[-1],
                    k_out_packed,
                    v_out_packed,
                    group_size,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
                kv_fused = True
            elif use_rotate_fused:
                gemv_awq_dual_pack8_transposed_rotate_staged_fp16(
                    scratch.attn_input.ptr,
                    scratch.q_rot.ptr,
                    scratch.k_rot.ptr,
                    self.tensor(f"{q}.pairs").ptr,
                    self.tensor(f"{k}.pairs").ptr,
                    self.tensor(f"{q}.theta").ptr,
                    self.tensor(f"{k}.theta").ptr,
                    self.tensor(f"{q}.channel_scales").ptr,
                    self.tensor(f"{k}.channel_scales").ptr,
                    q_qweight.ptr,
                    self.tensor(f"{q}.qzeros").ptr,
                    self.tensor(f"{q}.scales").ptr,
                    k_qweight.ptr,
                    self.tensor(f"{k}.qzeros").ptr,
                    self.tensor(f"{k}.scales").ptr,
                    scratch.q_proj_key.ptr,
                    scratch.rotate_fuse_barrier.ptr,
                    tokens,
                    scratch.q_rot.shape[-1],
                    q_out_packed,
                    k_out_packed,
                    group_size,
                    _rotation_krot(self.tensor(f"{q}.pairs")),
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
                self._rotate_fuse_ready.discard(scratch.rotate_fuse_barrier.ptr)
            else:
                gemv_awq_dual_pack8_transposed_fp16(
                    scratch.q_rot.ptr,
                    scratch.k_rot.ptr,
                    q_qweight.ptr,
                    self.tensor(f"{q}.qzeros").ptr,
                    self.tensor(f"{q}.scales").ptr,
                    k_qweight.ptr,
                    self.tensor(f"{k}.qzeros").ptr,
                    self.tensor(f"{k}.scales").ptr,
                    scratch.q_proj_key.ptr,
                    tokens,
                    scratch.q_rot.shape[-1],
                    q_out_packed,
                    k_out_packed,
                    group_size,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
        else:
            awq_fusedw4_prefill_dual_fp16(
                scratch.q_rot.ptr,
                scratch.k_rot.ptr,
                q_qweight.ptr,
                self.tensor(f"{q}.qzeros").ptr,
                self.tensor(f"{q}.scales").ptr,
                k_qweight.ptr,
                self.tensor(f"{k}.qzeros").ptr,
                self.tensor(f"{k}.scales").ptr,
                scratch.q_proj.ptr,
                scratch.key_bf16.ptr,
                tokens,
                scratch.q_rot.shape[-1],
                q_out_packed,
                k_out_packed,
                group_size,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
        if producer_trace is not None:
            producer_trace("q_proj_key_after_project", scratch.q_proj_key)
        if not kv_fused:
            self.project_pack8_fp16(
                scratch.v_rot,
                scratch.value,
                weight_prefix=f"{prefix}.v_proj",
                rows=tokens,
                group_size=group_size,
                threads=64 if tokens > 1 else 128,
                library=library,
                stream=stream,
            )
        if producer_trace is not None:
            producer_trace("value_after_project", scratch.value)
        return scratch.q_proj, scratch.key_bf16, scratch.value

    def prepare_full_attention_qkv_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        cos_table: Tensor,
        sin_table: Tensor,
        position: Tensor,
        max_positions: int,
        tokens: int = 1,
        query_bf16_out: Tensor | None = None,
        producer_trace: Callable[[str, Tensor], None] | None = None,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        qwen35_split_qgate_fp16(
            scratch.q_proj.ptr,
            scratch.query_raw.ptr,
            scratch.gate.ptr,
            tokens,
            cfg.num_attention_heads,
            cfg.head_dim,
            stream=stream,
            library=_library_for(library, "qwen_rotary"),
            runtime=self.runtime,
        )
        if producer_trace is not None:
            producer_trace("query_raw_after_split", scratch.query_raw)
            producer_trace("gate_after_split", scratch.gate)
        fp16_to_f32(
            scratch.key_bf16.ptr,
            scratch.key_raw.ptr,
            tokens * kv_width,
            stream=stream,
            library=_library_for(library, "cast"),
            runtime=self.runtime,
        )
        if producer_trace is not None:
            producer_trace("key_raw_after_cast", scratch.key_raw)
        if query_bf16_out is not None:
            if query_bf16_out.dtype is not DType.BF16 or query_bf16_out.shape != scratch.query.shape:
                raise ValueError("AOTriton query BF16 output must match full-attention query shape")
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn"
        qwen_rotary_library = _library_for(library, "qwen_rotary")
        if tokens == 1:
            qwen35_head_rmsnorm_partial_rotary_position_f32_bf16(
                scratch.query_raw.ptr,
                scratch.key_raw.ptr,
                self.tensor(f"{prefix}.q_norm.weight").ptr,
                self.tensor(f"{prefix}.k_norm.weight").ptr,
                cos_table.ptr,
                sin_table.ptr,
                position.ptr,
                scratch.query.ptr,
                scratch.key.ptr,
                self.config.rms_norm_eps,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rotary_dim or cfg.head_dim,
                max_positions,
                stream=stream,
                library=qwen_rotary_library,
                runtime=self.runtime,
            )
        elif query_bf16_out is not None:
            qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32(
                scratch.query_raw.ptr,
                scratch.key_raw.ptr,
                self.tensor(f"{prefix}.q_norm.weight").ptr,
                self.tensor(f"{prefix}.k_norm.weight").ptr,
                cos_table.ptr,
                sin_table.ptr,
                position.ptr,
                query_bf16_out.ptr,
                scratch.key.ptr,
                self.config.rms_norm_eps,
                tokens,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rotary_dim or cfg.head_dim,
                max_positions,
                stream=stream,
                library=qwen_rotary_library,
                runtime=self.runtime,
            )
        else:
            qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16(
                scratch.query_raw.ptr,
                scratch.key_raw.ptr,
                self.tensor(f"{prefix}.q_norm.weight").ptr,
                self.tensor(f"{prefix}.k_norm.weight").ptr,
                cos_table.ptr,
                sin_table.ptr,
                position.ptr,
                scratch.query.ptr,
                scratch.key.ptr,
                self.config.rms_norm_eps,
                tokens,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rotary_dim or cfg.head_dim,
                max_positions,
                stream=stream,
                library=qwen_rotary_library,
                runtime=self.runtime,
            )
        query_out = query_bf16_out if query_bf16_out is not None else scratch.query
        if producer_trace is not None:
            producer_trace("query_after_prepare", query_out)
            producer_trace("key_after_prepare", scratch.key)
        return query_out, scratch.key, scratch.value, scratch.gate

    @staticmethod
    def _row_tensor_view(tensor: Tensor, row: int) -> Tensor:
        if not tensor.shape:
            raise ValueError("cannot row-slice a scalar tensor")
        rows = int(tensor.shape[0])
        if row < 0 or row >= rows:
            raise ValueError(f"row {row} outside tensor shape {tensor.shape}")
        row_elements = 1
        for dim in tensor.shape[1:]:
            row_elements *= int(dim)
        return Tensor.from_handle(
            tensor.ptr + int(row) * row_elements * tensor.dtype.itemsize,
            (1, *tuple(int(dim) for dim in tensor.shape[1:])),
            tensor.dtype,
            tensor.device,
        )

    def _decode_row_linear_attention_projection_scratch(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        row: int,
    ) -> Qwen35ParoLinearAttentionScratch:
        """Return row-local token-1 projection scratch for c>N linear decode.

        The token-1 linear-attention projection kernels write packed [qkv,z] and
        [a,b] rows, while the native segmented conv/GDN path consumes planar
        batch buffers.  Use a one-row temporary projection scratch and copy the
        resulting qkv/z/a/b rows into the batch-shaped scratch before native
        segmented state updates.
        """

        cfg = self.config
        lowp = scratch.attn_input.dtype
        qkv_width = _linear_qkv_width(cfg)
        z_width = _linear_value_width(cfg)
        qkv_z = self.workspace.reserve_tensor("linear_attn.decode_row.qkv_z", (1, qkv_width + z_width), lowp)
        qkv = Tensor.from_handle(qkv_z.ptr, (1, qkv_width), lowp, qkv_z.device)
        z = Tensor.from_handle(qkv_z.ptr + qkv_width * lowp.itemsize, (1, z_width), lowp, qkv_z.device)
        ab = self.workspace.reserve_tensor("linear_attn.decode_row.ab", (1, 2 * cfg.linear_num_value_heads), lowp)
        a = Tensor.from_handle(ab.ptr, (1, cfg.linear_num_value_heads), lowp, ab.device)
        b = Tensor.from_handle(
            ab.ptr + cfg.linear_num_value_heads * lowp.itemsize,
            (1, cfg.linear_num_value_heads),
            lowp,
            ab.device,
        )
        return Qwen35ParoLinearAttentionScratch(
            attn_input=self._row_tensor_view(hidden, row),
            qkv_rot=self.workspace.reserve_tensor("linear_attn.decode_row.qkv_rot", (1, cfg.hidden_size), lowp),
            z_rot=self.workspace.reserve_tensor("linear_attn.decode_row.z_rot", (1, cfg.hidden_size), lowp),
            rotate_fuse_barrier=self.workspace.reserve_tensor("linear_attn.decode_row.rotate_fuse_barrier", (2,), DType.INT32),
            qkv_z=qkv_z,
            qkv=qkv,
            z=z,
            qkv_f32=self._row_tensor_view(scratch.qkv_f32, row),
            ab=ab,
            a=a,
            b=b,
            conv_out=self._row_tensor_view(scratch.conv_out, row),
            prefill_query=self._row_tensor_view(scratch.prefill_query, row),
            prefill_key=self._row_tensor_view(scratch.prefill_key, row),
            prefill_value=self._row_tensor_view(scratch.prefill_value, row),
            prefill_beta=self._row_tensor_view(scratch.prefill_beta, row),
            prefill_decay=self._row_tensor_view(scratch.prefill_decay, row),
            recurrent_out=self._row_tensor_view(scratch.recurrent_out, row),
            recurrent_bf16=self._row_tensor_view(scratch.recurrent_bf16, row),
            out_rot=self._row_tensor_view(scratch.out_rot, row),
            out_proj=self._row_tensor_view(scratch.out_proj, row),
        )

    def _decode_row_linear_attention_planar_scratch(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        row: int,
    ) -> Qwen35ParoLinearAttentionScratch:
        """Return row-local planar scratch views for c>N linear decode state/out replay."""

        return Qwen35ParoLinearAttentionScratch(
            attn_input=self._row_tensor_view(scratch.attn_input, row),
            qkv_rot=self._row_tensor_view(scratch.qkv_rot, row),
            z_rot=self._row_tensor_view(scratch.z_rot, row),
            rotate_fuse_barrier=scratch.rotate_fuse_barrier,
            qkv_z=scratch.qkv_z,
            qkv=self._row_tensor_view(scratch.qkv, row),
            z=self._row_tensor_view(scratch.z, row),
            qkv_f32=self._row_tensor_view(scratch.qkv_f32, row),
            ab=self._row_tensor_view(scratch.ab, row),
            a=self._row_tensor_view(scratch.a, row),
            b=self._row_tensor_view(scratch.b, row),
            conv_out=self._row_tensor_view(scratch.conv_out, row),
            prefill_query=self._row_tensor_view(scratch.prefill_query, row),
            prefill_key=self._row_tensor_view(scratch.prefill_key, row),
            prefill_value=self._row_tensor_view(scratch.prefill_value, row),
            prefill_beta=self._row_tensor_view(scratch.prefill_beta, row),
            prefill_decay=self._row_tensor_view(scratch.prefill_decay, row),
            recurrent_out=self._row_tensor_view(scratch.recurrent_out, row),
            recurrent_bf16=self._row_tensor_view(scratch.recurrent_bf16, row),
            out_rot=self._row_tensor_view(scratch.out_rot, row),
            out_proj=self._row_tensor_view(scratch.out_proj, row),
        )

    def project_linear_attention_decode_rows_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Replay linear-attention decode projections with the token-1 path.

        This is a diagnostic bridge for c>N decode: projections use c1 kernels
        row by row, then the native segmented conv/GDN kernels consume the
        reconstructed batch-shaped qkv/z/a/b buffers.
        """

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        runtime = self.runtime or get_hip_runtime()
        for row in range(tokens):
            row_scratch = self._decode_row_linear_attention_projection_scratch(hidden, scratch, row)
            self.rotate_linear_attention_inputs_fp16(
                row_scratch.attn_input,
                row_scratch,
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
            self.project_linear_attention_qkv_z_fp16(
                row_scratch,
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
            self.project_linear_attention_ab_fp16(
                row_scratch.attn_input,
                row_scratch,
                tokens=1,
                library=library,
                stream=stream,
            )
            for dst, src in (
                (self._row_tensor_view(scratch.qkv, row), row_scratch.qkv),
                (self._row_tensor_view(scratch.z, row), row_scratch.z),
                (self._row_tensor_view(scratch.a, row), row_scratch.a),
                (self._row_tensor_view(scratch.b, row), row_scratch.b),
            ):
                runtime.memcpy_async(
                    dst.ptr,
                    src.ptr,
                    src.numel * src.dtype.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
        return scratch.qkv, scratch.z, scratch.a, scratch.b

    def project_linear_attention_decode_rows_qkv_z_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Replay only linear-attention QKV/Z projections with the token-1 path."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        runtime = self.runtime or get_hip_runtime()
        for row in range(tokens):
            row_scratch = self._decode_row_linear_attention_projection_scratch(hidden, scratch, row)
            self.rotate_linear_attention_inputs_fp16(
                row_scratch.attn_input,
                row_scratch,
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
            self.project_linear_attention_qkv_z_fp16(
                row_scratch,
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
            for dst, src in (
                (self._row_tensor_view(scratch.qkv, row), row_scratch.qkv),
                (self._row_tensor_view(scratch.z, row), row_scratch.z),
            ):
                runtime.memcpy_async(
                    dst.ptr,
                    src.ptr,
                    src.numel * src.dtype.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
        return scratch.qkv, scratch.z

    def project_linear_attention_decode_rows_ab_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Replay only linear-attention A/B projections with the token-1 path."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        runtime = self.runtime or get_hip_runtime()
        for row in range(tokens):
            row_scratch = self._decode_row_linear_attention_projection_scratch(hidden, scratch, row)
            self.project_linear_attention_ab_fp16(
                row_scratch.attn_input,
                row_scratch,
                tokens=1,
                library=library,
                stream=stream,
            )
            for dst, src in (
                (self._row_tensor_view(scratch.a, row), row_scratch.a),
                (self._row_tensor_view(scratch.b, row), row_scratch.b),
            ):
                runtime.memcpy_async(
                    dst.ptr,
                    src.ptr,
                    src.numel * src.dtype.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
        return scratch.a, scratch.b

    def run_linear_attention_decode_rows_state_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        state_pairs: Sequence[tuple[Tensor, Tensor]],
        tokens: int,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Replay linear-attention conv/GDN recurrent updates with token-1 state kernels."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if len(state_pairs) < tokens:
            raise ValueError("state_pairs must provide one conv/recurrent pair per token")
        for row in range(tokens):
            conv_state, recurrent_state = state_pairs[row]
            self.run_linear_attention_conv_gdn_fp16(
                self._decode_row_linear_attention_planar_scratch(scratch, row),
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                library=library,
                stream=stream,
            )
        return scratch.recurrent_out

    def project_linear_attention_decode_rows_out_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Replay linear-attention output projection with token-1 kernels per row.

        This path matches token-1 decode state replay: ``scratch.recurrent_out``
        already contains the post-GDN/RMSNorm/gate FP32 row and must be cast to
        lowp before rotation/output projection.
        """

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        for row in range(tokens):
            self.project_linear_attention_out_fp16(
                self._decode_row_linear_attention_planar_scratch(scratch, row),
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return scratch.out_proj

    def project_linear_attention_prefill_rows_out_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Replay output projection per row from segmented-state lowp output.

        Segment-aware state replay writes raw recurrent state to
        ``scratch.recurrent_out`` and the post-GDN/RMSNorm/gate activation to
        ``scratch.recurrent_bf16``.  Per-row output diagnostics for that state
        path must therefore consume ``recurrent_bf16`` directly instead of
        recasting the raw recurrent tensor as token-1 decode does.
        """

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        for row in range(tokens):
            self.project_linear_attention_prefill_out_fp16(
                self._decode_row_linear_attention_planar_scratch(scratch, row),
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return scratch.out_proj

    def _decode_row_full_attention_temp_scratch(
        self,
        scratch: Qwen35ParoAttentionScratch,
    ) -> Qwen35ParoAttentionScratch:
        """Return an independent token-1 scratch for full-attention decode diagnostics."""

        cfg = self.config
        q_width = cfg.num_attention_heads * cfg.head_dim
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        lowp = scratch.attn_input.dtype
        gated = scratch.gate.dtype
        query_dtype = scratch.query.dtype
        q_proj_key = self.workspace.reserve_tensor("attn.decode_row.q_proj_key", (1, 2 * q_width + kv_width), lowp)
        q_proj = Tensor.from_handle(q_proj_key.ptr, (1, 2 * q_width), lowp, q_proj_key.device)
        key_bf16 = Tensor.from_handle(
            q_proj_key.ptr + 2 * q_width * lowp.itemsize,
            (1, kv_width),
            lowp,
            q_proj_key.device,
        )
        kv_proj = None
        if _full_attn_kv_pack8_fused_enabled():
            kv_proj = self.workspace.reserve_tensor("attn.decode_row.kv_proj", (1, 2 * kv_width), lowp)
            key_bf16 = Tensor.from_handle(kv_proj.ptr, (1, kv_width), lowp, kv_proj.device)
            value = Tensor.from_handle(
                kv_proj.ptr + kv_width * lowp.itemsize,
                (1, cfg.num_key_value_heads, cfg.head_dim),
                lowp,
                kv_proj.device,
            )
        else:
            value = self.workspace.reserve_tensor(
                "attn.decode_row.value",
                (1, cfg.num_key_value_heads, cfg.head_dim),
                lowp,
            )
        return Qwen35ParoAttentionScratch(
            attn_input=self.workspace.reserve_tensor("attn.decode_row.input", (1, cfg.hidden_size), lowp),
            q_rot=self.workspace.reserve_tensor("attn.decode_row.q_rot", (1, cfg.hidden_size), lowp),
            k_rot=self.workspace.reserve_tensor("attn.decode_row.k_rot", (1, cfg.hidden_size), lowp),
            v_rot=self.workspace.reserve_tensor("attn.decode_row.v_rot", (1, cfg.hidden_size), lowp),
            rotate_fuse_barrier=self.workspace.reserve_tensor("attn.decode_row.rotate_fuse_barrier", (2,), DType.INT32),
            q_proj_key=q_proj_key,
            q_proj=q_proj,
            key_bf16=key_bf16,
            query_raw=self.workspace.reserve_tensor(
                "attn.decode_row.query_raw",
                (1, cfg.num_attention_heads, cfg.head_dim),
                DType.FP32,
            ),
            key_raw=self.workspace.reserve_tensor(
                "attn.decode_row.key_raw",
                (1, cfg.num_key_value_heads, cfg.head_dim),
                DType.FP32,
            ),
            query=self.workspace.reserve_tensor(
                "attn.decode_row.query",
                (1, cfg.num_attention_heads, cfg.head_dim),
                query_dtype,
            ),
            key=self.workspace.reserve_tensor(
                "attn.decode_row.key",
                (1, cfg.num_key_value_heads, cfg.head_dim),
                DType.FP32,
            ),
            value=value,
            kv_proj=kv_proj,
            gate=self.workspace.reserve_tensor(
                "attn.decode_row.gate",
                (1, cfg.num_attention_heads, cfg.head_dim),
                gated,
            ),
            partial_out=self.workspace.reserve_tensor(
                "attn.decode_row.partial_out",
                (cfg.num_attention_heads, 1, cfg.head_dim),
                DType.FP32,
            ),
            partial_m=self.workspace.reserve_tensor("attn.decode_row.partial_m", (cfg.num_attention_heads, 1), DType.FP32),
            partial_l=self.workspace.reserve_tensor("attn.decode_row.partial_l", (cfg.num_attention_heads, 1), DType.FP32),
            attn_out=self.workspace.reserve_tensor("attn.decode_row.out", (cfg.num_attention_heads, cfg.head_dim), DType.FP32),
            gated_attn=self.workspace.reserve_tensor("attn.decode_row.gated", (1, q_width), gated),
            o_rot=self.workspace.reserve_tensor("attn.decode_row.o_rot", (1, q_width), lowp),
            o_proj=self.workspace.reserve_tensor("attn.decode_row.o_proj", (1, cfg.hidden_size), lowp),
        )

    def _decode_row_full_attention_scratch(
        self,
        scratch: Qwen35ParoAttentionScratch,
        row: int,
    ) -> Qwen35ParoAttentionScratch:
        """Return row-local views for c>N decode Q/K/V preparation.

        ``q_proj_key`` is laid out as all Q/G rows followed by all K rows for
        batch prefill.  The token-1 decode projection kernel, however, writes a
        temporary row as contiguous ``[q_gate, key]``.  Use the start of that
        buffer as per-row temporary storage and copy only the prepared
        query/key/value/gate row into the batch-shaped scratch outputs.
        """

        cfg = self.config
        q_width = cfg.num_attention_heads * cfg.head_dim
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        temp = Tensor.from_handle(
            scratch.q_proj_key.ptr,
            (1, 2 * q_width + kv_width),
            scratch.q_proj_key.dtype,
            scratch.q_proj_key.device,
        )
        q_proj = Tensor.from_handle(temp.ptr, (1, 2 * q_width), temp.dtype, temp.device)
        key_bf16 = Tensor.from_handle(
            temp.ptr + 2 * q_width * temp.dtype.itemsize,
            (1, kv_width),
            temp.dtype,
            temp.device,
        )
        return Qwen35ParoAttentionScratch(
            attn_input=self._row_tensor_view(scratch.attn_input, row),
            q_rot=self._row_tensor_view(scratch.q_rot, row),
            k_rot=self._row_tensor_view(scratch.k_rot, row),
            v_rot=self._row_tensor_view(scratch.v_rot, row),
            rotate_fuse_barrier=scratch.rotate_fuse_barrier,
            q_proj_key=temp,
            q_proj=q_proj,
            key_bf16=key_bf16,
            query_raw=self._row_tensor_view(scratch.query_raw, row),
            key_raw=self._row_tensor_view(scratch.key_raw, row),
            query=self._row_tensor_view(scratch.query, row),
            key=self._row_tensor_view(scratch.key, row),
            value=self._row_tensor_view(scratch.value, row),
            kv_proj=None,
            gate=self._row_tensor_view(scratch.gate, row),
            partial_out=scratch.partial_out,
            partial_m=scratch.partial_m,
            partial_l=scratch.partial_l,
            attn_out=scratch.attn_out,
            gated_attn=self._row_tensor_view(scratch.gated_attn, row),
            o_rot=self._row_tensor_view(scratch.o_rot, row),
            o_proj=self._row_tensor_view(scratch.o_proj, row),
        )

    def prepare_full_attention_qkv_fp16_decode_rows(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        cos_table: Tensor,
        sin_table: Tensor,
        positions: Tensor,
        max_positions: int,
        tokens: int,
        group_size: int = 128,
        input_scratch_trace: Callable[[str, int, Qwen35ParoAttentionScratch], None] | None = None,
        qkv_tensor_trace: Callable[[str, int, Tensor], None] | None = None,
        force_per_row_scratch: bool = False,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prepare decode Q/K/V rows with the token-1 projection path.

        This keeps compact c>N decode numerically aligned with independent c=1
        decode while the prefill-oriented W4 projection path is audited for
        long autoregressive parity.
        """

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        runtime = self.runtime or get_hip_runtime()
        for row in range(tokens):
            row_scratch = self._decode_row_full_attention_scratch(scratch, row)
            active_scratch = row_scratch
            if force_per_row_scratch and tokens > 1:
                active_scratch = self._decode_row_full_attention_temp_scratch(scratch)
                runtime.memcpy_async(
                    active_scratch.attn_input.ptr,
                    row_scratch.attn_input.ptr,
                    row_scratch.attn_input.numel * row_scratch.attn_input.dtype.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            row_position = Tensor.from_handle(
                positions.ptr + row * DType.INT64.itemsize,
                (1,),
                DType.INT64,
                positions.device,
            )
            self.rotate_full_attention_inputs_fp16(
                active_scratch.attn_input,
                active_scratch,
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
            if input_scratch_trace is not None:
                input_scratch_trace("attn_input_after_rotate", row, active_scratch)
            def producer_trace(stage: str, tensor: Tensor, *, _row: int = row) -> None:
                if qkv_tensor_trace is not None:
                    qkv_tensor_trace(stage, _row, tensor)

            self.project_full_attention_qkv_fp16(
                active_scratch,
                tokens=1,
                group_size=group_size,
                producer_trace=producer_trace if qkv_tensor_trace is not None else None,
                library=library,
                stream=stream,
            )
            if input_scratch_trace is not None:
                input_scratch_trace("attn_input_after_project", row, active_scratch)

            self.prepare_full_attention_qkv_fp16(
                active_scratch,
                cos_table=cos_table,
                sin_table=sin_table,
                position=row_position,
                max_positions=max_positions,
                tokens=1,
                producer_trace=producer_trace if qkv_tensor_trace is not None else None,
                library=library,
                stream=stream,
            )
            if force_per_row_scratch and tokens > 1:
                for dst, src in (
                    (row_scratch.query, active_scratch.query),
                    (row_scratch.key, active_scratch.key),
                    (row_scratch.value, active_scratch.value),
                    (row_scratch.gate, active_scratch.gate),
                ):
                    runtime.memcpy_async(
                        dst.ptr,
                        src.ptr,
                        src.numel * src.dtype.itemsize,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
            if input_scratch_trace is not None:
                input_scratch_trace("attn_input_after_prepare", row, active_scratch)
        return scratch.query, scratch.key, scratch.value, scratch.gate

    def append_full_attention_kv_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        block_size: int = 256,
        library=None,
        stream: int = 0,
    ) -> None:
        self._append_full_attention_kv_resolved(
            scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=spans,
            kind=PagedKVWriteKind.DECODE,
            rows=1,
            block_size=block_size,
            library=library,
            stream=stream,
        )

    def prefill_full_attention_gqa_gate_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        rows: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        gate_tensor = scratch.gate if gate is None else gate
        qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            spans,
            rows,
            spans.max_live_count,
            block_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            gate_tensor.shape[-1],
            1,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def prefill_full_attention_varlen_gqa_gate_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        rows: int,
        segments: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        gate_tensor = scratch.gate if gate is None else gate
        qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            spans,
            cu_seqlens_q.ptr,
            cu_seqlens_k.ptr,
            rows,
            segments,
            spans.max_live_count,
            block_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            gate_tensor.shape[-1],
            1,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def prefill_full_attention_aotriton_varlen_gqa_bf16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        rows: int,
        segments: int,
        kv_rows: int | None = None,
        query_bf16: Tensor | None = None,
        key_cache: Tensor | None = None,
        value_cache: Tensor | None = None,
        attn_bf16_out: Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run AOTriton compact-varlen GQA prefill and return BF16 attention output."""

        _check_positive(rows, "rows")
        _check_positive(segments, "segments")
        key_rows = int(rows if kv_rows is None else kv_rows)
        if key_rows < rows:
            raise ValueError("AOTriton key/value rows must cover query rows")
        if cu_seqlens_q.dtype is not DType.INT32 or cu_seqlens_k.dtype is not DType.INT32:
            raise ValueError("AOTriton compact-varlen prefill expects int32 cu_seqlens tensors")
        if scratch.key.dtype is not DType.FP32 or scratch.value.dtype is not DType.FP16:
            raise ValueError("AOTriton prefill expects FP32 K source tensor and FP16 V scratch tensor")
        if query_bf16 is None and scratch.query.dtype is not DType.FP32:
            raise ValueError("AOTriton prefill expects an FP32 Q source tensor unless query_bf16 is provided")
        lse = self.workspace.reserve_tensor("attn.aotriton_lse", (self.config.num_attention_heads, rows), DType.FP32)
        q_heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        q_width = q_heads * head_dim
        kv_width = kv_heads * head_dim
        if query_bf16 is None:
            q_bf16 = self.workspace.reserve_tensor("attn.aotriton_q_bf16", scratch.query.shape, DType.BF16)
        else:
            if query_bf16.dtype is not DType.BF16 or query_bf16.shape != scratch.query.shape:
                raise ValueError("AOTriton query BF16 tensor must match full-attention query shape")
            q_bf16 = query_bf16
        if attn_bf16_out is None:
            attn_bf16 = self.workspace.reserve_tensor("attn.aotriton_out_bf16", scratch.query.shape, DType.BF16)
        else:
            if attn_bf16_out.dtype is not DType.BF16 or attn_bf16_out.shape != scratch.query.shape:
                raise ValueError("AOTriton BF16 attention output tensor must match query shape")
            attn_bf16 = attn_bf16_out
        atomic_counter = self.workspace.reserve_tensor("attn.aotriton_atomic", (1,), DType.INT32)
        cast_library = _library_for(library, "cast")
        if query_bf16 is None:
            f32_to_bf16(
                scratch.query.ptr,
                q_bf16.ptr,
                rows * q_width,
                stream=stream,
                library=cast_library,
                runtime=self.runtime,
            )
        if key_cache is None or value_cache is None:
            if kv_rows is not None and key_rows != rows:
                raise ValueError("AOTriton scratch-backed K/V cannot use more key rows than query rows")
            k_bf16 = self.workspace.reserve_tensor("attn.aotriton_k_bf16", scratch.key.shape, DType.BF16)
            v_bf16 = self.workspace.reserve_tensor("attn.aotriton_v_bf16", scratch.value.shape, DType.BF16)
            f32_to_bf16(
                scratch.key.ptr,
                k_bf16.ptr,
                rows * kv_width,
                stream=stream,
                library=cast_library,
                runtime=self.runtime,
            )
            fp16_to_bf16(
                scratch.value.ptr,
                v_bf16.ptr,
                rows * kv_width,
                stream=stream,
                library=cast_library,
                runtime=self.runtime,
            )
            k_tensor = aotriton_tensor4(
                k_bf16.ptr,
                (1, kv_heads, rows, head_dim),
                (kv_width * rows, head_dim, kv_width, 1),
                DType.BF16,
            )
            v_tensor = aotriton_tensor4(
                v_bf16.ptr,
                (1, kv_heads, rows, head_dim),
                (kv_width * rows, head_dim, kv_width, 1),
                DType.BF16,
            )
        else:
            if key_cache.dtype is not DType.BF16 or value_cache.dtype is not DType.BF16:
                raise ValueError("AOTriton cache-backed K/V expects BF16 KV cache tensors")
            if len(key_cache.shape) != 4 or len(value_cache.shape) != 4:
                raise ValueError("AOTriton cache-backed K/V expects [blocks, block, kv_heads, head_dim] tensors")
            if key_cache.shape != value_cache.shape:
                raise ValueError("AOTriton key/value cache shapes must match")
            if int(key_cache.shape[2]) != kv_heads or int(key_cache.shape[3]) != head_dim:
                raise ValueError("AOTriton KV cache shape does not match attention head layout")
            cache_rows = int(key_cache.shape[0]) * int(key_cache.shape[1])
            if key_rows > cache_rows:
                raise ValueError("AOTriton KV cache is too small for prefill rows")
            # The single-request prompt path appends K/V into an identity block table before
            # AOTriton runs.  That BF16 cache image is bit-identical to the prior
            # scratch-to-BF16 casts, so reuse it and skip two full-row cast kernels.
            k_tensor = aotriton_tensor4(
                key_cache.ptr,
                (1, kv_heads, key_rows, head_dim),
                (kv_width * key_rows, head_dim, kv_width, 1),
                DType.BF16,
            )
            v_tensor = aotriton_tensor4(
                value_cache.ptr,
                (1, kv_heads, key_rows, head_dim),
                (kv_width * key_rows, head_dim, kv_width, 1),
                DType.BF16,
            )
        aotriton_library = _library_for(library, "aotriton")
        aotriton_attn_fwd_v3_compact_varlen(
            aotriton_tensor4(q_bf16.ptr, (1, q_heads, rows, head_dim), (q_width * rows, head_dim, q_width, 1), DType.BF16),
            k_tensor,
            v_tensor,
            aotriton_tensor1(cu_seqlens_q.ptr, (segments + 1,), (1,), DType.INT32),
            aotriton_tensor1(cu_seqlens_k.ptr, (segments + 1,), (1,), DType.INT32),
            aotriton_tensor2(lse.ptr, (q_heads, rows), (rows, 1), DType.FP32),
            aotriton_tensor4(attn_bf16.ptr, (1, q_heads, rows, head_dim), (q_width * rows, head_dim, q_width, 1), DType.BF16),
            persistent_atomic_counter_ptr=atomic_counter.ptr,
            max_seqlen_q=int(rows if max_seqlen_q is None else max_seqlen_q),
            max_seqlen_k=int(key_rows if max_seqlen_k is None else max_seqlen_k),
            sm_scale=(self.config.head_dim ** -0.5) if scale is None else scale,
            is_causal=True,
            stream=stream,
            library=aotriton_library,
            runtime=self.runtime,
        )
        return attn_bf16

    def prefill_full_attention_aotriton_varlen_gqa_gate_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        rows: int,
        segments: int,
        kv_rows: int | None = None,
        gate: Tensor | None = None,
        query_bf16: Tensor | None = None,
        key_cache: Tensor | None = None,
        value_cache: Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run AOTriton compact-varlen GQA prefill and apply Qwen3.5 sigmoid gate."""

        if scratch.gated_attn.dtype is not DType.FP16:
            raise ValueError("AOTriton gate post-pass currently writes FP16 attention output")
        gate_tensor = scratch.gate if gate is None else gate
        if gate_tensor.dtype is not DType.FP16:
            raise ValueError("AOTriton gate post-pass currently expects FP16 gate tensor")
        if gate_tensor.shape != scratch.query.shape:
            raise ValueError("gate tensor must match query shape for AOTriton post-pass")
        attn_bf16 = self.prefill_full_attention_aotriton_varlen_gqa_bf16(
            scratch,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            rows=rows,
            segments=segments,
            kv_rows=kv_rows,
            query_bf16=query_bf16,
            key_cache=key_cache,
            value_cache=value_cache,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            scale=scale,
            library=library,
            stream=stream,
        )
        aotriton_gate_mul_bf16_to_fp16(
            attn_bf16.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            rows * self.config.num_attention_heads * self.config.head_dim,
            stream=stream,
            library=_library_for(library, "aotriton"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def append_full_attention_kv_fp16_batch(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        rows: int,
        block_size: int = 256,
        library=None,
        stream: int = 0,
    ) -> None:
        """Append prompt K/V rows into one request's paged KV cache."""

        self._append_full_attention_kv_resolved(
            scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=spans,
            kind=PagedKVWriteKind.PROMPT,
            rows=rows,
            block_size=block_size,
            library=library,
            stream=stream,
        )

    def append_full_attention_kv_fp16_decode_batch(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        rows: int,
        block_size: int = 256,
        library=None,
        stream: int = 0,
    ) -> None:
        """Append one decode K/V row per active request into row-major KV slots."""

        self._append_full_attention_kv_resolved(
            scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=spans,
            kind=PagedKVWriteKind.BATCH,
            rows=rows,
            block_size=block_size,
            library=library,
            stream=stream,
        )

    def append_full_attention_kv_int8_per_token_head_fp16_batch(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        rows: int,
        block_size: int = 256,
        library=None,
        stream: int = 0,
    ) -> None:
        """Append FP16-prefill K/V rows into an INT8 retained KV cache."""

        if spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD:
            raise ValueError("INT8 retained prefill append requires int8_per_token_head spans")
        self._append_full_attention_kv_resolved(
            scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=spans,
            kind=PagedKVWriteKind.PROMPT,
            rows=rows,
            block_size=block_size,
            library=library,
            stream=stream,
        )

    def decode_full_attention_context_gate_fp16_batch(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        rows: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Decode BF16 paged attention for one active token per row.

        This batch path intentionally covers the retained 512/128 bring-up
        protocol where context remains below the split-K threshold. Longer
        contexts still need a row-aware split-K reducer before they can be
        labelled native c-aware decode.
        """

        if rows <= 0:
            raise ValueError("rows must be positive")
        if spans.storage_dtype != DType.BF16:
            raise NotImplementedError("native batch context decode currently requires BF16 KV")
        if spans.max_live_count >= 1024:
            raise NotImplementedError("native batch split-K full-attention decode is not wired")
        gate_tensor = scratch.gate if gate is None else gate
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            scratch.query_raw.ptr,
            spans,
            rows,
            spans.max_live_count,
            block_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        qwen35_full_attn_gate_mul_fp16(
            scratch.query_raw.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            rows * self.config.num_attention_heads * self.config.head_dim,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def decode_full_attention_context_gate_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        gate_tensor = scratch.gate if gate is None else gate
        if spans.max_live_count < 1024:
            qwen35_full_attn_decode_context_bf16(
                scratch.query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                scratch.attn_out.ptr,
                spans.live_counts.ptr,
                spans.max_live_count,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                (self.config.head_dim ** -0.5) if scale is None else scale,
                stream=stream,
                library=_library_for(library, "attention"),
                runtime=self.runtime,
            )
        else:
            qwen35_paged_full_attn_decode_context_bf16_spans(
                scratch.query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                scratch.attn_out.ptr,
                spans,
                spans.max_live_count,
                block_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                (self.config.head_dim ** -0.5) if scale is None else scale,
                stream=stream,
                library=_library_for(library, "attention"),
                runtime=self.runtime,
            )
        qwen35_full_attn_gate_mul_fp16(
            scratch.attn_out.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            self.config.num_attention_heads * self.config.head_dim,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def decode_full_attention_gqa_gate_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        chunk_size: int,
        num_splits: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            return self._decode_full_attention_int8_gqa_gate(
                scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=spans,
                chunk_size=chunk_size,
                num_splits=num_splits,
                kind=PagedAttnDecodeKind.GQA_SPLITK_GATE_FP16,
                gate=gate,
                block_size=block_size,
                scale=scale,
                library=library,
                stream=stream,
            )
        gate_tensor = scratch.gate if gate is None else gate
        split_kernel = (
            qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans
            if (
                block_size == 256
                and self.config.num_attention_heads == 16
                and self.config.num_key_value_heads == 2
                and self.config.head_dim == 256
            )
            else qwen35_paged_full_attn_decode_split_k_gate_fp16_spans
        )
        split_kernel(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            scratch.partial_out.ptr,
            scratch.partial_m.ptr,
            scratch.partial_l.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            gate_tensor.shape[-1],
            1,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def decode_full_attention_split_gate_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        chunk_size: int,
        num_splits: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            return self._decode_full_attention_int8_gqa_gate(
                scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=spans,
                chunk_size=chunk_size,
                num_splits=num_splits,
                kind=PagedAttnDecodeKind.GQA_SPLITK_GATE_FP16,
                gate=gate,
                block_size=block_size,
                scale=scale,
                library=library,
                stream=stream,
            )
        gate_tensor = scratch.gate if gate is None else gate
        decode_fn = _full_attention_split_gate_fp16_fn(
            self.config,
            block_size=block_size,
            num_splits=num_splits,
            max_live_count=spans.max_live_count,
        )
        decode_fn(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            scratch.partial_out.ptr,
            scratch.partial_m.ptr,
            scratch.partial_l.ptr,
            spans,
            chunk_size,
            num_splits,
            block_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            gate_tensor.shape[-1],
            1,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            stream=stream,
            library=_library_for(library, "attention"),
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def project_full_attention_o_fp16(
        self,
        gated_attn: Tensor,
        scratch: Qwen35ParoAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        force_pack8_gemv: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn.o_proj"
        q_width = self.config.num_attention_heads * self.config.head_dim
        pairs = self.tensor(f"{prefix}.pairs")
        paro_rotate1_fp16(
            gated_attn.ptr,
            scratch.o_rot.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.theta").ptr,
            self.tensor(f"{prefix}.channel_scales").ptr,
            tokens,
            q_width,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        self.project_pack8_fp16(
            scratch.o_rot,
            scratch.o_proj,
            weight_prefix=prefix,
            rows=tokens,
            in_features=q_width,
            group_size=group_size,
            threads=64 if tokens > 1 else 128,
            force_gemv=force_pack8_gemv,
            library=library,
            stream=stream,
        )
        return scratch.o_proj

    def project_full_attention_o_rows_fp16(
        self,
        gated_attn: Tensor,
        scratch: Qwen35ParoAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Replay full-attention O projection with token-1 kernels per row."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        for row in range(tokens):
            self.project_full_attention_o_fp16(
                self._row_tensor_view(gated_attn, row),
                self._decode_row_full_attention_scratch(scratch, row),
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return scratch.o_proj

    def project_full_attention_o_bf16_attn_gate_fp16(
        self,
        attn_bf16: Tensor,
        gate: Tensor,
        scratch: Qwen35ParoAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Fuse BF16 attention gating with PARO rotate1 before the FP16 O projection."""

        if attn_bf16.dtype is not DType.BF16:
            raise ValueError("fused full-attention O projection expects BF16 attention output")
        if gate.dtype is not DType.FP16:
            raise ValueError("fused full-attention O projection expects FP16 gate tensor")
        if scratch.o_rot.dtype is not DType.FP16 or scratch.o_proj.dtype is not DType.FP16:
            raise ValueError("fused full-attention O projection expects FP16 output scratch")
        if attn_bf16.shape != gate.shape or attn_bf16.numel != scratch.o_rot.numel:
            raise ValueError("attention output, gate, and O-rotation scratch sizes must match")
        prefix = f"layers.{self.layer_weights.layer_id}.self_attn.o_proj"
        q_width = self.config.num_attention_heads * self.config.head_dim
        pairs = self.tensor(f"{prefix}.pairs")
        paro_rotate1_bf16_gate_fp16(
            attn_bf16.ptr,
            gate.ptr,
            scratch.o_rot.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.theta").ptr,
            self.tensor(f"{prefix}.channel_scales").ptr,
            tokens,
            q_width,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        self.project_pack8_fp16(
            scratch.o_rot,
            scratch.o_proj,
            weight_prefix=prefix,
            rows=tokens,
            in_features=q_width,
            group_size=group_size,
            threads=64 if tokens > 1 else 128,
            library=library,
            stream=stream,
        )
        return scratch.o_proj

    def run_full_attention_moe_c1_layer_fp16(
        self,
        hidden: Tensor,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        append_spans: KVLiveSpans,
        decode_spans: KVLiveSpans,
        cos_table: Tensor,
        sin_table: Tensor,
        position: Tensor,
        max_positions: int,
        attention_scratch: Qwen35ParoAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoDenseMlpScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        block_size: int = 256,
        chunk_size: int = 256,
        num_splits: int = 1,
        post_input_rmsnorm_trace: Callable[[Qwen35ParoAttentionScratch], None] | None = None,
        input_scratch_trace: Callable[[str, Qwen35ParoAttentionScratch], None] | None = None,
        qkv_tensor_trace: Callable[[str, Tensor], None] | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if tokens != 1:
            raise ValueError("full-attention+MoE c=1 layer orchestrator currently requires tokens=1")
        attention_scratch = attention_scratch or self.reserve_full_attention_scratch(
            tokens=tokens,
            num_splits=num_splits,
            activation_dtype=DType.FP16,
            gated_dtype=DType.FP16,
        )
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        if dense_mlp:
            if not isinstance(moe_scratch, Qwen35ParoDenseMlpScratch):
                moe_scratch = self.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        else:
            moe_scratch = moe_scratch or self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.input_rmsnorm_fp16(hidden, attention_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        if post_input_rmsnorm_trace is not None:
            post_input_rmsnorm_trace(attention_scratch)
        self.rotate_full_attention_inputs_fp16(
            attention_scratch.attn_input,
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        if input_scratch_trace is not None:
            input_scratch_trace("attn_input_after_rotate", attention_scratch)
        self.project_full_attention_qkv_fp16(
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            producer_trace=qkv_tensor_trace,
            library=library,
            stream=stream,
        )
        if input_scratch_trace is not None:
            input_scratch_trace("attn_input_after_project", attention_scratch)
        _query, _key, _value, gate = self.prepare_full_attention_qkv_fp16(
            attention_scratch,
            cos_table=cos_table,
            sin_table=sin_table,
            position=position,
            max_positions=max_positions,
            tokens=tokens,
            producer_trace=qkv_tensor_trace,
            library=library,
            stream=stream,
        )
        if input_scratch_trace is not None:
            input_scratch_trace("attn_input_after_prepare", attention_scratch)
        self.append_full_attention_kv_fp16(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=append_spans,
            block_size=block_size,
            library=library,
            stream=stream,
        )
        if not _requires_full_attention_split_decode(decode_spans):
            gated = self.decode_full_attention_context_gate_fp16(
                attention_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=decode_spans,
                gate=gate,
                block_size=block_size,
                library=library,
                stream=stream,
            )
        else:
            gated = self.decode_full_attention_split_gate_fp16(
                attention_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=decode_spans,
                chunk_size=chunk_size,
                num_splits=num_splits,
                gate=gate,
                block_size=block_size,
                library=library,
                stream=stream,
            )
        attn_out = self.project_full_attention_o_fp16(
            gated,
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        mlp_input, residual = self.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        if dense_mlp:
            return self.run_dense_mlp_residual_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_c1_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_full_attention_moe_decode_batch_layer_fp16(
        self,
        hidden: Tensor,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        append_spans: KVLiveSpans,
        decode_spans: KVLiveSpans,
        cos_table: Tensor,
        sin_table: Tensor,
        positions: Tensor,
        max_positions: int,
        attention_scratch: Qwen35ParoAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch | None = None,
        tokens: int,
        group_size: int = 128,
        block_size: int = 256,
        force_selected_c1_moe: bool = False,
        force_per_row_input_rmsnorm: bool = False,
        force_per_row_qkv_scratch: bool = False,
        force_per_row_layer_scratch: bool = False,
        force_per_row_context: bool = False,
        per_row_contexts: Sequence[tuple[Tensor, Tensor, KVLiveSpans]] | None = None,
        force_per_row_kv_append: bool = False,
        per_row_append_contexts: Sequence[tuple[Tensor, Tensor, KVLiveSpans]] | None = None,
        force_per_row_append_context: bool = False,
        force_per_row_suffix: bool = False,
        force_per_row_output: bool = False,
        force_batch_gemv_output: bool = False,
        force_per_row_post_attention: bool = False,
        force_per_row_moe: bool = False,
        post_input_rmsnorm_trace: Callable[[Qwen35ParoAttentionScratch], None] | None = None,
        input_scratch_trace: Callable[[str, int, Qwen35ParoAttentionScratch], None] | None = None,
        qkv_tensor_trace: Callable[[str, int, Tensor], None] | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run one native full-attention decode token for each active batch row."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        attention_scratch = attention_scratch or self.reserve_full_attention_scratch(
            tokens=tokens,
            num_splits=1,
            activation_dtype=DType.FP16,
            gated_dtype=DType.FP16,
        )
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        if dense_mlp:
            if not isinstance(moe_scratch, Qwen35ParoDenseMlpScratch):
                moe_scratch = self.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif tokens > 1 and not (force_selected_c1_moe or force_per_row_moe or force_per_row_layer_scratch):
            if not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
                moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif not isinstance(moe_scratch, Qwen35ParoMoeScratch):
            moe_scratch = self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        input_rmsnorm_fn = (
            self.input_rmsnorm_fp16_per_row
            if force_per_row_input_rmsnorm and tokens > 1
            else self.input_rmsnorm_fp16
        )
        input_rmsnorm_fn(
            hidden,
            attention_scratch.attn_input,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        if post_input_rmsnorm_trace is not None:
            post_input_rmsnorm_trace(attention_scratch)
        if force_per_row_layer_scratch and tokens > 1:
            if per_row_append_contexts is None or len(per_row_append_contexts) != tokens:
                raise ValueError("per_row_append_contexts must provide one key/value/span tuple per decode row")
            if per_row_contexts is None or len(per_row_contexts) != tokens:
                raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
            if dense_mlp:
                raise NotImplementedError("per-row full-attention scratch diagnostic is currently wired for MoE layers")
            if not isinstance(moe_scratch, Qwen35ParoMoeScratch):
                raise ValueError("per-row full-attention scratch diagnostic requires token-row MoE scratch")
            row_moe_scratch = self.reserve_moe_c1_scratch(
                tokens=1,
                activation_dtype=hidden.dtype,
                prefix="moe.decode_row_layer_scratch",
            )
            runtime = self.runtime or get_hip_runtime()
            for row, ((row_key_cache, row_value_cache, row_append_spans), row_context_tuple) in enumerate(
                zip(per_row_append_contexts, per_row_contexts, strict=True)
            ):
                context_key_cache, context_value_cache, row_decode_spans = row_context_tuple
                if context_key_cache.ptr != row_key_cache.ptr or context_value_cache.ptr != row_value_cache.ptr:
                    raise ValueError("per-row full-attention scratch diagnostics must use matching row cache views")
                row_hidden = self._row_tensor_view(hidden, row)
                row_scratch = self._decode_row_full_attention_temp_scratch(attention_scratch)
                row_position = Tensor.from_handle(
                    positions.ptr + row * DType.INT64.itemsize,
                    (1,),
                    DType.INT64,
                    positions.device,
                )

                def row_input_scratch_trace(stage: str, scratch: Qwen35ParoAttentionScratch, *, _row: int = row) -> None:
                    if input_scratch_trace is not None:
                        input_scratch_trace(stage, _row, scratch)

                def row_qkv_tensor_trace(stage: str, tensor: Tensor, *, _row: int = row) -> None:
                    if qkv_tensor_trace is not None:
                        qkv_tensor_trace(stage, _row, tensor)

                row_out = self.run_full_attention_moe_c1_layer_fp16(
                    row_hidden,
                    key_cache=row_key_cache,
                    value_cache=row_value_cache,
                    append_spans=row_append_spans,
                    decode_spans=row_decode_spans,
                    cos_table=cos_table,
                    sin_table=sin_table,
                    position=row_position,
                    max_positions=max_positions,
                    attention_scratch=row_scratch,
                    moe_scratch=row_moe_scratch,
                    tokens=1,
                    group_size=group_size,
                    block_size=block_size,
                    input_scratch_trace=row_input_scratch_trace if input_scratch_trace is not None else None,
                    qkv_tensor_trace=row_qkv_tensor_trace if qkv_tensor_trace is not None else None,
                    library=library,
                    stream=stream,
                )
                dst = self._row_tensor_view(moe_scratch.moe_out, row)
                runtime.memcpy_async(
                    dst.ptr,
                    row_out.ptr,
                    row_out.numel * row_out.dtype.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            return moe_scratch.moe_out
        _query, _key, _value, gate = self.prepare_full_attention_qkv_fp16_decode_rows(
            attention_scratch,
            cos_table=cos_table,
            sin_table=sin_table,
            positions=positions,
            max_positions=max_positions,
            tokens=tokens,
            group_size=group_size,
            input_scratch_trace=input_scratch_trace,
            qkv_tensor_trace=qkv_tensor_trace,
            force_per_row_scratch=force_per_row_qkv_scratch,
            library=library,
            stream=stream,
        )
        if force_per_row_suffix and tokens > 1:
            if not (
                force_per_row_kv_append
                and force_per_row_context
                and force_per_row_output
                and force_per_row_post_attention
                and force_per_row_moe
            ):
                raise ValueError("per-row suffix interleave requires per-row KV append, context, output, post-attention, and MoE diagnostics")
            if per_row_append_contexts is None or len(per_row_append_contexts) != tokens:
                raise ValueError("per_row_append_contexts must provide one key/value/span tuple per decode row")
            if per_row_contexts is None or len(per_row_contexts) != tokens:
                raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
            if dense_mlp:
                raise NotImplementedError("per-row suffix diagnostic is currently wired for MoE layers")
            if not isinstance(moe_scratch, Qwen35ParoMoeScratch):
                raise ValueError("per-row suffix diagnostic requires token-row MoE scratch")
            row_moe_scratch = self.reserve_moe_c1_scratch(
                tokens=1,
                activation_dtype=hidden.dtype,
                prefix="moe.decode_row_suffix",
            )
            runtime = self.runtime or get_hip_runtime()
            for row, ((row_key_cache, row_value_cache, row_append_spans), row_context_tuple) in enumerate(
                zip(per_row_append_contexts, per_row_contexts, strict=True)
            ):
                context_key_cache, context_value_cache, row_decode_spans = row_context_tuple
                if context_key_cache.ptr != row_key_cache.ptr or context_value_cache.ptr != row_value_cache.ptr:
                    raise ValueError("per-row suffix diagnostics must use matching row cache views")
                row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
                row_hidden = self._row_tensor_view(hidden, row)
                self.append_full_attention_kv_fp16(
                    row_scratch,
                    key_cache=row_key_cache,
                    value_cache=row_value_cache,
                    spans=row_append_spans,
                    block_size=block_size,
                    library=library,
                    stream=stream,
                )
                row_gate = self._row_tensor_view(gate, row)
                self.decode_full_attention_context_gate_fp16(
                    row_scratch,
                    key_cache=row_key_cache,
                    value_cache=row_value_cache,
                    spans=row_decode_spans,
                    gate=row_gate,
                    block_size=block_size,
                    library=library,
                    stream=stream,
                )
                row_attn_out = self.project_full_attention_o_fp16(
                    self._row_tensor_view(attention_scratch.gated_attn, row),
                    row_scratch,
                    tokens=1,
                    group_size=group_size,
                    library=library,
                    stream=stream,
                )
                row_mlp_input, row_residual = self.post_attention_add_rmsnorm_fp16(
                    row_hidden,
                    row_attn_out,
                    row_moe_scratch,
                    tokens=1,
                    library=library,
                    stream=stream,
                )
                row_out = self.run_moe_c1_fp16(
                    row_mlp_input,
                    row_residual,
                    scratch=row_moe_scratch,
                    tokens=1,
                    group_size=group_size,
                    library=library,
                    stream=stream,
                )
                dst = self._row_tensor_view(moe_scratch.moe_out, row)
                runtime.memcpy_async(
                    dst.ptr,
                    row_out.ptr,
                    row_out.numel * row_out.dtype.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            return moe_scratch.moe_out
        if force_per_row_append_context and tokens > 1 and not (force_per_row_kv_append and force_per_row_context):
            raise ValueError("per-row append+context interleave requires per-row KV append and context diagnostics")
        if force_per_row_append_context and tokens > 1:
            if per_row_append_contexts is None or len(per_row_append_contexts) != tokens:
                raise ValueError("per_row_append_contexts must provide one key/value/span tuple per decode row")
            if per_row_contexts is None or len(per_row_contexts) != tokens:
                raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
            for row, ((row_key_cache, row_value_cache, row_append_spans), row_context_tuple) in enumerate(
                zip(per_row_append_contexts, per_row_contexts, strict=True)
            ):
                context_key_cache, context_value_cache, row_decode_spans = row_context_tuple
                if context_key_cache.ptr != row_key_cache.ptr or context_value_cache.ptr != row_value_cache.ptr:
                    raise ValueError("per-row append/context diagnostics must use matching row cache views")
                row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
                self.append_full_attention_kv_fp16(
                    row_scratch,
                    key_cache=row_key_cache,
                    value_cache=row_value_cache,
                    spans=row_append_spans,
                    block_size=block_size,
                    library=library,
                    stream=stream,
                )
                row_gate = self._row_tensor_view(gate, row)
                self.decode_full_attention_context_gate_fp16(
                    row_scratch,
                    key_cache=row_key_cache,
                    value_cache=row_value_cache,
                    spans=row_decode_spans,
                    gate=row_gate,
                    block_size=block_size,
                    library=library,
                    stream=stream,
                )
                row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                self.runtime.memcpy_async(
                    row_context.ptr,
                    attention_scratch.attn_out.ptr,
                    self.config.num_attention_heads * self.config.head_dim * DType.FP32.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            gated = attention_scratch.gated_attn
        else:
            if force_per_row_kv_append and tokens > 1:
                if per_row_append_contexts is None or len(per_row_append_contexts) != tokens:
                    raise ValueError("per_row_append_contexts must provide one key/value/span tuple per decode row")
                for row, (row_key_cache, row_value_cache, row_append_spans) in enumerate(per_row_append_contexts):
                    row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
                    self.append_full_attention_kv_fp16(
                        row_scratch,
                        key_cache=row_key_cache,
                        value_cache=row_value_cache,
                        spans=row_append_spans,
                        block_size=block_size,
                        library=library,
                        stream=stream,
                    )
            else:
                self.append_full_attention_kv_fp16_decode_batch(
                    attention_scratch,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    spans=append_spans,
                    rows=tokens,
                    block_size=block_size,
                    library=library,
                    stream=stream,
                )
            if force_per_row_context and tokens > 1:
                if per_row_contexts is None or len(per_row_contexts) != tokens:
                    raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
                for row, (row_key_cache, row_value_cache, row_decode_spans) in enumerate(per_row_contexts):
                    row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
                    row_gate = self._row_tensor_view(gate, row)
                    self.decode_full_attention_context_gate_fp16(
                        row_scratch,
                        key_cache=row_key_cache,
                        value_cache=row_value_cache,
                        spans=row_decode_spans,
                        gate=row_gate,
                        block_size=block_size,
                        library=library,
                        stream=stream,
                    )
                    row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                    self.runtime.memcpy_async(
                        row_context.ptr,
                        attention_scratch.attn_out.ptr,
                        self.config.num_attention_heads * self.config.head_dim * DType.FP32.itemsize,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                gated = attention_scratch.gated_attn
            else:
                gated = self.decode_full_attention_context_gate_fp16_batch(
                    attention_scratch,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    spans=decode_spans,
                    rows=tokens,
                    gate=gate,
                    block_size=block_size,
                    library=library,
                    stream=stream,
                )
        if force_per_row_output and tokens > 1:
            attn_out = self.project_full_attention_o_rows_fp16(
                gated,
                attention_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        else:
            attn_out = self.project_full_attention_o_fp16(
                gated,
                attention_scratch,
                tokens=tokens,
                group_size=group_size,
                force_pack8_gemv=force_batch_gemv_output,
                library=library,
                stream=stream,
            )
        post_attention_fn = (
            self.post_attention_add_rmsnorm_fp16_per_row
            if force_per_row_post_attention and tokens > 1
            else self.post_attention_add_rmsnorm_fp16
        )
        mlp_input, residual = post_attention_fn(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        if dense_mlp:
            return self.run_dense_mlp_residual_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        if force_per_row_moe and tokens > 1:
            return self.run_moe_c1_rows_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        if tokens > 1 and not force_selected_c1_moe:
            return self.run_moe_grouped_compact_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_c1_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_full_attention_moe_prefill_layer_fp16(
        self,
        hidden: Tensor,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        append_spans: KVLiveSpans,
        prefill_spans: KVLiveSpans,
        cos_table: Tensor,
        sin_table: Tensor,
        positions: Tensor,
        max_positions: int,
        attention_scratch: Qwen35ParoAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch | None = None,
        cu_seqlens_q: Tensor | None = None,
        cu_seqlens_k: Tensor | None = None,
        aotriton_attention: bool = False,
        aotriton_kv_rows: int | None = None,
        retained_key_cache: Tensor | None = None,
        retained_value_cache: Tensor | None = None,
        retained_append_spans: KVLiveSpans | None = None,
        tokens: int,
        group_size: int = 128,
        block_size: int = 256,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run native multi-token full-attention prefill plus grouped MoE.

        This is the single-request bulk prefill counterpart to
        :meth:`run_full_attention_moe_c1_layer_fp16`: all prompt K/V rows are
        appended first, the causal GQA prefill kernel attends each query row up
        to its row position, and the post-attention MoE uses the grouped compact
        route rather than selected c=1 rows.
        """

        if tokens <= 1:
            raise ValueError("full-attention native prefill requires tokens > 1")
        retained_int8 = retained_append_spans is not None
        if retained_int8:
            if retained_key_cache is None or retained_value_cache is None:
                raise ValueError("INT8 retained prefill append requires retained key/value cache tensors")
            if retained_append_spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD:
                raise ValueError("INT8 retained prefill append requires int8_per_token_head spans")
        attention_scratch = attention_scratch or self.reserve_full_attention_scratch(
            tokens=tokens,
            num_splits=1,
            activation_dtype=DType.FP16,
            gated_dtype=DType.FP16,
        )
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        use_grouped_moe = False if dense_mlp else _use_moe_grouped_compact_prefill(tokens)
        if dense_mlp:
            if not isinstance(moe_scratch, Qwen35ParoDenseMlpScratch):
                moe_scratch = self.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif use_grouped_moe:
            if not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
                moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif not isinstance(moe_scratch, Qwen35ParoMoeScratch):
            moe_scratch = self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.input_rmsnorm_fp16(hidden, attention_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        self.rotate_full_attention_inputs_fp16(
            attention_scratch.attn_input,
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        self.project_full_attention_qkv_fp16(
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        aotriton_query_bf16 = None
        if aotriton_attention:
            # Reuse the caller-owned prefill query buffer for AOTriton's BF16
            # query input. Allocating this in each layer state's decode
            # workspace makes long INT8 prefill accumulate one [chunk, Hq, D]
            # BF16 buffer per full-attention layer even though only the current
            # chunk needs it.
            aotriton_query_bf16 = Tensor.from_handle(
                attention_scratch.query.ptr,
                attention_scratch.query.shape,
                DType.BF16,
                attention_scratch.query.device,
            )
        _query, _key, _value, gate = self.prepare_full_attention_qkv_fp16(
            attention_scratch,
            cos_table=cos_table,
            sin_table=sin_table,
            position=positions,
            max_positions=max_positions,
            tokens=tokens,
            query_bf16_out=aotriton_query_bf16,
            library=library,
            stream=stream,
        )
        self.append_full_attention_kv_fp16_batch(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=append_spans,
            rows=tokens,
            block_size=block_size,
            library=library,
            stream=stream,
        )
        if retained_int8:
            self.append_full_attention_kv_int8_per_token_head_fp16_batch(
                attention_scratch,
                key_cache=retained_key_cache,
                value_cache=retained_value_cache,
                spans=retained_append_spans,
                rows=tokens,
                block_size=block_size,
                library=library,
                stream=stream,
            )
        if aotriton_attention:
            if cu_seqlens_q is None or cu_seqlens_k is None:
                raise ValueError("AOTriton prefill requires cu_seqlens_q/k tensors")
            # AOTriton returns BF16 and the fused gate+rotate path does not need
            # the old FP16 gated-attention scratch. Reinterpret those bytes as
            # BF16 output to avoid another full-width intermediate allocation.
            aotriton_attn_bf16_out = Tensor.from_handle(
                attention_scratch.gated_attn.ptr,
                attention_scratch.query.shape,
                DType.BF16,
                attention_scratch.gated_attn.device,
            )
            attn_bf16 = self.prefill_full_attention_aotriton_varlen_gqa_bf16(
                attention_scratch,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                rows=tokens,
                segments=1,
                kv_rows=aotriton_kv_rows,
                query_bf16=aotriton_query_bf16,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_bf16_out=aotriton_attn_bf16_out,
                library=library,
                stream=stream,
            )
            attn_out = self.project_full_attention_o_bf16_attn_gate_fp16(
                attn_bf16,
                gate,
                attention_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        else:
            gated = self.prefill_full_attention_gqa_gate_fp16(
                attention_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=prefill_spans,
                rows=tokens,
                gate=gate,
                block_size=block_size,
                library=library,
                stream=stream,
            )
            attn_out = self.project_full_attention_o_fp16(
                gated,
                attention_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        mlp_input, residual = self.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        if dense_mlp:
            return self.run_dense_mlp_residual_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        if use_grouped_moe:
            return self.run_moe_grouped_compact_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_c1_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_full_attention_moe_prefill_varlen_layer_fp16(
        self,
        hidden: Tensor,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        append_spans: KVLiveSpans,
        prefill_spans: KVLiveSpans,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        segments: int,
        cos_table: Tensor,
        sin_table: Tensor,
        positions: Tensor,
        max_positions: int,
        attention_scratch: Qwen35ParoAttentionScratch | None = None,
        moe_scratch: Qwen35ParoGroupedMoeScratch | None = None,
        tokens: int,
        group_size: int = 128,
        block_size: int = 256,
        aotriton_attention: bool = False,
        aotriton_max_seqlen_q: int | None = None,
        aotriton_max_seqlen_k: int | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if segments <= 0:
            raise ValueError("segments must be positive")
        attention_scratch = attention_scratch or self.reserve_full_attention_scratch(
            tokens=tokens,
            num_splits=1,
            activation_dtype=DType.FP16,
            gated_dtype=DType.FP16,
            query_dtype=DType.BF16 if aotriton_attention else DType.FP32,
        )
        if not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
            moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.input_rmsnorm_fp16(hidden, attention_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        self.rotate_full_attention_inputs_fp16(
            attention_scratch.attn_input,
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        self.project_full_attention_qkv_fp16(
            attention_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        aotriton_query_bf16 = None
        if aotriton_attention:
            aotriton_query_bf16 = Tensor.from_handle(
                attention_scratch.query.ptr,
                attention_scratch.query.shape,
                DType.BF16,
                attention_scratch.query.device,
            )
        _query, _key, _value, gate = self.prepare_full_attention_qkv_fp16(
            attention_scratch,
            cos_table=cos_table,
            sin_table=sin_table,
            position=positions,
            max_positions=max_positions,
            tokens=tokens,
            query_bf16_out=aotriton_query_bf16,
            library=library,
            stream=stream,
        )
        self.append_full_attention_kv_fp16_batch(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=append_spans,
            rows=tokens,
            block_size=block_size,
            library=library,
            stream=stream,
        )
        if aotriton_attention:
            attn_bf16 = self.prefill_full_attention_aotriton_varlen_gqa_bf16(
                attention_scratch,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                rows=tokens,
                segments=segments,
                query_bf16=aotriton_query_bf16,
                max_seqlen_q=aotriton_max_seqlen_q,
                max_seqlen_k=aotriton_max_seqlen_k,
                library=library,
                stream=stream,
            )
            attn_out = self.project_full_attention_o_bf16_attn_gate_fp16(
                attn_bf16,
                gate,
                attention_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        else:
            gated = self.prefill_full_attention_varlen_gqa_gate_fp16(
                attention_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=prefill_spans,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                rows=tokens,
                segments=segments,
                gate=gate,
                block_size=block_size,
                library=library,
                stream=stream,
            )
            attn_out = self.project_full_attention_o_fp16(
                gated,
                attention_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        mlp_input, residual = self.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        return self.run_moe_grouped_compact_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def rotate_linear_attention_inputs_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv = f"{prefix}.in_proj_qkv"
        z = f"{prefix}.in_proj_z"
        pairs_qkv = self.tensor(f"{qkv}.pairs")
        pairs_z = self.tensor(f"{z}.pairs")
        if tokens == 1 and hidden.ptr == scratch.attn_input.ptr and _rotate_dual_pack8_fused_enabled():
            self._rotate_fuse_ready.add(scratch.rotate_fuse_barrier.ptr)
            return scratch.qkv_rot, scratch.z_rot
        self._rotate_fuse_ready.discard(scratch.rotate_fuse_barrier.ptr)
        paro_rotate2_fp16(
            hidden.ptr,
            scratch.qkv_rot.ptr,
            scratch.z_rot.ptr,
            pairs_qkv.ptr,
            pairs_z.ptr,
            self.tensor(f"{qkv}.theta").ptr,
            self.tensor(f"{z}.theta").ptr,
            self.tensor(f"{qkv}.channel_scales").ptr,
            self.tensor(f"{z}.channel_scales").ptr,
            tokens,
            self.config.hidden_size,
            group_size,
            _rotation_krot(pairs_qkv),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        return scratch.qkv_rot, scratch.z_rot

    def project_linear_attention_qkv_z_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        force_gemv: bool = False,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv = f"{prefix}.in_proj_qkv"
        z = f"{prefix}.in_proj_z"
        qkv_qweight = self.tensor(f"{qkv}.qweight_pack8_decode")
        z_qweight = self.tensor(f"{z}.qweight_pack8_decode")
        qkv_out_packed = _out_packed_from_generic_transposed_qweight(qkv_qweight)
        z_out_packed = _out_packed_from_generic_transposed_qweight(z_qweight)
        if tokens == 1:
            awq_library = _library_for(library, "awq")
            use_rotate_fused = scratch.rotate_fuse_barrier.ptr in self._rotate_fuse_ready
            if use_rotate_fused:
                gemv_awq_dual_pack8_transposed_rotate_staged_fp16(
                    scratch.attn_input.ptr,
                    scratch.qkv_rot.ptr,
                    scratch.z_rot.ptr,
                    self.tensor(f"{qkv}.pairs").ptr,
                    self.tensor(f"{z}.pairs").ptr,
                    self.tensor(f"{qkv}.theta").ptr,
                    self.tensor(f"{z}.theta").ptr,
                    self.tensor(f"{qkv}.channel_scales").ptr,
                    self.tensor(f"{z}.channel_scales").ptr,
                    qkv_qweight.ptr,
                    self.tensor(f"{qkv}.qzeros").ptr,
                    self.tensor(f"{qkv}.scales").ptr,
                    z_qweight.ptr,
                    self.tensor(f"{z}.qzeros").ptr,
                    self.tensor(f"{z}.scales").ptr,
                    scratch.qkv_z.ptr,
                    scratch.rotate_fuse_barrier.ptr,
                    tokens,
                    scratch.qkv_rot.shape[-1],
                    qkv_out_packed,
                    z_out_packed,
                    group_size,
                    _rotation_krot(self.tensor(f"{qkv}.pairs")),
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
                self._rotate_fuse_ready.discard(scratch.rotate_fuse_barrier.ptr)
            else:
                gemv_awq_dual_pack8_transposed_fp16(
                    scratch.qkv_rot.ptr,
                    scratch.z_rot.ptr,
                    qkv_qweight.ptr,
                    self.tensor(f"{qkv}.qzeros").ptr,
                    self.tensor(f"{qkv}.scales").ptr,
                    z_qweight.ptr,
                    self.tensor(f"{z}.qzeros").ptr,
                    self.tensor(f"{z}.scales").ptr,
                    scratch.qkv_z.ptr,
                    tokens,
                    scratch.qkv_rot.shape[-1],
                    qkv_out_packed,
                    z_out_packed,
                    group_size,
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
        elif force_gemv:
            awq_library = _library_for(library, "awq")
            gemv_awq_pack8_transposed_fp16(
                scratch.qkv_rot.ptr,
                qkv_qweight.ptr,
                self.tensor(f"{qkv}.qzeros").ptr,
                self.tensor(f"{qkv}.scales").ptr,
                scratch.qkv.ptr,
                tokens,
                scratch.qkv_rot.shape[-1],
                qkv_out_packed,
                group_size,
                threads=64,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
            gemv_awq_pack8_transposed_fp16(
                scratch.z_rot.ptr,
                z_qweight.ptr,
                self.tensor(f"{z}.qzeros").ptr,
                self.tensor(f"{z}.scales").ptr,
                scratch.z.ptr,
                tokens,
                scratch.z_rot.shape[-1],
                z_out_packed,
                group_size,
                threads=64,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
        else:
            awq_fusedw4_prefill_dual_fp16(
                scratch.qkv_rot.ptr,
                scratch.z_rot.ptr,
                qkv_qweight.ptr,
                self.tensor(f"{qkv}.qzeros").ptr,
                self.tensor(f"{qkv}.scales").ptr,
                z_qweight.ptr,
                self.tensor(f"{z}.qzeros").ptr,
                self.tensor(f"{z}.scales").ptr,
                scratch.qkv.ptr,
                scratch.z.ptr,
                tokens,
                scratch.qkv_rot.shape[-1],
                qkv_out_packed,
                z_out_packed,
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        return scratch.qkv, scratch.z

    def project_linear_attention_ab_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int = 1,
        threads: int = 64,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        a_weight = self.tensor(f"{prefix}.in_proj_a.weight")
        b_weight = self.tensor(f"{prefix}.in_proj_b.weight")
        dense_library = _library_for(library, "dense")
        if tokens == 1:
            dense_dual_gemv_out_fp16(
                hidden.ptr,
                a_weight.ptr,
                b_weight.ptr,
                scratch.ab.ptr,
                tokens,
                self.config.hidden_size,
                self.config.linear_num_value_heads,
                self.config.linear_num_value_heads,
                threads=threads,
                stream=stream,
                library=dense_library,
                runtime=self.runtime,
            )
        else:
            # The dual GEMV writes row-major [a,b] per token.  Native prefill
            # GDN consumes contiguous [tokens,a] and [tokens,b] streams.
            if _use_linear_ab_prefill_rocblas(tokens):
                rocblas_gemm_ex_rowmajor_nt_fp16_compute_f32(
                    hidden.ptr,
                    a_weight.ptr,
                    scratch.a.ptr,
                    rows=tokens,
                    in_features=self.config.hidden_size,
                    out_features=self.config.linear_num_value_heads,
                    stream=stream,
                )
                rocblas_gemm_ex_rowmajor_nt_fp16_compute_f32(
                    hidden.ptr,
                    b_weight.ptr,
                    scratch.b.ptr,
                    rows=tokens,
                    in_features=self.config.hidden_size,
                    out_features=self.config.linear_num_value_heads,
                    stream=stream,
                )
            else:
                dense_gemv_out_fp16(
                    hidden.ptr,
                    a_weight.ptr,
                    scratch.a.ptr,
                    tokens,
                    self.config.hidden_size,
                    self.config.linear_num_value_heads,
                    threads=threads,
                    stream=stream,
                    library=dense_library,
                    runtime=self.runtime,
                )
                dense_gemv_out_fp16(
                    hidden.ptr,
                    b_weight.ptr,
                    scratch.b.ptr,
                    tokens,
                    self.config.hidden_size,
                    self.config.linear_num_value_heads,
                    threads=threads,
                    stream=stream,
                    library=dense_library,
                    runtime=self.runtime,
                )
        return scratch.a, scratch.b

    def run_linear_attention_conv_gdn_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qwen35_linear_attn_conv_decode_fp16(
            scratch.qkv.ptr,
            conv_state.ptr,
            self.tensor(f"{prefix}.conv1d.weight").ptr,
            scratch.conv_out.ptr,
            _linear_qkv_width(self.config),
            self.config.linear_conv_kernel_dim,
            stream=stream,
            library=_library_for(library, "linear_conv"),
            runtime=self.runtime,
        )
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16(
            scratch.conv_out.ptr,
            scratch.z.ptr,
            scratch.a.ptr,
            scratch.b.ptr,
            self.tensor(f"{prefix}.dt_bias").ptr,
            self.tensor(f"{prefix}.A_log").ptr,
            self.tensor(f"{prefix}.norm.weight").ptr,
            recurrent_state.ptr,
            scratch.recurrent_out.ptr,
            self.config.rms_norm_eps if eps is None else eps,
            self.config.linear_num_key_heads,
            self.config.linear_num_value_heads,
            self.config.linear_key_head_dim,
            self.config.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        return scratch.recurrent_out

    def run_linear_attention_prefill_recurrent_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        tokens: int,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        cfg = self.config
        if tokens < cfg.linear_conv_kernel_dim:
            raise ValueError("native linear-attention prefill requires tokens >= linear_conv_kernel_dim")
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv_width = _linear_qkv_width(cfg)
        qwen35_linear_attn_conv_prefill_fp16(
            scratch.qkv.ptr,
            conv_state.ptr,
            self.tensor(f"{prefix}.conv1d.weight").ptr,
            scratch.conv_out.ptr,
            tokens,
            qkv_width,
            cfg.linear_conv_kernel_dim,
            stream=stream,
            library=_library_for(library, "linear_conv"),
            runtime=self.runtime,
        )
        qwen35_linear_attn_prefill_prepare_f32_fp16(
            scratch.conv_out.ptr,
            scratch.a.ptr,
            scratch.b.ptr,
            self.tensor(f"{prefix}.dt_bias").ptr,
            self.tensor(f"{prefix}.A_log").ptr,
            scratch.prefill_query.ptr,
            scratch.prefill_key.ptr,
            scratch.prefill_value.ptr,
            scratch.prefill_beta.ptr,
            scratch.prefill_decay.ptr,
            tokens,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        qwen35_gdn_prefill_recurrent_k2_f32(
            scratch.prefill_query.ptr,
            scratch.prefill_key.ptr,
            scratch.prefill_value.ptr,
            scratch.prefill_beta.ptr,
            scratch.prefill_decay.ptr,
            recurrent_state.ptr,
            scratch.recurrent_out.ptr,
            tokens,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        return scratch.recurrent_out

    def run_linear_attention_prefill_conv_gdn_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        tokens: int,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        cfg = self.config
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        z_width = _linear_value_width(cfg)
        self.run_linear_attention_prefill_recurrent_fp16(
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        qwen35_gdn_prefill_rmsnorm_gate_fp16(
            scratch.recurrent_out.ptr,
            scratch.z.ptr,
            self.tensor(f"{prefix}.norm.weight").ptr,
            scratch.recurrent_bf16.ptr,
            cfg.rms_norm_eps if eps is None else eps,
            tokens,
            cfg.linear_num_value_heads,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        if scratch.recurrent_bf16.shape[-1] != z_width:
            raise ValueError("linear-attention recurrent scratch width mismatch")
        return scratch.recurrent_bf16

    def run_linear_attention_prefill_conv_gdn_segments_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        cu_seqlens: Tensor,
        state_indices: Tensor,
        tokens: int,
        segments: int,
        decode_order_state: bool = False,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        cfg = self.config
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if segments <= 0:
            raise ValueError("segments must be positive")
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv_width = _linear_qkv_width(cfg)
        z_width = _linear_value_width(cfg)
        fp16_to_f32(
            scratch.qkv.ptr,
            scratch.qkv_f32.ptr,
            tokens * qkv_width,
            stream=stream,
            library=_library_for(library, "cast"),
            runtime=self.runtime,
        )
        qwen35_linear_attn_conv_prefill_segments_f32(
            scratch.qkv_f32.ptr,
            conv_state.ptr,
            self.tensor(f"{prefix}.conv1d.weight").ptr,
            scratch.conv_out.ptr,
            cu_seqlens.ptr,
            state_indices.ptr,
            tokens,
            segments,
            qkv_width,
            cfg.linear_conv_kernel_dim,
            stream=stream,
            library=_library_for(library, "linear_conv"),
            runtime=self.runtime,
        )
        if decode_order_state:
            qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16(
                scratch.conv_out.ptr,
                scratch.z.ptr,
                scratch.a.ptr,
                scratch.b.ptr,
                self.tensor(f"{prefix}.dt_bias").ptr,
                self.tensor(f"{prefix}.A_log").ptr,
                self.tensor(f"{prefix}.norm.weight").ptr,
                recurrent_state.ptr,
                scratch.recurrent_out.ptr,
                cu_seqlens.ptr,
                state_indices.ptr,
                tokens,
                segments,
                cfg.rms_norm_eps if eps is None else eps,
                cfg.linear_num_key_heads,
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
                stream=stream,
                library=_library_for(library, "linear_gdn"),
                runtime=self.runtime,
            )
            if scratch.recurrent_out.shape[-1] != z_width:
                raise ValueError("linear-attention recurrent scratch width mismatch")
            return scratch.recurrent_out
        qwen35_linear_attn_prefill_prepare_f32_fp16(
            scratch.conv_out.ptr,
            scratch.a.ptr,
            scratch.b.ptr,
            self.tensor(f"{prefix}.dt_bias").ptr,
            self.tensor(f"{prefix}.A_log").ptr,
            scratch.prefill_query.ptr,
            scratch.prefill_key.ptr,
            scratch.prefill_value.ptr,
            scratch.prefill_beta.ptr,
            scratch.prefill_decay.ptr,
            tokens,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        qwen35_gdn_prefill_recurrent_segments_k2_f32(
            scratch.prefill_query.ptr,
            scratch.prefill_key.ptr,
            scratch.prefill_value.ptr,
            scratch.prefill_beta.ptr,
            scratch.prefill_decay.ptr,
            recurrent_state.ptr,
            scratch.recurrent_out.ptr,
            cu_seqlens.ptr,
            state_indices.ptr,
            tokens,
            segments,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        qwen35_gdn_prefill_rmsnorm_gate_fp16(
            scratch.recurrent_out.ptr,
            scratch.z.ptr,
            self.tensor(f"{prefix}.norm.weight").ptr,
            scratch.recurrent_bf16.ptr,
            cfg.rms_norm_eps if eps is None else eps,
            tokens,
            cfg.linear_num_value_heads,
            cfg.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        if scratch.recurrent_bf16.shape[-1] != z_width:
            raise ValueError("linear-attention recurrent scratch width mismatch")
        return scratch.recurrent_bf16

    def project_linear_attention_out_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        force_pack8_gemv: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn.out_proj"
        width = scratch.recurrent_out.shape[-1]
        f32_to_fp16(
            scratch.recurrent_out.ptr,
            scratch.recurrent_bf16.ptr,
            tokens * width,
            stream=stream,
            library=_library_for(library, "cast"),
            runtime=self.runtime,
        )
        pairs = self.tensor(f"{prefix}.pairs")
        paro_rotate1_fp16(
            scratch.recurrent_bf16.ptr,
            scratch.out_rot.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.theta").ptr,
            self.tensor(f"{prefix}.channel_scales").ptr,
            tokens,
            width,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        self.project_pack8_fp16(
            scratch.out_rot,
            scratch.out_proj,
            weight_prefix=prefix,
            rows=tokens,
            in_features=width,
            group_size=group_size,
            threads=64 if tokens > 1 else 128,
            force_gemv=force_pack8_gemv,
            library=library,
            stream=stream,
        )
        return scratch.out_proj

    def project_linear_attention_prefill_out_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        force_pack8_gemv: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn.out_proj"
        width = scratch.recurrent_bf16.shape[-1]
        pairs = self.tensor(f"{prefix}.pairs")
        paro_rotate1_fp16(
            scratch.recurrent_bf16.ptr,
            scratch.out_rot.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.theta").ptr,
            self.tensor(f"{prefix}.channel_scales").ptr,
            tokens,
            width,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        self.project_pack8_fp16(
            scratch.out_rot,
            scratch.out_proj,
            weight_prefix=prefix,
            rows=tokens,
            in_features=width,
            group_size=group_size,
            threads=64 if tokens > 1 else 128,
            force_gemv=force_pack8_gemv,
            library=library,
            stream=stream,
        )
        return scratch.out_proj

    def project_linear_attention_prefill_gdn_rotate_out_fp16(
        self,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        out_prefix = f"{prefix}.out_proj"
        width = _linear_value_width(self.config)
        pairs = self.tensor(f"{out_prefix}.pairs")
        qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16(
            scratch.recurrent_out.ptr,
            scratch.z.ptr,
            self.tensor(f"{prefix}.norm.weight").ptr,
            scratch.out_rot.ptr,
            pairs.ptr,
            self.tensor(f"{out_prefix}.theta").ptr,
            self.tensor(f"{out_prefix}.channel_scales").ptr,
            self.config.rms_norm_eps,
            tokens,
            self.config.linear_num_value_heads,
            self.config.linear_value_head_dim,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        self.project_pack8_fp16(
            scratch.out_rot,
            scratch.out_proj,
            weight_prefix=out_prefix,
            rows=tokens,
            in_features=width,
            group_size=group_size,
            threads=64 if tokens > 1 else 128,
            library=library,
            stream=stream,
        )
        return scratch.out_proj

    def run_linear_attention_state_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if tokens != 1:
            raise ValueError("linear-attention state orchestrator currently requires tokens=1")
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.rotate_linear_attention_inputs_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_qkv_z_fp16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_ab_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        return self.run_linear_attention_conv_gdn_fp16(
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            library=library,
            stream=stream,
        )

    def run_linear_attention_out_proj_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if tokens != 1:
            raise ValueError("linear-attention out-proj orchestrator currently requires tokens=1")
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.run_linear_attention_state_fp16(
            hidden,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            scratch=scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        return self.project_linear_attention_out_fp16(
            scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_linear_attention_prefill_state_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.rotate_linear_attention_inputs_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_qkv_z_fp16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_ab_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        return self.run_linear_attention_prefill_conv_gdn_fp16(
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            tokens=tokens,
            library=library,
            stream=stream,
        )

    def run_linear_attention_prefill_out_proj_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        if _use_linear_gdn_prefill_rotate_fused(self.config, tokens=tokens, group_size=group_size):
            self.rotate_linear_attention_inputs_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
            self.project_linear_attention_qkv_z_fp16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
            self.project_linear_attention_ab_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
            self.run_linear_attention_prefill_recurrent_fp16(
                scratch,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                tokens=tokens,
                library=library,
                stream=stream,
            )
            return self.project_linear_attention_prefill_gdn_rotate_out_fp16(
                scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        self.run_linear_attention_prefill_state_fp16(
            hidden,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            scratch=scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        return self.project_linear_attention_prefill_out_fp16(
            scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_linear_attention_prefill_state_segments_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        cu_seqlens: Tensor,
        state_indices: Tensor,
        segments: int,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int,
        group_size: int = 128,
        force_selected_c1_projections: bool = False,
        force_selected_c1_qkv_z_projections: bool = False,
        force_selected_c1_ab_projections: bool = False,
        force_batch_gemv_projections: bool = False,
        force_selected_c1_state: bool = False,
        selected_c1_state_pairs: Sequence[tuple[Tensor, Tensor]] | None = None,
        decode_order_state: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        if force_selected_c1_projections:
            self.project_linear_attention_decode_rows_fp16(
                hidden,
                scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        else:
            if force_selected_c1_qkv_z_projections:
                self.project_linear_attention_decode_rows_qkv_z_fp16(
                    hidden,
                    scratch,
                    tokens=tokens,
                    group_size=group_size,
                    library=library,
                    stream=stream,
                )
            else:
                self.rotate_linear_attention_inputs_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
                self.project_linear_attention_qkv_z_fp16(
                    scratch,
                    tokens=tokens,
                    group_size=group_size,
                    force_gemv=force_batch_gemv_projections,
                    library=library,
                    stream=stream,
                )
            if force_selected_c1_ab_projections:
                self.project_linear_attention_decode_rows_ab_fp16(
                    hidden,
                    scratch,
                    tokens=tokens,
                    library=library,
                    stream=stream,
                )
            else:
                self.project_linear_attention_ab_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        if force_selected_c1_state:
            if selected_c1_state_pairs is None:
                raise ValueError("selected_c1_state_pairs are required for selected-c1 linear state replay")
            return self.run_linear_attention_decode_rows_state_fp16(
                scratch,
                state_pairs=selected_c1_state_pairs,
                tokens=tokens,
                library=library,
                stream=stream,
            )
        return self.run_linear_attention_prefill_conv_gdn_segments_fp16(
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            tokens=tokens,
            segments=segments,
            decode_order_state=decode_order_state,
            library=library,
            stream=stream,
        )

    def run_linear_attention_prefill_out_proj_segments_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        cu_seqlens: Tensor,
        state_indices: Tensor,
        segments: int,
        scratch: Qwen35ParoLinearAttentionScratch | None = None,
        tokens: int,
        group_size: int = 128,
        force_selected_c1_projections: bool = False,
        force_selected_c1_qkv_z_projections: bool = False,
        force_selected_c1_ab_projections: bool = False,
        force_batch_gemv_projections: bool = False,
        force_selected_c1_state: bool = False,
        selected_c1_state_pairs: Sequence[tuple[Tensor, Tensor]] | None = None,
        force_selected_c1_out: bool | None = None,
        force_batch_gemv_out: bool = False,
        decode_order_state: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.run_linear_attention_prefill_state_segments_fp16(
            hidden,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            segments=segments,
            scratch=scratch,
            tokens=tokens,
            group_size=group_size,
            force_selected_c1_projections=force_selected_c1_projections,
            force_selected_c1_qkv_z_projections=force_selected_c1_qkv_z_projections,
            force_selected_c1_ab_projections=force_selected_c1_ab_projections,
            force_batch_gemv_projections=force_batch_gemv_projections,
            force_selected_c1_state=force_selected_c1_state,
            selected_c1_state_pairs=selected_c1_state_pairs,
            decode_order_state=decode_order_state,
            library=library,
            stream=stream,
        )
        if force_selected_c1_out is None:
            force_selected_c1_out = force_selected_c1_state
        if force_selected_c1_out:
            if force_selected_c1_state or decode_order_state:
                return self.project_linear_attention_decode_rows_out_fp16(
                    scratch,
                    tokens=tokens,
                    group_size=group_size,
                    library=library,
                    stream=stream,
                )
            return self.project_linear_attention_prefill_rows_out_fp16(
                scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        if force_selected_c1_state or decode_order_state:
            return self.project_linear_attention_out_fp16(
                scratch,
                tokens=tokens,
                group_size=group_size,
                force_pack8_gemv=force_batch_gemv_out,
                library=library,
                stream=stream,
            )
        return self.project_linear_attention_prefill_out_fp16(
            scratch,
            tokens=tokens,
            group_size=group_size,
            force_pack8_gemv=force_batch_gemv_out,
            library=library,
            stream=stream,
        )

    def input_rmsnorm_fp16(
        self,
        hidden: Tensor,
        out: Tensor,
        *,
        tokens: int = 1,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        weight = self.tensor(f"layers.{self.layer_weights.layer_id}.input_layernorm.weight")
        paro_rmsnorm_out_fp16(
            hidden.ptr,
            weight.ptr,
            out.ptr,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps if eps is None else eps,
            stream=stream,
            library=_library_for(library, "norm"),
            runtime=self.runtime,
        )
        return out

    def input_rmsnorm_fp16_per_row(
        self,
        hidden: Tensor,
        out: Tensor,
        *,
        tokens: int = 1,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Diagnostic c>N input RMSNorm path using the token-1 RMS kernel per row."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens == 1:
            return self.input_rmsnorm_fp16(
                hidden,
                out,
                tokens=tokens,
                eps=eps,
                library=library,
                stream=stream,
            )
        weight = self.tensor(f"layers.{self.layer_weights.layer_id}.input_layernorm.weight")
        norm_library = _library_for(library, "norm")
        norm_eps = self.config.rms_norm_eps if eps is None else eps
        for row in range(tokens):
            paro_rmsnorm_out_fp16(
                self._row_tensor_view(hidden, row).ptr,
                weight.ptr,
                self._row_tensor_view(out, row).ptr,
                1,
                self.config.hidden_size,
                norm_eps,
                stream=stream,
                library=norm_library,
                runtime=self.runtime,
            )
        return out

    def post_attention_add_rmsnorm_fp16(
        self,
        hidden: Tensor,
        attn_out: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        weight = self.tensor(f"layers.{self.layer_weights.layer_id}.post_attention_layernorm.weight")
        paro_add_rmsnorm_out_fp16(
            hidden.ptr,
            attn_out.ptr,
            weight.ptr,
            scratch.normed.ptr,
            scratch.residual.ptr,
            tokens,
            self.config.hidden_size,
            self.config.rms_norm_eps if eps is None else eps,
            stream=stream,
            library=_library_for(library, "norm"),
            runtime=self.runtime,
        )
        return scratch.normed, scratch.residual

    def post_attention_add_rmsnorm_fp16_per_row(
        self,
        hidden: Tensor,
        attn_out: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        eps: float | None = None,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Diagnostic c>N post-attention path using the token-1 RMS kernel per row."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens == 1:
            return self.post_attention_add_rmsnorm_fp16(
                hidden,
                attn_out,
                scratch,
                tokens=tokens,
                eps=eps,
                library=library,
                stream=stream,
            )
        weight = self.tensor(f"layers.{self.layer_weights.layer_id}.post_attention_layernorm.weight")
        norm_library = _library_for(library, "norm")
        norm_eps = self.config.rms_norm_eps if eps is None else eps
        for row in range(tokens):
            paro_add_rmsnorm_out_fp16(
                self._row_tensor_view(hidden, row).ptr,
                self._row_tensor_view(attn_out, row).ptr,
                weight.ptr,
                self._row_tensor_view(scratch.normed, row).ptr,
                self._row_tensor_view(scratch.residual, row).ptr,
                1,
                self.config.hidden_size,
                norm_eps,
                stream=stream,
                library=norm_library,
                runtime=self.runtime,
            )
        return scratch.normed, scratch.residual

    def run_linear_attention_moe_c1_layer_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        linear_scratch: Qwen35ParoLinearAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        linear_scratch = linear_scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        use_grouped_moe = False if dense_mlp else _use_moe_grouped_compact_prefill(tokens)
        if dense_mlp:
            if not isinstance(moe_scratch, Qwen35ParoDenseMlpScratch):
                moe_scratch = self.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif not use_grouped_moe:
            if not isinstance(moe_scratch, Qwen35ParoMoeScratch):
                moe_scratch = self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
            moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.input_rmsnorm_fp16(hidden, linear_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        if tokens == 1:
            attn_out = self.run_linear_attention_out_proj_fp16(
                linear_scratch.attn_input,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                scratch=linear_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        else:
            attn_out = self.run_linear_attention_prefill_out_proj_fp16(
                linear_scratch.attn_input,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                scratch=linear_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        mlp_input, residual = self.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        if dense_mlp:
            return self.run_dense_mlp_residual_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        if not use_grouped_moe:
            return self.run_moe_c1_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_grouped_compact_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_linear_attention_moe_packed_prefill_layer_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        cu_seqlens: Tensor,
        state_indices: Tensor,
        segments: int,
        linear_scratch: Qwen35ParoLinearAttentionScratch | None = None,
        moe_scratch: Qwen35ParoGroupedMoeScratch | None = None,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        linear_scratch = linear_scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        if not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
            moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.input_rmsnorm_fp16(hidden, linear_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        attn_out = self.run_linear_attention_prefill_out_proj_segments_fp16(
            linear_scratch.attn_input,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            segments=segments,
            scratch=linear_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        mlp_input, residual = self.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        return self.run_moe_grouped_compact_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_linear_attention_moe_decode_batch_layer_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        cu_seqlens: Tensor,
        state_indices: Tensor,
        segments: int,
        linear_scratch: Qwen35ParoLinearAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch | None = None,
        tokens: int,
        group_size: int = 128,
        force_selected_c1_moe: bool = False,
        force_selected_c1_linear_projections: bool = False,
        force_selected_c1_qkv_z_linear_projections: bool = False,
        force_selected_c1_ab_linear_projections: bool = False,
        force_batch_gemv_linear_projections: bool = False,
        force_selected_c1_linear_state: bool = False,
        selected_c1_linear_state_pairs: Sequence[tuple[Tensor, Tensor]] | None = None,
        force_selected_c1_linear_out: bool | None = None,
        force_batch_gemv_linear_out: bool = False,
        force_per_row_moe: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run native compact decode rows with grouped MoE for batch lanes."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        linear_scratch = linear_scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        if dense_mlp:
            if not isinstance(moe_scratch, Qwen35ParoDenseMlpScratch):
                moe_scratch = self.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif tokens > 1 and not (force_selected_c1_moe or force_per_row_moe):
            if not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
                moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif not isinstance(moe_scratch, Qwen35ParoMoeScratch):
            moe_scratch = self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.input_rmsnorm_fp16(hidden, linear_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        attn_out = self.run_linear_attention_prefill_out_proj_segments_fp16(
            linear_scratch.attn_input,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            segments=segments,
            scratch=linear_scratch,
            tokens=tokens,
            group_size=group_size,
            force_selected_c1_projections=force_selected_c1_linear_projections,
            force_selected_c1_qkv_z_projections=force_selected_c1_qkv_z_linear_projections,
            force_selected_c1_ab_projections=force_selected_c1_ab_linear_projections,
            force_batch_gemv_projections=force_batch_gemv_linear_projections,
            force_selected_c1_state=force_selected_c1_linear_state,
            selected_c1_state_pairs=selected_c1_linear_state_pairs,
            force_selected_c1_out=force_selected_c1_linear_out,
            force_batch_gemv_out=force_batch_gemv_linear_out,
            decode_order_state=not force_selected_c1_linear_state,
            library=library,
            stream=stream,
        )
        mlp_input, residual = self.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        if dense_mlp:
            return self.run_dense_mlp_residual_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        if force_per_row_moe and tokens > 1:
            return self.run_moe_c1_rows_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        if tokens > 1 and not force_selected_c1_moe:
            return self.run_moe_grouped_compact_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_c1_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def route_moe_topk_shared_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        threads: int = 512,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        combined = self.tensor(f"layers.{self.layer_weights.layer_id}.mlp.router_shared_gate.weight")
        prefill_threads = 256 if tokens > 1 else threads
        router_library = _library_for(library, "router")
        if _use_prefill_router_shared_gate_sigmoid_fused(
            tokens=tokens,
            legacy_shared=self._shared_expert_is_legacy_w8a16(),
        ):
            router_fn = qwen35_router_topk_shared_sigmoid_out_fp16
        elif tokens == 1 and _router_topk_coop_enabled():
            router_fn = qwen35_router_topk_shared_coop_out_fp16
        else:
            router_fn = qwen35_router_topk_shared_out_fp16
        router_fn(
            hidden.ptr,
            combined.ptr,
            scratch.router_logits.ptr,
            scratch.selected_experts.ptr,
            scratch.routing_weights.ptr,
            tokens,
            cfg.hidden_size,
            cfg.num_experts + 1,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            threads=prefill_threads,
            stream=stream,
            library=router_library,
            runtime=self.runtime,
        )
        return scratch.selected_experts, scratch.routing_weights

    def selected_moe_gate_up_pack8_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        gate_up_pairs = self.tensor(f"{prefix}.gate_up_weight_pairs")
        paro_rotate1_fp16(
            hidden.ptr,
            scratch.gate_up_input.ptr,
            gate_up_pairs.ptr,
            self.tensor(f"{prefix}.gate_up_weight_theta").ptr,
            self.tensor(f"{prefix}.gate_up_weight_channel_scales").ptr,
            tokens,
            self.config.hidden_size,
            group_size,
            _rotation_krot(gate_up_pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        gate_qweight = self.tensor(f"{prefix}.stacked_gate_qweight_pack8_decode")
        up_qweight = self.tensor(f"{prefix}.stacked_up_qweight_pack8_decode")
        rows = tokens * self.config.num_experts_per_tok
        gemv_awq_selected_dual_pack8_transposed_fp16(
            scratch.gate_up_input.ptr,
            scratch.selected_experts.ptr,
            gate_qweight.ptr,
            self.tensor(f"{prefix}.stacked_gate_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_gate_scales").ptr,
            up_qweight.ptr,
            self.tensor(f"{prefix}.stacked_up_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_up_scales").ptr,
            scratch.gate_up.ptr,
            tokens,
            rows,
            hidden.shape[-1],
            _out_packed_from_transposed_qweight(gate_qweight),
            _out_packed_from_transposed_qweight(up_qweight),
            self.config.num_experts,
            group_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
            runtime=self.runtime,
        )
        return scratch.gate_up

    def activate_rotate_moe_down_fp16(
        self,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        pairs = self.tensor(f"{prefix}.down_weight_pairs")
        silu_mul_dual_rotate_out_fp16(
            scratch.gate_up.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.down_weight_theta").ptr,
            self.tensor(f"{prefix}.down_weight_channel_scales").ptr,
            scratch.down_input.ptr,
            tokens * self.config.num_experts_per_tok,
            self.config.moe_intermediate_size,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "silu"),
            runtime=self.runtime,
        )
        return scratch.down_input

    def selected_moe_down_pack8_fp16(
        self,
        down_input: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        qweight = self.tensor(f"{prefix}.stacked_down_qweight_pack8_decode")
        rows = tokens * self.config.num_experts_per_tok
        gemv_awq_selected_pack8_transposed_fp16(
            down_input.ptr,
            scratch.selected_experts.ptr,
            qweight.ptr,
            self.tensor(f"{prefix}.stacked_down_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_down_scales").ptr,
            scratch.down_out.ptr,
            rows,
            down_input.shape[-1],
            _out_packed_from_transposed_qweight(qweight),
            self.config.num_experts,
            group_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
            runtime=self.runtime,
        )
        return scratch.down_out

    def shared_expert_gate_up_silu_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch,
        *,
        tokens: int = 1,
        threads: int = 64,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        w8a16_library = _library_for(library, "w8a16")
        token_tile = _use_shared_gate_up_prefill_token_tiled(tokens)
        if token_tile:
            w8a16_shared_gate_up_silu_fp16_token_tiled(
                hidden.ptr,
                self.tensor(f"{prefix}.gate_up_weight_w8a16").ptr,
                self.tensor(f"{prefix}.gate_up_weight_w8a16_scale").ptr,
                scratch.shared_intermediate.ptr,
                tokens,
                self.config.hidden_size,
                self.config.shared_expert_intermediate_size,
                token_tile=token_tile,
                threads=threads,
                stream=stream,
                library=w8a16_library,
                runtime=self.runtime,
            )
        else:
            w8a16_shared_gate_up_silu_fp16(
                hidden.ptr,
                self.tensor(f"{prefix}.gate_up_weight_w8a16").ptr,
                self.tensor(f"{prefix}.gate_up_weight_w8a16_scale").ptr,
                scratch.shared_intermediate.ptr,
                tokens,
                self.config.hidden_size,
                self.config.shared_expert_intermediate_size,
                threads=threads,
                stream=stream,
                library=w8a16_library,
                runtime=self.runtime,
            )
        return scratch.shared_intermediate

    def shared_expert_down_combine_residual_fp16(
        self,
        scratch: Qwen35ParoGroupedMoeScratch,
        residual: Tensor,
        *,
        tokens: int = 1,
        threads: int = 64,
        shared_gate_already_sigmoid: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        w8a16_library = _library_for(library, "w8a16")
        shared_gate_logits_ptr = scratch.router_logits.ptr + self.config.num_experts * DType.FP32.itemsize
        if not shared_gate_already_sigmoid:
            # Overwrite the shared-gate logit column in place with sigmoid(logit).
            # Router top-k/weights have already been materialized, and this avoids
            # recomputing the same expf once per hidden row tile below.  The P3.2
            # diagnostic path can do this in the prefill router select kernel instead.
            w8a16_shared_gate_sigmoid_fp32(
                shared_gate_logits_ptr,
                shared_gate_logits_ptr,
                tokens,
                self.config.num_experts + 1,
                threads=128,
                stream=stream,
                library=w8a16_library,
                runtime=self.runtime,
            )
        token_tile = _use_shared_down_combine_prefill_token_tiled(tokens)
        if token_tile:
            w8a16_shared_down_combine_residual_fp16_token_tiled(
                scratch.shared_intermediate.ptr,
                self.tensor(f"{prefix}.down_weight_w8a16").ptr,
                self.tensor(f"{prefix}.down_weight_w8a16_scale").ptr,
                scratch.selected_out.ptr,
                shared_gate_logits_ptr,
                residual.ptr,
                scratch.moe_out.ptr,
                tokens,
                self.config.hidden_size,
                self.config.shared_expert_intermediate_size,
                self.config.num_experts + 1,
                token_tile=token_tile,
                threads=threads,
                stream=stream,
                library=w8a16_library,
                runtime=self.runtime,
            )
        else:
            w8a16_shared_down_combine_residual_fp16(
                scratch.shared_intermediate.ptr,
                self.tensor(f"{prefix}.down_weight_w8a16").ptr,
                self.tensor(f"{prefix}.down_weight_w8a16_scale").ptr,
                scratch.selected_out.ptr,
                shared_gate_logits_ptr,
                residual.ptr,
                scratch.moe_out.ptr,
                tokens,
                self.config.hidden_size,
                self.config.shared_expert_intermediate_size,
                self.config.num_experts + 1,
                threads=threads,
                stream=stream,
                library=w8a16_library,
                runtime=self.runtime,
            )
        return scratch.moe_out

    def shared_expert_w8a16_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        threads: int = 64,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        w8a16_library = _library_for(library, "w8a16")
        if tokens > 1:
            token_tile = _use_shared_gate_up_prefill_token_tiled(tokens)
            if token_tile:
                w8a16_shared_gate_up_silu_fp16_token_tiled(
                    hidden.ptr,
                    self.tensor(f"{prefix}.gate_up_weight_w8a16").ptr,
                    self.tensor(f"{prefix}.gate_up_weight_w8a16_scale").ptr,
                    scratch.shared_intermediate.ptr,
                    tokens,
                    self.config.hidden_size,
                    self.config.shared_expert_intermediate_size,
                    token_tile=token_tile,
                    threads=threads,
                    stream=stream,
                    library=w8a16_library,
                    runtime=self.runtime,
                )
            else:
                w8a16_shared_gate_up_silu_fp16(
                    hidden.ptr,
                    self.tensor(f"{prefix}.gate_up_weight_w8a16").ptr,
                    self.tensor(f"{prefix}.gate_up_weight_w8a16_scale").ptr,
                    scratch.shared_intermediate.ptr,
                    tokens,
                    self.config.hidden_size,
                    self.config.shared_expert_intermediate_size,
                    threads=threads,
                    stream=stream,
                    library=w8a16_library,
                    runtime=self.runtime,
                )
        else:
            w8a16_linear_fp16_lowp_out(
                hidden.ptr,
                self.tensor(f"{prefix}.gate_up_weight_w8a16").ptr,
                self.tensor(f"{prefix}.gate_up_weight_w8a16_scale").ptr,
                scratch.shared_up.ptr,
                tokens,
                self.config.hidden_size,
                2 * self.config.shared_expert_intermediate_size,
                threads=threads,
                stream=stream,
                library=w8a16_library,
                runtime=self.runtime,
            )
            silu_mul_dual_out_fp16(
                scratch.shared_up.ptr,
                scratch.shared_intermediate.ptr,
                tokens,
                self.config.shared_expert_intermediate_size,
                stream=stream,
                library=_library_for(library, "silu"),
                runtime=self.runtime,
            )
        w8a16_linear_fp16_lowp_out(
            scratch.shared_intermediate.ptr,
            self.tensor(f"{prefix}.down_weight_w8a16").ptr,
            self.tensor(f"{prefix}.down_weight_w8a16_scale").ptr,
            scratch.shared_out.ptr,
            tokens,
            self.config.shared_expert_intermediate_size,
            self.config.hidden_size,
            threads=threads,
            stream=stream,
            library=w8a16_library,
            runtime=self.runtime,
        )
        return scratch.shared_out

    def shared_expert_paro_w4_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run the W4 PARO dense shared expert on ``hidden``.

        The shared expert uses three independent dense PARO linears
        (gate_proj, up_proj, down_proj) with their own rotation params; for
        tokens=1/small batches we use a fused gate/up rotate2, the dual GEMV
        with separate inputs + packed gate||up output, then fused SiLU +
        down-rotation. For larger batches we use the fused W4 prefill kernel
        which writes gate/up to separate buffers and pair them via
        silu_mul_separate_out.
        """
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        cfg = self.config
        gate_base = f"{prefix}.gate_proj"
        up_base = f"{prefix}.up_proj"
        down_base = f"{prefix}.down_proj"

        gate_pairs = self.tensor(f"{gate_base}.pairs")
        up_pairs = self.tensor(f"{up_base}.pairs")
        down_pairs = self.tensor(f"{down_base}.pairs")

        gate_krot = _rotation_krot(gate_pairs)
        up_krot = _rotation_krot(up_pairs)
        if gate_krot == up_krot:
            paro_rotate2_fp16(
                hidden.ptr,
                scratch.shared_gate_input.ptr,
                scratch.shared_up_input.ptr,
                gate_pairs.ptr,
                up_pairs.ptr,
                self.tensor(f"{gate_base}.theta").ptr,
                self.tensor(f"{up_base}.theta").ptr,
                self.tensor(f"{gate_base}.channel_scales").ptr,
                self.tensor(f"{up_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                gate_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
        else:
            paro_rotate1_fp16(
                hidden.ptr,
                scratch.shared_gate_input.ptr,
                gate_pairs.ptr,
                self.tensor(f"{gate_base}.theta").ptr,
                self.tensor(f"{gate_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                gate_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
            paro_rotate1_fp16(
                hidden.ptr,
                scratch.shared_up_input.ptr,
                up_pairs.ptr,
                self.tensor(f"{up_base}.theta").ptr,
                self.tensor(f"{up_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                up_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )

        gate_qweight = self.tensor(f"{gate_base}.qweight_pack8_decode")
        up_qweight = self.tensor(f"{up_base}.qweight_pack8_decode")
        if tokens == 1:
            gemv_awq_dual_pack8_transposed_fp16(
                scratch.shared_gate_input.ptr,
                scratch.shared_up_input.ptr,
                gate_qweight.ptr,
                self.tensor(f"{gate_base}.qzeros").ptr,
                self.tensor(f"{gate_base}.scales").ptr,
                up_qweight.ptr,
                self.tensor(f"{up_base}.qzeros").ptr,
                self.tensor(f"{up_base}.scales").ptr,
                scratch.shared_up.ptr,
                tokens,
                cfg.hidden_size,
                _out_packed_from_generic_transposed_qweight(gate_qweight),
                _out_packed_from_generic_transposed_qweight(up_qweight),
                group_size,
                threads=threads,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
            silu_mul_dual_rotate_out_fp16(
                scratch.shared_up.ptr,
                down_pairs.ptr,
                self.tensor(f"{down_base}.theta").ptr,
                self.tensor(f"{down_base}.channel_scales").ptr,
                scratch.shared_down_input.ptr,
                tokens,
                cfg.shared_expert_intermediate_size,
                group_size,
                _rotation_krot(down_pairs),
                stream=stream,
                library=_library_for(library, "silu"),
                runtime=self.runtime,
            )
        else:
            awq_fusedw4_prefill_dual_fp16(
                scratch.shared_gate_input.ptr,
                scratch.shared_up_input.ptr,
                gate_qweight.ptr,
                self.tensor(f"{gate_base}.qzeros").ptr,
                self.tensor(f"{gate_base}.scales").ptr,
                up_qweight.ptr,
                self.tensor(f"{up_base}.qzeros").ptr,
                self.tensor(f"{up_base}.scales").ptr,
                scratch.shared_gate_out.ptr,
                scratch.shared_up_out.ptr,
                tokens,
                cfg.hidden_size,
                _out_packed_from_generic_transposed_qweight(gate_qweight),
                _out_packed_from_generic_transposed_qweight(up_qweight),
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
            silu_mul_separate_out_fp16(
                scratch.shared_gate_out.ptr,
                scratch.shared_up_out.ptr,
                scratch.shared_intermediate.ptr,
                tokens,
                cfg.shared_expert_intermediate_size,
                stream=stream,
                library=_library_for(library, "silu"),
                runtime=self.runtime,
            )

        if tokens != 1:
            paro_rotate1_fp16(
                scratch.shared_intermediate.ptr,
                scratch.shared_down_input.ptr,
                down_pairs.ptr,
                self.tensor(f"{down_base}.theta").ptr,
                self.tensor(f"{down_base}.channel_scales").ptr,
                tokens,
                cfg.shared_expert_intermediate_size,
                group_size,
                _rotation_krot(down_pairs),
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
        down_qweight = self.tensor(f"{down_base}.qweight_pack8_decode")
        if tokens == 1:
            gemv_awq_pack8_transposed_fp16(
                scratch.shared_down_input.ptr,
                down_qweight.ptr,
                self.tensor(f"{down_base}.qzeros").ptr,
                self.tensor(f"{down_base}.scales").ptr,
                scratch.shared_out.ptr,
                tokens,
                cfg.shared_expert_intermediate_size,
                _out_packed_from_generic_transposed_qweight(down_qweight),
                group_size,
                threads=threads,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        else:
            awq_fusedw4_prefill_fp16(
                scratch.shared_down_input.ptr,
                down_qweight.ptr,
                self.tensor(f"{down_base}.qzeros").ptr,
                self.tensor(f"{down_base}.scales").ptr,
                scratch.shared_out.ptr,
                tokens,
                cfg.shared_expert_intermediate_size,
                _out_packed_from_generic_transposed_qweight(down_qweight),
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        return scratch.shared_out

    def dense_mlp_paro_w4_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoDenseMlpScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run dense Qwen3.5 PARO MLP gate/up/down without the residual add."""

        prefix = f"layers.{self.layer_weights.layer_id}.mlp"
        cfg = self.config
        intermediate = cfg.moe_intermediate_size
        gate_base = f"{prefix}.gate_proj"
        up_base = f"{prefix}.up_proj"
        down_base = f"{prefix}.down_proj"

        gate_pairs = self.tensor(f"{gate_base}.pairs")
        up_pairs = self.tensor(f"{up_base}.pairs")
        down_pairs = self.tensor(f"{down_base}.pairs")

        gate_krot = _rotation_krot(gate_pairs)
        up_krot = _rotation_krot(up_pairs)
        if gate_krot == up_krot:
            paro_rotate2_fp16(
                hidden.ptr,
                scratch.shared_gate_input.ptr,
                scratch.shared_up_input.ptr,
                gate_pairs.ptr,
                up_pairs.ptr,
                self.tensor(f"{gate_base}.theta").ptr,
                self.tensor(f"{up_base}.theta").ptr,
                self.tensor(f"{gate_base}.channel_scales").ptr,
                self.tensor(f"{up_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                gate_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
        else:
            paro_rotate1_fp16(
                hidden.ptr,
                scratch.shared_gate_input.ptr,
                gate_pairs.ptr,
                self.tensor(f"{gate_base}.theta").ptr,
                self.tensor(f"{gate_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                gate_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
            paro_rotate1_fp16(
                hidden.ptr,
                scratch.shared_up_input.ptr,
                up_pairs.ptr,
                self.tensor(f"{up_base}.theta").ptr,
                self.tensor(f"{up_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                up_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )

        gate_qweight = self.tensor(f"{gate_base}.qweight_pack8_decode")
        up_qweight = self.tensor(f"{up_base}.qweight_pack8_decode")
        if tokens == 1:
            gemv_awq_dual_pack8_transposed_fp16(
                scratch.shared_gate_input.ptr,
                scratch.shared_up_input.ptr,
                gate_qweight.ptr,
                self.tensor(f"{gate_base}.qzeros").ptr,
                self.tensor(f"{gate_base}.scales").ptr,
                up_qweight.ptr,
                self.tensor(f"{up_base}.qzeros").ptr,
                self.tensor(f"{up_base}.scales").ptr,
                scratch.shared_up.ptr,
                tokens,
                cfg.hidden_size,
                _out_packed_from_generic_transposed_qweight(gate_qweight),
                _out_packed_from_generic_transposed_qweight(up_qweight),
                group_size,
                threads=threads,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
            silu_mul_dual_rotate_out_fp16(
                scratch.shared_up.ptr,
                down_pairs.ptr,
                self.tensor(f"{down_base}.theta").ptr,
                self.tensor(f"{down_base}.channel_scales").ptr,
                scratch.shared_down_input.ptr,
                tokens,
                intermediate,
                group_size,
                _rotation_krot(down_pairs),
                stream=stream,
                library=_library_for(library, "silu"),
                runtime=self.runtime,
            )
        else:
            awq_fusedw4_prefill_dual_fp16(
                scratch.shared_gate_input.ptr,
                scratch.shared_up_input.ptr,
                gate_qweight.ptr,
                self.tensor(f"{gate_base}.qzeros").ptr,
                self.tensor(f"{gate_base}.scales").ptr,
                up_qweight.ptr,
                self.tensor(f"{up_base}.qzeros").ptr,
                self.tensor(f"{up_base}.scales").ptr,
                scratch.shared_gate_out.ptr,
                scratch.shared_up_out.ptr,
                tokens,
                cfg.hidden_size,
                _out_packed_from_generic_transposed_qweight(gate_qweight),
                _out_packed_from_generic_transposed_qweight(up_qweight),
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
            silu_mul_separate_out_fp16(
                scratch.shared_gate_out.ptr,
                scratch.shared_up_out.ptr,
                scratch.shared_intermediate.ptr,
                tokens,
                intermediate,
                stream=stream,
                library=_library_for(library, "silu"),
                runtime=self.runtime,
            )

        if tokens != 1:
            paro_rotate1_fp16(
                scratch.shared_intermediate.ptr,
                scratch.shared_down_input.ptr,
                down_pairs.ptr,
                self.tensor(f"{down_base}.theta").ptr,
                self.tensor(f"{down_base}.channel_scales").ptr,
                tokens,
                intermediate,
                group_size,
                _rotation_krot(down_pairs),
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
        down_qweight = self.tensor(f"{down_base}.qweight_pack8_decode")
        if tokens == 1:
            gemv_awq_pack8_transposed_fp16(
                scratch.shared_down_input.ptr,
                down_qweight.ptr,
                self.tensor(f"{down_base}.qzeros").ptr,
                self.tensor(f"{down_base}.scales").ptr,
                scratch.shared_out.ptr,
                tokens,
                intermediate,
                _out_packed_from_generic_transposed_qweight(down_qweight),
                group_size,
                threads=threads,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        else:
            awq_fusedw4_prefill_fp16(
                scratch.shared_down_input.ptr,
                down_qweight.ptr,
                self.tensor(f"{down_base}.qzeros").ptr,
                self.tensor(f"{down_base}.scales").ptr,
                scratch.shared_out.ptr,
                tokens,
                intermediate,
                _out_packed_from_generic_transposed_qweight(down_qweight),
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        return scratch.shared_out

    def run_dense_mlp_residual_fp16(
        self,
        hidden: Tensor,
        residual: Tensor,
        *,
        scratch: Qwen35ParoDenseMlpScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        dense_out = self.dense_mlp_paro_w4_fp16(
            hidden,
            scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        runtime = self.runtime or get_hip_runtime()
        self._memset_tensor(scratch.shared_zero, stream=stream, runtime=runtime)
        self._memset_tensor(scratch.gate_logits, stream=stream, runtime=runtime)
        shared_gate_combine_residual_batch_out_fp16(
            dense_out.ptr,
            scratch.shared_zero.ptr,
            scratch.gate_logits.ptr,
            residual.ptr,
            scratch.moe_out.ptr,
            tokens,
            self.config.hidden_size,
            1,
            stream=stream,
            library=_library_for(library, "combine"),
            runtime=self.runtime,
        )
        return scratch.moe_out

    def shared_expert_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if self._shared_expert_is_legacy_w8a16():
            return self.shared_expert_w8a16_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        if self._shared_expert_is_packed_paro_w4():
            return self.shared_expert_paro_w4_fp16(
                hidden,
                scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        raise KeyError(f"no supported shared_expert tensors found under {prefix}")

    def combine_moe_c1_shared_residual_fp16(
        self,
        scratch: Qwen35ParoMoeScratch,
        *,
        shared: Tensor,
        residual: Tensor,
        out: Tensor | None = None,
        tokens: int = 1,
        threads: int = 256,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        target = out or scratch.moe_out
        shared_gate_logits_ptr = scratch.router_logits.ptr + self.config.num_experts * DType.FP32.itemsize
        if tokens == 1:
            weighted_sum_shared_gate_combine_residual_out_fp16_f32w(
                scratch.down_out.ptr,
                scratch.routing_weights.ptr,
                shared.ptr,
                shared_gate_logits_ptr,
                residual.ptr,
                target.ptr,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                threads=threads,
                stream=stream,
                library=_library_for(library, "combine"),
                runtime=self.runtime,
            )
        else:
            weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w(
                scratch.down_out.ptr,
                scratch.routing_weights.ptr,
                shared.ptr,
                shared_gate_logits_ptr,
                residual.ptr,
                target.ptr,
                tokens,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.num_experts + 1,
                threads=threads,
                stream=stream,
                library=_library_for(library, "combine"),
                runtime=self.runtime,
            )
        return target

    def _prepare_grouped_moe_prefill_metadata(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoGroupedMoeScratch,
        *,
        tokens: int,
        library=None,
        stream: int = 0,
    ) -> int:
        cfg = self.config
        top_k = cfg.num_experts_per_tok
        total_lanes = tokens * top_k
        runtime = self.runtime or get_hip_runtime()
        self._memset_tensor(scratch.counts, stream=stream, runtime=runtime)
        qwen35_moe_group_count(
            scratch.selected_experts.ptr,
            scratch.counts.ptr,
            total_lanes,
            cfg.num_experts,
            stream=stream,
            library=_library_for(library, "group_scatter"),
            runtime=runtime,
        )
        qwen35_moe_group_prefix(
            scratch.counts.ptr,
            scratch.padded_counts.ptr,
            scratch.expert_start.ptr,
            scratch.total_padded.ptr,
            cfg.num_experts,
            1,
            stream=stream,
            library=_library_for(library, "group_scatter"),
            runtime=runtime,
        )
        self._memset_tensor(scratch.tile_expert, value=0xFF, stream=stream, runtime=runtime)
        qwen35_moe_wmma_tile_map(
            scratch.expert_start.ptr,
            scratch.wmma_expert_start.ptr,
            scratch.tile_expert.ptr,
            scratch.wmma_total.ptr,
            cfg.num_experts,
            stream=stream,
            library=_library_for(library, "group_scatter"),
            runtime=runtime,
        )
        self._memset_tensor(scratch.scatter_offsets, stream=stream, runtime=runtime)
        qwen35_moe_group_scatter_gather_lowp(
            hidden.ptr,
            scratch.selected_experts.ptr,
            scratch.routing_weights.ptr,
            scratch.expert_start.ptr,
            scratch.scatter_offsets.ptr,
            scratch.sorted_lanes.ptr,
            scratch.sorted_experts.ptr,
            scratch.sorted_weights.ptr,
            scratch.packed_hidden.ptr,
            total_lanes,
            cfg.num_experts,
            top_k,
            cfg.hidden_size,
            stream=stream,
            library=_library_for(library, "group_scatter"),
            runtime=runtime,
        )
        return total_lanes

    def run_moe_grouped_compact_fp16(
        self,
        hidden: Tensor,
        residual: Tensor,
        *,
        scratch: Qwen35ParoGroupedMoeScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        cfg = self.config
        top_k = cfg.num_experts_per_tok
        self.route_moe_topk_shared_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        total_lanes = self._prepare_grouped_moe_prefill_metadata(
            hidden,
            scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        gate_up_pairs = self.tensor(f"{prefix}.gate_up_weight_pairs")
        paro_rotate1_fp16(
            scratch.packed_hidden.ptr,
            scratch.packed_gate_up_input.ptr,
            gate_up_pairs.ptr,
            self.tensor(f"{prefix}.gate_up_weight_theta").ptr,
            self.tensor(f"{prefix}.gate_up_weight_channel_scales").ptr,
            total_lanes,
            cfg.hidden_size,
            group_size,
            _rotation_krot(gate_up_pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        gate_qweight = self.tensor(f"{prefix}.stacked_gate_qweight_pack8_decode")
        gate_qzeros = self.tensor(f"{prefix}.stacked_gate_qzeros")
        gate_scales = self.tensor(f"{prefix}.stacked_gate_scales")
        up_qweight = self.tensor(f"{prefix}.stacked_up_qweight_pack8_decode")
        up_qzeros = self.tensor(f"{prefix}.stacked_up_qzeros")
        up_scales = self.tensor(f"{prefix}.stacked_up_scales")
        wmma_total_rows = scratch.tile_expert.numel * 16
        gemm_awq_selected_dual_pack8_wmma_compact_fp16(
            scratch.packed_gate_up_input.ptr,
            scratch.expert_start.ptr,
            scratch.wmma_expert_start.ptr,
            scratch.tile_expert.ptr,
            gate_qweight.ptr,
            gate_qzeros.ptr,
            gate_scales.ptr,
            up_qweight.ptr,
            up_qzeros.ptr,
            up_scales.ptr,
            scratch.gate_up.ptr,
            total_lanes,
            cfg.hidden_size,
            _out_packed_from_transposed_qweight(gate_qweight),
            _out_packed_from_transposed_qweight(up_qweight),
            cfg.num_experts,
            group_size,
            wmma_total_rows,
            stream=stream,
            library=_library_for(library, "wmma"),
            runtime=self.runtime,
        )
        pairs = self.tensor(f"{prefix}.down_weight_pairs")
        silu_mul_dual_rotate_out_fp16(
            scratch.gate_up.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.down_weight_theta").ptr,
            self.tensor(f"{prefix}.down_weight_channel_scales").ptr,
            scratch.down_input.ptr,
            total_lanes,
            cfg.moe_intermediate_size,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "silu"),
            runtime=self.runtime,
        )
        down_qweight = self.tensor(f"{prefix}.stacked_down_qweight_pack8_decode")
        gemm_awq_selected_pack8_wmma_compact_fp16(
            scratch.down_input.ptr,
            scratch.expert_start.ptr,
            scratch.wmma_expert_start.ptr,
            scratch.tile_expert.ptr,
            down_qweight.ptr,
            self.tensor(f"{prefix}.stacked_down_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_down_scales").ptr,
            scratch.down_out.ptr,
            total_lanes,
            cfg.moe_intermediate_size,
            _out_packed_from_transposed_qweight(down_qweight),
            cfg.num_experts,
            group_size,
            wmma_total_rows,
            stream=stream,
            library=_library_for(library, "wmma"),
            runtime=self.runtime,
        )
        weighted_lanes_sum_out_fp16_f32w(
            scratch.down_out.ptr,
            scratch.sorted_weights.ptr,
            scratch.sorted_lanes.ptr,
            scratch.lane_to_row.ptr,
            scratch.selected_out.ptr,
            tokens,
            top_k,
            cfg.hidden_size,
            stream=stream,
            library=_library_for(library, "combine"),
            runtime=self.runtime,
        )
        if self._shared_expert_is_legacy_w8a16():
            self.shared_expert_gate_up_silu_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
            return self.shared_expert_down_combine_residual_fp16(
                scratch,
                residual,
                tokens=tokens,
                shared_gate_already_sigmoid=_use_prefill_router_shared_gate_sigmoid_fused(
                    tokens=tokens,
                    legacy_shared=True,
                ),
                library=library,
                stream=stream,
            )
        shared = self.shared_expert_paro_w4_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        shared_gate_logits_ptr = scratch.router_logits.ptr + cfg.num_experts * DType.FP32.itemsize
        shared_gate_combine_residual_batch_out_fp16(
            scratch.selected_out.ptr,
            shared.ptr,
            shared_gate_logits_ptr,
            residual.ptr,
            scratch.moe_out.ptr,
            tokens,
            cfg.hidden_size,
            cfg.num_experts + 1,
            stream=stream,
            library=_library_for(library, "combine"),
            runtime=self.runtime,
        )
        return scratch.moe_out

    def run_moe_c1_rows_fp16(
        self,
        hidden: Tensor,
        residual: Tensor,
        *,
        scratch: Qwen35ParoMoeScratch | None = None,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Replay MoE with true token-1 kernels for each decode row."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        scratch = scratch or self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        row_scratch = self.reserve_moe_c1_scratch(tokens=1, activation_dtype=DType.FP16, prefix="moe.decode_row")
        runtime = self.runtime or get_hip_runtime()
        for row in range(tokens):
            row_out = self.run_moe_c1_fp16(
                self._row_tensor_view(hidden, row),
                self._row_tensor_view(residual, row),
                scratch=row_scratch,
                tokens=1,
                group_size=group_size,
                library=library,
                stream=stream,
            )
            runtime.memcpy_async(
                scratch.moe_out.ptr + row * self.config.hidden_size * scratch.moe_out.dtype.itemsize,
                row_out.ptr,
                self.config.hidden_size * scratch.moe_out.dtype.itemsize,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
        return scratch.moe_out

    def run_moe_c1_fp16(
        self,
        hidden: Tensor,
        residual: Tensor,
        *,
        scratch: Qwen35ParoMoeScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.route_moe_topk_shared_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        self.selected_moe_gate_up_pack8_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.activate_rotate_moe_down_fp16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.selected_moe_down_pack8_fp16(scratch.down_input, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        shared = self.shared_expert_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        return self.combine_moe_c1_shared_residual_fp16(
            scratch,
            shared=shared,
            residual=residual,
            tokens=tokens,
            library=library,
            stream=stream,
        )

    def route_moe_topk_shared_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        threads: int = 512,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        combined = self.tensor(f"layers.{self.layer_weights.layer_id}.mlp.router_shared_gate.weight")
        router_library = _library_for(library, "router")
        router_fn = (
            qwen35_router_topk_shared_coop_out_bf16
            if tokens == 1 and _router_topk_coop_enabled()
            else qwen35_router_topk_shared_out_bf16
        )
        router_fn(
            hidden.ptr,
            combined.ptr,
            scratch.router_logits.ptr,
            scratch.selected_experts.ptr,
            scratch.routing_weights.ptr,
            tokens,
            cfg.hidden_size,
            cfg.num_experts + 1,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            threads=threads,
            stream=stream,
            library=router_library,
            runtime=self.runtime,
        )
        return scratch.selected_experts, scratch.routing_weights

    def selected_moe_gate_up_pack8_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        gate_up_pairs = self.tensor(f"{prefix}.gate_up_weight_pairs")
        paro_rotate1_bf16(
            hidden.ptr,
            scratch.gate_up_input.ptr,
            gate_up_pairs.ptr,
            self.tensor(f"{prefix}.gate_up_weight_theta").ptr,
            self.tensor(f"{prefix}.gate_up_weight_channel_scales").ptr,
            tokens,
            self.config.hidden_size,
            group_size,
            _rotation_krot(gate_up_pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        gate_qweight = self.tensor(f"{prefix}.stacked_gate_qweight_pack8_decode")
        gate_qzeros = self.tensor(f"{prefix}.stacked_gate_qzeros")
        gate_scales = self.tensor(f"{prefix}.stacked_gate_scales")
        up_qweight = self.tensor(f"{prefix}.stacked_up_qweight_pack8_decode")
        up_qzeros = self.tensor(f"{prefix}.stacked_up_qzeros")
        up_scales = self.tensor(f"{prefix}.stacked_up_scales")
        rows = tokens * self.config.num_experts_per_tok
        gemv_awq_selected_dual_pack8_transposed_bf16(
            scratch.gate_up_input.ptr,
            scratch.selected_experts.ptr,
            gate_qweight.ptr,
            gate_qzeros.ptr,
            gate_scales.ptr,
            up_qweight.ptr,
            up_qzeros.ptr,
            up_scales.ptr,
            scratch.gate_up.ptr,
            tokens,
            rows,
            hidden.shape[-1],
            _out_packed_from_transposed_qweight(gate_qweight),
            _out_packed_from_transposed_qweight(up_qweight),
            self.config.num_experts,
            group_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
            runtime=self.runtime,
        )
        return scratch.gate_up

    def activate_rotate_moe_down_bf16(
        self,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        pairs = self.tensor(f"{prefix}.down_weight_pairs")
        theta = self.tensor(f"{prefix}.down_weight_theta")
        scales = self.tensor(f"{prefix}.down_weight_channel_scales")
        silu_mul_dual_rotate_out_bf16(
            scratch.gate_up.ptr,
            pairs.ptr,
            theta.ptr,
            scales.ptr,
            scratch.down_input.ptr,
            tokens * self.config.num_experts_per_tok,
            self.config.moe_intermediate_size,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "silu"),
            runtime=self.runtime,
        )
        return scratch.down_input

    def selected_moe_down_pack8_bf16(
        self,
        down_input: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        qweight = self.tensor(f"{prefix}.stacked_down_qweight_pack8_decode")
        qzeros = self.tensor(f"{prefix}.stacked_down_qzeros")
        scales = self.tensor(f"{prefix}.stacked_down_scales")
        rows = tokens * self.config.num_experts_per_tok
        gemv_awq_selected_pack8_transposed_bf16(
            down_input.ptr,
            scratch.selected_experts.ptr,
            qweight.ptr,
            qzeros.ptr,
            scales.ptr,
            scratch.down_out.ptr,
            rows,
            down_input.shape[-1],
            _out_packed_from_transposed_qweight(qweight),
            self.config.num_experts,
            group_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
            runtime=self.runtime,
        )
        return scratch.down_out

    def shared_expert_w8a16_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch,
        *,
        tokens: int = 1,
        threads: int = 64,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        gate_up_weight = self.tensor(f"{prefix}.gate_up_weight_w8a16")
        gate_up_scale = self.tensor(f"{prefix}.gate_up_weight_w8a16_scale")
        down_weight = self.tensor(f"{prefix}.down_weight_w8a16")
        down_scale = self.tensor(f"{prefix}.down_weight_w8a16_scale")
        w8a16_linear_bf16_lowp_out(
            hidden.ptr,
            gate_up_weight.ptr,
            gate_up_scale.ptr,
            scratch.shared_up.ptr,
            tokens,
            self.config.hidden_size,
            2 * self.config.shared_expert_intermediate_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "w8a16"),
            runtime=self.runtime,
        )
        silu_mul_dual_out_bf16(
            scratch.shared_up.ptr,
            scratch.shared_intermediate.ptr,
            tokens,
            self.config.shared_expert_intermediate_size,
            stream=stream,
            library=_library_for(library, "silu"),
            runtime=self.runtime,
        )
        w8a16_linear_bf16_lowp_out(
            scratch.shared_intermediate.ptr,
            down_weight.ptr,
            down_scale.ptr,
            scratch.shared_out.ptr,
            tokens,
            self.config.shared_expert_intermediate_size,
            self.config.hidden_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "w8a16"),
            runtime=self.runtime,
        )
        return scratch.shared_out

    def shared_expert_paro_w4_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """BF16 W4 PARO dense shared expert path.

        Mirrors :meth:`shared_expert_paro_w4_fp16`, but BF16 has no fused
        prefill kernel, so the same dual GEMV (which accepts ``rows`` > 1)
        is used for every ``tokens`` value. Suboptimal for large prefill
        batches relative to a hypothetical BF16 fused W4 prefill kernel
        but functionally correct.
        """
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        cfg = self.config
        gate_base = f"{prefix}.gate_proj"
        up_base = f"{prefix}.up_proj"
        down_base = f"{prefix}.down_proj"

        gate_pairs = self.tensor(f"{gate_base}.pairs")
        up_pairs = self.tensor(f"{up_base}.pairs")
        down_pairs = self.tensor(f"{down_base}.pairs")

        gate_krot = _rotation_krot(gate_pairs)
        up_krot = _rotation_krot(up_pairs)
        if gate_krot == up_krot:
            paro_rotate2_bf16(
                hidden.ptr,
                scratch.shared_gate_input.ptr,
                scratch.shared_up_input.ptr,
                gate_pairs.ptr,
                up_pairs.ptr,
                self.tensor(f"{gate_base}.theta").ptr,
                self.tensor(f"{up_base}.theta").ptr,
                self.tensor(f"{gate_base}.channel_scales").ptr,
                self.tensor(f"{up_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                gate_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
        else:
            paro_rotate1_bf16(
                hidden.ptr,
                scratch.shared_gate_input.ptr,
                gate_pairs.ptr,
                self.tensor(f"{gate_base}.theta").ptr,
                self.tensor(f"{gate_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                gate_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
            paro_rotate1_bf16(
                hidden.ptr,
                scratch.shared_up_input.ptr,
                up_pairs.ptr,
                self.tensor(f"{up_base}.theta").ptr,
                self.tensor(f"{up_base}.channel_scales").ptr,
                tokens,
                cfg.hidden_size,
                group_size,
                up_krot,
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )

        gate_qweight = self.tensor(f"{gate_base}.qweight_pack8_decode")
        up_qweight = self.tensor(f"{up_base}.qweight_pack8_decode")
        gemv_awq_dual_pack8_transposed_bf16(
            scratch.shared_gate_input.ptr,
            scratch.shared_up_input.ptr,
            gate_qweight.ptr,
            self.tensor(f"{gate_base}.qzeros").ptr,
            self.tensor(f"{gate_base}.scales").ptr,
            up_qweight.ptr,
            self.tensor(f"{up_base}.qzeros").ptr,
            self.tensor(f"{up_base}.scales").ptr,
            scratch.shared_up.ptr,
            tokens,
            cfg.hidden_size,
            _out_packed_from_generic_transposed_qweight(gate_qweight),
            _out_packed_from_generic_transposed_qweight(up_qweight),
            group_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
            runtime=self.runtime,
        )
        silu_mul_dual_rotate_out_bf16(
            scratch.shared_up.ptr,
            down_pairs.ptr,
            self.tensor(f"{down_base}.theta").ptr,
            self.tensor(f"{down_base}.channel_scales").ptr,
            scratch.shared_down_input.ptr,
            tokens,
            cfg.shared_expert_intermediate_size,
            group_size,
            _rotation_krot(down_pairs),
            stream=stream,
            library=_library_for(library, "silu"),
            runtime=self.runtime,
        )
        down_qweight = self.tensor(f"{down_base}.qweight_pack8_decode")
        gemv_awq_pack8_transposed_bf16(
            scratch.shared_down_input.ptr,
            down_qweight.ptr,
            self.tensor(f"{down_base}.qzeros").ptr,
            self.tensor(f"{down_base}.scales").ptr,
            scratch.shared_out.ptr,
            tokens,
            cfg.shared_expert_intermediate_size,
            _out_packed_from_generic_transposed_qweight(down_qweight),
            group_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
            runtime=self.runtime,
        )
        return scratch.shared_out

    def shared_expert_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        if self._shared_expert_is_legacy_w8a16():
            return self.shared_expert_w8a16_bf16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        if self._shared_expert_is_packed_paro_w4():
            return self.shared_expert_paro_w4_bf16(
                hidden,
                scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        raise KeyError(f"no supported shared_expert tensors found under {prefix}")

    def combine_moe_c1_shared_residual_bf16(
        self,
        scratch: Qwen35ParoMoeScratch,
        *,
        shared: Tensor,
        residual: Tensor,
        out: Tensor | None = None,
        tokens: int = 1,
        threads: int = 256,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        target = out or scratch.moe_out
        shared_gate_logits_ptr = scratch.router_logits.ptr + self.config.num_experts * DType.FP32.itemsize
        if tokens == 1:
            weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
                scratch.down_out.ptr,
                scratch.routing_weights.ptr,
                shared.ptr,
                shared_gate_logits_ptr,
                residual.ptr,
                target.ptr,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                threads=threads,
                stream=stream,
                library=_library_for(library, "combine"),
                runtime=self.runtime,
            )
        else:
            weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(
                scratch.down_out.ptr,
                scratch.routing_weights.ptr,
                shared.ptr,
                shared_gate_logits_ptr,
                residual.ptr,
                target.ptr,
                tokens,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.num_experts + 1,
                threads=threads,
                stream=stream,
                library=_library_for(library, "combine"),
                runtime=self.runtime,
            )
        return target

    def run_moe_grouped_compact_bf16(
        self,
        hidden: Tensor,
        residual: Tensor,
        *,
        scratch: Qwen35ParoGroupedMoeScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_moe_grouped_prefill_scratch(tokens=tokens)
        cfg = self.config
        top_k = cfg.num_experts_per_tok
        self.route_moe_topk_shared_bf16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        total_lanes = self._prepare_grouped_moe_prefill_metadata(
            hidden,
            scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        gate_up_pairs = self.tensor(f"{prefix}.gate_up_weight_pairs")
        paro_rotate1_bf16(
            scratch.packed_hidden.ptr,
            scratch.packed_gate_up_input.ptr,
            gate_up_pairs.ptr,
            self.tensor(f"{prefix}.gate_up_weight_theta").ptr,
            self.tensor(f"{prefix}.gate_up_weight_channel_scales").ptr,
            total_lanes,
            cfg.hidden_size,
            group_size,
            _rotation_krot(gate_up_pairs),
            stream=stream,
            library=_library_for(library, "rotate"),
            runtime=self.runtime,
        )
        gate_qweight = self.tensor(f"{prefix}.stacked_gate_qweight_pack8_decode")
        gate_qzeros = self.tensor(f"{prefix}.stacked_gate_qzeros")
        gate_scales = self.tensor(f"{prefix}.stacked_gate_scales")
        up_qweight = self.tensor(f"{prefix}.stacked_up_qweight_pack8_decode")
        up_qzeros = self.tensor(f"{prefix}.stacked_up_qzeros")
        up_scales = self.tensor(f"{prefix}.stacked_up_scales")
        wmma_total_rows = scratch.tile_expert.numel * 16
        gemm_awq_selected_dual_pack8_wmma_compact_bf16(
            scratch.packed_gate_up_input.ptr,
            scratch.expert_start.ptr,
            scratch.wmma_expert_start.ptr,
            scratch.tile_expert.ptr,
            gate_qweight.ptr,
            gate_qzeros.ptr,
            gate_scales.ptr,
            up_qweight.ptr,
            up_qzeros.ptr,
            up_scales.ptr,
            scratch.gate_up.ptr,
            total_lanes,
            cfg.hidden_size,
            _out_packed_from_transposed_qweight(gate_qweight),
            _out_packed_from_transposed_qweight(up_qweight),
            cfg.num_experts,
            group_size,
            wmma_total_rows,
            stream=stream,
            library=_library_for(library, "wmma"),
            runtime=self.runtime,
        )
        pairs = self.tensor(f"{prefix}.down_weight_pairs")
        silu_mul_dual_rotate_out_bf16(
            scratch.gate_up.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.down_weight_theta").ptr,
            self.tensor(f"{prefix}.down_weight_channel_scales").ptr,
            scratch.down_input.ptr,
            total_lanes,
            cfg.moe_intermediate_size,
            group_size,
            _rotation_krot(pairs),
            stream=stream,
            library=_library_for(library, "silu"),
            runtime=self.runtime,
        )
        down_qweight = self.tensor(f"{prefix}.stacked_down_qweight_pack8_decode")
        gemm_awq_selected_pack8_wmma_compact_bf16(
            scratch.down_input.ptr,
            scratch.expert_start.ptr,
            scratch.wmma_expert_start.ptr,
            scratch.tile_expert.ptr,
            down_qweight.ptr,
            self.tensor(f"{prefix}.stacked_down_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_down_scales").ptr,
            scratch.down_out.ptr,
            total_lanes,
            cfg.moe_intermediate_size,
            _out_packed_from_transposed_qweight(down_qweight),
            cfg.num_experts,
            group_size,
            wmma_total_rows,
            stream=stream,
            library=_library_for(library, "wmma"),
            runtime=self.runtime,
        )
        weighted_lanes_sum_out_bf16_f32w(
            scratch.down_out.ptr,
            scratch.sorted_weights.ptr,
            scratch.sorted_lanes.ptr,
            scratch.lane_to_row.ptr,
            scratch.selected_out.ptr,
            tokens,
            top_k,
            cfg.hidden_size,
            stream=stream,
            library=_library_for(library, "combine"),
            runtime=self.runtime,
        )
        shared = self.shared_expert_bf16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        shared_gate_logits_ptr = scratch.router_logits.ptr + cfg.num_experts * DType.FP32.itemsize
        shared_gate_combine_residual_batch_out_bf16(
            scratch.selected_out.ptr,
            shared.ptr,
            shared_gate_logits_ptr,
            residual.ptr,
            scratch.moe_out.ptr,
            tokens,
            cfg.hidden_size,
            cfg.num_experts + 1,
            stream=stream,
            library=_library_for(library, "combine"),
            runtime=self.runtime,
        )
        return scratch.moe_out

    def run_moe_c1_bf16(
        self,
        hidden: Tensor,
        residual: Tensor,
        *,
        scratch: Qwen35ParoMoeScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_moe_c1_scratch(tokens=tokens)
        self.route_moe_topk_shared_bf16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        self.selected_moe_gate_up_pack8_bf16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.activate_rotate_moe_down_bf16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.selected_moe_down_pack8_bf16(scratch.down_input, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        shared = self.shared_expert_bf16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        return self.combine_moe_c1_shared_residual_bf16(
            scratch,
            shared=shared,
            residual=residual,
            tokens=tokens,
            library=library,
            stream=stream,
        )

    def reserve_dense_mlp_scratch(
        self,
        *,
        tokens: int = 1,
        activation_dtype: str | DType = DType.FP16,
    ) -> Qwen35ParoDenseMlpScratch:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        lowp = DType.parse(activation_dtype)
        if lowp not in {DType.BF16, DType.FP16}:
            raise ValueError("activation_dtype must be bf16 or fp16")
        cfg = self.config
        intermediate = cfg.moe_intermediate_size
        if intermediate <= 0:
            raise ValueError("config.moe_intermediate_size must be positive for dense MLP")
        return Qwen35ParoDenseMlpScratch(
            normed=self.workspace.reserve_tensor("dense_mlp.normed", (tokens, cfg.hidden_size), lowp),
            residual=self.workspace.reserve_tensor("dense_mlp.residual", (tokens, cfg.hidden_size), lowp),
            shared_gate_input=self.workspace.reserve_tensor("dense_mlp.gate_input", (tokens, cfg.hidden_size), lowp),
            shared_up_input=self.workspace.reserve_tensor("dense_mlp.up_input", (tokens, cfg.hidden_size), lowp),
            shared_gate_out=self.workspace.reserve_tensor("dense_mlp.gate_out", (tokens, intermediate), lowp),
            shared_up_out=self.workspace.reserve_tensor("dense_mlp.up_out", (tokens, intermediate), lowp),
            shared_up=self.workspace.reserve_tensor("dense_mlp.gate_up", (tokens, 2 * intermediate), lowp),
            shared_intermediate=self.workspace.reserve_tensor("dense_mlp.intermediate", (tokens, intermediate), lowp),
            shared_down_input=self.workspace.reserve_tensor("dense_mlp.down_input", (tokens, intermediate), lowp),
            shared_out=self.workspace.reserve_tensor("dense_mlp.out", (tokens, cfg.hidden_size), lowp),
            shared_zero=self.workspace.reserve_tensor("dense_mlp.zero", (tokens, cfg.hidden_size), lowp),
            gate_logits=self.workspace.reserve_tensor("dense_mlp.gate_logits", (tokens, 1), DType.FP32),
            moe_out=self.workspace.reserve_tensor("dense_mlp.residual_out", (tokens, cfg.hidden_size), lowp),
        )

    def reserve_moe_grouped_prefill_scratch(
        self,
        *,
        tokens: int,
        activation_dtype: str | DType = DType.BF16,
    ) -> Qwen35ParoGroupedMoeScratch:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        lowp = DType.parse(activation_dtype)
        if lowp not in {DType.BF16, DType.FP16}:
            raise ValueError("activation_dtype must be bf16 or fp16")
        cfg = self.config
        top_k = cfg.num_experts_per_tok
        if top_k <= 0:
            raise ValueError("config.num_experts_per_tok must be positive")
        total_lanes = tokens * top_k
        max_wmma_tiles = (total_lanes + 15 * cfg.num_experts + 15) // 16
        return Qwen35ParoGroupedMoeScratch(
            normed=self.workspace.reserve_tensor("moe.grouped.normed", (tokens, cfg.hidden_size), lowp),
            residual=self.workspace.reserve_tensor("moe.grouped.residual", (tokens, cfg.hidden_size), lowp),
            router_logits=self.workspace.reserve_tensor("moe.grouped.router_logits", (tokens, cfg.num_experts + 1), DType.FP32),
            routing_weights=self.workspace.reserve_tensor("moe.grouped.routing_weights", (tokens, top_k), DType.FP32),
            selected_experts=self.workspace.reserve_tensor("moe.grouped.selected_experts", (tokens, top_k), DType.INT64),
            counts=self.workspace.reserve_tensor("moe.grouped.counts", (cfg.num_experts,), DType.INT32),
            padded_counts=self.workspace.reserve_tensor("moe.grouped.padded_counts", (cfg.num_experts,), DType.INT32),
            expert_start=self.workspace.reserve_tensor("moe.grouped.expert_start", (cfg.num_experts + 1,), DType.INT64),
            total_padded=self.workspace.reserve_tensor("moe.grouped.total_padded", (1,), DType.INT64),
            scatter_offsets=self.workspace.reserve_tensor("moe.grouped.scatter_offsets", (cfg.num_experts,), DType.INT32),
            sorted_lanes=self.workspace.reserve_tensor("moe.grouped.sorted_lanes", (total_lanes,), DType.INT64),
            sorted_experts=self.workspace.reserve_tensor("moe.grouped.sorted_experts", (total_lanes,), DType.INT64),
            sorted_weights=self.workspace.reserve_tensor("moe.grouped.sorted_weights", (total_lanes,), DType.FP32),
            lane_to_row=self.workspace.reserve_tensor("moe.grouped.lane_to_row", (total_lanes,), DType.INT64),
            wmma_expert_start=self.workspace.reserve_tensor("moe.grouped.wmma_expert_start", (cfg.num_experts + 1,), DType.INT64),
            tile_expert=self.workspace.reserve_tensor("moe.grouped.tile_expert", (max_wmma_tiles,), DType.INT64),
            wmma_total=self.workspace.reserve_tensor("moe.grouped.wmma_total", (1,), DType.INT64),
            packed_hidden=self.workspace.reserve_tensor("moe.grouped.packed_hidden", (total_lanes, cfg.hidden_size), lowp),
            packed_gate_up_input=self.workspace.reserve_tensor(
                "moe.grouped.packed_gate_up_input",
                (total_lanes, cfg.hidden_size),
                lowp,
            ),
            gate_up=self.workspace.reserve_tensor(
                "moe.grouped.gate_up",
                (total_lanes, 2 * cfg.moe_intermediate_size),
                lowp,
            ),
            down_input=self.workspace.reserve_tensor("moe.grouped.down_input", (total_lanes, cfg.moe_intermediate_size), lowp),
            down_out=self.workspace.reserve_tensor("moe.grouped.down_out", (total_lanes, cfg.hidden_size), lowp),
            selected_out=self.workspace.reserve_tensor("moe.grouped.selected_out", (tokens, cfg.hidden_size), lowp),
            shared_gate_input=self.workspace.reserve_tensor(
                "moe.grouped.shared_gate_input",
                (tokens, cfg.hidden_size),
                lowp,
            ),
            shared_up_input=self.workspace.reserve_tensor(
                "moe.grouped.shared_up_input",
                (tokens, cfg.hidden_size),
                lowp,
            ),
            shared_gate_out=self.workspace.reserve_tensor(
                "moe.grouped.shared_gate_out",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_up_out=self.workspace.reserve_tensor(
                "moe.grouped.shared_up_out",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_up=self.workspace.reserve_tensor(
                "moe.grouped.shared_up",
                (tokens, 2 * cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_intermediate=self.workspace.reserve_tensor(
                "moe.grouped.shared_intermediate",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_down_input=self.workspace.reserve_tensor(
                "moe.grouped.shared_down_input",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_out=self.workspace.reserve_tensor("moe.grouped.shared_out", (tokens, cfg.hidden_size), lowp),
            moe_out=self.workspace.reserve_tensor("moe.grouped.out", (tokens, cfg.hidden_size), lowp),
        )

    def reserve_moe_c1_scratch(
        self,
        *,
        tokens: int = 1,
        activation_dtype: str | DType = DType.BF16,
        prefix: str = "moe",
    ) -> Qwen35ParoMoeScratch:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        lowp = DType.parse(activation_dtype)
        if lowp not in {DType.BF16, DType.FP16}:
            raise ValueError("activation_dtype must be bf16 or fp16")
        cfg = self.config
        top_k = cfg.num_experts_per_tok
        if top_k <= 0:
            raise ValueError("config.num_experts_per_tok must be positive")
        return Qwen35ParoMoeScratch(
            normed=self.workspace.reserve_tensor(f"{prefix}.normed", (tokens, cfg.hidden_size), lowp),
            residual=self.workspace.reserve_tensor(f"{prefix}.residual", (tokens, cfg.hidden_size), lowp),
            gate_up_input=self.workspace.reserve_tensor(f"{prefix}.gate_up_input", (tokens, cfg.hidden_size), lowp),
            router_logits=self.workspace.reserve_tensor(f"{prefix}.router_logits", (tokens, cfg.num_experts + 1), DType.FP32),
            routing_weights=self.workspace.reserve_tensor(f"{prefix}.routing_weights", (tokens, top_k), DType.FP32),
            selected_experts=self.workspace.reserve_tensor(f"{prefix}.selected_experts", (tokens, top_k), DType.INT64),
            gate_up=self.workspace.reserve_tensor(
                f"{prefix}.gate_up",
                (tokens, top_k, 2 * cfg.moe_intermediate_size),
                lowp,
            ),
            down_input=self.workspace.reserve_tensor(f"{prefix}.down_input", (tokens, top_k, cfg.moe_intermediate_size), lowp),
            down_out=self.workspace.reserve_tensor(f"{prefix}.down_out", (tokens, top_k, cfg.hidden_size), lowp),
            shared_gate_input=self.workspace.reserve_tensor(
                f"{prefix}.shared_gate_input",
                (tokens, cfg.hidden_size),
                lowp,
            ),
            shared_up_input=self.workspace.reserve_tensor(
                f"{prefix}.shared_up_input",
                (tokens, cfg.hidden_size),
                lowp,
            ),
            shared_gate_out=self.workspace.reserve_tensor(
                f"{prefix}.shared_gate_out",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_up_out=self.workspace.reserve_tensor(
                f"{prefix}.shared_up_out",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_up=self.workspace.reserve_tensor(
                f"{prefix}.shared_up",
                (tokens, 2 * cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_intermediate=self.workspace.reserve_tensor(
                f"{prefix}.shared_intermediate",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_down_input=self.workspace.reserve_tensor(
                f"{prefix}.shared_down_input",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_out=self.workspace.reserve_tensor(f"{prefix}.shared_out", (tokens, cfg.hidden_size), lowp),
            moe_out=self.workspace.reserve_tensor(f"{prefix}.out", (tokens, cfg.hidden_size), lowp),
        )

    def _memset_tensor(self, tensor: Tensor, *, stream: int, runtime, value: int = 0) -> None:
        nbytes = tensor.numel * tensor.dtype.itemsize
        if stream:
            runtime.memset_async(tensor.ptr, value, nbytes, stream)
        else:
            runtime.memset(tensor.ptr, value, nbytes)

    def free(self) -> None:
        self.workspace.free()
        self.layer_weights.free(runtime=self.runtime)



def _rotate_dual_pack8_fused_enabled() -> bool:
    value = os.environ.get("HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() not in {"0", "false", "off", "no"}



def _full_attn_kv_pack8_fused_enabled() -> bool:
    value = os.environ.get("HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() not in {"0", "false", "off", "no"}



def _router_topk_coop_enabled() -> bool:
    value = os.environ.get("HIPENGINE_PARO_ROUTER_TOPK_COOP")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _env_value(name: str, *aliases: str) -> str | None:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def _env_flag(name: str, default: bool, *aliases: str) -> bool:
    value = _env_value(name, *aliases)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "off", "no"}


def _env_int(name: str, default: int, *aliases: str) -> int:
    value = _env_value(name, *aliases)
    return default if value is None else int(value)


def _full_attention_split_decode_min_context() -> int:
    return max(
        0,
        _env_int(
            "HIPENGINE_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT",
            1024,
            "NANOVLLM_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT",
        ),
    )


def _use_full_attention_split_decode(max_live_count: int) -> bool:
    threshold = _full_attention_split_decode_min_context()
    return threshold > 0 and int(max_live_count) >= threshold


def _requires_full_attention_split_decode(spans: KVLiveSpans) -> bool:
    return spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD or _use_full_attention_split_decode(
        spans.max_live_count
    )


def _paged_attn_gqa_grouped_min_splits() -> int:
    return max(1, _env_int("HIPENGINE_PAGED_ATTN_GQA_GROUPED_MIN_SPLITS", 64))


def _paged_attn_gqa_grouped_min_context() -> int:
    return max(0, _env_int("HIPENGINE_PAGED_ATTN_GQA_GROUPED_MIN_CONTEXT", 4096))


def _paged_attn_gqa_grouped_enabled() -> bool:
    return _env_flag(
        "HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX",
        True,
        "NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX",
    )


def _paged_attn_warp_split_enabled() -> bool:
    return _env_flag(
        "HIPENGINE_PAGED_ATTN_WARP_SPLIT_CTX",
        True,
        "NANOVLLM_AMD_PAGED_ATTN_WARP_SPLIT_CTX",
    )


def _qwen35_gqa_decode_shape(config, *, block_size: int) -> bool:
    return (
        int(block_size) == 256
        and int(config.num_attention_heads) == 16
        and int(config.num_key_value_heads) == 2
        and int(config.head_dim) == 256
    )


def _use_paged_attn_gqa_grouped(max_live_count: int, num_splits: int) -> bool:
    if not _paged_attn_gqa_grouped_enabled():
        return False
    return int(num_splits) >= _paged_attn_gqa_grouped_min_splits() or int(
        max_live_count
    ) >= _paged_attn_gqa_grouped_min_context()


def _full_attention_split_gate_bf16_fn(config, *, block_size: int, num_splits: int, max_live_count: int):
    if _qwen35_gqa_decode_shape(config, block_size=block_size):
        if _use_paged_attn_gqa_grouped(max_live_count, num_splits):
            return qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans
        if _paged_attn_warp_split_enabled():
            return qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans
    return qwen35_paged_full_attn_decode_split_k_gate_bf16_spans


def _full_attention_split_gate_fp16_fn(config, *, block_size: int, num_splits: int, max_live_count: int):
    if _qwen35_gqa_decode_shape(config, block_size=block_size):
        if _use_paged_attn_gqa_grouped(max_live_count, num_splits):
            return qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans
        if _paged_attn_warp_split_enabled():
            return qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans
    return qwen35_paged_full_attn_decode_split_k_gate_fp16_spans


def _moe_prefill_compact_wmma_min_tokens() -> int:
    value = os.environ.get("HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS")
    if value is None or value.strip() == "":
        return 2
    return max(2, int(value))


def _use_moe_grouped_compact_prefill(tokens: int) -> bool:
    return tokens > 1 and tokens >= _moe_prefill_compact_wmma_min_tokens()


def _linear_ab_prefill_rocblas_min_tokens() -> int:
    value = os.environ.get("HIPENGINE_LINEAR_AB_PREFILL_ROCBLAS_MIN_TOKENS")
    if value is None or value.strip() == "":
        return 0
    return max(0, int(value))


def _use_linear_ab_prefill_rocblas(tokens: int) -> bool:
    threshold = _linear_ab_prefill_rocblas_min_tokens()
    return threshold > 0 and tokens >= threshold


def _use_linear_gdn_prefill_rotate_fused(config, *, tokens: int, group_size: int) -> bool:
    return (
        tokens > 1
        and _env_flag("HIPENGINE_LINEAR_GDN_PREFILL_ROTATE_FUSED", False)
        and int(group_size) == int(config.linear_value_head_dim)
    )


def _use_prefill_router_shared_gate_sigmoid_fused(*, tokens: int, legacy_shared: bool) -> bool:
    return (
        tokens > 1
        and legacy_shared
        and _env_flag("HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED", False)
    )


def _shared_gate_up_prefill_token_tile() -> int:
    value = os.environ.get("HIPENGINE_SHARED_GATE_UP_PREFILL_TOKEN_TILE")
    if value is None or value.strip() == "":
        return 2
    tile = int(value)
    if tile not in (0, 2, 4):
        raise ValueError("HIPENGINE_SHARED_GATE_UP_PREFILL_TOKEN_TILE must be 0, 2, or 4")
    return tile


def _shared_gate_up_prefill_min_tokens() -> int:
    value = os.environ.get("HIPENGINE_SHARED_GATE_UP_PREFILL_MIN_TOKENS")
    if value is None or value.strip() == "":
        return 1024
    return max(2, int(value))


def _use_shared_gate_up_prefill_token_tiled(tokens: int) -> int:
    tile = _shared_gate_up_prefill_token_tile()
    return tile if tile > 0 and tokens >= _shared_gate_up_prefill_min_tokens() else 0


def _shared_down_combine_prefill_token_tile() -> int:
    value = os.environ.get("HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE")
    if value is None or value.strip() == "":
        return 2
    tile = int(value)
    if tile not in (0, 2, 4):
        raise ValueError("HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE must be 0, 2, or 4")
    return tile


def _shared_down_combine_prefill_min_tokens() -> int:
    value = os.environ.get("HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_MIN_TOKENS")
    if value is None or value.strip() == "":
        return 2
    return max(2, int(value))


def _use_shared_down_combine_prefill_token_tiled(tokens: int) -> int:
    tile = _shared_down_combine_prefill_token_tile()
    return tile if tile > 0 and tokens >= _shared_down_combine_prefill_min_tokens() else 0


def _library_for(library, family: str):
    if isinstance(library, dict):
        return library.get(family)
    return library


def _check_positive(value: int, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _out_packed_from_strided_qweight(qweight: Tensor) -> int:
    if len(qweight.shape) < 2:
        raise ValueError("strided qweight must have at least two dimensions")
    return qweight.shape[-1]


def _out_packed_from_transposed_qweight(qweight: Tensor) -> int:
    if len(qweight.shape) < 3:
        raise ValueError("transposed stacked qweight must have shape [experts, out_packed, in_features]")
    return qweight.shape[1]


def _out_packed_from_generic_transposed_qweight(qweight: Tensor) -> int:
    if len(qweight.shape) != 2:
        raise ValueError("generic transposed qweight must have shape [out_packed, in_features]")
    return qweight.shape[0]


def _out_packed_from_marlin_qweight(qweight: Tensor) -> int:
    if len(qweight.shape) != 3 or qweight.shape[-1] != 128:
        raise ValueError("Marlin-K qweight must have shape [out_packed, groups, 128]")
    return qweight.shape[0]


def _linear_value_width(config) -> int:
    return int(config.linear_num_value_heads) * int(config.linear_value_head_dim)


def _linear_qkv_width(config) -> int:
    return 2 * int(config.linear_num_key_heads) * int(config.linear_key_head_dim) + _linear_value_width(config)


def _rotation_krot(pairs: Tensor) -> int:
    if not pairs.shape:
        raise ValueError("rotation pairs tensor must have at least one dimension")
    return pairs.shape[0]
