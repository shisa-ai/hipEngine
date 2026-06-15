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
    PagedAttnPrefillKind,
    PagedKVWriteKind,
    resolve_paged_attn_decode,
    resolve_paged_attn_prefill,
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
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans,
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
    silu_mul_separate_out_fp16,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    dense_dual_gemv_out_bf16,
    dense_dual_gemv_out_fp16,
    dense_dual_gemv_separate_out_fp16,
    dense_gemv_out_bf16,
    dense_gemv_out_fp16,
    dense_gemv_out_fp16_wmma,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_chain_conv_decode_fp16_tloop,
    qwen35_linear_attn_conv_decode_bf16,
    qwen35_linear_attn_conv_decode_fp16,
    qwen35_linear_attn_conv_prefill_f32,
    qwen35_linear_attn_conv_prefill_fp16,
    qwen35_linear_attn_conv_prefill_segments_f32,
    qwen35_linear_attn_tree_conv_decode_fp16_tloop,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_recurrent_segments_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_gdn_prefill_rmsnorm_gate_fp16,
    qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16,
    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16,
    qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16,
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
from hipengine.kernels.hip_gfx1100.quant.paro_marlin_k import (
    gemv_paro_marlin_k_fma_fp16,
    gemv_paro_marlin_k_fma_multi_row_fp16,
    marlin_k_default_threads,
)
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
    awq_fusedw4_prefill_dual_fp16,
    awq_fusedw4_prefill_fp16,
    awq_fusedw4_prefill_strided_fp16,
    gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16,
    gemv_awq_dual_pack8_multi_row_split_transposed_fp16,
    gemv_awq_pack8_multi_row_decode_transposed_fp16,
    gemv_awq_pack8_multi_row_strided_fp16,
    gemv_awq_pack8_multi_row_transposed_fp16,
    gemv_awq_dual_pack8_transposed_bf16,
    gemv_awq_dual_pack8_transposed_fp16,
    gemv_awq_dual_pack8_transposed_rotate_staged_bf16,
    gemv_awq_dual_pack8_transposed_rotate_staged_fp16,
    gemv_awq_dual_pack8_transposed_rotate_staged_keyed_fp16,
    gemv_awq_pack8_output_tiled_bf16,
    gemv_awq_pack8_output_tiled_fp16,
    gemv_awq_dual_pack8_output_tiled_split_transposed_fp16,
    gemv_awq_dual_pack8_output_tiled_transposed_fp16,
    gemv_awq_pack8_output_tiled_transposed_bf16,
    gemv_awq_pack8_output_tiled_transposed_fp16,
    gemv_awq_pack8_strided_bf16,
    gemv_awq_pack8_strided_fp16,
    gemv_awq_pack8_transposed_bf16,
    gemv_awq_pack8_transposed_fp16,
    gemv_awq_selected_dual_pack8_transposed_bf16,
    gemv_awq_selected_dual_pack8_transposed_fp16,
    gemv_awq_selected_dual_pack8_transposed_rotate_out_fp16,
    gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16,
    gemv_awq_selected_pack8_transposed_bf16,
    gemv_awq_selected_pack8_transposed_fp16,
    gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16,
)
from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import (
    paro_rmsnorm_rotate2_fp16,
    paro_rotate1_bf16,
    paro_rotate1_bf16_gate_fp16,
    paro_rotate1_f32_to_fp16,
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
    qwen35_split_qgate_fp16_key_f32,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.qwen35_paro import Qwen35ParoLayerDeviceWeights, normalize_qwen35_weight_name
from hipengine.runtime.moe_c1_dispatch import moe_c1_c_dispatch_enabled
from hipengine.runtime.workspace import RuntimeWorkspace


_SHARED_ROTATE_FUSE_BARRIER_STATE: dict[int, tuple[int, int]] = {}

# When true, keyed staged-rotate barriers fall back to memset-per-launch
# (zero the barrier on-stream, target=blocks, epoch=1). Cumulative host
# epochs are baked by-value into HIP-graph captures, so replays skip the
# producer/consumer sync (race) and later direct launches spin forever on
# epochs the device never reaches (#107). Memset-per-launch is stream-ordered
# and capture-safe: the memset is recorded into the graph with the kernel.
_SHARED_ROTATE_FUSE_BARRIER_MEMSET_MODE = False


def _set_shared_rotate_fuse_barrier_memset_mode(enabled: bool) -> None:
    """Toggle capture-safe memset-per-launch keyed barriers (verify graph mode)."""

    global _SHARED_ROTATE_FUSE_BARRIER_MEMSET_MODE
    _SHARED_ROTATE_FUSE_BARRIER_MEMSET_MODE = bool(enabled)


def _reset_shared_rotate_fuse_barrier_state() -> None:
    """Clear process-local keyed barrier counters for a new resident session."""

    _SHARED_ROTATE_FUSE_BARRIER_STATE.clear()
_PAGED_KV_REGISTRY_BACKEND = "hip_gfx1100"

# C3.0c: row counts for which the output-column-tiled pack8 GEMV is instantiated
# (templated C in the kernel). Other c>1 row counts fall back to the per-row
# strided GEMV. Byte-exact equivalence is gated in
# tests/test_paro_awq_output_tiled_gemv.py.
_PACK8_OUTPUT_TILED_ROWS = (
    frozenset()
    if os.environ.get("HIPENGINE_DISABLE_PACK8_OUTPUT_TILED")
    else frozenset({2, 4, 8})
)


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
    tree_conv_state: Tensor
    tree_recurrent_state: Tensor
    tree_gdn_acc: Tensor


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
    # M13.B.2: barrier int32[2] for fused shared-expert rotate+dual GEMV.
    shared_rotate_fuse_barrier: Tensor


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
    # M13.B.2: barrier int32[2] for fused shared-expert rotate+dual GEMV.
    shared_rotate_fuse_barrier: Tensor


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
        # M14.fuse.barrier: per-barrier cumulative (rotate_count, ready_epoch)
        # for keyed HBM-staged rotate+dual-GEMV launches.  This is module-global
        # because verifier layers can pass a scratch barrier owned by the
        # runner's prefill workspace rather than by the layer runtime's own
        # workspace.  Qwen35ParoResidentSession resets it at session start.
        self._shared_rotate_fuse_barrier_state = _SHARED_ROTATE_FUSE_BARRIER_STATE
        # M14.dispatch.1-beta: lazy per-layer cache for the C-side MoE C1
        # dispatcher.  Key: layer_kind ('linear_attention' | 'full_attention').
        # Populated on first matching call from run_moe_c1_fp16.
        self._moe_c1_dispatch_cache: object | None = None  # MoeC1DispatchCache
        self._tensor_lookup_cache_enabled = _weight_tensor_lookup_cache_enabled()
        self._tensor_lookup_cache: dict[str, Tensor] = {}
        shared_prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        tensors = self.layer_weights.weights.tensors
        if normalize_qwen35_weight_name(f"{shared_prefix}.gate_up_weight_w8a16") in tensors:
            self._shared_expert_kind = "legacy_w8a16"
        elif normalize_qwen35_weight_name(f"{shared_prefix}.gate_proj.qweight_pack8_decode") in tensors:
            self._shared_expert_kind = "packed_paro_w4"
        else:
            self._shared_expert_kind = None

    @property
    def config(self):
        return self.layer_weights.config

    def tensor(self, name: str) -> Tensor:
        if self._tensor_lookup_cache_enabled:
            cached = self._tensor_lookup_cache.get(name)
            if cached is not None:
                return cached
        normalized = normalize_qwen35_weight_name(name)
        tensor = self.layer_weights.weights.tensors[normalized].tensor
        if self._tensor_lookup_cache_enabled:
            self._tensor_lookup_cache[name] = tensor
        return tensor

    def has_tensor(self, name: str) -> bool:
        return normalize_qwen35_weight_name(name) in self.layer_weights.weights.tensors

    def _shared_expert_is_legacy_w8a16(self) -> bool:
        return self._shared_expert_kind == "legacy_w8a16"

    def _shared_expert_is_packed_paro_w4(self) -> bool:
        return self._shared_expert_kind == "packed_paro_w4"

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
            # C3.0c: output-column-tiled GEMV amortizes the weight load across all
            # c active columns for c in {2,4,8}; byte-exact vs the per-row strided
            # kernel (tests/test_paro_awq_output_tiled_gemv.py).
            pack8_fn = (
                gemv_awq_pack8_output_tiled_bf16
                if rows in _PACK8_OUTPUT_TILED_ROWS
                else gemv_awq_pack8_strided_bf16
            )
            pack8_fn(
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
            pack8_t_fn = (
                gemv_awq_pack8_output_tiled_transposed_bf16
                if rows in _PACK8_OUTPUT_TILED_ROWS
                else gemv_awq_pack8_transposed_bf16
            )
            pack8_t_fn(
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
        if (
            rows == 1
            or force_gemv
            or (rows > 1 and rows <= 8 and _marlin_k_multi_row_site_enabled(prefix))
        ) and self.has_tensor(f"{prefix}.qweight_mk"):
            # The Marlin-K GEMV kernel has a row grid and matches token-1 output
            # projection numerics for c>N diagnostic GEMV paths.  Keep normal
            # rows>1 prefill on the fused prefill kernels unless force_gemv asks
            # for a decode-style row-aware GEMV projection.
            qweight_mk = self.tensor(f"{prefix}.qweight_mk")
            out_packed = _out_packed_from_marlin_qweight(qweight_mk)
            marlin_args = (
                x.ptr,
                qweight_mk.ptr,
                self.tensor(f"{prefix}.qzeros_mk").ptr,
                self.tensor(f"{prefix}.scales_mk").ptr,
                out.ptr,
                rows,
                width,
                out_packed,
                group_size,
            )
            if rows == 1:
                gemv_paro_marlin_k_fma_fp16(
                    *marlin_args,
                    threads=marlin_k_default_threads(width, out_packed * 8),
                    stream=stream,
                    library=_library_for(library, "marlin_k"),
                    runtime=self.runtime,
                )
            else:
                # M15.2: weight-amortize across the B+1 verifier rows while
                # preserving the single-row Marlin-K per-row accumulation order,
                # so each row is bit-identical to AR's rows==1 Marlin-K output.
                gemv_paro_marlin_k_fma_multi_row_fp16(
                    *marlin_args,
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
            # M7.C investigation (2026-05-21): bumping this to
            # ``rows > _small_batch_decode_threshold()`` is functionally correct in
            # isolation (the single-output GEMV has no row-stride aliasing) but on
            # gfx1151 the resulting cache footprint shifts the downstream MoE/GDN
            # kernels by ~+3 ms per pass, wiping out the local -0.44 ms saving.
            # Re-enable once M7.C.6 lands the safe path for sites #1/#2 below so the
            # net reach justifies the secondary kernel-cache cost.
            if (
                rows > 1
                and rows <= 8
                and not force_gemv
                and _w4_multi_row_single_site_enabled(prefix)
                and group_size % 16 == 0
                and width % group_size == 0
            ):
                # M12.6: weight-sharing multi-row pack8 for B+1 <= 8 verifier rows.
                # See gemv_awq_pack8_multi_row_kernel in paro_awq_gemv.hip.
                gemv_awq_pack8_multi_row_strided_fp16(
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
            elif (
                rows > 1
                and not force_gemv
                and group_size % 16 == 0
                and width % group_size == 0
                and not (_w4_output_tiled_prefill_enabled() and rows in _PACK8_OUTPUT_TILED_ROWS)
            ):
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
                pack8_fn = (
                    gemv_awq_pack8_output_tiled_fp16
                    if rows in _PACK8_OUTPUT_TILED_ROWS
                    else gemv_awq_pack8_strided_fp16
                )
                pack8_fn(
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
            # M7.C investigation (2026-05-21): bumping this to
            # ``rows > _small_batch_decode_threshold()`` is functionally correct in
            # isolation but the resulting cache shift adds ~+3 ms in downstream
            # MoE/GDN kernels.  Re-enable once M7.C.6 unlocks the dual-GEMV reach.
            if (
                rows > 1
                and rows <= 8
                and not force_gemv
                and _w4_multi_row_single_site_enabled(prefix)
                and group_size % 16 == 0
                and width % group_size == 0
            ):
                gemv_awq_pack8_multi_row_transposed_fp16(
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
            elif (
                rows > 1
                and not force_gemv
                and group_size % 16 == 0
                and width % group_size == 0
                and not (_w4_output_tiled_prefill_enabled() and rows in _PACK8_OUTPUT_TILED_ROWS)
            ):
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
                pack8_t_fn = (
                    gemv_awq_pack8_output_tiled_transposed_fp16
                    if rows in _PACK8_OUTPUT_TILED_ROWS
                    else gemv_awq_pack8_transposed_fp16
                )
                pack8_t_fn(
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
        include_tree_state: bool = True,
    ) -> Qwen35ParoLinearAttentionScratch:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        lowp = DType.parse(activation_dtype)
        if lowp not in {DType.BF16, DType.FP16}:
            raise ValueError("activation_dtype must be bf16 or fp16")
        cfg = self.config
        qkv_width = _linear_qkv_width(cfg)
        z_width = _linear_value_width(cfg)
        tree_rows = tokens if include_tree_state else 1
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
            tree_conv_state=self.workspace.reserve_tensor(
                "linear_attn.tree_conv_state",
                (tree_rows, qkv_width, cfg.linear_conv_kernel_dim),
                DType.FP32,
            ),
            tree_recurrent_state=self.workspace.reserve_tensor(
                "linear_attn.tree_recurrent_state",
                (tree_rows, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim),
                DType.FP32,
            ),
            tree_gdn_acc=self.workspace.reserve_tensor("linear_attn.tree_gdn_acc", (tree_rows, z_width), DType.FP32),
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
        # M7.C.6: small-batch path mirrors the bf16 sibling at
        # ``project_linear_attention_qkv_z_bf16`` line 1075+.  The dual GEMV
        # writes row-major [q,k] per token into the combined ``q_proj_key``
        # buffer; ``q_proj`` and ``key_bf16`` are views with strides that do
        # NOT match that layout at tokens > 1.  Split into two single GEMVs
        # writing the views' backing memory directly (matches
        # ``awq_fusedw4_prefill_dual_fp16``'s two-output ABI without paying for
        # the prefill-tuned kernel at small batch).
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
                kv_dual_fn = (
                    gemv_awq_dual_pack8_output_tiled_transposed_fp16
                    if tokens in _PACK8_OUTPUT_TILED_ROWS
                    else gemv_awq_dual_pack8_transposed_fp16
                )
                kv_dual_fn(
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
                qk_dual_fn = (
                    gemv_awq_dual_pack8_output_tiled_transposed_fp16
                    if tokens in _PACK8_OUTPUT_TILED_ROWS
                    else gemv_awq_dual_pack8_transposed_fp16
                )
                qk_dual_fn(
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
        elif tokens <= _small_batch_decode_threshold():
            # M7.C.6: two single GEMVs, one for Q (writes scratch.q_proj.ptr)
            # and one for K (writes scratch.key_bf16.ptr).  Mirrors the bf16
            # sibling at project_linear_attention_qkv_z_bf16 line 1090+ — the
            # views' contiguous strides match the single-GEMV row strides, so
            # the downstream qwen35_split_qgate / attention path reads correct
            # rows.  scratch.q_proj_key (the combined view) is intentionally
            # left inconsistent here; nothing downstream reads it directly.
            # M15.1/M15.3: weight-amortize across the B+1 verifier rows via the
            # bit-exact multi-row decode kernel; M15.3 fuses the q+k pair into
            # one split-dual launch (bit-identical to two decode singles).
            if _w4_multi_row_small_batch_site_enabled("full_qk"):
                gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16(
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
            else:
                gemv_awq_pack8_transposed_fp16(
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
                gemv_awq_pack8_transposed_fp16(
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
        elif _w4_multi_row_dual_site_eligible("full_qk", tokens, scratch.q_rot.shape[-1], group_size):
            # M12.6: weight-sharing multi-row dual W4 GEMV for small verifier batches.
            gemv_awq_dual_pack8_multi_row_split_transposed_fp16(
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
        if _full_qkv_split_key_fused_enabled(tokens):
            qwen35_split_qgate_fp16_key_f32(
                scratch.q_proj.ptr,
                scratch.key_bf16.ptr,
                scratch.query_raw.ptr,
                scratch.key_raw.ptr,
                scratch.gate.ptr,
                tokens,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                stream=stream,
                library=_library_for(library, "qwen_rotary"),
                runtime=self.runtime,
            )
        else:
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
            fp16_to_f32(
                scratch.key_bf16.ptr,
                scratch.key_raw.ptr,
                tokens * kv_width,
                stream=stream,
                library=_library_for(library, "cast"),
                runtime=self.runtime,
            )
        if producer_trace is not None:
            producer_trace("query_raw_after_split", scratch.query_raw)
            producer_trace("gate_after_split", scratch.gate)
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
            tree_conv_state=self._row_tensor_view(scratch.tree_conv_state, row),
            tree_recurrent_state=self._row_tensor_view(scratch.tree_recurrent_state, row),
            tree_gdn_acc=self._row_tensor_view(scratch.tree_gdn_acc, row),
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
            tree_conv_state=self._row_tensor_view(scratch.tree_conv_state, row),
            tree_recurrent_state=self._row_tensor_view(scratch.tree_recurrent_state, row),
            tree_gdn_acc=self._row_tensor_view(scratch.tree_gdn_acc, row),
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

    def project_linear_attention_decode_rows_qkv_z_subset_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        copy_qkv: bool,
        copy_z: bool,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Replay selected QKV/Z projection outputs with token-1 rows."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if not copy_qkv and not copy_z:
            return scratch.qkv, scratch.z
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
            copies: list[tuple[Tensor, Tensor]] = []
            if copy_qkv:
                copies.append((self._row_tensor_view(scratch.qkv, row), row_scratch.qkv))
            if copy_z:
                copies.append((self._row_tensor_view(scratch.z, row), row_scratch.z))
            for dst, src in copies:
                runtime.memcpy_async(
                    dst.ptr,
                    src.ptr,
                    src.numel * src.dtype.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
        return scratch.qkv, scratch.z

    def rotate_linear_attention_decode_rows_qkv_z_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoLinearAttentionScratch,
        *,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Replay only the QKV/Z rotary-input stage with token-1 rows."""

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
            for dst, src in (
                (self._row_tensor_view(scratch.qkv_rot, row), row_scratch.qkv_rot),
                (self._row_tensor_view(scratch.z_rot, row), row_scratch.z_rot),
            ):
                runtime.memcpy_async(
                    dst.ptr,
                    src.ptr,
                    src.numel * src.dtype.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
        return scratch.qkv_rot, scratch.z_rot

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

    def prefill_full_attention_int8_gqa_gate_fp16(
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
        if key_cache.dtype is not DType.INT8 or value_cache.dtype is not DType.INT8:
            raise ValueError("INT8 paged attention prefill requires INT8 key/value cache tensors")
        metadata = spans.scale_metadata
        if metadata is None:
            raise ValueError("INT8 paged attention prefill requires scale metadata")
        gate_tensor = scratch.gate if gate is None else gate
        prefill_fn = resolve_paged_attn_prefill(
            backend=_PAGED_KV_REGISTRY_BACKEND,
            spans=spans,
            kind=PagedAttnPrefillKind.GQA_GATE_FP16,
        )
        prefill_fn(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            metadata.k_scale.ptr,
            metadata.v_scale.ptr,
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

    def prefill_full_attention_gqa_gate_tree_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        rows: int,
        ancestor_mask: Tensor,
        tree_committed_count_ptr: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Tree-aware variant of ``prefill_full_attention_gqa_gate_fp16``.

        ``ancestor_mask`` is a ``[rows, rows]`` ``DType.UINT8`` tensor where
        ``ancestor_mask[i, j] == 1`` iff verifier row ``j`` is an ancestor of
        verifier row ``i`` (a row is its own ancestor).  Committed-context
        positions in ``[0, tree_committed_count)`` are visible to every row;
        the mask only constrains the verifier-row K/V block at
        ``[tree_committed_count, tree_committed_count + rows)``.
        """

        if ancestor_mask.dtype not in {DType.BOOL, DType.INT8} or ancestor_mask.shape != (rows, rows):
            raise ValueError("ancestor_mask must be a 1-byte tensor (BOOL or INT8) with shape (rows, rows)")
        if tree_committed_count_ptr == 0:
            raise ValueError("tree_committed_count_ptr must be a device int64 scalar")
        gate_tensor = scratch.gate if gate is None else gate
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            spans,
            ancestor_mask.ptr,
            tree_committed_count_ptr,
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
        force_paged_context: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        gate_tensor = scratch.gate if gate is None else gate
        if force_paged_context:
            # Per-row diagnostics after a batch KV append must honor the same
            # KVLiveSpans block-table addressing as the native batch context
            # kernel; the dense token-1 context kernel assumes a compact row
            # cache and is kept only for the normal c=1 path.
            if spans.storage_dtype != DType.BF16:
                raise NotImplementedError("forced paged full-attention context diagnostic currently requires BF16 KV")
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
        elif spans.max_live_count < 1024:
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

    def decode_full_attention_gqa_gate_fp16_batch(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        rows: int,
        partial_out: Tensor,
        partial_m: Tensor,
        partial_l: Tensor,
        chunk_size: int,
        num_splits: int,
        gate: Tensor | None = None,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run small-B row-batched Qwen3.5 decode attention with FP16 gate."""

        gate_tensor = scratch.gate if gate is None else gate
        if num_splits == 1 and _env_flag("HIPENGINE_QWEN35_DECODE_BATCHED_DIRECT_GATE", True):
            qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans(
                scratch.query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                gate_tensor.ptr,
                scratch.gated_attn.ptr,
                spans,
                rows,
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
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            partial_out.ptr,
            partial_m.ptr,
            partial_l.ptr,
            spans,
            rows,
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
        out: Tensor | None = None,
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
        # M13.B.0: ``out`` forwards into the final MoE combine so the per-row
        # next_hidden D2D copy in ``_run_full_attention_chain_c1_loop`` becomes
        # a no-op when the caller passes ``out=row_out``.
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
                out=out,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_c1_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            out=out,
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
        force_per_row_layer_batch_scratch: bool = False,
        force_per_row_attention_batch_moe: bool = False,
        force_per_row_attention_batch_post_moe: bool = False,
        force_per_row_attention_batch_o_post_moe: bool = False,
        force_per_row_preqkv_append_batch_context_o_post_moe: bool = False,
        force_per_row_preqkv_append_context_batch_gate_o_post_moe: bool = False,
        force_per_row_preqkv_append_context_gate_batch_o_post_moe: bool = False,
        force_per_row_context: bool = False,
        force_per_row_context_only: bool = False,
        force_per_row_dense_context_only: bool = False,
        force_per_row_dense_context_batch_gate: bool = False,
        force_per_row_paged_context_only: bool = False,
        force_batch_temp_context: bool = False,
        force_batch_compact_context: bool = False,
        force_per_row_gate: bool = False,
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
        elif tokens > 1 and not (
            force_selected_c1_moe
            or force_per_row_moe
            or force_per_row_layer_scratch
            or force_per_row_layer_batch_scratch
        ):
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
        if (
            force_per_row_attention_batch_moe
            or force_per_row_attention_batch_post_moe
            or force_per_row_attention_batch_o_post_moe
            or force_per_row_preqkv_append_batch_context_o_post_moe
            or force_per_row_preqkv_append_context_batch_gate_o_post_moe
            or force_per_row_preqkv_append_context_gate_batch_o_post_moe
        ) and tokens > 1:
            if per_row_append_contexts is None or len(per_row_append_contexts) != tokens:
                raise ValueError("per_row_append_contexts must provide one key/value/span tuple per decode row")
            if per_row_contexts is None or len(per_row_contexts) != tokens:
                raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
            if dense_mlp:
                raise NotImplementedError("per-row attention / batch-MoE diagnostic is currently wired for MoE layers")
            if not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
                raise ValueError("per-row attention / batch-MoE diagnostic requires grouped MoE scratch")
            for row, ((row_key_cache, row_value_cache, row_append_spans), row_context_tuple) in enumerate(
                zip(per_row_append_contexts, per_row_contexts, strict=True)
            ):
                context_key_cache, context_value_cache, row_decode_spans = row_context_tuple
                if context_key_cache.ptr != row_key_cache.ptr or context_value_cache.ptr != row_value_cache.ptr:
                    raise ValueError("per-row attention / batch-MoE diagnostics must use matching row cache views")
                row_hidden = self._row_tensor_view(hidden, row)
                row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
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

                self.input_rmsnorm_fp16(row_hidden, row_scratch.attn_input, tokens=1, library=library, stream=stream)
                self.rotate_full_attention_inputs_fp16(
                    row_scratch.attn_input,
                    row_scratch,
                    tokens=1,
                    group_size=group_size,
                    library=library,
                    stream=stream,
                )
                row_input_scratch_trace("attn_input_after_rotate", row_scratch)
                self.project_full_attention_qkv_fp16(
                    row_scratch,
                    tokens=1,
                    group_size=group_size,
                    producer_trace=row_qkv_tensor_trace if qkv_tensor_trace is not None else None,
                    library=library,
                    stream=stream,
                )
                row_input_scratch_trace("attn_input_after_project", row_scratch)
                _row_query, _row_key, _row_value, row_gate = self.prepare_full_attention_qkv_fp16(
                    row_scratch,
                    cos_table=cos_table,
                    sin_table=sin_table,
                    position=row_position,
                    max_positions=max_positions,
                    tokens=1,
                    producer_trace=row_qkv_tensor_trace if qkv_tensor_trace is not None else None,
                    library=library,
                    stream=stream,
                )
                row_input_scratch_trace("attn_input_after_prepare", row_scratch)
                self.append_full_attention_kv_fp16(
                    row_scratch,
                    key_cache=row_key_cache,
                    value_cache=row_value_cache,
                    spans=row_append_spans,
                    block_size=block_size,
                    library=library,
                    stream=stream,
                )
                if not force_per_row_preqkv_append_batch_context_o_post_moe:
                    if not _requires_full_attention_split_decode(row_decode_spans):
                        row_gated = self.decode_full_attention_context_gate_fp16(
                            row_scratch,
                            key_cache=row_key_cache,
                            value_cache=row_value_cache,
                            spans=row_decode_spans,
                            gate=row_gate,
                            block_size=block_size,
                            library=library,
                            stream=stream,
                        )
                    else:
                        raise NotImplementedError(
                            "per-row attention / batch-MoE diagnostic does not cover split-K full-attention decode"
                        )
                    if force_per_row_preqkv_append_context_batch_gate_o_post_moe:
                        row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                        (self.runtime or get_hip_runtime()).memcpy_async(
                            row_context.ptr,
                            attention_scratch.attn_out.ptr,
                            self.config.num_attention_heads * self.config.head_dim * DType.FP32.itemsize,
                            HipMemcpyKind.DEVICE_TO_DEVICE,
                            stream,
                        )
                    elif not (
                        force_per_row_attention_batch_o_post_moe
                        or force_per_row_preqkv_append_context_gate_batch_o_post_moe
                    ):
                        self.project_full_attention_o_fp16(
                            row_gated,
                            row_scratch,
                            tokens=1,
                            group_size=group_size,
                            library=library,
                            stream=stream,
                        )
            if force_per_row_preqkv_append_batch_context_o_post_moe:
                if _requires_full_attention_split_decode(decode_spans):
                    raise NotImplementedError(
                        "per-row pre-QKV/append + batch context diagnostic does not cover split-K full-attention decode"
                    )
                gated = self.decode_full_attention_context_gate_fp16_batch(
                    attention_scratch,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    spans=decode_spans,
                    rows=tokens,
                    gate=attention_scratch.gate,
                    block_size=block_size,
                    library=library,
                    stream=stream,
                )
                attn_out = self.project_full_attention_o_fp16(
                    gated,
                    attention_scratch,
                    tokens=tokens,
                    group_size=group_size,
                    force_pack8_gemv=force_batch_gemv_output,
                    library=library,
                    stream=stream,
                )
            elif force_per_row_preqkv_append_context_batch_gate_o_post_moe:
                qwen35_full_attn_gate_mul_fp16(
                    attention_scratch.query_raw.ptr,
                    attention_scratch.gate.ptr,
                    attention_scratch.gated_attn.ptr,
                    tokens * self.config.num_attention_heads * self.config.head_dim,
                    stream=stream,
                    library=_library_for(library, "attention"),
                    runtime=self.runtime,
                )
                attn_out = self.project_full_attention_o_fp16(
                    attention_scratch.gated_attn,
                    attention_scratch,
                    tokens=tokens,
                    group_size=group_size,
                    force_pack8_gemv=force_batch_gemv_output,
                    library=library,
                    stream=stream,
                )
            else:
                attn_out = (
                    self.project_full_attention_o_fp16(
                        attention_scratch.gated_attn,
                        attention_scratch,
                        tokens=tokens,
                        group_size=group_size,
                        force_pack8_gemv=force_batch_gemv_output,
                        library=library,
                        stream=stream,
                    )
                    if (
                        force_per_row_attention_batch_o_post_moe
                        or force_per_row_preqkv_append_context_gate_batch_o_post_moe
                    )
                    else attention_scratch.o_proj
                )
            post_attention_fn = (
                self.post_attention_add_rmsnorm_fp16
                if (
                    force_per_row_attention_batch_post_moe
                    or force_per_row_attention_batch_o_post_moe
                    or force_per_row_preqkv_append_batch_context_o_post_moe
                    or force_per_row_preqkv_append_context_batch_gate_o_post_moe
                    or force_per_row_preqkv_append_context_gate_batch_o_post_moe
                )
                else self.post_attention_add_rmsnorm_fp16_per_row
            )
            mlp_input, residual = post_attention_fn(
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
        if (force_per_row_layer_scratch or force_per_row_layer_batch_scratch) and tokens > 1:
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
                row_scratch = (
                    self._decode_row_full_attention_scratch(attention_scratch, row)
                    if force_per_row_layer_batch_scratch
                    else self._decode_row_full_attention_temp_scratch(attention_scratch)
                )
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
        if force_per_row_append_context and not (force_per_row_kv_append and force_per_row_context):
            raise ValueError("per-row append+context interleave requires per-row KV append and context diagnostics")
        if force_per_row_append_context:
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
                # c2 needs paged per-row context plus the batch gate replay to
                # avoid the historical row0 [103,137] diagnostic signature.
                # For rows>2, c4 evidence shows row-local context+gate remains
                # the stable diagnostic while batch-gating replayed contexts can
                # reproduce the native row3 failure.
                if tokens == 2:
                    for row, (row_key_cache, row_value_cache, row_decode_spans) in enumerate(per_row_contexts):
                        row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
                        if row_decode_spans.storage_dtype != DType.BF16:
                            raise NotImplementedError("per-row full-attention context diagnostic currently requires BF16 KV")
                        qwen35_paged_full_attn_decode_context_bf16_spans(
                            row_scratch.query.ptr,
                            row_key_cache.ptr,
                            row_value_cache.ptr,
                            attention_scratch.attn_out.ptr,
                            row_decode_spans,
                            row_decode_spans.max_live_count,
                            block_size,
                            self.config.num_attention_heads,
                            self.config.num_key_value_heads,
                            self.config.head_dim,
                            self.config.head_dim ** -0.5,
                            stream=stream,
                            library=_library_for(library, "attention"),
                            runtime=self.runtime,
                        )
                        row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                        self.runtime.memcpy_async(
                            row_context.ptr,
                            attention_scratch.attn_out.ptr,
                            self.config.num_attention_heads * self.config.head_dim * DType.FP32.itemsize,
                            HipMemcpyKind.DEVICE_TO_DEVICE,
                            stream,
                        )
                    qwen35_full_attn_gate_mul_fp16(
                        attention_scratch.query_raw.ptr,
                        gate.ptr,
                        attention_scratch.gated_attn.ptr,
                        tokens * self.config.num_attention_heads * self.config.head_dim,
                        stream=stream,
                        library=_library_for(library, "attention"),
                        runtime=self.runtime,
                    )
                else:
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
            elif force_per_row_dense_context_batch_gate and tokens > 1:
                if per_row_contexts is None or len(per_row_contexts) != tokens:
                    raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
                q_width = self.config.num_attention_heads * self.config.head_dim
                for row, (row_key_cache, row_value_cache, row_decode_spans) in enumerate(per_row_contexts):
                    row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
                    if row_decode_spans.storage_dtype != DType.BF16:
                        raise NotImplementedError("per-row dense full-attention context/batch-gate diagnostic currently requires BF16 KV")
                    if row_decode_spans.max_live_count >= 1024:
                        raise NotImplementedError("per-row dense full-attention context/batch-gate diagnostic does not cover split-K decode")
                    qwen35_full_attn_decode_context_bf16(
                        row_scratch.query.ptr,
                        row_key_cache.ptr,
                        row_value_cache.ptr,
                        attention_scratch.attn_out.ptr,
                        row_decode_spans.live_counts.ptr,
                        row_decode_spans.max_live_count,
                        self.config.num_attention_heads,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                        self.config.head_dim ** -0.5,
                        stream=stream,
                        library=_library_for(library, "attention"),
                        runtime=self.runtime,
                    )
                    row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                    self.runtime.memcpy_async(
                        row_context.ptr,
                        attention_scratch.attn_out.ptr,
                        q_width * DType.FP32.itemsize,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                qwen35_full_attn_gate_mul_fp16(
                    attention_scratch.query_raw.ptr,
                    gate.ptr,
                    attention_scratch.gated_attn.ptr,
                    tokens * q_width,
                    stream=stream,
                    library=_library_for(library, "attention"),
                    runtime=self.runtime,
                )
                gated = attention_scratch.gated_attn
            elif (force_per_row_dense_context_only or force_per_row_paged_context_only) and tokens > 1:
                if per_row_contexts is None or len(per_row_contexts) != tokens:
                    raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
                q_width = self.config.num_attention_heads * self.config.head_dim
                for row, (row_key_cache, row_value_cache, row_decode_spans) in enumerate(per_row_contexts):
                    row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
                    if row_decode_spans.storage_dtype != DType.BF16:
                        raise NotImplementedError("per-row full-attention context-only diagnostic currently requires BF16 KV")
                    if force_per_row_paged_context_only or row_decode_spans.max_live_count >= 1024:
                        qwen35_paged_full_attn_decode_context_bf16_spans(
                            row_scratch.query.ptr,
                            row_key_cache.ptr,
                            row_value_cache.ptr,
                            attention_scratch.attn_out.ptr,
                            row_decode_spans,
                            row_decode_spans.max_live_count,
                            block_size,
                            self.config.num_attention_heads,
                            self.config.num_key_value_heads,
                            self.config.head_dim,
                            self.config.head_dim ** -0.5,
                            stream=stream,
                            library=_library_for(library, "attention"),
                            runtime=self.runtime,
                        )
                    else:
                        qwen35_full_attn_decode_context_bf16(
                            row_scratch.query.ptr,
                            row_key_cache.ptr,
                            row_value_cache.ptr,
                            attention_scratch.attn_out.ptr,
                            row_decode_spans.live_counts.ptr,
                            row_decode_spans.max_live_count,
                            self.config.num_attention_heads,
                            self.config.num_key_value_heads,
                            self.config.head_dim,
                            self.config.head_dim ** -0.5,
                            stream=stream,
                            library=_library_for(library, "attention"),
                            runtime=self.runtime,
                        )
                    row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                    self.runtime.memcpy_async(
                        row_context.ptr,
                        attention_scratch.attn_out.ptr,
                        q_width * DType.FP32.itemsize,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    row_gate = self._row_tensor_view(gate, row)
                    row_gated = self._row_tensor_view(attention_scratch.gated_attn, row)
                    qwen35_full_attn_gate_mul_fp16(
                        row_context.ptr,
                        row_gate.ptr,
                        row_gated.ptr,
                        q_width,
                        stream=stream,
                        library=_library_for(library, "attention"),
                        runtime=self.runtime,
                    )
                gated = attention_scratch.gated_attn
            elif force_per_row_context_only and tokens > 1:
                if per_row_contexts is None or len(per_row_contexts) != tokens:
                    raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
                for row, (row_key_cache, row_value_cache, row_decode_spans) in enumerate(per_row_contexts):
                    row_scratch = self._decode_row_full_attention_scratch(attention_scratch, row)
                    if row_decode_spans.storage_dtype != DType.BF16:
                        raise NotImplementedError("per-row full-attention context-only diagnostic currently requires BF16 KV")
                    if tokens == 2 or row_decode_spans.max_live_count >= 1024:
                        qwen35_paged_full_attn_decode_context_bf16_spans(
                            row_scratch.query.ptr,
                            row_key_cache.ptr,
                            row_value_cache.ptr,
                            attention_scratch.attn_out.ptr,
                            row_decode_spans,
                            row_decode_spans.max_live_count,
                            block_size,
                            self.config.num_attention_heads,
                            self.config.num_key_value_heads,
                            self.config.head_dim,
                            self.config.head_dim ** -0.5,
                            stream=stream,
                            library=_library_for(library, "attention"),
                            runtime=self.runtime,
                        )
                    else:
                        qwen35_full_attn_decode_context_bf16(
                            row_scratch.query.ptr,
                            row_key_cache.ptr,
                            row_value_cache.ptr,
                            attention_scratch.attn_out.ptr,
                            row_decode_spans.live_counts.ptr,
                            row_decode_spans.max_live_count,
                            self.config.num_attention_heads,
                            self.config.num_key_value_heads,
                            self.config.head_dim,
                            self.config.head_dim ** -0.5,
                            stream=stream,
                            library=_library_for(library, "attention"),
                            runtime=self.runtime,
                        )
                    row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                    self.runtime.memcpy_async(
                        row_context.ptr,
                        attention_scratch.attn_out.ptr,
                        self.config.num_attention_heads * self.config.head_dim * DType.FP32.itemsize,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                q_width = self.config.num_attention_heads * self.config.head_dim
                if tokens == 2:
                    qwen35_full_attn_gate_mul_fp16(
                        attention_scratch.query_raw.ptr,
                        gate.ptr,
                        attention_scratch.gated_attn.ptr,
                        tokens * q_width,
                        stream=stream,
                        library=_library_for(library, "attention"),
                        runtime=self.runtime,
                    )
                else:
                    for row in range(tokens):
                        row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                        row_gate = self._row_tensor_view(gate, row)
                        row_gated = self._row_tensor_view(attention_scratch.gated_attn, row)
                        qwen35_full_attn_gate_mul_fp16(
                            row_context.ptr,
                            row_gate.ptr,
                            row_gated.ptr,
                            q_width,
                            stream=stream,
                            library=_library_for(library, "attention"),
                            runtime=self.runtime,
                        )
                gated = attention_scratch.gated_attn
            elif force_batch_temp_context and tokens > 1:
                if decode_spans.storage_dtype != DType.BF16:
                    raise NotImplementedError("temp-output full-attention context diagnostic currently requires BF16 KV")
                if decode_spans.max_live_count >= 1024:
                    raise NotImplementedError("temp-output full-attention context diagnostic does not cover split-K decode")
                temp_context = self.workspace.reserve_tensor(
                    "attn.decode.batch_context_tmp",
                    (tokens, self.config.num_attention_heads, self.config.head_dim),
                    DType.FP32,
                )
                qwen35_paged_full_attn_decode_context_bf16_batch_spans(
                    attention_scratch.query.ptr,
                    key_cache.ptr,
                    value_cache.ptr,
                    temp_context.ptr,
                    decode_spans,
                    tokens,
                    decode_spans.max_live_count,
                    block_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.head_dim ** -0.5,
                    stream=stream,
                    library=_library_for(library, "attention"),
                    runtime=self.runtime,
                )
                runtime = self.runtime or get_hip_runtime()
                runtime.memcpy_async(
                    attention_scratch.query_raw.ptr,
                    temp_context.ptr,
                    temp_context.numel * DType.FP32.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                qwen35_full_attn_gate_mul_fp16(
                    attention_scratch.query_raw.ptr,
                    gate.ptr,
                    attention_scratch.gated_attn.ptr,
                    tokens * self.config.num_attention_heads * self.config.head_dim,
                    stream=stream,
                    library=_library_for(library, "attention"),
                    runtime=self.runtime,
                )
                gated = attention_scratch.gated_attn
            elif force_batch_compact_context and tokens > 1:
                if per_row_contexts is None or len(per_row_contexts) != tokens:
                    raise ValueError("per_row_contexts must provide one key/value/span tuple per decode row")
                if decode_spans.storage_dtype != DType.BF16:
                    raise NotImplementedError("compact-cache full-attention context diagnostic currently requires BF16 KV")
                if decode_spans.max_live_count >= 1024:
                    raise NotImplementedError("compact-cache full-attention context diagnostic does not cover split-K decode")
                row_blocks = int(per_row_contexts[0][2].base_offsets.numel)
                if row_blocks <= 0:
                    raise ValueError("compact-cache full-attention context diagnostic requires non-empty block tables")
                compact_key_cache = self.workspace.reserve_tensor(
                    "attn.decode.batch_compact_key_cache",
                    (tokens * row_blocks, block_size, self.config.num_key_value_heads, self.config.head_dim),
                    DType.BF16,
                )
                compact_value_cache = self.workspace.reserve_tensor(
                    "attn.decode.batch_compact_value_cache",
                    (tokens * row_blocks, block_size, self.config.num_key_value_heads, self.config.head_dim),
                    DType.BF16,
                )
                compact_block_table = self.workspace.reserve_tensor(
                    "attn.decode.batch_compact_block_table",
                    (tokens, row_blocks),
                    DType.INT32,
                )
                compact_live_counts = self.workspace.reserve_tensor(
                    "attn.decode.batch_compact_live_counts",
                    (tokens,),
                    DType.INT64,
                )
                row_cache_bytes = row_blocks * block_size * self.config.num_key_value_heads * self.config.head_dim * DType.BF16.itemsize
                row_table_bytes = row_blocks * DType.INT32.itemsize
                runtime = self.runtime or get_hip_runtime()
                for row, (row_key_cache, row_value_cache, row_decode_spans) in enumerate(per_row_contexts):
                    if row_decode_spans.storage_dtype != DType.BF16:
                        raise NotImplementedError("compact-cache full-attention context diagnostic currently requires BF16 KV")
                    if int(row_decode_spans.base_offsets.numel) != row_blocks:
                        raise ValueError("compact-cache full-attention context diagnostic requires uniform row block-table length")
                    runtime.memcpy_async(
                        compact_key_cache.ptr + row * row_cache_bytes,
                        row_key_cache.ptr,
                        row_cache_bytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    runtime.memcpy_async(
                        compact_value_cache.ptr + row * row_cache_bytes,
                        row_value_cache.ptr,
                        row_cache_bytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    runtime.memcpy_async(
                        compact_block_table.ptr + row * row_table_bytes,
                        row_decode_spans.base_offsets.ptr,
                        row_table_bytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    runtime.memcpy_async(
                        compact_live_counts.ptr + row * DType.INT64.itemsize,
                        row_decode_spans.live_counts.ptr,
                        DType.INT64.itemsize,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                compact_spans = KVLiveSpans.paged_uniform(
                    block_table=compact_block_table,
                    live_counts=compact_live_counts,
                    max_live_count=decode_spans.max_live_count,
                    storage_dtype=DType.BF16,
                )
                qwen35_paged_full_attn_decode_context_bf16_batch_spans(
                    attention_scratch.query.ptr,
                    compact_key_cache.ptr,
                    compact_value_cache.ptr,
                    attention_scratch.query_raw.ptr,
                    compact_spans,
                    tokens,
                    decode_spans.max_live_count,
                    block_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.head_dim ** -0.5,
                    stream=stream,
                    library=_library_for(library, "attention"),
                    runtime=self.runtime,
                )
                qwen35_full_attn_gate_mul_fp16(
                    attention_scratch.query_raw.ptr,
                    gate.ptr,
                    attention_scratch.gated_attn.ptr,
                    tokens * self.config.num_attention_heads * self.config.head_dim,
                    stream=stream,
                    library=_library_for(library, "attention"),
                    runtime=self.runtime,
                )
                gated = attention_scratch.gated_attn
            elif force_per_row_gate and tokens > 1:
                if decode_spans.storage_dtype != DType.BF16:
                    raise NotImplementedError("per-row full-attention gate diagnostic currently requires BF16 KV")
                if decode_spans.max_live_count >= 1024:
                    raise NotImplementedError("per-row full-attention gate diagnostic does not cover split-K decode")
                qwen35_paged_full_attn_decode_context_bf16_batch_spans(
                    attention_scratch.query.ptr,
                    key_cache.ptr,
                    value_cache.ptr,
                    attention_scratch.query_raw.ptr,
                    decode_spans,
                    tokens,
                    decode_spans.max_live_count,
                    block_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.head_dim ** -0.5,
                    stream=stream,
                    library=_library_for(library, "attention"),
                    runtime=self.runtime,
                )
                q_width = self.config.num_attention_heads * self.config.head_dim
                for row in range(tokens):
                    row_context = self._row_tensor_view(attention_scratch.query_raw, row)
                    row_gate = self._row_tensor_view(gate, row)
                    row_gated = self._row_tensor_view(attention_scratch.gated_attn, row)
                    qwen35_full_attn_gate_mul_fp16(
                        row_context.ptr,
                        row_gate.ptr,
                        row_gated.ptr,
                        q_width,
                        stream=stream,
                        library=_library_for(library, "attention"),
                        runtime=self.runtime,
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
        direct_int8 = (
            append_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
            or prefill_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
        )
        retained_int8 = retained_append_spans is not None
        if direct_int8:
            if append_spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD or prefill_spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD:
                raise ValueError("direct INT8 prefill requires INT8 append and attention spans")
            if key_cache.dtype is not DType.INT8 or value_cache.dtype is not DType.INT8:
                raise ValueError("direct INT8 prefill requires INT8 key/value cache tensors")
            if append_spans.scale_metadata is None or prefill_spans.scale_metadata is None:
                raise ValueError("direct INT8 prefill requires scale metadata")
            if aotriton_attention:
                raise ValueError("AOTriton prefill requires BF16 K/V; disable it for direct INT8 prefill")
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
        if append_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            self.append_full_attention_kv_int8_per_token_head_fp16_batch(
                attention_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                spans=append_spans,
                rows=tokens,
                block_size=block_size,
                library=library,
                stream=stream,
            )
        else:
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
            if prefill_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD:
                gated = self.prefill_full_attention_int8_gqa_gate_fp16(
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
        # M7.C.6: small-batch path mirrors the bf16 sibling at
        # ``project_linear_attention_qkv_z_bf16`` line 1083+.  The dual GEMV
        # writes row-major [qkv,z] per token sharing the combined ``qkv_z``
        # buffer with ``qkv`` and ``z`` as views; at tokens > 1 the view strides
        # don't match the dual GEMV row stride.  Split into two single GEMVs
        # writing the views' backing memory directly.
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
                qkvz_dual_fn = (
                    gemv_awq_dual_pack8_output_tiled_transposed_fp16
                    if tokens in _PACK8_OUTPUT_TILED_ROWS
                    else gemv_awq_dual_pack8_transposed_fp16
                )
                qkvz_dual_fn(
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
            qkv_proj_fn = (
                gemv_awq_pack8_output_tiled_transposed_fp16
                if tokens in _PACK8_OUTPUT_TILED_ROWS
                else gemv_awq_pack8_transposed_fp16
            )
            qkv_proj_fn(
                scratch.qkv_rot.ptr,
                qkv_qweight.ptr,
                self.tensor(f"{qkv}.qzeros").ptr,
                self.tensor(f"{qkv}.scales").ptr,
                scratch.qkv.ptr,
                tokens,
                scratch.qkv_rot.shape[-1],
                qkv_out_packed,
                group_size,
                threads=128,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
            z_proj_fn = (
                gemv_awq_pack8_output_tiled_transposed_fp16
                if tokens in _PACK8_OUTPUT_TILED_ROWS
                else gemv_awq_pack8_transposed_fp16
            )
            z_proj_fn(
                scratch.z_rot.ptr,
                z_qweight.ptr,
                self.tensor(f"{z}.qzeros").ptr,
                self.tensor(f"{z}.scales").ptr,
                scratch.z.ptr,
                tokens,
                scratch.z_rot.shape[-1],
                z_out_packed,
                group_size,
                threads=128,
                stream=stream,
                library=awq_library,
                runtime=self.runtime,
            )
        elif tokens <= _small_batch_decode_threshold():
            # M7.C.6: two single GEMVs, one for QKV (writes scratch.qkv.ptr)
            # and one for Z (writes scratch.z.ptr).  Direct port of the bf16
            # sibling pattern at project_linear_attention_qkv_z_bf16 line 1090+.
            # M15.1/M15.3: weight-amortize across the B+1 verifier rows via the
            # bit-exact multi-row decode kernel; M15.3 fuses the qkv+z pair into
            # one split-dual launch (bit-identical to two decode singles).
            awq_library = _library_for(library, "awq")
            if _w4_multi_row_small_batch_site_enabled("linear_qkv_z"):
                gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16(
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
                    library=awq_library,
                    runtime=self.runtime,
                )
            else:
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
                    stream=stream,
                    library=awq_library,
                    runtime=self.runtime,
                )
        else:
            if _w4_multi_row_dual_site_eligible("linear_qkv_z", tokens, scratch.qkv_rot.shape[-1], group_size):
                # M12.6: multi-row dual W4 GEMV for the linear-attn QKV+Z projection.
                gemv_awq_dual_pack8_multi_row_split_transposed_fp16(
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
                if _use_linear_ab_dual_separate(tokens):
                    dense_dual_gemv_separate_out_fp16(
                        hidden.ptr,
                        a_weight.ptr,
                        b_weight.ptr,
                        scratch.a.ptr,
                        scratch.b.ptr,
                        tokens,
                        self.config.hidden_size,
                        self.config.linear_num_value_heads,
                        self.config.linear_num_value_heads,
                        threads=threads,
                        stream=stream,
                        library=dense_library,
                        runtime=self.runtime,
                    )
                elif _use_verify_dense_gemv_wmma(tokens, self.config.hidden_size):
                    dense_gemv_out_fp16_wmma(
                        hidden.ptr,
                        a_weight.ptr,
                        scratch.a.ptr,
                        tokens,
                        self.config.hidden_size,
                        self.config.linear_num_value_heads,
                        stream=stream,
                        library=dense_library,
                        runtime=self.runtime,
                    )
                    dense_gemv_out_fp16_wmma(
                        hidden.ptr,
                        b_weight.ptr,
                        scratch.b.ptr,
                        tokens,
                        self.config.hidden_size,
                        self.config.linear_num_value_heads,
                        stream=stream,
                        library=dense_library,
                        runtime=self.runtime,
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
        if _linear_out_c1_exact_rows_enabled(tokens):
            return self.project_linear_attention_decode_rows_out_fp16(
                scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn.out_proj"
        width = scratch.recurrent_out.shape[-1]
        pairs = self.tensor(f"{prefix}.pairs")
        if _linear_out_cast_rotate_fused_enabled(tokens):
            paro_rotate1_f32_to_fp16(
                scratch.recurrent_out.ptr,
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
        else:
            f32_to_fp16(
                scratch.recurrent_out.ptr,
                scratch.recurrent_bf16.ptr,
                tokens * width,
                stream=stream,
                library=_library_for(library, "cast"),
                runtime=self.runtime,
            )
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
        force_selected_c1_qkv_z_input: bool = False,
        force_selected_c1_qkv_projections: bool = False,
        force_selected_c1_z_projections: bool = False,
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
                if force_selected_c1_qkv_z_input:
                    self.rotate_linear_attention_decode_rows_qkv_z_fp16(
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
                if force_selected_c1_qkv_projections or force_selected_c1_z_projections:
                    self.project_linear_attention_decode_rows_qkv_z_subset_fp16(
                        hidden,
                        scratch,
                        tokens=tokens,
                        copy_qkv=force_selected_c1_qkv_projections,
                        copy_z=force_selected_c1_z_projections,
                        group_size=group_size,
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
        force_selected_c1_qkv_z_input: bool = False,
        force_selected_c1_qkv_projections: bool = False,
        force_selected_c1_z_projections: bool = False,
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
            force_selected_c1_qkv_z_input=force_selected_c1_qkv_z_input,
            force_selected_c1_qkv_projections=force_selected_c1_qkv_projections,
            force_selected_c1_z_projections=force_selected_c1_z_projections,
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

    def run_linear_attention_moe_tree_tloop_layer_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        parent_rows: Tensor,
        linear_scratch: Qwen35ParoLinearAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch | None = None,
        out: Tensor | None = None,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        # M13.B.0: ``out`` forwards into the final MoE combine so the per-layer
        # ``next_hidden`` D2D copy in ``_iterate_verify_chain_layers`` becomes a
        # no-op when the caller passes ``out=next_hidden``.
        """Run one linear-attention layer for a parent-indexed verifier tree.

        ``parent_rows`` is a row-major/topological int64 vector where roots use
        ``-1`` and every non-root row references an earlier row.  The t-loop
        Conv/GDN kernels fill ``linear_scratch.tree_conv_state`` and
        ``linear_scratch.tree_recurrent_state`` for every row so the caller can
        later commit the selected row without replaying rejected candidates.
        """

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if parent_rows.dtype is not DType.INT64 or parent_rows.shape != (tokens,):
            raise ValueError("parent_rows must be int64 with shape (tokens,)")
        linear_scratch = linear_scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        if (
            linear_scratch.tree_conv_state.shape[0] < tokens
            or linear_scratch.tree_recurrent_state.shape[0] < tokens
            or linear_scratch.tree_gdn_acc.shape[0] < tokens
        ):
            raise ValueError("linear-attention tree scratch must include one tree-state row per token")
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        # Verifier chains/trees are tiny (typically B+1 <= 8).  The grouped
        # compact/WMMA MoE route wins for real prefill chunks, but its fixed
        # routing/compaction overhead dominates these small speculative rows.
        use_grouped_moe = False if dense_mlp else tokens >= _verify_moe_grouped_min_tokens()
        if dense_mlp:
            if not isinstance(moe_scratch, Qwen35ParoDenseMlpScratch):
                moe_scratch = self.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif use_grouped_moe:
            if not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
                moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif not isinstance(moe_scratch, Qwen35ParoMoeScratch):
            moe_scratch = self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)

        self.input_rmsnorm_fp16(hidden, linear_scratch.attn_input, tokens=tokens, library=library, stream=stream)
        self.rotate_linear_attention_inputs_fp16(
            linear_scratch.attn_input,
            linear_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        self.project_linear_attention_qkv_z_fp16(
            linear_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        self.project_linear_attention_ab_fp16(
            linear_scratch.attn_input,
            linear_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv_width = _linear_qkv_width(self.config)
        z_width = _linear_value_width(self.config)
        qwen35_linear_attn_tree_conv_decode_fp16_tloop(
            linear_scratch.qkv.ptr,
            conv_state.ptr,
            linear_scratch.tree_conv_state.ptr,
            self.tensor(f"{prefix}.conv1d.weight").ptr,
            parent_rows.ptr,
            linear_scratch.conv_out.ptr,
            tokens,
            qkv_width,
            self.config.linear_conv_kernel_dim,
            stream=stream,
            library=_library_for(library, "linear_conv"),
            runtime=self.runtime,
        )
        qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16(
            linear_scratch.conv_out.ptr,
            linear_scratch.z.ptr,
            linear_scratch.a.ptr,
            linear_scratch.b.ptr,
            self.tensor(f"{prefix}.dt_bias").ptr,
            self.tensor(f"{prefix}.A_log").ptr,
            self.tensor(f"{prefix}.norm.weight").ptr,
            recurrent_state.ptr,
            linear_scratch.tree_recurrent_state.ptr,
            parent_rows.ptr,
            linear_scratch.tree_gdn_acc.ptr,
            linear_scratch.recurrent_out.ptr,
            self.config.rms_norm_eps,
            tokens,
            self.config.linear_num_key_heads,
            self.config.linear_num_value_heads,
            self.config.linear_key_head_dim,
            self.config.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        if linear_scratch.recurrent_out.shape[-1] != z_width:
            raise ValueError("linear-attention recurrent scratch width mismatch")
        attn_out = self.project_linear_attention_out_fp16(
            linear_scratch,
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
                out=out,
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
                out=out,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_c1_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            out=out,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )

    def run_linear_attention_moe_chain_tloop_layer_fp16(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        chain_conv_state: Tensor,
        chain_recurrent_state: Tensor,
        linear_scratch: Qwen35ParoLinearAttentionScratch | None = None,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch | None = None,
        out: Tensor | None = None,
        tokens: int,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        # M13.B.0: ``out`` forwards into the final MoE combine so the per-layer
        # ``next_hidden`` D2D copy in ``_iterate_verify_chain_layers`` becomes a
        # no-op when the caller passes ``out=next_hidden``.
        """Run one linear-attention layer for a single verifier chain.

        For row topology ``[-1, 0, 1, ...]`` this avoids parent-row global
        state reloads by carrying the Conv/GDN state forward in-kernel, while
        still materializing every row for exact partial-accept commits.
        """

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        linear_scratch = linear_scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        use_grouped_moe = False if dense_mlp else tokens >= _verify_moe_grouped_min_tokens()
        if dense_mlp:
            if not isinstance(moe_scratch, Qwen35ParoDenseMlpScratch):
                moe_scratch = self.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif use_grouped_moe:
            if not isinstance(moe_scratch, Qwen35ParoGroupedMoeScratch):
                moe_scratch = self.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)
        elif not isinstance(moe_scratch, Qwen35ParoMoeScratch):
            moe_scratch = self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)

        if _fused_rmsnorm_rotate_enabled() and _fused_rmsnorm_rotate2_shape_ok(
            tokens, self.config.hidden_size, group_size
        ):
            # M15.4: one launch for input RMSNorm + qkv/z rotate (also writes the
            # unrotated RMSNorm output to attn_input for the AB projection).
            _lin_prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
            _qkv = f"{_lin_prefix}.in_proj_qkv"
            _z = f"{_lin_prefix}.in_proj_z"
            _pairs_qkv = self.tensor(f"{_qkv}.pairs")
            paro_rmsnorm_rotate2_fp16(
                hidden.ptr,
                self.tensor(f"layers.{self.layer_weights.layer_id}.input_layernorm.weight").ptr,
                linear_scratch.attn_input.ptr,
                linear_scratch.qkv_rot.ptr,
                linear_scratch.z_rot.ptr,
                _pairs_qkv.ptr,
                self.tensor(f"{_z}.pairs").ptr,
                self.tensor(f"{_qkv}.theta").ptr,
                self.tensor(f"{_z}.theta").ptr,
                self.tensor(f"{_qkv}.channel_scales").ptr,
                self.tensor(f"{_z}.channel_scales").ptr,
                self.config.rms_norm_eps,
                tokens,
                self.config.hidden_size,
                group_size,
                _rotation_krot(_pairs_qkv),
                stream=stream,
                library=_library_for(library, "rotate"),
                runtime=self.runtime,
            )
        else:
            self.input_rmsnorm_fp16(hidden, linear_scratch.attn_input, tokens=tokens, library=library, stream=stream)
            self.rotate_linear_attention_inputs_fp16(
                linear_scratch.attn_input,
                linear_scratch,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        self.project_linear_attention_qkv_z_fp16(
            linear_scratch,
            tokens=tokens,
            group_size=group_size,
            library=library,
            stream=stream,
        )
        self.project_linear_attention_ab_fp16(
            linear_scratch.attn_input,
            linear_scratch,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        prefix = f"layers.{self.layer_weights.layer_id}.linear_attn"
        qkv_width = _linear_qkv_width(self.config)
        z_width = _linear_value_width(self.config)
        qwen35_linear_attn_chain_conv_decode_fp16_tloop(
            linear_scratch.qkv.ptr,
            conv_state.ptr,
            chain_conv_state.ptr,
            self.tensor(f"{prefix}.conv1d.weight").ptr,
            linear_scratch.conv_out.ptr,
            tokens,
            qkv_width,
            self.config.linear_conv_kernel_dim,
            stream=stream,
            library=_library_for(library, "linear_conv"),
            runtime=self.runtime,
        )
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16(
            linear_scratch.conv_out.ptr,
            linear_scratch.z.ptr,
            linear_scratch.a.ptr,
            linear_scratch.b.ptr,
            self.tensor(f"{prefix}.dt_bias").ptr,
            self.tensor(f"{prefix}.A_log").ptr,
            self.tensor(f"{prefix}.norm.weight").ptr,
            recurrent_state.ptr,
            chain_recurrent_state.ptr,
            linear_scratch.tree_gdn_acc.ptr,
            linear_scratch.recurrent_out.ptr,
            self.config.rms_norm_eps,
            tokens,
            self.config.linear_num_key_heads,
            self.config.linear_num_value_heads,
            self.config.linear_key_head_dim,
            self.config.linear_value_head_dim,
            stream=stream,
            library=_library_for(library, "linear_gdn"),
            runtime=self.runtime,
        )
        if linear_scratch.recurrent_out.shape[-1] != z_width:
            raise ValueError("linear-attention recurrent scratch width mismatch")
        attn_out = self.project_linear_attention_out_fp16(
            linear_scratch,
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
                out=out,
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
                out=out,
                tokens=tokens,
                group_size=group_size,
                library=library,
                stream=stream,
            )
        return self.run_moe_c1_fp16(
            mlp_input,
            residual,
            scratch=moe_scratch,
            out=out,
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
        force_selected_c1_qkv_z_linear_input: bool = False,
        force_selected_c1_qkv_linear_projections: bool = False,
        force_selected_c1_z_linear_projections: bool = False,
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
            force_selected_c1_qkv_z_input=force_selected_c1_qkv_z_linear_input,
            force_selected_c1_qkv_projections=force_selected_c1_qkv_linear_projections,
            force_selected_c1_z_projections=force_selected_c1_z_linear_projections,
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
        threads: int = 64,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        gate_up_pairs = self.tensor(f"{prefix}.gate_up_weight_pairs")
        gate_qweight = self.tensor(f"{prefix}.stacked_gate_qweight_pack8_decode")
        up_qweight = self.tensor(f"{prefix}.stacked_up_qweight_pack8_decode")
        rows = tokens * self.config.num_experts_per_tok
        if _moe_fused_rotate_enabled():
            # M13.B.1: fused rotate + selected dual pack8 GEMV.  Bit-exact with
            # the unfused chain below via an LDS scalar_t round-trip in the
            # kernel.  scratch.gate_up_input is unused on this path but stays
            # allocated for the unfused fallback.
            gemv_awq_selected_dual_pack8_transposed_rotate_out_fp16(
                hidden.ptr,
                scratch.selected_experts.ptr,
                gate_up_pairs.ptr,
                self.tensor(f"{prefix}.gate_up_weight_theta").ptr,
                self.tensor(f"{prefix}.gate_up_weight_channel_scales").ptr,
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
                _rotation_krot(gate_up_pairs),
                threads=threads,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
            return scratch.gate_up
        if tokens > 1 and _selected_moe_staged_rotate_enabled() and hasattr(scratch, "shared_rotate_fuse_barrier"):
            runtime = self.runtime or get_hip_runtime()
            barrier_target, barrier_epoch = self._next_shared_rotate_fuse_barrier_key(
                scratch.shared_rotate_fuse_barrier,
                rows=tokens,
                in_features=self.config.hidden_size,
                group_size=group_size,
                rotations=1,
                stream=stream,
                runtime=runtime,
            )
            gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16(
                hidden.ptr,
                scratch.gate_up_input.ptr,
                scratch.selected_experts.ptr,
                gate_up_pairs.ptr,
                self.tensor(f"{prefix}.gate_up_weight_theta").ptr,
                self.tensor(f"{prefix}.gate_up_weight_channel_scales").ptr,
                gate_qweight.ptr,
                self.tensor(f"{prefix}.stacked_gate_qzeros").ptr,
                self.tensor(f"{prefix}.stacked_gate_scales").ptr,
                up_qweight.ptr,
                self.tensor(f"{prefix}.stacked_up_qzeros").ptr,
                self.tensor(f"{prefix}.stacked_up_scales").ptr,
                scratch.gate_up.ptr,
                scratch.shared_rotate_fuse_barrier.ptr,
                tokens,
                rows,
                hidden.shape[-1],
                _out_packed_from_transposed_qweight(gate_qweight),
                _out_packed_from_transposed_qweight(up_qweight),
                self.config.num_experts,
                group_size,
                _rotation_krot(gate_up_pairs),
                barrier_target,
                barrier_epoch,
                threads=threads,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=runtime,
            )
            return scratch.gate_up
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
        threads: int = 64,
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

    def selected_moe_ffn_megakernel_fp16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 256,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """B4: fused selected-expert PARO FFN megakernel -> ``scratch.down_out``.

        Numerically-equivalent (KL ~2e-5) replacement for
        ``selected_moe_gate_up_pack8_fp16`` + ``activate_rotate_moe_down_fp16``
        + ``selected_moe_down_pack8_fp16``.  Reads the un-rotated MoE input
        ``hidden`` and applies rotate1 internally, so it does not depend on the
        ``gate_up_input`` / ``gate_up`` / ``down_input`` scratch buffers.
        Requires gate_up and down rotations to share ``krot`` (true for the
        deployed model; raises otherwise so the caller can fall back).
        """

        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        gate_up_pairs = self.tensor(f"{prefix}.gate_up_weight_pairs")
        down_pairs = self.tensor(f"{prefix}.down_weight_pairs")
        krot = _rotation_krot(gate_up_pairs)
        if krot != _rotation_krot(down_pairs):
            raise ValueError(
                "selected_moe_ffn_megakernel_fp16 requires gate_up krot == down krot"
            )
        from hipengine.kernels.hip_gfx1100.quant.paro_moe_ffn_fused import (
            paro_selected_ffn_fused_fp16_fp16_out,
        )

        rows = tokens * self.config.num_experts_per_tok
        global _PARO_FFN_MEGAKERNEL_FIRED
        if os.environ.get("HIPENGINE_PARO_FFN_MEGAKERNEL_DEBUG"):
            _PARO_FFN_MEGAKERNEL_FIRED += 1
            if _PARO_FFN_MEGAKERNEL_FIRED == 1:
                import sys as _sys
                print(
                    f"[paro-ffn-megakernel] first fire: tokens={tokens} rows={rows} krot={krot} "
                    f"hidden={self.config.hidden_size} ffn={self.config.moe_intermediate_size}",
                    file=_sys.stderr, flush=True,
                )
        paro_selected_ffn_fused_fp16_fp16_out(
            hidden.ptr,
            scratch.selected_experts.ptr,
            self.tensor(f"{prefix}.stacked_gate_qweight_pack8_decode").ptr,
            self.tensor(f"{prefix}.stacked_gate_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_gate_scales").ptr,
            self.tensor(f"{prefix}.stacked_up_qweight_pack8_decode").ptr,
            self.tensor(f"{prefix}.stacked_up_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_up_scales").ptr,
            self.tensor(f"{prefix}.stacked_down_qweight_pack8_decode").ptr,
            self.tensor(f"{prefix}.stacked_down_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_down_scales").ptr,
            gate_up_pairs.ptr,
            self.tensor(f"{prefix}.gate_up_weight_theta").ptr,
            self.tensor(f"{prefix}.gate_up_weight_channel_scales").ptr,
            down_pairs.ptr,
            self.tensor(f"{prefix}.down_weight_theta").ptr,
            self.tensor(f"{prefix}.down_weight_channel_scales").ptr,
            scratch.down_out.ptr,
            tokens,
            rows,
            self.config.num_experts,
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            group_size,
            krot,
            threads=threads,
            stream=stream,
            library=_paro_ffn_megakernel_library(library),
            runtime=self.runtime,
        )
        return scratch.down_out

    def selected_moe_activate_down_pack8_fp16(
        self,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        group_size: int = 128,
        threads: int = 64,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Fused selected SiLU/down-rotate + selected down GEMV.

        The kernel stages the same FP16 ``down_input`` that the unfused
        ``activate_rotate_moe_down_fp16`` path writes, then runs the existing
        selected ids-tensor down GEMV after an in-kernel barrier.  This
        preserves the unfused path's numerics while removing one launch.
        """

        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        pairs = self.tensor(f"{prefix}.down_weight_pairs")
        qweight = self.tensor(f"{prefix}.stacked_down_qweight_pack8_decode")
        rows = tokens * self.config.num_experts_per_tok
        runtime = self.runtime or get_hip_runtime()
        barrier_target, barrier_epoch = self._next_shared_rotate_fuse_barrier_key(
            scratch.shared_rotate_fuse_barrier,
            rows=rows,
            in_features=self.config.moe_intermediate_size,
            group_size=group_size,
            stream=stream,
            runtime=runtime,
            rotations=1,
        )
        gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16(
            scratch.gate_up.ptr,
            scratch.down_input.ptr,
            scratch.selected_experts.ptr,
            pairs.ptr,
            self.tensor(f"{prefix}.down_weight_theta").ptr,
            self.tensor(f"{prefix}.down_weight_channel_scales").ptr,
            qweight.ptr,
            self.tensor(f"{prefix}.stacked_down_qzeros").ptr,
            self.tensor(f"{prefix}.stacked_down_scales").ptr,
            scratch.down_out.ptr,
            scratch.shared_rotate_fuse_barrier.ptr,
            rows,
            self.config.moe_intermediate_size,
            _out_packed_from_transposed_qweight(qweight),
            self.config.num_experts,
            group_size,
            _rotation_krot(pairs),
            barrier_target,
            barrier_epoch,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
            runtime=runtime,
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
        out: Tensor | None = None,
        tokens: int = 1,
        threads: int = 64,
        shared_gate_already_sigmoid: bool = False,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        # M13.B.0: ``out`` lets the grouped MoE write the combined residual
        # directly into the caller's ``next_hidden`` buffer instead of
        # ``scratch.moe_out`` + a follow-up D2D copy.
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.shared_expert"
        target = out if out is not None else scratch.moe_out
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
                target.ptr,
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
                target.ptr,
                tokens,
                self.config.hidden_size,
                self.config.shared_expert_intermediate_size,
                self.config.num_experts + 1,
                threads=threads,
                stream=stream,
                library=w8a16_library,
                runtime=self.runtime,
            )
        return target

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
        gate_qweight = self.tensor(f"{gate_base}.qweight_pack8_decode")
        up_qweight = self.tensor(f"{up_base}.qweight_pack8_decode")
        layer_type = self.config.layer_types[self.layer_weights.layer_id]
        # M12: the all-layer threshold bump regressed because it perturbed the
        # 30 linear-attention layers.  Batched verifier mode adds shared-expert
        # prefill overhead mainly in the 10 full-attention layers that c1_loop
        # used to run as tokens=1.  This site has no view aliasing, so use GEMV
        # only for tokens==1 or small full-attention verifier batches.
        small_batch = tokens == 1 or (layer_type == "full_attention" and tokens <= _small_batch_decode_threshold())
        # M13.B.2: in the small-batch path, replace `paro_rotate2 +
        # gemv_awq_dual_pack8_transposed` with the HBM-staged fused kernel
        # when gate_krot == up_krot (the kernel takes a single krot).  The
        # staged kernel rotates exactly once per (group, row) and barriers
        # before the GEMV phase so it's bit-exact with the unfused chain.
        # Skipped when krots differ (rare upstream variant; falls back to two
        # paro_rotate1 launches + the unfused dual GEMV).
        fused_shared_rotate = (
            small_batch
            and gate_krot == up_krot
            and _shared_expert_fused_rotate_enabled()
            and hasattr(scratch, "shared_rotate_fuse_barrier")
        )
        if not fused_shared_rotate:
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

        if small_batch:
            if fused_shared_rotate:
                runtime = self.runtime or get_hip_runtime()
                barrier_target, barrier_epoch = self._next_shared_rotate_fuse_barrier_key(
                    scratch.shared_rotate_fuse_barrier,
                    rows=tokens,
                    in_features=cfg.hidden_size,
                    group_size=group_size,
                    stream=stream,
                    runtime=runtime,
                )
                gemv_awq_dual_pack8_transposed_rotate_staged_keyed_fp16(
                    hidden.ptr,
                    scratch.shared_gate_input.ptr,
                    scratch.shared_up_input.ptr,
                    gate_pairs.ptr,
                    up_pairs.ptr,
                    self.tensor(f"{gate_base}.theta").ptr,
                    self.tensor(f"{up_base}.theta").ptr,
                    self.tensor(f"{gate_base}.channel_scales").ptr,
                    self.tensor(f"{up_base}.channel_scales").ptr,
                    gate_qweight.ptr,
                    self.tensor(f"{gate_base}.qzeros").ptr,
                    self.tensor(f"{gate_base}.scales").ptr,
                    up_qweight.ptr,
                    self.tensor(f"{up_base}.qzeros").ptr,
                    self.tensor(f"{up_base}.scales").ptr,
                    scratch.shared_up.ptr,
                    scratch.shared_rotate_fuse_barrier.ptr,
                    tokens,
                    cfg.hidden_size,
                    _out_packed_from_generic_transposed_qweight(gate_qweight),
                    _out_packed_from_generic_transposed_qweight(up_qweight),
                    group_size,
                    gate_krot,
                    barrier_target,
                    barrier_epoch,
                    stream=stream,
                    library=_library_for(library, "awq"),
                    runtime=runtime,
                )
            else:
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
        elif _w4_dual_output_tiled_split_site_eligible("shared_gate_up", tokens, cfg.hidden_size, group_size):
            # M16.4 follow-up: output-column-tiled dual GEMV for split gate/up buffers.
            gemv_awq_dual_pack8_output_tiled_split_transposed_fp16(
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
        elif _w4_multi_row_dual_site_eligible("shared_gate_up", tokens, cfg.hidden_size, group_size):
            # M12.6: shared-expert gate/up multi-row dual W4 GEMV.
            gemv_awq_dual_pack8_multi_row_split_transposed_fp16(
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

        if not small_batch:
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
        down_site = _w4_multi_row_single_site(down_base)
        down_mode = _w4_down_proj_small_batch_mode(down_site)
        if small_batch or (1 < tokens <= 8 and down_mode == "gemv"):
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
        elif (
            1 < tokens <= 8
            and down_mode in {"multi_row", "multi_row_decode"}
            and group_size % 16 == 0
            and cfg.shared_expert_intermediate_size % group_size == 0
        ):
            down_kernel = (
                gemv_awq_pack8_multi_row_decode_transposed_fp16
                if down_mode == "multi_row_decode"
                else gemv_awq_pack8_multi_row_transposed_fp16
            )
            down_kernel(
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
        # M7.C investigation (2026-05-21): see shared_expert_paro_w4_fp16 above
        # for the same rationale.  Safe but not in any verifier path today; left
        # at ``tokens == 1`` to avoid cache-pressure noise.  Tracked under M7.C.6.
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
        elif _w4_multi_row_dual_site_eligible("dense_gate_up", tokens, cfg.hidden_size, group_size):
            # M12.6: dense MLP gate/up multi-row dual W4 GEMV.
            gemv_awq_dual_pack8_multi_row_split_transposed_fp16(
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
        down_site = _w4_multi_row_single_site(down_base)
        down_mode = _w4_down_proj_small_batch_mode(down_site)
        if tokens == 1 or (1 < tokens <= 8 and down_mode == "gemv"):
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
        elif (
            1 < tokens <= 8
            and down_mode in {"multi_row", "multi_row_decode"}
            and group_size % 16 == 0
            and intermediate % group_size == 0
        ):
            down_kernel = (
                gemv_awq_pack8_multi_row_decode_transposed_fp16
                if down_mode == "multi_row_decode"
                else gemv_awq_pack8_multi_row_transposed_fp16
            )
            down_kernel(
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
        out: Tensor | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Same write-through semantics as ``run_moe_c1_fp16`` for the dense MLP
        variant.  When ``out`` is provided the residual combine writes directly
        into it (M12.6 layer-output write-through)."""

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
        target = out if out is not None else scratch.moe_out
        shared_gate_combine_residual_batch_out_fp16(
            dense_out.ptr,
            scratch.shared_zero.ptr,
            scratch.gate_logits.ptr,
            residual.ptr,
            target.ptr,
            tokens,
            self.config.hidden_size,
            1,
            stream=stream,
            library=_library_for(library, "combine"),
            runtime=self.runtime,
        )
        return target

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
        out: Tensor | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        # M13.B.0: ``out`` writes the final combine into the caller's buffer
        # instead of ``scratch.moe_out`` + a follow-up D2D copy.  Matches the
        # ``run_moe_c1_fp16`` contract so the verifier orchestrator can pass
        # ``out=next_hidden`` through every helper uniformly.
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
        use_decode_gemv = tokens <= 8
        if use_decode_gemv:
            gemv_awq_selected_dual_pack8_transposed_fp16(
                scratch.packed_gate_up_input.ptr,
                scratch.sorted_experts.ptr,
                gate_qweight.ptr,
                gate_qzeros.ptr,
                gate_scales.ptr,
                up_qweight.ptr,
                up_qzeros.ptr,
                up_scales.ptr,
                scratch.gate_up.ptr,
                total_lanes,
                total_lanes,
                cfg.hidden_size,
                _out_packed_from_transposed_qweight(gate_qweight),
                _out_packed_from_transposed_qweight(up_qweight),
                cfg.num_experts,
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        else:
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
        if use_decode_gemv:
            gemv_awq_selected_pack8_transposed_fp16(
                scratch.down_input.ptr,
                scratch.sorted_experts.ptr,
                down_qweight.ptr,
                self.tensor(f"{prefix}.stacked_down_qzeros").ptr,
                self.tensor(f"{prefix}.stacked_down_scales").ptr,
                scratch.down_out.ptr,
                total_lanes,
                cfg.moe_intermediate_size,
                _out_packed_from_transposed_qweight(down_qweight),
                cfg.num_experts,
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        else:
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
        self._memset_tensor(scratch.lane_to_row, value=0xFF, stream=stream, runtime=self.runtime)
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
        target = out if out is not None else scratch.moe_out
        if self._shared_expert_is_legacy_w8a16():
            self.shared_expert_gate_up_silu_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
            return self.shared_expert_down_combine_residual_fp16(
                scratch,
                residual,
                out=target,
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
            target.ptr,
            tokens,
            cfg.hidden_size,
            cfg.num_experts + 1,
            stream=stream,
            library=_library_for(library, "combine"),
            runtime=self.runtime,
        )
        return target

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
        out: Tensor | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        """Run the per-layer MoE pipeline.  When ``out`` is provided the final
        combine writes directly into it, avoiding the per-layer
        ``next_hidden = scratch.moe_out`` D2D copy in the verifier orchestrator
        (M12.6 layer-output write-through).

        M14.dispatch.1-beta: by default (unless
        ``HIPENGINE_MOE_C1_C_DISPATCH=0``) and when the call pattern matches
        the C dispatcher's contract (paro_w4 shared, tokens>1, no M13.B.1/B.2
        fused-rotate, group_size==128), bundle the 6 sub-method calls + 11
        underlying kernel launches into one extern-C call to cut ~10 ctypes ABI
        transitions per layer.  Falls back to the Python pipeline below for any
        unsupported call pattern.
        """

        scratch = scratch or self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        if (
            _selected_moe_ffn_megakernel_enabled()
            and tokens > 1
            and group_size == 128
            and self._shared_expert_kind == "packed_paro_w4"
        ):
            # B4: fused selected-expert FFN megakernel replaces the selected
            # sub-chain (gate_up + activate/down-rotate + down) with one launch
            # per (token, expert).  Router, shared expert and combine stay as
            # their own kernels.  Bypasses the C dispatcher.
            self.route_moe_topk_shared_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
            self.selected_moe_ffn_megakernel_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
            shared = self.shared_expert_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
            return self.combine_moe_c1_shared_residual_fp16(
                scratch,
                shared=shared,
                residual=residual,
                out=out,
                tokens=tokens,
                library=library,
                stream=stream,
            )
        if self._try_moe_c1_c_dispatch(
            hidden=hidden,
            residual=residual,
            out=out,
            scratch=scratch,
            tokens=tokens,
            group_size=group_size,
            stream=stream,
        ) is not None:
            return out or scratch.moe_out
        self.route_moe_topk_shared_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        self.selected_moe_gate_up_pack8_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        if tokens > 1 and _selected_moe_down_staged_enabled() and hasattr(scratch, "shared_rotate_fuse_barrier"):
            self.selected_moe_activate_down_pack8_fp16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        else:
            self.activate_rotate_moe_down_fp16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
            self.selected_moe_down_pack8_fp16(scratch.down_input, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        shared = self.shared_expert_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        return self.combine_moe_c1_shared_residual_fp16(
            scratch,
            shared=shared,
            residual=residual,
            out=out,
            tokens=tokens,
            library=library,
            stream=stream,
        )

    def _try_moe_c1_c_dispatch(
        self,
        *,
        hidden: Tensor,
        residual: Tensor,
        out: Tensor | None,
        scratch: Qwen35ParoMoeScratch,
        tokens: int,
        group_size: int,
        stream: int,
    ) -> Tensor | None:
        """Return ``out`` if the C dispatcher was used, else None for fallback.

        Preconditions for the C path (any failure -> Python fallback):

        - ``HIPENGINE_MOE_C1_C_DISPATCH`` is not explicitly disabled.
        - paro_w4 shared expert (not legacy w8a16).
        - ``tokens > 1`` (decode tokens=1 uses coop router; C path doesn't).
        - ``group_size == 128`` (cached value; mismatches fall back).
        - M13.B.1 fused-rotate disabled (else selected gate-up flow differs).
        - M13.B.2 fused-shared-rotate disabled (else shared expert flow differs).
        - Linear-attn layer-type OR full-attn at tokens<=small_batch_threshold.
        - Shared-expert krots match (rotate2 fast path).
        """

        if not moe_c1_c_dispatch_enabled():
            return None
        if self._shared_expert_kind != "packed_paro_w4":
            return None
        if tokens < 2:
            return None
        if group_size != 128:
            return None
        if _moe_fused_rotate_enabled():
            return None
        if _shared_expert_fused_rotate_enabled():
            return None

        layer_id = self.layer_weights.layer_id
        layer_kind = self.config.layer_types[layer_id]
        if layer_kind not in {"linear_attention", "full_attention"}:
            return None

        # Shared-expert rotate2 requires gate_krot == up_krot.
        shared = f"layers.{layer_id}.mlp.shared_expert"
        gate_pairs = self.tensor(f"{shared}.gate_proj.pairs")
        up_pairs = self.tensor(f"{shared}.up_proj.pairs")
        if _rotation_krot(gate_pairs) != _rotation_krot(up_pairs):
            return None

        # Lazy-construct the per-layer cache on first matching call.
        cache = self._moe_c1_dispatch_cache
        if cache is None or cache.layer_kind != layer_kind:
            from hipengine.runtime.moe_c1_dispatch import MoeC1DispatchCache
            cache = MoeC1DispatchCache(
                self,
                layer_kind=layer_kind,
                small_batch_threshold=_small_batch_decode_threshold(),
            )
            self._moe_c1_dispatch_cache = cache

        if not cache.supports_call(tokens=tokens):
            return None

        target_out = out if out is not None else scratch.moe_out
        selected_barrier_target = 0
        selected_barrier_epoch = 0
        selected_down_barrier_target = 0
        selected_down_barrier_epoch = 0
        selected_barrier = None
        if hasattr(scratch, "shared_rotate_fuse_barrier"):
            runtime = self.runtime or get_hip_runtime()
            selected_barrier = scratch.shared_rotate_fuse_barrier
            if _selected_moe_staged_rotate_enabled():
                selected_barrier_target, selected_barrier_epoch = self._next_shared_rotate_fuse_barrier_key(
                    selected_barrier,
                    rows=tokens,
                    in_features=self.config.hidden_size,
                    group_size=group_size,
                    stream=stream,
                    runtime=runtime,
                    rotations=1,
                )
            if _selected_moe_down_staged_enabled():
                selected_down_barrier_target, selected_down_barrier_epoch = self._next_shared_rotate_fuse_barrier_key(
                    selected_barrier,
                    rows=tokens * self.config.num_experts_per_tok,
                    in_features=self.config.moe_intermediate_size,
                    group_size=group_size,
                    stream=stream,
                    runtime=runtime,
                    rotations=1,
                )
        cache.dispatch(
            hidden=hidden,
            residual=residual,
            out=target_out,
            scratch=scratch,
            tokens=tokens,
            group_size=group_size,
            stream=stream,
            selected_rotate_fuse_barrier=selected_barrier,
            selected_rotate_barrier_target=selected_barrier_target,
            selected_rotate_barrier_epoch=selected_barrier_epoch,
            selected_down_barrier_target=selected_down_barrier_target,
            selected_down_barrier_epoch=selected_down_barrier_epoch,
        )
        return target_out

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
        use_decode_gemv = tokens <= 8
        if use_decode_gemv:
            gemv_awq_selected_dual_pack8_transposed_bf16(
                scratch.packed_gate_up_input.ptr,
                scratch.sorted_experts.ptr,
                gate_qweight.ptr,
                gate_qzeros.ptr,
                gate_scales.ptr,
                up_qweight.ptr,
                up_qzeros.ptr,
                up_scales.ptr,
                scratch.gate_up.ptr,
                total_lanes,
                total_lanes,
                cfg.hidden_size,
                _out_packed_from_transposed_qweight(gate_qweight),
                _out_packed_from_transposed_qweight(up_qweight),
                cfg.num_experts,
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        else:
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
        if use_decode_gemv:
            gemv_awq_selected_pack8_transposed_bf16(
                scratch.down_input.ptr,
                scratch.sorted_experts.ptr,
                down_qweight.ptr,
                self.tensor(f"{prefix}.stacked_down_qzeros").ptr,
                self.tensor(f"{prefix}.stacked_down_scales").ptr,
                scratch.down_out.ptr,
                total_lanes,
                cfg.moe_intermediate_size,
                _out_packed_from_transposed_qweight(down_qweight),
                cfg.num_experts,
                group_size,
                stream=stream,
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        else:
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
        self._memset_tensor(scratch.lane_to_row, value=0xFF, stream=stream, runtime=self.runtime)
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
            shared_rotate_fuse_barrier=self.workspace.reserve_tensor(
                f"moe.grouped.layer{self.layer_weights.layer_id}.shared_rotate_fuse_barrier", (2,), DType.INT32,
            ),
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
            shared_out=self.workspace.reserve_tensor("moe.shared_out", (tokens, cfg.hidden_size), lowp),
            moe_out=self.workspace.reserve_tensor("moe.out", (tokens, cfg.hidden_size), lowp),
            shared_rotate_fuse_barrier=self.workspace.reserve_tensor(
                f"moe.layer{self.layer_weights.layer_id}.shared_rotate_fuse_barrier", (2,), DType.INT32,
            ),
        )

    def _next_shared_rotate_fuse_barrier_key(
        self,
        barrier: Tensor,
        *,
        rows: int,
        in_features: int,
        group_size: int,
        stream: int,
        runtime,
        rotations: int = 2,
    ) -> tuple[int, int]:
        """Return cumulative barrier target/epoch for a keyed staged rotate GEMV.

        The underlying kernel increments ``barrier[0]`` once for every
        (rotation, group, row) staging block and publishes ``barrier[1]`` when
        the cumulative target is reached.  Initializing once per barrier pointer
        preserves stream ordering while avoiding M13.B.2's per-launch memset.
        """

        if rows <= 0 or in_features <= 0 or group_size <= 0 or rotations <= 0 or in_features % group_size:
            raise ValueError("invalid keyed staged-rotate barrier shape")
        rotate_blocks = (in_features // group_size) * rotations * rows
        ptr = int(barrier.ptr)
        if _SHARED_ROTATE_FUSE_BARRIER_MEMSET_MODE:
            # Capture-safe mode (#107): zero the barrier on-stream before every
            # staged launch so the captured graph replays a self-contained
            # memset→produce→consume sequence. Reset host counters so a later
            # keyed (non-capture) launch starts from a zeroed barrier.
            self._memset_tensor(barrier, stream=stream, runtime=runtime)
            self._shared_rotate_fuse_barrier_state[ptr] = (0, 0)
            return rotate_blocks, 1
        count, epoch = self._shared_rotate_fuse_barrier_state.get(ptr, (0, 0))
        if count == 0 and epoch == 0:
            self._memset_tensor(barrier, stream=stream, runtime=runtime)
        if count + rotate_blocks > 0x7FFFFFFF or epoch + 1 > 0x7FFFFFFF:
            self._memset_tensor(barrier, stream=stream, runtime=runtime)
            count = 0
            epoch = 0
        count += rotate_blocks
        epoch += 1
        self._shared_rotate_fuse_barrier_state[ptr] = (count, epoch)
        return count, epoch

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


def _weight_tensor_lookup_cache_enabled() -> bool:
    return _env_flag("HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE", True)


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


def _verify_moe_grouped_min_tokens() -> int:
    value = os.environ.get("HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS")
    if value is None or value.strip() == "":
        return 16
    return max(2, int(value))


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _moe_fused_rotate_enabled() -> bool:
    """M13.B.1 gate for the fused rotate + selected_dual_pack8 GEMV kernel.

    Defaults to ``off`` based on the M13.B.1 measurement: the existing
    ``gemv_awq_selected_dual_pack8_*_rotate_out_kernel`` design re-does the
    full LDS rotation in every ``(out_pack, row)`` block.  At the verifier
    shape (tokens=4, top_k=8, out_packs~192) this multiplies rotation work
    by ~1500× vs the unfused paro_rotate1 + selected_dual chain, blowing up
    ``moe_gate_up_dual_gemv`` kernel time by +12.4 ms/pass for only -40
    paro_rotate launches/pass saved.  Enable via
    ``HIPENGINE_MOE_FUSED_ROTATE={1,on,yes,true}`` only at shapes where the
    redundant in-LDS rotation cost is acceptable (e.g. very small
    ``out_pack × top_k`` totals).  A future properly-staged variant (HBM
    barrier, rotate once per x_row) is tracked as M13.B.3-style work.  The
    kernel itself is bit-exact with the unfused chain via an LDS scalar_t
    round-trip after rotation (Option C).
    """

    return _env_enabled("HIPENGINE_MOE_FUSED_ROTATE", default=False)


def _selected_moe_staged_rotate_enabled() -> bool:
    """Gate for HBM-staged selected-MoE rotate + ids-tensor dual GEMV.

    This replaces the verifier's ``paro_rotate1_fp16`` +
    ``gemv_awq_selected_dual_pack8_transposed_fp16`` pair with one staged
    kernel for ``tokens > 1``.  The staged kernel rotates each verifier x-row
    exactly once into the existing ``gate_up_input`` scratch, then the selected
    GEMV phase reads that FP16 buffer after an in-kernel barrier, preserving the
    unfused path's numerics while cutting one launch per MoE layer.

    Default remains off after the W7900 diagnostic: launch count dropped, but
    staged-kernel duration and verifier economics regressed.
    """

    return _env_enabled("HIPENGINE_SELECTED_MOE_STAGED_ROTATE", default=False)


def _selected_moe_down_staged_enabled() -> bool:
    """Gate for staged selected SiLU/down-rotate + ids-tensor down GEMV.

    Replaces ``silu_mul_dual_rotate_out_fp16`` +
    ``gemv_awq_selected_pack8_transposed_fp16`` with a single HBM-staged kernel
    for verifier ``tokens > 1``.  This was default-on after an early verifier
    rocprof win, but the current graph-auto MTP stack pays capture-safe
    barrier/fill overhead and the unfused fallback is faster on the D32
    9-prompt exact suite.  Keep the staged path opt-in for bisection.
    """

    return _env_enabled("HIPENGINE_SELECTED_MOE_DOWN_STAGED", default=False)


_PARO_FFN_MEGAKERNEL_LIBRARY = None
_PARO_FFN_MEGAKERNEL_FIRED = 0


def _paro_ffn_megakernel_fire_count() -> int:
    return _PARO_FFN_MEGAKERNEL_FIRED


if os.environ.get("HIPENGINE_PARO_FFN_MEGAKERNEL_DEBUG"):
    import atexit as _atexit

    @_atexit.register
    def _report_paro_ffn_megakernel_fires() -> None:
        import sys as _sys

        print(
            f"[paro-ffn-megakernel] total fires this process = {_PARO_FFN_MEGAKERNEL_FIRED}",
            file=_sys.stderr, flush=True,
        )


def _selected_moe_ffn_megakernel_enabled() -> bool:
    """B4 gate for the fused selected-expert PARO FFN megakernel.

    One launch per (token, expert) computes the whole selected expert FFN
    (rotate1 -> dual gate/up AWQ GEMV -> silu*mul -> down-rotate -> down AWQ
    GEMV) with both incoherence rotations and the ffn-wide intermediate held
    on-chip, writing the per-selected-row down projection into
    ``scratch.down_out``.  Replaces the 3-call/~5-launch unfused selected
    sub-chain.  Default off; enable with ``HIPENGINE_PARO_FFN_MEGAKERNEL=1``.
    Verify-path only (tokens > 1); AR decode (tokens=1) keeps the unfused
    path so the AR baseline is untouched.
    """

    return _env_enabled("HIPENGINE_PARO_FFN_MEGAKERNEL", default=False)


def _paro_ffn_megakernel_library(library=None):
    """Resolve the fused PARO FFN megakernel .so, cached process-wide."""

    if isinstance(library, dict):
        lib = library.get("paro_ffn_fused")
        if lib is not None:
            return lib
    global _PARO_FFN_MEGAKERNEL_LIBRARY
    if _PARO_FFN_MEGAKERNEL_LIBRARY is None:
        from hipengine.kernels.hip_gfx1100.quant.paro_moe_ffn_fused import (
            build_paro_moe_ffn_fused,
        )

        _PARO_FFN_MEGAKERNEL_LIBRARY = build_paro_moe_ffn_fused(load=True)
    return _PARO_FFN_MEGAKERNEL_LIBRARY


def _shared_expert_fused_rotate_enabled() -> bool:
    """M13.B.2 gate for the shared-expert fused rotate + dual_pack8 GEMV path.

    Replaces ``paro_rotate2_fp16 + gemv_awq_dual_pack8_transposed_fp16``
    with the HBM-staged keyed-barrier kernel
    ``gemv_awq_dual_pack8_transposed_rotate_staged_keyed_fp16``.  Unlike
    the M13.B.1 selected variant, this kernel uses an HBM staging buffer
    so each row rotates exactly once (no per-block redundancy), and the
    M14.fuse.barrier keyed counter avoids the per-launch
    ``hipMemsetAsync(barrier, 0, 8)`` that cancelled the original M13.B.2
    dispatch saving.

    Default remains ``off`` until a fresh W7900 exact/rocprof row proves
    the expected net launch reduction and no hidden barrier-spin loss.
    Enable via ``HIPENGINE_SHARED_EXPERT_FUSED_ROTATE={1,on,yes,true}``.
    """

    return _env_enabled("HIPENGINE_SHARED_EXPERT_FUSED_ROTATE", default=False)


def _w4_output_tiled_prefill_enabled() -> bool:
    """M16.4: route the small verifier batch (rows in {2,4,8}) for single-output
    W4 projections through the weight-amortized output-column-tiled GEMV
    (``gemv_awq_pack8_output_tiled_(transposed_)fp16``) instead of the WMMA
    ``awq_fusedw4_prefill_*`` small-batch kernel.

    The output-tiled GEMV is bit-identical to the per-row strided/transposed
    pack8 GEMV (``tests/test_paro_awq_output_tiled_gemv.py``) -- i.e. byte-exact
    to AR's rows==1 projection -- so exact-AR cannot regress; only throughput is
    at stake.  The WMMA prefill kernel starves at tokens=4 (the B+1 verifier
    shape), so the amortized GEMV is the M16.4 lever.

    Default-on (2026-06-09): W7900/gfx1100 B=3 verifier rocprof shows the
    single-output W4 projections move from ``awq_fusedw4_prefill_fp16`` to the
    byte-exact output-tiled GEMV, cutting verifier kernel time ~-5.6% at
    decode-tokens=8 (17.02 -> 16.07 ms/pass) / ~-10.7% at decode-tokens=4
    (13.60 -> 12.15) with exact-AR preserved.  Opt out with ``=0``.
    """
    return _env_enabled("HIPENGINE_W4_OUTPUT_TILED_PREFILL", default=True)


_W4_DUAL_OUTPUT_TILED_SPLIT_DEFAULT_SITES = frozenset({"shared_gate_up"})


def _w4_dual_output_tiled_split_prefill_enabled(site: str) -> bool:
    """M16.4 follow-up: split-output dual output-tiled W4 prefill path.

    Default-on for the prompt-suite-safe ``shared_gate_up`` site after the
    2026-06-11 W7900 D32 9-prompt gate: stacked with linear out cast+rotate,
    exact ``9/9`` and same acceptance as the off baseline, with mean verify
    wall ``22.98 -> 22.37 ms/cycle``. Opt out with
    ``HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL=0``.
    """

    if not _env_enabled("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", default=True):
        return False
    raw = os.environ.get("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_SITES")
    if raw is None or raw.strip() == "":
        sites = _W4_DUAL_OUTPUT_TILED_SPLIT_DEFAULT_SITES
    else:
        sites = {part.strip().lower() for part in raw.split(",") if part.strip()}
        if "none" in sites:
            return False
        if "all" in sites:
            return True
    return site.lower() in sites


def _w4_dual_output_tiled_split_site_eligible(site: str, tokens: int, in_features: int, group_size: int) -> bool:
    return (
        _w4_output_tiled_prefill_enabled()
        and _w4_dual_output_tiled_split_prefill_enabled(site)
        and int(tokens) in _PACK8_OUTPUT_TILED_ROWS
        and int(in_features) % int(group_size) == 0
    )


def _w4_multi_row_pack8_enabled() -> bool:
    """M12.6 umbrella gate for multi-row pack8 W4 GEMV.

    Defaults to ``on``.  Disable via ``HIPENGINE_W4_MULTI_ROW_PACK8={0,off,no,false}``.
    More specific gates can override single-output or dual-output dispatch:
    ``HIPENGINE_W4_MULTI_ROW_PACK8_SINGLE`` and
    ``HIPENGINE_W4_MULTI_ROW_PACK8_DUAL``.
    """

    return _env_enabled("HIPENGINE_W4_MULTI_ROW_PACK8", default=True)


def _w4_multi_row_single_enabled() -> bool:
    return _env_enabled(
        "HIPENGINE_W4_MULTI_ROW_PACK8_SINGLE",
        default=_w4_multi_row_pack8_enabled(),
    )


def _w4_multi_row_dual_enabled() -> bool:
    return _env_enabled(
        "HIPENGINE_W4_MULTI_ROW_PACK8_DUAL",
        default=_w4_multi_row_pack8_enabled(),
    )


_W4_MULTI_ROW_DEFAULT_SAFE_SITES = frozenset(
    {
        # Passed the llama.cpp-compatible 9-prompt exactness suite on 2026-05-22.
        "full_qk",
        "linear_qkv_z",
        "dense_gate_up",
        "single_full_o",
        "single_shared_down",
        "single_dense_down",
        # Promoted on 2026-06-11 after the current MTP stack's 9-prompt D32
        # exact suite passed and the default translation prompt mismatch was
        # restored to exact AR.
        "single_linear_out",
        # Promoted later on 2026-06-11 after a fresh default A/B stayed exact
        # and moved D32 suite wall/verify plus the locked verifier profile down.
        "single_full_v",
    }
)


def _w4_multi_row_site_enabled(site: str) -> bool:
    """M12.6 per-callsite correctness mask.

    Full M12.6 (all sites enabled) improves the stable quicksort prompt but the
    new prompt suite found exact-AR mismatches in numerically fragile sites.
    With ``HIPENGINE_W4_MULTI_ROW_PACK8_SITES`` unset, only the exact-suite-safe
    subset in ``_W4_MULTI_ROW_DEFAULT_SAFE_SITES`` is enabled.  Override values:
    comma-separated site names, ``all`` for risky/full M12.6, or ``none`` to
    disable every multi-row W4 site while leaving the umbrella env gate on.
    """

    raw = os.environ.get("HIPENGINE_W4_MULTI_ROW_PACK8_SITES")
    if raw is None or raw.strip() == "":
        return site.lower() in _W4_MULTI_ROW_DEFAULT_SAFE_SITES
    sites = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not sites:
        return site.lower() in _W4_MULTI_ROW_DEFAULT_SAFE_SITES
    if "all" in sites:
        return True
    if "none" in sites:
        return False
    return site.lower() in sites


def _w4_multi_row_single_site(prefix: str) -> str:
    if prefix.endswith(".self_attn.v_proj"):
        return "single_full_v"
    if prefix.endswith(".self_attn.o_proj"):
        return "single_full_o"
    if prefix.endswith(".linear_attn.out_proj"):
        return "single_linear_out"
    if prefix.endswith(".mlp.shared_expert.down_proj"):
        return "single_shared_down"
    if prefix.endswith(".mlp.down_proj"):
        return "single_dense_down"
    return "single_other"


def _w4_multi_row_single_site_enabled(prefix: str) -> bool:
    return _w4_multi_row_single_enabled() and _w4_multi_row_site_enabled(
        _w4_multi_row_single_site(prefix)
    )


def _w4_down_proj_small_batch_mode(site: str) -> str:
    """Dispatch mode for verifier-sized W4 down projections.

    Defaults to ``multi_row_decode`` because it passed the 27B DFlash
    exact-suite gate while preserving the standard row-wise pack8 GEMV
    dequantization and sharing weights across verifier rows.  ``gemv`` keeps the
    old row-wise exact fallback.  ``multi_row`` remains diagnostic only: it is
    faster, but currently fails branch-copy exactness for dense down because it
    follows the FP16 prefill-WMMA dequantization path.  Site filtering
    reuses ``HIPENGINE_W4_MULTI_ROW_PACK8_SITES`` so experiments can isolate
    ``single_shared_down`` vs ``single_dense_down``.
    """

    raw = os.environ.get("HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH")
    mode = "multi_row_decode" if raw is None or raw.strip() == "" else raw.strip().lower()
    aliases = {
        "0": "prefill",
        "off": "prefill",
        "false": "prefill",
        "no": "prefill",
        "decode": "gemv",
        "decode_gemv": "gemv",
        "single": "gemv",
        "single_gemv": "gemv",
        "1": "multi_row_decode",
        "on": "multi_row_decode",
        "true": "multi_row_decode",
        "yes": "multi_row_decode",
        "multi_row_exact": "multi_row_decode",
        "decode_multi_row": "multi_row_decode",
        "multi_row_gemv": "multi_row_decode",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"prefill", "gemv", "multi_row", "multi_row_decode"}:
        raise ValueError("HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH must be prefill, gemv, multi_row, or multi_row_decode")
    if mode != "prefill" and not _w4_multi_row_site_enabled(site):
        return "prefill"
    return mode


def _w4_multi_row_dual_site_eligible(site: str, tokens: int, in_features: int, group_size: int) -> bool:
    """M12.6: shared eligibility check for multi-row dual W4 GEMV."""

    return (
        _w4_multi_row_dual_enabled()
        and _w4_multi_row_site_enabled(site)
        and 1 < int(tokens) <= 8
        and int(in_features) % int(group_size) == 0
    )


def _fused_rmsnorm_rotate_enabled() -> bool:
    """M15.4: fuse the linear-attention input RMSNorm + paro_rotate2 into one
    launch on the verifier path (tokens > 1).

    The fused kernel is bit-identical to ``input_rmsnorm_fp16`` followed by
    ``rotate_linear_attention_inputs_fp16`` (see
    ``tests/test_paro_rmsnorm_rotate2.py``), so exact-AR cannot regress; it just
    removes one launch + one HBM round-trip per linear layer.  Default-off
    pending the verifier economics; opt in with
    ``HIPENGINE_FUSED_RMSNORM_ROTATE=1``.
    """

    return _env_enabled("HIPENGINE_FUSED_RMSNORM_ROTATE", default=False)


def _linear_out_cast_rotate_fused_enabled(tokens: int) -> bool:
    """Fuse verifier linear-attention out-proj FP32->FP16 cast into PARO rotate1.

    Default-on for verifier shapes after the 2026-06-11 W7900 D32 prompt-suite
    gate. The fused kernel is raw-FP16 bit-exact vs ``f32_to_fp16`` followed by
    ``paro_rotate1_fp16`` and removes 30 launches/pass in the B=3 verifier.
    Opt out with ``HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED=0``.
    """

    return int(tokens) > 1 and _env_flag("HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED", True)


def _linear_out_c1_exact_rows_enabled(tokens: int) -> bool:
    """Replay verifier linear-attention out projection through token-1 rows.

    Diagnostic exactness gate for the MTP D64 drift lane.  The retained
    verifier path uses faster multi-row W4 projection for
    ``linear_attn.out_proj``; this opt-in keeps the same public row-shaped
    verifier buffers but replays each row through the serial c1 projection
    order.
    """

    return int(tokens) > 1 and _env_flag("HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS", False)


def _full_qkv_split_key_fused_enabled(tokens: int) -> bool:
    """Fuse verifier full-attention Q/Gate split with the FP16->FP32 key cast.

    The fused kernel writes the same FP32 query, FP32 key, and FP16 gate buffers
    as ``qwen35_split_qgate_fp16`` followed by ``fp16_to_f32``. It is verifier-
    only so same-session AR remains on the old c=1 launch chain. Default-off:
    the 2026-06-11 profile removed 10 launches/pass, but two exact 9-prompt
    D32 A/B pairs both regressed aggregate wall/verify. Opt in with
    ``HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED=1`` for diagnostics.
    """

    return int(tokens) > 1 and _env_flag("HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED", False)


def _fused_rmsnorm_rotate2_shape_ok(tokens: int, hidden: int, group_size: int) -> bool:
    if int(tokens) <= 0 or int(group_size) <= 0 or int(group_size) % 2 != 0:
        return False
    half_group = int(group_size) // 2
    if half_group <= 0 or 256 % half_group != 0:
        return False
    if int(hidden) % int(group_size) != 0:
        return False
    groups_per_batch = 256 // half_group
    return (int(hidden) // int(group_size)) % groups_per_batch == 0


def _w4_multi_row_small_batch_enabled() -> bool:
    """M15.1: route the small-batch (``2 <= tokens <= threshold``) verifier
    projections through the weight-amortized multi-row *decode* GEMV instead of
    per-row single GEMVs.

    ``gemv_awq_pack8_multi_row_decode_transposed_fp16`` streams each weight tile
    from HBM exactly once and accumulates all ``B+1`` verifier rows, vs the
    per-row ``gemv_awq_pack8_transposed_fp16`` whose ``grid=(out_pack, row)``
    re-streams the tile for every row.  The decode variant is *bit-identical* to
    the per-row kernel (``tests/test_paro_awq_gemv_multi_row_decode.py``), so
    exact-AR cannot regress; only throughput is at stake.

    Default-on (2026-06-08): W7900 B=3 verifier rocprof shows the
    ``w4_single_gemv`` projection family drop ``2.460 -> 1.873 ms/pass``
    (-23.9%) with launches/pass unchanged (981); economics ``verify_ms/cycle``
    falls -2.4% at B=3 and -4.0% at B=5 (the win scales with row count, as
    weight-amortization predicts). Disable via
    ``HIPENGINE_W4_MULTI_ROW_SMALL_BATCH={0,off,no,false}``.
    """

    return _env_enabled("HIPENGINE_W4_MULTI_ROW_SMALL_BATCH", default=True)


def _w4_multi_row_small_batch_site_enabled(site: str) -> bool:
    """Per-site gate for the M15.1 small-batch multi-row decode path.

    Reuses the M12.6 umbrella + site mask so experiments can isolate sites, but
    is independent of the WMMA-numerics safe mask because the decode kernel is
    bit-exact for every site it covers.
    """

    return (
        _w4_multi_row_small_batch_enabled()
        and _w4_multi_row_single_enabled()
        and _w4_multi_row_site_enabled(site)
    )


# M15.2: sites where the bit-exact multi-row Marlin-K GEMV (preserves AR's
# rows==1 per-row accumulation order) weight-amortizes the verifier projection
# across the B+1 rows in ``project_pack8_fp16``.  This is exact for *every*
# prompt (unlike the decode-dequant multi-row, which matches AR's dequant
# formula but reorders the fp32 reduction and flips top-1 on fragile prompts
# such as ``translation`` for single_full_v/single_linear_out).  A site is added
# here (default-on) only after exact-AR passes on the 9-prompt suite.
_MARLIN_K_MULTI_ROW_DEFAULT_SITES: frozenset[str] = frozenset()


def _marlin_k_multi_row_site_enabled(prefix: str) -> bool:
    """M15.2: route the small verifier batch (rows 2..8) for a Marlin-K
    (``qweight_mk``) projection through the multi-row Marlin-K GEMV.

    The multi-row kernel reads each weight element once and FMAs into all B+1
    rows in the *same* k-order and reduction the single-row Marlin-K kernel uses,
    so each verifier row is bit-identical to AR's rows==1 Marlin-K output.

    Gated by ``HIPENGINE_W4_MULTI_ROW_SMALL_BATCH`` (umbrella) plus the site list
    ``HIPENGINE_MARLIN_K_MULTI_ROW_SITES`` (comma list, ``all`` or ``none``);
    unset uses the validated default set above.
    """

    if not _w4_multi_row_small_batch_enabled():
        return False
    raw = os.environ.get("HIPENGINE_MARLIN_K_MULTI_ROW_SITES")
    if raw is None or raw.strip() == "":
        sites = _MARLIN_K_MULTI_ROW_DEFAULT_SITES
    else:
        parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
        if "none" in parts:
            return False
        if "all" in parts:
            return True
        sites = parts
    return _w4_multi_row_single_site(normalize_qwen35_weight_name(prefix)) in sites


def _small_batch_decode_threshold() -> int:
    """Largest ``tokens`` value that still routes through the decode-style GEMV /
    fused-silu kernel chain instead of the multi-token prefill kernel.

    The fp16 dispatch sites in ``project_full_attention_qkv_fp16``,
    ``project_linear_attention_qkv_z_fp16``, ``shared_expert_paro_w4_fp16``,
    and ``dense_mlp_paro_w4_fp16`` were originally gated
    ``if tokens == 1 … else awq_fusedw4_prefill_*``.  M7.0 (rocprofv3 baseline)
    showed those prefill kernels firing at ~123 / ~60 µs per launch for the
    B=3 verifier (tokens=4), 60 launches/pass each, costing ~11 ms / pass.
    The bf16 sibling paths (``shared_expert_paro_w4_bf16`` etc.) already use
    the same GEMV kernel for every ``tokens`` value — the kernels accept
    ``rows > 1`` (grid is ``(out_packed_a + out_packed_b, row)``).

    The threshold controls the changeover.  Default ``7`` matches llama.cpp’s
    ``MMVF_MAX_BATCH_SIZE = 8`` minus one (the verifier B=3 batch is 4 rows;
    keep headroom for B ∈ {4, 5, 6}).  Override via
    ``HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD``.  Set to ``1`` to restore the
    pre-M7.C behavior exactly.
    """

    value = os.environ.get("HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD")
    if value is None or value.strip() == "":
        return 7
    parsed = int(value)
    if parsed < 1:
        return 1
    return parsed


def _linear_ab_prefill_rocblas_min_tokens() -> int:
    value = os.environ.get("HIPENGINE_LINEAR_AB_PREFILL_ROCBLAS_MIN_TOKENS")
    if value is None or value.strip() == "":
        return 0
    return max(0, int(value))


def _use_linear_ab_prefill_rocblas(tokens: int) -> bool:
    threshold = _linear_ab_prefill_rocblas_min_tokens()
    return threshold > 0 and tokens >= threshold


def _use_verify_dense_gemv_wmma(tokens: int, in_features: int) -> bool:
    return (
        _env_enabled("HIPENGINE_VERIFY_DENSE_GEMV_WMMA", default=False)
        and 1 < int(tokens) <= 16
        and (int(in_features) % 16) == 0
    )


def _use_linear_ab_dual_separate(tokens: int) -> bool:
    return (
        _env_enabled("HIPENGINE_LINEAR_AB_DUAL_SEPARATE", default=True)
        and 1 < int(tokens) <= _small_batch_decode_threshold()
    )


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
