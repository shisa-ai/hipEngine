"""Qwen3.5/PARO runtime-state scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
import os

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.rocblas import rocblas_gemm_ex_rowmajor_nt_fp16_compute_f32
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    aotriton_attn_fwd_v3_compact_varlen,
    aotriton_gate_mul_bf16_to_fp16,
    qwen35_full_attn_decode_context_bf16,
    qwen35_full_attn_gate_mul_bf16,
    qwen35_full_attn_gate_mul_fp16,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans,
    qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
    qwen35_write_paged_kv_mixed_value_fp16_batch_spans,
    qwen35_write_paged_kv_mixed_value_fp16_prompt_spans,
    qwen35_write_paged_kv_mixed_value_fp16_spans,
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
    gemv_awq_dual_pack8_multi_row_split_transposed_fp16,
    gemv_awq_pack8_multi_row_decode_transposed_fp16,
    gemv_awq_pack8_multi_row_strided_fp16,
    gemv_awq_pack8_multi_row_transposed_fp16,
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
    gemv_awq_selected_dual_pack8_transposed_rotate_out_fp16,
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
from hipengine.runtime.moe_c1_dispatch import moe_c1_c_dispatch_enabled
from hipengine.runtime.workspace import RuntimeWorkspace


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
        # M14.dispatch.1-beta: lazy per-layer cache for the C-side MoE C1
        # dispatcher.  Key: layer_kind ('linear_attention' | 'full_attention').
        # Populated on first matching call from run_moe_c1_fp16.
        self._moe_c1_dispatch_cache: object | None = None  # MoeC1DispatchCache
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
        return self.layer_weights.tensor(name)

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
            query=self.workspace.reserve_tensor("attn.query", (tokens, cfg.num_attention_heads, cfg.head_dim), DType.FP32),
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
            elif rows > 1 and group_size % 16 == 0 and width % group_size == 0:
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
            # M7.C investigation (2026-05-21): bumping this to
            # ``rows > _small_batch_decode_threshold()`` is functionally correct in
            # isolation but the resulting cache shift adds ~+3 ms in downstream
            # MoE/GDN kernels.  Re-enable once M7.C.6 unlocks the dual-GEMV reach.
            if (
                rows > 1
                and rows <= 8
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
            elif rows > 1 and group_size % 16 == 0 and width % group_size == 0:
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
            tree_conv_state=self.workspace.reserve_tensor(
                "linear_attn.tree_conv_state",
                (tokens, qkv_width, cfg.linear_conv_kernel_dim),
                DType.FP32,
            ),
            tree_recurrent_state=self.workspace.reserve_tensor(
                "linear_attn.tree_recurrent_state",
                (tokens, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim),
                DType.FP32,
            ),
            tree_gdn_acc=self.workspace.reserve_tensor("linear_attn.tree_gdn_acc", (tokens, z_width), DType.FP32),
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
        qwen35_write_paged_kv_mixed_value_bf16_spans(
            scratch.key.ptr,
            scratch.value.ptr,
            key_cache.ptr,
            value_cache.ptr,
            spans,
            block_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
            stream=stream,
            library=_library_for(library, "kv"),
            runtime=self.runtime,
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
        if not _use_full_attention_split_decode(decode_spans.max_live_count):
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
        elif tokens <= _small_batch_decode_threshold():
            # M7.C.6: two single GEMVs, one for Q (writes scratch.q_proj.ptr)
            # and one for K (writes scratch.key_bf16.ptr).  Mirrors the bf16
            # sibling at project_linear_attention_qkv_z_bf16 line 1090+ — the
            # views' contiguous strides match the single-GEMV row strides, so
            # the downstream qwen35_split_qgate / attention path reads correct
            # rows.  scratch.q_proj_key (the combined view) is intentionally
            # left inconsistent here; nothing downstream reads it directly.
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
        fp16_to_f32(
            scratch.key_bf16.ptr,
            scratch.key_raw.ptr,
            tokens * kv_width,
            stream=stream,
            library=_library_for(library, "cast"),
            runtime=self.runtime,
        )
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
        return query_out, scratch.key, scratch.value, scratch.gate

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
        qwen35_write_paged_kv_mixed_value_fp16_spans(
            scratch.key.ptr,
            scratch.value.ptr,
            key_cache.ptr,
            value_cache.ptr,
            spans,
            block_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
            stream=stream,
            library=_library_for(library, "kv"),
            runtime=self.runtime,
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

    def prefill_full_attention_gqa_gate_tree_fp16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        rows: int,
        ancestor_mask: Tensor,
        tree_committed_count: int,
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
        if tree_committed_count < 0:
            raise ValueError("tree_committed_count must be non-negative")
        gate_tensor = scratch.gate if gate is None else gate
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            gate_tensor.ptr,
            scratch.gated_attn.ptr,
            spans,
            ancestor_mask.ptr,
            tree_committed_count,
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
        if scratch.query.dtype is not DType.FP32 or scratch.key.dtype is not DType.FP32 or scratch.value.dtype is not DType.FP16:
            raise ValueError("AOTriton prefill expects FP32 Q/K source tensors and FP16 V scratch tensor")
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
            max_seqlen_q=rows,
            max_seqlen_k=key_rows,
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

        qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(
            scratch.key.ptr,
            scratch.value.ptr,
            key_cache.ptr,
            value_cache.ptr,
            spans,
            rows,
            block_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
            stream=stream,
            library=_library_for(library, "kv"),
            runtime=self.runtime,
        )

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
        _query, _key, _value, gate = self.prepare_full_attention_qkv_fp16(
            attention_scratch,
            cos_table=cos_table,
            sin_table=sin_table,
            position=position,
            max_positions=max_positions,
            tokens=tokens,
            library=library,
            stream=stream,
        )
        self.append_full_attention_kv_fp16(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=append_spans,
            block_size=block_size,
            library=library,
            stream=stream,
        )
        if not _use_full_attention_split_decode(decode_spans.max_live_count):
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
            aotriton_query_bf16 = self.workspace.reserve_tensor(
                "attn.aotriton_q_bf16",
                attention_scratch.query.shape,
                DType.BF16,
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
        _query, _key, _value, gate = self.prepare_full_attention_qkv_fp16(
            attention_scratch,
            cos_table=cos_table,
            sin_table=sin_table,
            position=positions,
            max_positions=max_positions,
            tokens=tokens,
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
        elif tokens <= _small_batch_decode_threshold():
            # M7.C.6: two single GEMVs, one for QKV (writes scratch.qkv.ptr)
            # and one for Z (writes scratch.z.ptr).  Direct port of the bf16
            # sibling pattern at project_linear_attention_qkv_z_bf16 line 1090+.
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
                if _use_verify_dense_gemv_wmma(tokens, self.config.hidden_size):
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
        library=None,
        stream: int = 0,
    ) -> Tensor:
        scratch = scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        self.rotate_linear_attention_inputs_fp16(hidden, scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_qkv_z_fp16(scratch, tokens=tokens, group_size=group_size, library=library, stream=stream)
        self.project_linear_attention_ab_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        return self.run_linear_attention_prefill_conv_gdn_segments_fp16(
            scratch,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            tokens=tokens,
            segments=segments,
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
                gemv_awq_dual_pack8_transposed_rotate_staged_fp16(
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
                    stream=stream,
                    library=_library_for(library, "awq"),
                    runtime=self.runtime,
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
        cache.dispatch(
            hidden=hidden,
            residual=residual,
            out=target_out,
            scratch=scratch,
            tokens=tokens,
            group_size=group_size,
            stream=stream,
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
            shared_rotate_fuse_barrier=self.workspace.reserve_tensor(
                "moe.grouped.shared_rotate_fuse_barrier", (2,), DType.INT32,
            ),
        )

    def reserve_moe_c1_scratch(
        self,
        *,
        tokens: int = 1,
        activation_dtype: str | DType = DType.BF16,
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
            normed=self.workspace.reserve_tensor("moe.normed", (tokens, cfg.hidden_size), lowp),
            residual=self.workspace.reserve_tensor("moe.residual", (tokens, cfg.hidden_size), lowp),
            gate_up_input=self.workspace.reserve_tensor("moe.gate_up_input", (tokens, cfg.hidden_size), lowp),
            router_logits=self.workspace.reserve_tensor("moe.router_logits", (tokens, cfg.num_experts + 1), DType.FP32),
            routing_weights=self.workspace.reserve_tensor("moe.routing_weights", (tokens, top_k), DType.FP32),
            selected_experts=self.workspace.reserve_tensor("moe.selected_experts", (tokens, top_k), DType.INT64),
            gate_up=self.workspace.reserve_tensor(
                "moe.gate_up",
                (tokens, top_k, 2 * cfg.moe_intermediate_size),
                lowp,
            ),
            down_input=self.workspace.reserve_tensor("moe.down_input", (tokens, top_k, cfg.moe_intermediate_size), lowp),
            down_out=self.workspace.reserve_tensor("moe.down_out", (tokens, top_k, cfg.hidden_size), lowp),
            shared_gate_input=self.workspace.reserve_tensor(
                "moe.shared_gate_input",
                (tokens, cfg.hidden_size),
                lowp,
            ),
            shared_up_input=self.workspace.reserve_tensor(
                "moe.shared_up_input",
                (tokens, cfg.hidden_size),
                lowp,
            ),
            shared_gate_out=self.workspace.reserve_tensor(
                "moe.shared_gate_out",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_up_out=self.workspace.reserve_tensor(
                "moe.shared_up_out",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_up=self.workspace.reserve_tensor(
                "moe.shared_up",
                (tokens, 2 * cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_intermediate=self.workspace.reserve_tensor(
                "moe.shared_intermediate",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_down_input=self.workspace.reserve_tensor(
                "moe.shared_down_input",
                (tokens, cfg.shared_expert_intermediate_size),
                lowp,
            ),
            shared_out=self.workspace.reserve_tensor("moe.shared_out", (tokens, cfg.hidden_size), lowp),
            moe_out=self.workspace.reserve_tensor("moe.out", (tokens, cfg.hidden_size), lowp),
            shared_rotate_fuse_barrier=self.workspace.reserve_tensor(
                "moe.shared_rotate_fuse_barrier", (2,), DType.INT32,
            ),
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


def _shared_expert_fused_rotate_enabled() -> bool:
    """M13.B.2 gate for the shared-expert fused rotate + dual_pack8 GEMV path.

    Replaces ``paro_rotate2_fp16 + gemv_awq_dual_pack8_transposed_fp16``
    with the already-existing HBM-staged kernel
    ``gemv_awq_dual_pack8_transposed_rotate_staged_fp16``.  Unlike the
    M13.B.1 selected variant, this kernel uses an HBM barrier so each row
    rotates exactly once (no per-block redundancy), so it is
    structurally bit-exact and does not blow up kernel time.

    Defaulted ``off`` after measurement: the kernel does save the
    expected 10 ``moe_paro_rotate_in`` launches/pass (one per
    full-attention MoE layer at B=3 batched chain), but its launcher
    does an implicit ``hipMemsetAsync(barrier, 0, 8)`` to reset the
    2-int32 atomic barrier on every call, which rocprof counts as
    another runtime launch.  Net launch delta is therefore ~0 and the
    fused kernel runs ~0.1 ms/pass slower (extra barrier spin on the
    GEMV blocks).  Kept opt-in so a future keyed-barrier variant
    (rotate counter persists, host bumps a uniform value each launch)
    can flip it on without re-plumbing.  Tracked as M14.fuse.barrier.
    Enable via ``HIPENGINE_SHARED_EXPERT_FUSED_ROTATE={1,on,yes,true}``.
    """

    return _env_enabled("HIPENGINE_SHARED_EXPERT_FUSED_ROTATE", default=False)


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
