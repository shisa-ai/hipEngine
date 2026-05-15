"""Qwen3.5/PARO runtime-state scaffolding."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    qwen35_full_attn_decode_context_bf16,
    qwen35_full_attn_gate_mul_bf16,
    qwen35_full_attn_gate_mul_fp16,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
    qwen35_write_paged_kv_mixed_value_fp16_batch_spans,
    qwen35_write_paged_kv_mixed_value_fp16_spans,
)
from hipengine.kernels.hip_gfx1100.convert import bf16_to_f32, f32_to_bf16, f32_to_fp16, fp16_to_f32
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
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_gdn_prefill_rmsnorm_gate_fp16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16,
    qwen35_linear_attn_prefill_prepare_f32_bf16,
    qwen35_linear_attn_prefill_prepare_f32_fp16,
)
from hipengine.kernels.hip_gfx1100.norm import (
    paro_add_rmsnorm_out_bf16,
    paro_add_rmsnorm_out_fp16,
    paro_rmsnorm_out_bf16,
    paro_rmsnorm_out_fp16,
)
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import w8a16_linear_bf16_lowp_out, w8a16_linear_fp16_lowp_out
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    qwen35_moe_group_count,
    qwen35_moe_group_prefix,
    qwen35_moe_group_scatter_gather_lowp,
    qwen35_moe_wmma_tile_map,
)
from hipengine.kernels.hip_gfx1100.moe.router import qwen35_router_topk_shared_out_bf16, qwen35_router_topk_shared_out_fp16
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
    gemv_awq_dual_pack8_transposed_bf16,
    gemv_awq_dual_pack8_transposed_fp16,
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
    qwen35_split_qgate_bf16,
    qwen35_split_qgate_fp16,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.qwen35_paro import Qwen35ParoLayerDeviceWeights, normalize_qwen35_weight_name
from hipengine.runtime.workspace import RuntimeWorkspace


@dataclass(frozen=True)
class Qwen35ParoAttentionScratch:
    attn_input: Tensor
    q_rot: Tensor
    k_rot: Tensor
    v_rot: Tensor
    q_proj_key: Tensor
    q_proj: Tensor
    key_bf16: Tensor
    query_raw: Tensor
    key_raw: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
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
    shared_up: Tensor
    shared_intermediate: Tensor
    shared_out: Tensor
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
    shared_up: Tensor
    shared_intermediate: Tensor
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

    @property
    def config(self):
        return self.layer_weights.config

    def tensor(self, name: str) -> Tensor:
        return self.layer_weights.tensor(name)

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
        return Qwen35ParoAttentionScratch(
            attn_input=self.workspace.reserve_tensor("attn.input", (tokens, cfg.hidden_size), lowp),
            q_rot=self.workspace.reserve_tensor("attn.q_rot", (tokens, cfg.hidden_size), lowp),
            k_rot=self.workspace.reserve_tensor("attn.k_rot", (tokens, cfg.hidden_size), lowp),
            v_rot=self.workspace.reserve_tensor("attn.v_rot", (tokens, cfg.hidden_size), lowp),
            q_proj_key=q_proj_key,
            q_proj=q_proj,
            key_bf16=key_bf16,
            query_raw=self.workspace.reserve_tensor("attn.query_raw", (tokens, cfg.num_attention_heads, cfg.head_dim), DType.FP32),
            key_raw=self.workspace.reserve_tensor("attn.key_raw", (tokens, cfg.num_key_value_heads, cfg.head_dim), DType.FP32),
            query=self.workspace.reserve_tensor("attn.query", (tokens, cfg.num_attention_heads, cfg.head_dim), DType.FP32),
            key=self.workspace.reserve_tensor("attn.key", (tokens, cfg.num_key_value_heads, cfg.head_dim), DType.FP32),
            value=self.workspace.reserve_tensor("attn.value", (tokens, cfg.num_key_value_heads, cfg.head_dim), lowp),
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
        qweight = self.tensor(f"{prefix}.qweight")
        qzeros = self.tensor(f"{prefix}.qzeros")
        scales = self.tensor(f"{prefix}.scales")
        if not qweight.shape:
            raise ValueError(f"{prefix}.qweight must have at least one dimension")
        gemv_awq_pack8_strided_bf16(
            x.ptr,
            qweight.ptr,
            qzeros.ptr,
            scales.ptr,
            out.ptr,
            rows,
            x.shape[-1] if in_features is None else in_features,
            _out_packed_from_strided_qweight(qweight),
            group_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
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
        qweight = self.tensor(f"{prefix}.qweight")
        qzeros = self.tensor(f"{prefix}.qzeros")
        scales = self.tensor(f"{prefix}.scales")
        if not qweight.shape:
            raise ValueError(f"{prefix}.qweight must have at least one dimension")
        gemv_awq_pack8_strided_fp16(
            x.ptr,
            qweight.ptr,
            qzeros.ptr,
            scales.ptr,
            out.ptr,
            rows,
            x.shape[-1] if in_features is None else in_features,
            _out_packed_from_strided_qweight(qweight),
            group_size,
            threads=threads,
            stream=stream,
            library=_library_for(library, "awq"),
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
        q_qweight = self.tensor(f"{q}.qweight_pack8_decode")
        k_qweight = self.tensor(f"{k}.qweight_pack8_decode")
        q_out_packed = _out_packed_from_generic_transposed_qweight(q_qweight)
        k_out_packed = _out_packed_from_generic_transposed_qweight(k_qweight)
        awq_library = _library_for(library, "awq")
        if tokens == 1:
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
                library=_library_for(library, "awq"),
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
        if decode_spans.max_live_count <= 4096:
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
            gated = self.decode_full_attention_gqa_gate_bf16(
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
        q_qweight = self.tensor(f"{q}.qweight_pack8_decode")
        k_qweight = self.tensor(f"{k}.qweight_pack8_decode")
        q_out_packed = _out_packed_from_generic_transposed_qweight(q_qweight)
        k_out_packed = _out_packed_from_generic_transposed_qweight(k_qweight)
        awq_library = _library_for(library, "awq")
        if tokens == 1:
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
        self.project_pack8_fp16(
            scratch.v_rot,
            scratch.value,
            weight_prefix=f"{prefix}.v_proj",
            rows=tokens,
            group_size=group_size,
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
        qwen35_write_paged_kv_mixed_value_fp16_batch_spans(
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
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans(
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
        attention_scratch = attention_scratch or self.reserve_full_attention_scratch(
            tokens=tokens,
            num_splits=num_splits,
            activation_dtype=DType.FP16,
            gated_dtype=DType.FP16,
        )
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
        if decode_spans.max_live_count <= 4096:
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
            gated = self.decode_full_attention_gqa_gate_fp16(
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
        return self.run_moe_c1_fp16(
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
        if tokens == 1:
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
                library=_library_for(library, "awq"),
                runtime=self.runtime,
            )
        else:
            # The dual GEMV writes row-major [qkv,z] per token.  Native
            # prefill conv/GDN consumes contiguous [tokens,qkv] and [tokens,z]
            # streams, so split multi-token prefill into two projections.
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
        if tokens < cfg.linear_conv_kernel_dim:
            raise ValueError("native linear-attention prefill requires tokens >= linear_conv_kernel_dim")
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
        qwen35_linear_attn_conv_prefill_f32(
            scratch.qkv_f32.ptr,
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
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
        stream: int = 0,
    ) -> Tensor:
        linear_scratch = linear_scratch or self.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
        if tokens == 1:
            moe_scratch = moe_scratch or self.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
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
        if tokens == 1:
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
        qwen35_router_topk_shared_out_fp16(
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
            library=_library_for(library, "router"),
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
            library=_library_for(library, "w8a16"),
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
            library=_library_for(library, "w8a16"),
            runtime=self.runtime,
        )
        return scratch.shared_out

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
        shared = self.shared_expert_w8a16_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
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
        shared = self.shared_expert_w8a16_fp16(hidden, scratch, tokens=tokens, library=library, stream=stream)
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
        qwen35_router_topk_shared_out_bf16(
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
            library=_library_for(library, "router"),
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
        scratch: Qwen35ParoMoeScratch,
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
        shared = self.shared_expert_w8a16_bf16(hidden, scratch, tokens=tokens, library=library, stream=stream)
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
        shared = self.shared_expert_w8a16_bf16(hidden, scratch, tokens=tokens, library=library, stream=stream)
        return self.combine_moe_c1_shared_residual_bf16(
            scratch,
            shared=shared,
            residual=residual,
            tokens=tokens,
            library=library,
            stream=stream,
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
            shared_out=self.workspace.reserve_tensor("moe.grouped.shared_out", (tokens, cfg.hidden_size), lowp),
            moe_out=self.workspace.reserve_tensor("moe.grouped.out", (tokens, cfg.hidden_size), lowp),
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
            shared_out=self.workspace.reserve_tensor("moe.shared_out", (tokens, cfg.hidden_size), lowp),
            moe_out=self.workspace.reserve_tensor("moe.out", (tokens, cfg.hidden_size), lowp),
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



def _library_for(library, family: str):
    if isinstance(library, dict):
        return library.get(family)
    return library

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


def _linear_value_width(config) -> int:
    return int(config.linear_num_value_heads) * int(config.linear_value_head_dim)


def _linear_qkv_width(config) -> int:
    return 2 * int(config.linear_num_key_heads) * int(config.linear_key_head_dim) + _linear_value_width(config)


def _rotation_krot(pairs: Tensor) -> int:
    if not pairs.shape:
        raise ValueError("rotation pairs tensor must have at least one dimension")
    return pairs.shape[0]
