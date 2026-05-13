"""Qwen3.5/PARO runtime-state scaffolding."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.fused.paro_combine import weighted_sum_shared_gate_combine_residual_out_bf16_f32w
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_dual_out_bf16, silu_mul_dual_rotate_out_bf16
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import w8a16_linear_bf16_lowp_out
from hipengine.kernels.hip_gfx1100.moe.router import qwen35_router_topk_shared_out_bf16
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
    gemv_awq_pack8_strided_bf16,
    gemv_awq_selected_dual_pack8_transposed_bf16,
    gemv_awq_selected_pack8_transposed_bf16,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.qwen35_paro import Qwen35ParoLayerDeviceWeights, normalize_qwen35_weight_name
from hipengine.runtime.workspace import RuntimeWorkspace


@dataclass(frozen=True)
class Qwen35ParoAttentionScratch:
    query: Tensor
    key: Tensor
    value: Tensor
    gate: Tensor
    partial_out: Tensor
    partial_m: Tensor
    partial_l: Tensor
    attn_out: Tensor
    gated_attn: Tensor


@dataclass(frozen=True)
class Qwen35ParoMoeScratch:
    normed: Tensor
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
        gated_dtype: str | DType = DType.BF16,
    ) -> Qwen35ParoAttentionScratch:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if num_splits <= 0:
            raise ValueError("num_splits must be positive")
        cfg = self.config
        q_width = cfg.num_attention_heads * cfg.head_dim
        gated = DType.parse(gated_dtype)
        if gated not in {DType.BF16, DType.FP16, DType.FP32}:
            raise ValueError("gated_dtype must be bf16, fp16, or fp32")
        return Qwen35ParoAttentionScratch(
            query=self.workspace.reserve_tensor("attn.query", (tokens, cfg.num_attention_heads, cfg.head_dim), DType.FP32),
            key=self.workspace.reserve_tensor("attn.key", (tokens, cfg.num_key_value_heads, cfg.head_dim), DType.FP32),
            value=self.workspace.reserve_tensor("attn.value", (tokens, cfg.num_key_value_heads, cfg.head_dim), DType.BF16),
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
            qweight.shape[0],
            group_size,
            threads=threads,
            library=library,
            runtime=self.runtime,
        )
        return out

    def append_full_attention_kv(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        block_size: int = 256,
        library=None,
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
            library=library,
            runtime=self.runtime,
        )

    def decode_full_attention_gqa_gate_bf16(
        self,
        scratch: Qwen35ParoAttentionScratch,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        spans: KVLiveSpans,
        chunk_size: int,
        num_splits: int,
        block_size: int = 256,
        scale: float | None = None,
        library=None,
    ) -> Tensor:
        qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans(
            scratch.query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            scratch.gate.ptr,
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
            scratch.gate.shape[-1],
            1,
            (self.config.head_dim ** -0.5) if scale is None else scale,
            library=library,
            runtime=self.runtime,
        )
        return scratch.gated_attn

    def route_moe_topk_shared_bf16(
        self,
        hidden: Tensor,
        scratch: Qwen35ParoMoeScratch,
        *,
        tokens: int = 1,
        threads: int = 512,
        library=None,
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
            library=library,
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
    ) -> Tensor:
        prefix = f"layers.{self.layer_weights.layer_id}.mlp.experts"
        gate_qweight = self.tensor(f"{prefix}.stacked_gate_qweight_pack8_decode")
        gate_qzeros = self.tensor(f"{prefix}.stacked_gate_qzeros")
        gate_scales = self.tensor(f"{prefix}.stacked_gate_scales")
        up_qweight = self.tensor(f"{prefix}.stacked_up_qweight_pack8_decode")
        up_qzeros = self.tensor(f"{prefix}.stacked_up_qzeros")
        up_scales = self.tensor(f"{prefix}.stacked_up_scales")
        rows = tokens * self.config.num_experts_per_tok
        gemv_awq_selected_dual_pack8_transposed_bf16(
            hidden.ptr,
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
            library=library,
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
            library=library,
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
            library=library,
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
            library=library,
            runtime=self.runtime,
        )
        silu_mul_dual_out_bf16(
            scratch.shared_up.ptr,
            scratch.shared_intermediate.ptr,
            tokens,
            self.config.shared_expert_intermediate_size,
            library=library,
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
            library=library,
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
    ) -> Tensor:
        if tokens != 1:
            raise ValueError("combined MoE c=1 residual helper currently requires tokens=1")
        target = out or scratch.moe_out
        shared_gate_logits_ptr = scratch.router_logits.ptr + self.config.num_experts * DType.FP32.itemsize
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
            library=library,
            runtime=self.runtime,
        )
        return target

    def run_moe_c1_bf16(
        self,
        hidden: Tensor,
        residual: Tensor,
        *,
        scratch: Qwen35ParoMoeScratch | None = None,
        tokens: int = 1,
        group_size: int = 128,
        library=None,
    ) -> Tensor:
        if tokens != 1:
            raise ValueError("MoE c=1 orchestrator currently requires tokens=1")
        scratch = scratch or self.reserve_moe_c1_scratch(tokens=tokens)
        self.route_moe_topk_shared_bf16(hidden, scratch, tokens=tokens, library=library)
        self.selected_moe_gate_up_pack8_bf16(hidden, scratch, tokens=tokens, group_size=group_size, library=library)
        self.activate_rotate_moe_down_bf16(scratch, tokens=tokens, group_size=group_size, library=library)
        self.selected_moe_down_pack8_bf16(scratch.down_input, scratch, tokens=tokens, group_size=group_size, library=library)
        shared = self.shared_expert_w8a16_bf16(hidden, scratch, tokens=tokens, library=library)
        return self.combine_moe_c1_shared_residual_bf16(
            scratch,
            shared=shared,
            residual=residual,
            tokens=tokens,
            library=library,
        )

    def reserve_moe_c1_scratch(self, *, tokens: int = 1) -> Qwen35ParoMoeScratch:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        cfg = self.config
        top_k = cfg.num_experts_per_tok
        if top_k <= 0:
            raise ValueError("config.num_experts_per_tok must be positive")
        return Qwen35ParoMoeScratch(
            normed=self.workspace.reserve_tensor("moe.normed", (tokens, cfg.hidden_size), DType.BF16),
            router_logits=self.workspace.reserve_tensor("moe.router_logits", (tokens, cfg.num_experts), DType.FP32),
            routing_weights=self.workspace.reserve_tensor("moe.routing_weights", (tokens, top_k), DType.FP32),
            selected_experts=self.workspace.reserve_tensor("moe.selected_experts", (tokens, top_k), DType.INT32),
            gate_up=self.workspace.reserve_tensor(
                "moe.gate_up",
                (tokens, top_k, 2 * cfg.moe_intermediate_size),
                DType.BF16,
            ),
            down_input=self.workspace.reserve_tensor("moe.down_input", (tokens, top_k, cfg.moe_intermediate_size), DType.BF16),
            down_out=self.workspace.reserve_tensor("moe.down_out", (tokens, top_k, cfg.hidden_size), DType.BF16),
            shared_up=self.workspace.reserve_tensor(
                "moe.shared_up",
                (tokens, 2 * cfg.shared_expert_intermediate_size),
                DType.BF16,
            ),
            shared_intermediate=self.workspace.reserve_tensor(
                "moe.shared_intermediate",
                (tokens, cfg.shared_expert_intermediate_size),
                DType.BF16,
            ),
            shared_out=self.workspace.reserve_tensor("moe.shared_out", (tokens, cfg.hidden_size), DType.BF16),
            moe_out=self.workspace.reserve_tensor("moe.out", (tokens, cfg.hidden_size), DType.BF16),
        )

    def free(self) -> None:
        self.workspace.free()
        self.layer_weights.free(runtime=self.runtime)


def _out_packed_from_transposed_qweight(qweight: Tensor) -> int:
    if len(qweight.shape) < 3:
        raise ValueError("transposed stacked qweight must have shape [experts, out_packed, in_features]")
    return qweight.shape[1]


def _rotation_krot(pairs: Tensor) -> int:
    if not pairs.shape:
        raise ValueError("rotation pairs tensor must have at least one dimension")
    return pairs.shape[0]
