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
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.qwen35_paro import Qwen35ParoLayerDeviceWeights
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
    shared_up: Tensor
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
            shared_up=self.workspace.reserve_tensor(
                "moe.shared_up",
                (tokens, 2 * cfg.shared_expert_intermediate_size),
                DType.BF16,
            ),
            moe_out=self.workspace.reserve_tensor("moe.out", (tokens, cfg.hidden_size), DType.BF16),
        )

    def free(self) -> None:
        self.workspace.free()
        self.layer_weights.free(runtime=self.runtime)
