"""Host-side Qwen GGUF diagnostic snapshots and explicit buffer readers.

The caller owns token setup, layer execution and synchronization. These readers
borrow scratch and copy selected buffers; they never advance a session or select
kernels. Copy operations are explicit inputs so diagnostic tooling can intercept
the same reads without acquiring a runner or session dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from hipengine.core.dtype import DType
from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION


@dataclass(frozen=True)
class CaptureReaders:
    bf16: Callable
    f32: Callable
    i64: Callable


@dataclass(frozen=True)
class Qwen35GGUFLinearAttentionBoundaryCapture:
    """Host-visible diagnostic snapshot for one GGUF linear-attention boundary."""

    layer_id: int
    token_id: int
    position: int
    hidden_size: int
    ssm_time_step_rank: int
    linear_qkv_width: int
    ssm_inner_size: int
    attn_norm_f32: np.ndarray
    linear_qkv_f32: np.ndarray
    linear_z_f32: np.ndarray
    ssm_alpha_f32: np.ndarray
    ssm_beta_f32: np.ndarray
    conv_out_f32: np.ndarray
    recurrent_out_f32: np.ndarray
    recurrent_bf16_f32: np.ndarray
    attn_out_f32: np.ndarray

    def as_summary_dict(self) -> dict[str, object]:
        return {
            "layer_id": int(self.layer_id),
            "token_id": int(self.token_id),
            "position": int(self.position),
            "hidden_size": int(self.hidden_size),
            "ssm_time_step_rank": int(self.ssm_time_step_rank),
            "linear_qkv_width": int(self.linear_qkv_width),
            "ssm_inner_size": int(self.ssm_inner_size),
            "attn_norm_shape": list(self.attn_norm_f32.shape),
            "linear_qkv_shape": list(self.linear_qkv_f32.shape),
            "linear_z_shape": list(self.linear_z_f32.shape),
            "ssm_alpha_shape": list(self.ssm_alpha_f32.shape),
            "ssm_beta_shape": list(self.ssm_beta_f32.shape),
            "conv_out_shape": list(self.conv_out_f32.shape),
            "recurrent_out_shape": list(self.recurrent_out_f32.shape),
            "recurrent_bf16_shape": list(self.recurrent_bf16_f32.shape),
            "attn_out_shape": list(self.attn_out_f32.shape),
            "finite": bool(
                np.all(np.isfinite(self.attn_norm_f32))
                and np.all(np.isfinite(self.linear_qkv_f32))
                and np.all(np.isfinite(self.linear_z_f32))
                and np.all(np.isfinite(self.ssm_alpha_f32))
                and np.all(np.isfinite(self.ssm_beta_f32))
                and np.all(np.isfinite(self.conv_out_f32))
                and np.all(np.isfinite(self.recurrent_out_f32))
                and np.all(np.isfinite(self.recurrent_bf16_f32))
                and np.all(np.isfinite(self.attn_out_f32))
            ),
        }


@dataclass(frozen=True)
class Qwen35GGUFLinearAttentionLayerCapture:
    """Host-visible diagnostic snapshot for a full attention layer."""

    layer_id: int
    layer_type: str
    token_id: int
    position: int
    hidden_size: int
    is_moe: bool
    top_k: int
    preceding_layer_count: int
    hidden_in_f32: np.ndarray
    attn_norm_f32: np.ndarray
    attn_out_f32: np.ndarray
    post_norm_f32: np.ndarray
    residual_f32: np.ndarray
    ffn_or_moe_down_f32: np.ndarray
    layer_out_f32: np.ndarray
    moe_router_logits_f32: np.ndarray | None = None
    moe_selected_intermediate_f32: np.ndarray | None = None
    moe_shared_intermediate_f32: np.ndarray | None = None
    moe_shared_out_f32: np.ndarray | None = None
    moe_routing_weights_f32: np.ndarray | None = None
    moe_shared_gate_f32: np.ndarray | None = None
    moe_selected_experts_i64: np.ndarray | None = None
    linear_qkv_f32: np.ndarray | None = None
    linear_z_f32: np.ndarray | None = None
    ssm_alpha_f32: np.ndarray | None = None
    ssm_beta_f32: np.ndarray | None = None
    conv_out_f32: np.ndarray | None = None
    recurrent_out_f32: np.ndarray | None = None
    recurrent_bf16_f32: np.ndarray | None = None
    post_norm_source: str = "bf16_scratch.post_norm"

    def as_summary_dict(self) -> dict[str, object]:
        optional_finite = True
        optional_arrays = (
            self.moe_router_logits_f32,
            self.moe_selected_intermediate_f32,
            self.moe_shared_intermediate_f32,
            self.moe_shared_out_f32,
            self.moe_routing_weights_f32,
            self.moe_shared_gate_f32,
            self.linear_qkv_f32,
            self.linear_z_f32,
            self.ssm_alpha_f32,
            self.ssm_beta_f32,
            self.conv_out_f32,
            self.recurrent_out_f32,
            self.recurrent_bf16_f32,
        )
        for array in optional_arrays:
            if array is not None:
                optional_finite = optional_finite and bool(np.all(np.isfinite(array)))
        return {
            "layer_id": int(self.layer_id),
            "layer_type": str(self.layer_type),
            "token_id": int(self.token_id),
            "position": int(self.position),
            "hidden_size": int(self.hidden_size),
            "is_moe": bool(self.is_moe),
            "top_k": int(self.top_k),
            "preceding_layer_count": int(self.preceding_layer_count),
            "hidden_in_shape": list(self.hidden_in_f32.shape),
            "attn_norm_shape": list(self.attn_norm_f32.shape),
            "attn_out_shape": list(self.attn_out_f32.shape),
            "post_norm_shape": list(self.post_norm_f32.shape),
            "post_norm_source": str(self.post_norm_source),
            "residual_shape": list(self.residual_f32.shape),
            "ffn_or_moe_down_shape": list(self.ffn_or_moe_down_f32.shape),
            "moe_router_logits_shape": (
                None
                if self.moe_router_logits_f32 is None
                else list(self.moe_router_logits_f32.shape)
            ),
            "moe_selected_intermediate_shape": (
                None
                if self.moe_selected_intermediate_f32 is None
                else list(self.moe_selected_intermediate_f32.shape)
            ),
            "moe_shared_intermediate_shape": (
                None
                if self.moe_shared_intermediate_f32 is None
                else list(self.moe_shared_intermediate_f32.shape)
            ),
            "moe_shared_out_shape": (
                None
                if self.moe_shared_out_f32 is None
                else list(self.moe_shared_out_f32.shape)
            ),
            "moe_routing_weights_shape": (
                None
                if self.moe_routing_weights_f32 is None
                else list(self.moe_routing_weights_f32.shape)
            ),
            "moe_shared_gate_shape": (
                None
                if self.moe_shared_gate_f32 is None
                else list(self.moe_shared_gate_f32.shape)
            ),
            "moe_selected_experts_shape": (
                None
                if self.moe_selected_experts_i64 is None
                else list(self.moe_selected_experts_i64.shape)
            ),
            "linear_qkv_shape": (
                None if self.linear_qkv_f32 is None else list(self.linear_qkv_f32.shape)
            ),
            "linear_z_shape": (
                None if self.linear_z_f32 is None else list(self.linear_z_f32.shape)
            ),
            "ssm_alpha_shape": (
                None if self.ssm_alpha_f32 is None else list(self.ssm_alpha_f32.shape)
            ),
            "ssm_beta_shape": (
                None if self.ssm_beta_f32 is None else list(self.ssm_beta_f32.shape)
            ),
            "conv_out_shape": (
                None if self.conv_out_f32 is None else list(self.conv_out_f32.shape)
            ),
            "recurrent_out_shape": (
                None
                if self.recurrent_out_f32 is None
                else list(self.recurrent_out_f32.shape)
            ),
            "recurrent_bf16_shape": (
                None
                if self.recurrent_bf16_f32 is None
                else list(self.recurrent_bf16_f32.shape)
            ),
            "layer_out_shape": list(self.layer_out_f32.shape),
            "finite": bool(
                np.all(np.isfinite(self.hidden_in_f32))
                and np.all(np.isfinite(self.attn_norm_f32))
                and np.all(np.isfinite(self.attn_out_f32))
                and np.all(np.isfinite(self.post_norm_f32))
                and np.all(np.isfinite(self.residual_f32))
                and np.all(np.isfinite(self.ffn_or_moe_down_f32))
                and np.all(np.isfinite(self.layer_out_f32))
                and optional_finite
            ),
        }


@dataclass(frozen=True)
class Qwen35GGUFRouterTraceLayerCapture:
    """Host-visible router snapshot for one decoded target layer."""

    layer_id: int
    layer_type: str
    token_id: int
    position: int
    hidden_size: int
    expert_count: int
    top_k: int
    hidden_in_f32: np.ndarray
    layer_out_f32: np.ndarray
    moe_router_logits_f32: np.ndarray
    moe_routing_weights_f32: np.ndarray
    moe_shared_gate_f32: np.ndarray
    moe_selected_experts_i64: np.ndarray

    def as_summary_dict(self) -> dict[str, object]:
        return {
            "layer_id": int(self.layer_id),
            "layer_type": str(self.layer_type),
            "token_id": int(self.token_id),
            "position": int(self.position),
            "hidden_size": int(self.hidden_size),
            "expert_count": int(self.expert_count),
            "top_k": int(self.top_k),
            "hidden_in_shape": list(self.hidden_in_f32.shape),
            "layer_out_shape": list(self.layer_out_f32.shape),
            "moe_router_logits_shape": list(self.moe_router_logits_f32.shape),
            "moe_routing_weights_shape": list(self.moe_routing_weights_f32.shape),
            "moe_shared_gate_shape": list(self.moe_shared_gate_f32.shape),
            "moe_selected_experts_shape": list(self.moe_selected_experts_i64.shape),
            "finite": bool(
                np.all(np.isfinite(self.hidden_in_f32))
                and np.all(np.isfinite(self.layer_out_f32))
                and np.all(np.isfinite(self.moe_router_logits_f32))
                and np.all(np.isfinite(self.moe_routing_weights_f32))
                and np.all(np.isfinite(self.moe_shared_gate_f32))
            ),
        }


def read_router_trace_layer(
    *, layer_id: int, layer_type: str, token_id: int, position: int,
    hidden_size: int, expert_count: int, top_k: int, hidden_in: np.ndarray,
    target_dst_ptr: int, scratch, runtime, readers: CaptureReaders,
) -> Qwen35GGUFRouterTraceLayerCapture:
    shared_gate_ptr = int(scratch.moe_router_logits.ptr) + expert_count * DType.FP32.itemsize
    return Qwen35GGUFRouterTraceLayerCapture(
        layer_id=int(layer_id),
        layer_type=str(layer_type),
        token_id=int(token_id),
        position=int(position),
        hidden_size=hidden_size,
        expert_count=expert_count,
        top_k=top_k,
        hidden_in_f32=np.ascontiguousarray(hidden_in, dtype=np.float32),
        layer_out_f32=readers.bf16(target_dst_ptr, hidden_size, runtime=runtime),
        moe_router_logits_f32=readers.f32(
            int(scratch.moe_router_logits.ptr), expert_count, runtime=runtime,
        ),
        moe_routing_weights_f32=readers.f32(
            int(scratch.moe_routing_weights.ptr), top_k, runtime=runtime,
        ),
        moe_shared_gate_f32=readers.f32(shared_gate_ptr, 1, runtime=runtime),
        moe_selected_experts_i64=readers.i64(
            int(scratch.moe_selected_experts.ptr), top_k, runtime=runtime,
        ),
    )


def read_linear_attention_boundary(*, cfg, hidden_size: int, linear_qkv_width: int,
                                   scratch, layer_id: int, token_id: int, position: int,
                                   runtime, readers: CaptureReaders):
    rank = int(cfg.ssm_time_step_rank)
    ssm_inner_size = int(cfg.ssm_inner_size)
    alpha_ptr = int(scratch.linear_alpha.ptr)
    beta_ptr = int(scratch.linear_beta.ptr)

    return Qwen35GGUFLinearAttentionBoundaryCapture(
        layer_id=int(layer_id),
        token_id=int(token_id),
        position=int(position),
        hidden_size=hidden_size,
        ssm_time_step_rank=rank,
        linear_qkv_width=linear_qkv_width,
        ssm_inner_size=ssm_inner_size,
        attn_norm_f32=readers.bf16(
            int(scratch.norm.ptr), hidden_size, runtime=runtime
        ),
        linear_qkv_f32=readers.bf16(
            int(scratch.linear_qkv.ptr), linear_qkv_width, runtime=runtime
        ),
        linear_z_f32=readers.bf16(
            int(scratch.linear_z.ptr), ssm_inner_size, runtime=runtime
        ),
        ssm_alpha_f32=readers.bf16(alpha_ptr, rank, runtime=runtime),
        ssm_beta_f32=readers.bf16(beta_ptr, rank, runtime=runtime),
        conv_out_f32=readers.f32(
            int(scratch.conv_out.ptr), linear_qkv_width, runtime=runtime
        ),
        recurrent_out_f32=readers.f32(
            int(scratch.recurrent_out.ptr), ssm_inner_size, runtime=runtime
        ),
        recurrent_bf16_f32=readers.bf16(
            int(scratch.recurrent_bf16.ptr), ssm_inner_size, runtime=runtime
        ),
        attn_out_f32=readers.bf16(
            int(scratch.attn_out.ptr), hidden_size, runtime=runtime
        ),
    )


def read_attention_layer(*, cfg, hidden_size: int, linear_qkv_width: int,
                         scratch, layer_id: int, layer_type: str, token_id: int,
                         position: int, run_preceding_layers: bool,
                         target_src_ptr: int, target_dst_ptr: int,
                         runtime, readers: CaptureReaders):
    top_k = int(cfg.expert_used_count) if cfg.is_moe else 1
    down_ptr = int(
        scratch.moe_down_out.ptr if cfg.is_moe else scratch.ffn_down.ptr
    )
    down_elements = hidden_size * top_k if cfg.is_moe else hidden_size
    moe_shared_out = None
    moe_routing_weights = None
    moe_shared_gate = None
    moe_selected_experts = None
    moe_router_logits = None
    moe_selected_intermediate = None
    moe_shared_intermediate = None
    linear_qkv = None
    linear_z = None
    ssm_alpha = None
    ssm_beta = None
    conv_out = None
    recurrent_out = None
    recurrent_bf16 = None
    if layer_type == LINEAR_ATTENTION:
        ssm_inner_size = int(cfg.ssm_inner_size)
        ssm_time_step_rank = int(cfg.ssm_time_step_rank)
        linear_qkv = readers.bf16(
            int(scratch.linear_qkv.ptr),
            linear_qkv_width,
            runtime=runtime,
        )
        linear_z = readers.bf16(
            int(scratch.linear_z.ptr),
            ssm_inner_size,
            runtime=runtime,
        )
        ssm_alpha = readers.bf16(
            int(scratch.linear_alpha.ptr),
            ssm_time_step_rank,
            runtime=runtime,
        )
        ssm_beta = readers.bf16(
            int(scratch.linear_beta.ptr),
            ssm_time_step_rank,
            runtime=runtime,
        )
        conv_out = readers.f32(
            int(scratch.conv_out.ptr),
            linear_qkv_width,
            runtime=runtime,
        )
        recurrent_out = readers.f32(
            int(scratch.recurrent_out.ptr),
            ssm_inner_size,
            runtime=runtime,
        )
        recurrent_bf16 = readers.bf16(
            int(scratch.recurrent_bf16.ptr),
            ssm_inner_size,
            runtime=runtime,
        )
    if cfg.is_moe:
        moe_router_logits = readers.f32(
            int(scratch.moe_router_logits.ptr),
            int(cfg.expert_count),
            runtime=runtime,
        )
        moe_selected_intermediate = readers.bf16(
            int(scratch.ffn_intermediate.ptr),
            top_k * int(cfg.expert_feed_forward_length),
            runtime=runtime,
        )
        moe_shared_intermediate = readers.bf16(
            int(scratch.moe_shared_intermediate.ptr),
            int(cfg.expert_shared_feed_forward_length),
            runtime=runtime,
        )
        moe_shared_out = readers.bf16(
            int(scratch.moe_shared_out.ptr), hidden_size, runtime=runtime
        )
        moe_routing_weights = readers.f32(
            int(scratch.moe_routing_weights.ptr), top_k, runtime=runtime
        )
        shared_gate_ptr = (
            int(scratch.moe_router_logits.ptr)
            + int(cfg.expert_count) * DType.FP32.itemsize
        )
        moe_shared_gate = readers.f32(shared_gate_ptr, 1, runtime=runtime)
        moe_selected_experts = readers.i64(
            int(scratch.moe_selected_experts.ptr), top_k, runtime=runtime
        )

    return Qwen35GGUFLinearAttentionLayerCapture(
        layer_id=int(layer_id),
        layer_type=layer_type,
        token_id=int(token_id),
        position=int(position),
        hidden_size=hidden_size,
        is_moe=bool(cfg.is_moe),
        top_k=top_k,
        preceding_layer_count=int(layer_id) if run_preceding_layers else 0,
        hidden_in_f32=readers.bf16(
            target_src_ptr, hidden_size, runtime=runtime
        ),
        attn_norm_f32=readers.bf16(
            int(scratch.norm.ptr), hidden_size, runtime=runtime
        ),
        attn_out_f32=readers.bf16(
            int(scratch.attn_out.ptr), hidden_size, runtime=runtime
        ),
        post_norm_f32=readers.bf16(
            int(scratch.post_norm.ptr), hidden_size, runtime=runtime
        ),
        post_norm_source="bf16_scratch.post_norm",
        residual_f32=readers.bf16(
            int(scratch.residual.ptr), hidden_size, runtime=runtime
        ),
        ffn_or_moe_down_f32=readers.bf16(
            down_ptr, down_elements, runtime=runtime
        ),
        layer_out_f32=readers.bf16(
            target_dst_ptr, hidden_size, runtime=runtime
        ),
        moe_router_logits_f32=moe_router_logits,
        moe_selected_intermediate_f32=moe_selected_intermediate,
        moe_shared_intermediate_f32=moe_shared_intermediate,
        moe_shared_out_f32=moe_shared_out,
        moe_routing_weights_f32=moe_routing_weights,
        moe_shared_gate_f32=moe_shared_gate,
        moe_selected_experts_i64=moe_selected_experts,
        linear_qkv_f32=linear_qkv,
        linear_z_f32=linear_z,
        ssm_alpha_f32=ssm_alpha,
        ssm_beta_f32=ssm_beta,
        conv_out_f32=conv_out,
        recurrent_out_f32=recurrent_out,
        recurrent_bf16_f32=recurrent_bf16,
    )
