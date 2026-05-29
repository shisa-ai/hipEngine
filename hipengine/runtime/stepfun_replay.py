"""Deterministic StepFun CPU-reference layer replay harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from hipengine.kernels.cpu_reference.ops import (
    linear,
    step_apply_rope,
    step_dense_mlp,
    step_gqa_attention_decode,
    step_headwise_attention_gate,
    step_moe_mlp,
    step_rmsnorm,
)

ArrayLike = Any


@dataclass(frozen=True)
class StepFunAttentionReplaySpec:
    q_weight: ArrayLike
    k_weight: ArrayLike
    v_weight: ArrayLike
    o_weight: ArrayLike
    gate_weight: ArrayLike
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    rope_partial_factor: float = 1.0
    rope_theta: float = 10_000.0
    rope_llama3_scaling: bool = False
    sliding_window: int | None = None


@dataclass(frozen=True)
class StepFunDenseMLPReplaySpec:
    gate_weight: ArrayLike
    up_weight: ArrayLike
    down_weight: ArrayLike
    swiglu_limit: float | None = None


@dataclass(frozen=True)
class StepFunMoEMLPReplaySpec:
    router_weight: ArrayLike
    expert_gate_weight: ArrayLike
    expert_up_weight: ArrayLike
    expert_down_weight: ArrayLike
    router_bias: ArrayLike | None = None
    shared_gate_weight: ArrayLike | None = None
    shared_up_weight: ArrayLike | None = None
    shared_down_weight: ArrayLike | None = None
    top_k: int = 8
    routing_scale: float = 3.0
    swiglu_limit: float | None = None
    shared_swiglu_limit: float | None = None


@dataclass(frozen=True)
class StepFunLayerReplaySpec:
    name: str
    input_norm_weight: ArrayLike
    ffn_norm_weight: ArrayLike
    attention: StepFunAttentionReplaySpec
    mlp: StepFunDenseMLPReplaySpec | StepFunMoEMLPReplaySpec


@dataclass(frozen=True)
class StepFunReplayTolerance:
    atol: float = 1e-3
    rtol: float = 1e-3


@dataclass(frozen=True)
class StepFunReplayResult:
    name: str
    output: np.ndarray
    stages: Mapping[str, np.ndarray]

    def stage(self, name: str) -> np.ndarray:
        return self.stages[name]


class StepFunReplayMismatch(AssertionError):
    def __init__(self, stage: str, max_abs: float, max_rel: float, tolerance: StepFunReplayTolerance):
        self.stage = stage
        self.max_abs = max_abs
        self.max_rel = max_rel
        self.tolerance = tolerance
        super().__init__(
            f"StepFun replay mismatch at stage {stage!r}: "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
            f"tolerance(atol={tolerance.atol:.6g}, rtol={tolerance.rtol:.6g})"
        )


def replay_stepfun_layer(
    hidden_states: ArrayLike,
    spec: StepFunLayerReplaySpec,
    *,
    positions: ArrayLike,
    key_cache: ArrayLike | None = None,
    value_cache: ArrayLike | None = None,
    live_counts: ArrayLike | None = None,
) -> StepFunReplayResult:
    """Replay one StepFun text layer with CPU-reference primitives.

    The harness is intentionally small and deterministic. It records every major
    substage so dense, full-attention MoE, and sliding-attention MoE blocks can be
    compared against CPU-derived references before all 45 layers are integrated.
    """

    hidden = np.asarray(hidden_states, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("hidden_states must have shape [rows, hidden_size]")
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos.shape != (hidden.shape[0],):
        raise ValueError("positions must have one entry per hidden row")

    attention = spec.attention
    stages: dict[str, np.ndarray] = {"input": hidden.copy()}
    input_norm = step_rmsnorm(hidden, spec.input_norm_weight)
    stages["input_norm"] = input_norm

    q = _project_heads(input_norm, attention.q_weight, attention.num_query_heads, attention.head_dim, "q_weight")
    k = _project_heads(input_norm, attention.k_weight, attention.num_kv_heads, attention.head_dim, "k_weight")
    v = _project_heads(input_norm, attention.v_weight, attention.num_kv_heads, attention.head_dim, "v_weight")
    stages["q"] = q
    stages["k"] = k
    stages["v"] = v

    q_rope = step_apply_rope(
        q,
        pos,
        head_dim=attention.head_dim,
        partial_factor=attention.rope_partial_factor,
        theta=attention.rope_theta,
        llama3_scaling=attention.rope_llama3_scaling,
    )
    k_rope = step_apply_rope(
        k,
        pos,
        head_dim=attention.head_dim,
        partial_factor=attention.rope_partial_factor,
        theta=attention.rope_theta,
        llama3_scaling=attention.rope_llama3_scaling,
    )
    stages["q_rope"] = q_rope
    stages["k_rope"] = k_rope

    if key_cache is None:
        key_for_attn = k_rope[:, None, :, :]
        value_for_attn = v[:, None, :, :]
        counts = np.ones((hidden.shape[0],), dtype=np.int64) if live_counts is None else np.asarray(live_counts, dtype=np.int64)
    else:
        if value_cache is None:
            raise ValueError("value_cache is required when key_cache is provided")
        key_for_attn = np.asarray(key_cache, dtype=np.float32)
        value_for_attn = np.asarray(value_cache, dtype=np.float32)
        if live_counts is None:
            raise ValueError("live_counts is required when key_cache is provided")
        counts = np.asarray(live_counts, dtype=np.int64)
    stages["key_cache"] = key_for_attn.copy()
    stages["value_cache"] = value_for_attn.copy()
    stages["live_counts"] = counts.reshape(-1).astype(np.int64)

    attn = step_gqa_attention_decode(
        q_rope,
        key_for_attn,
        value_for_attn,
        counts,
        sliding_window=attention.sliding_window,
    )
    gate_logits = linear(input_norm, attention.gate_weight)
    gated_attn = step_headwise_attention_gate(attn, gate_logits)
    attn_out = linear(gated_attn.reshape(hidden.shape[0], -1), attention.o_weight)
    stages["attention"] = attn
    stages["attention_gate_logits"] = gate_logits
    stages["attention_gated"] = gated_attn
    stages["attention_output"] = attn_out

    attention_residual = hidden + attn_out
    ffn_norm = step_rmsnorm(attention_residual, spec.ffn_norm_weight)
    stages["attention_residual"] = attention_residual
    stages["ffn_norm"] = ffn_norm

    if isinstance(spec.mlp, StepFunDenseMLPReplaySpec):
        ffn = step_dense_mlp(
            ffn_norm,
            spec.mlp.gate_weight,
            spec.mlp.up_weight,
            spec.mlp.down_weight,
            swiglu_limit=spec.mlp.swiglu_limit,
        )
    else:
        ffn, router_weights, router_indices = step_moe_mlp(
            ffn_norm,
            spec.mlp.router_weight,
            spec.mlp.expert_gate_weight,
            spec.mlp.expert_up_weight,
            spec.mlp.expert_down_weight,
            router_bias=spec.mlp.router_bias,
            shared_gate_weight=spec.mlp.shared_gate_weight,
            shared_up_weight=spec.mlp.shared_up_weight,
            shared_down_weight=spec.mlp.shared_down_weight,
            top_k=spec.mlp.top_k,
            routing_scale=spec.mlp.routing_scale,
            swiglu_limit=spec.mlp.swiglu_limit,
            shared_swiglu_limit=spec.mlp.shared_swiglu_limit,
            return_router=True,
        )
        stages["router_weights"] = router_weights
        stages["router_indices"] = router_indices.astype(np.int64)
    stages["ffn"] = ffn

    output = attention_residual + ffn
    stages["output"] = output
    return StepFunReplayResult(name=spec.name, output=output, stages=stages)


def capture_stepfun_replay_reference(result: StepFunReplayResult) -> dict[str, np.ndarray]:
    """Return compact, in-memory stage references derived from CPU replay."""

    return {name: np.asarray(value).copy() for name, value in result.stages.items()}


def assert_stepfun_replay_close(
    actual: StepFunReplayResult,
    reference: StepFunReplayResult | Mapping[str, ArrayLike],
    *,
    tolerance: StepFunReplayTolerance = StepFunReplayTolerance(),
) -> None:
    """Compare replay stages and report the first mismatching substage."""

    reference_stages = reference.stages if isinstance(reference, StepFunReplayResult) else reference
    for stage, expected_value in reference_stages.items():
        if stage not in actual.stages:
            raise StepFunReplayMismatch(stage, float("inf"), float("inf"), tolerance)
        actual_value = np.asarray(actual.stages[stage])
        expected = np.asarray(expected_value)
        if actual_value.shape != expected.shape:
            raise StepFunReplayMismatch(stage, float("inf"), float("inf"), tolerance)
        if np.allclose(actual_value, expected, atol=tolerance.atol, rtol=tolerance.rtol):
            continue
        diff = np.abs(actual_value.astype(np.float32) - expected.astype(np.float32))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        denom = np.maximum(np.abs(expected.astype(np.float32)), np.float32(1e-12))
        max_rel = float(np.max(diff / denom)) if diff.size else 0.0
        raise StepFunReplayMismatch(stage, max_abs, max_rel, tolerance)


def _project_heads(
    hidden: np.ndarray,
    weight: ArrayLike,
    heads: int,
    head_dim: int,
    name: str,
) -> np.ndarray:
    projected = linear(hidden, weight)
    expected = int(heads) * int(head_dim)
    if projected.shape != (hidden.shape[0], expected):
        raise ValueError(f"{name} must project to rows x heads*head_dim = {hidden.shape[0]} x {expected}")
    return projected.reshape(hidden.shape[0], int(heads), int(head_dim)).astype(np.float32)


__all__ = [
    "StepFunAttentionReplaySpec",
    "StepFunDenseMLPReplaySpec",
    "StepFunLayerReplaySpec",
    "StepFunMoEMLPReplaySpec",
    "StepFunReplayMismatch",
    "StepFunReplayResult",
    "StepFunReplayTolerance",
    "assert_stepfun_replay_close",
    "capture_stepfun_replay_reference",
    "replay_stepfun_layer",
]
