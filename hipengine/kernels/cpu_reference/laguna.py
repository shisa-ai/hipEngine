"""Torch-free NumPy reference primitives for Laguna."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.kernels.registry import KernelKey, register

ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class LagunaRoutingResult:
    """Intermediate and selected values from Laguna sigmoid MoE routing."""

    router_logits: np.ndarray
    routing_scores: np.ndarray
    selection_scores: np.ndarray
    selected_experts: np.ndarray
    routing_weights: np.ndarray
    scaled_routing_weights: np.ndarray


def laguna_softplus_head_gate(
    attention: ArrayLike,
    gate_logits: ArrayLike,
) -> np.ndarray:
    """Apply Laguna's FP32 softplus per-head attention gate.

    ``attention`` is shaped ``[..., heads, head_dim]`` and ``gate_logits`` is
    shaped ``[..., heads]``. One scalar is broadcast across each head's final
    dimension before the output projection.
    """

    attn = np.asarray(attention, dtype=np.float32)
    gate = np.asarray(gate_logits, dtype=np.float32)
    if attn.ndim < 2:
        raise ValueError("attention must have shape [..., heads, head_dim]")
    if gate.shape != attn.shape[:-1]:
        raise ValueError(
            "gate_logits must match attention leading dimensions [..., heads]"
        )
    if not np.isfinite(attn).all() or not np.isfinite(gate).all():
        raise ValueError("attention and gate_logits must be finite")
    scale = np.logaddexp(np.float32(0.0), gate).astype(np.float32)
    return (attn * scale[..., None]).astype(np.float32)


def laguna_sigmoid_correction_topk(
    hidden: ArrayLike,
    router_weight: ArrayLike,
    correction_bias: ArrayLike,
    *,
    experts_used: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
    router_logit_softcapping: float = 0.0,
) -> LagunaRoutingResult:
    """Reference Laguna sigmoid routing with selection-only correction bias.

    Selection uses ``sigmoid(logits) + correction_bias``. Returned routing
    weights are gathered from the uncorrected sigmoid scores, optionally
    sum-normalized, then exposed both before and after routed-output scaling.
    Stable descending selection keeps the lower expert ID on exact ties.
    """

    x = np.asarray(hidden, dtype=np.float32)
    router = np.asarray(router_weight, dtype=np.float32)
    bias = np.asarray(correction_bias, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    if router.ndim != 2 or router.shape[1] != x.shape[1]:
        raise ValueError("router_weight must have shape [experts, hidden]")
    if bias.shape != (router.shape[0],):
        raise ValueError("correction_bias must have shape [experts]")
    if not np.isfinite(x).all() or not np.isfinite(router).all() or not np.isfinite(bias).all():
        raise ValueError("hidden, router_weight, and correction_bias must be finite")

    top_k = int(experts_used)
    if top_k <= 0 or top_k > router.shape[0]:
        raise ValueError("experts_used must be within [1, number of experts]")
    softcap = float(router_logit_softcapping)
    if softcap < 0.0:
        raise ValueError("router_logit_softcapping must be non-negative")
    scale = float(routed_scaling_factor)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("routed_scaling_factor must be finite and positive")

    logits = np.matmul(x, router.T).astype(np.float32)
    if softcap > 0.0:
        logits = (np.tanh(logits / np.float32(softcap)) * np.float32(softcap)).astype(
            np.float32
        )
    routing_scores = _stable_sigmoid(logits)
    selection_scores = (routing_scores + bias[None, :]).astype(np.float32)
    selected = np.argsort(-selection_scores, axis=-1, kind="stable")[:, :top_k].astype(
        np.int64
    )
    weights = np.take_along_axis(routing_scores, selected, axis=-1).astype(np.float32)
    if norm_topk_prob:
        denominator = np.maximum(
            weights.sum(axis=-1, keepdims=True, dtype=np.float32),
            np.float32(np.finfo(np.float32).tiny),
        )
        weights = (weights / denominator).astype(np.float32)
    scaled = (weights * np.float32(scale)).astype(np.float32)
    return LagunaRoutingResult(
        router_logits=logits,
        routing_scores=routing_scores,
        selection_scores=selection_scores,
        selected_experts=selected,
        routing_weights=weights,
        scaled_routing_weights=scaled,
    )


def register_laguna_cpu_reference_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "cpu_reference",
            "softplus_head_gate",
            "fp32",
            "laguna_per_head",
        ),
        laguna_softplus_head_gate,
        replace=replace,
    )
    register(
        KernelKey(
            "cpu_reference",
            "laguna_sigmoid_router_topk",
            "gguf_f32",
            "correction_bias",
        ),
        laguna_sigmoid_correction_topk,
        replace=replace,
    )


def _stable_sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-x[positive]).astype(np.float32)
    )
    negative_exp = np.exp(x[~positive]).astype(np.float32)
    out[~positive] = negative_exp / (np.float32(1.0) + negative_exp)
    return out


__all__ = [
    "LagunaRoutingResult",
    "laguna_sigmoid_correction_topk",
    "laguna_softplus_head_gate",
    "register_laguna_cpu_reference_kernels",
]
