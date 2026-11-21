"""Torch-free reference operations for the Inkling 2 Mini decoder."""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -x))


def routed_feed_forward(weights, layer: int, hidden: np.ndarray, *, top_k: int,
                        route_scale: float, gate_cap: float, up_cap: float) -> np.ndarray:
    """One token, with decoded expert weights and FP32 accumulation."""
    prefix = f"layers.{layer}.mlp.gate."
    scores = _sigmoid(weights.dense(prefix + "weight") @ hidden)
    if not 1 <= top_k <= scores.size:
        raise ValueError(f"invalid Inkling 2 top-k: {top_k} for {scores.size} experts")
    ranking = scores + weights.dense(prefix + "bias")
    # Use expert index to break ties, and accumulate in expert order.
    ids = np.sort(np.argsort(-ranking, kind="stable")[:top_k])
    mix = scores[ids]
    mix *= route_scale / (mix.sum() + np.float32(1e-20))
    result = np.zeros_like(hidden, dtype=np.float32)
    for expert, factor in zip(ids.tolist(), mix):
        paired = weights.expert(layer, "w13", expert) @ hidden
        gate, up = paired[0::2], paired[1::2]
        activation = (gate_cap * np.tanh(gate / gate_cap) * _sigmoid(gate)
                      * up_cap * np.tanh(up / up_cap))
        result += factor * (weights.expert(layer, "w2", expert) @ activation)
    return result
