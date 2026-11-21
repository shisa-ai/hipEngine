"""Torch-free reference operations for the Inkling 2 Mini decoder."""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -x))


def routed_feed_forward(weights, layer: int, hidden: np.ndarray, *, top_k: int,
                        route_scale: float, gate_cap: float, up_cap: float, project=None, project_dense=None,
                        project_group=None, feed_group=None) -> np.ndarray:
    """One token, with FP32 routing and accumulation."""
    if project is None:
        def project(layer, stem, expert, x):
            return weights.expert(layer, stem, expert) @ x
    prefix = f"layers.{layer}.mlp.gate."
    name = prefix + "weight"
    scores = _sigmoid(weights.dense(name) @ hidden if project_dense is None else project_dense(name, hidden))
    if not 1 <= top_k <= scores.size:
        raise ValueError(f"invalid Inkling 2 top-k: {top_k} for {scores.size} experts")
    ranking = scores + weights.dense(prefix + "bias")
    # Use expert index to break ties, and accumulate in expert order.
    ids = np.sort(np.argsort(-ranking, kind="stable")[:top_k])
    mix = scores[ids]
    mix *= route_scale / (mix.sum() + np.float32(1e-20))
    if feed_group is not None:
        return feed_group(layer, ids, mix, hidden, gate_cap, up_cap)

    def activate(paired):
        gate, up = paired[0::2], paired[1::2]
        return gate_cap * np.tanh(gate / gate_cap) * _sigmoid(gate) * up_cap * np.tanh(up / up_cap)

    result = np.zeros_like(hidden, dtype=np.float32)
    if project_group is not None:
        paired = project_group(layer, "w13", ids, hidden)
        inputs = np.stack([activate(row) for row in paired])
        down = project_group(layer, "w2", ids, inputs)
        for factor, row in zip(mix, down):
            result += factor * row
    else:
        for expert, factor in zip(ids.tolist(), mix):
            paired = project(layer, "w13", expert, hidden)
            result += factor * project(layer, "w2", expert, activate(paired))
    return result
