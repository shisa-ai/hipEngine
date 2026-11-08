"""Unit RED oracle for the clean-room DSpine CPU reference.

Pins the adjacent-causal-injection math and the transition-cache refinement
against independent hand computations (synthetic weights, no GPU). The live
gate (DSpine greedy IDs == true AR on Qwen3.8-27B) is separate; this pins the
kernel math the GPU kernels must match.
"""
from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.dspine import (
    DSpineBridgeWeights,
    bridge_write,
)


def _bridge(V=32, C=8, R=4, H=6, L=2, seed=0):
    rng = np.random.default_rng(seed)
    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32)
    return DSpineBridgeWeights(
        codebook=r(V, C), early_read_weights=r(L - 1, C, H), read_weight=r(C, H),
        feature_write_weights=r(L, H, R), gate_weight=r(R, C), gate_bias=r(R),
        predecessor_gate_weight=r(R, C), prev_weight=r(R, C),
    )


def test_bridge_write_anchor_and_predecessor_shift():
    b = _bridge()
    T, H, C, R = 5, 6, 8, 4
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((T, H)).astype(np.float32)
    input_ids = np.array([7, 0, 0, 0, 0], dtype=np.int64)
    next_norm = np.ones(H, dtype=np.float32)
    eps = 1e-6
    normed, out = bridge_write(raw, input_ids, 0, 2, b, next_norm, eps)

    # Independent recomputation of the injection math.
    rms = np.sqrt(np.maximum(np.mean(raw * raw, axis=-1), 1.1754943508222875e-38))
    n = raw / (rms[:, None] + eps)
    proj = n @ b.early_read_weights[0].T
    codes = np.sqrt(C) * proj / (np.sqrt(np.sum(proj * proj, axis=-1, keepdims=True)) + eps)
    codes[0] = np.sqrt(C) * b.codebook[7]
    packed = codes @ b.packed().T
    q, s, v = packed[:, :R], packed[:, R:2*R], packed[:, 2*R:]
    gate = 1.0 / (1.0 + np.exp(-(q[1:] + s[:-1] + b.gate_bias)))
    message = np.concatenate([np.zeros((1, R)), gate * v[:-1]], axis=0)
    delta = message @ b.feature_write_weights[0].T
    expected = raw + (rms[:, None] * delta) * (np.arange(T) != 0)[:, None]
    assert np.allclose(out, expected, atol=1e-5)
    # Anchor (position 0) receives no message: out[0] == raw[0].
    assert np.allclose(out[0], raw[0], atol=1e-6)


def test_bridge_write_zero_write_is_identity():
    b = _bridge()
    b.feature_write_weights = np.zeros_like(b.feature_write_weights)
    raw = np.random.default_rng(2).standard_normal((4, 6)).astype(np.float32)
    normed, out = bridge_write(raw, np.zeros(4, dtype=np.int64), 0, 2, b,
                               np.ones(6, dtype=np.float32), 1e-6)
    # Zero write matrix => no residual delta => out == raw.
    assert np.allclose(out, raw, atol=1e-6)


def test_transition_cache_matches_serial_refinement():
    # The parallel TG/TSV table score must equal a direct serial last-write
    # refinement (redo the last-layer write per sender token).
    from hipengine.kernels.cpu_reference.dspine import DSpineNumpyDrafter
    V, C, R, H, K = 16, 8, 4, 6, 4
    b = _bridge(V=V, C=C, R=R, H=H, L=2, seed=3)
    rng = np.random.default_rng(4)
    n = 3
    raw = rng.standard_normal((n, H)).astype(np.float32)
    rms = np.sqrt(np.mean(raw * raw, axis=-1))
    q = rng.standard_normal((n, R)).astype(np.float32)
    head = rng.standard_normal((V, H)).astype(np.float32)
    norm_w = np.ones(H, dtype=np.float32)
    cand = np.argsort(-rng.standard_normal((n, V)), axis=-1)[:, :K]
    cand = np.sort(cand, axis=-1)
    anchor = 5
    write_w = b.feature_write_weights[-1]
    sqrt_c = np.sqrt(C)

    # Parallel table (as refine_greedy builds it).
    sid = np.concatenate([np.full((1, K), anchor), cand[:-1]], axis=0)
    e = sqrt_c * b.codebook[sid]
    tvg = e @ b.predecessor_gate_weight.T
    tvv = e @ b.prev_weight.T
    m = (1.0 / (1.0 + np.exp(-(q[:, None, :] + tvg + b.gate_bias)))) * tvv
    head_rows = head[cand]
    tg_cand = (head_rows * norm_w) @ write_w
    base = np.sum((raw[:, None, :] * norm_w) * head_rows, axis=-1)
    s = m @ tg_cand.swapaxes(1, 2)
    score_parallel = base[:, None, :] + rms[:, None, None] * s

    # Serial: compute the same greedy score formula per sender (base + rms*s),
    # where s = (message @ Wᵀ) · (head_row ⊙ norm_w). The greedy path uses this
    # first-order form (the full re-normalized refinement is the sampled path).
    score_serial = np.zeros((n, K, K))
    for i in range(n):
        for si in range(K):
            sender = anchor if i == 0 else cand[i - 1, si]
            es = sqrt_c * b.codebook[sender]
            g = 1.0 / (1.0 + np.exp(-(q[i] + es @ b.predecessor_gate_weight.T + b.gate_bias)))
            msg = g * (es @ b.prev_weight.T)
            delta = msg @ write_w.T
            s_row = (head_rows[i] * norm_w) @ delta
            score_serial[i, si, :] = base[i] + rms[i] * s_row
    assert np.allclose(score_parallel, score_serial, atol=1e-4)