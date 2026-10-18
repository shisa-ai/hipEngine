"""CPU FP32 reference for the VibeVoice-ASR Qwen2 text backbone.

Mirrors ``transformers`` Qwen2 exactly for the torch-free hipEngine port:
plain-weight RMSNorm, GQA with QKV biases, scalar RoPE (theta 1e6,
rotate-half, head_dim 128), SiLU-gated MLP, untied lm_head. Attention is
causal over the KV cache; the caller stages cache slices. Accumulation FP32.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

ArrayLike = Any


def qwen2_rmsnorm(x: ArrayLike, weight: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    xb = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    variance = np.mean(xb * xb, axis=-1, keepdims=True)
    normed = xb / np.sqrt(variance + eps)
    return (normed * w.reshape(*([1] * (xb.ndim - 1)), -1)).astype(np.float32)


def qwen2_silu(x: ArrayLike) -> np.ndarray:
    xb = np.asarray(x, dtype=np.float32)
    return (xb / (1.0 + np.exp(-xb))).astype(np.float32)


def qwen2_rope(
    q: ArrayLike,
    k: ArrayLike,
    positions: ArrayLike,
    theta: float = 1_000_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar RoPE in-place on (tokens, heads, head_dim) arrays, rotate-half."""
    q_tokens = np.asarray(q, dtype=np.float32)
    k_tokens = np.asarray(k, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    head_dim = q_tokens.shape[-1]
    inv_freq = (1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))).astype(np.float32)
    angles = pos[:, None].astype(np.float32) * inv_freq[None, :]  # (tokens, head_dim/2)
    cos = np.cos(angles)
    sin = np.sin(angles)

    def rotate(x: np.ndarray) -> np.ndarray:
        half = head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = np.empty_like(x)
        rotated[..., :half] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        rotated[..., half:] = x1 * sin[:, None, :] + x2 * cos[:, None, :]
        return rotated

    return rotate(q_tokens), rotate(k_tokens)


def qwen2_layer_forward(
    layer: Any,
    hidden: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    positions: Sequence[int],
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> np.ndarray:
    """One Qwen2 layer over ``tokens`` new rows with caches pre-staged.

    ``k_cache``/``v_cache`` cover absolute positions ``0..max(positions)``;
    this function writes the new rows at their positions, then attends
    causally: row ``t`` sees cache positions ``0..positions[t]``.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    tokens = hidden.shape[0]
    normed = qwen2_rmsnorm(hidden, layer.input_layernorm, rms_norm_eps)
    q = normed @ layer.q_weight.T + layer.q_bias
    k = normed @ layer.k_weight.T + layer.k_bias
    v = normed @ layer.v_weight.T + layer.v_bias

    q = q.reshape(tokens, num_attention_heads, head_dim)
    k = k.reshape(tokens, num_key_value_heads, head_dim)
    v = v.reshape(tokens, num_key_value_heads, head_dim)

    pos = np.asarray(positions, dtype=np.int64)
    q, k = qwen2_rope(q, k, pos, theta=rope_theta)

    for t in range(tokens):
        k_cache[pos[t]] = k[t]
        v_cache[pos[t]] = v[t]

    attn = _gqa_attention_causal(q, k_cache, v_cache, pos, num_key_value_heads, head_dim)
    attn_flat = attn.reshape(tokens, num_attention_heads * head_dim)
    residual = hidden + attn_flat @ layer.o_weight.T

    normed2 = qwen2_rmsnorm(residual, layer.post_attention_layernorm, rms_norm_eps)
    gate = normed2 @ layer.gate_proj.T
    up = normed2 @ layer.up_proj.T
    mlp = (qwen2_silu(gate) * up) @ layer.down_proj.T
    return residual + mlp


def _gqa_attention_causal(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    positions: np.ndarray,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    tokens, nq, _ = q.shape
    group = nq // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    out = np.empty((tokens, nq, head_dim), dtype=np.float32)
    for t in range(tokens):
        ctx = positions[t] + 1
        for h in range(num_kv_heads):
            qh = q[t, h * group : (h + 1) * group]
            scores = np.einsum("gh,lh->gl", qh, k_cache[:ctx, h, :]) * scale
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            out[t, h * group : (h + 1) * group] = np.einsum("gl,lh->gh", probs, v_cache[:ctx, h, :])
    return out


def qwen2_model_forward(
    weights: Any,
    input_ids: Sequence[int] | np.ndarray,
    k_caches: Sequence[np.ndarray],
    v_caches: Sequence[np.ndarray],
    positions: Sequence[int],
) -> np.ndarray:
    """Full prefill/decode pass; returns final hidden states (tokens, hidden)."""
    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    hidden = weights.embed_tokens[ids[0]]
    spec = weights.spec
    for layer, kc, vc in zip(weights.layers, k_caches, v_caches):
        hidden = qwen2_layer_forward(
            layer,
            hidden,
            kc,
            vc,
            positions,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            rope_theta=spec.rope_theta,
            rms_norm_eps=spec.rms_norm_eps,
        )
    return qwen2_rmsnorm(hidden, weights.final_norm, spec.rms_norm_eps)


def qwen2_logits(weights: Any, hidden: np.ndarray) -> np.ndarray:
    """Untied lm_head logits: (tokens, vocab)."""
    return np.asarray(hidden, dtype=np.float32) @ weights.lm_head.T
