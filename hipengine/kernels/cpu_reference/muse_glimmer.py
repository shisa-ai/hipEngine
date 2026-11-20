"""NumPy reference for the Muse Glimmer text architecture.

Contract source: the ``muse-glimmer`` GGUF metadata plus the llama.cpp
reference graph (``src/models/muse-glimmer.cpp``, build e0f142d5d). The
architecture is a dense decoder with:

* sandwich norms: pre-norm RMS (eps from metadata, ``weight+1`` folded at
  conversion) and post-norm RMS (eps 1e-8, plain weight) around both the
  attention and FFN sublayers;
* per-head QK RMS-norm whose Q weight absorbs ``qk_scale_factor``;
* interleaved (NORM-layout) RoPE on sliding-window layers only; full layers
  are NoPE;
* a sigmoid output gate computed from the pre-attention normed hidden state
  and applied to the attention output before ``o_proj``;
* a weightless embedding RMS-norm, then final norm, ``lm_head``, a logit
  scale, and a tanh softcap.

Weights are consumed as dequantized ``[out, in]`` float32 matrices, matching
``hipengine.kernels.cpu_reference.ops.linear``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from hipengine.loading.gguf import GGUFModelInfo


@dataclass(frozen=True)
class MuseGlimmerConfig:
    n_layers: int
    n_embd: int
    n_ff: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    rope_freq_base: float
    rms_eps: float
    post_norm_eps: float
    sliding_window: int
    swa_pattern: tuple[bool, ...]
    logit_scale: float
    final_logit_softcapping: float | None

    def __post_init__(self) -> None:
        if len(self.swa_pattern) != self.n_layers:
            raise ValueError("swa_pattern must have one entry per layer")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")

    @property
    def qkv_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @classmethod
    def from_gguf_info(cls, info: GGUFModelInfo, *, vocab_size: int) -> "MuseGlimmerConfig":
        md = info.metadata
        arch = md.get("general.architecture")
        if arch != "muse-glimmer":
            raise ValueError(f"expected muse-glimmer architecture, got {arch!r}")
        prefix = "muse-glimmer."
        pattern = md.get(prefix + "attention.sliding_window_pattern")
        n_layers = int(md[prefix + "block_count"])
        if pattern is None:
            swa = (True,) * n_layers
        else:
            swa = tuple(bool(v) for v in pattern)
        softcap = md.get(prefix + "final_logit_softcapping")
        return cls(
            n_layers=n_layers,
            n_embd=int(md[prefix + "embedding_length"]),
            n_ff=int(md[prefix + "feed_forward_length"]),
            n_heads=int(md[prefix + "attention.head_count"]),
            n_kv_heads=int(md[prefix + "attention.head_count_kv"]),
            head_dim=int(md[prefix + "attention.key_length"]),
            vocab_size=int(vocab_size),
            rope_freq_base=float(md[prefix + "rope.freq_base"]),
            rms_eps=float(md[prefix + "attention.layer_norm_rms_epsilon"]),
            post_norm_eps=1e-8,
            sliding_window=int(md[prefix + "attention.sliding_window"]),
            swa_pattern=swa,
            logit_scale=float(md.get(prefix + "logit_scale", 1.0)),
            final_logit_softcapping=None if softcap is None else float(softcap),
        )


def rmsnorm(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    out = x * np.reciprocal(np.sqrt(variance + eps))
    if weight is not None:
        out = out * np.asarray(weight, dtype=np.float32)
    return out.astype(np.float32)


def head_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Per-head RMS norm; ``x`` is [tokens, heads, head_dim]."""

    x = np.asarray(x, dtype=np.float32)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    out = x * np.reciprocal(np.sqrt(variance + eps))
    return (out * np.asarray(weight, dtype=np.float32)).astype(np.float32)


def rope_tables(
    positions: np.ndarray, head_dim: int, freq_base: float
) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved-pair rotary tables: cos/sin [n_positions, head_dim // 2]."""

    positions = np.asarray(positions, dtype=np.float32)
    half = head_dim // 2
    inv_freq = freq_base ** (-np.arange(half, dtype=np.float32) * (2.0 / head_dim))
    angles = positions[:, None] * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope_interleaved(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate adjacent pairs; ``x`` is [tokens, heads, head_dim]."""

    x = np.asarray(x, dtype=np.float32)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos_b - x1 * sin_b
    out[..., 1::2] = x0 * sin_b + x1 * cos_b
    return out


def causal_swa_mask(
    positions: np.ndarray, n_kv: int, kv_positions: np.ndarray, sliding_window: int | None
) -> np.ndarray:
    """[n_queries, n_kv] bool mask; True where attention is allowed."""

    q = np.asarray(positions)[:, None]
    k = np.asarray(kv_positions)[None, :]
    mask = k <= q
    if sliding_window is not None:
        mask &= k > q - int(sliding_window)
    return mask


def gqa_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    *,
    n_heads: int,
    n_kv_heads: int,
    scale: float,
) -> np.ndarray:
    """q [nq, n_heads, d]; k/v [nk, n_kv_heads, d]; returns [nq, n_heads, d]."""

    group = n_heads // n_kv_heads
    k = np.repeat(np.asarray(k, dtype=np.float32), group, axis=1)
    v = np.repeat(np.asarray(v, dtype=np.float32), group, axis=1)
    q = np.asarray(q, dtype=np.float32)
    logits = np.einsum("qhd,khd->hqk", q, k) * np.float32(scale)
    logits = np.where(mask[None, :, :], logits, np.float32(-np.inf))
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", weights, v).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


WeightSource = Callable[[str], np.ndarray]


def gguf_weight_source(reader) -> WeightSource:
    """Stream dequantized [out, in] float32 tensors from a GGUFReader."""

    def get(name: str) -> np.ndarray:
        return np.ascontiguousarray(reader.dequantize_tensor(name), dtype=np.float32)

    return get


def layer_forward(
    x: np.ndarray,
    positions: np.ndarray,
    kv_k: np.ndarray | None,
    kv_v: np.ndarray | None,
    get: WeightSource,
    config: MuseGlimmerConfig,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One decoder layer over new tokens.

    ``kv_k``/``kv_v`` hold the cached keys/values for prior positions
    ([n_past, n_kv_heads, head_dim]) or None. Returns (output, full_k, full_v)
    where full_* include the new tokens.
    """

    p = f"blk.{layer}."
    n_heads, n_kv_heads, d = config.n_heads, config.n_kv_heads, config.head_dim
    is_swa = config.swa_pattern[layer]

    normed = rmsnorm(x, get(p + "attn_norm.weight"), config.rms_eps)
    q = np.matmul(normed, get(p + "attn_q.weight").T).reshape(-1, n_heads, d)
    k = np.matmul(normed, get(p + "attn_k.weight").T).reshape(-1, n_kv_heads, d)
    v = np.matmul(normed, get(p + "attn_v.weight").T).reshape(-1, n_kv_heads, d)

    q = head_rmsnorm(q, get(p + "attn_q_norm.weight"), config.rms_eps)
    k = head_rmsnorm(k, get(p + "attn_k_norm.weight"), config.rms_eps)

    if is_swa:
        cos, sin = rope_tables(positions, d, config.rope_freq_base)
        q = apply_rope_interleaved(q, cos, sin)
        k = apply_rope_interleaved(k, cos, sin)

    if kv_k is None:
        full_k, full_v = k, v
        kv_positions = np.asarray(positions)
    else:
        full_k = np.concatenate([kv_k, k], axis=0)
        full_v = np.concatenate([kv_v, v], axis=0)
        kv_positions = np.arange(full_k.shape[0], dtype=np.int64)
        # Cached rows were written at positions 0..n_past-1 in order.
    mask = causal_swa_mask(
        positions, full_k.shape[0], kv_positions,
        config.sliding_window if is_swa else None,
    )
    attn = gqa_attention(
        q, full_k, full_v, mask,
        n_heads=n_heads, n_kv_heads=n_kv_heads, scale=d ** -0.5,
    )
    gate = sigmoid(np.matmul(normed, get(p + "attn_gate.weight").T))
    attn = attn.reshape(-1, n_heads * d) * gate
    attn = np.matmul(attn, get(p + "attn_output.weight").T)
    attn = rmsnorm(attn, None, config.post_norm_eps) * get(p + "post_attention_norm.weight")
    h = (x + attn).astype(np.float32)

    normed_ff = rmsnorm(h, get(p + "ffn_norm.weight"), config.rms_eps)
    gate_ff = np.matmul(normed_ff, get(p + "ffn_gate.weight").T)
    up = np.matmul(normed_ff, get(p + "ffn_up.weight").T)
    ff = np.matmul(silu(gate_ff) * up, get(p + "ffn_down.weight").T)
    ff = rmsnorm(ff, None, config.post_norm_eps) * get(p + "post_ffw_norm.weight")
    out = (h + ff).astype(np.float32)
    return out, full_k, full_v


def forward(
    token_ids: Sequence[int],
    get: WeightSource,
    config: MuseGlimmerConfig,
    *,
    capture_layers: Sequence[int] = (),
    past_kv: list[tuple[np.ndarray, np.ndarray]] | None = None,
    position_offset: int = 0,
) -> tuple[np.ndarray, dict[int, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Full forward over new tokens; returns (logits, captures, kv_cache).

    Logits are for every input token (prefill) with scale+softcap applied.
    ``capture_layers`` records each named layer's output (the DFlash residual
    tap semantics: the input to layer L is the output of layer L-1).
    """

    ids = np.asarray(token_ids, dtype=np.int64)
    embd = get("token_embd.weight")
    x = embd[ids]
    x = rmsnorm(x, None, config.rms_eps)
    positions = np.arange(position_offset, position_offset + ids.size, dtype=np.int64)

    if past_kv is None:
        past_kv = [(None, None)] * config.n_layers
    if len(past_kv) != config.n_layers:
        raise ValueError("past_kv must have one entry per layer")

    captures: dict[int, np.ndarray] = {}
    new_kv: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in range(config.n_layers):
        if layer in capture_layers:
            captures[layer] = x.copy()
        kv_k, kv_v = past_kv[layer]
        x, full_k, full_v = layer_forward(x, positions, kv_k, kv_v, get, config, layer)
        new_kv.append((full_k, full_v))
    if config.n_layers in capture_layers:
        captures[config.n_layers] = x.copy()

    x = rmsnorm(x, get("output_norm.weight"), config.rms_eps)
    logits = np.matmul(x, get("output.weight").T)
    logits = logits * np.float32(config.logit_scale)
    cap = config.final_logit_softcapping
    if cap:
        logits = np.float32(cap) * np.tanh(logits / np.float32(cap))
    return logits.astype(np.float32), captures, new_kv


__all__ = [
    "MuseGlimmerConfig",
    "apply_rope_interleaved",
    "causal_swa_mask",
    "forward",
    "gguf_weight_source",
    "gqa_attention",
    "head_rmsnorm",
    "layer_forward",
    "rmsnorm",
    "rope_tables",
    "sigmoid",
    "silu",
]
