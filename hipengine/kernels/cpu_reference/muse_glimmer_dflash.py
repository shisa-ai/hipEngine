"""NumPy reference for the Muse Glimmer DFlash draft model.

Contract source: the ``dflash`` GGUF metadata, llama.cpp ``src/models/dflash.cpp``
(basic DFlash dual-mode graph), and ``common/speculative.cpp``'s draft-dflash
driver. The drafter is a 5-layer pre-norm decoder (no attention gate, no
post-norms) with:

* **injection mode**: concatenated target residual taps [T, 5*6656] -> fc ->
  enc RMSNorm -> per-layer K/V projections -> K head-norm -> NEOX RoPE -> KV
  cache at the target positions;
* **token mode**: a noise block [anchor, MASK x (block-1)] embedded through the
  *target's* token table (no embedding norm), 5 layers of QK-norm + NEOX RoPE +
  **non-causal** attention over [injected context + whole block], SwiGLU FFN,
  final RMSNorm, and the *target's* lm_head (argmax-equivalent without the
  target's monotone scale/softcap).

Norm weights are the raw HF tensors (the draft conversion applies no folds);
ggml applies them as-is, so this reference does too. RoPE is NEOX (rotate_half)
— the draft keeps HF layout, unlike the target's interleaved NORM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from hipengine.kernels.cpu_reference.muse_glimmer import (
    head_rmsnorm,
    rmsnorm,
    silu,
)
from hipengine.loading.gguf import GGUFModelInfo

WeightSource = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class MuseGlimmerDFlashConfig:
    n_layers: int
    n_embd: int
    n_ff: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    block_size: int
    sliding_window: int
    rope_freq_base: float
    rms_eps: float
    target_layers: tuple[int, ...]
    mask_token_id: int

    @classmethod
    def from_gguf_info(cls, info: GGUFModelInfo, *, mask_token_id: int) -> "MuseGlimmerDFlashConfig":
        md = info.metadata
        if md.get("general.architecture") != "dflash":
            raise ValueError(f"expected dflash architecture, got {md.get('general.architecture')!r}")
        p = "dflash."
        return cls(
            n_layers=int(md[p + "block_count"]),
            n_embd=int(md[p + "embedding_length"]),
            n_ff=int(md[p + "feed_forward_length"]),
            n_heads=int(md[p + "attention.head_count"]),
            n_kv_heads=int(md[p + "attention.head_count_kv"]),
            head_dim=int(md[p + "attention.key_length"]),
            block_size=int(md[p + "block_size"]),
            sliding_window=int(md[p + "attention.sliding_window"]),
            rope_freq_base=float(md[p + "rope.freq_base"]),
            rms_eps=float(md[p + "attention.layer_norm_rms_epsilon"]),
            target_layers=tuple(int(v) for v in md[p + "target_layers"]),
            mask_token_id=int(mask_token_id),
        )


def rope_tables_neox(
    positions: np.ndarray, head_dim: int, freq_base: float
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float32)
    half = head_dim // 2
    inv_freq = freq_base ** (-np.arange(half, dtype=np.float32) * (2.0 / head_dim))
    angles = positions[:, None] * inv_freq[None, :]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=1)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=1)
    return cos.astype(np.float32), sin.astype(np.float32)


def apply_rope_neox(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """rotate_half over the full head width; x [tokens, heads, head_dim]."""

    x = np.asarray(x, dtype=np.float32)
    half = x.shape[-1] // 2
    x0, x1 = x[..., :half], x[..., half:]
    cb, sb = cos[:, None, :half], sin[:, None, :half]
    return np.concatenate([x0 * cb - x1 * sb, x0 * sb + x1 * cb], axis=-1).astype(np.float32)


def inject_features(
    features: np.ndarray,
    positions: np.ndarray,
    get: WeightSource,
    config: MuseGlimmerDFlashConfig,
    kv_k: list[np.ndarray],
    kv_v: list[np.ndarray],
) -> None:
    """Injection mode: append target-feature-derived K/V at ``positions``.

    ``features`` is [T, n_taps*n_embd], taps concatenated in target_layers
    order. ``kv_k``/``kv_v`` are per-layer lists grown by concatenation (the
    caller keeps them position-ordered).
    """

    cfg = config
    h = np.matmul(features, get("fc.weight").T)
    h = rmsnorm(h, get("enc.output_norm.weight"), cfg.rms_eps)
    cos, sin = rope_tables_neox(positions, cfg.head_dim, cfg.rope_freq_base)
    for il in range(cfg.n_layers):
        p = f"blk.{il}."
        k = np.matmul(h, get(p + "attn_k.weight").T).reshape(-1, cfg.n_kv_heads, cfg.head_dim)
        v = np.matmul(h, get(p + "attn_v.weight").T).reshape(-1, cfg.n_kv_heads, cfg.head_dim)
        k = head_rmsnorm(k, get(p + "attn_k_norm.weight"), cfg.rms_eps)
        k = apply_rope_neox(k, cos, sin)
        kv_k[il] = k if kv_k[il] is None else np.concatenate([kv_k[il], k], axis=0)
        kv_v[il] = v if kv_v[il] is None else np.concatenate([kv_v[il], v], axis=0)


def draft_block(
    token_ids: Sequence[int],
    positions: np.ndarray,
    get: WeightSource,
    config: MuseGlimmerDFlashConfig,
    kv_k: list[np.ndarray | None],
    kv_v: list[np.ndarray | None],
    target_embd: np.ndarray,
    target_head: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Token mode: run the noise block; return (logits, full_k, full_v).

    Attention is non-causal: every block query sees the injected context and
    the whole block. The block's own K/V are appended to the returned caches
    (the caller discards them after drafting — only injected context persists).
    """

    cfg = config
    ids = np.asarray(token_ids, dtype=np.int64)
    tokens = ids.size
    x = target_embd[ids].astype(np.float32)
    cos, sin = rope_tables_neox(positions, cfg.head_dim, cfg.rope_freq_base)
    n_past = 0 if kv_k[0] is None else kv_k[0].shape[0]
    n_total = n_past + tokens
    # Non-causal over [context + block]; context positions precede the block.
    mask = np.ones((tokens, n_total), dtype=bool)

    new_k: list[np.ndarray] = []
    new_v: list[np.ndarray] = []
    for il in range(cfg.n_layers):
        p = f"blk.{il}."
        normed = rmsnorm(x, get(p + "attn_norm.weight"), cfg.rms_eps)
        q = np.matmul(normed, get(p + "attn_q.weight").T).reshape(tokens, cfg.n_heads, cfg.head_dim)
        k = np.matmul(normed, get(p + "attn_k.weight").T).reshape(tokens, cfg.n_kv_heads, cfg.head_dim)
        v = np.matmul(normed, get(p + "attn_v.weight").T).reshape(tokens, cfg.n_kv_heads, cfg.head_dim)
        q = head_rmsnorm(q, get(p + "attn_q_norm.weight"), cfg.rms_eps)
        k = head_rmsnorm(k, get(p + "attn_k_norm.weight"), cfg.rms_eps)
        q = apply_rope_neox(q, cos, sin)
        k = apply_rope_neox(k, cos, sin)
        full_k = k if kv_k[il] is None else np.concatenate([kv_k[il], k], axis=0)
        full_v = v if kv_v[il] is None else np.concatenate([kv_v[il], v], axis=0)
        new_k.append(full_k)
        new_v.append(full_v)

        group = cfg.n_heads // cfg.n_kv_heads
        kr = np.repeat(full_k, group, axis=1)
        vr = np.repeat(full_v, group, axis=1)
        logits = np.einsum("qhd,khd->hqk", q, kr) * np.float32(cfg.head_dim ** -0.5)
        logits = np.where(mask[None, :, :], logits, np.float32(-np.inf))
        logits = logits - logits.max(axis=-1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(axis=-1, keepdims=True)
        attn = np.einsum("hqk,khd->qhd", w, vr).reshape(tokens, cfg.n_heads * cfg.head_dim)
        x = (x + np.matmul(attn, get(p + "attn_output.weight").T)).astype(np.float32)
        normed_ff = rmsnorm(x, get(p + "ffn_norm.weight"), cfg.rms_eps)
        gate = np.matmul(normed_ff, get(p + "ffn_gate.weight").T)
        up = np.matmul(normed_ff, get(p + "ffn_up.weight").T)
        ff = np.matmul(silu(gate) * up, get(p + "ffn_down.weight").T)
        x = (x + ff).astype(np.float32)
    x = rmsnorm(x, get("output_norm.weight"), cfg.rms_eps)
    logits = np.matmul(x, target_head.T)
    return logits.astype(np.float32), new_k, new_v


def gguf_weight_source(reader) -> WeightSource:
    def get(name: str) -> np.ndarray:
        return np.ascontiguousarray(reader.dequantize_tensor(name), dtype=np.float32)
    return get


__all__ = [
    "MuseGlimmerDFlashConfig",
    "apply_rope_neox",
    "draft_block",
    "gguf_weight_source",
    "inject_features",
    "rope_tables_neox",
]