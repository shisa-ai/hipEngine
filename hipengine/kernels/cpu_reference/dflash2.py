"""Torch-free NumPy reference primitives for DFlash2.

Implements the two DFlash2 drafter mechanisms exactly as described in the
z-lab/dflash reference (``dflash/model.py`` @ 07ebd93):

* ``grouped_dynamic_convolve`` / ``grouped_dynamic_conv_prepare`` /
  ``grouped_dynamic_conv_finish`` — the two-tap grouped dynamic causal conv
  that runs before/after each attention and MLP sublayer.
* ``candidate_selector_select`` / ``dflash2_topk`` — the low-rank bilinear
  top-16 path selector with a greedy (T=0) walk.

These are the strict RED oracles for the native DFlash2 kernels; the golden
fixtures are generated from the torch reference implementation (test-time
only, never on the hot path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.kernels.cpu_reference.ops import linear, rmsnorm
from hipengine.kernels.registry import KernelKey, register

ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def grouped_dynamic_convolve(
    hidden: ArrayLike,
    dynamic: ArrayLike,
    base: ArrayLike,
    group_size: int,
) -> np.ndarray:
    """Causal grouped dynamic conv (DFlash2 ``_grouped_dynamic_convolve``).

    ``hidden`` is ``(..., length, hidden_size)``; ``dynamic`` is
    ``(..., length, kernel_size, groups)``; ``base`` is
    ``(kernel_size, hidden_size)``.

    For each tap offset ``o`` the contribution is
    ``(base[o] + dynamic[..., o, g]) * hidden[..., t - o, :]`` broadcast per
    channel group ``g`` (zero-padded at the sequence start), matching the
    torch reference exactly.
    """

    hidden_arr = np.asarray(hidden, dtype=np.float32)
    dynamic_arr = np.asarray(dynamic, dtype=np.float32)
    base_arr = np.asarray(base, dtype=np.float32)
    if base_arr.ndim != 2:
        raise ValueError(f"base must be (kernel_size, hidden_size), got {base_arr.shape}")
    *_, length, hidden_size = hidden_arr.shape
    kernel_size, base_hidden = base_arr.shape
    if base_hidden != hidden_size:
        raise ValueError(f"base hidden_size {base_hidden} != hidden hidden_size {hidden_size}")
    if group_size <= 0 or hidden_size % group_size != 0:
        raise ValueError(f"group_size {group_size} must divide hidden_size {hidden_size}")
    groups = hidden_size // group_size
    expected_dynamic = hidden_arr.shape[:-1] + (kernel_size, groups)
    if dynamic_arr.shape != expected_dynamic:
        raise ValueError(f"dynamic {dynamic_arr.shape} != expected {expected_dynamic}")

    base_g = base_arr.reshape(kernel_size, groups, group_size)
    output_g = np.zeros(hidden_arr.shape[:-1] + (groups, group_size), dtype=np.float32)
    for offset in range(kernel_size):
        if offset == 0:
            values = hidden_arr
        else:
            values = np.pad(
                hidden_arr[..., :-offset, :],
                [(0, 0)] * (hidden_arr.ndim - 2) + [(offset, 0), (0, 0)],
                mode="constant",
            )
        values_g = values.reshape(*values.shape[:-1], groups, group_size)
        dyn = dynamic_arr[..., offset, :]  # (..., length, groups)
        kernel = base_g[offset][np.newaxis, np.newaxis, :, :] + dyn[..., :, :, np.newaxis]
        output_g += kernel * values_g
    return output_g.reshape(hidden_arr.shape)


def grouped_dynamic_conv_prepare(
    hidden: ArrayLike,
    kernel_projection: ArrayLike,
    base_kernel: ArrayLike,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """DFlash2 ``GroupedDynamicCausalConv.prepare``.

    Returns the input-side convolved hidden and the output-side dynamic
    coefficients ``(..., length, kernel_size, groups)`` for ``finish``.
    """

    base_arr = np.asarray(base_kernel, dtype=np.float32)
    if base_arr.ndim != 3 or base_arr.shape[0] != 2:
        raise ValueError(f"base_kernel must be (2, kernel_size, hidden), got {base_arr.shape}")
    kernel_size, hidden_size = base_arr.shape[1], base_arr.shape[2]
    proj = linear(hidden, kernel_projection)
    groups = hidden_size // group_size
    expected_proj = proj.shape[:-1] + (2 * kernel_size * groups,)
    if proj.shape != expected_proj:
        raise ValueError(
            f"kernel_projection output {proj.shape} != expected {expected_proj}"
        )
    dynamic = proj.reshape(*proj.shape[:-1], 2, kernel_size, groups)
    convolved = grouped_dynamic_convolve(
        hidden, dynamic[..., 0, :, :], base_arr[0], group_size
    )
    return convolved, dynamic[..., 1, :, :]


def grouped_dynamic_conv_finish(
    hidden: ArrayLike,
    dynamic: ArrayLike,
    base_kernel: ArrayLike,
    group_size: int,
) -> np.ndarray:
    """DFlash2 ``GroupedDynamicCausalConv.finish`` (output-side conv)."""

    base_arr = np.asarray(base_kernel, dtype=np.float32)
    return grouped_dynamic_convolve(hidden, dynamic, base_arr[1], group_size)


def dflash2_topk(logits: ArrayLike, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``top_k`` values and indices over the last dim, sorted descending.

    The reference returns ``torch.topk(..., sorted=False)`` whose candidate
    *order* is unspecified; the greedy walk's argmax is order-independent, so
    the deterministic descending order is a strict superset of the reference's
    path selection contract.
    """

    logits_arr = np.asarray(logits, dtype=np.float32)
    top_k = int(top_k)
    if logits_arr.shape[-1] < top_k:
        raise ValueError(f"top_k {top_k} exceeds vocab {logits_arr.shape[-1]}")
    full_indices = np.argsort(logits_arr, axis=-1)[..., ::-1]
    indices = full_indices[..., :top_k]
    values = np.take_along_axis(logits_arr, indices, axis=-1)
    return values, indices


@dataclass(frozen=True)
class DFlash2SelectorResult:
    """Selected greedy path, candidate table, and per-position scores."""

    path: np.ndarray
    candidates: np.ndarray
    unary: np.ndarray
    scores: np.ndarray


def candidate_selector_select(
    hidden: ArrayLike,
    logits: ArrayLike,
    anchor_ids: ArrayLike,
    predecessor_codebook: ArrayLike,
    successor_codebook: ArrayLike,
    hidden_projection: ArrayLike,
    *,
    top_k: int,
) -> DFlash2SelectorResult:
    """DFlash2 ``CandidateSelector.select`` with a greedy (T=0) walk.

    ``hidden`` is ``(batch, length, hidden_size)``, ``logits`` is
    ``(batch, length, vocab_size)``, ``anchor_ids`` is ``(batch,)`` (the last
    verified target token). Codebooks are ``(vocab_size, rank)``.
    """

    hidden_arr = np.asarray(hidden, dtype=np.float32)
    logits_arr = np.asarray(logits, dtype=np.float32)
    anchor_arr = np.asarray(anchor_ids, dtype=np.int64)
    codebook_a = np.asarray(predecessor_codebook, dtype=np.float32)
    codebook_b = np.asarray(successor_codebook, dtype=np.float32)
    if hidden_arr.ndim != 3:
        raise ValueError(f"hidden must be (batch, length, hidden), got {hidden_arr.shape}")
    batch, length, hidden_size = hidden_arr.shape
    if logits_arr.shape[:2] != (batch, length):
        raise ValueError(f"logits {logits_arr.shape} must match hidden {hidden_arr.shape[:2]}")
    if codebook_a.shape != codebook_b.shape or codebook_a.ndim != 2:
        raise ValueError(
            f"codebooks must be equal 2-D (vocab, rank), got {codebook_a.shape} vs {codebook_b.shape}"
        )
    vocab, rank = codebook_a.shape
    # Logits may cover a subset of the codebook vocab (the selector only gathers
    # codebook rows at candidate ids, which are < logits vocab).
    if logits_arr.shape[-1] > vocab:
        raise ValueError(f"logits vocab {logits_arr.shape[-1]} exceeds codebook vocab {vocab}")
    if anchor_arr.shape != (batch,):
        raise ValueError(f"anchor_ids must be (batch,), got {anchor_arr.shape}")
    if not (0 < top_k <= vocab):
        raise ValueError(f"top_k {top_k} must be in (0, vocab {vocab}]")

    unary, candidates = dflash2_topk(logits_arr, top_k)
    h = linear(hidden_arr, hidden_projection)  # (batch, length, rank)
    predecessor = anchor_arr.copy()
    path = np.zeros((batch, length), dtype=np.int64)
    scores = np.zeros((batch, length, top_k), dtype=np.float32)
    rows = np.arange(batch)
    for position in range(length):
        ah = codebook_a[predecessor] * h[:, position, :]  # (batch, rank)
        bc = codebook_b[candidates[:, position, :]]  # (batch, top_k, rank)
        sc = unary[:, position, :] + np.sum(ah[:, None, :] * bc, axis=-1)
        scores[:, position, :] = sc
        index = np.argmax(sc, axis=-1)  # (batch,), first-max tie-break matches torch
        predecessor = candidates[rows, position, index]
        path[:, position] = predecessor
    return DFlash2SelectorResult(
        path=path,
        candidates=candidates,
        unary=unary,
        scores=scores,
    )


def candidate_selector_greedy_path(
    hidden: ArrayLike,
    logits: ArrayLike,
    anchor_ids: ArrayLike,
    predecessor_codebook: ArrayLike,
    successor_codebook: ArrayLike,
    hidden_projection: ArrayLike,
    *,
    top_k: int,
) -> np.ndarray:
    """Return only the greedy selector path (int token ids).

    Thin wrapper for the single-array LayerFixture flow; the full result is
    available via :func:`candidate_selector_select`.
    """

    return candidate_selector_select(
        hidden,
        logits,
        anchor_ids,
        predecessor_codebook,
        successor_codebook,
        hidden_projection,
        top_k=top_k,
    ).path


def dflash2_rope_tables(
    positions: ArrayLike,
    *,
    rope_theta: float,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Qwen3-style rotary tables: cos/sin of shape (length, head_dim).

    ``inv_freq = 1 / theta**(arange(0, head_dim, 2) / head_dim)``, then each
    frequency is repeated twice to fill ``head_dim``, matching the reference
    ``Qwen3RotaryEmbedding`` (``emb = cat([freqs, freqs], -1)``).
    """

    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 1:
        raise ValueError(f"positions must be rank-1, got {pos.shape}")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")
    dims = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = np.float32(1.0) / (np.float32(rope_theta) ** (dims / np.float32(head_dim)))
    freqs = pos[:, None] * inv_freq[None, :]  # (length, head_dim/2)
    # Block-repeat: emb = cat([freqs, freqs], -1); the rope pairs channel c with
    # channel c + head_dim/2 (rotate_half splits at head_dim/2).
    emb = np.concatenate((freqs, freqs), axis=-1)  # (length, head_dim)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """Reference ``transformers.rotate_half``: cat([-x2, x1]) over the last dim."""

    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def dflash2_attention_forward(
    hidden: ArrayLike,
    target_hidden: ArrayLike,
    positions: ArrayLike,
    q_proj: ArrayLike,
    k_proj: ArrayLike,
    v_proj: ArrayLike,
    o_proj: ArrayLike,
    q_norm: ArrayLike,
    k_norm: ArrayLike,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
    sliding_window: int,
    is_causal: bool,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """DFlash2 attention: projected-context K/V plus draft hidden queries.

    Context rows are the projected target hidden (always visible predecessors);
    draft rows follow. q/k are per-head RMSNormed and rotary-embedded; the mask
    is a sliding window (bidirectional when ``is_causal`` is false), exactly the
    reference ``Qwen3DFlashAttention``.

    Returns the attention output (draft rows only) of shape
    ``(batch, draft_length, hidden_size)``.
    """

    hidden_arr = np.asarray(hidden, dtype=np.float32)
    target_arr = np.asarray(target_hidden, dtype=np.float32)
    pos_arr = np.asarray(positions, dtype=np.int64)
    if hidden_arr.ndim != 3:
        raise ValueError(f"hidden must be (batch, length, hidden), got {hidden_arr.shape}")
    batch, draft_len, hidden_size = hidden_arr.shape
    ctx_len = target_arr.shape[1]
    if target_arr.shape != (batch, ctx_len, hidden_size):
        raise ValueError(f"target_hidden {target_arr.shape} must be (batch, ctx_len, hidden)")
    if pos_arr.shape != (batch, ctx_len + draft_len):
        raise ValueError(
            f"positions {pos_arr.shape} must be (batch, ctx_len+draft_len) "
            f"= ({batch}, {ctx_len + draft_len})"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads {num_heads} must divide by num_kv_heads {num_kv_heads}")

    q = linear(hidden_arr, q_proj)  # (B, L_d, nH*Hd)
    q = q.reshape(batch, draft_len, num_heads, head_dim)
    q = _head_rmsnorm(q, q_norm, eps)
    k_ctx = linear(target_arr, k_proj)
    k_noise = linear(hidden_arr, k_proj)
    k_cat = np.concatenate((k_ctx, k_noise), axis=1)  # (B, L_ctx+L_d, nKV*Hd)
    k_len = k_cat.shape[1]
    k_cat = k_cat.reshape(batch, k_len, num_kv_heads, head_dim)
    k_cat = _head_rmsnorm(k_cat, k_norm, eps)
    v_cat = np.concatenate(
        (linear(target_arr, v_proj), linear(hidden_arr, v_proj)), axis=1
    ).reshape(batch, k_len, num_kv_heads, head_dim)

    cos, sin = dflash2_rope_tables(
        pos_arr.reshape(-1), rope_theta=rope_theta, head_dim=head_dim
    )
    cos_k = cos[:k_len][None, None, :, :]
    sin_k = sin[:k_len][None, None, :, :]
    cos_q = cos_k[..., -draft_len:, :]
    sin_q = sin_k[..., -draft_len:, :]
    q = np.swapaxes(q, 1, 2)  # (B, nH, L_d, Hd)
    q = q * cos_q + _rotate_half(q) * sin_q
    k_cat = np.swapaxes(k_cat, 1, 2)  # (B, nKV, k_len, Hd)
    k_cat = k_cat * cos_k + _rotate_half(k_cat) * sin_k
    v_cat = np.swapaxes(v_cat, 1, 2)

    # Bidirectional (or causal) sliding-window visibility mask, matching the
    # reference _attention_mask over the concatenated key sequence.
    query_position = k_len - draft_len + np.arange(draft_len)[:, None]  # (L_d, 1)
    key_position = np.arange(k_len)[None, :]  # (1, k_len)
    visible = np.ones((draft_len, k_len), dtype=bool)
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window and sliding_window > 0:
        visible &= query_position - key_position < sliding_window
        if not is_causal:
            visible &= key_position - query_position < sliding_window
    mask = np.where(visible, np.float32(0.0), np.float32(-np.inf))

    scale = np.float32(head_dim) ** np.float32(-0.5)
    groups = num_heads // num_kv_heads
    out = np.zeros((batch, num_heads, draft_len, head_dim), dtype=np.float32)
    for b in range(batch):
        for h in range(num_heads):
            kv = h // groups
            scores = (q[b, h] @ k_cat[b, kv].T) * scale + mask
            scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            scores = scores / scores.sum(axis=-1, keepdims=True)
            out[b, h] = scores @ v_cat[b, kv]
    out = out.transpose(0, 2, 1, 3).reshape(batch, draft_len, num_heads * head_dim)
    return linear(out, o_proj)


def _head_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Per-head RMSNorm over the last axis (head_dim), matching q/k norms."""

    w = np.asarray(weight, dtype=np.float32)
    if w.shape[-1] != x.shape[-1]:
        raise ValueError(f"head norm weight {w.shape} does not match head dim {x.shape[-1]}")
    return rmsnorm(x, w, eps=eps).astype(np.float32)


def register_dflash2_cpu_reference_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("cpu_reference", "dflash2_grouped_conv", "fp32"),
        grouped_dynamic_convolve,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "dflash2_selector", "fp32"),
        candidate_selector_select,
        replace=replace,
    )
    register(
        KernelKey("cpu_reference", "dflash2_selector_path", "fp32"),
        candidate_selector_greedy_path,
        replace=replace,
    )


register_dflash2_cpu_reference_kernels()

__all__ = [
    "DFlash2SelectorResult",
    "candidate_selector_greedy_path",
    "candidate_selector_select",
    "dflash2_topk",
    "dflash2_rope_tables",
    "dflash2_attention_forward",
    "grouped_dynamic_convolve",
    "grouped_dynamic_conv_finish",
    "grouped_dynamic_conv_prepare",
    "register_dflash2_cpu_reference_kernels",
]
