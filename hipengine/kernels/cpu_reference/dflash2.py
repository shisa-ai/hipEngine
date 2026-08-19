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

from hipengine.kernels.cpu_reference.ops import linear
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
    if logits_arr.shape[-1] != vocab:
        raise ValueError(f"logits vocab {logits_arr.shape[-1]} != codebook vocab {vocab}")
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
    "grouped_dynamic_convolve",
    "grouped_dynamic_conv_finish",
    "grouped_dynamic_conv_prepare",
    "register_dflash2_cpu_reference_kernels",
]
