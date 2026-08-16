"""Stable BF16-relative full-logit quantization-quality metrics.

The definitions mirror ParoQuant's canonical evaluator. PPL/NLL values here are
named ``teacher_*`` because portable prompt-suite labels come from the BF16
teacher trajectory rather than a held-out natural-text corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


_PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.9)


def _log_softmax(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.float64)
    shifted = values - np.max(values)
    return shifted - np.log(np.exp(shifted).sum())


def per_row_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, np.ndarray]:
    """Return aligned per-row vectors used by summaries and paired bootstrap."""
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape != candidate.shape:
        raise ValueError(
            f"reference/candidate shape mismatch: {reference.shape!r} vs {candidate.shape!r}"
        )
    n_rows, vocab_size = reference.shape
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.shape != (n_rows,):
        raise ValueError(f"labels must have shape ({n_rows},), got {labels_array.shape!r}")
    if np.any(labels_array < 0) or np.any(labels_array >= vocab_size):
        raise ValueError("labels contain token IDs outside the logits vocabulary")
    if top_k <= 0 or top_k > vocab_size:
        raise ValueError(f"top_k must be in [1, {vocab_size}], got {top_k}")

    rows = {
        "kl_nats": np.empty(n_rows, dtype=np.float64),
        "reference_teacher_nll_nats": np.empty(n_rows, dtype=np.float64),
        "teacher_nll_nats": np.empty(n_rows, dtype=np.float64),
        "delta_p": np.empty(n_rows, dtype=np.float64),
        "top1_equal": np.empty(n_rows, dtype=np.bool_),
        "topk_set_overlap": np.empty(n_rows, dtype=np.float64),
        "max_abs_logit_delta": np.empty(n_rows, dtype=np.float64),
    }
    for i in range(n_rows):
        ref_lp = _log_softmax(reference[i])
        candidate_lp = _log_softmax(candidate[i])
        ref_p = np.exp(ref_lp)
        candidate_p = np.exp(candidate_lp)
        label = int(labels_array[i])

        # Small negative values can arise only from floating-point summation.
        rows["kl_nats"][i] = max(float(np.sum(ref_p * (ref_lp - candidate_lp))), 0.0)
        rows["reference_teacher_nll_nats"][i] = -ref_lp[label]
        rows["teacher_nll_nats"][i] = -candidate_lp[label]
        rows["delta_p"][i] = candidate_p[label] - ref_p[label]
        rows["top1_equal"][i] = int(np.argmax(reference[i])) == int(np.argmax(candidate[i]))
        rows["max_abs_logit_delta"][i] = float(
            np.max(np.abs(np.asarray(reference[i], dtype=np.float64) - candidate[i]))
        )

        ref_topk = np.argpartition(reference[i], -top_k)[-top_k:]
        candidate_topk = np.argpartition(candidate[i], -top_k)[-top_k:]
        rows["topk_set_overlap"][i] = np.intersect1d(ref_topk, candidate_topk).size / top_k
    return rows


def _summarize_vectors(
    rows: dict[str, np.ndarray],
    *,
    vocab_size: int,
    top_k: int,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = rows if mask is None else {key: values[mask] for key, values in rows.items()}
    n_rows = int(selected["kl_nats"].size)
    if n_rows == 0:
        raise ValueError("cannot summarize zero metric rows")
    mean_ref_nll = float(selected["reference_teacher_nll_nats"].mean())
    mean_candidate_nll = float(selected["teacher_nll_nats"].mean())
    percentiles = np.percentile(selected["kl_nats"], _PERCENTILES)
    percentile_names = ("p50", "p90", "p95", "p99", "p99_9")
    mismatch_rows = np.flatnonzero(~selected["top1_equal"])
    return {
        "rows": n_rows,
        "vocab_size": int(vocab_size),
        "mean_kl_nats": float(selected["kl_nats"].mean()),
        "max_kl_nats": float(selected["kl_nats"].max()),
        "kl_percentiles": {
            name: float(value) for name, value in zip(percentile_names, percentiles, strict=True)
        },
        "reference_teacher_mean_nll_nats": mean_ref_nll,
        "teacher_mean_nll_nats": mean_candidate_nll,
        "teacher_delta_nll_nats": mean_candidate_nll - mean_ref_nll,
        "reference_teacher_ppl": float(np.exp(mean_ref_nll)),
        "teacher_ppl": float(np.exp(mean_candidate_nll)),
        "mean_delta_p_pct": float(100.0 * selected["delta_p"].mean()),
        "rms_delta_p_pct": float(100.0 * np.sqrt(np.mean(selected["delta_p"] ** 2))),
        "top1_agreement_pct": float(100.0 * selected["top1_equal"].mean()),
        "top1_mismatch_count": int(mismatch_rows.size),
        "first_top1_mismatch_row": None if mismatch_rows.size == 0 else int(mismatch_rows[0]),
        f"top{top_k}_set_overlap_pct": float(100.0 * selected["topk_set_overlap"].mean()),
        "max_abs_logit_delta": float(selected["max_abs_logit_delta"].max()),
    }


def compare_logits(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    *,
    groups: Sequence[str] | np.ndarray | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare aligned ``[rows, vocab]`` candidate logits against BF16.

    The function accepts regular arrays or ``numpy.memmap`` values so a caller
    can compare large ``.npy`` caches without loading both fully into RAM.
    """
    rows = per_row_metrics(reference, candidate, labels, top_k=top_k)
    n_rows, vocab_size = reference.shape
    result = _summarize_vectors(rows, vocab_size=vocab_size, top_k=top_k)
    if groups is None:
        return result

    groups_array = np.asarray(groups, dtype=str)
    if groups_array.shape != (n_rows,):
        raise ValueError(f"groups must have shape ({n_rows},), got {groups_array.shape!r}")
    result["by_group"] = {}
    for group in sorted(np.unique(groups_array)):
        result["by_group"][str(group)] = _summarize_vectors(
            rows,
            vocab_size=vocab_size,
            top_k=top_k,
            mask=groups_array == group,
        )
    return result
