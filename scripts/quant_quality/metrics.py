"""Stable BF16-relative full-logit quantization-quality metrics.

The definitions mirror ParoQuant's canonical evaluator. PPL/NLL values here are
named ``teacher_*`` because portable prompt-suite labels come from the BF16
teacher trajectory rather than a held-out natural-text corpus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        "reference_margin_logit": np.empty(n_rows, dtype=np.float64),
        "total_variation": np.empty(n_rows, dtype=np.float64),
        "entropy_delta_nats": np.empty(n_rows, dtype=np.float64),
        "max_probability_delta": np.empty(n_rows, dtype=np.float64),
    }
    for i in range(n_rows):
        ref_lp = _log_softmax(reference[i])
        candidate_lp = _log_softmax(candidate[i])
        ref_p = np.exp(ref_lp)
        candidate_p = np.exp(candidate_lp)
        label = int(labels_array[i])
        rows["total_variation"][i] = float(0.5 * np.abs(candidate_p - ref_p).sum())
        rows["entropy_delta_nats"][i] = float(
            np.sum(ref_p * ref_lp) - np.sum(candidate_p * candidate_lp)
        )
        rows["max_probability_delta"][i] = float(candidate_p.max() - ref_p.max())
        if vocab_size == 1:
            rows["reference_margin_logit"][i] = 0.0
        else:
            top_two = np.partition(np.asarray(reference[i], dtype=np.float64), -2)[-2:]
            rows["reference_margin_logit"][i] = float(top_two.max() - top_two.min())

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


def _flip_summary(equal: np.ndarray) -> dict[str, Any]:
    """Report a descriptive rate and a row-independent Wilson approximation."""
    count = int(equal.size)
    flips = int(np.count_nonzero(~equal))
    interval = None
    rate = None
    if count:
        rate = flips / count
        z2 = 1.959963984540054 ** 2
        denominator = 1.0 + z2 / count
        center = (rate + z2 / (2 * count)) / denominator
        radius = np.sqrt(rate * (1 - rate) / count + z2 / (4 * count ** 2)) / denominator * np.sqrt(z2)
        interval = {"low": max(0.0, float(center - radius)),
                    "high": min(1.0, float(center + radius))}
    return {"rows": count, "flips": flips, "rate": rate, "wilson95_iid_rows": interval}


def summarize_diagnostics(
    rows: Mapping[str, np.ndarray], *, sequence_count: int | None = None,
) -> dict[str, Any]:
    """Summarize report-only vectors; signed deltas are candidate minus reference.

    Sequence counts describe capture coverage, not independent observations.
    Wilson intervals assume independent, identically distributed token rows;
    correlated contexts and deterministic repeats do not justify that assumption.
    """
    equal = rows["top1_equal"]
    margins = rows["reference_margin_logit"]
    bounds = ((0.0, 0.05), (0.05, 0.5), (0.5, 2.0), (2.0, None))
    buckets = []
    for lower, upper in bounds:
        mask = margins >= lower
        if upper is not None:
            mask &= margins < upper
        buckets.append({"lower_inclusive": lower, "upper_exclusive": upper,
                        **_flip_summary(equal[mask])})
    result = {
        "gating": False,
        "rows": int(equal.size),
        "sequence_count": sequence_count,
        "uncertainty_note": (
            "Wilson intervals assume independent identically distributed rows; "
            "token rows can be correlated. Sequence counts are coverage, not "
            "independent sample counts. Do not pool deterministic repeats."
        ),
        "top1_flips": _flip_summary(equal),
        "margin_buckets": buckets,
    }
    for key in ("total_variation", "entropy_delta_nats", "max_probability_delta"):
        values = rows[key]
        percentiles = np.percentile(values, (5, 50, 95, 99))
        result[key] = {
            "mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()),
            **dict(zip(("p05", "p50", "p95", "p99"), map(float, percentiles), strict=True)),
        }
    return result


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
        "diagnostics": summarize_diagnostics(selected),
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
