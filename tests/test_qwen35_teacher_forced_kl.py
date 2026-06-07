"""Unit tests for the teacher-forced KL / top-1 gate pure metric core.

These tests exercise only the host-side math (no HIP/ROCm, no GPU), so they run
on no-ROCm CI/publish runners.  The GPU harness in
``scripts/qwen35_batch_teacher_forced_kl.py`` guards on ``libamdhip64.so`` and is
validated separately on RX 7900 XTX.
"""

from __future__ import annotations

import math

import numpy as np

from scripts.qwen35_batch_teacher_forced_kl import (
    kl_divergence_from_logits,
    summarize_metrics,
    top1_from_logits,
)


def test_kl_identical_logits_is_zero():
    logits = np.array([0.3, -1.2, 4.5, 2.1], dtype=np.float32)
    assert kl_divergence_from_logits(logits, logits) == 0.0


def test_kl_is_nonnegative_and_offset_invariant():
    p = np.array([1.0, 2.0, 3.0])
    q = np.array([0.5, 0.1, -2.0])
    kl = kl_divergence_from_logits(p, q)
    assert kl >= 0.0
    # Softmax is invariant to a constant logit offset, so KL must be too.
    kl_shifted = kl_divergence_from_logits(p + 7.0, q - 3.0)
    assert math.isclose(kl, kl_shifted, rel_tol=1e-9, abs_tol=1e-12)


def test_kl_matches_hand_computed_value():
    # P = softmax([0, 0]) = [0.5, 0.5]; Q = softmax([0, ln 3]) = [0.25, 0.75].
    # KL(P||Q) = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75) = 0.5*ln2 + 0.5*ln(2/3).
    p_logits = np.array([0.0, 0.0])
    q_logits = np.array([0.0, math.log(3.0)])
    expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
    assert math.isclose(kl_divergence_from_logits(p_logits, q_logits), expected, rel_tol=1e-12, abs_tol=1e-12)


def test_top1_from_logits():
    assert top1_from_logits(np.array([0.1, 9.0, -3.0, 2.0])) == 1
    assert top1_from_logits(np.array([5.0, 1.0, 1.0])) == 0


def test_summary_passes_when_kl_low_and_top1_high():
    records = [{"row": r, "step": s, "kl": 0.01, "top1_match": True} for r in range(2) for s in range(8)]
    summary = summarize_metrics(records, kl_threshold=0.05, top1_threshold=0.90)
    assert summary["passed"] is True
    assert summary["kl_passed"] is True
    assert summary["top1_passed"] is True
    assert summary["n"] == 16
    assert math.isclose(summary["mean_kl"], 0.01, abs_tol=1e-12)
    assert summary["top1_fraction"] == 1.0
    assert set(summary["per_row"].keys()) == {"0", "1"}


def test_summary_fails_on_low_top1_even_with_low_kl():
    # Half the steps disagree on top-1 -> top1_fraction 0.5 < 0.90 -> fail, despite low KL.
    records = []
    for s in range(10):
        records.append({"row": 0, "step": s, "kl": 0.001, "top1_match": s % 2 == 0})
    summary = summarize_metrics(records, kl_threshold=0.05, top1_threshold=0.90)
    assert summary["passed"] is False
    assert summary["kl_passed"] is True
    assert summary["top1_passed"] is False
    assert math.isclose(summary["top1_fraction"], 0.5, abs_tol=1e-12)


def test_summary_fails_on_high_mean_kl_even_with_perfect_top1():
    records = [{"row": 0, "step": s, "kl": 0.2, "top1_match": True} for s in range(10)]
    summary = summarize_metrics(records, kl_threshold=0.05, top1_threshold=0.90)
    assert summary["passed"] is False
    assert summary["kl_passed"] is False
    assert summary["top1_passed"] is True


def test_summary_empty_records():
    summary = summarize_metrics([], kl_threshold=0.05, top1_threshold=0.90)
    assert summary["passed"] is False
    assert summary["n"] == 0
