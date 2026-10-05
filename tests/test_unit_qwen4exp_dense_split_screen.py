"""CPU-only checks for the split screen's independent distribution metrics."""

import numpy as np

from scripts.qwen4exp_dense_split_screen import distribution_metrics


def test_split_screen_distribution_checks_detect_wrong_math():
    reference = np.array([[1.0, 2.0, 4.0], [2.0, -1.0, 0.0]])
    same = distribution_metrics(reference, reference.copy())
    assert same["kl_mean"] == 0
    assert same["relative_l2"] == 0
    assert same["top1"] == 1
    wrong = distribution_metrics(reference, -reference)
    assert wrong["kl_mean"] > 0.05
    assert wrong["top1"] < 0.9
    assert wrong["relative_l2"] == 2


def test_split_screen_distribution_is_row_shift_invariant():
    reference = np.array([[1.0, 2.0, 4.0], [2.0, -1.0, 0.0]])
    shifted = reference + np.array([[5.0], [-2.0]])
    metrics = distribution_metrics(reference, shifted)
    assert metrics["kl_mean"] == 0
    assert metrics["top1"] == 1
    assert metrics["relative_l2"] > 0
