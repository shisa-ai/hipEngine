from __future__ import annotations

import numpy as np
import pytest

from scripts.qwen4_exp_compare_logits import compare_logits


def test_qwen4_exp_compare_logits_reports_kl_top1_and_errors() -> None:
    teacher = np.array([0.0, 2.0, -1.0], dtype=np.float32)
    actual = np.array([0.1, 1.8, -0.8], dtype=np.float32)

    report = compare_logits(teacher, actual)

    assert report["teacher_top1"] == 1
    assert report["hipengine_top1"] == 1
    assert report["top1_agreement"] is True
    assert 0.0 < report["kl_teacher_to_hipengine"] < 0.05
    assert report["mean_absolute_logit_error"] == pytest.approx(1.0 / 6.0)
    assert report["max_absolute_logit_error"] == pytest.approx(0.2)


def test_qwen4_exp_compare_logits_rejects_shape_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="same 1D shape"):
        compare_logits(np.zeros(2, dtype=np.float32), np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        compare_logits(
            np.array([0.0, np.inf], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
