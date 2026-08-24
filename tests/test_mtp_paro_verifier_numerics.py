from __future__ import annotations

import numpy as np
import pytest

from scripts.mtp_paro_verifier_numerics import _summary


def test_verifier_numerical_summary_reports_binding_tail_and_top1() -> None:
    result = _summary(
        np.asarray([0.0, 1.0e-3, 2.0e-2, 5.0e-2], dtype=np.float64),
        np.asarray([True, True, False, True], dtype=np.bool_),
    )

    assert result["rows"] == 4
    assert result["mean_kl"] == pytest.approx(0.01775)
    assert result["max_kl"] == 0.05
    assert result["top1_agreement"] == 0.75
    assert result["p99_kl"] > result["p95_kl"]
