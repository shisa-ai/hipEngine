from __future__ import annotations

import numpy as np
import pytest

from scripts.laguna_gguf_correctness import (
    _greedy_step_metrics,
    _kl_from_reference_log_probs,
)


def test_laguna_correctness_kl_ignores_shared_logit_offset() -> None:
    reference = np.asarray([-3.0, -1.0, -2.0], dtype=np.float32)
    candidate = reference + np.float32(17.0)

    assert _kl_from_reference_log_probs(reference, candidate) == pytest.approx(0.0, abs=1e-12)


def test_laguna_greedy_step_metrics_exposes_low_margin_mismatch() -> None:
    logits = np.asarray([-2.0, 4.75, 5.0, 1.0, 3.0, 2.0], dtype=np.float32)

    metrics = _greedy_step_metrics(logits, expected_id=1, top_n=3)

    assert metrics["expected_id"] == 1
    assert metrics["expected_logit"] == pytest.approx(4.75)
    assert metrics["expected_is_top1"] is False
    assert metrics["expected_margin_to_top1"] == pytest.approx(-0.25)
    assert metrics["top1_margin"] == pytest.approx(0.25)
    assert [item["id"] for item in metrics["top"]] == [2, 1, 4]
