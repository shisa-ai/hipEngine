from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.quant_quality.metrics import compare_logits


def _log_softmax(row: np.ndarray) -> np.ndarray:
    row = row.astype(np.float64)
    shifted = row - row.max()
    return shifted - math.log(float(np.exp(shifted).sum()))


def test_compare_logits_matches_direct_kl_and_teacher_metrics() -> None:
    reference = np.array(
        [
            [2.0, 1.0, 0.0, -1.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float16,
    )
    candidate = np.array(
        [
            [2.0, 1.0, 0.0, -1.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.9, 1.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 2], dtype=np.int64)

    result = compare_logits(
        reference,
        candidate,
        labels,
        groups=np.array(["code", "code", "general"]),
        top_k=2,
    )

    expected_kl = []
    for ref_row, cand_row in zip(reference, candidate, strict=True):
        ref_lp = _log_softmax(ref_row)
        cand_lp = _log_softmax(cand_row)
        expected_kl.append(float((np.exp(ref_lp) * (ref_lp - cand_lp)).sum()))

    assert result["rows"] == 3
    assert result["vocab_size"] == 4
    assert result["mean_kl_nats"] == pytest.approx(np.mean(expected_kl), abs=1e-12)
    assert result["max_kl_nats"] == pytest.approx(max(expected_kl), abs=1e-12)
    assert result["top1_agreement_pct"] == pytest.approx(200.0 / 3.0)
    assert result["top1_mismatch_count"] == 1
    assert result["first_top1_mismatch_row"] == 1
    assert result["max_abs_logit_delta"] == pytest.approx(3.0)
    assert result["teacher_delta_nll_nats"] > 0
    assert result["teacher_ppl"] > result["reference_teacher_ppl"]
    assert result["rms_delta_p_pct"] > 0
    assert result["top2_set_overlap_pct"] == pytest.approx(250.0 / 3.0)
    assert result["by_group"]["code"]["rows"] == 2
    assert result["by_group"]["general"]["top1_agreement_pct"] == 100.0


def test_compare_logits_rejects_misaligned_inputs() -> None:
    ref = np.zeros((2, 4), dtype=np.float32)
    candidate = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        compare_logits(ref, candidate, np.array([0, 1]))

    with pytest.raises(ValueError, match="labels"):
        compare_logits(ref, ref, np.array([0]))

    with pytest.raises(ValueError, match="groups"):
        compare_logits(
            ref,
            ref,
            np.array([0, 1]),
            groups=np.array(["only-one"]),
            top_k=2,
        )
