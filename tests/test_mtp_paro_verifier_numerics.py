from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mtp_paro_verifier_numerics
from scripts.mtp_paro_verifier_numerics import (
    _capture_sha256,
    _review_manifests,
    _row_review_diagnostic,
    _scope_summaries,
    _summary,
    _wilson_interval,
)


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


def test_top1_summary_reports_resolution_and_wilson_interval() -> None:
    result = _summary(
        np.zeros((68,), dtype=np.float64),
        np.asarray([True] * 67 + [False], dtype=np.bool_),
    )
    expected = 67 / 68
    low, high = _wilson_interval(67, 68)

    assert result["top1_agreement"] == pytest.approx(expected)
    assert result["top1_matches"] == 67
    assert result["top1_mismatches"] == 1
    assert result["top1_wilson95_low"] == pytest.approx(low)
    assert result["top1_wilson95_high"] == pytest.approx(high)
    assert low < expected < high


def test_row_review_diagnostic_localizes_near_tie_flip() -> None:
    strict = np.asarray([5.0, 4.8, 1.0, 0.0], dtype=np.float32)
    candidate = np.asarray([4.9, 5.0, 1.0, 0.0], dtype=np.float32)
    result = _row_review_diagnostic(strict, candidate, top_k=3)

    assert result["strict_topk"] == [0, 1, 2]
    assert result["candidate_topk"] == [1, 0, 2]
    assert result["strict_margin"] == pytest.approx(0.2)
    assert result["candidate_margin"] == pytest.approx(0.1)
    assert result["strict_top1_candidate_rank"] == 2
    assert result["candidate_top1_strict_rank"] == 2
    assert result["strict_gap_to_candidate_top1"] == pytest.approx(0.2)
    assert result["candidate_gap_to_strict_top1"] == pytest.approx(0.1)


def test_scope_summaries_bind_each_row_role() -> None:
    common = {
        "category": "general_en",
        "shape": "c2_b1",
        "transition": "verify_to_verify",
        "decision_role": "draft_acceptance_or_reject_correction",
        "strict_selected_for_commit": True,
        "top5_overlap": 1.0,
        "strict_margin": 0.1,
    }
    rows = [
        {**common, "row_role": "root", "kl": 1.0e-4, "top1_equal": False},
        {
            **common,
            "row_role": "draft_candidate",
            "strict_selected_for_commit": False,
            "kl": 1.0e-4,
            "top1_equal": True,
        },
    ]
    result = _scope_summaries(rows)

    assert result["row_role"]["root"]["top1_agreement"] == 0.0
    assert result["row_role"]["root"]["passed"] is False
    assert result["row_role"]["draft_candidate"]["top1_agreement"] == 1.0
    assert result["row_role"]["draft_candidate"]["passed"] is True


def test_capture_hash_binds_prompt_manifest_rows_and_cycles() -> None:
    kwargs = {
        "prompt": {"name": "fixture", "tokens": 4},
        "candidate_manifest_sha256": "a" * 64,
        "rows": [{"row": 0, "kl": 1.0e-4}],
        "cycles": [{"cycle": 1, "root": 7}],
    }
    first = _capture_sha256(**kwargs)
    second = _capture_sha256(**kwargs)
    changed = _capture_sha256(**{**kwargs, "cycles": [{"cycle": 1, "root": 8}]})

    assert first == second
    assert len(first) == 64
    assert changed != first


def test_fast_review_manifest_is_unselected_with_strict_fallback() -> None:
    result = _review_manifests()
    rows = result["candidate_review_manifest"]["selections"]
    verifier = next(row for row in rows if row["layer"] == "mtp_verifier_route")

    assert result["candidate_registered_but_uncertified"] is True
    assert result["candidate_selected_by_production"] is False
    assert verifier["selected_variant"] == "b1_graph_off_fast_d64_candidate"
    assert verifier["strict_fallback_variant"] == "b1_graph_off_strict_exact"
    assert len(result["candidate_review_sha256"]) == 64


def test_verifier_capture_reads_materialized_verify_logits(monkeypatch) -> None:
    copied: list[tuple[int, int]] = []

    def fake_copy(_host_ptr, buffer, nbytes, *, runtime):
        del runtime
        copied.append((int(buffer.ptr), int(nbytes)))

    monkeypatch.setattr(mtp_paro_verifier_numerics, "copy_device_to_host", fake_copy)
    session = SimpleNamespace(
        vocab_size=4,
        verify_lm_logits=SimpleNamespace(ptr=0xABC000),
        batch_lm_logits=SimpleNamespace(ptr=0xBAD000),
        runtime=object(),
    )
    logits = mtp_paro_verifier_numerics._copy_logits(session, rows=2)

    assert logits.shape == (2, 4)
    assert copied == [(0xABC000, 2 * 4 * 4)]
