from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.qwen35_paro_kv_functional_mc import (
    DEFAULT_SUITE,
    _validate_mc_tasks,
    pair_policy_results,
    score_choices,
)
from scripts.qwen35_paro_kv_quality_smoke import SuiteError, _load_suite


def test_committed_multiple_choice_suite_is_valid_and_covers_categories() -> None:
    tasks = _load_suite(DEFAULT_SUITE)

    _validate_mc_tasks(tasks)

    assert [task["category"] for task in tasks] == [
        "retrieval",
        "multihop",
        "aggregation",
        "long_doc",
        "code",
    ]
    assert len({task["expected_choice"] for task in tasks}) > 1


def test_validate_mc_tasks_rejects_expected_choice_text_mismatch(tmp_path: Path) -> None:
    task = json.loads(DEFAULT_SUITE.read_text(encoding="utf-8").splitlines()[0])
    task["expected_choice"] = "A"

    with pytest.raises(SuiteError, match="must match expected"):
        _validate_mc_tasks([task])


def test_score_choices_reports_restricted_winner_probability_and_margin() -> None:
    logits = np.zeros((10,), dtype=np.float32)
    logits[[1, 2, 3, 4]] = [0.0, 1.0, 3.0, 2.0]

    score = score_choices(logits, choice_token_ids=[1, 2, 3, 4], expected_choice="C")

    assert score["selected_choice"] == "C"
    assert score["passed"] is True
    assert score["expected_margin_vs_strongest_wrong"] == 1.0
    assert sum(score["restricted_probabilities"].values()) == pytest.approx(1.0)


def test_pair_policy_results_counts_only_reference_qualified_tasks() -> None:
    tasks = [
        {"id": "qualified", "category": "retrieval", "expected_choice": "A"},
        {"id": "unqualified", "category": "code", "expected_choice": "B"},
    ]
    reference = {
        "rows": [
            {"id": "qualified", "score": {"passed": True, "selected_choice": "A"}},
            {"id": "unqualified", "score": {"passed": False, "selected_choice": "C"}},
        ]
    }
    candidate = {
        "rows": [
            {"id": "qualified", "score": {"passed": False, "selected_choice": "D"}},
            {"id": "unqualified", "score": {"passed": True, "selected_choice": "B"}},
        ]
    }
    reference_logits = {
        "qualified": np.asarray([3.0, 2.0], dtype=np.float32),
        "unqualified": np.asarray([2.0, 3.0], dtype=np.float32),
    }
    candidate_logits = {
        "qualified": np.asarray([2.0, 3.0], dtype=np.float32),
        "unqualified": np.asarray([2.0, 3.0], dtype=np.float32),
    }

    paired, status, summary = pair_policy_results(
        tasks,
        reference,
        candidate,
        reference_logits,
        candidate_logits,
    )

    assert status == "partially_scorable"
    assert summary == {
        "reference_qualified": 1,
        "total": 2,
        "candidate_retained": 0,
        "candidate_regressions": ["qualified"],
        "reference_failures": ["unqualified"],
        "fully_scorable": False,
    }
    assert paired[0]["candidate_regression"] is True
    assert paired[1]["candidate_retained"] is False
