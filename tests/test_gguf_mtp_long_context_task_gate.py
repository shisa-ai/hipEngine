from __future__ import annotations

import json

import pytest

from scripts.gguf_mtp_long_context_task_gate import (
    CATEGORIES,
    TaskSuiteError,
    load_tasks,
    score_task_output,
)


def _row(task_id: str, category: str, expected: str = "B") -> dict[str, object]:
    return {
        "id": task_id,
        "category": category,
        "target_context_tokens": 32,
        "prefix": "prefix",
        "filler": "filler",
        "evidence": [{"position": 0.5, "text": "fact"}],
        "suffix": "suffix",
        "expected": ["answer"],
        "scorer": "choice_exact",
        "choices": {"A": "wrong-a", "B": "answer", "C": "wrong-c", "D": "wrong-d"},
        "expected_choice": expected,
    }


def test_committed_task_fixture_covers_all_categories() -> None:
    rows = load_tasks(
        __import__("pathlib").Path("benchmarks/prompts/mtp-realworld-long-context.jsonl")
    )

    assert tuple(row["category"] for row in rows) == CATEGORIES
    assert all(row["target_context_tokens"] == 4096 for row in rows)
    assert len({row["id"] for row in rows}) == len(CATEGORIES)


def test_load_tasks_fails_closed_on_missing_category(tmp_path) -> None:
    path = tmp_path / "suite.jsonl"
    path.write_text(json.dumps(_row("one", "retrieval")) + "\n")

    with pytest.raises(TaskSuiteError, match="every RF1 category"):
        load_tasks(path)


def test_score_task_output_accepts_choice_or_declared_answer_text() -> None:
    task = _row("one", "retrieval")

    assert score_task_output("B", task)["passed"] is True
    assert score_task_output("The answer is answer.", task)["passed"] is True
    assert score_task_output("C", task)["passed"] is False
