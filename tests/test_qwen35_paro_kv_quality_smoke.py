from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import qwen35_paro_kv_quality_smoke as smoke


class CharacterTokenizer:
    def encode(self, text: str):
        return type("Encoding", (), {"ids": [ord(char) for char in text]})()

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)


def test_committed_suite_has_one_task_per_bounded_category() -> None:
    rows = smoke._load_suite(smoke.DEFAULT_SUITE)

    assert [row["category"] for row in rows] == list(smoke.CATEGORIES)
    assert len({row["id"] for row in rows}) == len(smoke.CATEGORIES)
    assert all(row["target_context_tokens"] == 4096 for row in rows)


def test_load_suite_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = {
        "id": "duplicate",
        "category": "retrieval",
        "target_context_tokens": 32,
        "prefix": "prefix",
        "filler": "filler",
        "evidence": [{"position": 0.5, "text": "fact"}],
        "suffix": "question",
        "expected": ["answer"],
        "scorer": "final_exact",
    }
    path = tmp_path / "suite.jsonl"
    path.write_text("\n".join((json.dumps(row), json.dumps(row))) + "\n", encoding="utf-8")

    with pytest.raises(smoke.SuiteError, match="duplicate id"):
        smoke._load_suite(path)


def test_prompt_builder_preserves_exact_length_and_evidence_positions() -> None:
    task = {
        "id": "builder",
        "prompt_format": "raw",
        "target_context_tokens": 100,
        "prefix": "PP",
        "filler": "f",
        "evidence": [
            {"position": 0.25, "text": "AAA"},
            {"position": 0.75, "text": "BBB"},
        ],
        "suffix": "SS",
    }

    tokens, metadata = smoke._build_prompt_tokens(CharacterTokenizer(), task)
    text = "".join(chr(token) for token in tokens)

    assert len(tokens) == 100
    assert text.startswith("PP")
    assert text.endswith("SS")
    assert text.count("AAA") == 1
    assert text.count("BBB") == 1
    assert metadata["fixed_tokens"] == 10
    assert metadata["filler_tokens"] == 90
    assert metadata["evidence_offsets"][0]["token_offset"] == 24
    assert metadata["evidence_offsets"][1]["token_offset"] == 73
    assert len(metadata["prompt_token_ids_sha256"]) == 64


def test_prompt_builder_rejects_target_smaller_than_fixed_content() -> None:
    task = {
        "id": "too-small",
        "prompt_format": "raw",
        "target_context_tokens": 3,
        "prefix": "PP",
        "filler": "f",
        "evidence": [{"position": 0.5, "text": "AAA"}],
        "suffix": "SS",
    }

    with pytest.raises(smoke.SuiteError, match="above target"):
        smoke._build_prompt_tokens(CharacterTokenizer(), task)


@pytest.mark.parametrize(
    ("output", "expected", "passed", "normalized"),
    [
        ("FINAL: ORCHID - 7319\n", ["ORCHID-7319"], True, "orchid-7319"),
        ("work\nFINAL: 90 days.\n", ["90 days", "90"], True, "90 days"),
        ("FINAL: 59", ["60"], False, "59"),
        ("the answer is 60", ["60"], False, None),
    ],
)
def test_final_exact_scorer(output, expected, passed, normalized) -> None:
    score = smoke._score_output(output, expected)

    assert score["passed"] is passed
    assert score["normalized_answer"] == normalized


def _task(task_id: str, category: str = "retrieval") -> dict[str, str]:
    return {"id": task_id, "category": category}


def _result(task_id: str, *, passed: bool, token_ids: list[int]) -> dict[str, object]:
    return {"id": task_id, "score": {"passed": passed}, "output_token_ids": token_ids}


def test_pair_results_accepts_task_score_retention_without_exact_tokens() -> None:
    tasks = [_task("a"), _task("b", "code")]
    reference = [_result("a", passed=True, token_ids=[1]), _result("b", passed=True, token_ids=[2])]
    candidate = [_result("a", passed=True, token_ids=[9]), _result("b", passed=True, token_ids=[2])]

    paired, status, summary = smoke._pair_results(tasks, reference, candidate)

    assert status == "accepted_smoke"
    assert summary["paired_non_regression"] is True
    assert paired[0]["output_token_ids_match"] is False
    assert paired[0]["candidate_regression"] is False


def test_pair_results_distinguishes_candidate_regression_and_reference_failure() -> None:
    tasks = [_task("a")]
    regression = smoke._pair_results(
        tasks,
        [_result("a", passed=True, token_ids=[1])],
        [_result("a", passed=False, token_ids=[2])],
    )
    unscorable = smoke._pair_results(
        tasks,
        [_result("a", passed=False, token_ids=[1])],
        [_result("a", passed=True, token_ids=[2])],
    )

    assert regression[1] == "candidate_quality_regression"
    assert regression[2]["candidate_regressions"] == ["a"]
    assert unscorable[1] == "reference_unscorable"
    assert unscorable[2]["reference_failures"] == ["a"]


def test_select_tasks_rejects_unknown_category() -> None:
    with pytest.raises(smoke.SuiteError, match="unknown categories"):
        smoke._select_tasks([_task("a")], categories="not-real", limit=None)
