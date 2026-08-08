from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pm4_promotion_gate import (
    _compare_rows,
    _load_prompt_cases,
    _message_content,
    _repeat_to_length,
    build_parser,
)


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(text), sum(text.encode("utf-8")) % 251]


def test_pm4_promotion_gate_loads_categories_and_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite.jsonl"
    records = [
        {
            "id": "code_a",
            "category": "code",
            "messages": [{"role": "user", "content": "write code"}],
        },
        {
            "id": "ja_a",
            "category": "general_ja",
            "messages": [{"role": "user", "content": "説明してください"}],
        },
    ]
    suite.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    cases = _load_prompt_cases([suite], _FakeTokenizer())

    assert [case.name for case in cases] == ["code_a", "ja_a"]
    assert [case.category for case in cases] == ["code", "general_ja"]
    assert all(case.token_ids for case in cases)
    with pytest.raises(ValueError, match="duplicate"):
        _load_prompt_cases([suite, suite], _FakeTokenizer())


def test_pm4_promotion_gate_requires_nonempty_user_messages() -> None:
    assert (
        _message_content(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second"},
                ]
            }
        )
        == "first\n\nsecond"
    )
    with pytest.raises(ValueError, match="user messages only"):
        _message_content({"messages": [{"role": "system", "content": "no"}]})


def test_pm4_promotion_gate_context_repeat_and_exact_comparison() -> None:
    assert _repeat_to_length((1, 2, 3), 8) == (1, 2, 3, 1, 2, 3, 1, 2)
    with pytest.raises(ValueError):
        _repeat_to_length((), 8)

    row = {
        "seed_token_id": 7,
        "final_token_id": 8,
        "state_sha256": "state",
        "final_logits_sha256": "logits",
    }
    assert all(_compare_rows(row, dict(row)).values())
    mismatch = dict(row, state_sha256="other")
    comparison = _compare_rows(row, mismatch)
    assert comparison["state_exact"] is False
    assert comparison["final_logits_exact"] is True


def test_pm4_promotion_gate_defaults_cover_natural_and_4k_context() -> None:
    args = build_parser().parse_args(["--json", "/tmp/result.json"])

    assert len(args.suite_files) == 2
    assert args.steps == 3
    assert args.context_stress_length == 4096
