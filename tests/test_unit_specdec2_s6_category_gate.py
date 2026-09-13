from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.specdec2_s6_category_gate import (
    _HELDOUT_IDS,
    counterbalanced_route_order,
    load_prompt_suite,
)


def test_category_route_order_is_balanced_and_prompt_content_independent() -> None:
    orders = tuple(counterbalanced_route_order(index) for index in range(10))

    assert orders.count(("ar", "mtp")) == 5
    assert orders.count(("mtp", "ar")) == 5


def test_specdec2_category_gate_loads_canonical_train_and_heldout_contract() -> None:
    path = Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl")

    rows = load_prompt_suite(path)

    assert len(rows) == 10
    assert {row["category"] for row in rows} == {
        "code",
        "general_en",
        "general_ja",
        "mixed_ja_en",
    }
    assert {row["id"] for row in rows if row["id"] in _HELDOUT_IDS} == _HELDOUT_IDS
    assert all(len(row["prompt_sha256"]) == 64 for row in rows)
    assert all(row["rendered_prompt"].endswith("<|im_start|>assistant\n") for row in rows)


def test_specdec2_category_gate_rejects_missing_explicit_category(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="category"):
        load_prompt_suite(path)
