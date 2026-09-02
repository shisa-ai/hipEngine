from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qwen4exp_mtp_phase_census.py"


def _load():
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_phase_census", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_rows_requires_and_orders_all_categories(tmp_path: Path) -> None:
    module = _load()
    rows = [
        {"id": category, "category": category, "messages": [{"content": category}]}
        for category in reversed(module.CATEGORIES)
    ]
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    selected = module._load_rows(path)

    assert [row["category"] for row in selected] == list(module.CATEGORIES)
    assert module._prompt_text(selected[0]) == "code"


def test_load_rows_rejects_missing_category(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"id": "code", "category": "code", "messages": [{"content": "x"}]})
        + "\n"
    )

    with pytest.raises(ValueError, match="general_en"):
        module._load_rows(path)


def test_summary_reports_distribution() -> None:
    module = _load()
    assert module._summary([3.0, 1.0, 2.0]) == {
        "sum_ms": 6.0,
        "median_ms": 2.0,
        "min_ms": 1.0,
        "max_ms": 3.0,
    }
