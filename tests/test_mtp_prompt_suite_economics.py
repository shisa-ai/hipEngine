from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOL_PATH = Path("scripts/mtp_prompt_suite_economics.py")


def _load_tool():
    module_name = "hipengine_mtp_prompt_suite_economics_tool"
    spec = importlib.util.spec_from_file_location(module_name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_load_prompt_suite_accepts_canonical_category_jsonl(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "prompts.jsonl"
    rows = [
        {
            "id": "code_merge_intervals",
            "category": "code",
            "messages": [{"role": "user", "content": "Write code."}],
        },
        {
            "id": "general_en_explain",
            "category": "general_en",
            "messages": [{"role": "user", "content": "Explain it."}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    suite = tool._load_prompt_suite(path)

    assert suite["source_format"] == "jsonl"
    assert suite["prompts"] == [
        {
            "name": "code_merge_intervals",
            "prompt": "Write code.",
            "category": "code",
            "split": "train",
        },
        {
            "name": "general_en_explain",
            "prompt": "Explain it.",
            "category": "general_en",
            "split": "heldout",
        },
    ]
    metadata = tool._prompt_suite_metadata(path, suite, suite["prompts"])
    assert metadata["source"] == str(path.resolve())
    assert metadata["source_format"] == "jsonl"
    assert metadata["prompt_ids"] == ["code_merge_intervals", "general_en_explain"]
    assert metadata["category_counts"] == {"code": 1, "general_en": 1}
    assert metadata["split_counts"] == {"train": 1, "heldout": 1}
    assert metadata["train_ids"] == ["code_merge_intervals"]
    assert metadata["heldout_ids"] == ["general_en_explain"]


def test_load_prompt_suite_preserves_legacy_json(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps({"source": "legacy", "prompts": [{"name": "one", "prompt": "Hello"}]}),
        encoding="utf-8",
    )

    suite = tool._load_prompt_suite(path)

    assert suite["source"] == "legacy"
    assert suite["source_format"] == "json"
    assert suite["prompts"] == [{"name": "one", "prompt": "Hello"}]


def test_load_prompt_suite_rejects_non_user_jsonl_messages(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "category": "code",
                "messages": [{"role": "assistant", "content": "No"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one user message"):
        tool._load_prompt_suite(path)
