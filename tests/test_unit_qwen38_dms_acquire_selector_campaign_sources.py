from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from scripts.qwen38_dms_acquire_selector_campaign_sources import (
    WIKI_REVISION,
    _cli_documents,
    acquire_records,
    build_parser,
    write_atomic,
)


class Tok:
    def __call__(self, text: str) -> list[int]:
        marker = 101 if text.startswith("general_en") else 106 if text.startswith("general_ja") else 99
        return [marker] * int(text.split(":")[-1])


def docs(n: int = 18) -> list[dict]:
    rows = []
    for category, count in (("code", 9), ("general_en", n), ("general_ja", n)):
        for index in range(count):
            rows.append({"source_id": f"{category}-{index}", "path": f"{category}/{index}", "category": category, "text": f"{category}:{index}:12", "dataset": category, "revision": "r", "license": "CC0"})
    return rows


def build(rows: list[dict], **kwargs: object) -> list[dict]:
    return acquire_records(rows, tokenizer=Tok(), tokenizer_identity="fixture", tokenizer_sha256="a" * 64, model_sha256="b" * 64, target_tokens=8, mixed_chunk_tokens=2, **kwargs)


def test_exact_streams_are_deterministic_disjoint_and_mixed(tmp_path: Path) -> None:
    first = build(docs())
    second = build(list(reversed(docs())))
    assert first == second
    assert len(first) == 36
    assert all(len(row["token_ids"]) == 8 for row in first)
    owned = [component["source_id"] for row in first for component in row["source_records"]]
    assert len(owned) == len(set(owned))
    mixed = [row for row in first if row["category"] == "mixed_ja_en"]
    assert len(mixed) == 9
    assert all(row["token_ids"][0] != row["token_ids"][2] for row in mixed)
    assert all("text" not in component for row in first for component in row["source_records"])
    output = tmp_path / "records.json"
    binding = {"identity": "fixture", "sha256": "a" * 64}
    write_atomic(first, output, {"tokenizer": binding})
    sealed = json.loads(output.read_text(encoding="utf-8"))
    assert sealed["tokenizer"] == binding
    assert '"text"' not in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError): write_atomic(first, output, {})


def test_cli_pins_wikipedia_revision_and_uses_campaign_sized_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def fake_load_dataset(*args, **kwargs):
        calls.append({"args": args, **kwargs})
        return iter([{"id": "1", "title": "title", "text": "x" * 1200}])

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setattr(
        "scripts.qwen38_dms_acquire_selector_campaign_sources.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="", returncode=0),
    )
    monkeypatch.setattr(
        "scripts.qwen38_dms_acquire_selector_campaign_sources.sysconfig.get_path",
        lambda name: str(tmp_path),
    )
    rows = _cli_documents(tmp_path, (set(), set(), set()), record_limit=1)
    assert len(rows) == 2
    assert all(call["revision"] == WIKI_REVISION for call in calls)
    assert build_parser().get_default("record_limit") == 4096


def test_exclusions_and_duplicate_rejection(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"source_records": [{"source_id": "code-0", "path": "old", "normalized_text_sha256": "0" * 64}]}), encoding="utf-8")
    rows = docs(); rows[0]["path"] = "new"
    with pytest.raises(ValueError, match="insufficient code"):
        build(rows, old_manifests=[old])
    duplicate = docs() + [dict(docs()[1], path="other")]
    with pytest.raises(ValueError, match="duplicate source"):
        build(duplicate)


def test_insufficient_source_and_tokenizer_mismatch_fail_closed() -> None:
    with pytest.raises(ValueError, match="insufficient general_ja"):
        build([row for row in docs() if row["category"] != "general_ja"])
    rows = docs(); rows[0]["tokenizer"] = {"identity": "wrong", "sha256": "a" * 64}
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        build(rows)
