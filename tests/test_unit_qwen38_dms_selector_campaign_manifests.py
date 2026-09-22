from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qwen38_dms_build_selector_campaign_manifests import (
    CAMPAIGN_MATRIX,
    CATEGORIES,
    build_campaign_manifests,
)


class _Tokenizer:
    def __call__(self, text: str) -> list[int]:
        return [7] * len(text)


def _records() -> list[dict]:
    # Nine per category is the exact total required by the five split quotas.
    return [
        {
            "source_id": f"source-{category}-{index}",
            "path": f"corpus/{category}/{index}.txt",
            "category": category,
            "text": f"{category} document {index} " + ("x" * 131100),
        }
        for category in CATEGORIES
        for index in range(10)
    ]


def _old_manifests(tmp_path: Path) -> list[Path]:
    paths = []
    for version in range(1, 5):
        path = tmp_path / f"v{version}.json"
        path.write_text(
            json.dumps({"sources": [{"source_id": "historical-id", "path": "historical/path"}]}),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _build(tmp_path: Path, output: Path, records: list[dict] | None = None) -> dict:
    return build_campaign_manifests(
        _records() if records is None else records,
        old_source_manifests=_old_manifests(tmp_path),
        model_path=Path("/models/qwen.gguf"),
        output_root=output,
        tokenizer_identity="qwen35:tokenizer.model:default",
        tokenizer_sha256="a" * 64,
        tokenizer=_Tokenizer(),
        model_sha256="b" * 64,
    )


def test_campaign_matrix_has_exact_category_and_length_counts(tmp_path: Path) -> None:
    result = _build(tmp_path, tmp_path / "sealed")
    assert set(result["manifests"]) == set(CAMPAIGN_MATRIX)
    for split, spec in CAMPAIGN_MATRIX.items():
        payload = json.loads((tmp_path / "sealed" / f"{split}-data.json").read_text())
        assert payload["count"] == spec["per_category"] * len(CATEGORIES)
        assert payload["length_tokens"] == spec["length"]
        assert {row["category"] for row in payload["sequences"]} == set(CATEGORIES)
        assert all(len(row["token_ids"]) == spec["length"] for row in payload["sequences"])
        assert all(
            sum(row["category"] == category for row in payload["sequences"]) == spec["per_category"]
            for category in CATEGORIES
        )


def test_assignment_is_independent_of_input_iteration_order(tmp_path: Path) -> None:
    records = _records()
    output = tmp_path / "sealed"
    _build(tmp_path, output, records)
    first = {name: (output / name).read_bytes() for name in ("sealed-index.json", *[f"{split}-data.json" for split in CAMPAIGN_MATRIX])}
    _build(tmp_path, output, list(reversed(records)))
    second = {name: (output / name).read_bytes() for name in first}
    assert first == second


def test_duplicate_source_id_path_and_normalized_text_are_rejected(tmp_path: Path) -> None:
    records = _records()
    with pytest.raises(ValueError, match="duplicate source ID/path"):
        _build(tmp_path, tmp_path / "id", records + [dict(records[0], path="other/path")])
    with pytest.raises(ValueError, match="duplicate source ID/path"):
        _build(tmp_path, tmp_path / "path", records + [dict(records[0], source_id="other-id")])
    with pytest.raises(ValueError, match="duplicate normalized text"):
        _build(tmp_path, tmp_path / "text", records + [dict(records[0], source_id="other-id", path="other/path")])


def test_tokenizer_provenance_and_old_source_exclusions_are_sealed(tmp_path: Path) -> None:
    records = _records()
    records[0] = dict(records[0], source_id="historical-id", path="new/path")
    result = _build(tmp_path, tmp_path / "sealed", records)
    payload = json.loads((tmp_path / "sealed" / "training-expansion-data.json").read_text())
    assert all(row["provenance"]["tokenizer"]["identity"].startswith("qwen35:") for row in payload["sequences"])
    assert all(row["provenance"]["tokenizer"]["sha256"] == "a" * 64 for row in payload["sequences"])
    assert all(row["provenance"]["model_sha256"] == "b" * 64 for row in payload["sequences"])
    assert all(row["provenance"]["source_id"] != "historical-id" for row in payload["sequences"])
    assert result["sealed_index"]["sha256"]


def test_final_manifests_are_separate_and_index_has_no_candidate_result(tmp_path: Path) -> None:
    _build(tmp_path, tmp_path / "sealed")
    root = tmp_path / "sealed"
    index = json.loads((root / "sealed-index.json").read_text())
    assert index["sealed"] is True
    assert index["candidate_result"] is None
    assert set(index["manifests"]) == set(CAMPAIGN_MATRIX)
    assert (root / "final-32k-data.json").exists()
    assert (root / "final-128k-data.json").exists()
    assert (root / "final-32k-sources.json").exists()
    assert (root / "final-128k-sources.json").exists()
    assert json.loads((root / "final-32k-data.json").read_text())["split"] != "final-128k"
