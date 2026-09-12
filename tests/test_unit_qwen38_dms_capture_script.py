from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.qwen38_dms_capture import load_capture_sequences


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


def _manifest() -> dict:
    categories = ("code", "general_en", "general_ja", "mixed_ja_en")
    return {
        "schema_version": 1,
        "kind": "hipengine_dms_training_data_manifest",
        "tokenizer": {"identity": "fixture-tokenizer", "sha256": "a" * 64},
        "datasets": [
            {
                "name": "fixture-corpus",
                "revision": "rev-1",
                "license": "CC0",
                "splits": ["train", "validation"],
                "filters": {"dedupe": "sha256"},
            }
        ],
        "sequence_construction": {"format": "plain", "separator": "eos"},
        "context_length_distribution": {"4": 4},
        "split_policy": {"method": "sha256-mod", "validation_mod": 10},
        "seed": 0,
        "sequences": [
            {
                "sequence_id": f"fixture-{category}",
                "category": category,
                "split": "train" if index < 3 else "validation",
                "token_ids": [index + 1, 2, 3, 4, 5],
                "provenance": {"dataset": "fixture-corpus", "row": index},
            }
            for index, category in enumerate(categories)
        ],
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_data_manifest_binds_tokenizer_categories_and_sequence_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.json"
    payload = _manifest()
    payload["sequences"][0].pop("token_ids")
    payload["sequences"][0]["text"] = "abcdef"
    _write(path, payload)

    manifest_hash, sequences = load_capture_sequences(
        path,
        tokenizer=_Tokenizer(),
        tokenizer_identity_value="fixture-tokenizer",
        tokenizer_sha256="a" * 64,
        max_sequence_tokens=4,
    )

    assert manifest_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(sequences) == 4
    assert sequences[0].token_ids == tuple(ord(char) for char in "abcd")
    assert sequences[0].provenance["captured_token_count"] == 4
    assert {sequence.category for sequence in sequences} == {
        "code",
        "general_en",
        "general_ja",
        "mixed_ja_en",
    }


def test_data_manifest_rejects_evaluation_suite_references(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    payload = _manifest()
    payload["datasets"][0]["filters"] = {
        "copied_from": "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
    }
    _write(path, payload)

    with pytest.raises(ValueError, match="evaluation-only mtp-bench"):
        load_capture_sequences(
            path,
            tokenizer=_Tokenizer(),
            tokenizer_identity_value="fixture-tokenizer",
            tokenizer_sha256="a" * 64,
            max_sequence_tokens=4,
        )


def test_data_manifest_rejects_tokenizer_mismatch_and_missing_categories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.json"
    payload = _manifest()
    _write(path, payload)
    with pytest.raises(ValueError, match="tokenizer hash"):
        load_capture_sequences(
            path,
            tokenizer=_Tokenizer(),
            tokenizer_identity_value="fixture-tokenizer",
            tokenizer_sha256="b" * 64,
            max_sequence_tokens=4,
        )

    payload["sequences"] = payload["sequences"][:1]
    _write(path, payload)
    with pytest.raises(ValueError, match="requires all training categories"):
        load_capture_sequences(
            path,
            tokenizer=_Tokenizer(),
            tokenizer_identity_value="fixture-tokenizer",
            tokenizer_sha256="a" * 64,
            max_sequence_tokens=4,
        )
