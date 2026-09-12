from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.benchmark.prompts import (
    DEFAULT_STABLE_PROMPT_FIXTURE,
    STABLE_PROMPT_SPECS,
    build_prompt_records,
    load_prompt_records,
    load_tokenizer,
    token_ids_sha256,
    validate_prompt_records,
)

LOCAL_TOKENIZER_MODEL = Path(
    "/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/"
    "snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
)


def test_stable_prompt_specs_cover_required_groups() -> None:
    categories = {spec.category for spec in STABLE_PROMPT_SPECS}
    groups = {spec.benchmark_group for spec in STABLE_PROMPT_SPECS}

    assert {"code", "math", "instruct", "prose", "general", "multilingual"}.issubset(categories)
    assert "code_promotion" in groups
    assert "robustness" in groups
    assert sum(1 for spec in STABLE_PROMPT_SPECS if spec.promotion_gate) >= 4
    assert all(spec.prompt_id for spec in STABLE_PROMPT_SPECS)


def test_committed_dflash_prompt_fixture_hashes_and_groups() -> None:
    records = load_prompt_records(DEFAULT_STABLE_PROMPT_FIXTURE)
    errors = validate_prompt_records(records)

    assert errors == ()
    assert len(records) == 15
    assert sum(1 for row in records if row["benchmark_group"] == "code_promotion") == 6
    assert sum(1 for row in records if row["benchmark_group"] == "robustness") == 7
    assert sum(1 for row in records if row["benchmark_group"] == "synthetic_stress") == 2
    assert sum(1 for row in records if row["category"] == "general") == 1
    assert sum(1 for row in records if row["category"] == "multilingual") == 2
    assert all(row["prompt_tokens"] == len(row["prompt_ids"]) for row in records)
    assert all(row["prompt_ids_sha256"] == token_ids_sha256(row["prompt_ids"]) for row in records)
    assert all(row["prompt_preview"] for row in records)
    assert all(row["promotion_gate"] is (row["benchmark_group"] == "code_promotion") for row in records)


@pytest.mark.skipif(not (LOCAL_TOKENIZER_MODEL / "tokenizer.json").exists(), reason="local shisa tokenizer not cached")
def test_prompt_fixture_regenerates_from_local_tokenizer() -> None:
    tokenizer = load_tokenizer(LOCAL_TOKENIZER_MODEL)
    regenerated = build_prompt_records(tokenizer, tokenizer_path=LOCAL_TOKENIZER_MODEL)
    committed = load_prompt_records(DEFAULT_STABLE_PROMPT_FIXTURE)

    assert validate_prompt_records(regenerated) == ()
    assert [(r["id"], r["prompt_ids_sha256"]) for r in regenerated] == [
        (r["id"], r["prompt_ids_sha256"]) for r in committed
    ]
