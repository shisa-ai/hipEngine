from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import gguf_ngram_mtp_code_bench as code_bench
from scripts import specdec2_perf_bridge as bridge


def test_code_repetition_wrapper_binds_fixed_train_and_heldout_contract() -> None:
    code_bench.configure_code_repetition_contract()

    rows = bridge.load_prompt_suite(code_bench.DEFAULT_CODE_REPETITION_PROMPTS)

    assert tuple(row["id"] for row in rows) == code_bench.CODE_REPETITION_PROMPT_IDS
    assert {row["category"] for row in rows} == {"code"}
    assert bridge._HELDOUT_IDS == code_bench.CODE_REPETITION_HELDOUT_IDS
    assert bridge._REQUIRED_CATEGORIES == frozenset({"code"})
    assert all(row["rendered_prompt"].endswith("<|im_start|>assistant\n") for row in rows)
    assert all(
        row["prompt_sha256"]
        == hashlib.sha256(row["rendered_prompt"].encode("utf-8")).hexdigest()
        for row in rows
    )


def test_code_repetition_fixture_is_repo_owned_and_has_no_prompt_aliases() -> None:
    path = code_bench.DEFAULT_CODE_REPETITION_PROMPTS

    assert path.is_file()
    assert path.resolve().is_relative_to(Path.cwd().resolve())
    source = path.read_text(encoding="utf-8")
    assert '"split":"train"' in source
    assert '"split":"heldout"' in source
    assert source.count('"id":') == len(code_bench.CODE_REPETITION_PROMPT_IDS)
