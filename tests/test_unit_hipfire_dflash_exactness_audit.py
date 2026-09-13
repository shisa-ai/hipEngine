from __future__ import annotations

import json
from pathlib import Path

from scripts.hipfire_dflash_exactness_audit import (
    compare_token_rows,
    load_prompt_rows,
    parse_token_rows,
    write_hipfire_prompts,
)


def test_parse_token_rows_splits_hipfire_multirow_stderr() -> None:
    stderr = """
=== dflash_spec_demo ===
@@@ ROW 0: code:quicksort_prefix @@@
decode_tokens_emitted: 4
AR tokens: [10, 11, 12, 13]
@@@ ROW 0 END @@@
@@@ ROW 1: instruct:simple_qa_no_template @@@
decode_tok_s: 22.5
AR tokens: [20, 21]
@@@ ROW 1 END @@@
"""

    tokens, metrics, labels = parse_token_rows(stderr, kind="AR")

    assert tokens == {0: [10, 11, 12, 13], 1: [20, 21]}
    assert metrics[0]["decode_tokens_emitted"] == 4.0
    assert metrics[1]["decode_tok_s"] == 22.5
    assert labels == {0: "code:quicksort_prefix", 1: "instruct:simple_qa_no_template"}


def test_compare_token_rows_distinguishes_overemit_from_hard_mismatch(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "prefix-extra", "prompt_text": "a", "max": 3}),
                json.dumps({"id": "hard-mismatch", "prompt_text": "b", "max": 4}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_prompt_rows(prompt_path, default_max_tokens=4)

    comparisons, aggregate = compare_token_rows(
        rows,
        ar_tokens={0: [1, 2, 3], 1: [4, 5, 6, 7]},
        dflash_tokens={0: [1, 2, 3, 99], 1: [4, 5, 99, 7]},
    )

    assert comparisons[0]["strict_exact"] is False
    assert comparisons[0]["prefix_equal_to_shared_len"] is True
    assert comparisons[0]["dflash_prefix_matches_full_ar"] is True
    assert comparisons[0]["over_emitted_vs_max"] is True
    assert comparisons[0]["hard_mismatch_before_shared_len"] is False

    assert comparisons[1]["hard_mismatch_before_shared_len"] is True
    assert comparisons[1]["first_mismatch_index"] == 2
    assert comparisons[1]["ar_token_at_mismatch"] == 6
    assert comparisons[1]["dflash_token_at_mismatch"] == 99

    assert aggregate["strict_exact_rows"] == 0
    assert aggregate["prefix_equal_to_shared_len_rows"] == 1
    assert aggregate["hard_mismatch_before_shared_len_rows"] == 1
    assert aggregate["all_exact_speculative_decode"] is False


def test_write_hipfire_prompts_uses_label_prompt_and_max(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "row-a", "prompt_text": "hello", "max_tokens": 8}) + "\n", encoding="utf-8")
    rows = load_prompt_rows(source, default_max_tokens=4)
    out = tmp_path / "hipfire.jsonl"

    write_hipfire_prompts(rows, out)

    assert out.read_text(encoding="utf-8") == json.dumps({"label": "row-a", "prompt": "hello", "max": 8}) + "\n"
