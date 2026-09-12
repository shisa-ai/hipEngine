from __future__ import annotations

from pathlib import Path

import pytest

from scripts.qwen35_gguf_int8_kv_correctness import _prompt_tokens_for_length


def test_prompt_tokens_for_length_reads_exact_token_ids_without_tokenizer(tmp_path: Path) -> None:
    prompt = tmp_path / "mixed.tokens"
    prompt.write_text("1 2\n3 2\n", encoding="utf-8")

    tokens, source = _prompt_tokens_for_length(
        model=tmp_path / "missing.gguf",
        prompt_file=None,
        prompt_text=None,
        prompt_token_file=prompt,
        token_id=9707,
        prompt_length=4,
    )

    assert tokens == [1, 2, 3, 2]
    assert source == {
        "type": "prompt_token_file",
        "path": str(prompt),
        "available_tokens": 4,
        "distinct_tokens": 3,
        "prefix_token_ids_sample": [1, 2, 3, 2],
        "prompt_token_sha256": "a8aaa835a9d64a57862dbac5fdcc0704bc4284fff4f36f1c73833de117b4cab3",
    }


def test_prompt_tokens_for_length_rejects_multiple_prompt_sources(tmp_path: Path) -> None:
    prompt = tmp_path / "mixed.tokens"
    prompt.write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        _prompt_tokens_for_length(
            model=tmp_path / "missing.gguf",
            prompt_file=tmp_path / "prompt.txt",
            prompt_text=None,
            prompt_token_file=prompt,
            token_id=9707,
            prompt_length=1,
        )
