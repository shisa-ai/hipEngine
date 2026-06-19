from __future__ import annotations


import pytest

from scripts.gguf_mtp_bench import (
    build_chat_prompt,
    IM_END_TOKEN,
    IM_START_TOKEN,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
)


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_build_chat_prompt_uses_supplied_user_prompt() -> None:
    tokens = build_chat_prompt(FakeTokenizer(), "Say hi")
    assert tokens[0] == IM_START_TOKEN
    assert IM_END_TOKEN in tokens
    assert tokens.count(IM_START_TOKEN) == 2
    assert tokens.count(IM_END_TOKEN) == 1
    assert tokens.count(THINK_START_TOKEN) == 1
    assert tokens.count(THINK_END_TOKEN) == 1
    decoded_text_parts = FakeTokenizer().decode(
        [t for t in tokens if t not in (IM_START_TOKEN, IM_END_TOKEN, THINK_START_TOKEN, THINK_END_TOKEN)]
    )
    assert decoded_text_parts == "user\nSay hi\nassistant\n\n\n\n\n\n"


def test_build_chat_prompt_can_omit_reasoning_suffix_for_legacy_diagnostics() -> None:
    tokens = build_chat_prompt(FakeTokenizer(), "Say hi", reasoning="none")

    assert THINK_START_TOKEN not in tokens
    assert THINK_END_TOKEN not in tokens
    decoded_text_parts = FakeTokenizer().decode([t for t in tokens if t not in (IM_START_TOKEN, IM_END_TOKEN)])
    assert decoded_text_parts == "user\nSay hi\nassistant\n"


def test_build_chat_prompt_rejects_unimplemented_reasoning_modes() -> None:
    with pytest.raises(ValueError, match="reasoning"):
        build_chat_prompt(FakeTokenizer(), "Say hi", reasoning="auto")


def test_build_chat_prompt_default_matches_france_prompt() -> None:
    tokens = build_chat_prompt(FakeTokenizer())
    decoded = FakeTokenizer().decode(tokens)

    assert "What is the capital of France?" in decoded
