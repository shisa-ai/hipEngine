from pathlib import Path

import numpy as np

from scripts.qwen4_exp_qsa_retrieval_gate import (
    _read_llama_tap,
    build_retrieval_prompt,
)


class _WhitespaceTokenizer:
    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()


def test_qsa_retrieval_prompt_is_bounded_and_retains_early_needle() -> None:
    tokenizer = _WhitespaceTokenizer()
    prompt, needle = build_retrieval_prompt(tokenizer, target_tokens=2_000)

    tokens = tokenizer.encode(prompt)
    assert len(tokens) <= 2_000
    assert len(tokens) > 1_900
    assert tokens[needle] == "VIOLET-7391."
    assert "Reasoning effort is set to low" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_read_llama_tap_returns_last_query_vector(tmp_path: Path) -> None:
    shape = np.asarray([4, 3, 1, 1], dtype=np.int64)
    values = np.arange(12, dtype=np.int32).reshape(1, 1, 3, 4)
    path = tmp_path / "tap.bin"
    path.write_bytes(shape.tobytes() + values.tobytes())

    actual_shape, actual = _read_llama_tap(path)

    assert actual_shape == [4, 3, 1, 1]
    np.testing.assert_array_equal(actual, [8, 9, 10, 11])
