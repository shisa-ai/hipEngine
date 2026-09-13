from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import scan_gguf
from hipengine.tokenization.gguf import Qwen4ExpGGUFTokenizer

UNSLOTH_PART0 = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
)


def _tokenizer() -> Qwen4ExpGGUFTokenizer:
    if not UNSLOTH_PART0.exists():
        pytest.skip(f"local qwen4exp metadata split missing: {UNSLOTH_PART0}")
    return Qwen4ExpGGUFTokenizer.from_gguf_info(scan_gguf(UNSLOTH_PART0))


def test_qwen4_exp_tokenizer_matches_frozen_real_gguf_ids() -> None:
    tokenizer = _tokenizer()

    assert tokenizer.encode("Hello, world!") == [9419, 11, 1814, 0]
    assert tokenizer.encode("こんにちは") == [85951]
    assert tokenizer.encode("def fibonacci(n):") == [727, 73111, 1393, 1590]
    assert tokenizer.decode([9419, 11, 1814, 0]) == "Hello, world!"


def test_qwen4_exp_tokenizer_preserves_template_and_special_tokens() -> None:
    tokenizer = _tokenizer()

    assert tokenizer.bos_token_id == 248_044
    assert tokenizer.eos_token_id == 248_046
    assert tokenizer.padding_token_id == 248_044
    assert tokenizer.add_bos_token is False
    assert tokenizer.stop_token_ids == (248_046,)
    assert tokenizer.token_to_id["<|im_start|>"] == 248_045
    assert tokenizer.token_to_id["<|im_end|>"] == 248_046
    assert tokenizer.token_to_id["<|image_pad|>"] == 248_056
    assert "reasoning_effort" in tokenizer.chat_template
    assert "<|im_start|>assistant" in tokenizer.chat_template
