from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def _tokenizer() -> Qwen35GGUFTokenizer:
    return Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(MODEL).info)


def test_qwen35_gguf_tokenizer_matches_e2e_fixture() -> None:
    tokenizer = _tokenizer()

    assert tokenizer.encode("The answer is") == [760, 4087, 369]
    assert tokenizer.decode([220, 16, 13, 271]) == " 1.\n\n"
    assert tokenizer.encode(" 1.\n\n") == [220, 16, 13, 271]
    assert tokenizer.decode([760, 4087, 369, 220, 16, 13, 271]) == "The answer is 1.\n\n"


def test_qwen35moe_gguf_tokenizer_matches_smoke_fixture() -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(MOE_MODEL).info)

    assert tokenizer.encode("Hello") == [9419]
    assert tokenizer.decode([9419]) == "Hello"
    assert tokenizer.encode("izio.") == [43482, 13]
    assert tokenizer.decode([43482, 13]) == "izio."
    assert tokenizer.decode([9419, 43482, 13]) == "Helloizio."


def test_qwen35_gguf_tokenizer_round_trips_common_ascii_prompts() -> None:
    tokenizer = _tokenizer()

    examples = [
        "Hello",
        "AMD GPUs are",
        "The answer is 1.",
        "line one\nline two",
    ]
    for text in examples:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_qwen35_gguf_tokenizer_decodes_special_tokens() -> None:
    tokenizer = _tokenizer()

    assert tokenizer.decode([248046]) == "<|im_end|>"
    assert tokenizer.decode([248046], skip_special=True) == ""
    assert tokenizer.eos_token_id == 248046
    assert tokenizer.padding_token_id == 248055


def test_qwen35_gguf_tokenizer_does_not_import_torch() -> None:
    torch_preloaded = "torch" in sys.modules
    tokenizer = _tokenizer()

    assert tokenizer.encode("The answer is") == [760, 4087, 369]
    assert ("torch" in sys.modules) is torch_preloaded
