"""Live tier: Gemma 4 GGUF tokenizer against the reference HF tokenizer.

The unit tier pins the recipe's mechanics on a hand-built vocabulary. This file
pins the reconstruction itself: it encodes a mixed corpus with both the
tokenizer built from the GGUF and the ``tokenizer.json`` that shipped with the
same model, and requires the ids to agree exactly.

The corpus is generated deterministically from real source text plus randomized
Unicode and whitespace, so it covers the cases a hand-written fixture list
misses: long newline runs, combining marks, emoji, astral-plane characters,
tabs, and embedded control tokens.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import Gemma4GGUFTokenizer

MODEL = Path(
    "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
HF_REFERENCE = MODEL.parent / "hf-ref" / "tokenizer.json"

CORPUS_SEED = 20260922
RANDOM_CHARS = (
    "abcdefgHIJKL 0123456789\n\t.,!?;:()[]{}\"'`~@#$%^&*-+=/\\|<>_"
    "\u65e5\u672c\u8a9e\U0001f389\u00e9\u03c0\U0001d573\u00a0\u200b"
)


@pytest.fixture(scope="module")
def tokenizer() -> Gemma4GGUFTokenizer:
    if not MODEL.exists():
        pytest.skip(f"local Gemma 4 GGUF not found: {MODEL}")
    return Gemma4GGUFTokenizer.from_gguf_info(GGUFReader(MODEL).info)


@pytest.fixture(scope="module")
def reference() -> Tokenizer:
    if not HF_REFERENCE.exists():
        pytest.skip(f"reference HF tokenizer not found: {HF_REFERENCE}")
    return Tokenizer.from_file(str(HF_REFERENCE))


def build_corpus() -> list[str]:
    rng = random.Random(CORPUS_SEED)
    corpus: list[str] = [
        "",
        " ",
        "\n",
        "Hello world",
        "The capital of France is",
        "  leading and trailing spaces  ",
        "\n" * 40,
        "a\r\nb",
        "def f(x):\n    return x + 1\n",
        "<|turn>user\nhello<turn|>\n",
        "<|tool_response>{\"ok\": true}<turn|>",
        "\U0001f389\U0001f680\u00e9\u0301\u03c0",
        "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059",
        "\U0001d573\U0001d576\U0001d591\U0001d591\U0001d594",
        "\u0000\u0001\u0002\u007f",
        "SELECT * FROM t WHERE id = 1;",
    ]
    # Real source text, so the corpus contains long natural token streams.
    for relative in (
        "hipengine/tokenization/gguf.py",
        "hipengine/models/gemma4.py",
        "hipengine/loading/gemma4_gguf.py",
        "AGENTS.md",
        "docs/PLAN.md",
    ):
        source = Path(__file__).resolve().parent.parent / relative
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8")
        for start in range(0, len(text), 997):
            corpus.append(text[start : start + 400])
    for _ in range(600):
        length = rng.randint(1, 80)
        corpus.append("".join(rng.choice(RANDOM_CHARS) for _ in range(length)))
    return [text for text in corpus if text]


def test_tokenizer_matches_the_reference_hf_ids(
    tokenizer: Gemma4GGUFTokenizer,
    reference: Tokenizer,
) -> None:
    corpus = build_corpus()
    mismatches: list[tuple[str, list[int], list[int]]] = []
    for text in corpus:
        expected = reference.encode(text, add_special_tokens=False).ids
        actual = tokenizer.encode(text, add_special_tokens=False)
        if expected != actual:
            mismatches.append((text, expected, actual))
    assert not mismatches, (
        f"{len(mismatches)} of {len(corpus)} texts disagreed with the reference "
        f"tokenizer; first: {mismatches[0][0][:80]!r}"
    )


def test_encoding_round_trips_through_the_gguf_decoder(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    failures = [
        text
        for text in build_corpus()
        if tokenizer.decode(tokenizer.encode(text)) != text
    ]
    assert not failures, f"round trip lost text for {failures[:3]!r}"


def test_special_contract_comes_from_the_real_artifact(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    assert len(tokenizer.tokens) == 262_144
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 106
    assert tokenizer.tokens[tokenizer.eos_token_id] == "<turn|>"
    assert tokenizer.add_bos_token is True
    assert tokenizer.eog_token_ids == (106, 1, 50)
    assert [tokenizer.tokens[index] for index in tokenizer.eog_token_ids] == [
        "<turn|>",
        "<eos>",
        "<|tool_response>",
    ]
    assert "<|turn>" in tokenizer.chat_template


def test_prompt_fixture_ids_are_stable(tokenizer: Gemma4GGUFTokenizer) -> None:
    """Ids captured from the reference tokenizer, pinned so drift is visible."""

    fixtures = {
        "Hello world": [9259, 1902],
        "The capital of France is": [818, 5279, 529, 7001, 563],
        "def f(x):\n    return x + 1\n": [
            2063,
            517,
            236769,
            236781,
            1473,
            107,
            140,
            2060,
            1123,
            900,
            236743,
            236770,
            107,
        ],
        "<|turn>user\nhello<turn|>\n": [105, 2364, 107, 23391, 106, 107],
        "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059": [
            94951,
            236945,
            95830,
            3652,
        ],
    }
    for text, expected in fixtures.items():
        assert tokenizer.encode(text, add_special_tokens=False) == expected, text


def test_chat_template_renders_and_encodes(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    """The template must render, and its markers must be single tokens."""

    from jinja2 import Environment

    if not tokenizer.chat_template:
        pytest.skip("artifact carries no chat template")

    def raise_helper(message: str):
        raise AssertionError(message)

    environment = Environment(trim_blocks=True, lstrip_blocks=True, autoescape=False)
    environment.globals["raise_exception"] = raise_helper
    template = environment.from_string(tokenizer.chat_template)
    rendered = template.render(
        messages=[{"role": "user", "content": "hello"}],
        bos_token="<bos>",
        eos_token="<eos>",
        add_generation_prompt=True,
    )
    assert "<|turn>user" in rendered
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    assert tokenizer.token_to_id["<|turn>"] in ids
    assert tokenizer.token_to_id["<turn|>"] in ids
    assert json.dumps(rendered)[0] == '"'
