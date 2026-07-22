from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import LagunaGGUFTokenizer, _pretokenize_laguna

MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
POOLSIDE_TEMPLATE_FIXTURE = (
    Path(__file__).parent / "fixtures" / "laguna_poolside_v1_template.json"
)


@pytest.fixture(scope="module")
def tokenizer() -> LagunaGGUFTokenizer:
    if not MODEL.exists():
        pytest.skip(f"local Laguna GGUF not found: {MODEL}")
    return LagunaGGUFTokenizer.from_gguf_info(GGUFReader(MODEL).info)


def test_laguna_pretokenizer_keeps_lf_runs_as_boundaries() -> None:
    assert _pretokenize_laguna("line 1\n\nline 2") == [
        "line",
        " ",
        "1",
        "\n\n",
        "line",
        " ",
        "2",
    ]


def test_laguna_tokenizer_loads_special_contract(tokenizer: LagunaGGUFTokenizer) -> None:
    assert len(tokenizer.tokens) == 100_352
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 2
    assert tokenizer.eot_token_id == 24
    assert tokenizer.padding_token_id == 9
    assert tokenizer.separator_token_id == 8
    assert tokenizer.mask_token_id == 12
    assert tokenizer.unknown_token_id == 0
    assert tokenizer.add_bos_token is True
    assert tokenizer.tokens[tokenizer.eot_token_id] == "</assistant>"
    assert "laguna_glm_thinking_v8" in tokenizer.chat_template


def test_laguna_tokenizer_matches_hf_fast_tokenizer_prompt_fixtures(
    tokenizer: LagunaGGUFTokenizer,
) -> None:
    fixtures = {
        "Hello": [6_352],
        "The capital of France is": [785, 9_626, 377, 15_360, 395],
        "line 1\nline 2": [1_030, 290, 86, 268, 1_030, 290, 87],
        "日本語と English 123!": [76_211, 53_336, 16_262, 4_618, 290, 86, 87, 88, 70],
        "We're testing 123!": [2_583, 2_264, 2_607, 290, 86, 87, 88, 70],
        "a\r\nb": [134, 489, 135],
        "x!!!\n\ny": [157, 14_906, 350, 158],
        "e\u0301 cafe": [138, 41_879, 70_551],
        "〈|EOS|〉": [2],
        "<think>reason</think>": [18, 32_781, 19],
        "abc</assistant>def": [9_873, 24, 1_172],
    }
    for text, expected in fixtures.items():
        assert tokenizer.encode(text, add_special_tokens=False) == expected
        assert tokenizer.encode(text, add_special_tokens=True) == [2, *expected]


def test_laguna_poolside_template_fixture_matches_gguf_tokenizer(
    tokenizer: LagunaGGUFTokenizer,
) -> None:
    fixture = json.loads(POOLSIDE_TEMPLATE_FIXTURE.read_text())
    assert fixture["poolside_llama_commit"] == (
        "04b2b72cb54048ead292884adbe11f284e3ec950"
    )
    assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == fixture[
        "template_sha256"
    ]
    cases = {case["name"]: case for case in fixture["cases"]}
    assert cases["no_thinking"]["rendered"].endswith("<assistant></think>")
    assert cases["thinking"]["rendered"].endswith("<assistant><think>")
    assert "<arg_key>city</arg_key><arg_value>Paris</arg_value>" in cases[
        "tool_history"
    ]["rendered"]
    for case in cases.values():
        assert tokenizer.encode(case["rendered"]) == case["token_ids"]


def test_laguna_tokenizer_suppresses_eot_text_when_skipping_special(
    tokenizer: LagunaGGUFTokenizer,
) -> None:
    assert tokenizer.decode([6_352, 24]) == "Hello</assistant>"
    assert tokenizer.decode([6_352, 24], skip_special=True) == "Hello"
    assert tokenizer.decode([18, 19, 23, 25, 26], skip_special=True) == (
        "<think></think><assistant><tool_call></tool_call>"
    )
    assert tokenizer.stop_token_ids == (2, 24)


def test_laguna_tokenizer_rejects_wrong_pretokenizer() -> None:
    if not MODEL.exists():
        pytest.skip(f"local Laguna GGUF not found: {MODEL}")
    info = GGUFReader(MODEL).info
    metadata = dict(info.metadata)
    metadata["tokenizer.ggml.pre"] = "qwen35"
    wrong = type(info)(
        path=info.path,
        version=info.version,
        alignment=info.alignment,
        metadata=metadata,
        tensors=info.tensors,
        tensor_data_offset=info.tensor_data_offset,
    )

    with pytest.raises(ValueError, match="gpt2.*laguna"):
        LagunaGGUFTokenizer.from_gguf_info(wrong)
