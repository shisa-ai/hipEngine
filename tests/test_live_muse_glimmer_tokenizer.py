from __future__ import annotations

import json
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import MuseGlimmerGGUFTokenizer

MODEL = Path(
    "/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)
FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "muse_glimmer"
    / "tokenizer_llamacpp_e0f142d5d.json"
)


@pytest.fixture(scope="module")
def tokenizer() -> MuseGlimmerGGUFTokenizer:
    if not MODEL.exists():
        pytest.skip(f"local Muse Glimmer GGUF not found: {MODEL}")
    return MuseGlimmerGGUFTokenizer.from_gguf_info(GGUFReader(MODEL).info)


@pytest.fixture(scope="module")
def oracle_rows() -> list[dict]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["kind"] == "muse_glimmer_tokenizer_oracle"
    return fixture["rows"]


def test_muse_glimmer_tokenizer_loads_special_contract(
    tokenizer: MuseGlimmerGGUFTokenizer,
) -> None:
    assert len(tokenizer.tokens) == 202_048
    assert tokenizer.bos_token_id == 200_000
    assert tokenizer.eos_token_id == 200_001
    assert tokenizer.eot_token_id == 200_008
    assert tokenizer.padding_token_id == 200_018
    assert tokenizer.add_bos_token is True
    assert tokenizer.tokens[tokenizer.bos_token_id] == "<|begin_of_text|>"
    assert tokenizer.tokens[tokenizer.eot_token_id] == "<|eot|>"
    assert "ATEM" in tokenizer.chat_template
    assert tokenizer.stop_token_ids == (200_001, 200_008)


def test_muse_glimmer_tokenizer_matches_llamacpp_oracle(
    tokenizer: MuseGlimmerGGUFTokenizer,
    oracle_rows: list[dict],
) -> None:
    checked = 0
    for row in oracle_rows:
        if row["id"] == "edge_special_literal":
            # llama.cpp --no-parse-special byte-splits special literals; the
            # product tokenizer always overlays added special tokens (the
            # parse_special=true row below pins that behavior).
            continue
        expected = row["ids"]
        actual = tokenizer.encode(row["text"], add_special_tokens=False)
        assert actual == expected, f"{row['id']}: {actual} != {expected}"
        checked += 1
    assert checked >= 18


def test_muse_glimmer_tokenizer_parses_special_literals(
    tokenizer: MuseGlimmerGGUFTokenizer,
    oracle_rows: list[dict],
) -> None:
    row = next(r for r in oracle_rows if r["id"] == "edge_special_literal_parsed")
    assert tokenizer.encode(row["text"], add_special_tokens=False) == row["ids"]
    assert row["ids"][0] == tokenizer.eot_token_id


def test_muse_glimmer_tokenizer_bos_insertion(
    tokenizer: MuseGlimmerGGUFTokenizer,
    oracle_rows: list[dict],
) -> None:
    row = next(r for r in oracle_rows if r["id"] == "edge_english")
    assert tokenizer.encode(row["text"], add_special_tokens=True) == [
        tokenizer.bos_token_id,
        *row["ids"],
    ]


def test_muse_glimmer_tokenizer_decode_roundtrip(
    tokenizer: MuseGlimmerGGUFTokenizer,
    oracle_rows: list[dict],
) -> None:
    for row in oracle_rows:
        if row["parse_special"]:
            continue
        assert tokenizer.decode(row["ids"]) == row["text"], row["id"]
        assert tokenizer.decode(row["ids"], skip_special=True) == row["text"], row["id"]


def test_muse_glimmer_tokenizer_rejects_wrong_pretokenizer() -> None:
    if not MODEL.exists():
        pytest.skip(f"local Muse Glimmer GGUF not found: {MODEL}")
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
    with pytest.raises(ValueError, match="gpt2.*llama4"):
        MuseGlimmerGGUFTokenizer.from_gguf_info(wrong)
