"""Unit tier: Gemma 4 tokenizer reconstructed from GGUF metadata.

Gemma 4 is the first hipEngine tokenizer that is not a GPT-2 byte-BPE model, so
these tests pin the three properties that make it different: an SPM-style space
marker with raw UTF-8 merges, real ``<0xXX>`` byte fallback, and literal
matching of control and user-defined tokens.

The vocabulary here is hand-built and tiny, so the expected ids are stated
symbolically through ``FIXTURE_TOKEN_IDS``. Agreement with the reference
HuggingFace tokenizer on real text is covered by
``tests/test_live_gemma4_gguf_tokenizer.py``, which needs the artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from hipengine.loading.gguf import scan_gguf
from hipengine.tokenization.gguf import Gemma4GGUFTokenizer
from tests._gemma4_gguf_fixture import (
    FIXTURE_BOS_TOKEN_ID,
    FIXTURE_EOS_TOKEN_ID,
    FIXTURE_MERGES,
    FIXTURE_SPM_SPACE,
    FIXTURE_TOKEN_IDS,
    FIXTURE_TOKEN_STRINGS,
    default_fixture_tensors,
    fixture_metadata,
    tokenizer_fixture_metadata,
    write_fixture_gguf,
)

IDS = FIXTURE_TOKEN_IDS


def build_tokenizer(tmp_path: Path, **overrides: object) -> Gemma4GGUFTokenizer:
    metadata = tokenizer_fixture_metadata(**overrides)
    path = write_fixture_gguf(
        tmp_path / "tokenizer.gguf",
        default_fixture_tensors(),
        metadata,
    )
    return Gemma4GGUFTokenizer.from_gguf_info(scan_gguf(path))


@pytest.fixture
def tokenizer(tmp_path: Path) -> Gemma4GGUFTokenizer:
    return build_tokenizer(tmp_path)


def test_tokenizer_is_reconstructed_from_gguf_metadata(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    assert tokenizer.encoder_backend == "huggingface_tokenizers"
    assert isinstance(tokenizer.encoder, Tokenizer)
    assert len(tokenizer.tokens) == len(FIXTURE_TOKEN_STRINGS)
    assert tokenizer.merges == FIXTURE_MERGES
    assert not hasattr(tokenizer, "_bpe")
    assert not hasattr(tokenizer, "_cache")


def test_special_contract_comes_from_the_metadata(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    assert tokenizer.bos_token_id == FIXTURE_BOS_TOKEN_ID
    assert tokenizer.eos_token_id == FIXTURE_EOS_TOKEN_ID
    assert tokenizer.tokens[tokenizer.eos_token_id] == "<turn|>"
    assert tokenizer.add_bos_token is True
    assert "<|turn>" in tokenizer.chat_template


def test_normalizer_replaces_spaces_with_the_spm_marker(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    # Only the interior space becomes U+2581; the words merge with it.
    assert tokenizer.encode("hello world") == [IDS["hello"], IDS[FIXTURE_SPM_SPACE + "world"]]
    # A leading space is an SPM marker attached to the following word.
    assert tokenizer.encode(" abc") == [IDS[FIXTURE_SPM_SPACE + "abc"]]
    # A bare marker is its own token when no merge consumes it.
    assert tokenizer.encode("a b") == [IDS["a"], IDS[FIXTURE_SPM_SPACE], IDS["b"]]


def test_merges_run_on_raw_utf8_without_a_byte_encoding(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    # ``hello`` is a merged token and ``▁hello`` is a different merged token;
    # under a GPT-2 byte encoding neither string would appear at all.
    assert tokenizer.encode("hello") == [IDS["hello"]]
    assert tokenizer.encode(FIXTURE_SPM_SPACE + "hello") == [
        IDS[FIXTURE_SPM_SPACE + "hello"]
    ]
    assert IDS["hello"] != IDS[FIXTURE_SPM_SPACE + "hello"]


def test_newline_runs_are_split_before_merging(tokenizer: Gemma4GGUFTokenizer) -> None:
    assert tokenizer.encode("hello world\n") == [
        IDS["hello"],
        IDS[FIXTURE_SPM_SPACE + "world"],
        IDS["\n"],
    ]
    # llama.cpp splits on newlines so a merge can never span one.
    assert tokenizer.encode("\n\n\n\n") == [IDS["\n"]] * 4


def test_control_tokens_match_literally(tokenizer: Gemma4GGUFTokenizer) -> None:
    assert tokenizer.encode("<turn|>") == [IDS["<turn|>"]]
    assert tokenizer.encode("a<turn|>b") == [IDS["a"], IDS["<turn|>"], IDS["b"]]


def test_user_defined_tokens_match_literally(tokenizer: Gemma4GGUFTokenizer) -> None:
    assert tokenizer.encode("<|tool_response>") == [IDS["<|tool_response>"]]


def test_a_non_control_eos_is_still_matched_literally(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    # The real artifact writes ``<eos>`` with GGUF type NORMAL. llama.cpp
    # overrides that to control and warns the file is wrong; we match it, so the
    # literal string never byte-splits into ordinary tokens.
    assert FIXTURE_TOKEN_STRINGS[IDS["<eos>"]][1] == 1
    assert tokenizer.encode("<eos>") == [IDS["<eos>"]]


def test_eog_token_ids_cover_the_gemma4_end_of_generation_set(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    assert tokenizer.eog_token_ids == (
        IDS["<turn|>"],
        IDS["<eos>"],
        IDS["<|tool_response>"],
    )
    assert tokenizer.stop_token_ids == tokenizer.eog_token_ids


def test_control_tokens_are_skipped_only_when_decoding_asks_for_it(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    ids = [IDS["<turn|>"], IDS["abc"]]
    assert tokenizer.decode(ids) == "<turn|>abc"
    assert tokenizer.decode(ids, skip_special=True) == "abc"


def test_user_defined_tokens_render_even_when_special_tokens_are_skipped(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    ids = [IDS["<turn|>"], IDS["<|tool_response>"], IDS["abc"]]
    assert tokenizer.decode(ids, skip_special=True) == "<|tool_response>abc"


def test_spm_markers_decode_back_to_spaces(tokenizer: Gemma4GGUFTokenizer) -> None:
    assert (
        tokenizer.decode([IDS[FIXTURE_SPM_SPACE + "hello"], IDS[FIXTURE_SPM_SPACE + "world"]])
        == " hello world"
    )


def test_out_of_vocabulary_characters_fall_back_to_byte_tokens(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    assert tokenizer.encode("z") == [IDS["<0x7A>"]]
    # ``é`` is two UTF-8 bytes and neither is in the vocabulary.
    assert tokenizer.encode("\u00e9") == [IDS["<0xC3>"], IDS["<0xA9>"]]


def test_byte_tokens_decode_back_to_their_text(tokenizer: Gemma4GGUFTokenizer) -> None:
    assert tokenizer.decode([IDS["<0x7A>"]]) == "z"
    assert tokenizer.decode([IDS["<0xC3>"], IDS["<0xA9>"]]) == "\u00e9"


def test_an_undecodable_byte_sequence_keeps_its_literal_form(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    # The source decoder chain keeps ``<0xXX>`` text for a broken sequence.
    assert tokenizer.decode([IDS["<0xFF>"]]) == "<0xFF>"
    assert tokenizer.decode([IDS["<0xC3>"], IDS["abc"]]) == "<0xC3>abc"


def test_bos_is_inserted_only_when_requested(tokenizer: Gemma4GGUFTokenizer) -> None:
    assert tokenizer.encode("abc") == [IDS["abc"]]
    assert tokenizer.encode("abc", add_special_tokens=True) == [
        FIXTURE_BOS_TOKEN_ID,
        IDS["abc"],
    ]


def test_bos_insertion_is_refused_without_a_bos_id(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path, drop=("tokenizer.ggml.bos_token_id",))
    assert tokenizer.bos_token_id is None
    with pytest.raises(ValueError, match="BOS insertion"):
        tokenizer.encode("abc", add_special_tokens=True)


def test_round_trip_preserves_text(tokenizer: Gemma4GGUFTokenizer) -> None:
    for text in (
        "",
        "abc",
        " abc",
        "hello world",
        "hello world\n",
        "\n\n\n",
        "\u00e9",
        "z\u00e9",
        "<|tool_response>abc",
    ):
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_decoding_rejects_a_token_id_outside_the_vocabulary(
    tokenizer: Gemma4GGUFTokenizer,
) -> None:
    with pytest.raises(ValueError, match="outside vocabulary size"):
        tokenizer.decode([len(FIXTURE_TOKEN_STRINGS)])
    with pytest.raises(ValueError, match="outside vocabulary size"):
        tokenizer.decode([-1])


def test_a_foreign_tokenizer_model_is_refused(tmp_path: Path) -> None:
    metadata = tokenizer_fixture_metadata(tokenizer_model="gpt2")
    path = write_fixture_gguf(
        tmp_path / "foreign.gguf",
        default_fixture_tensors(),
        metadata,
    )
    with pytest.raises(ValueError, match="expected model 'gemma4'"):
        Gemma4GGUFTokenizer.from_gguf_info(scan_gguf(path))


def test_a_missing_token_type_array_is_reported(tmp_path: Path) -> None:
    metadata = [
        entry
        for entry in fixture_metadata()
        if entry[0] != "tokenizer.ggml.token_type"
    ]
    path = write_fixture_gguf(tmp_path / "plain.gguf", default_fixture_tensors(), metadata)
    with pytest.raises(KeyError):
        Gemma4GGUFTokenizer.from_gguf_info(scan_gguf(path))
