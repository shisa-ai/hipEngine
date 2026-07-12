from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from hipengine.generation import GenerationRequest
from hipengine.generation.qwen35_paro import (
    _is_eos,
    _request_with_tokenizer_eos,
    _tokenizer_eos_id,
    _tokenizer_eos_ids,
)


@dataclass
class _HFTokenizer:
    """Mimics ``tokenizers.Tokenizer``: a callable ``token_to_id`` and no
    ``eos_token_id`` attribute (that only exists on ``transformers``)."""

    vocab: dict[str, int] = field(default_factory=dict)

    def token_to_id(self, token: str) -> int | None:
        return self.vocab.get(token)


@dataclass
class _AttrTokenizer:
    """Mimics a ``transformers``-style tokenizer exposing ``eos_token_id``."""

    eos_token_id: Any = None


@dataclass
class _MappingTokenizer:
    """Mimics tokenizers that expose a token-to-id mapping."""

    token_to_id: dict[str, int] = field(default_factory=dict)


# Token strings via chr() to survive tooling that strips <|...|> sequences.
_IM_END = chr(60) + chr(124) + "im_end" + chr(124) + chr(62)
_IM_START = chr(60) + chr(124) + "im_start" + chr(124) + chr(62)
_END_OF_TEXT = chr(60) + chr(124) + "endoftext" + chr(124) + chr(62)
_S_END = "</s>"


def _request(**overrides: Any) -> GenerationRequest:
    values: dict[str, Any] = {
        "prompts": ("hello",),
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def test_none_tokenizer_returns_empty() -> None:
    assert _tokenizer_eos_ids(None) == ()
    assert _tokenizer_eos_id(None) is None
    assert not _is_eos(None, 1)


def test_eos_token_id_attribute_wins_first() -> None:
    tk = _AttrTokenizer(eos_token_id=42)
    assert _tokenizer_eos_id(tk) == 42
    assert _is_eos(tk, 42)


def test_eos_token_id_attribute_accepts_sequences_and_ignores_invalid_ids() -> None:
    tk = _AttrTokenizer(eos_token_id=(42, "43", 42, -1, None, "invalid"))

    assert _tokenizer_eos_ids(tk) == (42, 43)


def test_qwen_chat_tokenizer_resolves_im_end_and_endoftext() -> None:
    # The bug on main: <|im_end|> is the turn terminator but was never looked
    # up, so chat generations never stopped at end-of-turn.
    tk = _HFTokenizer(
        vocab={
            _END_OF_TEXT: 248044,
            _IM_START: 248045,
            _IM_END: 248046,
            _S_END: 248047,
        }
    )
    ids = _tokenizer_eos_ids(tk)
    assert ids == (248046, 248044)
    assert _is_eos(tk, 248046)  # <|im_end|>   -> stops (the fix)
    assert _is_eos(tk, 248044)  # <|endoftext|> -> stops
    assert not _is_eos(tk, 248045)  # <|im_start|> is a turn START, not EOS
    assert not _is_eos(tk, 248047)  # non-Qwen </s> is not inferred as EOS
    assert not _is_eos(tk, 846)  # ordinary token -> keeps generating


def test_qwen_eos_fallback_supports_token_to_id_mappings() -> None:
    tk = _MappingTokenizer(token_to_id={_END_OF_TEXT: 248044, _IM_END: 248046})

    assert _tokenizer_eos_ids(tk) == (248046, 248044)


def test_tokenizer_without_any_eos_returns_empty() -> None:
    assert _tokenizer_eos_ids(_HFTokenizer(vocab={"hello": 1})) == ()


def test_request_uses_primary_qwen_eos_without_overriding_explicit_eos() -> None:
    tk = _HFTokenizer(vocab={_END_OF_TEXT: 248044, _IM_END: 248046})
    inferred = _request_with_tokenizer_eos(_request(), tk)

    assert inferred.eos_token_id == 248046
    explicit = replace(inferred, eos_token_id=7)
    assert _request_with_tokenizer_eos(explicit, tk) is explicit


def test_tokenizers_runtime_interface_resolves_qwen_eos_ids() -> None:
    tk = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                _END_OF_TEXT: 248044,
                _IM_START: 248045,
                _IM_END: 248046,
            },
            unk_token="[UNK]",
        )
    )

    assert not hasattr(tk, "eos_token_id")
    assert _tokenizer_eos_ids(tk) == (248046, 248044)


def _find_paro_model_fixture() -> Path | None:
    configured = os.environ.get("HIPENGINE_TEST_PARO_MODEL")
    candidates = [
        *([Path(configured)] if configured else []),
        Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"),
        Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16"),
        Path("/mnt/storage/models/qwen3.6-35b-a3b-paro-packed"),
    ]
    for candidate in candidates:
        model = candidate.parent if candidate.name == "tokenizer.json" else candidate
        required = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")
        if all((model / name).is_file() for name in required):
            return model
    return None


# Optional integration against a local PARO model; the portable WordLevel test
# above always exercises the production ``tokenizers.Tokenizer`` interface.
_PARO_MODEL = _find_paro_model_fixture()


@pytest.mark.skipif(_PARO_MODEL is None, reason="missing local PARO tokenizer fixture")
def test_paro_tokenizer_eos_integration() -> None:
    assert _PARO_MODEL is not None
    generation_config = json.loads((_PARO_MODEL / "generation_config.json").read_text())
    tokenizer_config = json.loads((_PARO_MODEL / "tokenizer_config.json").read_text())
    tk = Tokenizer.from_file(str(_PARO_MODEL / "tokenizer.json"))
    ids = _tokenizer_eos_ids(tk)

    assert generation_config["eos_token_id"] == [248046, 248044]
    assert tokenizer_config["eos_token"] == _IM_END
    assert ids == (248046, 248044)
    assert _is_eos(tk, 248046)
    assert not _is_eos(tk, 248045)
