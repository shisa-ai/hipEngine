from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer

from hipengine.generation.qwen35_paro import (
    _is_eos,
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

    eos_token_id: int | None = None


# Token strings via chr() to survive tooling that strips <|...|> sequences.
_IM_END = chr(60) + chr(124) + "im_end" + chr(124) + chr(62)
_IM_START = chr(60) + chr(124) + "im_start" + chr(124) + chr(62)
_END_OF_TEXT = chr(60) + chr(124) + "endoftext" + chr(124) + chr(62)


def test_none_tokenizer_returns_empty() -> None:
    assert _tokenizer_eos_ids(None) == ()
    assert _tokenizer_eos_id(None) is None
    assert not _is_eos(None, 1)


def test_eos_token_id_attribute_wins_first() -> None:
    tk = _AttrTokenizer(eos_token_id=42)
    assert _tokenizer_eos_id(tk) == 42
    assert _is_eos(tk, 42)


def test_qwen_chat_tokenizer_resolves_im_end_and_endoftext() -> None:
    # The bug on main: <|im_end|> is the turn terminator but was never looked
    # up, so chat generations never stopped at end-of-turn.
    tk = _HFTokenizer(vocab={_END_OF_TEXT: 248044, _IM_START: 248045, _IM_END: 248046})
    ids = _tokenizer_eos_ids(tk)
    assert set(ids) == {248044, 248046}  # both real end tokens
    assert ids[0] == 248046  # <|im_end|> is the declared primary eos_token
    assert _is_eos(tk, 248046)  # <|im_end|>   -> stops (the fix)
    assert _is_eos(tk, 248044)  # <|endoftext|> -> stops
    assert not _is_eos(tk, 248045)  # <|im_start|> is a turn START, not EOS
    assert not _is_eos(tk, 846)  # ordinary token -> keeps generating


def test_tokenizer_without_any_eos_returns_empty() -> None:
    assert _tokenizer_eos_ids(_HFTokenizer(vocab={"hello": 1})) == ()


# --- Integration: the real PARO-shipped tokenizer.json ------------------------

_PARO_MODEL = Path("/mnt/storage/models/qwen3.6-35b-a3b-paro-packed/tokenizer.json")


@pytest.mark.skipif(not _PARO_MODEL.exists(), reason=f"missing fixture: {_PARO_MODEL}")
def test_paro_tokenizer_eos_integration() -> None:
    tk: Any = Tokenizer.from_file(str(_PARO_MODEL))
    ids = _tokenizer_eos_ids(tk)
    # generation_config.json declares eos_token_id = [248046, 248044].
    assert 248046 in ids  # <|im_end|>
    assert 248044 in ids  # <|endoftext|>
    assert 248045 not in ids  # <|im_start|> must NOT be treated as EOS
    assert _is_eos(tk, 248046)
    assert not _is_eos(tk, 248045)
