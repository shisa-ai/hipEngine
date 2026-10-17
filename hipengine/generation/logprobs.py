"""Byte-aware logprob text alignment; probabilities always belong to whole tokens."""
from __future__ import annotations

import codecs
from collections.abc import Iterator, Sequence
from typing import Any

from hipengine.generation.registry import TokenLogprob


def token_bytes_for_id(tokenizer: Any, token_id: int) -> tuple[int, ...] | None:
    decode = getattr(tokenizer, "decode_token_bytes", None)
    return tuple(decode(int(token_id))) if callable(decode) else None


def _bytes(token: TokenLogprob) -> bytes:
    return bytes(token.token_bytes) if token.token_bytes is not None else token.token_text.encode("utf-8")


def logprob_text(tokens: Sequence[TokenLogprob]) -> str:
    return b"".join(_bytes(token) for token in tokens).decode("utf-8", errors="replace")


def logprob_text_spans(tokens: Sequence[TokenLogprob]) -> Iterator[tuple[int, int]]:
    """Yield character spans, allowing adjacent tokens to share a codepoint.

    A pending UTF-8 prefix occupies a character even before its final byte
    arrives. This prevents attaching a whole-token probability to a visible
    substring that ends inside that token. Final incomplete UTF-8 is a single
    replacement character, as in the GGUF decoder.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    cursor = 0
    for token in tokens:
        data = _bytes(token)
        start = cursor
        pending, _ = decoder.getstate()
        if pending and data:
            try:
                (pending + data[:1]).decode("utf-8")
            except UnicodeDecodeError as error:
                if error.end <= len(pending):
                    # This byte invalidates the previous prefix rather than
                    # sharing its codepoint. That prefix becomes one U+FFFD.
                    start += 1
        cursor += len(decoder.decode(data))
        pending, _ = decoder.getstate()
        yield start, cursor + bool(pending)
