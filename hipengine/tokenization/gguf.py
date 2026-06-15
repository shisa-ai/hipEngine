"""Torch-free tokenizer helpers for GGUF metadata."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from hipengine.loading.gguf import GGUFModelInfo

# Torch-free Qwen3.5 byte-level pre-tokenizer.  This mirrors llama.cpp's
# LLAMA_VOCAB_PRE_TYPE_QWEN35 splitter for Python str code points; bytes are
# still mapped through the GPT-2 byte encoder before BPE merges.


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte->unicode map used by GPT-2 byte BPE."""

    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(value) for value in cs), strict=True))


@dataclass
class Qwen35GGUFTokenizer:
    """Minimal torch-free byte-BPE tokenizer loaded from Qwen3.5 GGUF metadata."""

    tokens: Sequence[str]
    merges: Sequence[str]
    token_types: Sequence[int]
    eos_token_id: int | None = None
    padding_token_id: int | None = None
    token_to_id: dict[str, int] = field(init=False)
    merge_ranks: dict[tuple[str, str], int] = field(init=False)
    byte_encoder: dict[int, str] = field(default_factory=bytes_to_unicode, init=False)
    byte_decoder: dict[str, int] = field(init=False)
    _cache: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.token_types):
            raise ValueError("token and token_type arrays must have the same length")
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.merge_ranks = {}
        for rank, merge in enumerate(self.merges):
            left, sep, right = merge.partition(" ")
            if not sep or not left or not right:
                continue
            self.merge_ranks[(left, right)] = rank
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}

    @classmethod
    def from_gguf_info(cls, info: GGUFModelInfo) -> "Qwen35GGUFTokenizer":
        metadata = info.metadata
        model = metadata.get("tokenizer.ggml.model")
        pre = metadata.get("tokenizer.ggml.pre")
        if model != "gpt2" or pre != "qwen35":
            raise ValueError(f"unsupported GGUF tokenizer model/pre pair: {model!r}/{pre!r}")
        return cls(
            tokens=tuple(str(token) for token in metadata["tokenizer.ggml.tokens"]),
            merges=tuple(str(merge) for merge in metadata["tokenizer.ggml.merges"]),
            token_types=tuple(int(kind) for kind in metadata["tokenizer.ggml.token_type"]),
            eos_token_id=_optional_int(metadata.get("tokenizer.ggml.eos_token_id")),
            padding_token_id=_optional_int(metadata.get("tokenizer.ggml.padding_token_id")),
        )

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _pretokenize_qwen35(text):
            encoded = "".join(self.byte_encoder[byte] for byte in chunk.encode("utf-8"))
            for piece in self._bpe(encoded):
                try:
                    ids.append(self.token_to_id[piece])
                except KeyError as exc:
                    raise ValueError(f"BPE piece {piece!r} is missing from GGUF vocabulary") from exc
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = False) -> str:
        pieces: list[str] = []
        literal: list[str] = []
        for token_id in token_ids:
            idx = int(token_id)
            if idx < 0 or idx >= len(self.tokens):
                raise ValueError(f"token id {idx} is outside vocabulary size {len(self.tokens)}")
            token = self.tokens[idx]
            if skip_special and self.token_types[idx] != 1:
                continue
            if all(char in self.byte_decoder for char in token):
                pieces.append(token)
            else:
                if pieces:
                    literal.append(_decode_byte_pieces(pieces, self.byte_decoder))
                    pieces.clear()
                if not skip_special:
                    literal.append(token)
        if pieces:
            literal.append(_decode_byte_pieces(pieces, self.byte_decoder))
        return "".join(literal)

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        if not token:
            return ()
        word = tuple(token)
        if len(word) == 1:
            self._cache[token] = word
            return word
        while True:
            pairs = _pairs(word)
            ranked = [pair for pair in pairs if pair in self.merge_ranks]
            if not ranked:
                break
            bigram = min(ranked, key=self.merge_ranks.__getitem__)
            word = _merge_pair(word, bigram)
            if len(word) == 1:
                break
        self._cache[token] = word
        return word


def _pretokenize_qwen35(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    end = len(text)

    def char(pos: int) -> str | None:
        return text[pos] if 0 <= pos < end else None

    pos = 0
    while pos < end:
        token_start = pos
        current = char(pos)
        assert current is not None

        # regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if current == "'" and pos + 1 < end:
            tail2 = text[pos + 1 : pos + 3].lower()
            if text[pos + 1].lower() in {"s", "t", "m", "d"}:
                pos += 2
                chunks.append(text[token_start:pos])
                start = pos
                continue
            if tail2 in {"re", "ve", "ll"}:
                pos += 3
                chunks.append(text[token_start:pos])
                start = pos
                continue

        # regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if current not in {"\r", "\n"} and not _is_number(current):
            next_char = char(pos + 1)
            if (
                _is_letter(current)
                or _is_mark(current)
                or _is_mark(next_char)
                or _is_letter(next_char)
            ):
                pos += 1
                while _is_letter(char(pos)) or _is_mark(char(pos)):
                    pos += 1
                chunks.append(text[token_start:pos])
                start = pos
                continue

        # regex: \p{N}
        if _is_number(current):
            pos += 1
            chunks.append(text[token_start:pos])
            start = pos
            continue

        # regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        flags_char = char(pos + 1) if current == " " else current
        if (
            flags_char is not None
            and not _is_whitespace(flags_char)
            and not _is_letter(flags_char)
            and not _is_mark(flags_char)
            and not _is_number(flags_char)
        ):
            if current == " ":
                pos += 1
            while True:
                flags_char = char(pos)
                if (
                    flags_char is None
                    or _is_whitespace(flags_char)
                    or _is_letter(flags_char)
                    or _is_mark(flags_char)
                    or _is_number(flags_char)
                ):
                    break
                pos += 1
            while char(pos) in {"\r", "\n"}:
                pos += 1
            chunks.append(text[token_start:pos])
            start = pos
            continue

        num_whitespaces = 0
        last_end_r_or_n = 0
        while _is_whitespace(char(pos + num_whitespaces)):
            cpt = char(pos + num_whitespaces)
            if cpt in {"\r", "\n"}:
                last_end_r_or_n = pos + num_whitespaces + 1
            num_whitespaces += 1

        # regex: \s*[\r\n]+
        if last_end_r_or_n > 0:
            pos = last_end_r_or_n
            chunks.append(text[token_start:pos])
            start = pos
            continue

        # regex: \s+(?!\S)
        if num_whitespaces > 1 and char(pos + num_whitespaces) is not None:
            pos += num_whitespaces - 1
            chunks.append(text[token_start:pos])
            start = pos
            continue

        # regex: \s+
        if num_whitespaces > 0:
            pos += num_whitespaces
            chunks.append(text[token_start:pos])
            start = pos
            continue

        pos += 1
        chunks.append(text[token_start:pos])
        start = pos

    assert start == end
    return chunks


def _category(char: str | None) -> str:
    return "" if char is None else unicodedata.category(char)


def _is_letter(char: str | None) -> bool:
    return _category(char).startswith("L")


def _is_mark(char: str | None) -> bool:
    return _category(char).startswith("M")


def _is_number(char: str | None) -> bool:
    return _category(char).startswith("N")


def _is_whitespace(char: str | None) -> bool:
    return False if char is None else char.isspace()


def _decode_byte_pieces(pieces: Sequence[str], byte_decoder: Mapping[str, int]) -> str:
    data = bytearray()
    for piece in pieces:
        for char in piece:
            data.append(byte_decoder[char])
    return data.decode("utf-8", errors="replace")


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:], strict=False))


def _merge_pair(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    first, second = pair
    out: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            out.append(first + second)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


__all__ = ["Qwen35GGUFTokenizer", "bytes_to_unicode"]
