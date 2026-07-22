"""Torch-free tokenizer helpers for GGUF metadata."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
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
    atomic_tokens_by_prefix: dict[str, tuple[tuple[str, int], ...]] = field(
        init=False, repr=False
    )
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
        atomic: dict[str, list[tuple[str, int]]] = {}
        for token_id, (token, kind) in enumerate(zip(self.tokens, self.token_types, strict=True)):
            if token and int(kind) in {3, 4}:
                atomic.setdefault(token[0], []).append((token, token_id))
        self.atomic_tokens_by_prefix = {
            prefix: tuple(sorted(items, key=lambda item: (-len(item[0]), item[1])))
            for prefix, items in atomic.items()
        }

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
        return self._encode_text(text, _pretokenize_qwen35)

    def _encode_text(
        self,
        text: str,
        pretokenize: Callable[[str], list[str]],
    ) -> list[int]:
        ids: list[int] = []
        segment_start = 0
        pos = 0
        while pos < len(text):
            match = next(
                (
                    item
                    for item in self.atomic_tokens_by_prefix.get(text[pos], ())
                    if text.startswith(item[0], pos)
                ),
                None,
            )
            if match is None:
                pos += 1
                continue
            if segment_start < pos:
                ids.extend(self._encode_chunks(pretokenize(text[segment_start:pos])))
            token, token_id = match
            ids.append(token_id)
            pos += len(token)
            segment_start = pos
        if segment_start < len(text):
            ids.extend(self._encode_chunks(pretokenize(text[segment_start:])))
        return ids

    def _encode_chunks(self, chunks: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for chunk in chunks:
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
            if skip_special and self.token_types[idx] == 3:
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


@dataclass
class LagunaGGUFTokenizer(Qwen35GGUFTokenizer):
    """Torch-free Laguna byte-BPE tokenizer loaded from GGUF metadata."""

    bos_token_id: int | None = None
    eot_token_id: int | None = None
    separator_token_id: int | None = None
    mask_token_id: int | None = None
    unknown_token_id: int | None = None
    add_bos_token: bool = False
    chat_template: str = ""

    @classmethod
    def from_gguf_info(cls, info: GGUFModelInfo) -> "LagunaGGUFTokenizer":
        metadata = info.metadata
        model = metadata.get("tokenizer.ggml.model")
        pre = metadata.get("tokenizer.ggml.pre")
        if model != "gpt2" or pre != "laguna":
            raise ValueError(
                "Laguna GGUF tokenizer expected 'gpt2'/'laguna', "
                f"got {model!r}/{pre!r}"
            )
        return cls(
            tokens=tuple(str(token) for token in metadata["tokenizer.ggml.tokens"]),
            merges=tuple(str(merge) for merge in metadata["tokenizer.ggml.merges"]),
            token_types=tuple(int(kind) for kind in metadata["tokenizer.ggml.token_type"]),
            bos_token_id=_optional_int(metadata.get("tokenizer.ggml.bos_token_id")),
            eos_token_id=_optional_int(metadata.get("tokenizer.ggml.eos_token_id")),
            eot_token_id=_optional_int(metadata.get("tokenizer.ggml.eot_token_id")),
            padding_token_id=_optional_int(metadata.get("tokenizer.ggml.padding_token_id")),
            separator_token_id=_optional_int(
                metadata.get("tokenizer.ggml.seperator_token_id")
            ),
            mask_token_id=_optional_int(metadata.get("tokenizer.ggml.mask_token_id")),
            unknown_token_id=_optional_int(metadata.get("tokenizer.ggml.unknown_token_id")),
            add_bos_token=bool(metadata.get("tokenizer.ggml.add_bos_token", False)),
            chat_template=str(metadata.get("tokenizer.chat_template", "")),
        )

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        values = (self.eos_token_id, self.eot_token_id)
        return tuple(dict.fromkeys(int(value) for value in values if value is not None))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = self._encode_text(text, _pretokenize_laguna)
        if add_special_tokens and self.add_bos_token:
            if self.bos_token_id is None:
                raise ValueError("Laguna tokenizer requests BOS insertion but has no BOS token ID")
            ids.insert(0, self.bos_token_id)
        return ids


def _pretokenize_qwen35(text: str) -> list[str]:
    return _pretokenize_qwen_family(text, include_marks=True)


def _pretokenize_laguna(text: str) -> list[str]:
    """Mirror Poolside llama.cpp's newline split followed by Qwen2-style split."""

    chunks: list[str] = []
    for run in _split_newline_runs(text):
        chunks.extend(_pretokenize_qwen_family(run, include_marks=False))
    return chunks


def _split_newline_runs(text: str) -> list[str]:
    """Split around ``(?:\r?\n)+`` while retaining CRLF as one run."""

    runs: list[str] = []
    segment_start = 0
    pos = 0
    while pos < len(text):
        newline_start = pos
        if text[pos] == "\r" and pos + 1 < len(text) and text[pos + 1] == "\n":
            pos += 2
        elif text[pos] == "\n":
            pos += 1
        else:
            pos += 1
            continue
        while pos < len(text):
            if text[pos] == "\r" and pos + 1 < len(text) and text[pos + 1] == "\n":
                pos += 2
            elif text[pos] == "\n":
                pos += 1
            else:
                break
        if newline_start > segment_start:
            runs.append(text[segment_start:newline_start])
        runs.append(text[newline_start:pos])
        segment_start = pos
    if segment_start < len(text):
        runs.append(text[segment_start:])
    return runs


def _pretokenize_qwen_family(text: str, *, include_marks: bool) -> list[str]:
    chunks: list[str] = []
    end = len(text)

    def char(pos: int) -> str | None:
        return text[pos] if 0 <= pos < end else None

    def word_char(value: str | None) -> bool:
        return _is_letter(value) or (include_marks and _is_mark(value))

    def punctuation_boundary(value: str | None) -> bool:
        return (
            value is None
            or _is_whitespace(value)
            or _is_letter(value)
            or _is_number(value)
            or (include_marks and _is_mark(value))
        )

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
                continue
            if tail2 in {"re", "ve", "ll"}:
                pos += 3
                chunks.append(text[token_start:pos])
                continue

        # Qwen35 consumes marks in the letter run; Laguna uses plain \p{L}+.
        if current not in {"\r", "\n"} and not _is_number(current):
            if word_char(current) or word_char(char(pos + 1)):
                pos += 1
                while word_char(char(pos)):
                    pos += 1
                chunks.append(text[token_start:pos])
                continue

        # regex: \p{N}
        if _is_number(current):
            pos += 1
            chunks.append(text[token_start:pos])
            continue

        # Punctuation branch; Qwen35 excludes marks, while Laguna does not.
        flags_char = char(pos + 1) if current == " " else current
        if not punctuation_boundary(flags_char):
            if current == " ":
                pos += 1
            while not punctuation_boundary(char(pos)):
                pos += 1
            while char(pos) in {"\r", "\n"}:
                pos += 1
            chunks.append(text[token_start:pos])
            continue

        num_whitespaces = 0
        last_end_r_or_n = 0
        while _is_whitespace(char(pos + num_whitespaces)):
            cpt = char(pos + num_whitespaces)
            if cpt in {"\r", "\n"}:
                last_end_r_or_n = pos + num_whitespaces + 1
            num_whitespaces += 1

        if last_end_r_or_n > 0:
            pos = last_end_r_or_n
            chunks.append(text[token_start:pos])
            continue
        if num_whitespaces > 1 and char(pos + num_whitespaces) is not None:
            pos += num_whitespaces - 1
            chunks.append(text[token_start:pos])
            continue
        if num_whitespaces > 0:
            pos += num_whitespaces
            chunks.append(text[token_start:pos])
            continue

        pos += 1
        chunks.append(text[token_start:pos])

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


__all__ = ["LagunaGGUFTokenizer", "Qwen35GGUFTokenizer", "bytes_to_unicode"]
