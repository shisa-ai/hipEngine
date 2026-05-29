"""Torch-free tokenizer helpers for GGUF metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from jinja2 import Environment

from hipengine.loading.gguf import GGUFModelInfo, GGUFSplitModelInfo

# GPT-2/Qwen byte-level pre-tokenizer approximation.  It intentionally avoids
# optional dependencies while matching the ASCII fixture and common text paths.
_PRETOKEN_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


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
    return [match.group(0) for match in _PRETOKEN_RE.finditer(text)]


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


@dataclass
class StepFunGGUFTokenizer:
    """Torch-free StepFun DeepSeek-V3 byte-BPE tokenizer from GGUF metadata."""

    tokens: Sequence[str]
    merges: Sequence[str]
    token_types: Sequence[int]
    bos_token_id: int | None = None
    eos_token_ids: tuple[int, ...] = ()
    padding_token_id: int | None = None
    add_bos_token: bool = True
    chat_template: str | None = None
    token_to_id: dict[str, int] = field(init=False)
    _tokenizer: Any = field(init=False, repr=False)
    _jinja: Environment = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.token_types):
            raise ValueError("token and token_type arrays must have the same length")
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self._tokenizer = _build_tokenizers_bpe(self.tokens, self.merges, self.token_types)
        self._jinja = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
        self._jinja.filters["fromjson"] = json.loads

    @classmethod
    def from_gguf_info(cls, info: GGUFModelInfo | GGUFSplitModelInfo) -> "StepFunGGUFTokenizer":
        metadata = info.metadata
        model = metadata.get("tokenizer.ggml.model")
        pre = metadata.get("tokenizer.ggml.pre")
        if model != "gpt2" or pre != "deepseek-v3":
            raise ValueError(
                f"unsupported StepFun GGUF tokenizer model/pre pair: {model!r}/{pre!r}"
            )
        eos_ids = tuple(
            dict.fromkeys((1, 2, int(metadata["tokenizer.ggml.eos_token_id"]))).keys()
        )
        return cls(
            tokens=tuple(str(token) for token in metadata["tokenizer.ggml.tokens"]),
            merges=tuple(str(merge) for merge in metadata["tokenizer.ggml.merges"]),
            token_types=tuple(int(kind) for kind in metadata["tokenizer.ggml.token_type"]),
            bos_token_id=_optional_int(metadata.get("tokenizer.ggml.bos_token_id")),
            eos_token_ids=eos_ids,
            padding_token_id=_optional_int(metadata.get("tokenizer.ggml.padding_token_id")),
            add_bos_token=bool(metadata.get("tokenizer.ggml.add_bos_token", True)),
            chat_template=_optional_str(metadata.get("tokenizer.chat_template")),
        )

    @property
    def bos_token(self) -> str | None:
        if self.bos_token_id is None:
            return None
        return self.tokens[self.bos_token_id]

    def encode(self, text: str, *, add_bos: bool | None = None) -> list[int]:
        ids = list(self._tokenizer.encode(str(text)).ids)
        should_add_bos = self.add_bos_token if add_bos is None else bool(add_bos)
        if should_add_bos and self.bos_token_id is not None:
            if not ids or ids[0] != self.bos_token_id:
                ids.insert(0, self.bos_token_id)
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = True) -> str:
        ids = [int(token_id) for token_id in token_ids]
        return str(self._tokenizer.decode(ids, skip_special_tokens=skip_special))

    def is_eos(self, token_id: int) -> bool:
        return int(token_id) in self.eos_token_ids

    def render_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        reasoning_effort: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        if not self.chat_template:
            raise ValueError("StepFun GGUF metadata does not include tokenizer.chat_template")
        if not messages:
            raise ValueError("messages must be non-empty")
        template = self._jinja.from_string(self.chat_template)
        context: dict[str, Any] = {
            "messages": list(messages),
            "add_generation_prompt": bool(add_generation_prompt),
            "bos_token": self.bos_token or "",
            "tools": None if tools is None else list(tools),
        }
        if reasoning_effort is not None:
            context["reasoning_effort"] = str(reasoning_effort)
        return str(template.render(**context))

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        reasoning_effort: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[int]:
        return self.encode(
            self.render_chat(
                messages,
                add_generation_prompt=add_generation_prompt,
                reasoning_effort=reasoning_effort,
                tools=tools,
            ),
            add_bos=False,
        )


def _build_tokenizers_bpe(
    tokens: Sequence[str], merges: Sequence[str], token_types: Sequence[int]
) -> Any:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    except Exception as exc:  # pragma: no cover - dependency is declared in pyproject
        raise RuntimeError("tokenizers is required for StepFun GGUF tokenization") from exc

    vocab = {str(token): idx for idx, token in enumerate(tokens)}
    merge_pairs: list[tuple[str, str]] = []
    for merge in merges:
        left, sep, right = str(merge).partition(" ")
        if sep and left and right:
            merge_pairs.append((left, right))
    tokenizer = Tokenizer(
        models.BPE(vocab=vocab, merges=merge_pairs, unk_token=None, fuse_unk=False)
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = [
        str(token) for token, kind in zip(tokens, token_types, strict=True) if int(kind) != 1
    ]
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _optional_str(value) -> str | None:
    return None if value is None else str(value)


__all__ = ["Qwen35GGUFTokenizer", "StepFunGGUFTokenizer", "bytes_to_unicode"]
