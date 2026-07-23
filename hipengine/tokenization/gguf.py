"""Torch-free HF tokenizer wrappers reconstructed from GGUF metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers
from tokenizers import pre_tokenizers as hf_pre_tokenizers

from hipengine.loading.gguf import GGUFModelInfo

_QWEN35_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)
_LAGUNA_NEWLINE_SPLIT = r"(?:\r?\n)+(?!\r?\n)"
_LAGUNA_QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)
_ADDED_TOKEN_TYPES = frozenset({3, 4})
_BASE_TOKEN_TYPES = frozenset({1, 2, 6})


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte->unicode map used by GPT-2 byte BPE."""

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = byte_values[:]
    next_codepoint = 256
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(next_codepoint)
            next_codepoint += 1
    return dict(zip(byte_values, (chr(value) for value in codepoints), strict=True))


@dataclass
class Qwen35GGUFTokenizer:
    """HF byte-BPE tokenizer reconstructed entirely from Qwen GGUF metadata."""

    encoder_backend: ClassVar[str] = "huggingface_tokenizers"
    _encoder_recipe: ClassVar[str] = "qwen35"

    tokens: Sequence[str]
    merges: Sequence[str]
    token_types: Sequence[int]
    eos_token_id: int | None = None
    padding_token_id: int | None = None
    token_to_id: dict[str, int] = field(init=False)
    byte_encoder: dict[int, str] = field(default_factory=bytes_to_unicode, init=False)
    byte_decoder: dict[str, int] = field(init=False)
    encoder: Tokenizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.token_types):
            raise ValueError("token and token_type arrays must have the same length")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("GGUF tokenizer vocabulary contains duplicate token strings")
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.encoder = _build_hf_encoder(
            self.tokens,
            self.merges,
            self.token_types,
            recipe=self._encoder_recipe,
        )

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
        return [
            int(token_id)
            for token_id in self.encoder.encode(
                str(text),
                add_special_tokens=False,
            ).ids
        ]

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = False) -> str:
        pieces: list[str] = []
        literal: list[str] = []
        for token_id in token_ids:
            idx = int(token_id)
            if idx < 0 or idx >= len(self.tokens):
                raise ValueError(
                    f"token id {idx} is outside vocabulary size {len(self.tokens)}"
                )
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


@dataclass
class LagunaGGUFTokenizer(Qwen35GGUFTokenizer):
    """HF byte-BPE tokenizer reconstructed entirely from Laguna GGUF metadata."""

    _encoder_recipe: ClassVar[str] = "laguna"

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
        ids = super().encode(text)
        if add_special_tokens and self.add_bos_token:
            if self.bos_token_id is None:
                raise ValueError(
                    "Laguna tokenizer requests BOS insertion but has no BOS token ID"
                )
            ids.insert(0, self.bos_token_id)
        return ids


def _build_hf_encoder(
    tokens: Sequence[str],
    merges: Sequence[str],
    token_types: Sequence[int],
    *,
    recipe: str,
) -> Tokenizer:
    token_values = tuple(str(token) for token in tokens)
    type_values = tuple(int(kind) for kind in token_types)
    if recipe == "qwen35":
        vocabulary = {
            token: token_id
            for token_id, (token, kind) in enumerate(
                zip(token_values, type_values, strict=True)
            )
            if kind in _BASE_TOKEN_TYPES
        }
        expected_ids = tuple(range(len(vocabulary)))
        if tuple(vocabulary.values()) != expected_ids:
            raise ValueError(
                "Qwen GGUF base vocabulary must precede added and unused tokens"
            )
    elif recipe == "laguna":
        # Laguna's source tokenizer keeps its low-ID added tokens in the BPE
        # vocabulary and overlays AddedToken matching semantics on those IDs.
        vocabulary = {token: token_id for token_id, token in enumerate(token_values)}
    else:  # pragma: no cover - guarded by the concrete tokenizer classes
        raise ValueError(f"unsupported HF GGUF tokenizer recipe: {recipe!r}")

    encoder = Tokenizer(
        models.BPE(
            vocab=vocabulary,
            merges=_parse_merges(merges),
            fuse_unk=False,
            byte_fallback=False,
        )
    )
    if recipe == "qwen35":
        encoder.normalizer = normalizers.NFC()
        encoder.pre_tokenizer = hf_pre_tokenizers.Sequence(
            [
                hf_pre_tokenizers.Split(
                    Regex(_QWEN35_SPLIT),
                    behavior="isolated",
                    invert=False,
                ),
                hf_pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=False,
                    use_regex=False,
                ),
            ]
        )
        encoder.decoder = decoders.ByteLevel(
            add_prefix_space=False,
            trim_offsets=False,
            use_regex=False,
        )
    else:
        encoder.pre_tokenizer = hf_pre_tokenizers.Sequence(
            [
                hf_pre_tokenizers.Split(
                    Regex(_LAGUNA_NEWLINE_SPLIT),
                    behavior="merged_with_next",
                    invert=False,
                ),
                hf_pre_tokenizers.Split(
                    Regex(_LAGUNA_QWEN2_SPLIT),
                    behavior="isolated",
                    invert=False,
                ),
                hf_pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=True,
                    use_regex=False,
                ),
            ]
        )
        encoder.decoder = decoders.ByteLevel(
            add_prefix_space=True,
            trim_offsets=True,
            use_regex=True,
        )

    for token_id, (token, kind) in enumerate(
        zip(token_values, type_values, strict=True)
    ):
        if kind not in _ADDED_TOKEN_TYPES:
            continue
        added = AddedToken(
            token,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=kind == 3,
        )
        if kind == 3:
            encoder.add_special_tokens([added])
        else:
            encoder.add_tokens([added])
        actual_id = encoder.token_to_id(token)
        if actual_id != token_id:
            raise ValueError(
                "HF reconstruction changed GGUF token ID "
                f"for {token!r}: expected {token_id}, got {actual_id}"
            )
    return encoder


def _parse_merges(merges: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for merge in merges:
        left, separator, right = str(merge).partition(" ")
        if not separator or not left or not right:
            raise ValueError(f"invalid GGUF BPE merge: {merge!r}")
        parsed.append((left, right))
    return parsed


def _decode_byte_pieces(
    pieces: Sequence[str],
    byte_decoder: Mapping[str, int],
) -> str:
    data = bytearray()
    for piece in pieces:
        for char in piece:
            data.append(byte_decoder[char])
    return data.decode("utf-8", errors="replace")


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


__all__ = ["LagunaGGUFTokenizer", "Qwen35GGUFTokenizer", "bytes_to_unicode"]
