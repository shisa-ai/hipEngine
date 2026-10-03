"""Checkpoint-native YuE2 text/ABC BPE (``qwen.tiktoken``), torch-free.

The pinned release ships a tiktoken-format merge file (151643 ordinary ranks) and
a fixed pre-tokenization regex. ``tiktoken`` itself is not a hipEngine dependency,
so this module reproduces the same split with ``tokenizers`` (already a hard
dependency, and the same Rust ``regex`` engine the reference package uses) and
applies a pure-Python byte-pair merge over each piece.

``encode`` is deliberately ``encode_ordinary`` semantics: the special tokens are
registered for decoding and identity, but ordinary text never emits them. Special
IDs above the ordinary vocabulary are appended by the request protocol, not by the
tokenizer.

This is a text/ABC BPE. It is not the semantic audio tokenizer.
"""

from __future__ import annotations

import base64
import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Regex
from tokenizers.pre_tokenizers import Split

ORDINARY_VOCAB = 151643
SPECIAL_NAMES = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<R>",
    "<S>",
    "<X>",
    "<mask>",
    "<sep>",
)
SPECIAL_NAMES = SPECIAL_NAMES + tuple(f"<extra_{index}>" for index in range(200))
SPECIAL_NAMES = SPECIAL_NAMES[:204] + ("<abc>", "</abc>") + SPECIAL_NAMES[206:]

#: The released pre-tokenization pattern (verbatim).
PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


@dataclass(frozen=True)
class TokenizerIdentity:
    path: str
    sha256: str
    ranks: int
    specials: int

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "ordinary_ranks": self.ranks,
            "special_tokens": self.specials,
            "pattern": PATTERN,
        }


class YuE2TextTokenizer:
    """Frozen text/ABC BPE with ``encode_ordinary`` semantics."""

    def __init__(self, merge_file: str | Path):
        self.merge_file = Path(merge_file)
        if not self.merge_file.is_file():
            raise FileNotFoundError(f"qwen.tiktoken not found: {self.merge_file}")
        raw = self.merge_file.read_bytes()
        ranks: dict[bytes, int] = {}
        for line in raw.splitlines():
            if not line:
                continue
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
        if len(ranks) != ORDINARY_VOCAB:
            raise ValueError(
                f"Expected checkpoint-native qwen.tiktoken ({ORDINARY_VOCAB} ordinary tokens), "
                f"found {len(ranks)}"
            )
        self._ranks = ranks
        self._single = {token: rank for token, rank in ranks.items() if len(token) == 1}
        self.specials = {name: ORDINARY_VOCAB + index for index, name in enumerate(SPECIAL_NAMES)}
        self.n_vocab = ORDINARY_VOCAB + len(SPECIAL_NAMES)
        self._splitter = Split(Regex(PATTERN), behavior="isolated")
        self.identity = TokenizerIdentity(
            path=str(self.merge_file),
            sha256=hashlib.sha256(raw).hexdigest(),
            ranks=len(ranks),
            specials=len(SPECIAL_NAMES),
        )

    # -- encoding ---------------------------------------------------------
    def _byte_pair(self, piece: bytes) -> list[int]:
        ranks = self._ranks
        if len(piece) == 1:
            return [ranks[piece]]
        parts = [piece[index : index + 1] for index in range(len(piece))]
        while len(parts) > 1:
            best_rank = None
            best_index = None
            for index in range(len(parts) - 1):
                rank = ranks.get(parts[index] + parts[index + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, index
            if best_index is None:
                break
            parts[best_index : best_index + 2] = [parts[best_index] + parts[best_index + 1]]
        return [ranks[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        """``encode_ordinary`` over NFC-normalized text (never emits specials)."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize("NFC", text)
        tokens: list[int] = []
        for piece, _ in self._splitter.pre_tokenize_str(normalized):
            tokens.extend(self._byte_pair(piece.encode("utf-8")))
        return tokens

    def decode(self, ids) -> str:
        pieces = []
        reverse = {rank: token for token, rank in self._ranks.items()}
        for value in ids:
            token = int(value)
            if 0 <= token < ORDINARY_VOCAB:
                pieces.append(reverse.get(token, b""))
            elif token in range(ORDINARY_VOCAB, self.n_vocab):
                pieces.append(SPECIAL_NAMES[token - ORDINARY_VOCAB].encode("utf-8"))
        return b"".join(pieces).decode("utf-8", errors="replace")

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "YuE2TextTokenizer":
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"model directory not found: {directory}")
        return cls(directory / "qwen.tiktoken")
