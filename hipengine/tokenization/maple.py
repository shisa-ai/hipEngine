"""Torch-free tokenizer adapter for Maple-Preview's HF tokenizer.json."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class MapleTokenizer:
    encoder: Tokenizer
    vocab_size: int
    eos_token_id: int
    bos_token_id: int

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        *,
        model_vocab_size: int,
        eos_token_id: int,
        bos_token_id: int,
    ) -> MapleTokenizer:
        path = Path(model_path) / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(f"Maple tokenizer.json not found: {path}")
        encoder = Tokenizer.from_file(str(path))
        tokenizer_vocab_size = int(encoder.get_vocab_size(with_added_tokens=True))
        if tokenizer_vocab_size > model_vocab_size:
            raise ValueError(
                f"Maple tokenizer vocab {tokenizer_vocab_size} exceeds model vocab "
                f"{model_vocab_size}"
            )
        if encoder.token_to_id("<|im_end|>") != eos_token_id:
            raise ValueError("Maple tokenizer <|im_end|> ID differs from config.eos_token_id")
        if encoder.token_to_id("<|endoftext|>") != bos_token_id:
            raise ValueError("Maple tokenizer <|endoftext|> ID differs from config.bos_token_id")
        return cls(
            encoder=encoder,
            vocab_size=tokenizer_vocab_size,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(
            int(token_id)
            for token_id in self.encoder.encode(str(text), add_special_tokens=False).ids
        )

    def encode_chat(self, user: str, *, system: str | None = None) -> tuple[int, ...]:
        """Apply the checkpoint's simple no-tools chat template and tokenize it."""

        return self.encode(self.format_chat_prompt(user, system=system))

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = False) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError(f"Maple token IDs must be in [0, {self.vocab_size})")
        return str(self.encoder.decode(ids, skip_special_tokens=bool(skip_special)))

    @staticmethod
    def format_chat_prompt(user: str, *, system: str | None = None) -> str:
        prefix = "" if system is None else f"<|im_start|>system\n{system}<|im_end|>\n"
        return (
            f"{prefix}<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )


__all__ = ["MapleTokenizer"]
