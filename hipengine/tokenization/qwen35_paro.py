"""Qwen3.5/PARO tokenizer with checkpoint-owned special-token metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def normalize_token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError:
            values = (value,)
    normalized: list[int] = []
    for raw_value in values:
        try:
            token_id = int(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if token_id >= 0 and token_id not in normalized:
            normalized.append(token_id)
    return tuple(normalized)


def _metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _token_ids(encoder: Any, value: Any) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    found: list[int] = []
    for token in values:
        if isinstance(token, dict):
            token = token.get("content")
        if isinstance(token, str):
            token_id = encoder.token_to_id(token)
            if token_id is not None and token_id >= 0 and token_id not in found:
                found.append(int(token_id))
    return tuple(found)


@dataclass(frozen=True)
class Qwen35ParoTokenizer:
    encoder: Any
    eos_token_id: tuple[int, ...]
    bos_token_id: int | None
    pad_token_id: int | None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.encoder, name)

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> Qwen35ParoTokenizer:
        from tokenizers import Tokenizer

        model = Path(model_path)
        encoder = Tokenizer.from_file(str(model / "tokenizer.json"))
        generation = _metadata(model / "generation_config.json")
        tokenizer = _metadata(model / "tokenizer_config.json")
        config = _metadata(model / "config.json")
        text_config = config.get("text_config", {})
        sources = (generation, tokenizer, text_config, config)

        def special_ids(role: str) -> tuple[int, ...]:
            for source in sources:
                if not isinstance(source, dict):
                    continue
                ids = normalize_token_ids(source.get(f"{role}_token_id"))
                if not ids:
                    ids = _token_ids(encoder, source.get(f"{role}_token"))
                if ids:
                    return ids
            return ()

        # Checkpoints without sidecar metadata keep the Qwen end-of-turn
        # compatibility rule, resolved once rather than during generation.
        eos = special_ids("eos") or _token_ids(
            encoder, ("<|im_end|>", "<|endoftext|>")
        )
        bos = special_ids("bos")
        pad = special_ids("pad")
        return cls(
            encoder=encoder,
            eos_token_id=eos,
            bos_token_id=bos[0] if bos else None,
            pad_token_id=pad[0] if pad else None,
        )
