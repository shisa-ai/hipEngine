"""Model-bound selected-vocabulary maps for GGUF MTP proposal heads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.loading.gguf import GGUFModelInfo

HOT_VOCAB_SCHEMA_VERSION = 1
HOT_VOCAB_KIND = "hipengine.gguf_mtp_hot_vocab"


@dataclass(frozen=True)
class GGUFHotVocabSelection:
    """Validated compact-to-full token map for one GGUF tokenizer."""

    token_ids: tuple[int, ...]
    tokenizer_tokens_sha256: str
    source_path: Path
    metadata: Mapping[str, Any]

    @property
    def size(self) -> int:
        return len(self.token_ids)


def gguf_tokenizer_tokens_sha256(info: GGUFModelInfo) -> str:
    """Hash the ordered tokenizer vocabulary without ambiguous separators."""

    tokens = info.metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
        raise ValueError("GGUF metadata does not contain tokenizer.ggml.tokens")
    digest = hashlib.sha256()
    for token in tokens:
        encoded = str(token).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def load_gguf_hot_vocab_selection(
    path: str | Path,
    info: GGUFModelInfo,
) -> GGUFHotVocabSelection:
    """Load and validate a model-bound selected-vocabulary JSON artifact."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GGUF MTP hot-vocabulary artifact must be a JSON object")
    if int(payload.get("schema_version", -1)) != HOT_VOCAB_SCHEMA_VERSION:
        raise ValueError("unsupported GGUF MTP hot-vocabulary schema version")
    if payload.get("kind") != HOT_VOCAB_KIND:
        raise ValueError("invalid GGUF MTP hot-vocabulary artifact kind")

    vocab_size = len(info.metadata.get("tokenizer.ggml.tokens", ()))
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("GGUF MTP hot-vocabulary artifact is missing model metadata")
    if int(model.get("vocab_size", -1)) != vocab_size:
        raise ValueError("GGUF MTP hot-vocabulary vocab size does not match the model")
    expected_hash = gguf_tokenizer_tokens_sha256(info)
    if str(model.get("tokenizer_tokens_sha256", "")) != expected_hash:
        raise ValueError("GGUF MTP hot-vocabulary tokenizer hash does not match the model")

    raw_ids = payload.get("token_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("GGUF MTP hot-vocabulary token_ids must be a nonempty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids):
        raise ValueError("GGUF MTP hot-vocabulary token IDs must be integers")
    token_ids = tuple(int(value) for value in raw_ids)
    if len(token_ids) % 16:
        raise ValueError("GGUF MTP hot-vocabulary size must be divisible by 16")
    if tuple(sorted(set(token_ids))) != token_ids:
        raise ValueError("GGUF MTP hot-vocabulary token IDs must be sorted and unique")
    if token_ids[0] < 0 or token_ids[-1] >= vocab_size:
        raise ValueError("GGUF MTP hot-vocabulary token ID is outside the model vocabulary")

    metadata = payload.get("selection")
    return GGUFHotVocabSelection(
        token_ids=token_ids,
        tokenizer_tokens_sha256=expected_hash,
        source_path=source_path,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


__all__ = [
    "GGUFHotVocabSelection",
    "HOT_VOCAB_KIND",
    "HOT_VOCAB_SCHEMA_VERSION",
    "gguf_tokenizer_tokens_sha256",
    "load_gguf_hot_vocab_selection",
]
