"""Tokenizer-bound selected-vocabulary maps for GGUF MTP proposal heads."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.loading.gguf import GGUFModelInfo

HOT_VOCAB_SCHEMA_VERSION = 1
HOT_VOCAB_KIND = "hipengine.gguf_mtp_hot_vocab"
_DEFAULT_HOT_VOCAB_IDENTITIES = {
    "b7f4906b5bf6a845baf3f41fdcdcd70f0f2f234eb702aabb830ce1536604e5d6":
        "qwen38-27b-hot131072-cjk-v1.json",
}


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


def default_gguf_hot_vocab_path(
    info: GGUFModelInfo,
    artifact_preset_key: str | None = None,
) -> Path | None:
    """Resolve a packaged token map for the model's ordered vocabulary.

    Token IDs are meaningful only for the same tokenizer vocabulary. Model
    names, layer counts, weight quantization and admission presets do not
    change that compatibility. ``artifact_preset_key`` remains accepted for
    callers using the previous interface but does not restrict selection.
    """

    key = gguf_tokenizer_tokens_sha256(info)
    filename = _DEFAULT_HOT_VOCAB_IDENTITIES.get(key)
    if filename is None:
        return None
    path = Path(__file__).with_name("mtp_hot_vocab_data") / filename
    if not path.is_file():
        raise FileNotFoundError(f"packaged GGUF MTP hot-vocabulary map is missing: {path}")
    return path


def load_gguf_hot_vocab_selection(
    path: str | Path,
    info: GGUFModelInfo,
) -> GGUFHotVocabSelection:
    """Load a selected-vocabulary map bound to compatible tokenizer IDs."""

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
    bitmap_encoded = payload.get("token_bitmap_base64")
    if (raw_ids is None) == (bitmap_encoded is None):
        raise ValueError(
            "GGUF MTP hot-vocabulary artifact requires exactly one token encoding"
        )
    if raw_ids is not None:
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError("GGUF MTP hot-vocabulary token_ids must be a nonempty list")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids):
            raise ValueError("GGUF MTP hot-vocabulary token IDs must be integers")
        token_ids = tuple(int(value) for value in raw_ids)
    else:
        if not isinstance(bitmap_encoded, str) or not bitmap_encoded:
            raise ValueError("GGUF MTP hot-vocabulary token bitmap must be nonempty")
        try:
            bitmap = base64.b64decode(bitmap_encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("GGUF MTP hot-vocabulary token bitmap is invalid") from exc
        expected_nbytes = (vocab_size + 7) // 8
        if len(bitmap) != expected_nbytes:
            raise ValueError("GGUF MTP hot-vocabulary token bitmap has the wrong size")
        if vocab_size % 8 and bitmap[-1] >> (vocab_size % 8):
            raise ValueError("GGUF MTP hot-vocabulary token bitmap sets padding bits")
        token_ids = tuple(
            token_id
            for token_id in range(vocab_size)
            if bitmap[token_id // 8] & (1 << (token_id % 8))
        )
        declared_size = payload.get("selection", {}).get("selected_tokens")
        if declared_size is not None and int(declared_size) != len(token_ids):
            raise ValueError("GGUF MTP hot-vocabulary token bitmap count is inconsistent")
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
    "default_gguf_hot_vocab_path",
    "gguf_tokenizer_tokens_sha256",
    "load_gguf_hot_vocab_selection",
]
