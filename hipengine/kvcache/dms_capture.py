"""Bounded, checksummed offline capture artifacts for external DMS training."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

DMS_CAPTURE_INPUT_STAGE = "post_attn_rmsnorm_pre_q_projection"
_DMS_CAPTURE_SCHEMA_VERSION = 1
_HEX_DIGITS = frozenset("0123456789abcdef")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in _HEX_DIGITS for char in text):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return text


def _json_safe_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    payload = dict(value)
    try:
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc
    return payload


class DMSCaptureSink(Protocol):
    """Runtime-facing sink for exact full-attention prefill intermediates."""

    physical_layer_ids: tuple[int, ...]
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    input_stage: str

    def capture_chunk(
        self,
        *,
        physical_layer_id: int,
        compact_layer_index: int,
        positions: np.ndarray,
        hidden_bf16: np.ndarray,
        query_f32: np.ndarray,
        key_f32: np.ndarray,
    ) -> None: ...

    def capture_teacher_logits(self, logits: np.ndarray) -> None: ...


class DMSCaptureWriter:
    """Stream one sequence/layer chunk per NPZ and retain strict checksums."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model_path: str,
        model_sha256: str,
        data_manifest_sha256: str,
        tokenizer_identity: str,
        tokenizer_sha256: str,
        physical_layer_ids: Sequence[int],
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        input_stage: str = DMS_CAPTURE_INPUT_STAGE,
        qk_storage_dtype: str = "float32",
        teacher_topk: int = 64,
        max_shard_bytes: int = 512 * 1024 * 1024,
        capture_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise FileExistsError(f"DMS capture output directory must be empty: {self.output_dir}")
        self.model_path = str(model_path)
        if not self.model_path.strip():
            raise ValueError("DMS capture model_path must be non-empty")
        self.model_sha256 = _validated_sha256(model_sha256, label="model_sha256")
        self.data_manifest_sha256 = _validated_sha256(
            data_manifest_sha256,
            label="data_manifest_sha256",
        )
        self.tokenizer_identity = str(tokenizer_identity)
        if not self.tokenizer_identity.strip():
            raise ValueError("DMS capture tokenizer identity must be non-empty")
        self.tokenizer_sha256 = _validated_sha256(
            tokenizer_sha256,
            label="tokenizer_sha256",
        )
        self.physical_layer_ids = tuple(int(layer) for layer in physical_layer_ids)
        if (
            not self.physical_layer_ids
            or len(set(self.physical_layer_ids)) != len(self.physical_layer_ids)
            or tuple(sorted(self.physical_layer_ids)) != self.physical_layer_ids
            or any(layer < 0 for layer in self.physical_layer_ids)
        ):
            raise ValueError("physical_layer_ids must be non-empty, sorted, unique, and non-negative")
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.hidden_size = int(hidden_size)
        if min(self.num_q_heads, self.num_kv_heads, self.head_dim, self.hidden_size) <= 0:
            raise ValueError("DMS capture geometry dimensions must be positive")
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError("DMS capture query heads must be divisible by KV heads")
        self.input_stage = str(input_stage)
        if self.input_stage != DMS_CAPTURE_INPUT_STAGE:
            raise ValueError(f"DMS capture input_stage must be {DMS_CAPTURE_INPUT_STAGE!r}")
        if qk_storage_dtype not in {"float16", "float32"}:
            raise ValueError("DMS capture qk_storage_dtype must be float16 or float32")
        self.qk_storage_dtype = str(qk_storage_dtype)
        self._qk_dtype = np.dtype(qk_storage_dtype)
        self.teacher_topk = int(teacher_topk)
        if self.teacher_topk <= 0:
            raise ValueError("DMS capture teacher_topk must be positive")
        self.max_shard_bytes = int(max_shard_bytes)
        if self.max_shard_bytes <= 0:
            raise ValueError("DMS capture max_shard_bytes must be positive")
        self.capture_provenance = _json_safe_mapping(
            {} if capture_provenance is None else capture_provenance,
            label="DMS capture provenance",
        )
        self._sequences: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._finalized = False

    def begin_sequence(
        self,
        *,
        sequence_id: str,
        token_ids: Sequence[int],
        category: str,
        provenance: Mapping[str, Any],
    ) -> None:
        if self._finalized:
            raise RuntimeError("DMS capture writer is already finalized")
        if self._active is not None:
            raise RuntimeError("finish the active DMS sequence before beginning another")
        normalized_id = str(sequence_id)
        normalized_category = str(category)
        if not normalized_id.strip() or normalized_id != normalized_id.strip():
            raise ValueError("DMS capture sequence_id must be a non-empty trimmed string")
        if any(row["sequence_id"] == normalized_id for row in self._sequences):
            raise ValueError(f"duplicate DMS capture sequence_id {normalized_id!r}")
        if not normalized_category.strip() or normalized_category != normalized_category.strip():
            raise ValueError("DMS capture category must be a non-empty trimmed string")
        tokens = tuple(int(token) for token in token_ids)
        if not tokens or any(token < 0 for token in tokens):
            raise ValueError("DMS capture token_ids must be non-empty and non-negative")
        sequence_index = len(self._sequences)
        self._active = {
            "sequence_index": sequence_index,
            "sequence_id": normalized_id,
            "category": normalized_category,
            "token_ids": tokens,
            "token_ids_sha256": _sha256_bytes(
                np.asarray(tokens, dtype=np.int64).tobytes(order="C")
            ),
            "provenance": _json_safe_mapping(provenance, label="DMS sequence provenance"),
            "next_position_by_layer": {
                int(layer_id): 0 for layer_id in self.physical_layer_ids
            },
            "shards": [],
            "teacher_logits": None,
        }

    def capture_chunk(
        self,
        *,
        physical_layer_id: int,
        compact_layer_index: int,
        positions: np.ndarray,
        hidden_bf16: np.ndarray,
        query_f32: np.ndarray,
        key_f32: np.ndarray,
    ) -> None:
        active = self._require_active()
        layer_id = int(physical_layer_id)
        compact_index = int(compact_layer_index)
        if layer_id not in self.physical_layer_ids:
            raise ValueError(f"physical layer {layer_id} is not in the DMS capture map")
        expected_compact = self.physical_layer_ids.index(layer_id)
        if compact_index != expected_compact:
            raise ValueError(
                f"DMS compact layer index mismatch for physical layer {layer_id}: "
                f"expected {expected_compact}, got {compact_index}"
            )
        pos = np.ascontiguousarray(positions, dtype=np.int32)
        if pos.ndim != 1 or pos.size <= 0:
            raise ValueError("DMS capture positions must be a non-empty rank-1 array")
        start = int(active["next_position_by_layer"][layer_id])
        expected_positions = np.arange(start, start + pos.size, dtype=np.int32)
        if not np.array_equal(pos, expected_positions):
            raise ValueError(
                f"DMS capture chunk must begin at the next uncaptured position {start} "
                f"for physical layer {layer_id}"
            )
        token_count = len(active["token_ids"])
        if start + pos.size > token_count:
            raise ValueError("DMS capture chunk exceeds the active sequence token count")
        hidden = np.asarray(hidden_bf16)
        query = np.asarray(query_f32)
        key = np.asarray(key_f32)
        expected_hidden = (pos.size, self.hidden_size)
        expected_query = (pos.size, self.num_q_heads, self.head_dim)
        expected_key = (pos.size, self.num_kv_heads, self.head_dim)
        if hidden.dtype != np.uint16 or hidden.shape != expected_hidden:
            raise ValueError(
                f"DMS capture hidden_bf16 shape/dtype must be {expected_hidden}/uint16"
            )
        if query.shape != expected_query or query.dtype != np.float32:
            raise ValueError(f"DMS capture query_f32 shape/dtype must be {expected_query}/float32")
        if key.shape != expected_key or key.dtype != np.float32:
            raise ValueError(f"DMS capture key_f32 shape/dtype must be {expected_key}/float32")
        if not np.all(np.isfinite(query)) or not np.all(np.isfinite(key)):
            raise ValueError("DMS capture Q/K tensors must be finite")
        token_ids = np.asarray(
            active["token_ids"][start : start + pos.size],
            dtype=np.int32,
        )
        arrays = {
            "positions": pos,
            "token_ids": token_ids,
            "hidden_bf16": np.ascontiguousarray(hidden, dtype=np.uint16),
            "query": np.ascontiguousarray(query, dtype=self._qk_dtype),
            "key": np.ascontiguousarray(key, dtype=self._qk_dtype),
        }
        payload_bytes = sum(int(array.nbytes) for array in arrays.values())
        if payload_bytes > self.max_shard_bytes:
            raise ValueError(
                f"DMS capture chunk payload {payload_bytes} exceeds max_shard_bytes "
                f"{self.max_shard_bytes}"
            )
        filename = (
            f"seq-{int(active['sequence_index']):06d}-layer-{compact_index:02d}-"
            f"p{start:08d}-{start + pos.size:08d}.npz"
        )
        destination = self.output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, destination)
        shard = {
            "path": filename,
            "sha256": _sha256_file(destination),
            "nbytes": destination.stat().st_size,
            "payload_nbytes": payload_bytes,
            "physical_layer_id": layer_id,
            "compact_layer_index": compact_index,
            "position_start": start,
            "position_end": start + pos.size,
            "rows": int(pos.size),
            "hidden_shape": list(expected_hidden),
            "query_shape": list(expected_query),
            "key_shape": list(expected_key),
        }
        active["shards"].append(shard)
        active["next_position_by_layer"][layer_id] = start + int(pos.size)

    def capture_teacher_logits(self, logits: np.ndarray) -> None:
        active = self._require_active()
        if active["teacher_logits"] is not None:
            raise RuntimeError("DMS teacher logits were already captured for this sequence")
        values = np.ascontiguousarray(logits, dtype=np.float32).reshape(-1)
        if values.size <= 0 or not np.all(np.isfinite(values)):
            raise ValueError("DMS teacher logits must be a non-empty finite vector")
        topk = min(self.teacher_topk, int(values.size))
        token_ids = np.arange(values.size, dtype=np.int64)
        order = np.lexsort((token_ids, -values.astype(np.float64)))[:topk]
        maximum = float(np.max(values))
        logsumexp = maximum + float(
            np.log(np.sum(np.exp(values.astype(np.float64) - maximum), dtype=np.float64))
        )
        active["teacher_logits"] = {
            "scope": "next_token_after_sequence",
            "vocab_size": int(values.size),
            "topk": topk,
            "topk_token_ids": [int(token) for token in order.tolist()],
            "topk_logits": [float(values[token]) for token in order.tolist()],
            "logsumexp": logsumexp,
        }

    def finish_sequence(self) -> None:
        active = self._require_active()
        token_count = len(active["token_ids"])
        if any(
            int(active["next_position_by_layer"][layer]) != token_count
            for layer in self.physical_layer_ids
        ):
            raise ValueError(
                "every DMS physical layer must have contiguous full-sequence coverage"
            )
        if active["teacher_logits"] is None:
            raise ValueError("DMS capture sequence is missing teacher logits")
        self._sequences.append(
            {
                "sequence_index": int(active["sequence_index"]),
                "sequence_id": active["sequence_id"],
                "category": active["category"],
                "token_count": token_count,
                "token_ids_sha256": active["token_ids_sha256"],
                "provenance": active["provenance"],
                "teacher_logits": active["teacher_logits"],
                "shards": list(active["shards"]),
            }
        )
        self._active = None

    def finalize(self) -> Path:
        if self._active is not None:
            raise RuntimeError("finish the active DMS sequence before finalizing")
        if self._finalized:
            raise RuntimeError("DMS capture writer is already finalized")
        if not self._sequences:
            raise ValueError("DMS capture manifest requires at least one sequence")
        shard_count = sum(len(row["shards"]) for row in self._sequences)
        manifest = {
            "schema_version": _DMS_CAPTURE_SCHEMA_VERSION,
            "kind": "hipengine_dms_capture_manifest",
            "created_at": datetime.now(UTC).isoformat(),
            "model": {"path": self.model_path, "sha256": self.model_sha256},
            "data_manifest_sha256": self.data_manifest_sha256,
            "capture_provenance": self.capture_provenance,
            "tokenizer": {
                "identity": self.tokenizer_identity,
                "sha256": self.tokenizer_sha256,
            },
            "geometry": {
                "physical_layer_ids": list(self.physical_layer_ids),
                "num_layers": len(self.physical_layer_ids),
                "num_q_heads": self.num_q_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "hidden_size": self.hidden_size,
                "input_stage": self.input_stage,
            },
            "storage": {
                "container": "npz",
                "hidden_dtype": "bfloat16_bits_uint16",
                "qk_source_dtype": "float32",
                "qk_dtype": self.qk_storage_dtype,
                "max_shard_bytes": self.max_shard_bytes,
            },
            "summary": {
                "sequence_count": len(self._sequences),
                "token_count": sum(int(row["token_count"]) for row in self._sequences),
                "shard_count": shard_count,
            },
            "sequences": self._sequences,
        }
        destination = self.output_dir / "capture_manifest.json"
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        destination.with_suffix(destination.suffix + ".sha256").write_text(
            _sha256_bytes(payload) + "\n",
            encoding="ascii",
        )
        self._finalized = True
        return destination

    def _require_active(self) -> dict[str, Any]:
        if self._active is None:
            raise RuntimeError("begin a DMS capture sequence first")
        return self._active


def load_dms_capture_manifest(
    path: str | Path,
    *,
    verify_shards: bool = True,
) -> dict[str, Any]:
    """Load and optionally verify every bounded capture shard."""

    manifest_path = Path(path).expanduser().resolve()
    companion = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not companion.is_file():
        raise FileNotFoundError(f"DMS capture manifest checksum is missing: {companion}")
    expected_manifest_sha256 = companion.read_text(encoding="ascii").strip()
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("DMS capture manifest hash mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DMS capture manifest must be an object")
    if payload.get("schema_version") != _DMS_CAPTURE_SCHEMA_VERSION:
        raise ValueError("unsupported DMS capture manifest schema")
    if payload.get("kind") != "hipengine_dms_capture_manifest":
        raise ValueError("invalid DMS capture manifest kind")
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("DMS capture manifest has no sequences")
    if not verify_shards:
        return payload
    root = manifest_path.parent
    seen: set[Path] = set()
    for sequence in sequences:
        if not isinstance(sequence, dict) or not isinstance(sequence.get("shards"), list):
            raise TypeError("DMS capture sequence/shards must be objects/lists")
        for shard in sequence["shards"]:
            if not isinstance(shard, dict):
                raise TypeError("DMS capture shard record must be an object")
            relative = Path(str(shard.get("path", "")))
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError("DMS capture shard path escapes its manifest directory") from exc
            if candidate in seen:
                raise ValueError(f"duplicate DMS capture shard path: {relative}")
            seen.add(candidate)
            if not candidate.is_file():
                raise FileNotFoundError(f"DMS capture shard is missing: {candidate}")
            if int(shard.get("nbytes", -1)) != candidate.stat().st_size:
                raise ValueError(f"DMS capture shard size mismatch: {relative}")
            if _sha256_file(candidate) != str(shard.get("sha256", "")):
                raise ValueError(f"DMS capture shard hash mismatch: {relative}")
    return payload


__all__ = [
    "DMS_CAPTURE_INPUT_STAGE",
    "DMSCaptureSink",
    "DMSCaptureWriter",
    "load_dms_capture_manifest",
]
