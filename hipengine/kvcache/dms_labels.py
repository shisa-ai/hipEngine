"""Deterministic future-attention oracle labels for external DMS sidecars."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.kvcache.dms_capture import load_dms_capture_manifest

_DMS_LABEL_SCHEMA_VERSION = 1
_DMS_LABEL_KIND = "hipengine_dms_label_manifest"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def future_attention_mass_cpu(
    query: np.ndarray,
    key: np.ndarray,
    *,
    window_size: int,
    scale: float | None = None,
) -> np.ndarray:
    """Sum dense causal attention received after each key leaves the grace window."""

    queries = np.asarray(query, dtype=np.float32)
    keys = np.asarray(key, dtype=np.float32)
    if queries.ndim != 3 or keys.ndim != 3:
        raise ValueError("DMS oracle Q/K must be [tokens,heads,head_dim]")
    tokens, q_heads, head_dim = queries.shape
    if tokens <= 0 or keys.shape[0] != tokens or keys.shape[2] != head_dim:
        raise ValueError("DMS oracle Q/K token/head dimensions do not align")
    kv_heads = int(keys.shape[1])
    if kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("DMS oracle Q heads must be divisible by KV heads")
    window = int(window_size)
    if window < 0:
        raise ValueError("DMS oracle window_size must be non-negative")
    factor = 1.0 / sqrt(float(head_dim)) if scale is None else float(scale)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("DMS oracle attention scale must be finite and positive")
    if not np.all(np.isfinite(queries)) or not np.all(np.isfinite(keys)):
        raise ValueError("DMS oracle Q/K values must be finite")

    group_size = q_heads // kv_heads
    mass = np.zeros((tokens, kv_heads), dtype=np.float64)
    queries64 = queries.astype(np.float64)
    keys64 = keys.astype(np.float64)
    for kv_head in range(kv_heads):
        group = queries64[:, kv_head * group_size : (kv_head + 1) * group_size]
        head_keys = keys64[:, kv_head]
        for query_position in range(tokens):
            old_key_count = max(0, query_position - window)
            if old_key_count == 0:
                continue
            logits = (
                group[query_position] @ head_keys[: query_position + 1].T
            ) * factor
            logits -= np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            mass[:old_key_count, kv_head] += np.sum(
                probabilities[:, :old_key_count],
                axis=0,
            )
    return mass


def build_eviction_labels(
    future_attention_mass: np.ndarray,
    *,
    positions: np.ndarray,
    current_position: int,
    window_size: int,
    target_compression_ratio: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | float]]]:
    """Rank eligible keys independently per KV head under one exact live budget."""

    scores = np.asarray(future_attention_mass, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.int64)
    if scores.ndim != 2 or pos.shape != (scores.shape[0],):
        raise ValueError("DMS label scores/positions must align as [tokens,kv_heads]")
    if scores.shape[0] <= 0 or scores.shape[1] <= 0:
        raise ValueError("DMS labels require non-empty token and KV-head axes")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("DMS future-attention scores must be finite and non-negative")
    if np.any(np.diff(pos) <= 0):
        raise ValueError("DMS label positions must be strictly increasing")
    window = int(window_size)
    target_cr = int(target_compression_ratio)
    current = int(current_position)
    if window < 0:
        raise ValueError("DMS label window_size must be non-negative")
    if target_cr <= 0:
        raise ValueError("DMS target_compression_ratio must be positive")
    if current < int(pos[-1]):
        raise ValueError("DMS current_position cannot precede captured positions")

    eligible = (current - pos) > window
    eligible_indices = np.flatnonzero(eligible)
    protected_count = int(scores.shape[0] - eligible_indices.size)
    unconstrained_live = int(ceil(scores.shape[0] / target_cr))
    target_live = max(protected_count, unconstrained_live)
    evict_count = min(int(eligible_indices.size), int(scores.shape[0] - target_live))
    labels = np.zeros(scores.shape, dtype=np.bool_)
    stats: list[dict[str, int | float]] = []
    for kv_head in range(scores.shape[1]):
        if evict_count:
            order = np.lexsort(
                (
                    pos[eligible_indices],
                    scores[eligible_indices, kv_head],
                )
            )
            selected = eligible_indices[order[:evict_count]]
            labels[selected, kv_head] = True
        actual_live = int(scores.shape[0] - np.count_nonzero(labels[:, kv_head]))
        stats.append(
            {
                "kv_head": kv_head,
                "token_count": int(scores.shape[0]),
                "eligible_count": int(eligible_indices.size),
                "protected_count": protected_count,
                "target_live_count": target_live,
                "evict_count": int(np.count_nonzero(labels[:, kv_head])),
                "actual_live_count": actual_live,
                "actual_compression_ratio": float(scores.shape[0] / actual_live),
            }
        )
    if np.any(labels[~eligible]):
        raise AssertionError("DMS label builder evicted a protected-window token")
    return labels, np.asarray(eligible, dtype=np.bool_), stats


def _load_layer_arrays(
    root: Path,
    shard_records: Sequence[dict[str, Any]],
    *,
    physical_layer_id: int,
    compact_layer_index: int,
    token_count: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
) -> dict[str, np.ndarray]:
    required = {"positions", "token_ids", "hidden_bf16", "query", "key"}
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in required}
    expected_start = 0
    for record in sorted(shard_records, key=lambda row: int(row["position_start"])):
        if int(record.get("physical_layer_id", -1)) != int(physical_layer_id):
            raise ValueError("DMS capture shard physical-layer record mismatch")
        if int(record.get("compact_layer_index", -1)) != int(compact_layer_index):
            raise ValueError("DMS capture shard compact-layer record mismatch")
        if int(record.get("position_start", -1)) != expected_start:
            raise ValueError("DMS capture layer chunks are not contiguous")
        shard_path = (root / str(record["path"])).resolve()
        with np.load(shard_path, allow_pickle=False) as shard:
            missing = sorted(required - set(shard.files))
            if missing:
                raise ValueError(f"DMS capture shard is missing required arrays: {missing}")
            arrays = {name: np.asarray(shard[name]) for name in required}
        rows = int(arrays["positions"].shape[0])
        expected_end = expected_start + rows
        if rows <= 0 or int(record.get("position_end", -1)) != expected_end:
            raise ValueError("DMS capture shard row/position metadata mismatch")
        if arrays["positions"].shape != (rows,) or arrays["token_ids"].shape != (rows,):
            raise ValueError("DMS capture positions/token_ids must be rank-1 and row-aligned")
        if arrays["hidden_bf16"].shape != (rows, hidden_size):
            raise ValueError("DMS capture hidden tensor shape mismatch")
        if arrays["hidden_bf16"].dtype != np.uint16:
            raise ValueError("DMS capture hidden tensor must contain BF16 uint16 bits")
        if arrays["query"].shape != (rows, num_q_heads, head_dim):
            raise ValueError("DMS capture query tensor shape mismatch")
        if arrays["key"].shape != (rows, num_kv_heads, head_dim):
            raise ValueError("DMS capture key tensor shape mismatch")
        if not np.array_equal(
            arrays["positions"].astype(np.int64),
            np.arange(expected_start, expected_end, dtype=np.int64),
        ):
            raise ValueError("DMS capture positions are not canonical contiguous positions")
        for name in required:
            chunks[name].append(np.ascontiguousarray(arrays[name]))
        expected_start = expected_end
    if expected_start != int(token_count):
        raise ValueError("DMS capture layer does not cover the full sequence")
    return {
        name: np.ascontiguousarray(np.concatenate(parts, axis=0))
        for name, parts in chunks.items()
    }


def build_dms_label_artifact(
    capture_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    target_compression_ratio: int,
    window_size: int,
    mass_builder: Callable[..., np.ndarray] | None = None,
    compute_backend: str = "cpu_numpy_float64",
    compute_score_dtype: str = "float64",
    compute_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Verify capture shards, build labels, and emit a compact checksummed artifact."""

    capture_path = Path(capture_manifest_path).expanduser().resolve()
    capture = load_dms_capture_manifest(capture_path, verify_shards=True)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"DMS label output directory must be empty: {output}")
    target_cr = int(target_compression_ratio)
    window = int(window_size)
    if target_cr <= 0 or window < 0:
        raise ValueError("DMS label target CR must be positive and window non-negative")
    builder = future_attention_mass_cpu if mass_builder is None else mass_builder
    geometry = capture.get("geometry")
    if not isinstance(geometry, dict):
        raise TypeError("DMS capture geometry must be an object")
    physical_layer_ids = tuple(int(layer) for layer in geometry["physical_layer_ids"])
    num_q_heads = int(geometry["num_q_heads"])
    num_kv_heads = int(geometry["num_kv_heads"])
    head_dim = int(geometry["head_dim"])
    hidden_size = int(geometry["hidden_size"])
    root = capture_path.parent
    label_sequences: list[dict[str, Any]] = []
    total_tokens = 0
    total_shards = 0
    category_counts: dict[str, dict[str, int]] = {}
    for sequence in capture["sequences"]:
        if not isinstance(sequence, dict):
            raise TypeError("DMS capture sequence must be an object")
        token_count = int(sequence["token_count"])
        if token_count <= 0:
            raise ValueError("DMS capture sequence token_count must be positive")
        records = sequence.get("shards")
        if not isinstance(records, list):
            raise TypeError("DMS capture sequence shards must be a list")
        by_layer: dict[int, list[dict[str, Any]]] = {
            layer: [] for layer in physical_layer_ids
        }
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("DMS capture shard record must be an object")
            layer_id = int(record.get("physical_layer_id", -1))
            if layer_id not in by_layer:
                raise ValueError(f"DMS capture contains undeclared physical layer {layer_id}")
            by_layer[layer_id].append(record)
        label_shards: list[dict[str, Any]] = []
        reference_positions: np.ndarray | None = None
        reference_token_ids: np.ndarray | None = None
        for compact_index, layer_id in enumerate(physical_layer_ids):
            arrays = _load_layer_arrays(
                root,
                by_layer[layer_id],
                physical_layer_id=layer_id,
                compact_layer_index=compact_index,
                token_count=token_count,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
            )
            positions = np.asarray(arrays["positions"], dtype=np.int32)
            token_ids = np.asarray(arrays["token_ids"], dtype=np.int32)
            if reference_positions is None:
                reference_positions = positions
                reference_token_ids = token_ids
            elif not np.array_equal(reference_positions, positions) or not np.array_equal(
                reference_token_ids, token_ids
            ):
                raise ValueError("DMS capture token/position provenance differs across layers")
            scores = np.asarray(
                builder(
                    np.asarray(arrays["query"], dtype=np.float32),
                    np.asarray(arrays["key"], dtype=np.float32),
                    window_size=window,
                ),
                dtype=np.float64,
            )
            if scores.shape != (token_count, num_kv_heads):
                raise ValueError("DMS future-attention builder returned the wrong shape")
            labels, eligible, per_head = build_eviction_labels(
                scores,
                positions=positions,
                current_position=int(positions[-1]),
                window_size=window,
                target_compression_ratio=target_cr,
            )
            filename = (
                f"seq-{int(sequence['sequence_index']):06d}-layer-{compact_index:02d}.npz"
            )
            destination = output / filename
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    positions=positions,
                    token_ids=token_ids,
                    hidden_bf16=np.asarray(arrays["hidden_bf16"], dtype=np.uint16),
                    future_attention_mass=np.asarray(scores, dtype=np.float32),
                    eligible_mask=eligible,
                    evict_labels=labels,
                )
            os.replace(temporary, destination)
            label_shards.append(
                {
                    "path": filename,
                    "sha256": _sha256_file(destination),
                    "nbytes": destination.stat().st_size,
                    "physical_layer_id": layer_id,
                    "compact_layer_index": compact_index,
                    "rows": token_count,
                    "hidden_shape": [token_count, hidden_size],
                    "label_shape": [token_count, num_kv_heads],
                    "per_head": per_head,
                }
            )
        category = str(sequence["category"])
        bucket = category_counts.setdefault(category, {"sequences": 0, "tokens": 0})
        bucket["sequences"] += 1
        bucket["tokens"] += token_count
        total_tokens += token_count
        total_shards += len(label_shards)
        label_sequences.append(
            {
                "sequence_index": int(sequence["sequence_index"]),
                "sequence_id": str(sequence["sequence_id"]),
                "category": category,
                "token_count": token_count,
                "token_ids_sha256": str(sequence["token_ids_sha256"]),
                "provenance": sequence["provenance"],
                "teacher_logits": sequence["teacher_logits"],
                "shards": label_shards,
            }
        )
    manifest = {
        "schema_version": _DMS_LABEL_SCHEMA_VERSION,
        "kind": _DMS_LABEL_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "source_capture": {
            "path": str(capture_path),
            "sha256": _sha256_file(capture_path),
        },
        "model": capture["model"],
        "data_manifest_sha256": capture["data_manifest_sha256"],
        "tokenizer": capture["tokenizer"],
        "geometry": geometry,
        "objective": {
            "method": "future_attention_distillation_v1",
            "target_compression_ratio": target_cr,
            "window_size": window,
            "tie_break": "ascending_score_then_position",
        },
        "compute": {
            "backend": str(compute_backend),
            "score_dtype": str(compute_score_dtype),
            "stored_score_dtype": "float32",
            "provenance": dict(compute_provenance or {}),
        },
        "summary": {
            "sequence_count": len(label_sequences),
            "token_count": total_tokens,
            "shard_count": total_shards,
            "categories": category_counts,
        },
        "sequences": label_sequences,
    }
    manifest_path = output / "label_manifest.json"
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, manifest_path)
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )
    return manifest_path


def load_dms_label_manifest(
    path: str | Path,
    *,
    verify_shards: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    companion = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not companion.is_file():
        raise FileNotFoundError(f"DMS label manifest checksum is missing: {companion}")
    if _sha256_file(manifest_path) != companion.read_text(encoding="ascii").strip():
        raise ValueError("DMS label manifest hash mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DMS label manifest must be an object")
    if payload.get("schema_version") != _DMS_LABEL_SCHEMA_VERSION:
        raise ValueError("unsupported DMS label manifest schema")
    if payload.get("kind") != _DMS_LABEL_KIND:
        raise ValueError("invalid DMS label manifest kind")
    if not verify_shards:
        return payload
    root = manifest_path.parent
    seen: set[Path] = set()
    for sequence in payload.get("sequences", []):
        for shard in sequence.get("shards", []):
            candidate = (root / str(shard.get("path", ""))).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError("DMS label shard path escapes its manifest directory") from exc
            if candidate in seen:
                raise ValueError("duplicate DMS label shard path")
            seen.add(candidate)
            if not candidate.is_file():
                raise FileNotFoundError(f"DMS label shard is missing: {candidate}")
            if candidate.stat().st_size != int(shard.get("nbytes", -1)):
                raise ValueError(f"DMS label shard size mismatch: {candidate.name}")
            if _sha256_file(candidate) != str(shard.get("sha256", "")):
                raise ValueError(f"DMS label shard hash mismatch: {candidate.name}")
    return payload


__all__ = [
    "build_dms_label_artifact",
    "build_eviction_labels",
    "future_attention_mass_cpu",
    "load_dms_label_manifest",
]
