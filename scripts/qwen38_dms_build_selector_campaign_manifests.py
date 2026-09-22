#!/usr/bin/env python3
"""Build sealed, source-disjoint manifests for the Qwen3.8 DMS selector campaign.

The builder is deliberately independent of corpus acquisition.  Input records are
JSON objects with ``source_id``, ``path``, ``category``, and either ``token_ids``
or ``text``.  Text records are tokenized by the caller's exact tokenizer binding.
The command-line form accepts tokenized records; this keeps corpus generation and
model access outside this reproducible sealing step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
CAMPAIGN_MATRIX = {
    "training-expansion": {"length": 32768, "per_category": 3},
    "qualification": {"length": 32768, "per_category": 2},
    "boundary-qualification": {"length": 49157, "per_category": 1},
    "final-32k": {"length": 32768, "per_category": 2},
    "final-128k": {"length": 131072, "per_category": 1},
}
SCHEMA_VERSION = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = " ".join(normalized.split())
    return sha256_bytes(normalized.encode("utf-8"))


def _stable_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"builder_commit": commit, "builder": Path(__file__).name}


def _walk_source_values(value: Any, *, ids: set[str], paths: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key = str(key).lower()
            if key in {"source_id", "sourceid", "id"} and isinstance(item, (str, int)):
                ids.add(str(item))
            if key in {"path", "source_path", "filepath", "file_path"} and isinstance(item, (str, int)):
                paths.add(str(item))
            _walk_source_values(item, ids=ids, paths=paths)
    elif isinstance(value, list):
        for item in value:
            _walk_source_values(item, ids=ids, paths=paths)


def source_exclusions(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    source_paths: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _walk_source_values(payload, ids=ids, paths=source_paths)
    return ids, source_paths


def _record_source(record: dict[str, Any]) -> tuple[str, str]:
    source_id = str(record.get("source_id", "")).strip()
    source_path = str(record.get("path", "")).strip()
    if not source_id or not source_path:
        raise ValueError("every candidate requires non-empty source_id and path")
    return source_id, source_path


def _component_sources(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("source_records", [])
    if not isinstance(raw, list):
        raise TypeError("source_records must be a list")
    components: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            raise TypeError("every source_records entry must be an object")
        source_id, source_path = _record_source(row)
        text_hash = str(row.get("normalized_text_sha256", row.get("sha256", ""))).strip()
        if len(text_hash) != 64:
            raise ValueError("every component source requires a SHA-256 text digest")
        components.append(
            {
                key: value
                for key, value in row.items()
                if key != "text"
            }
            | {
                "source_id": source_id,
                "path": source_path,
                "normalized_text_sha256": text_hash,
            }
        )
    return components


def _tokens(record: dict[str, Any], tokenizer: Callable[[str], Iterable[int]] | None) -> list[int]:
    if "token_ids" in record:
        values = record["token_ids"]
        if not isinstance(values, list):
            raise TypeError("token_ids must be a list")
        return [int(value) for value in values]
    if "text" not in record or tokenizer is None:
        raise ValueError("text records require the exact tokenizer binding")
    return [int(value) for value in tokenizer(str(record["text"]))]


def _candidate_key(record: dict[str, Any], text_hash: str, seed: int) -> str:
    source_id, source_path = _record_source(record)
    return sha256_bytes(f"{seed}:{source_id}:{source_path}:{text_hash}".encode())


def _load_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("input must be a record list or an object containing records")
    return payload["records"], dict(payload.get("tokenizer", {}))


def build_campaign_manifests(
    records: Iterable[dict[str, Any]],
    *,
    old_source_manifests: Iterable[Path],
    model_path: Path,
    output_root: Path,
    tokenizer_identity: str,
    tokenizer_sha256: str,
    tokenizer: Callable[[str], Iterable[int]] | None = None,
    model_sha256: str | None = None,
    seed: int = 20260922,
) -> dict[str, Any]:
    """Select and seal the complete campaign matrix without exposing text in output summaries."""
    old_paths = [Path(path).expanduser().resolve() for path in old_source_manifests]
    if len(old_paths) != 4:
        raise ValueError("exactly four v1-v4 source manifests are required")
    excluded_ids, excluded_paths = source_exclusions(old_paths)
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_text: set[str] = set()
    seen_component_ids: set[str] = set()
    seen_component_paths: set[str] = set()
    seen_component_text: set[str] = set()
    for raw in records:
        record = dict(raw)
        source_id, source_path = _record_source(record)
        if source_id in excluded_ids or source_path in excluded_paths:
            continue
        if source_id in seen_ids or source_path in seen_paths:
            raise ValueError("duplicate source ID/path in candidate pool")
        components = _component_sources(record)
        component_ids = {str(row["source_id"]) for row in components}
        component_paths = {str(row["path"]) for row in components}
        component_text = {str(row["normalized_text_sha256"]) for row in components}
        if len(component_ids) != len(components) or len(component_paths) != len(components):
            raise ValueError("duplicate component source ID/path within candidate record")
        if len(component_text) != len(components):
            raise ValueError("duplicate component normalized text within candidate record")
        if component_ids & excluded_ids or component_paths & excluded_paths:
            continue
        if (
            component_ids & seen_component_ids
            or component_paths & seen_component_paths
            or component_text & seen_component_text
        ):
            raise ValueError("duplicate component source ID/path/text across candidate pool")
        seen_ids.add(source_id)
        seen_paths.add(source_path)
        seen_component_ids.update(component_ids)
        seen_component_paths.update(component_paths)
        seen_component_text.update(component_text)
        tokens = _tokens(record, tokenizer)
        if any(token < 0 for token in tokens):
            raise ValueError(f"negative token ID for source {source_id}")
        text = str(record.get("text", ""))
        text_hash = normalized_text_sha256(text) if "text" in record else sha256_bytes(
            _stable_json(tokens)
        )
        if text_hash in seen_text:
            raise ValueError("duplicate normalized text hash in candidate pool")
        seen_text.add(text_hash)
        candidates.append({
            "record": record, "source_id": source_id, "path": source_path,
            "tokens": tokens, "text_sha256": text_hash,
            "source_records": components,
        })

    selected_text: set[str] = set()
    selected_ids: set[str] = set()
    selected_paths: set[str] = set()
    outputs: dict[str, dict[str, Any]] = {}
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    old_records = [{"path": str(path), "sha256": sha256_file(path)} for path in old_paths]

    for split, spec in CAMPAIGN_MATRIX.items():
        chosen: list[dict[str, Any]] = []
        for category in CATEGORIES:
            eligible = [row for row in candidates if row["record"].get("category") == category]
            eligible.sort(key=lambda row: _candidate_key(row["record"], row["text_sha256"], seed))
            for row in eligible:
                if row["source_id"] in selected_ids or row["path"] in selected_paths or row["text_sha256"] in selected_text:
                    continue
                if len(row["tokens"]) < int(spec["length"]):
                    continue
                chosen.append(row)
                selected_ids.add(row["source_id"])
                selected_paths.add(row["path"])
                selected_text.add(row["text_sha256"])
                if sum(item["record"].get("category") == category for item in chosen) == int(spec["per_category"]):
                    break
            if sum(item["record"].get("category") == category for item in chosen) != int(spec["per_category"]):
                raise ValueError(f"insufficient disjoint {category} records for {split}")

        sequences = []
        source_sequences = []
        for index, row in enumerate(chosen):
            length = int(spec["length"])
            sequence_id = f"selector-{split}-{row['record']['category']}-{index:02d}"
            sequences.append({
                "sequence_id": sequence_id,
                "category": row["record"]["category"], "split": split,
                "token_ids": row["tokens"][:length],
                "provenance": {
                    "source_id": row["source_id"], "source_path": row["path"],
                    "normalized_text_sha256": row["text_sha256"],
                    "tokenizer": {"identity": tokenizer_identity, "sha256": tokenizer_sha256},
                    "model_path": str(model_path), "model_sha256": model_sha256,
                    "source_records": row["source_records"],
                },
            })
            source_sequences.append({
                "sequence_id": sequence_id, "category": row["record"]["category"],
                "source_id": row["source_id"], "path": row["path"],
                "normalized_text_sha256": row["text_sha256"],
                "source_records": row["source_records"],
            })

        common = {
            "schema_version": SCHEMA_VERSION,
            "kind": "hipengine_dms_selector_campaign_manifest",
            "split": split, "length_tokens": int(spec["length"]),
            "tokenizer": {"identity": tokenizer_identity, "sha256": tokenizer_sha256},
            "model": {"path": str(model_path), "sha256": model_sha256},
            "source_exclusions": old_records,
            "provenance": _git_provenance(),
        }
        data_payload = {**common, "count": len(sequences), "sequences": sequences}
        source_payload = {**common, "count": len(source_sequences), "sequences": source_sequences}
        data_path = output_root / f"{split}-data.json"
        source_path = output_root / f"{split}-sources.json"
        data_path.write_text(json.dumps(data_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        source_path.write_text(json.dumps(source_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        outputs[split] = {
            "data": {"path": str(data_path), "sha256": sha256_file(data_path), "count": len(sequences)},
            "sources": {"path": str(source_path), "sha256": sha256_file(source_path), "count": len(source_sequences)},
        }

    index = {
        "schema_version": SCHEMA_VERSION, "kind": "hipengine_dms_selector_campaign_sealed_index",
        "campaign_matrix": CAMPAIGN_MATRIX, "tokenizer": {"identity": tokenizer_identity, "sha256": tokenizer_sha256},
        "model": {"path": str(model_path), "sha256": model_sha256},
        "old_source_manifests": old_records, "manifests": outputs,
        "sealed": True, "candidate_result": None, "provenance": _git_provenance(),
    }
    index_path = output_root / "sealed-index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"output_root": str(output_root), "sealed_index": {"path": str(index_path), "sha256": sha256_file(index_path)}, "manifests": outputs}


def build_parser() -> argparse.ArgumentParser:
    matrix = "; ".join(
        f"{name}: {spec['per_category']}/category x {spec['length']} tokens"
        for name, spec in CAMPAIGN_MATRIX.items()
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Fixed campaign matrix: {matrix}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="JSON record pool")
    parser.add_argument("--model", type=Path, required=True, help="explicit Qwen3.8 model path used for provenance")
    parser.add_argument("--old-source-manifest", type=Path, required=True, nargs=4, metavar="V1_V4_MANIFEST", help="exactly four v1/v2/v3/v4 source manifests")
    parser.add_argument("--output-root", type=Path, required=True, help="new unique output root")
    parser.add_argument("--tokenizer-identity", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--model-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records, binding = _load_records(args.input)
    identity = args.tokenizer_identity
    tokenizer_hash = args.tokenizer_sha256
    if binding and (binding.get("identity") != identity or binding.get("sha256") != tokenizer_hash):
        raise ValueError("input tokenizer binding does not match CLI tokenizer binding")
    result = build_campaign_manifests(
        records, old_source_manifests=args.old_source_manifest, model_path=args.model,
        output_root=args.output_root, tokenizer_identity=identity, tokenizer_sha256=tokenizer_hash,
        model_sha256=args.model_sha256,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
