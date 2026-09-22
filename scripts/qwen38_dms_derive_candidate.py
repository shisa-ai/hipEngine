#!/usr/bin/env python3
"""Derive an immutable DMS policy package from an existing linear sidecar.

This changes only policy metadata such as the protected window and exact
eligible-history budget. The sidecar tensor bytes are copied unchanged and
re-hashed. It never mutates the parent package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "working_tree_clean": not dirty}


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def derive_candidate(
    parent_metadata: str | Path,
    output_dir: str | Path,
    *,
    candidate_id: str,
    window_size: int,
    target_compression_ratio: int,
    provenance: dict[str, Any] | None = None,
) -> Path:
    parent_path = Path(parent_metadata).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    identifier = str(candidate_id).strip()
    window = int(window_size)
    target_cr = int(target_compression_ratio)
    if not identifier or identifier != str(candidate_id):
        raise ValueError("candidate_id must be a non-empty trimmed string")
    if window < 0:
        raise ValueError("window_size must be non-negative")
    if target_cr <= 0:
        raise ValueError("target_compression_ratio must be positive")
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")

    parent = _load_object(parent_path, label="parent DMS metadata")
    if int(parent.get("schema_version", -1)) != 2:
        raise ValueError("candidate derivation requires schema-v2 DMS metadata")
    if str(parent.get("decision_source", "")) != "external_linear_sidecar_v1":
        raise ValueError("candidate derivation requires an external linear sidecar")
    if str(parent.get("prefill_selection_mode", "")) != "exact_budget":
        raise ValueError("candidate derivation requires exact-budget prefill selection")
    sidecar = parent.get("sidecar")
    if not isinstance(sidecar, dict):
        raise TypeError("parent DMS metadata sidecar must be an object")
    source = (parent_path.parent / str(sidecar.get("path", ""))).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = _sha256(source)
    if source_hash != str(sidecar.get("sha256", "")):
        raise ValueError("parent DMS sidecar checksum mismatch")

    metadata = json.loads(json.dumps(parent))
    metadata["window_size"] = window
    metadata["target_compression_ratio"] = target_cr
    metadata["evidence_source"] = "policy_derivation.json"
    metadata["policy_derivation"] = {
        "method": "selector_campaign_policy_override_v1",
        "candidate_id": identifier,
        "parent_metadata_sha256": _sha256(parent_path),
        "sidecar_sha256": source_hash,
        "source_window_size": int(parent["window_size"]),
        "source_target_compression_ratio": int(parent["target_compression_ratio"]),
        "window_size": window,
        "target_compression_ratio": target_cr,
        "generator": dict(provenance or _git()),
    }

    output.mkdir(parents=True, exist_ok=False)
    destination = output / source.name
    try:
        shutil.copy2(source, destination)
        if _sha256(destination) != source_hash:
            raise RuntimeError("copied DMS sidecar checksum changed")
        metadata["sidecar"]["path"] = destination.name
        metadata_path = output / "dms_metadata.json"
        derivation_path = output / "policy_derivation.json"
        derivation = dict(metadata["policy_derivation"])
        derivation_path.write_text(
            json.dumps(derivation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "candidate_id": identifier,
            "metadata": {"path": metadata_path.name, "sha256": _sha256(metadata_path)},
            "policy_derivation": {
                "path": derivation_path.name,
                "sha256": _sha256(derivation_path),
            },
            "sidecar": {"path": destination.name, "sha256": source_hash},
        }
        temporary = output / "candidate_manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output / "candidate_manifest.json")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output / "candidate_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--target-cr", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = derive_candidate(
        args.parent_metadata,
        args.output_dir,
        candidate_id=args.candidate_id,
        window_size=args.window_size,
        target_compression_ratio=args.target_cr,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
