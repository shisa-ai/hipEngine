from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.qwen38_dms_derive_candidate import derive_candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent(tmp_path: Path) -> Path:
    root = tmp_path / "parent"
    root.mkdir(parents=True)
    sidecar = root / "sidecar.safetensors"
    sidecar.write_bytes(b"fixed-sidecar-bytes")
    metadata = {
        "schema_version": 2,
        "decision_source": "external_linear_sidecar_v1",
        "prefill_selection_mode": "exact_budget",
        "window_size": 8192,
        "target_compression_ratio": 2,
        "alpha_offset": 0.125,
        "sidecar": {
            "path": sidecar.name,
            "sha256": _sha256(sidecar),
            "format": "safetensors",
            "dtype": "bfloat16",
        },
        "training": {"method": "fixture"},
    }
    path = root / "dms_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_derive_candidate_copies_unchanged_sidecar_and_binds_policy(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    output = tmp_path / "candidate"
    manifest_path = derive_candidate(
        parent,
        output,
        candidate_id="p2-w2048-cr2",
        window_size=2048,
        target_compression_ratio=2,
        provenance={"commit": "a" * 40, "working_tree_clean": True},
    )

    parent_payload = json.loads(parent.read_text())
    metadata = json.loads((output / "dms_metadata.json").read_text())
    derivation = json.loads((output / "policy_derivation.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    copied = output / "sidecar.safetensors"

    assert copied.read_bytes() == (parent.parent / "sidecar.safetensors").read_bytes()
    assert _sha256(copied) == parent_payload["sidecar"]["sha256"]
    assert metadata["window_size"] == 2048
    assert metadata["target_compression_ratio"] == 2
    assert metadata["alpha_offset"] == parent_payload["alpha_offset"]
    assert metadata["training"] == parent_payload["training"]
    assert metadata["sidecar"]["path"] == copied.name
    assert metadata["policy_derivation"] == derivation
    assert derivation == {
        "candidate_id": "p2-w2048-cr2",
        "generator": {"commit": "a" * 40, "working_tree_clean": True},
        "method": "selector_campaign_policy_override_v1",
        "parent_metadata_sha256": _sha256(parent),
        "sidecar_sha256": _sha256(copied),
        "source_target_compression_ratio": 2,
        "source_window_size": 8192,
        "target_compression_ratio": 2,
        "window_size": 2048,
    }
    assert manifest["candidate_id"] == "p2-w2048-cr2"
    assert manifest["metadata"]["sha256"] == _sha256(output / "dms_metadata.json")
    assert manifest["policy_derivation"]["sha256"] == _sha256(
        output / "policy_derivation.json"
    )


def test_derive_candidate_rejects_mutation_invalid_policy_and_bad_checksum(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path)
    output = tmp_path / "candidate"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        derive_candidate(
            parent,
            output,
            candidate_id="p0",
            window_size=8192,
            target_compression_ratio=2,
        )
    output.rmdir()

    payload = json.loads(parent.read_text())
    payload["sidecar"]["sha256"] = "0" * 64
    parent.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        derive_candidate(
            parent,
            output,
            candidate_id="p0",
            window_size=8192,
            target_compression_ratio=2,
        )
    assert not output.exists()

    parent = _parent(tmp_path / "second")
    for candidate_id, window, cr, message in (
        ("", 1, 1, "candidate_id"),
        ("x", -1, 1, "window_size"),
        ("x", 1, 0, "target_compression_ratio"),
    ):
        with pytest.raises(ValueError, match=message):
            derive_candidate(
                parent,
                output,
                candidate_id=candidate_id,
                window_size=window,
                target_compression_ratio=cr,
            )
