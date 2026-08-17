#!/usr/bin/env python3
"""Adapt Qwen3.6 teacher caches plus exact controls to a profile capture.

This does not invent control telemetry. The caller must provide an actual
control capture carrying the same run ID; legacy full-logit caches alone are
insufficient to certify request/state/KV/route ownership. Expected scenario
controls remain a separate input to ``execution_profile_gate.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hipengine.benchmark.execution_profiles import (
    EXECUTION_PROFILE_CAPTURE_KIND,
    EXECUTION_PROFILE_CAPTURE_SCHEMA_VERSION,
    load_control_capture,
    parse_run_capture_manifest,
    qwen36_rows_from_teacher_fixture,
)
from hipengine.execution_profiles import (
    ExecutionProfile,
    manifest_sha256,
    validate_variant_manifest,
)


def _object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be an object")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--logits-manifest", type=Path, required=True)
    parser.add_argument("--actual-controls", type=Path, required=True)
    parser.add_argument("--variant-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--execution-profile",
        choices=tuple(profile.value for profile in ExecutionProfile),
        required=True,
    )
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--shape", default="c1")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    fixture_path = args.fixture.resolve()
    legacy_path = args.logits_manifest.resolve()
    fixture = _object(fixture_path, label="Qwen3.6 teacher fixture")
    legacy = _object(legacy_path, label="Qwen3.6 logits manifest")
    variant_manifest = validate_variant_manifest(
        _object(args.variant_manifest.resolve(), label="variant manifest")
    )
    if variant_manifest["execution_profile"] != args.execution_profile:
        raise ValueError("variant manifest profile differs from --execution-profile")
    variant_manifest_hash = manifest_sha256(variant_manifest)
    if legacy.get("kind") != "quant_quality_full_logits_cache":
        raise ValueError("legacy logits manifest kind must be quant_quality_full_logits_cache")
    fixture_sha = _sha256(fixture_path)
    if legacy.get("fixture_sha256") != fixture_sha:
        raise ValueError("legacy logits manifest does not match the teacher fixture")
    logits_path = Path(str(legacy.get("logits_path", "")))
    if not logits_path.is_absolute():
        logits_path = legacy_path.parent / logits_path
    logits_path = logits_path.resolve()
    logits_sha = _sha256(logits_path)
    if legacy.get("logits_sha256") != logits_sha:
        raise ValueError("legacy logits file hash differs from its manifest")

    rows = qwen36_rows_from_teacher_fixture(
        fixture,
        scenario_id=args.scenario_id,
        shape=args.shape,
    )
    logits = np.load(logits_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(logits, np.ndarray) or logits.ndim != 2 or logits.shape[0] != len(rows):
        raise ValueError("legacy logits are not aligned with adapted teacher rows")
    control_run_id, controls = load_control_capture(args.actual_controls)
    if control_run_id != args.run_id:
        raise ValueError("actual control telemetry run_id differs from --run-id")
    if any(control.scenario_id != args.scenario_id for control in controls):
        raise ValueError("actual control telemetry scenario differs from --scenario-id")
    control_rows = {(record.scenario_step, record.request_id) for record in controls}
    missing_controls = [
        row.logical_key
        for row in rows
        if (row.scenario_step, row.request_id) not in control_rows
    ]
    if missing_controls:
        raise ValueError(
            f"control fixture lacks {len(missing_controls)} adapted teacher rows; "
            f"first={missing_controls[0]!r}"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": EXECUTION_PROFILE_CAPTURE_KIND,
        "schema_version": EXECUTION_PROFILE_CAPTURE_SCHEMA_VERSION,
        "execution_profile": args.execution_profile,
        "scenario_id": args.scenario_id,
        "run_id": args.run_id,
        "variant_manifest_sha256": variant_manifest_hash,
        "repeat_index": args.repeat_index,
        "logits_path": os.path.relpath(logits_path, output.parent),
        "logits_sha256": logits_sha,
        "rows": [row.to_dict() for row in rows],
        "selected_token_ids": [int(token) for token in np.argmax(logits, axis=1)],
        "controls": [record.to_dict() for record in controls],
    }
    parse_run_capture_manifest(payload, base_dir=output.parent)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"capture": str(output), "rows": len(rows), "logits_sha256": logits_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
