#!/usr/bin/env python3
"""Evaluate strict-teacher execution-profile captures into one JSON artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.execution_profiles import (
    Bf16NoninferiorityThresholds,
    EvaluationThresholds,
    build_execution_profile_artifact,
    load_control_fixture,
    load_run_capture_manifest,
)


def _object(path: str | Path, *, label: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be an object")
    return dict(payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-manifest", required=True)
    parser.add_argument("--strict-manifest", required=True)
    parser.add_argument("--strict-capture", required=True)
    parser.add_argument("--candidate-capture", required=True)
    parser.add_argument("--expected-controls", required=True)
    parser.add_argument("--strict-expected-controls", required=True)
    parser.add_argument(
        "--comparison-controls",
        action="append",
        default=[],
        help="exact control fixture for each additional isolation/composition scenario",
    )
    parser.add_argument("--repeat-capture", action="append", default=[])
    parser.add_argument("--isolation-capture", action="append", default=[])
    parser.add_argument("--batch-invariant-capture", action="append", default=[])
    parser.add_argument("--task-results", required=True)
    parser.add_argument("--arithmetic-class", choices=("T0", "T1", "T2", "T3"), required=True)
    parser.add_argument("--thresholds", help="optional EvaluationThresholds JSON object")
    parser.add_argument("--bf16-logits", help="optional aligned BF16 .npy logits")
    parser.add_argument("--bf16-thresholds", help="optional BF16 non-inferiority JSON object")
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    thresholds = (
        EvaluationThresholds(**_object(args.thresholds, label="thresholds"))
        if args.thresholds
        else None
    )
    bf16_thresholds = (
        Bf16NoninferiorityThresholds(
            **_object(args.bf16_thresholds, label="BF16 thresholds")
        )
        if args.bf16_thresholds
        else None
    )
    bf16_logits = (
        np.load(args.bf16_logits, allow_pickle=False) if args.bf16_logits else None
    )
    if bf16_logits is not None and not isinstance(bf16_logits, np.ndarray):
        raise ValueError("--bf16-logits must refer to one .npy array")

    comparison_controls = {}
    for path in args.comparison_controls:
        controls = load_control_fixture(path)
        scenario_id = controls[0].scenario_id
        if scenario_id in comparison_controls:
            raise ValueError(f"duplicate --comparison-controls scenario: {scenario_id}")
        comparison_controls[scenario_id] = controls

    artifact = build_execution_profile_artifact(
        variant_manifest=_object(args.variant_manifest, label="variant manifest"),
        strict_manifest=_object(args.strict_manifest, label="strict manifest"),
        arithmetic_class=args.arithmetic_class,
        strict_capture=load_run_capture_manifest(args.strict_capture),
        candidate_capture=load_run_capture_manifest(args.candidate_capture),
        expected_controls=load_control_fixture(args.expected_controls),
        strict_expected_controls=load_control_fixture(args.strict_expected_controls),
        comparison_expected_controls=comparison_controls,
        repeat_captures=tuple(
            load_run_capture_manifest(path) for path in args.repeat_capture
        ),
        isolation_captures=tuple(
            load_run_capture_manifest(path) for path in args.isolation_capture
        ),
        batch_invariant_captures=tuple(
            load_run_capture_manifest(path) for path in args.batch_invariant_capture
        ),
        task_results=_object(args.task_results, label="task results"),
        thresholds=thresholds,
        bf16_logits=bf16_logits,
        bf16_thresholds=bf16_thresholds,
    )
    output = Path(args.output)
    _write_json_atomic(output, artifact)
    decision = artifact["decision"]
    print(
        f"execution_profile={artifact['execution_profile']} status={decision['status']} "
        f"automatic={str(decision['eligible_for_automatic_admission']).lower()} "
        f"artifact={output}"
    )
    return 0 if decision["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
