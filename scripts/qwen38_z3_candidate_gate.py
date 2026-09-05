#!/usr/bin/env python3
"""Fail-closed evidence gate for Qwen3.8 structural-differential candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

DECLARED_CLASSES = {
    "P1_F16_ACTIVATION_B": "T1",
    "M1_C1_ACCEPT_ROUTE_PARITY": "T2",
    "M2_C2_DRAFT_DEPTH": "T3",
    "M3_ACCEPT_BOUNDARY_DATAFLOW": "T0",
}
_COMMON = (
    "implemented",
    "strict_fallback_registered",
    "full_category_suite",
    "heldouts",
    "complete_wall_improved",
)
_REQUIRED = {
    "P1_F16_ACTIVATION_B": _COMMON + (
        "controls_exact", "production_numerics", "deterministic", "isolation",
        "lifecycle", "bf16_relative", "task_quality", "expected_kernel_trace",
    ),
    "M1_C1_ACCEPT_ROUTE_PARITY": _COMMON + (
        "controls_exact", "production_numerics", "deterministic", "isolation",
        "lifecycle", "task_quality", "expected_kernel_trace",
    ),
    "M2_C2_DRAFT_DEPTH": _COMMON + (
        "explicit_experiment", "controls_exact", "deterministic", "isolation",
        "lifecycle", "true_ar_control", "automatic_promotion_disabled",
    ),
    "M3_ACCEPT_BOUNDARY_DATAFLOW": _COMMON + (
        "strict_exact", "controls_exact", "deterministic", "isolation",
        "lifecycle", "cancellation", "compaction", "expected_kernel_trace",
    ),
}


def evaluate_candidate_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(payload.get("candidate_id") or "")
    expected_class = DECLARED_CLASSES.get(candidate_id)
    failures: list[str] = []
    if expected_class is None:
        failures.append("unknown_candidate_id")
        required: tuple[str, ...] = ()
    else:
        required = _REQUIRED[candidate_id]
        if str(payload.get("declared_class") or "") != expected_class:
            failures.append(f"declared_class_must_be_{expected_class}")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        checks = {}
        failures.append("checks_mapping_missing")
    for name in required:
        if checks.get(name) is not True:
            failures.append(f"check_failed:{name}")
    if candidate_id == "M2_C2_DRAFT_DEPTH" and checks.get("ordinary_production_default") is True:
        failures.append("t3_ordinary_production_default_forbidden")
    return {
        "schema": 1,
        "candidate_id": candidate_id,
        "declared_class": expected_class,
        "passed": not failures,
        "required_checks": list(required),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.evidence.read_text())
    if isinstance(payload.get("candidates"), list):
        results = [evaluate_candidate_evidence(row) for row in payload["candidates"]]
        result = {"schema": 1, "kind": "qwen38_z3_candidate_gate", "passed": all(r["passed"] for r in results), "results": results}
    else:
        result = evaluate_candidate_evidence(payload)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
