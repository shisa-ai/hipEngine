#!/usr/bin/env python3
"""Aggregate per-prompt PARO verifier numerical captures into a review matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.mtp_paro_verifier_numerics import _THRESHOLDS, _scope_summaries, _summary


def aggregate(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one verifier capture is required")
    captures = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    model = captures[0]["model"]
    backend = captures[0]["backend"]
    manifest_hash = captures[0]["manifests"]["candidate_review_sha256"]
    rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    decision_mismatches: list[dict[str, Any]] = []
    prompt_summaries: dict[str, Any] = {}
    for path, capture in zip(paths, captures, strict=True):
        if capture["model"] != model or capture["backend"] != backend:
            raise ValueError("captures must share model and backend")
        if capture["manifests"]["candidate_review_sha256"] != manifest_hash:
            raise ValueError("captures must share candidate review manifest")
        prompt = str(capture["prompt"]["name"])
        if prompt in prompt_summaries:
            raise ValueError(f"duplicate prompt capture: {prompt}")
        prompt_rows = [{**row, "prompt": prompt} for row in capture["rows"]]
        rows.extend(prompt_rows)
        mismatches = [
            {**row, "prompt": prompt}
            for row in capture["review"]["top1_mismatch_rows"]
        ]
        decisions = [
            {**row, "prompt": prompt}
            for row in capture["review"]["task_decision_mismatches"]
        ]
        mismatch_rows.extend(mismatches)
        decision_mismatches.extend(decisions)
        prompt_summaries[prompt] = {
            "source": str(path),
            "category": capture["prompt"]["category"],
            "split": capture["prompt"]["split"],
            "render": capture["prompt"]["render"],
            "rows": capture["aggregate"]["rows"],
            "aggregate": capture["aggregate"],
            "scope_failures": capture["scope_failures"],
            "top1_mismatches": len(mismatches),
            "task_decision_mismatches": len(decisions),
            "status": capture["status"],
        }

    kl = np.asarray([row["kl"] for row in rows], dtype=np.float64)
    top1 = np.asarray([row["top1_equal"] for row in rows], dtype=np.bool_)
    summary = _summary(kl, top1)
    summary["sample_resolution"] = 1.0 / float(summary["rows"])
    summary["top5_overlap_mean"] = float(
        np.mean([row["top5_overlap"] for row in rows])
    )
    summary["strict_margin_min"] = float(min(row["strict_margin"] for row in rows))
    scopes = _scope_summaries(rows)
    scope_failures = [
        {"dimension": dimension, "value": value}
        for dimension, groups in scopes.items()
        for value, scoped in groups.items()
        if bool(scoped["binding"]) and not bool(scoped["passed"])
    ]
    checks = {
        "finite": bool(np.isfinite(kl).all()),
        "mean_kl": summary["mean_kl"] <= _THRESHOLDS["mean_kl_max"],
        "p95_kl": summary["p95_kl"] <= _THRESHOLDS["p95_kl_max"],
        "p99_kl": summary["p99_kl"] <= _THRESHOLDS["p99_kl_max"],
        "max_kl": summary["max_kl"] <= _THRESHOLDS["max_kl_max"],
        "top1": summary["top1_agreement"] >= _THRESHOLDS["top1_min"],
        "per_scope": not scope_failures,
        "task_decision_proxy": not decision_mismatches,
    }
    automatic = all(checks.values())
    distribution_pass = all(
        checks[key] for key in ("finite", "mean_kl", "p95_kl", "p99_kl", "max_kl")
    )
    return {
        "schema": "hipengine.paro_mtp_verifier_review_matrix.v1",
        "status": "automatic_admission_passed" if automatic else "manual_review_required",
        "performance_claim": False,
        "model": model,
        "backend": backend,
        "capture_mode": "sequential_strict_then_fast_replay",
        "coverage": {
            "prompts": len(captures),
            "categories": sorted({str(row["category"]) for row in rows}),
            "rows": len(rows),
            "note": "quality rows come from one canonical trajectory per supplied prompt; repeat captures are handled separately and must not inflate this denominator",
        },
        "manifests": captures[0]["manifests"],
        "thresholds": _THRESHOLDS,
        "aggregate": summary,
        "scopes": scopes,
        "scope_failures": scope_failures,
        "checks": checks,
        "review": {
            "automatic_admission_threshold_unchanged": True,
            "eligible_for_automatic_admission": automatic,
            "distribution_gates_passed": distribution_pass,
            "top1_mismatch_rows": mismatch_rows,
            "task_decision_mismatches": decision_mismatches,
            "point_estimate_uncertain_at_99pct": bool(
                summary["top1_wilson95_low"] < _THRESHOLDS["top1_min"]
                <= summary["top1_wilson95_high"]
            ),
        },
        "prompts": prompt_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                **result["aggregate"],
                "scope_failures": result["scope_failures"],
                "task_decision_mismatches": len(
                    result["review"]["task_decision_mismatches"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
