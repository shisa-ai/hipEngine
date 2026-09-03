#!/usr/bin/env python3
"""Emit the Q8 MMQ route-coverage finding artifact.

Rule B of the PF-1 basis predeclaration
(``worklog/entries/20260903T033202.234656Z-lhl-qwen4exp-pf1-basis-predeclaration-6f6810.md``)
requires every production-numerics packet to record route-engagement coverage:
the fraction of compared rows whose trajectory dispatched the candidate route at
least once.  This script computes that coverage for the 2026-08-29 Q8 MMQ
admission fixture and the 2026-09-03 canonical fixture, and pairs it with the
measured per-prompt incumbent-versus-strict divergence from the reconciliation
gate run.

It makes no performance claim and admits nothing; it classifies existing
evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KIND = "hipengine_production_numerics_route_coverage_finding"
SCHEMA_VERSION = 1
# Every registered Qwen4Exp dense-Q8 shape shares this admission threshold.
MMQ_MIN_ROWS = 64
DECODE_ROWS = 1


def _fixture_coverage(fixture_path: Path, decode_steps: int) -> dict:
    fixture_path = fixture_path.resolve()
    fixture = json.loads(fixture_path.read_text())
    cases = fixture["cases"]
    per_case = []
    for case in cases:
        prompt_tokens = int(case["prompt_tokens"])
        # Prefill dispatches the policy once per chunk; decode dispatches at
        # rows==1, which never reaches the threshold.
        prefill_engages = prompt_tokens >= MMQ_MIN_ROWS
        per_case.append(
            {
                "id": str(case["id"]),
                "category": str(case["category"]),
                "prompt_tokens": prompt_tokens,
                "prefill_engages_q8_mmq": prefill_engages,
                "compared_rows": 1 + decode_steps,
            }
        )
    engaged_cases = [row for row in per_case if row["prefill_engages_q8_mmq"]]
    total_rows = sum(row["compared_rows"] for row in per_case)
    engaged_rows = sum(row["compared_rows"] for row in engaged_cases)
    return {
        "fixture": str(fixture_path.relative_to(REPO_ROOT)),
        "fixture_kind": fixture.get("kind"),
        "cases": len(per_case),
        "cases_engaging_route": len(engaged_cases),
        "compared_rows": total_rows,
        "rows_in_route_engaging_cases": engaged_rows,
        "route_engagement_coverage": engaged_rows / total_rows if total_rows else 0.0,
        "decode_rows_engage_route": DECODE_ROWS >= MMQ_MIN_ROWS,
        "per_case": per_case,
    }


def _measured_divergence(gate_path: Path) -> dict:
    gate_path = gate_path.resolve()
    gate = json.loads(gate_path.read_text())
    context = gate.get("context_quality") or {}
    summary = context.get("summary") or {}
    return {
        "gate_artifact": str(gate_path.relative_to(REPO_ROOT)),
        "route": gate.get("route"),
        "incumbent_vs_strict": {
            key: summary.get(key)
            for key in (
                "rows",
                "kl_mean",
                "kl_p50",
                "kl_p95",
                "kl_p99",
                "kl_max",
                "top1_agreement",
                "max_abs_logit_delta",
            )
        },
        "incumbent_vs_strict_by_category": {
            name: {
                "rows": scope.get("rows"),
                "kl_mean": scope.get("kl_mean"),
                "kl_max": scope.get("kl_max"),
                "top1_agreement": scope.get("top1_agreement"),
            }
            for name, scope in (context.get("by_scope", {}).get("category", {})).items()
        },
        "repeat_determinism_passed": (gate.get("context_repeat_determinism") or {}).get(
            "passed"
        ),
    }


def _single_fixture_artifact(fixture_path: Path, decode_steps: int, output: Path) -> int:
    """PF-0 mode: record route coverage for one fixture without gate inputs.

    Pure dispatch arithmetic (prompt_tokens vs MMQ_MIN_ROWS); no measurement
    and no claim.
    """
    cov = _fixture_coverage(fixture_path, decode_steps)
    classification = (
        "route_vacuous_for_scope"
        if cov["route_engagement_coverage"] < 0.5
        else "route_covering"
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "accepted_diagnostic_finding",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "profile_qualification_claim": False,
        "admits_or_rejects_candidate": False,
        "subject": {
            "route": "prefill_policy_qwen4exp_dense_q8_shapes",
            "selected_variant": "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
            "strict_fallback_variant": "coltile8_rowbatch4_f32_f32_out",
            "admission_threshold_rows": MMQ_MIN_ROWS,
            "threshold_source": "hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_mmq_prefill.py:104 _QWEN4EXP_MIN_ROWS",
        },
        "decision_rule": {
            "predeclared_in": "worklog/entries/20260903T033202.234656Z-lhl-qwen4exp-pf1-basis-predeclaration-6f6810.md",
            "rule_b_coverage_floor": 0.5,
        },
        "coverage": {"single_fixture": cov},
        "measured": None,
        "measured_note": (
            "single-fixture mode: dispatch-coverage diagnostic only; no "
            "incumbent-vs-strict gate run is paired"
        ),
        "verdict": {
            "single_fixture_classification": classification,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=1) + "\n")
    print(json.dumps(
        {
            "output": str(output),
            "single_fixture_classification": classification,
            "coverage": cov["route_engagement_coverage"],
            "cases": cov["cases"],
            "compared_rows": cov["compared_rows"],
        },
        indent=1,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-fixture", type=Path, default=None)
    parser.add_argument("--canonical-fixture", type=Path, default=None)
    parser.add_argument("--admission-gate", type=Path, default=None)
    parser.add_argument("--canonical-gate", type=Path, default=None)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--single-fixture", type=Path, default=None,
                        help="PF-0 mode: record coverage for one fixture only")
    parser.add_argument("--single-output", type=Path, default=None,
                        help="output path for the single-fixture artifact")
    args = parser.parse_args(argv)

    if args.single_fixture:
        if not args.single_output:
            parser.error("--single-output is required with --single-fixture")
        return _single_fixture_artifact(
            args.single_fixture, args.decode_steps, args.single_output
        )

    missing = [
        name
        for name, value in (
            ("--admission-fixture", args.admission_fixture),
            ("--canonical-fixture", args.canonical_fixture),
            ("--admission-gate", args.admission_gate),
            ("--canonical-gate", args.canonical_gate),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        parser.error(
            "the following arguments are required: " + ", ".join(missing)
        )

    admission_cov = _fixture_coverage(args.admission_fixture, args.decode_steps)
    canonical_cov = _fixture_coverage(args.canonical_fixture, args.decode_steps)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "accepted_diagnostic_finding",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "profile_qualification_claim": False,
        "admits_or_rejects_candidate": False,
        "subject": {
            "route": "prefill_policy_qwen4exp_dense_q8_shapes",
            "selected_variant": "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
            "strict_fallback_variant": "coltile8_rowbatch4_f32_f32_out",
            "admission_threshold_rows": MMQ_MIN_ROWS,
            "threshold_source": "hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_mmq_prefill.py:104 _QWEN4EXP_MIN_ROWS",
        },
        "decision_rule": {
            "predeclared_in": "worklog/entries/20260903T033202.234656Z-lhl-qwen4exp-pf1-basis-predeclaration-6f6810.md",
            "rule_b_coverage_floor": 0.5,
        },
        "coverage": {
            "admission_suite_2026_08_29": admission_cov,
            "canonical_suite_2026_09_03": canonical_cov,
        },
        "measured": {
            "admission_suite": _measured_divergence(args.admission_gate),
            "canonical_suite": _measured_divergence(args.canonical_gate),
        },
    }
    coverage = admission_cov["route_engagement_coverage"]
    artifact["verdict"] = {
        "admission_suite_classification": (
            "route_vacuous_for_scope" if coverage < 0.5 else "route_covering"
        ),
        "canonical_suite_classification": (
            "route_vacuous_for_scope"
            if canonical_cov["route_engagement_coverage"] < 0.5
            else "route_covering"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), **artifact["verdict"],
                      "admission_coverage": coverage,
                      "canonical_coverage": canonical_cov["route_engagement_coverage"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
