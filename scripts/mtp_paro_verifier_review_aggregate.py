#!/usr/bin/env python3
"""Aggregate per-prompt PARO verifier numerical captures into a review matrix.

Scope. Every input is one ``mtp_paro_verifier_numerics`` capture: a single-prompt
B1 teacher-forced screen whose strict teacher is a B1 verifier trajectory. Pooling
screens changes the sample, not the protocol, so this matrix is observational at
every sample size.

Envelopes. Both named envelopes from the captures are reported over the pooled
rows: ``current_sparse_moe_policy`` (``docs/EXECUTION-PROFILES.md`` 2.11), binding
for this sparse top-k MoE model class, and the 2026-08-16
``legacy_mean_tail_kl_envelope``, whose thresholds are unchanged so saved
historical rejections stay readable. Each metric is an observed sample comparison
carrying ``observed``, ``limit``, ``sample_within_limit`` and
``decides_policy: false``.

Policy. ``policy_decision`` is always ``inconclusive``. Nothing here admits a
profile, decides a rate gate, or claims qualification
(``docs/EXECUTION-PROFILES.md`` 6.4 rules 1-4), and a pooled row count never
establishes the full protocol: the protocol is 12 cases x 129 transitions under a
declared prompt suite, not a row total, so ``full_protocol_reached`` is always
false and ``qualification.automatic_admission`` is always false.

Status. ``status`` reports only whether this matrix is usable: ``failed`` when a
source capture recorded an independent execution failure (prefill-root mismatch,
nonfinite logits, route drift, decision inconsistency, manifest drift) or the
pooled rows are unusable, and ``inconclusive`` otherwise. Numerical observations
never set ``status``; a missed top-1 or a damaged row is reported, not judged.

Provenance. Model, backend, candidate-review manifest, registered capture
manifest and prompt uniqueness are validated across inputs. Each source's
capture hash, schema, attestation and qualification record is retained under
``provenance``.

Compatibility. ``checks`` keeps the flat boolean view (``finite``, ``mean_kl``,
``p95_kl``, ``p99_kl``, ``max_kl``, ``top1``, ``per_scope``,
``task_decision_proxy``) that ``mtp_paro_verifier_repeat_review`` reads. Its KL and
``top1`` entries restate the 2026-08-16 mean/tail limits, not the binding 2.11
ones, so they are kept only as that consumer's input: each value is an observed
comparison on this sample, not a gate, and no admission follows from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.mtp_paro_verifier_numerics import (
    _FULL_PROTOCOL_FIXTURE_ROWS,
    _THRESHOLDS,
    _current_policy_envelope,
    _legacy_envelope,
    _scope_summaries,
    _summary,
)

_BINDING_ENVELOPE = "current_sparse_moe_policy"
_LEGACY_ENVELOPE = "legacy_mean_tail_kl_envelope"


def _capture_provenance(path: Path, capture: Mapping[str, Any]) -> dict[str, Any]:
    """Retain one source capture's identity and what it recorded about itself."""

    manifests = capture.get("manifests") or {}
    attestation = capture.get("route_attestation") or {}
    qualification = capture.get("qualification") or {}
    rows = list(capture.get("rows") or [])
    declared_rows = (capture.get("aggregate") or {}).get("rows")
    return {
        "path": str(path),
        "prompt": str(capture["prompt"]["name"]),
        "schema": capture.get("schema"),
        "capture_sha256": capture.get("capture_sha256"),
        "capture_binding": capture.get("capture_binding"),
        "status": capture.get("status"),
        "policy_decision": capture.get("policy_decision"),
        "execution_failures": [str(name) for name in capture.get("execution_failures") or []],
        "execution_failures_known": "execution_failures" in capture,
        "route_flags_attested": attestation.get("attested"),
        "manifests_stable_during_capture": capture.get("manifests_stable_during_capture"),
        "backend_scope": capture.get("backend_scope"),
        "candidate_review_sha256": manifests.get("candidate_review_sha256"),
        "capture_manifest_sha256": manifests.get("capture_manifest_sha256"),
        "rows": len(rows),
        "declared_rows": declared_rows,
        "row_count_consistent": (
            declared_rows is None or int(declared_rows) == len(rows)
        ),
        "qualification_status": qualification.get("status"),
        "source_full_qualification": qualification.get("full_qualification"),
    }


def _source_execution_failures(source: Mapping[str, Any]) -> list[str]:
    """Independent execution failures one source recorded, never numerical ones.

    ``execution_failures`` is authoritative when the capture records it. A
    capture that reports ``status: failed`` without naming a check still fails
    here, and a recorded route drift or manifest drift fails even if the source
    did not list it. A pre-v5 capture's single ``passed``/``rejected`` status is
    not re-read as an execution failure: that status mixed numerical and
    execution outcomes, so it is reported as unknown instead.
    """

    failures = list(source["execution_failures"])
    if not failures and str(source["status"]) == "failed":
        failures.append("unattributed_source_failure")
    if source["route_flags_attested"] is False and "route_flags_attested" not in failures:
        failures.append("route_flags_attested")
    if (
        source["manifests_stable_during_capture"] is False
        and "manifests_stable_during_capture" not in failures
    ):
        failures.append("manifests_stable_during_capture")
    return failures


def _sample_scope(*, prompts: int, rows: int) -> dict[str, Any]:
    """What this pooled sample can and cannot decide (6.4)."""

    return {
        "prompts": int(prompts),
        "rows": int(rows),
        "full_protocol_rows": _FULL_PROTOCOL_FIXTURE_ROWS,
        "full_protocol_reached": False,
        "control_scopes": [],
        "screen_to_full_ratio_interval": None,
        "continuous_gate_policy": (
            "docs/EXECUTION-PROFILES.md 6.4 rules 1-3: median and p90 KL are "
            "reported as observed sample values. Reading a pooled sample against a "
            "full-protocol limit needs a control scope with a known full-gate "
            "verdict in the same batch and a screen-to-full ratio interval; this "
            "matrix has neither, so no continuous gate is decided and no ratio "
            "correction is applied."
        ),
        "rate_gate_policy": (
            "docs/EXECUTION-PROFILES.md 6.4 rule 1: top-1 agreement and the "
            "flip-damage rate are rate gates. They are reported as counts with "
            "intervals and never as a pass/fail, at any row count."
        ),
        "reason": (
            "Pooling single-prompt screens changes the sample, not the protocol. "
            "The canonical full protocol is 12 cases x 129 transitions under a "
            "declared prompt suite (1548 rows), so a pooled row total never "
            "establishes it: reaching that count is neither necessary nor "
            "sufficient, and no gate is decided by row count."
        ),
        "policy_decision": "inconclusive",
    }


def _qualification(
    *,
    prompts: int,
    rows: int,
    execution_failures: Sequence[Mapping[str, Any]],
    aggregation_failures: Sequence[str],
) -> dict[str, Any]:
    """State what this matrix does and does not certify."""

    return {
        "status": "not_qualified",
        "full_qualification": False,
        "policy_decision": "inconclusive",
        "automatic_admission": False,
        "binding_envelope": _BINDING_ENVELOPE,
        "scope": "b1_target_verifier_teacher_forced_review_matrix",
        "prompts": int(prompts),
        "rows": int(rows),
        "full_protocol_reached": False,
        "rate_gates_decided": False,
        "execution_failure_count": len(execution_failures),
        "aggregation_failures": list(aggregation_failures),
        "covers": [
            (
                "pooled B1 fixed-chain strict-versus-candidate target verifier "
                "full-vocabulary logits on strict-owned teacher-forced schedules"
            ),
            "per-prompt provenance, top-1 mismatch rows and task-decision mismatches",
        ],
        "does_not_cover": [
            "any admission decision: a pooled screen issues none at any row count",
            "the public adapter's route selection, rollback and asynchronous publication",
            "B2/B3 draft depths, including the depth-2 to depth-1 output-tail transition",
            "serial-AR equality: the strict teacher here is a B1 verifier trajectory",
            (
                "rate-gate decisions, which no reduced sample decides "
                "(EXECUTION-PROFILES 6.4 rule 1)"
            ),
        ],
        "full_qualification_requires": [
            (
                "task non-inferiority on the applicable prompt suite and "
                "heldouts under a predeclared paired criterion"
            ),
            "three bit-identical fixed-seed repeats of the same manifest and execution schedule",
            "single-slot isolation and unrelated-request replay",
            (
                "the declared full protocol across the prompt suite, whose top-1 "
                "and flip-damage rate gates a pooled screen cannot decide "
                "(EXECUTION-PROFILES 6.4)"
            ),
            "B2/B3 captures through the public adapter",
        ],
    }


def _checks(
    summary: Mapping[str, Any],
    *,
    scope_failures: Sequence[Mapping[str, Any]],
    decision_mismatches: Sequence[Mapping[str, Any]],
    finite_kl: bool,
) -> dict[str, bool]:
    """The flat boolean view ``mtp_paro_verifier_repeat_review`` reads.

    The KL and top-1 entries compare the pooled sample against the 2026-08-16
    mean/tail limits (``_THRESHOLDS``), not the binding 2.11 limits, because that
    consumer indexes these keys by name. It is a view of this sample, not a gate:
    ``policy_decision`` stays ``inconclusive`` and nothing is admitted from it.
    """

    return {
        "finite": bool(finite_kl),
        "mean_kl": summary["mean_kl"] <= _THRESHOLDS["mean_kl_max"],
        "p95_kl": summary["p95_kl"] <= _THRESHOLDS["p95_kl_max"],
        "p99_kl": summary["p99_kl"] <= _THRESHOLDS["p99_kl_max"],
        "max_kl": summary["max_kl"] <= _THRESHOLDS["max_kl_max"],
        "top1": summary["top1_agreement"] >= _THRESHOLDS["top1_min"],
        "per_scope": not scope_failures,
        "task_decision_proxy": not decision_mismatches,
    }


def aggregate(paths: Sequence[Path]) -> dict[str, Any]:
    """Build the review matrix from per-prompt capture artifacts."""

    if not paths:
        raise ValueError("at least one verifier capture is required")
    captures = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    model = captures[0]["model"]
    backend = captures[0]["backend"]
    manifest_hash = captures[0]["manifests"]["candidate_review_sha256"]
    capture_manifest_hash = captures[0]["manifests"].get("capture_manifest_sha256")
    rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    decision_mismatches: list[dict[str, Any]] = []
    prompt_summaries: dict[str, Any] = {}
    source_captures: list[dict[str, Any]] = []
    for path, capture in zip(paths, captures, strict=True):
        if capture["model"] != model or capture["backend"] != backend:
            raise ValueError("captures must share model and backend")
        if capture["manifests"]["candidate_review_sha256"] != manifest_hash:
            raise ValueError("captures must share candidate review manifest")
        if capture["manifests"].get("capture_manifest_sha256") != capture_manifest_hash:
            raise ValueError("captures must share registered production capture manifest")
        prompt = str(capture["prompt"]["name"])
        if prompt in prompt_summaries:
            raise ValueError(f"duplicate prompt capture: {prompt}")
        provenance = _capture_provenance(path, capture)
        source_captures.append(provenance)
        review = capture.get("review") or {}
        prompt_rows = [{**row, "prompt": prompt} for row in capture.get("rows") or []]
        rows.extend(prompt_rows)
        mismatches = [
            {**row, "prompt": prompt} for row in review.get("top1_mismatch_rows") or []
        ]
        decisions = [
            {**row, "prompt": prompt}
            for row in review.get("task_decision_mismatches") or []
        ]
        mismatch_rows.extend(mismatches)
        decision_mismatches.extend(decisions)
        prompt_summaries[prompt] = {
            "source": str(path),
            "capture_sha256": provenance["capture_sha256"],
            "schema": provenance["schema"],
            "category": capture["prompt"]["category"],
            "split": capture["prompt"]["split"],
            "render": capture["prompt"]["render"],
            "rows": len(prompt_rows),
            "status": provenance["status"],
            "policy_decision": provenance["policy_decision"],
            "execution_failures": provenance["execution_failures"],
            "execution_failures_known": provenance["execution_failures_known"],
            "qualification_status": provenance["qualification_status"],
            "source_full_qualification": provenance["source_full_qualification"],
            "aggregate": capture.get("aggregate"),
            "scope_failures": capture.get("scope_failures", []),
            "top1_mismatches": len(mismatches),
            "task_decision_mismatches": len(decisions),
        }
    if not rows:
        raise ValueError("captures contain no numerical rows")

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
    finite_kl = bool(np.isfinite(kl).all())

    aggregation_failures: list[str] = []
    if not finite_kl:
        aggregation_failures.append("finite_kl")
    if any(not source["row_count_consistent"] for source in source_captures):
        aggregation_failures.append("source_row_count_mismatch")

    execution_failures = [
        {"prompt": source["prompt"], "check": check}
        for source in source_captures
        for check in _source_execution_failures(source)
    ]
    execution_failures_known = all(
        source["execution_failures_known"] for source in source_captures
    )
    status = "failed" if (execution_failures or aggregation_failures) else "inconclusive"

    envelopes = {
        _BINDING_ENVELOPE: _current_policy_envelope(rows),
        _LEGACY_ENVELOPE: _legacy_envelope(summary, scopes=scopes),
    }
    current = envelopes[_BINDING_ENVELOPE]
    legacy = envelopes[_LEGACY_ENVELOPE]
    checks = _checks(
        summary,
        scope_failures=scope_failures,
        decision_mismatches=decision_mismatches,
        finite_kl=finite_kl,
    )
    return {
        "schema": "hipengine.paro_mtp_verifier_review_matrix.v2",
        "status": status,
        "policy_decision": "inconclusive",
        "qualification": _qualification(
            prompts=len(captures),
            rows=len(rows),
            execution_failures=execution_failures,
            aggregation_failures=aggregation_failures,
        ),
        "performance_claim": False,
        "model": model,
        "backend": backend,
        "capture_mode": "sequential_strict_then_fast_replay",
        "coverage": {
            "prompts": len(captures),
            "categories": sorted({str(row["category"]) for row in rows}),
            "rows": len(rows),
            "note": (
                "quality rows come from one canonical trajectory per supplied "
                "prompt; repeat captures are handled separately and must not "
                "inflate this denominator"
            ),
        },
        "provenance": {
            "source_captures": source_captures,
            "schemas": sorted({str(source["schema"]) for source in source_captures}),
            "mixed_schemas": len({str(source["schema"]) for source in source_captures}) > 1,
            "capture_sha256": [
                source["capture_sha256"] for source in source_captures
            ],
            "capture_binding": sorted(
                {str(source["capture_binding"]) for source in source_captures}
            ),
            "candidate_review_sha256": manifest_hash,
            "capture_manifest_sha256": [
                source["capture_manifest_sha256"] for source in source_captures
            ],
            "route_flags_attested_all": all(
                source["route_flags_attested"] is True for source in source_captures
            ),
            "manifests_stable_all": all(
                source["manifests_stable_during_capture"] is True
                for source in source_captures
            ),
            "execution_failures_known_all": execution_failures_known,
            "note": (
                "Retained from each source capture and validated for consistency; "
                "a false attestation or manifest-stability value is an execution "
                "finding and fails status, never a numerical observation."
            ),
        },
        "manifests": captures[0]["manifests"],
        # Legacy 2026-08-16 thresholds, retained with unchanged meaning so saved
        # matrices stay comparable. `envelopes` carries both named envelopes.
        "thresholds": _THRESHOLDS,
        "thresholds_envelope": _LEGACY_ENVELOPE,
        "sample_scope": _sample_scope(prompts=len(captures), rows=len(rows)),
        "aggregate": summary,
        "scopes": scopes,
        "scope_failures": scope_failures,
        "scope_failures_note": (
            "Binding scopes whose pooled sample top-1 missed the 6.1 floor. This "
            "is an observation on this sample, not a rate-gate decision."
        ),
        "envelopes": envelopes,
        "envelope_disagreement": {
            "current_sample_limits_exceeded": current["sample_limits_exceeded"],
            "legacy_sample_limits_exceeded": legacy["sample_limits_exceeded"],
            "disagree": (
                bool(current["sample_limits_exceeded"])
                != bool(legacy["sample_limits_exceeded"])
            ),
            "note": (
                "Both envelopes are reported as observed sample comparisons and "
                "neither decides admission or a rate gate. Where the 2026-08-16 "
                "mean/tail sample exceeds a limit the binding 2.11 sample does not, "
                "2.11 is the applicable envelope for this sparse top-k MoE model "
                "class; the legacy numbers stay reported so saved rejections remain "
                "readable."
            ),
        },
        "checks": checks,
        "review": {
            "binding_envelope": _BINDING_ENVELOPE,
            "policy_decision": "inconclusive",
            "current_sample_limits_exceeded": current["sample_limits_exceeded"],
            "legacy_sample_limits_exceeded": legacy["sample_limits_exceeded"],
            "legacy_envelope_thresholds_unchanged": True,
            "rate_gates_decided": current["rate_gate_policy"]["decided"],
            "source_execution_failures": execution_failures,
            "aggregation_failures": aggregation_failures,
            "top1_mismatch_rows": mismatch_rows,
            "task_decision_mismatches": decision_mismatches,
            "point_estimate_uncertain_at_99pct": bool(
                summary["top1_wilson95_low"] < _THRESHOLDS["top1_min"]
                <= summary["top1_wilson95_high"]
            ),
        },
        "prompts": prompt_summaries,
    }


def _summary_line(result: Mapping[str, Any]) -> dict[str, Any]:
    """The compact stdout summary the CLI prints for one matrix."""

    return {
        "status": result["status"],
        "policy_decision": result["policy_decision"],
        **result["aggregate"],
        "prompts": result["coverage"]["prompts"],
        "current_sample_limits_exceeded": result["review"][
            "current_sample_limits_exceeded"
        ],
        "legacy_sample_limits_exceeded": result["review"][
            "legacy_sample_limits_exceeded"
        ],
        "execution_failure_count": len(result["review"]["source_execution_failures"]),
        "aggregation_failures": result["review"]["aggregation_failures"],
        "scope_failures": result["scope_failures"],
        "task_decision_mismatches": len(result["review"]["task_decision_mismatches"]),
        "full_protocol_reached": result["sample_scope"]["full_protocol_reached"],
        "full_qualification": result["qualification"]["full_qualification"],
        "automatic_admission": result["qualification"]["automatic_admission"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 when the matrix is clean (status 'inconclusive', no "
            "policy decision, no qualification), 1 when a source capture recorded "
            "an execution failure or the pooled rows are unusable (status "
            "'failed'). Neither code admits a profile or certifies quality."
        ),
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_summary_line(result), sort_keys=True))
    return 0 if result["status"] == "inconclusive" else 1


if __name__ == "__main__":
    raise SystemExit(main())
