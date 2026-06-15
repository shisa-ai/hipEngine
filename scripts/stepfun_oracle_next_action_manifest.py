#!/usr/bin/env python3
"""Emit a StepFun oracle-parity next-action manifest.

The manifest consolidates the retained oracle blocker diagnosis, evidence
consistency check, and host top-logit margin into one handoff artifact for the
next logits/backend parity investigation. It intentionally keeps oracle parity,
KV readiness, e2e readiness, and performance claims disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_host_logit_margin as host_margin
from scripts import stepfun_oracle_blocker_diagnosis as diagnosis_mod
from scripts import stepfun_oracle_evidence_consistency_check as consistency_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-next-action-manifest.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnosis-artifact",
        type=Path,
        default=diagnosis_mod.DEFAULT_OUTPUT,
        help="Retained oracle blocker diagnosis artifact.",
    )
    parser.add_argument(
        "--consistency-artifact",
        type=Path,
        default=consistency_mod.DEFAULT_OUTPUT,
        help="Retained oracle evidence consistency-check artifact.",
    )
    parser.add_argument(
        "--host-logit-margin-artifact",
        type=Path,
        default=host_margin.DEFAULT_OUTPUT,
        help="Retained host top-logit margin artifact.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the manifest artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write JSON output atomically to this path instead of stdout. Use "
            "--default-output for the canonical StepFun artifact path."
        ),
    )
    parser.add_argument(
        "--default-output",
        action="store_true",
        help=f"Write to the canonical artifact path: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--status-only", action="store_true", help="Emit only status.")
    parser.add_argument(
        "--investigation-ready-only",
        action="store_true",
        help="Emit only whether all preconditions for the next investigation are met.",
    )
    parser.add_argument(
        "--next-action-only",
        action="store_true",
        help="Emit only the next-action summary string.",
    )
    parser.add_argument(
        "--required-inputs-only",
        action="store_true",
        help="Emit only required retained evidence input records.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the manifest payload.",
    )
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted oracle next-action manifest with current "
            f"diagnosis/consistency/logit-margin metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-manifest, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-manifest, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-manifest, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _artifact_ref(path: Path, payload: dict[str, object], *, role: str) -> dict[str, object]:
    return {
        "path": str(path),
        "artifact_kind": payload.get("artifact_kind"),
        "status": payload.get("status"),
        "sha256": status_mod._stable_json_sha256(payload),
        "role": role,
        "required": True,
    }


def _active_finding(diagnosis: dict[str, object], name: str) -> dict[str, object]:
    findings = diagnosis.get("active_findings")
    if not isinstance(findings, list):
        return {}
    for item in findings:
        if isinstance(item, dict) and item.get("finding") == name:
            return item
    return {}


def _ruled_out_names(diagnosis: dict[str, object]) -> list[str]:
    causes = diagnosis.get("ruled_out_causes")
    if not isinstance(causes, list):
        return []
    names: list[str] = []
    for item in causes:
        if isinstance(item, dict) and item.get("ruled_out") is True:
            cause = item.get("cause")
            if isinstance(cause, str):
                names.append(cause)
    return names


def _diagnosis_evidence_inputs(diagnosis: dict[str, object]) -> list[dict[str, object]]:
    evidence = diagnosis.get("evidence_artifacts")
    if not isinstance(evidence, dict):
        return []
    inputs: list[dict[str, object]] = []
    roles = {
        "backend_matrix": "canonical and comparison oracle backend outcomes",
        "token_mismatch": "canonical generated-text mismatch details",
        "rank_check": "generated token host-top-list absence",
        "top_token_roundtrip": "host top-token text/token-id round-trip evidence",
        "prompt_token_roundtrip": "host prompt input-id/tokenizer round-trip evidence",
    }
    for key in [
        "backend_matrix",
        "token_mismatch",
        "rank_check",
        "top_token_roundtrip",
        "prompt_token_roundtrip",
    ]:
        ref = evidence.get(key)
        if not isinstance(ref, dict):
            continue
        inputs.append(
            {
                "key": key,
                "path": ref.get("path"),
                "artifact_kind": ref.get("artifact_kind"),
                "status": ref.get("status"),
                "sha256": ref.get("sha256"),
                "role": roles[key],
                "required": True,
            }
        )
    return inputs


def build_oracle_next_action_manifest(
    *,
    diagnosis_artifact: Path = diagnosis_mod.DEFAULT_OUTPUT,
    consistency_artifact: Path = consistency_mod.DEFAULT_OUTPUT,
    host_logit_margin_artifact: Path = host_margin.DEFAULT_OUTPUT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the next-action manifest for the oracle-parity blocker."""

    diagnosis = _load_json_object(diagnosis_artifact)
    consistency = _load_json_object(consistency_artifact)
    margin = _load_json_object(host_logit_margin_artifact)
    mismatch = _active_finding(diagnosis, "generated_text_mismatch")
    rank_absence = _active_finding(
        diagnosis, "generated_token_absent_from_host_top_tokens"
    )
    hip_timeout = _active_finding(diagnosis, "hip_oracle_timeout")

    preconditions = [
        {
            "name": "oracle_diagnosis_status_blocked",
            "passed": diagnosis.get("status") == "blocked",
            "expected": "blocked",
            "actual": diagnosis.get("status"),
            "command": "python3 scripts/stepfun_oracle_blocker_diagnosis.py --status-only",
        },
        {
            "name": "oracle_evidence_consistency_matches",
            "passed": consistency.get("status") == "match",
            "expected": "match",
            "actual": consistency.get("status"),
            "command": "python3 scripts/stepfun_oracle_evidence_consistency_check.py --status-only",
        },
        {
            "name": "host_logit_margin_passes",
            "passed": margin.get("status") == "passed",
            "expected": "passed",
            "actual": margin.get("status"),
            "command": "python3 scripts/stepfun_host_logit_margin.py --status-only",
        },
        {
            "name": "expected_token_is_host_top1",
            "passed": margin.get("top1_matches_expected_id") is True,
            "expected": True,
            "actual": margin.get("top1_matches_expected_id"),
            "command": "python3 scripts/stepfun_host_logit_margin.py --expected-top1-only",
        },
        {
            "name": "generated_text_mismatch_is_active",
            "passed": mismatch.get("active") is True,
            "expected": True,
            "actual": mismatch.get("active"),
            "command": "python3 scripts/stepfun_validator_status.py --next-action-oracle-evidence-gaps-joined-only",
        },
    ]
    missing_preconditions = [
        item["name"] for item in preconditions if item.get("passed") is not True
    ]
    investigation_ready = not missing_preconditions
    required_inputs = [
        _artifact_ref(
            diagnosis_artifact,
            diagnosis,
            role="consolidated ruled-out causes and active oracle findings",
        ),
        _artifact_ref(
            consistency_artifact,
            consistency,
            role="cross-artifact freshness and consistency gate",
        ),
        _artifact_ref(
            host_logit_margin_artifact,
            margin,
            role="host top-logit margin context for the expected token",
        ),
        *_diagnosis_evidence_inputs(diagnosis),
    ]
    next_action = diagnosis.get("next_action")
    if not isinstance(next_action, str):
        next_action = (
            "investigate logits/backend parity for the canonical Vulkan executed oracle; "
            "do not claim oracle parity until generated_text_matches_target passes"
        )
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_oracle_next_action_manifest",
        "date": artifact_date,
        "status": "blocked",
        "ready": False,
        "investigation_ready": investigation_ready,
        "missing_preconditions": missing_preconditions,
        "next_action_kind": "logits_backend_parity_investigation",
        "next_action": next_action,
        "target": {
            "canonical_backend": "vulkan",
            "readiness_gate": "oracle_parity",
            "unresolved_evidence_gap": "generated_text_matches_target",
            "expected_next_token_id": mismatch.get("expected_next_token_id"),
            "expected_next_token_text": mismatch.get("expected_next_token_text"),
            "generated_first_token_id": mismatch.get("generated_first_token_id"),
            "generated_text": mismatch.get("generated_text"),
            "generated_text_token_ids": mismatch.get("generated_text_token_ids"),
            "generated_text_stripped_token_ids": mismatch.get(
                "generated_text_stripped_token_ids"
            ),
            "generated_first_token_matches_expected_id": mismatch.get(
                "generated_first_token_matches_expected_id"
            ),
            "generated_token_in_host_top_tokens": rank_absence.get(
                "generated_token_in_host_top_tokens"
            ),
            "generated_token_host_rank": rank_absence.get(
                "generated_token_host_rank"
            ),
            "host_top_token_ids": rank_absence.get("host_top_token_ids"),
            "host_top1_to_top2_margin": margin.get("top1_to_top2_margin"),
            "host_top1_to_top5_margin": margin.get("top1_to_top5_margin"),
        },
        "ruled_out_causes": _ruled_out_names(diagnosis),
        "active_findings": [
            item.get("finding")
            for item in diagnosis.get("active_findings", [])
            if isinstance(item, dict) and item.get("active") is True
        ],
        "comparison_only_findings": [
            name
            for name in ["hip_oracle_timeout" if hip_timeout.get("active") is True else None]
            if name is not None
        ],
        "preconditions": preconditions,
        "required_evidence_inputs": required_inputs,
        "required_commands_before_claiming_parity": [
            "python3 scripts/stepfun_oracle_evidence_consistency_check.py --status-only",
            "python3 scripts/stepfun_oracle_blocker_diagnosis.py --status-only",
            "python3 scripts/stepfun_correctness_status.py --blocked-gates-joined-only",
            "bash -lc 'set -euo pipefail; step_tests=$(find tests -maxdepth 1 -name \"test_stepfun_*.py\" -print | sort | tr \"\\n\" \" \" ); python3 -m compileall -q hipengine tests scripts; python3 -m pytest -q tests/test_gfx1151_backend.py tests/test_gguf_reader.py tests/test_model_quant_and_imports.py ${step_tests}; python3 scripts/check_fixtures.py'",
        ],
        "acceptance_to_clear_oracle_parity": [
            "canonical generated_text_matches_target evidence passes",
            "oracle_parity_ready becomes true in stepfun_correctness_status",
            "retained oracle/status/handoff artifacts are refreshed and verify as match",
            "full StepFun guard passes",
            "WORKLOG and docs/STEPFUN.md record the new evidence without performance claims",
        ],
        "non_goals_for_next_action": [
            "KV-backed decode wiring",
            "e2e generation readiness claim",
            "StepFun throughput or latency claim",
            "NVFP4, vision, or MTP work",
        ],
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This manifest describes the next investigation target; it does not "
                "resolve generated_text_matches_target."
            ),
        },
    }


def verify_oracle_next_action_manifest(
    manifest_artifact: Path,
    *,
    current_manifest: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted oracle next-action manifest with current metadata."""

    persisted = _load_json_object(manifest_artifact)
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_manifest)
    failures: list[dict[str, object]] = []
    if persisted != current_manifest:
        failures.append(
            {
                "name": "oracle_next_action_manifest_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted oracle next-action manifest differs from current "
                    "diagnosis/consistency/logit-margin metadata."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "artifact_path": str(manifest_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status"),
        "current_status": current_manifest.get("status"),
        "persisted_investigation_ready": persisted.get("investigation_ready"),
        "current_investigation_ready": current_manifest.get("investigation_ready"),
        "persisted_next_action": persisted.get("next_action"),
        "current_next_action": current_manifest.get("next_action"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_oracle_next_action_manifest(
        diagnosis_artifact=args.diagnosis_artifact,
        consistency_artifact=args.consistency_artifact,
        host_logit_margin_artifact=args.host_logit_margin_artifact,
        artifact_date=args.artifact_date,
    )
    if args.verify_manifest is not None:
        verification = verify_oracle_next_action_manifest(
            args.verify_manifest,
            current_manifest=report,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        elif args.verification_sha_only:
            payload = status_mod._stable_json_sha256(verification)
        else:
            payload = verification
        output = DEFAULT_OUTPUT if args.default_output else args.output
        status_mod._emit_json(payload, pretty=args.pretty, output=output)
        return (
            0
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    if args.status_only:
        payload: object = report["status"]
    elif args.investigation_ready_only:
        payload = report["investigation_ready"]
    elif args.next_action_only:
        payload = report["next_action"]
    elif args.required_inputs_only:
        payload = report["required_evidence_inputs"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
