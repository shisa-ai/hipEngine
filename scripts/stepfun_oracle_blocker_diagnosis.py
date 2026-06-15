#!/usr/bin/env python3
"""Emit a consolidated StepFun oracle-parity blocker diagnosis artifact.

The diagnosis reads retained oracle evidence artifacts and separates explanations
already ruled out (prompt-token drift, host top-token text-label drift, expected
next-token text tokenization drift) from the active generated-text mismatch. It
is handoff/blocker evidence only and does not claim oracle parity, KV readiness,
e2e readiness, or performance.
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
from scripts import stepfun_oracle_backend_matrix as backend_matrix
from scripts import stepfun_oracle_rank_check as rank_check
from scripts import stepfun_oracle_token_mismatch as token_mismatch
from scripts import stepfun_prompt_token_roundtrip as prompt_roundtrip
from scripts import stepfun_top_token_roundtrip as top_roundtrip

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-blocker-diagnosis.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-matrix-artifact",
        type=Path,
        default=backend_matrix.DEFAULT_OUTPUT,
        help="Retained oracle backend matrix artifact.",
    )
    parser.add_argument(
        "--token-mismatch-artifact",
        type=Path,
        default=token_mismatch.DEFAULT_OUTPUT,
        help="Retained oracle token-mismatch artifact.",
    )
    parser.add_argument(
        "--rank-check-artifact",
        type=Path,
        default=rank_check.DEFAULT_OUTPUT,
        help="Retained oracle rank-check artifact.",
    )
    parser.add_argument(
        "--top-token-roundtrip-artifact",
        type=Path,
        default=top_roundtrip.DEFAULT_OUTPUT,
        help="Retained host top-token text round-trip artifact.",
    )
    parser.add_argument(
        "--prompt-token-roundtrip-artifact",
        type=Path,
        default=prompt_roundtrip.DEFAULT_OUTPUT,
        help="Retained host prompt token round-trip artifact.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the diagnosis artifact.",
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
        "--ruled-out-causes-only",
        action="store_true",
        help="Emit only ruled-out cause records.",
    )
    parser.add_argument(
        "--active-findings-only",
        action="store_true",
        help="Emit only active finding records.",
    )
    parser.add_argument(
        "--next-action-only",
        action="store_true",
        help="Emit only the recommended next action string.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the diagnosis payload.",
    )
    parser.add_argument(
        "--verify-diagnosis",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted oracle blocker-diagnosis artifact with current "
            f"backend/token/rank/roundtrip metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-diagnosis, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-diagnosis, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-diagnosis, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _artifact_ref(path: Path, payload: dict[str, object]) -> dict[str, object]:
    return {
        "path": str(path),
        "artifact_kind": payload.get("artifact_kind"),
        "status": payload.get("status"),
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _hip_outcome(backend_payload: dict[str, object]) -> object:
    outcomes = backend_payload.get("backend_outcomes")
    if not isinstance(outcomes, list):
        return None
    for item in outcomes:
        if isinstance(item, dict) and item.get("backend") == "hip":
            return item.get("outcome")
    return None


def build_oracle_blocker_diagnosis(
    *,
    backend_matrix_artifact: Path = backend_matrix.DEFAULT_OUTPUT,
    token_mismatch_artifact: Path = token_mismatch.DEFAULT_OUTPUT,
    rank_check_artifact: Path = rank_check.DEFAULT_OUTPUT,
    top_token_roundtrip_artifact: Path = top_roundtrip.DEFAULT_OUTPUT,
    prompt_token_roundtrip_artifact: Path = prompt_roundtrip.DEFAULT_OUTPUT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the consolidated oracle blocker diagnosis."""

    backend_payload = _load_json_object(backend_matrix_artifact)
    mismatch_payload = _load_json_object(token_mismatch_artifact)
    rank_payload = _load_json_object(rank_check_artifact)
    top_payload = _load_json_object(top_token_roundtrip_artifact)
    prompt_payload = _load_json_object(prompt_token_roundtrip_artifact)
    diagnostic = mismatch_payload.get("tokenization_diagnostic")
    if not isinstance(diagnostic, dict):
        diagnostic = {}

    expected_text_roundtrip = (
        diagnostic.get("expected_next_token_text_single_token_matches_expected_id")
        is True
    )
    prompt_token_roundtrip_ok = (
        prompt_payload.get("prompt_tokenization_matches_host_input_ids") is True
    )
    top_token_roundtrip_ok = top_payload.get("all_top_token_texts_roundtrip") is True
    generated_text_matches = mismatch_payload.get("text_matches_expected_stripped") is True
    generated_token_in_top = rank_payload.get("generated_token_in_host_top_tokens") is True
    generated_first_token_id = rank_payload.get("generated_first_token_id")
    expected_next_token_id = rank_payload.get("expected_next_token_id")
    generated_vs_expected_ok = generated_first_token_id == expected_next_token_id

    ruled_out_causes = [
        {
            "cause": "prompt_token_drift",
            "ruled_out": prompt_token_roundtrip_ok,
            "evidence_artifact": str(prompt_token_roundtrip_artifact),
            "host_input_id_count": prompt_payload.get("host_input_id_count"),
            "llama_token_count": prompt_payload.get("llama_token_count"),
            "first_mismatch": prompt_payload.get("first_mismatch"),
        },
        {
            "cause": "host_top_token_text_label_drift",
            "ruled_out": top_token_roundtrip_ok,
            "evidence_artifact": str(top_token_roundtrip_artifact),
            "host_top_token_ids": top_payload.get("host_top_token_ids"),
            "mismatch_count": top_payload.get("mismatch_count"),
        },
        {
            "cause": "expected_next_token_text_tokenization_drift",
            "ruled_out": expected_text_roundtrip,
            "evidence_artifact": str(token_mismatch_artifact),
            "expected_next_token_id": expected_next_token_id,
            "expected_next_token_text": mismatch_payload.get("expected_next_token_text"),
            "expected_next_token_text_token_ids": diagnostic.get(
                "expected_next_token_text_token_ids"
            ),
        },
    ]
    active_findings = [
        {
            "finding": "generated_text_mismatch",
            "active": not generated_text_matches,
            "evidence_artifact": str(token_mismatch_artifact),
            "missing_evidence": mismatch_payload.get("missing_evidence"),
            "expected_next_token_id": expected_next_token_id,
            "expected_next_token_text": mismatch_payload.get("expected_next_token_text"),
            "generated_text": mismatch_payload.get("generated_text"),
            "generated_text_token_ids": diagnostic.get("generated_text_token_ids"),
            "generated_text_stripped_token_ids": diagnostic.get(
                "generated_text_stripped_token_ids"
            ),
            "generated_first_token_id": generated_first_token_id,
            "generated_first_token_matches_expected_id": generated_vs_expected_ok,
        },
        {
            "finding": "generated_token_absent_from_host_top_tokens",
            "active": not generated_token_in_top,
            "evidence_artifact": str(rank_check_artifact),
            "host_top_token_ids": rank_payload.get("host_top_token_ids"),
            "generated_token_host_rank": rank_payload.get("generated_token_host_rank"),
            "generated_token_in_host_top_tokens": generated_token_in_top,
        },
        {
            "finding": "hip_oracle_timeout",
            "active": _hip_outcome(backend_payload) == "timeout",
            "evidence_artifact": str(backend_matrix_artifact),
            "backend_outcomes": backend_payload.get("backend_outcomes"),
            "note": "HIP oracle is comparison evidence only; canonical executed oracle remains Vulkan.",
        },
    ]
    all_ruled_out = all(record["ruled_out"] is True for record in ruled_out_causes)
    active_blocker_count = sum(1 for record in active_findings if record.get("active") is True)
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_oracle_blocker_diagnosis",
        "date": artifact_date,
        "status": "blocked",
        "ready": False,
        "oracle_parity_ready": False,
        "evidence_artifacts": {
            "backend_matrix": _artifact_ref(backend_matrix_artifact, backend_payload),
            "token_mismatch": _artifact_ref(token_mismatch_artifact, mismatch_payload),
            "rank_check": _artifact_ref(rank_check_artifact, rank_payload),
            "top_token_roundtrip": _artifact_ref(
                top_token_roundtrip_artifact, top_payload
            ),
            "prompt_token_roundtrip": _artifact_ref(
                prompt_token_roundtrip_artifact, prompt_payload
            ),
        },
        "ruled_out_cause_count": sum(
            1 for record in ruled_out_causes if record["ruled_out"] is True
        ),
        "ruled_out_causes": ruled_out_causes,
        "all_tokenizer_prompt_drift_causes_ruled_out": all_ruled_out,
        "active_blocker_count": active_blocker_count,
        "active_findings": active_findings,
        "diagnosis": (
            "prompt/tokenizer drift ruled out; generated-text mismatch remains"
            if all_ruled_out
            else "prompt/tokenizer drift not fully ruled out; generated-text mismatch remains"
        ),
        "next_action": (
            "investigate logits/backend parity for the canonical Vulkan executed oracle; "
            "do not claim oracle parity until generated_text_matches_target passes"
        ),
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This diagnosis consolidates retained evidence and explicitly leaves "
                "generated_text_matches_target unresolved."
            ),
        },
    }


def verify_oracle_blocker_diagnosis(
    diagnosis_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted oracle blocker-diagnosis artifact with current metadata."""

    persisted = _load_json_object(diagnosis_artifact)
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "oracle_blocker_diagnosis_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted oracle blocker-diagnosis artifact differs from "
                    "current backend/token/rank/roundtrip metadata."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "artifact_path": str(diagnosis_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status"),
        "current_status": current_report.get("status"),
        "persisted_ruled_out_cause_count": persisted.get("ruled_out_cause_count"),
        "current_ruled_out_cause_count": current_report.get("ruled_out_cause_count"),
        "persisted_active_blocker_count": persisted.get("active_blocker_count"),
        "current_active_blocker_count": current_report.get("active_blocker_count"),
        "persisted_next_action": persisted.get("next_action"),
        "current_next_action": current_report.get("next_action"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_oracle_blocker_diagnosis(
        backend_matrix_artifact=args.backend_matrix_artifact,
        token_mismatch_artifact=args.token_mismatch_artifact,
        rank_check_artifact=args.rank_check_artifact,
        top_token_roundtrip_artifact=args.top_token_roundtrip_artifact,
        prompt_token_roundtrip_artifact=args.prompt_token_roundtrip_artifact,
        artifact_date=args.artifact_date,
    )
    if args.verify_diagnosis is not None:
        verification = verify_oracle_blocker_diagnosis(
            args.verify_diagnosis,
            current_report=report,
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
    elif args.ruled_out_causes_only:
        payload = report["ruled_out_causes"]
    elif args.active_findings_only:
        payload = report["active_findings"]
    elif args.next_action_only:
        payload = report["next_action"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
