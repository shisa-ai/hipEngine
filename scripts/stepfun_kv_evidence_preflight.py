#!/usr/bin/env python3
"""Emit a StepFun KV evidence preflight artifact.

The preflight answers a narrow handoff question: do the currently retained
metadata-only artifacts satisfy the two required KV-backed decode evidence files?
It does not run checkers against missing files, launch kernels, or generate a
token. It is blocker evidence only.
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
from scripts import stepfun_kv_blocker_status as kv_blocker
from scripts import stepfun_kv_session_contract as session_contract

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-kv-evidence-preflight.json"
)
DEFAULT_TRACE_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json"
)
DEFAULT_NEXT_TOKEN_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-contract-artifact",
        type=Path,
        default=session_contract.DEFAULT_OUTPUT,
        help="Retained StepFun KV session-contract artifact.",
    )
    parser.add_argument(
        "--kv-blocker-artifact",
        type=Path,
        default=kv_blocker.DEFAULT_OUTPUT,
        help="Retained StepFun KV blocker-status artifact.",
    )
    parser.add_argument(
        "--trace-artifact",
        type=Path,
        default=DEFAULT_TRACE_ARTIFACT,
        help="Required KV kernel-trace artifact path.",
    )
    parser.add_argument(
        "--next-token-artifact",
        type=Path,
        default=DEFAULT_NEXT_TOKEN_ARTIFACT,
        help="Required KV-backed next-token artifact path.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the artifact.",
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
        "--missing-paths-only",
        action="store_true",
        help="Emit only missing required artifact paths.",
    )
    parser.add_argument(
        "--next-action-only",
        action="store_true",
        help="Emit only next_action.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    parser.add_argument(
        "--verify-preflight",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted KV evidence preflight artifact with current "
            f"session/blocker/missing-artifact metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-preflight, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-preflight, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-preflight, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _contract_summary(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    if payload is None:
        return {
            "artifact": str(path),
            "present": False,
            "ready": False,
            "missing_evidence": ["session_contract_artifact_present"],
        }
    contract = payload.get("contract")
    contract_obj = contract if isinstance(contract, dict) else {}
    ready = (
        payload.get("status") == "blocked"
        and contract_obj.get("blocked_by") == "streaming_decode_loop_not_wired"
        and contract_obj.get("no_kernel_launches") is True
        and contract_obj.get("launch_operation_count") == 135
    )
    missing: list[str] = []
    if payload.get("status") != "blocked":
        missing.append("session_contract_status_blocked")
    if contract_obj.get("blocked_by") != "streaming_decode_loop_not_wired":
        missing.append("session_contract_blocked_by_streaming_loop")
    if contract_obj.get("no_kernel_launches") is not True:
        missing.append("session_contract_no_kernel_launches_true")
    if contract_obj.get("launch_operation_count") != 135:
        missing.append("session_contract_launch_operation_count_135")
    return {
        "artifact": str(path),
        "artifact_sha256": status_mod._stable_json_sha256(payload),
        "present": True,
        "ready": ready,
        "missing_evidence": missing,
        "status": payload.get("status"),
        "blocked_by": contract_obj.get("blocked_by"),
        "next_action": contract_obj.get("next_action"),
        "no_kernel_launches": contract_obj.get("no_kernel_launches"),
        "launch_operation_count": contract_obj.get("launch_operation_count"),
        "required_artifacts": contract_obj.get("required_artifacts"),
    }


def _kv_blocker_summary(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    if payload is None:
        return {
            "artifact": str(path),
            "present": False,
            "ready": False,
            "missing_evidence": ["kv_blocker_artifact_present"],
        }
    ready = payload.get("status") == "blocked" and payload.get("blocked_count") == 2
    return {
        "artifact": str(path),
        "artifact_sha256": status_mod._stable_json_sha256(payload),
        "present": True,
        "ready": ready,
        "missing_evidence": [] if ready else ["kv_blocker_status_blocked_count_2"],
        "status": payload.get("status"),
        "blocked_count": payload.get("blocked_count"),
        "missing_artifact_paths": payload.get("missing_artifact_paths"),
        "streaming_blocked_by": (
            payload.get("streaming_runner_source_status", {})
            if isinstance(payload.get("streaming_runner_source_status"), dict)
            else {}
        ).get("blocked_by"),
        "runtime_symbols_validated": (
            payload.get("runtime_wiring_symbol_validation", {})
            if isinstance(payload.get("runtime_wiring_symbol_validation"), dict)
            else {}
        ).get("all_symbols_present"),
    }


def _required_file_check(path: Path, *, checker_command: str) -> dict[str, object]:
    present = path.exists()
    return {
        "path": str(path),
        "present": present,
        "ready": present,
        "missing_evidence": [] if present else ["artifact_file_present"],
        "checker_command": checker_command,
        "checker_command_sha256": status_mod._stable_json_sha256(checker_command),
    }


def build_kv_evidence_preflight(
    *,
    session_contract_artifact: Path = session_contract.DEFAULT_OUTPUT,
    kv_blocker_artifact: Path = kv_blocker.DEFAULT_OUTPUT,
    trace_artifact: Path = DEFAULT_TRACE_ARTIFACT,
    next_token_artifact: Path = DEFAULT_NEXT_TOKEN_ARTIFACT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return a compact preflight report for the missing KV evidence."""

    contract = _contract_summary(session_contract_artifact)
    blocker = _kv_blocker_summary(kv_blocker_artifact)
    trace = _required_file_check(
        trace_artifact,
        checker_command=(
            "python3 scripts/stepfun_kv_trace_check.py --trace "
            f"{trace_artifact} --resource-artifact {status_mod.DEFAULT_RESOURCE_ARTIFACT} "
            "--summary-only --fail-on-missing --pretty"
        ),
    )
    next_token = _required_file_check(
        next_token_artifact,
        checker_command=(
            "python3 scripts/stepfun_kv_next_token_check.py --artifact "
            f"{next_token_artifact} --prompt-artifact {status_mod.DEFAULT_PROMPT_ARTIFACT} "
            "--summary-only --fail-on-missing --pretty"
        ),
    )
    missing_paths = [
        record["path"]
        for record in (trace, next_token)
        if record.get("present") is not True
    ]
    metadata_ready = bool(contract.get("ready")) and bool(blocker.get("ready"))
    required_artifacts_present = not missing_paths
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_kv_evidence_preflight",
        "date": artifact_date,
        "status": "blocked" if missing_paths else "ready_for_checkers",
        "metadata_contract_ready": metadata_ready,
        "required_artifacts_present": required_artifacts_present,
        "kv_backed_decode_ready": False,
        "next_action": (
            "wire_streaming_decode_loop"
            if missing_paths
            else "run_kv_trace_and_next_token_checkers"
        ),
        "missing_required_artifact_paths": missing_paths,
        "session_contract": contract,
        "kv_blocker_status": blocker,
        "required_artifact_checks": [trace, next_token],
        "generator_commands": {
            "session_contract": (
                "python3 scripts/stepfun_kv_session_contract.py --default-output --pretty"
            ),
            "kv_blocker_status": (
                "python3 scripts/stepfun_kv_blocker_status.py --default-output --pretty"
            ),
        },
        "no_claim_policy": {
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This preflight proves only that metadata-only contract evidence exists; "
                "the required KV trace and KV-backed next-token artifacts are still absent."
            ),
        },
    }


def verify_kv_evidence_preflight(
    preflight_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted KV evidence preflight artifact with current metadata."""

    persisted = json.loads(preflight_artifact.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "kv_evidence_preflight_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted KV evidence preflight artifact differs from current "
                    "session-contract/KV-blocker/missing-artifact metadata."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "artifact_path": str(preflight_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status")
        if isinstance(persisted, dict)
        else None,
        "current_status": current_report.get("status"),
        "persisted_missing_required_artifact_paths": persisted.get(
            "missing_required_artifact_paths"
        )
        if isinstance(persisted, dict)
        else None,
        "current_missing_required_artifact_paths": current_report.get(
            "missing_required_artifact_paths"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_kv_evidence_preflight(
        session_contract_artifact=args.session_contract_artifact,
        kv_blocker_artifact=args.kv_blocker_artifact,
        trace_artifact=args.trace_artifact,
        next_token_artifact=args.next_token_artifact,
        artifact_date=args.artifact_date,
    )
    if args.verify_preflight is not None:
        verification = verify_kv_evidence_preflight(
            args.verify_preflight,
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
    elif args.missing_paths_only:
        payload = report["missing_required_artifact_paths"]
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
