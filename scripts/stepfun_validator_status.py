#!/usr/bin/env python3
"""Aggregate StepFun final-blocker validator status.

This script reads the final-blocker manifest (or builds the current one), then
checks the concrete artifact paths for the oracle, KV trace, and KV next-token
validators without shelling out. Missing artifacts are reported as ``missing``;
present artifacts are validated by importing the dedicated checker modules.
Passing this script does not make a performance claim.
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
from scripts import stepfun_final_blocker_manifest as manifest_mod
from scripts import stepfun_kv_next_token_check as kv_next_token_check_mod
from scripts import stepfun_kv_trace_check as kv_trace_check_mod
from scripts import stepfun_oracle_artifact_check as oracle_check_mod

VALIDATOR_STATUS_SCHEMA_VERSION = 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Use this final-blocker manifest JSON instead of building the current one.",
    )
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Canonical prompt artifact used by oracle and KV token validators.",
    )
    parser.add_argument(
        "--oracle-artifact",
        type=Path,
        default=status_mod.DEFAULT_ORACLE_ARTIFACT,
        help="llama.cpp oracle artifact used when building the current manifest.",
    )
    parser.add_argument(
        "--resource-artifact",
        type=Path,
        default=status_mod.DEFAULT_RESOURCE_ARTIFACT,
        help="Text-resource artifact used by the KV trace validator.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=status_mod.DEFAULT_DOCS_PATH,
        help="docs/STEPFUN.md checklist source used when building the current manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON output atomically to this path instead of stdout.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit only the compact validator_status_summary payload.",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Emit only the per-validator result records.",
    )
    parser.add_argument(
        "--results-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of per-validator results.",
    )
    parser.add_argument(
        "--blocked-only",
        action="store_true",
        help="Emit only failed or missing validator result records.",
    )
    parser.add_argument(
        "--blocked-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of failed/missing validator records.",
    )
    parser.add_argument(
        "--next-blocker-only",
        action="store_true",
        help="Emit only the first failed or missing validator record, or null.",
    )
    parser.add_argument(
        "--next-blocker-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the first failed/missing validator record.",
    )
    parser.add_argument(
        "--next-command-only",
        action="store_true",
        help="Emit only the concrete validator command for the first failed/missing record, or null.",
    )
    parser.add_argument(
        "--next-command-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the next concrete validator command.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the full report or compact summary.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Emit only passed/blocked status.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when any validator is missing or failed.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args(argv)


def _load_manifest(path: Path | None, *, prompt_artifact: Path, oracle_artifact: Path, resource_artifact: Path, docs: Path) -> dict[str, object]:
    if path is not None:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return payload
    status = status_mod.build_status(
        prompt_artifact,
        oracle_artifact,
        docs,
        resource_artifact=resource_artifact,
    )
    return manifest_mod.build_final_blocker_manifest(status)


def _path_from_record(record: dict[str, object]) -> Path | None:
    path_value = record.get("validator_artifact_path") or record.get("path")
    if path_value in (None, ""):
        return None
    return Path(str(path_value))


def _summary_from_report(kind: str, report: dict[str, object]) -> dict[str, object]:
    if kind == "oracle_artifact_check_command":
        return dict(report.get("oracle_summary", {}))
    if kind == "kv_trace_check_command":
        return dict(report.get("trace_summary", {}))
    if kind == "kv_next_token_check_command":
        return dict(report.get("next_token_summary", {}))
    return {}


def _run_validator(
    record: dict[str, object],
    *,
    prompt_artifact: Path,
    resource_artifact: Path,
) -> dict[str, object]:
    kind = str(record.get("validator_command_kind") or "")
    artifact_path = _path_from_record(record)
    base = {
        "artifact_name": record.get("artifact_name") or record.get("name"),
        "readiness_gate": record.get("readiness_gate"),
        "required_for": record.get("required_for"),
        "validator_command_kind": kind,
        "validator_artifact_path": str(artifact_path) if artifact_path is not None else None,
        "validator_command_concrete": record.get("validator_command_concrete"),
        "validator_command_concrete_sha256": record.get("validator_command_concrete_sha256"),
    }
    if artifact_path is None:
        return {
            **base,
            "status": "missing",
            "ready": False,
            "reason": "validator_artifact_path_missing",
        }
    if not artifact_path.exists():
        return {
            **base,
            "status": "missing",
            "ready": False,
            "reason": "artifact_file_missing",
        }
    try:
        if kind == "oracle_artifact_check_command":
            report = oracle_check_mod.build_oracle_check_report(
                artifact_path,
                prompt_artifact=prompt_artifact,
            )
        elif kind == "kv_trace_check_command":
            report = kv_trace_check_mod.build_trace_check_report(
                artifact_path,
                resource_artifact=resource_artifact,
            )
        elif kind == "kv_next_token_check_command":
            report = kv_next_token_check_mod.build_next_token_check_report(
                artifact_path,
                prompt_artifact=prompt_artifact,
            )
        else:
            return {
                **base,
                "status": "failed",
                "ready": False,
                "reason": "unknown_validator_command_kind",
            }
    except Exception as exc:  # pragma: no cover - exercised by malformed-user inputs
        return {
            **base,
            "status": "failed",
            "ready": False,
            "reason": "validator_exception",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
    summary = _summary_from_report(kind, report)
    status = str(report.get("status") or "failed")
    ready = status == "passed"
    return {
        **base,
        "status": "passed" if ready else "failed",
        "ready": ready,
        "reason": None if ready else "validator_report_failed",
        "validator_report_sha256": report.get("report_sha256"),
        "validator_summary_sha256": status_mod._stable_json_sha256(summary),
        "validator_missing_evidence": summary.get("missing_evidence"),
        "validator_missing_evidence_count": summary.get("missing_evidence_count"),
    }


def build_validator_status_report(
    manifest: dict[str, object],
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    resource_artifact: Path = status_mod.DEFAULT_RESOURCE_ARTIFACT,
) -> dict[str, object]:
    """Return aggregate status for concrete final-blocker validators."""

    validator_commands = manifest.get("validator_commands_handoff", [])
    if not isinstance(validator_commands, list):
        validator_commands = []
    results = [
        _run_validator(
            dict(record) if isinstance(record, dict) else {},
            prompt_artifact=prompt_artifact,
            resource_artifact=resource_artifact,
        )
        for record in validator_commands
    ]
    blocked_results = [
        record for record in results if record.get("status") in {"missing", "failed"}
    ]
    next_blocker = blocked_results[0] if blocked_results else None
    next_blocker_command = (
        next_blocker.get("validator_command_concrete")
        if isinstance(next_blocker, dict)
        else None
    )
    passed = sum(1 for record in results if record.get("status") == "passed")
    missing = sum(1 for record in results if record.get("status") == "missing")
    failed = sum(1 for record in results if record.get("status") == "failed")
    ready = bool(results) and passed == len(results)
    summary = {
        "schema_version": VALIDATOR_STATUS_SCHEMA_VERSION,
        "status": "passed" if ready else "blocked",
        "ready": ready,
        "validator_count": len(results),
        "passed_count": passed,
        "missing_count": missing,
        "failed_count": failed,
        "validator_command_kinds": [
            record.get("validator_command_kind") for record in results
        ],
        "validator_artifact_paths": [
            record.get("validator_artifact_path") for record in results
        ],
        "validator_results_sha256": status_mod._stable_json_sha256(results),
        "blocked_validator_results_sha256": status_mod._stable_json_sha256(
            blocked_results
        ),
        "next_blocker_artifact_name": next_blocker.get("artifact_name")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_status": next_blocker.get("status")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_reason": next_blocker.get("reason")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_command": next_blocker_command,
        "next_blocker_command_sha256": status_mod._stable_json_sha256(
            next_blocker_command
        ),
        "next_blocker_sha256": status_mod._stable_json_sha256(next_blocker),
        "manifest_sha256": status_mod._stable_json_sha256(manifest),
        "no_claim_policy": {
            "validator_artifacts_passed": ready,
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "Aggregate validator status checks retained evidence artifacts only; "
                "readiness gates must still be updated from real oracle/KV evidence."
            ),
        },
    }
    report = {
        "schema_version": VALIDATOR_STATUS_SCHEMA_VERSION,
        "status": summary["status"],
        "validator_status_summary": summary,
        "validator_status_summary_sha256": status_mod._stable_json_sha256(summary),
        "validator_results": results,
        "validator_results_sha256": status_mod._stable_json_sha256(results),
        "blocked_validator_results": blocked_results,
        "blocked_validator_results_sha256": status_mod._stable_json_sha256(
            blocked_results
        ),
        "next_blocker": next_blocker,
        "next_blocker_sha256": status_mod._stable_json_sha256(next_blocker),
        "next_blocker_command": next_blocker_command,
        "next_blocker_command_sha256": status_mod._stable_json_sha256(
            next_blocker_command
        ),
        "readiness_impact": {
            "validator_artifacts_passed": ready,
            "oracle_parity": False,
            "kv_backed_decode_ready": False,
            "e2e_inference_ready": False,
            "reason": (
                "This report is an aggregate validator execution status; it does not by itself "
                "mark StepFun e2e inference ready."
            ),
        },
    }
    report["report_sha256"] = status_mod._stable_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = _load_manifest(
        args.manifest,
        prompt_artifact=args.prompt_artifact,
        oracle_artifact=args.oracle_artifact,
        resource_artifact=args.resource_artifact,
        docs=args.docs,
    )
    report = build_validator_status_report(
        manifest,
        prompt_artifact=args.prompt_artifact,
        resource_artifact=args.resource_artifact,
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.next_command_sha_only:
        payload = report["next_blocker_command_sha256"]
    elif args.next_command_only:
        payload = report["next_blocker_command"]
    elif args.next_blocker_sha_only:
        payload = report["next_blocker_sha256"]
    elif args.next_blocker_only:
        payload = report["next_blocker"]
    elif args.blocked_sha_only:
        payload = report["blocked_validator_results_sha256"]
    elif args.blocked_only:
        payload = report["blocked_validator_results"]
    elif args.results_sha_only:
        payload = report["validator_results_sha256"]
    elif args.results_only:
        payload = report["validator_results"]
    elif args.sha_only:
        payload = report["validator_status_summary_sha256"] if args.summary_only else report["report_sha256"]
    elif args.summary_only:
        payload = report["validator_status_summary"]
    else:
        payload = report
    status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
    if args.fail_on_blocked and report["status"] != "passed":
        return status_mod.BLOCKED_EXIT_CODE
    return status_mod.READY_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
