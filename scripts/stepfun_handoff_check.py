#!/usr/bin/env python3
"""Verify StepFun GGUF correctness-status and final-blocker handoff artifacts."""

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

DEFAULT_MANIFEST_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-final-blocker-manifest.json"
)
DEFAULT_HANDOFF_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-handoff-check.json"
)
HANDOFF_CHECK_SCHEMA_VERSION = 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Prompt/layer-prefix artifact used to rebuild current status.",
    )
    parser.add_argument(
        "--oracle-artifact",
        type=Path,
        default=status_mod.DEFAULT_ORACLE_ARTIFACT,
        help="llama.cpp oracle artifact used to rebuild current status.",
    )
    parser.add_argument(
        "--resource-artifact",
        type=Path,
        default=status_mod.DEFAULT_RESOURCE_ARTIFACT,
        help="StepFun text-resource dry-run artifact used to rebuild current status.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=status_mod.DEFAULT_DOCS_PATH,
        help="docs/STEPFUN.md checklist source used to rebuild current status.",
    )
    parser.add_argument(
        "--status-artifact",
        type=Path,
        default=status_mod.DEFAULT_STATUS_ARTIFACT,
        help="Persisted consolidated correctness-status artifact to verify.",
    )
    parser.add_argument(
        "--manifest-artifact",
        type=Path,
        default=DEFAULT_MANIFEST_ARTIFACT,
        help="Persisted final-blocker manifest artifact to verify.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON output to this path instead of stdout.",
    )
    parser.add_argument(
        "--verify-handoff-report",
        type=Path,
        nargs="?",
        const=DEFAULT_HANDOFF_ARTIFACT,
        default=None,
        help=(
            "Compare a persisted handoff-check report with current inputs. "
            f"If no path is supplied, uses {DEFAULT_HANDOFF_ARTIFACT}."
        ),
    )
    parser.add_argument(
        "--report-verification-status-only",
        action="store_true",
        help="With --verify-handoff-report, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--report-verification-failures-only",
        action="store_true",
        help="With --verify-handoff-report, emit only report verification failures.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit only the compact handoff summary.",
    )
    parser.add_argument(
        "--summary-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the compact handoff summary.",
    )
    parser.add_argument(
        "--artifact-verification-only",
        action="store_true",
        help="Emit only compact status/manifest artifact verification state.",
    )
    parser.add_argument(
        "--artifact-verification-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of artifact verification state.",
    )
    parser.add_argument(
        "--readiness-summary-only",
        action="store_true",
        help="Emit only the compact readiness summary from the verified handoff.",
    )
    parser.add_argument(
        "--readiness-summary-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the readiness summary.",
    )
    parser.add_argument(
        "--final-blocker-summary-only",
        action="store_true",
        help="Emit only remaining blocker kinds/gates/no-claim policy summary.",
    )
    parser.add_argument(
        "--final-blocker-summary-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of final blocker summary.",
    )
    parser.add_argument(
        "--exit-code-policy-only",
        action="store_true",
        help="Emit only handoff verifier exit-code policy for CI routing.",
    )
    parser.add_argument(
        "--exit-code-policy-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of exit-code policy.",
    )
    parser.add_argument(
        "--digest-summary-only",
        action="store_true",
        help="Emit only the compact digest summary for verified handoff drift polling.",
    )
    parser.add_argument(
        "--digest-summary-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the compact digest summary.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Emit only the handoff verification status string.",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Emit only handoff verification failures.",
    )
    parser.add_argument(
        "--failures-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of handoff verification failures.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when artifacts verify but readiness remains blocked.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args(argv)


def build_handoff_check(
    *,
    prompt_artifact: Path,
    oracle_artifact: Path,
    resource_artifact: Path,
    docs: Path,
    status_artifact: Path,
    manifest_artifact: Path,
) -> dict[str, object]:
    """Return a combined verification report for StepFun handoff artifacts."""

    current_status = status_mod.build_status(
        prompt_artifact,
        oracle_artifact,
        docs,
        resource_artifact=resource_artifact,
    )
    current_manifest = manifest_mod.build_final_blocker_manifest(current_status)
    status_verification = status_mod._verify_source_artifacts(status_artifact)
    manifest_verification = manifest_mod.verify_final_blocker_manifest(
        manifest_artifact,
        current_manifest=current_manifest,
    )
    failures: list[dict[str, object]] = []
    if status_verification.get("all_match") is not True:
        failures.append(
            {
                "name": "correctness_status_artifact_mismatch",
                "status": status_verification.get("status"),
                "verification_failures": status_verification.get(
                    "verification_failures"
                ),
            }
        )
    if manifest_verification.get("all_match") is not True:
        failures.append(
            {
                "name": "final_blocker_manifest_mismatch",
                "status": manifest_verification.get("status"),
                "verification_failures": manifest_verification.get(
                    "verification_failures"
                ),
            }
        )

    remaining_blocker_count = current_manifest.get("remaining_blocker_count")
    remaining_blocker_kinds = list(current_manifest.get("remaining_blocker_kinds", []))
    blocked_gates = list(current_manifest.get("blocked_gates", []))
    readiness_summary = dict(current_status.get("readiness_summary", {}))
    ready = bool(readiness_summary.get("ready"))
    verified = not failures
    blocked_verified = verified and not ready
    status = (
        "ready"
        if verified and ready
        else "blocked_verified"
        if blocked_verified
        else "mismatch"
    )
    artifact_verification = {
        "schema_version": HANDOFF_CHECK_SCHEMA_VERSION,
        "status": "match" if verified else "mismatch",
        "all_match": verified,
        "correctness_status": {
            "artifact": str(status_artifact),
            "status": status_verification.get("status"),
            "all_match": status_verification.get("all_match"),
            "source_artifacts_all_match": status_verification.get(
                "source_artifacts_all_match"
            ),
            "checked_count": status_verification.get("checked_count"),
            "verification_failures_sha256": status_verification.get(
                "verification_failures_sha256"
            ),
        },
        "final_blocker_manifest": {
            "artifact": str(manifest_artifact),
            "status": manifest_verification.get("status"),
            "all_match": manifest_verification.get("all_match"),
            "persisted_manifest_sha256": manifest_verification.get(
                "persisted_manifest_sha256"
            ),
            "current_manifest_sha256": manifest_verification.get(
                "current_manifest_sha256"
            ),
            "verification_failure_count": manifest_verification.get(
                "verification_failure_count"
            ),
        },
    }
    final_blocker_manifest_summary = {
        "remaining_blocker_count": remaining_blocker_count,
        "remaining_blocker_kinds": remaining_blocker_kinds,
        "blocked_gates": blocked_gates,
        "no_claim_policy": current_manifest.get("no_claim_policy"),
    }
    exit_code_policy = {
        "schema_version": HANDOFF_CHECK_SCHEMA_VERSION,
        "ready": status_mod.READY_EXIT_CODE,
        "mismatch": status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
        "blocked_verified_with_fail_on_blocked": status_mod.BLOCKED_EXIT_CODE,
        "current_without_fail_on_blocked": status_mod.READY_EXIT_CODE
        if status != "mismatch"
        else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
        "current_with_fail_on_blocked": status_mod.BLOCKED_EXIT_CODE
        if status == "blocked_verified"
        else status_mod.READY_EXIT_CODE
        if status == "ready"
        else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
        "status": status,
        "verified": verified,
        "ready_status": ready,
        "fail_on_blocked_option": "--fail-on-blocked",
    }
    summary = {
        "schema_version": HANDOFF_CHECK_SCHEMA_VERSION,
        "status": status,
        "verified": verified,
        "ready": ready,
        "blocked_verified": blocked_verified,
        "remaining_blocker_count": remaining_blocker_count,
        "remaining_blocker_kinds": remaining_blocker_kinds,
        "blocked_gates": blocked_gates,
        "open_or_partial_items_p0_p12": current_manifest.get(
            "open_or_partial_items_p0_p12"
        ),
        "status_artifact": str(status_artifact),
        "manifest_artifact": str(manifest_artifact),
        "source_artifacts_sha256": current_status.get("source_artifacts_sha256"),
        "manifest_sha256": status_mod._stable_json_sha256(current_manifest),
        "verification_failure_count": len(failures),
    }
    digest_summary = {
        "schema_version": HANDOFF_CHECK_SCHEMA_VERSION,
        "status": status,
        "summary_sha256": status_mod._stable_json_sha256(summary),
        "artifact_verification_sha256": status_mod._stable_json_sha256(
            artifact_verification
        ),
        "readiness_summary_sha256": status_mod._stable_json_sha256(
            readiness_summary
        ),
        "final_blocker_manifest_summary_sha256": status_mod._stable_json_sha256(
            final_blocker_manifest_summary
        ),
        "exit_code_policy_sha256": status_mod._stable_json_sha256(exit_code_policy),
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "source_artifacts_sha256": current_status.get("source_artifacts_sha256"),
        "manifest_sha256": status_mod._stable_json_sha256(current_manifest),
    }
    return {
        "schema_version": HANDOFF_CHECK_SCHEMA_VERSION,
        "status": status,
        "summary": summary,
        "summary_sha256": status_mod._stable_json_sha256(summary),
        "artifact_verification": artifact_verification,
        "artifact_verification_sha256": status_mod._stable_json_sha256(
            artifact_verification
        ),
        "readiness_summary_sha256": status_mod._stable_json_sha256(
            readiness_summary
        ),
        "final_blocker_manifest_summary": final_blocker_manifest_summary,
        "final_blocker_manifest_summary_sha256": status_mod._stable_json_sha256(
            final_blocker_manifest_summary
        ),
        "exit_code_policy": exit_code_policy,
        "exit_code_policy_sha256": status_mod._stable_json_sha256(exit_code_policy),
        "digest_summary": digest_summary,
        "digest_summary_sha256": status_mod._stable_json_sha256(digest_summary),
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "correctness_status_verification": status_verification,
        "final_blocker_manifest_verification": manifest_verification,
        "readiness_summary": readiness_summary,
    }


def verify_handoff_report(
    report_path: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted handoff-check report with the current report."""

    persisted = json.loads(report_path.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "handoff_report_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted handoff-check report differs from current "
                    "prompt/oracle/resource/docs/status/manifest inputs."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": HANDOFF_CHECK_SCHEMA_VERSION,
        "report_path": str(report_path),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_report_sha256": persisted_sha256,
        "current_report_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status")
        if isinstance(persisted, dict)
        else None,
        "current_status": current_report.get("status"),
        "persisted_summary_sha256": persisted.get("summary_sha256")
        if isinstance(persisted, dict)
        else None,
        "current_summary_sha256": current_report.get("summary_sha256"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_handoff_check(
        prompt_artifact=args.prompt_artifact,
        oracle_artifact=args.oracle_artifact,
        resource_artifact=args.resource_artifact,
        docs=args.docs,
        status_artifact=args.status_artifact,
        manifest_artifact=args.manifest_artifact,
    )
    if args.verify_handoff_report is not None:
        verification = verify_handoff_report(
            args.verify_handoff_report,
            current_report=report,
        )
        if args.report_verification_status_only:
            payload: object = verification["status"]
        elif args.report_verification_failures_only:
            payload = verification["verification_failures"]
        else:
            payload = verification
        status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
        if verification["all_match"] is not True:
            return status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        if args.fail_on_blocked and report["status"] == "blocked_verified":
            return status_mod.BLOCKED_EXIT_CODE
        return status_mod.READY_EXIT_CODE
    if args.status_only:
        payload: object = report["status"]
    elif args.artifact_verification_sha_only:
        payload = report["artifact_verification_sha256"]
    elif args.artifact_verification_only:
        payload = report["artifact_verification"]
    elif args.readiness_summary_sha_only:
        payload = report["readiness_summary_sha256"]
    elif args.readiness_summary_only:
        payload = report["readiness_summary"]
    elif args.final_blocker_summary_sha_only:
        payload = report["final_blocker_manifest_summary_sha256"]
    elif args.final_blocker_summary_only:
        payload = report["final_blocker_manifest_summary"]
    elif args.exit_code_policy_sha_only:
        payload = report["exit_code_policy_sha256"]
    elif args.exit_code_policy_only:
        payload = report["exit_code_policy"]
    elif args.digest_summary_sha_only:
        payload = report["digest_summary_sha256"]
    elif args.digest_summary_only:
        payload = report["digest_summary"]
    elif args.failures_sha_only:
        payload = report["verification_failures_sha256"]
    elif args.failures_only:
        payload = report["verification_failures"]
    elif args.summary_sha_only:
        payload = report["summary_sha256"]
    elif args.summary_only:
        payload = report["summary"]
    else:
        payload = report
    status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
    if report["status"] == "mismatch":
        return status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    if args.fail_on_blocked and report["status"] == "blocked_verified":
        return status_mod.BLOCKED_EXIT_CODE
    return status_mod.READY_EXIT_CODE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
