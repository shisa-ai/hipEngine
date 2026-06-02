#!/usr/bin/env python3
"""Summarize StepFun Q3_K_L correctness artifacts and remaining blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Sequence

DEFAULT_PROMPT_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json"
)
DEFAULT_ORACLE_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-step35-timeout.json"
)
DEFAULT_ORACLE_WRAPPER_TIMEOUT_ARTIFACT = Path(
    "benchmarks/results/2026-06-01-stepfun-q3kl-llamacpp-step35-180s-wrapper-timeout.json"
)
DEFAULT_RESOURCE_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-text-resource-dry-run.json"
)
DEFAULT_DOCS_PATH = Path("docs/STEPFUN.md")
DEFAULT_STATUS_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json"
)
DEFAULT_ORACLE_LONG_TIMEOUT_S = 900.0
STATUS_SCHEMA_VERSION = 1
HANDOFF_SUMMARY_SCHEMA_VERSION = 1
READINESS_SUMMARY_SCHEMA_VERSION = 1
READY_EXIT_CODE = 0
SOURCE_ARTIFACT_MISMATCH_EXIT_CODE = 1
BLOCKED_EXIT_CODE = 2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-artifact", type=Path, default=DEFAULT_PROMPT_ARTIFACT)
    parser.add_argument("--oracle-artifact", type=Path, default=DEFAULT_ORACLE_ARTIFACT)
    parser.add_argument("--resource-artifact", type=Path, default=DEFAULT_RESOURCE_ARTIFACT)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output to this path instead of stdout.")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when the summarized status is not ready.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit only the compact handoff_summary instead of the full status artifact.",
    )
    parser.add_argument(
        "--readiness-summary-only",
        action="store_true",
        help=(
            "Emit only the compact top-level readiness_summary for scheduler polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--readiness-summary-sha-only",
        action="store_true",
        help=(
            "Emit only readiness_summary_sha256 for top-level readiness drift polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--readiness-gates-only",
        action="store_true",
        help=(
            "Emit only readiness_gates for oracle/KV/e2e gate polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--readiness-gates-sha-only",
        action="store_true",
        help=(
            "Emit only readiness_gates_sha256 for oracle/KV/e2e gate drift polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--blocked-gates-only",
        action="store_true",
        help=(
            "Emit only blocked_gates for lightweight readiness gate routing. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--blocked-gates-sha-only",
        action="store_true",
        help=(
            "Emit only blocked_gates_sha256 for blocked readiness gate drift polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--remaining-blockers-report-only",
        action="store_true",
        help=(
            "Emit only remaining_blockers_report for compact final-blocker routing. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--remaining-blockers-report-sha-only",
        action="store_true",
        help=(
            "Emit only remaining_blockers_report_sha256 for compact final-blocker drift polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--first-remaining-blocker-report-only",
        action="store_true",
        help=(
            "Emit only first_remaining_blocker_report for compact front-blocker routing. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--first-remaining-blocker-report-sha-only",
        action="store_true",
        help=(
            "Emit only first_remaining_blocker_report_sha256 for compact front-blocker drift polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--handoff-summary-sha-only",
        action="store_true",
        help=(
            "Emit only handoff_summary_sha256 for blocker handoff drift polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--schema-versions-only",
        action="store_true",
        help=(
            "Emit only schema_versions for compact status/readiness/handoff "
            "contract checks. Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--status-integrity-only",
        action="store_true",
        help=(
            "Emit only status_integrity for compact embedded digest/schema checks. "
            "Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--status-integrity-sha-only",
        action="store_true",
        help=(
            "Emit only status_integrity_sha256 for compact embedded integrity drift polling. "
            "Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--status-integrity-failures-only",
        action="store_true",
        help=(
            "Emit only status_integrity.failed_checks for compact integrity failure routing. "
            "Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--persisted-status-integrity-only",
        action="store_true",
        help=(
            "Emit only persisted_status_integrity for checking that embedded status_integrity "
            "payload/SHA still match recomputed checks. Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--persisted-status-integrity-failures-only",
        action="store_true",
        help=(
            "Emit only persisted_status_integrity.failed_checks for compact embedded integrity "
            "payload/SHA drift routing. Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--oracle-wrapper-timeout-source-only",
        action="store_true",
        help=(
            "Emit only source_artifacts.oracle_wrapper_timeout for compact 180s oracle "
            "wrapper-timeout provenance polling. Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--oracle-wrapper-timeout-source-sha-only",
        action="store_true",
        help=(
            "Emit only source_artifacts.oracle_wrapper_timeout.sha256 for compact 180s oracle "
            "wrapper-timeout provenance drift polling. Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--text-resource-source-only",
        action="store_true",
        help=(
            "Emit only source_artifacts.text_resource for compact KV resource/run-plan "
            "provenance polling. Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--text-resource-source-sha-only",
        action="store_true",
        help=(
            "Emit only source_artifacts.text_resource.sha256 for compact KV resource/run-plan "
            "provenance drift polling. Overrides summary and queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--source-artifacts-sha-only",
        action="store_true",
        help=(
            "Emit only source_artifacts_sha256 for input provenance drift polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--source-artifact-failures-only",
        action="store_true",
        help=(
            "With --verify-source-artifacts, emit only source_artifact_failed_records "
            "for compact file provenance failure routing."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help=(
            "With --verify-source-artifacts, emit only verification status "
            "for compact match/mismatch routing."
        ),
    )
    parser.add_argument(
        "--verification-exit-code-only",
        action="store_true",
        help=(
            "With --verify-source-artifacts, emit only the verifier exit code "
            "for compact numeric routing."
        ),
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help=(
            "With --verify-source-artifacts, emit only verification_failures for compact "
            "source/status failure routing."
        ),
    )
    parser.add_argument(
        "--verification-failures-sha-only",
        action="store_true",
        help=(
            "With --verify-source-artifacts, emit only verification_failures_sha256 "
            "for compact source/status failure drift polling."
        ),
    )
    parser.add_argument(
        "--next-action-commands-only",
        action="store_true",
        help=(
            "Emit only next_action_commands for command handoff polling. Overrides "
            "--summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--next-action-commands-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands_sha256 for command handoff drift polling. "
            "Overrides --summary-only and blocker queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--blocker-kinds-only",
        action="store_true",
        help=(
            "Emit only blocker_kinds for lightweight blocker kind polling. "
            "Overrides --summary-only and readiness compact-output modes."
        ),
    )
    parser.add_argument(
        "--blocker-kinds-sha-only",
        action="store_true",
        help=(
            "Emit only blocker_kinds_sha256 for blocker kind drift polling. "
            "Overrides --summary-only and readiness compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-streaming-blockers-only",
        action="store_true",
        help=(
            "Emit only kv_streaming_runner_blocker_names for runtime-specific KV blocker polling. "
            "Overrides --summary-only and readiness compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-streaming-blockers-sha-only",
        action="store_true",
        help=(
            "Emit only kv_streaming_runner_blocker_names_sha256 for KV blocker drift polling. "
            "Overrides --summary-only and readiness compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-first-streaming-blocker-only",
        action="store_true",
        help=(
            "Emit only kv_backed_decode_gap_report.first_streaming_runner_blocker for "
            "routing the next KV-backed decode implementation task. Overrides --summary-only "
            "and readiness compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-first-streaming-blocker-sha-only",
        action="store_true",
        help=(
            "Emit only the SHA-256 digest of kv_backed_decode_gap_report.first_streaming_runner_blocker "
            "for first-KV-blocker drift polling. Overrides --summary-only and readiness "
            "compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-streaming-blueprint-only",
        action="store_true",
        help=(
            "Emit only kv_backed_decode_gap_report.streaming_decode_loop_blueprint "
            "for KV streaming loop contract handoff. Overrides --summary-only and readiness "
            "compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-streaming-blueprint-sha-only",
        action="store_true",
        help=(
            "Emit only the stable SHA-256 digest of kv_backed_decode_gap_report.streaming_decode_loop_blueprint "
            "for KV streaming loop contract drift polling. Overrides --summary-only and readiness "
            "compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-streaming-loop-status-only",
        action="store_true",
        help=(
            "Emit only kv_backed_decode_gap_report.streaming_decode_loop_status "
            "for KV streaming loop readiness handoff. Overrides --summary-only and readiness "
            "compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-streaming-loop-status-sha-only",
        action="store_true",
        help=(
            "Emit only kv_backed_decode_gap_report.streaming_decode_loop_status_sha256 "
            "for KV streaming loop readiness drift polling. Overrides --summary-only and readiness "
            "compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-streaming-loop-next-action-only",
        action="store_true",
        help=(
            "Emit only kv_backed_decode_gap_report.streaming_decode_loop_status.next_action "
            "for direct KV loop wiring handoff. Overrides --summary-only and readiness "
            "compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-streaming-loop-next-action-sha-only",
        action="store_true",
        help=(
            "Emit only the stable SHA-256 digest of kv_backed_decode_gap_report."
            "streaming_decode_loop_status.next_action for KV loop wiring drift polling. "
            "Overrides --summary-only and readiness compact-output modes."
        ),
    )
    parser.add_argument(
        "--blocker-work-queue-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.blocker_work_queue for lightweight blocker routing. "
            "Overrides --summary-only."
        ),
    )
    parser.add_argument(
        "--blocker-work-queue-meta-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.blocker_work_queue_meta for lightweight queue metadata. "
            "Overrides --summary-only and --blocker-work-queue-only."
        ),
    )
    parser.add_argument(
        "--blocker-work-queue-sha-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.blocker_work_queue_sha256 for queue-drift polling. "
            "Overrides --summary-only, --blocker-work-queue-only, and --blocker-work-queue-meta-only."
        ),
    )
    parser.add_argument(
        "--blocker-recommended-commands-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.blocker_recommended_commands for generic blocker-command "
            "routing. Overrides --summary-only and blocker-work-queue compact modes."
        ),
    )
    parser.add_argument(
        "--blocker-recommended-commands-sha-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.blocker_recommended_commands_sha256 for recommended-command "
            "queue drift polling. Overrides --summary-only and blocker-work-queue compact modes."
        ),
    )
    parser.add_argument(
        "--status-refresh-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.oracle_parity_blocked.status_refresh_command for "
            "consolidated status refresh. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-resource-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.kv_backed_decode_not_wired.resource_plan_refresh_command "
            "for KV resource dry-run refresh. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--kv-resource-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.kv_backed_decode_not_wired.resource_plan_refresh_command_sha256 "
            "for KV resource command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--status-refresh-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.oracle_parity_blocked.status_refresh_command_sha256 "
            "for status-refresh command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--source-verify-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.source_artifacts_verify_command "
            "for source artifact provenance checks. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--source-verify-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.source_artifacts_verify_command_sha256 "
            "for source artifact verification command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verification-status-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.verification_status_command "
            "for compact persisted verification status checks. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verification-status-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.verification_status_command_sha256 "
            "for compact persisted verification status command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verification-exit-code-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.verification_exit_code_command "
            "for compact persisted verification exit-code checks. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verification-exit-code-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.verification_exit_code_command_sha256 "
            "for compact persisted verification exit-code command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verification-failures-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.verification_failures_command "
            "for compact persisted verification failure checks. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verification-failures-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.verification_failures_command_sha256 "
            "for compact persisted verification failure command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verification-failures-sha-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.verification_failures_sha_command "
            "for compact persisted verification failure digest checks. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verification-failures-sha-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.handoff_integrity.verification_failures_sha_command_sha256 "
            "for compact persisted verification failure digest command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--oracle-helper-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.oracle_parity_blocked.oracle_helper_refresh_command "
            "for oracle artifact regeneration. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--oracle-helper-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.oracle_parity_blocked.oracle_helper_refresh_command_sha256 "
            "for oracle helper command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--oracle-helper-long-timeout-command-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.oracle_parity_blocked.oracle_helper_long_timeout_command "
            "for the longer-timeout oracle rerun. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--oracle-helper-long-timeout-command-sha-only",
        action="store_true",
        help=(
            "Emit only next_action_commands.oracle_parity_blocked.oracle_helper_long_timeout_command_sha256 "
            "for longer-timeout oracle command drift polling. Overrides readiness/queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--first-blocker-sha-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.first_blocker_work_item_sha256 for immediate-blocker "
            "drift polling. Overrides queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--first-blocker-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.first_blocker_work_item for immediate routing. "
            "Overrides --summary-only, --blocker-work-queue-only, --blocker-work-queue-meta-only, "
            "--blocker-work-queue-sha-only, and --first-blocker-sha-only."
        ),
    )
    parser.add_argument(
        "--first-blocker-recommended-command-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.first_blocker_work_item.recommended_command for "
            "generic immediate-blocker execution. Overrides queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--first-blocker-recommended-command-sha-only",
        action="store_true",
        help=(
            "Emit only handoff_summary.first_blocker_work_item.recommended_command_sha256 "
            "for generic immediate-blocker command drift polling. Overrides queue compact-output modes."
        ),
    )
    parser.add_argument(
        "--verify-source-artifacts",
        type=Path,
        default=None,
        metavar="STATUS_JSON",
        help="Verify source_artifacts in an existing status JSON against the current filesystem.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _artifact_record(path: Path) -> dict[str, object]:
    """Return stable provenance metadata for an input artifact."""

    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _source_artifacts(
    *,
    prompt_artifact: Path,
    oracle_artifact: Path,
    resource_artifact: Path,
    docs_path: Path,
) -> dict[str, object]:
    artifacts = {
        "prompt": _artifact_record(prompt_artifact),
        "oracle": _artifact_record(oracle_artifact),
        "text_resource": _artifact_record(resource_artifact),
        "docs": _artifact_record(docs_path),
    }
    if (
        prompt_artifact == DEFAULT_PROMPT_ARTIFACT
        and oracle_artifact == DEFAULT_ORACLE_ARTIFACT
        and resource_artifact == DEFAULT_RESOURCE_ARTIFACT
        and docs_path == DEFAULT_DOCS_PATH
    ):
        artifacts["oracle_wrapper_timeout"] = _artifact_record(
            DEFAULT_ORACLE_WRAPPER_TIMEOUT_ARTIFACT
        )
    return artifacts


def _status_integrity(status: dict[str, object]) -> dict[str, object]:
    """Verify embedded status digests/schema versions are self-consistent."""

    source_artifacts = status.get("source_artifacts", {})
    handoff_summary = status.get("handoff_summary", {})
    readiness_summary = status.get("readiness_summary", {})
    readiness_gates = status.get("readiness_gates", {})
    next_action_commands = status.get("next_action_commands", {})
    blocker_kinds = status.get("blocker_kinds", [])
    blocked_gates = status.get("blocked_gates", [])
    kv_gap_report = status.get("kv_backed_decode_gap_report", {})
    kv_streaming_blocker_names = (
        kv_gap_report.get("streaming_runner_blocker_names", [])
        if isinstance(kv_gap_report, dict)
        else []
    )
    kv_streaming_blocker_names_sha256 = (
        kv_gap_report.get("streaming_runner_blocker_names_sha256")
        if isinstance(kv_gap_report, dict)
        else None
    )
    kv_streaming_blocker_names_sha256_match = (
        kv_gap_report.get("streaming_runner_blocker_names_sha256_match")
        if isinstance(kv_gap_report, dict)
        else None
    )
    first_kv_streaming_blocker = (
        kv_gap_report.get("first_streaming_runner_blocker")
        if isinstance(kv_gap_report, dict)
        else None
    )
    first_kv_streaming_blocker_sha256 = (
        kv_gap_report.get("first_streaming_runner_blocker_sha256")
        if isinstance(kv_gap_report, dict)
        else None
    )
    kv_streaming_blueprint = (
        kv_gap_report.get("streaming_decode_loop_blueprint")
        if isinstance(kv_gap_report, dict)
        else None
    )
    kv_streaming_blueprint_sha256 = (
        kv_gap_report.get("streaming_decode_loop_blueprint_sha256")
        if isinstance(kv_gap_report, dict)
        else None
    )
    kv_streaming_loop_status = (
        kv_gap_report.get("streaming_decode_loop_status")
        if isinstance(kv_gap_report, dict)
        else None
    )
    kv_streaming_loop_status_sha256 = (
        kv_gap_report.get("streaming_decode_loop_status_sha256")
        if isinstance(kv_gap_report, dict)
        else None
    )
    kv_streaming_loop_next_action = (
        kv_streaming_loop_status.get("next_action")
        if isinstance(kv_streaming_loop_status, dict)
        else None
    )
    kv_streaming_loop_next_action_sha256 = (
        kv_streaming_loop_status.get("next_action_sha256")
        if isinstance(kv_streaming_loop_status, dict)
        else None
    )
    schema_versions = status.get("schema_versions", {})
    blocker_meta = (
        handoff_summary.get("blocker_work_queue_meta", {})
        if isinstance(handoff_summary, dict)
        else {}
    )
    blocker_recommended_commands = (
        handoff_summary.get("blocker_recommended_commands", [])
        if isinstance(handoff_summary, dict)
        else []
    )
    blocker_recommended_commands_sha256 = (
        handoff_summary.get("blocker_recommended_commands_sha256")
        if isinstance(handoff_summary, dict)
        else None
    )
    kv_streaming_mirror_records: list[dict[str, object]] = []
    if isinstance(kv_gap_report, dict):
        kv_streaming_mirror_records.append(kv_gap_report)
    if isinstance(next_action_commands, dict):
        kv_next_action = next_action_commands.get("kv_backed_decode_not_wired", {})
        if isinstance(kv_next_action, dict):
            kv_streaming_mirror_records.append(kv_next_action)
    if isinstance(handoff_summary, dict):
        handoff_gap_report = handoff_summary.get("kv_backed_decode_gap_report", {})
        if isinstance(handoff_gap_report, dict):
            kv_streaming_mirror_records.append(handoff_gap_report)
        for item in handoff_summary.get("blocker_work_queue", []):
            if isinstance(item, dict) and item.get("blocker_kind") == "kv_backed_decode_not_wired":
                kv_streaming_mirror_records.append(item)
                break
    kv_streaming_runner_blocker_mirrors = (
        bool(kv_streaming_mirror_records)
        and isinstance(kv_streaming_blocker_names, list)
        and kv_streaming_blocker_names_sha256 is not None
        and all(
            record.get("streaming_runner_blocker_names") == kv_streaming_blocker_names
            and record.get("streaming_runner_blocker_names_sha256")
            == kv_streaming_blocker_names_sha256
            and record.get("streaming_runner_blocker_names_sha256_match") is True
            for record in kv_streaming_mirror_records
        )
    )
    first_kv_streaming_runner_blocker_sha256_match = (
        first_kv_streaming_blocker_sha256 is None
        if first_kv_streaming_blocker is None
        else first_kv_streaming_blocker_sha256
        == _stable_json_sha256(first_kv_streaming_blocker)
    )
    first_kv_streaming_runner_blocker_mirrors = bool(
        kv_streaming_mirror_records
    ) and all(
        record.get("first_streaming_runner_blocker") == first_kv_streaming_blocker
        and record.get("first_streaming_runner_blocker_sha256")
        == first_kv_streaming_blocker_sha256
        for record in kv_streaming_mirror_records
    )
    kv_streaming_blueprint_sha256_match = (
        isinstance(kv_streaming_blueprint, dict)
        and kv_streaming_blueprint_sha256
        == _stable_json_sha256(kv_streaming_blueprint)
    )
    kv_streaming_blueprint_mirrors = bool(kv_streaming_mirror_records) and all(
        record.get("streaming_decode_loop_blueprint") == kv_streaming_blueprint
        and record.get("streaming_decode_loop_blueprint_sha256")
        == kv_streaming_blueprint_sha256
        for record in kv_streaming_mirror_records
    )
    kv_streaming_loop_status_sha256_match = (
        isinstance(kv_streaming_loop_status, dict)
        and kv_streaming_loop_status_sha256
        == _stable_json_sha256(kv_streaming_loop_status)
    )
    kv_streaming_loop_status_mirrors = bool(kv_streaming_mirror_records) and all(
        record.get("streaming_decode_loop_status") == kv_streaming_loop_status
        and record.get("streaming_decode_loop_status_sha256")
        == kv_streaming_loop_status_sha256
        for record in kv_streaming_mirror_records
    )
    kv_streaming_loop_next_action_sha256_match = (
        kv_streaming_loop_next_action_sha256
        == _stable_json_sha256(kv_streaming_loop_next_action)
    )
    blocker_recommended_commands_sha256_match = (
        isinstance(blocker_recommended_commands, list)
        and blocker_recommended_commands_sha256
        == _stable_json_sha256(blocker_recommended_commands)
    )
    blocker_recommended_commands_meta_mirror = (
        isinstance(blocker_meta, dict)
        and blocker_meta.get("recommended_commands_sha256")
        == blocker_recommended_commands_sha256
    )
    checks = {
        "source_artifacts_sha256": (
            isinstance(source_artifacts, dict)
            and status.get("source_artifacts_sha256") == _stable_json_sha256(source_artifacts)
        ),
        "handoff_summary_sha256": (
            isinstance(handoff_summary, dict)
            and status.get("handoff_summary_sha256") == _stable_json_sha256(handoff_summary)
        ),
        "readiness_summary_sha256": (
            isinstance(readiness_summary, dict)
            and status.get("readiness_summary_sha256") == _stable_json_sha256(readiness_summary)
        ),
        "readiness_gates_sha256": (
            isinstance(readiness_gates, dict)
            and status.get("readiness_gates_sha256") == _stable_json_sha256(readiness_gates)
        ),
        "next_action_commands_sha256": (
            isinstance(next_action_commands, dict)
            and status.get("next_action_commands_sha256") == _stable_json_sha256(next_action_commands)
        ),
        "blocker_kinds_sha256": (
            isinstance(blocker_kinds, list)
            and status.get("blocker_kinds_sha256") == _stable_json_sha256(blocker_kinds)
        ),
        "blocked_gates_sha256": (
            isinstance(blocked_gates, list)
            and status.get("blocked_gates_sha256") == _stable_json_sha256(blocked_gates)
        ),
        "kv_streaming_runner_blocker_names_sha256": (
            isinstance(kv_streaming_blocker_names, list)
            and kv_streaming_blocker_names_sha256
            == _stable_json_sha256(kv_streaming_blocker_names)
            and kv_streaming_blocker_names_sha256_match is True
        ),
        "kv_streaming_runner_blocker_mirrors": kv_streaming_runner_blocker_mirrors,
        "first_kv_streaming_runner_blocker_sha256": (
            first_kv_streaming_runner_blocker_sha256_match
        ),
        "first_kv_streaming_runner_blocker_mirrors": (
            first_kv_streaming_runner_blocker_mirrors
        ),
        "kv_streaming_blueprint_sha256": kv_streaming_blueprint_sha256_match,
        "kv_streaming_blueprint_mirrors": kv_streaming_blueprint_mirrors,
        "kv_streaming_loop_status_sha256": kv_streaming_loop_status_sha256_match,
        "kv_streaming_loop_status_mirrors": kv_streaming_loop_status_mirrors,
        "kv_streaming_loop_next_action_sha256": (
            kv_streaming_loop_next_action_sha256_match
        ),
        "blocker_recommended_commands_sha256": (
            blocker_recommended_commands_sha256_match
        ),
        "blocker_recommended_commands_meta_mirror": (
            blocker_recommended_commands_meta_mirror
        ),
        "schema_versions": schema_versions
        == {
            "status": status.get("schema_version"),
            "readiness_summary": readiness_summary.get("schema_version")
            if isinstance(readiness_summary, dict)
            else None,
            "handoff_summary": handoff_summary.get("schema_version")
            if isinstance(handoff_summary, dict)
            else None,
            "blocker_work_queue": handoff_summary.get("blocker_work_queue_schema_version")
            if isinstance(handoff_summary, dict)
            else None,
            "first_blocker_work_item": blocker_meta.get("first_work_item_schema_version")
            if isinstance(blocker_meta, dict)
            else None,
        },
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {"all_match": not failed_checks, "failed_checks": failed_checks, "checks": checks}


def _persisted_status_integrity(
    status: dict[str, object], computed_status_integrity: dict[str, object]
) -> dict[str, object]:
    """Verify persisted status_integrity fields against computed checks."""

    checks = {
        "status_integrity_payload": status.get("status_integrity")
        == computed_status_integrity,
        "status_integrity_sha256": status.get("status_integrity_sha256")
        == _stable_json_sha256(computed_status_integrity),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {"all_match": not failed_checks, "failed_checks": failed_checks, "checks": checks}


def _verify_source_artifacts(status_artifact: Path) -> dict[str, object]:
    """Verify embedded source_artifacts provenance against current files."""

    status = _load(status_artifact)
    source_artifacts = status.get("source_artifacts", {})
    if not isinstance(source_artifacts, dict) or not source_artifacts:
        status_integrity = _status_integrity(status)
        persisted_status_integrity = _persisted_status_integrity(status, status_integrity)
        source_artifact_failed_records: list[str] = []
        verification_failures = {
            "source_artifact_failed_records": source_artifact_failed_records,
            "status_integrity_failed_checks": status_integrity["failed_checks"],
            "persisted_status_integrity_failed_checks": persisted_status_integrity[
                "failed_checks"
            ],
        }
        return {
            "status": "missing_source_artifacts",
            "status_artifact": str(status_artifact),
            "verification_exit_code": SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
            "all_match": False,
            "source_artifacts_all_match": False,
            "source_artifact_failed_records": source_artifact_failed_records,
            "verification_failures": verification_failures,
            "verification_failures_sha256": _stable_json_sha256(verification_failures),
            "checked_count": 0,
            "records": {},
            "status_integrity": status_integrity,
            "persisted_status_integrity": persisted_status_integrity,
        }
    records: dict[str, object] = {}
    source_artifacts_all_match = True
    for name, recorded_obj in source_artifacts.items():
        recorded = dict(recorded_obj) if isinstance(recorded_obj, dict) else {}
        path_value = recorded.get("path")
        current = _artifact_record(Path(str(path_value))) if path_value is not None else {}
        matches = {
            "exists": recorded.get("exists") == current.get("exists"),
            "size_bytes": recorded.get("size_bytes") == current.get("size_bytes"),
            "sha256": recorded.get("sha256") == current.get("sha256"),
        }
        match = all(matches.values())
        source_artifacts_all_match = source_artifacts_all_match and match
        records[str(name)] = {
            "path": path_value,
            "match": match,
            "matches": matches,
            "recorded": recorded,
            "current": current,
        }
    source_artifact_failed_records = [
        name for name, record in records.items() if record["match"] is not True
    ]
    status_integrity = _status_integrity(status)
    persisted_status_integrity = _persisted_status_integrity(status, status_integrity)
    verification_failures = {
        "source_artifact_failed_records": source_artifact_failed_records,
        "status_integrity_failed_checks": status_integrity["failed_checks"],
        "persisted_status_integrity_failed_checks": persisted_status_integrity[
            "failed_checks"
        ],
    }
    all_match = (
        source_artifacts_all_match
        and status_integrity["all_match"] is True
        and persisted_status_integrity["all_match"] is True
    )
    verification_exit_code = (
        READY_EXIT_CODE if all_match is True else SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    return {
        "status": "match" if all_match else "mismatch",
        "status_artifact": str(status_artifact),
        "verification_exit_code": verification_exit_code,
        "all_match": all_match,
        "source_artifacts_all_match": source_artifacts_all_match,
        "source_artifact_failed_records": source_artifact_failed_records,
        "verification_failures": verification_failures,
        "verification_failures_sha256": _stable_json_sha256(verification_failures),
        "checked_count": len(records),
        "records": records,
        "status_integrity": status_integrity,
        "persisted_status_integrity": persisted_status_integrity,
    }


def _docs_checklist_status(docs_path: Path) -> dict[str, object]:
    text = docs_path.read_text()
    block = text.split("### P0", 1)[1].split("### P13", 1)[0]
    start_line = text[: text.index("### P0")].count("\n") + 1
    items: list[dict[str, object]] = []
    for offset, line in enumerate(block.splitlines(), start=0):
        match = re.match(r"^- \[( |~)\] (.*)", line)
        if match:
            items.append(
                {
                    "line": start_line + offset,
                    "state": "partial" if match.group(1) == "~" else "open",
                    "text": match.group(2),
                }
            )
    return {
        "docs_path": str(docs_path),
        "open_or_partial_count_p0_p12": len(items),
        "open_or_partial_items_p0_p12": items,
    }


_ATTENTION_LINEAR_SUFFIXES = ("attn_q", "attn_k", "attn_v", "attn_gate", "attn_output")
_DENSE_MLP_LINEAR_SUFFIXES = ("ffn_gate", "ffn_up", "ffn_down")
_MOE_ROUTER_SUFFIXES = ("ffn_gate_inp",)
_MOE_EXPERT_LINEAR_SUFFIXES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
_MOE_SHARED_LINEAR_SUFFIXES = ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
_RESIDENT_LINEAR_SUFFIXES = (
    *_ATTENTION_LINEAR_SUFFIXES,
    *_DENSE_MLP_LINEAR_SUFFIXES,
    *_MOE_EXPERT_LINEAR_SUFFIXES,
    *_MOE_SHARED_LINEAR_SUFFIXES,
)


def _linear_projection_progress(prompt: dict[str, object]) -> dict[str, object]:
    """Summarize resident linear projection coverage from a prompt artifact."""

    slots = [slot for slot in prompt.get("selected_slots", []) if isinstance(slot, str)]
    layer_suffixes: dict[int, set[str]] = {}
    for slot in slots:
        if not slot.startswith("layers."):
            continue
        parts = slot.split(".")
        if len(parts) != 3:
            continue
        try:
            layer_id = int(parts[1])
        except ValueError:
            continue
        layer_suffixes.setdefault(layer_id, set()).add(parts[2])

    suffix_counts: dict[str, int] = {}
    for suffixes in layer_suffixes.values():
        for suffix in suffixes:
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    def suffix_count(suffixes: Sequence[str]) -> int:
        return sum(suffix_counts.get(suffix, 0) for suffix in suffixes)

    def complete_layers(suffixes: Sequence[str]) -> int:
        required = set(suffixes)
        return sum(1 for layer_slots in layer_suffixes.values() if required <= layer_slots)

    try:
        layer_count = int(prompt.get("layer_count", 0))
    except (TypeError, ValueError):
        layer_count = 0
    selected_layer_count = len(layer_suffixes)
    expected_layer_count = layer_count or selected_layer_count
    root_lm_head_present = "root.lm_head" in slots
    attention_slot_count = suffix_count(_ATTENTION_LINEAR_SUFFIXES)
    dense_slot_count = suffix_count(_DENSE_MLP_LINEAR_SUFFIXES)
    moe_router_slot_count = suffix_count(_MOE_ROUTER_SUFFIXES)
    moe_expert_slot_count = suffix_count(_MOE_EXPERT_LINEAR_SUFFIXES)
    moe_shared_slot_count = suffix_count(_MOE_SHARED_LINEAR_SUFFIXES)
    resident_linear_projection_slot_count = int(root_lm_head_present) + suffix_count(_RESIDENT_LINEAR_SUFFIXES)
    attention_expected = expected_layer_count * len(_ATTENTION_LINEAR_SUFFIXES)
    return {
        "source": "prompt_artifact.selected_slots",
        "execution_mode": prompt.get("execution_mode"),
        "prompt_status": prompt.get("status"),
        "layer_count": layer_count,
        "selected_layer_count": selected_layer_count,
        "selected_slot_count": prompt.get("selected_slot_count", len(slots)),
        "root_lm_head_present": root_lm_head_present,
        "resident_linear_projection_slot_count": resident_linear_projection_slot_count,
        "host_reference_router_projection_slot_count": moe_router_slot_count,
        "attention": {
            "slot_count": attention_slot_count,
            "expected_slot_count": attention_expected,
            "complete_layer_count": complete_layers(_ATTENTION_LINEAR_SUFFIXES),
            "all_selected_layers_complete": (
                selected_layer_count == expected_layer_count
                and attention_slot_count == attention_expected
                and complete_layers(_ATTENTION_LINEAR_SUFFIXES) == expected_layer_count
            ),
        },
        "dense_mlp": {
            "slot_count": dense_slot_count,
            "complete_layer_count": complete_layers(_DENSE_MLP_LINEAR_SUFFIXES),
        },
        "moe_router": {
            "slot_count": moe_router_slot_count,
            "complete_layer_count": complete_layers(_MOE_ROUTER_SUFFIXES),
            "execution_note": "router weights are copied to the host CPU-reference router in current probes",
        },
        "moe_expert": {
            "slot_count": moe_expert_slot_count,
            "complete_layer_count": complete_layers(_MOE_EXPERT_LINEAR_SUFFIXES),
        },
        "moe_shared_expert": {
            "slot_count": moe_shared_slot_count,
            "complete_layer_count": complete_layers(_MOE_SHARED_LINEAR_SUFFIXES),
        },
        "note": (
            "Derived from selected slots in the all-layer host-composed prompt smoke. "
            "This records GGUF linear projection coverage only; it is not KV-backed decode, "
            "oracle parity, or performance evidence."
        ),
    }


def _oracle_progress(oracle: dict[str, object]) -> dict[str, object]:
    """Summarize the current deterministic oracle target and blocker."""

    stdout = str(oracle.get("stdout", ""))
    stderr = str(oracle.get("stderr", ""))
    generated = str(oracle.get("generated_text", ""))
    expected_top_tokens = oracle.get("expected_top_tokens", [])
    return {
        "source": "oracle_artifact",
        "status": oracle.get("status"),
        "returncode": oracle.get("returncode"),
        "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
        "oracle_blocker_detail": oracle.get("oracle_blocker_detail"),
        "llama_cli": oracle.get("llama_cli"),
        "llama_cpp_version": oracle.get("llama_cpp_version"),
        "model": oracle.get("model"),
        "prompt_length": oracle.get("prompt_length"),
        "n_predict": oracle.get("n_predict"),
        "timeout_s": oracle.get("timeout_s"),
        "diagnostic_logs": oracle.get("diagnostic_logs") is True,
        "elapsed_s": oracle.get("elapsed_s"),
        "extra_llama_args": list(oracle.get("extra_llama_args", [])),
        "command_shell": oracle.get("command_shell"),
        "expected_next_token_id": oracle.get("expected_next_token_id"),
        "expected_next_token_text": oracle.get("expected_next_token_text"),
        "expected_next_token_logit": oracle.get("expected_next_token_logit"),
        "expected_top_tokens": expected_top_tokens if isinstance(expected_top_tokens, list) else [],
        "generated_text_len": len(generated),
        "stdout_len": len(stdout),
        "stderr_len": len(stderr),
        "text_matches_expected_exact": oracle.get("text_matches_expected_exact") is True,
        "text_matches_expected_stripped": oracle.get("text_matches_expected_stripped") is True,
        "comparison_policy": dict(oracle.get("comparison_policy", {})),
        "step35_supported_by_oracle": oracle.get("step35_supported"),
        "note": (
            "This records the deterministic oracle target and current blocker only. "
            "It is not oracle parity unless a comparable generated token matches the expected text/logit policy."
        ),
    }


def _oracle_gap_report(oracle_progress: dict[str, object]) -> dict[str, object]:
    """Return machine-readable remaining evidence for the oracle parity blocker."""

    preconditions = [
        {
            "name": "deterministic_target_recorded",
            "ready": oracle_progress.get("command_shell") is not None
            and oracle_progress.get("expected_next_token_id") is not None
            and oracle_progress.get("expected_next_token_text") is not None
            and oracle_progress.get("prompt_length") is not None
            and oracle_progress.get("n_predict") is not None,
            "evidence": "oracle command, prompt length, n_predict, and expected token/text are recorded",
        },
        {
            "name": "oracle_binary_recorded",
            "ready": bool(oracle_progress.get("llama_cli"))
            and bool(oracle_progress.get("llama_cpp_version"))
            and bool(oracle_progress.get("model")),
            "evidence": "llama-cli path/version and GGUF model path are recorded",
        },
        {
            "name": "step35_not_rejected",
            "ready": oracle_progress.get("step35_supported_by_oracle") is not False
            and oracle_progress.get("oracle_blocker_kind")
            != "llama_cpp_missing_step35_architecture",
            "evidence": "current oracle blocker is not an explicit unknown step35 architecture rejection",
        },
    ]
    remaining_evidence = [
        {
            "name": "oracle_completed_successfully",
            "ready": oracle_progress.get("status") == "executed"
            and oracle_progress.get("returncode") == 0,
            "required_evidence": "llama.cpp/CPU oracle run must complete with status=executed and returncode=0",
            "current": {
                "status": oracle_progress.get("status"),
                "returncode": oracle_progress.get("returncode"),
                "oracle_blocker_kind": oracle_progress.get("oracle_blocker_kind"),
                "elapsed_s": oracle_progress.get("elapsed_s"),
                "timeout_s": oracle_progress.get("timeout_s"),
            },
        },
        {
            "name": "oracle_generated_comparable_text",
            "ready": int(oracle_progress.get("generated_text_len") or 0) > 0,
            "required_evidence": "oracle artifact must capture non-empty generated_text for the one-token run",
            "current": {
                "generated_text_len": oracle_progress.get("generated_text_len"),
                "stdout_len": oracle_progress.get("stdout_len"),
                "stderr_len": oracle_progress.get("stderr_len"),
            },
        },
        {
            "name": "oracle_exact_text_match",
            "ready": oracle_progress.get("text_matches_expected_exact") is True,
            "required_evidence": "oracle generated_text must exactly match expected_next_token_text",
            "current": {
                "expected_next_token_id": oracle_progress.get("expected_next_token_id"),
                "expected_next_token_text": oracle_progress.get("expected_next_token_text"),
                "text_matches_expected_exact": oracle_progress.get("text_matches_expected_exact"),
                "text_matches_expected_stripped": oracle_progress.get("text_matches_expected_stripped"),
            },
        },
    ]
    missing_preconditions = [
        str(item["name"]) for item in preconditions if item.get("ready") is not True
    ]
    missing_evidence = [
        str(item["name"]) for item in remaining_evidence if item.get("ready") is not True
    ]
    return {
        "source": "oracle_progress",
        "status": "ready" if not missing_preconditions and not missing_evidence else "blocked",
        "precondition_count": len(preconditions),
        "validated_precondition_count": sum(1 for item in preconditions if item.get("ready") is True),
        "validated_preconditions": [
            str(item["name"]) for item in preconditions if item.get("ready") is True
        ],
        "missing_preconditions": missing_preconditions,
        "missing_precondition_count": len(missing_preconditions),
        "first_missing_precondition": missing_preconditions[0] if missing_preconditions else None,
        "missing_evidence": missing_evidence,
        "missing_evidence_count": len(missing_evidence),
        "first_missing_evidence": missing_evidence[0] if missing_evidence else None,
        "preconditions": preconditions,
        "remaining_evidence": remaining_evidence,
        "oracle_blocker_kind": oracle_progress.get("oracle_blocker_kind"),
        "oracle_status": oracle_progress.get("status"),
        "returncode": oracle_progress.get("returncode"),
        "elapsed_s": oracle_progress.get("elapsed_s"),
        "timeout_s": oracle_progress.get("timeout_s"),
        "expected_next_token_id": oracle_progress.get("expected_next_token_id"),
        "expected_next_token_text": oracle_progress.get("expected_next_token_text"),
        "note": (
            "This separates recorded deterministic oracle prerequisites from the evidence still needed "
            "before oracle_parity can become true. It is not a performance or e2e-readiness claim."
        ),
    }



def _kv_decode_dispatch_progress(resource: dict[str, object]) -> dict[str, object]:
    """Summarize KV dispatch coverage from the text resource artifact."""

    plan = dict(resource.get("text_decode_resource_plan", {}))
    kv_plan = dict(plan.get("kv_decode_kernel_plan", {}))
    run_plan = dict(resource.get("kv_decode_run_plan", {}))
    decode_input_upload_plan = dict(run_plan.get("decode_input_upload_plan", {}))
    streaming_decode_loop_blueprint = dict(
        run_plan.get("streaming_decode_loop_blueprint", {})
    )
    streaming_decode_loop_blueprint_sha256 = (
        _stable_json_sha256(streaming_decode_loop_blueprint)
        if streaming_decode_loop_blueprint
        else None
    )
    streaming_decode_loop_status = dict(run_plan.get("streaming_decode_loop_status", {}))
    streaming_decode_loop_status_sha256 = (
        _stable_json_sha256(streaming_decode_loop_status)
        if streaming_decode_loop_status
        else None
    )
    registered = dict(kv_plan.get("registered", {}))
    blueprint_operation_count = streaming_decode_loop_blueprint.get("operation_count")
    launch_operation_count = dict(plan.get("kv_decode_launch_schedule", {})).get(
        "operation_count"
    )
    return {
        "source": "resource_artifact.text_decode_resource_plan.kv_decode_kernel_plan",
        "resource_status": resource.get("status"),
        "backend": kv_plan.get("backend") or plan.get("backend"),
        "model_quant": kv_plan.get("model_quant"),
        "kv_storage_dtype": kv_plan.get("kv_storage_dtype"),
        "decode_attention_kind": kv_plan.get("decode_attention_kind"),
        "max_context": kv_plan.get("max_context"),
        "max_new_tokens": kv_plan.get("max_new_tokens"),
        "max_prompt_rows": kv_plan.get("max_prompt_rows"),
        "attention_block_size": kv_plan.get("attention_block_size"),
        "attention_block_table_len": kv_plan.get("attention_block_table_len"),
        "attention_capacity_tokens": kv_plan.get("attention_capacity_tokens"),
        "decode_span": dict(kv_plan.get("decode_span", {})),
        "prompt_span": dict(kv_plan.get("prompt_span", {})),
        "decode_span_shape_compatible": kv_plan.get("decode_span_shape_compatible") is True,
        "prompt_span_shape_compatible": kv_plan.get("prompt_span_shape_compatible") is True,
        "span_shape_compatible": kv_plan.get("span_shape_compatible") is True,
        "launch_schedule": dict(plan.get("kv_decode_launch_schedule", {})),
        "run_plan": run_plan,
        "streaming_decode_loop_blueprint": streaming_decode_loop_blueprint,
        "streaming_decode_loop_blueprint_sha256": streaming_decode_loop_blueprint_sha256,
        "streaming_decode_loop_status": streaming_decode_loop_status,
        "streaming_decode_loop_status_sha256": streaming_decode_loop_status_sha256,
        "streaming_decode_loop_status_recorded": bool(
            streaming_decode_loop_status.get("blueprint_sha256")
        ),
        "streaming_decode_loop_status_matches_blueprint": (
            streaming_decode_loop_status.get("blueprint_sha256")
            == streaming_decode_loop_blueprint_sha256
            and streaming_decode_loop_status.get("blocked_by")
            == streaming_decode_loop_blueprint.get("blocked_by")
            and streaming_decode_loop_status.get("ready")
            == run_plan.get("streaming_runner_ready")
        ),
        "streaming_decode_loop_status_blocker_names_match": (
            list(streaming_decode_loop_status.get("blocker_names", []))
            == list(run_plan.get("streaming_runner_blocker_names", []))
        ),
        "streaming_decode_loop_blueprint_recorded": bool(
            streaming_decode_loop_blueprint.get("operation_count")
        ),
        "streaming_decode_loop_blueprint_matches_launch_schedule": (
            blueprint_operation_count == launch_operation_count
            and list(streaming_decode_loop_blueprint.get("per_layer_order", []))
            == list(
                dict(plan.get("kv_decode_launch_schedule", {})).get(
                    "per_layer_order", []
                )
            )
            and streaming_decode_loop_blueprint.get("layer_count")
            == dict(plan.get("kv_decode_launch_schedule", {})).get("layer_count")
        ),
        "streaming_decode_loop_blueprint_upload_order_matches": (
            list(streaming_decode_loop_blueprint.get("pre_run_upload_order", []))
            == list(decode_input_upload_plan.get("upload_order", []))
        ),
        "streaming_decode_loop_blueprint_blocker_matches": (
            streaming_decode_loop_blueprint.get("blocked_by")
            == run_plan.get("first_streaming_runner_blocker")
        ),
        "run_plan_prompt_fits_resource_plan": run_plan.get("prompt_fits_resource_plan") is True,
        "run_plan_context_fits_resource_plan": run_plan.get("context_fits_resource_plan") is True,
        "dispatch_keys": dict(kv_plan.get("dispatch_keys", {})),
        "registered": registered,
        "all_registered": kv_plan.get("all_registered") is True and all(bool(v) for v in registered.values()),
        "note": (
            "Derived from the text resource dry-run artifact. Registry dispatch readiness "
            "does not mean the streaming KV-backed runner or oracle parity is complete."
        ),
    }


def _kv_backed_decode_gap_report(
    kv_decode_dispatch_progress: dict[str, object],
) -> dict[str, object]:
    """Return machine-readable remaining evidence for the KV-backed runner blocker."""

    launch_schedule = dict(kv_decode_dispatch_progress.get("launch_schedule", {}))
    run_plan = dict(kv_decode_dispatch_progress.get("run_plan", {}))
    decode_input_upload_plan = dict(run_plan.get("decode_input_upload_plan", {}))
    streaming_decode_loop_blueprint = dict(
        kv_decode_dispatch_progress.get("streaming_decode_loop_blueprint", {})
    )
    streaming_decode_loop_status = dict(
        kv_decode_dispatch_progress.get("streaming_decode_loop_status", {})
    )
    launch_schedule_streaming_runner_blockers = list(
        launch_schedule.get("streaming_runner_blockers", [])
    )
    run_plan_streaming_runner_blockers = list(run_plan.get("streaming_runner_blockers", []))
    streaming_runner_blockers = (
        run_plan_streaming_runner_blockers or launch_schedule_streaming_runner_blockers
    )
    launch_schedule_streaming_runner_blocker_names = list(
        launch_schedule.get("streaming_runner_blocker_names", [])
    )
    run_plan_streaming_runner_blocker_names = list(
        run_plan.get("streaming_runner_blocker_names", [])
    )
    streaming_runner_blocker_names = (
        run_plan_streaming_runner_blocker_names
        or launch_schedule_streaming_runner_blocker_names
        or [str(blocker.get("name")) for blocker in streaming_runner_blockers]
    )
    streaming_runner_blocker_names_sha256 = run_plan.get(
        "streaming_runner_blocker_names_sha256"
    ) or launch_schedule.get("streaming_runner_blocker_names_sha256")
    computed_streaming_runner_blocker_names_sha256 = (
        _stable_json_sha256(streaming_runner_blocker_names)
        if streaming_runner_blocker_names
        else None
    )
    streaming_runner_blocker_names_sha256_match = (
        streaming_runner_blocker_names_sha256 == computed_streaming_runner_blocker_names_sha256
        if streaming_runner_blocker_names_sha256 is not None
        else None
    )
    first_streaming_runner_blocker = run_plan.get(
        "first_streaming_runner_blocker"
    ) or launch_schedule.get("first_streaming_runner_blocker")
    first_streaming_runner_blocker_sha256 = (
        _stable_json_sha256(first_streaming_runner_blocker)
        if first_streaming_runner_blocker is not None
        else None
    )
    streaming_runner_blocker_count = run_plan.get(
        "streaming_runner_blocker_count"
    ) or launch_schedule.get("streaming_runner_blocker_count")
    preconditions = [
        {
            "name": "dispatch_keys_registered",
            "ready": kv_decode_dispatch_progress.get("all_registered") is True,
            "evidence": "prompt/decode KV write and decode-attention registry keys resolve",
        },
        {
            "name": "span_shapes_compatible",
            "ready": kv_decode_dispatch_progress.get("span_shape_compatible") is True,
            "evidence": "prompt and decode KVLiveSpans geometry matches the 512-token bring-up window",
        },
        {
            "name": "launch_schedule_recorded",
            "ready": bool(launch_schedule.get("operation_count")),
            "evidence": f"{launch_schedule.get('operation_count')} planned KV operations",
        },
        {
            "name": "run_plan_fits_resource_plan",
            "ready": kv_decode_dispatch_progress.get("run_plan_prompt_fits_resource_plan") is True
            and kv_decode_dispatch_progress.get("run_plan_context_fits_resource_plan") is True,
            "evidence": "prompt rows and prompt+decode context fit the text resource plan",
        },
        {
            "name": "decode_input_upload_plan_consistent",
            "ready": decode_input_upload_plan.get("all_consistency_checks_passed") is True,
            "evidence": (
                f"{decode_input_upload_plan.get('entry_count')} staged input entries / "
                f"{decode_input_upload_plan.get('total_nbytes')} bytes"
            ),
        },
        {
            "name": "streaming_decode_loop_blueprint_recorded",
            "ready": kv_decode_dispatch_progress.get(
                "streaming_decode_loop_blueprint_recorded"
            ) is True
            and kv_decode_dispatch_progress.get(
                "streaming_decode_loop_blueprint_matches_launch_schedule"
            ) is True
            and kv_decode_dispatch_progress.get(
                "streaming_decode_loop_blueprint_upload_order_matches"
            ) is True
            and kv_decode_dispatch_progress.get(
                "streaming_decode_loop_blueprint_blocker_matches"
            ) is True,
            "evidence": (
                f"{streaming_decode_loop_blueprint.get('operation_count')} blueprint operations / "
                f"{streaming_decode_loop_blueprint.get('stage_count')} stages"
            ),
        },
        {
            "name": "streaming_decode_loop_status_recorded",
            "ready": kv_decode_dispatch_progress.get(
                "streaming_decode_loop_status_recorded"
            ) is True
            and kv_decode_dispatch_progress.get(
                "streaming_decode_loop_status_matches_blueprint"
            ) is True
            and kv_decode_dispatch_progress.get(
                "streaming_decode_loop_status_blocker_names_match"
            ) is True,
            "evidence": (
                f"{streaming_decode_loop_status.get('blocker_count')} blockers / "
                f"next_action={streaming_decode_loop_status.get('next_action')}"
            ),
        },
    ]
    remaining_evidence = [
        {
            "name": "streaming_runner_ready_flags",
            "ready": launch_schedule.get("streaming_runner_ready") is True
            and run_plan.get("streaming_runner_ready") is True,
            "required_evidence": (
                "text_decode_resource_plan.kv_decode_launch_schedule.streaming_runner_ready "
                "and kv_decode_run_plan.streaming_runner_ready must both be true"
            ),
            "current": {
                "launch_schedule_streaming_runner_ready": launch_schedule.get(
                    "streaming_runner_ready"
                ),
                "run_plan_streaming_runner_ready": run_plan.get("streaming_runner_ready"),
                "streaming_runner_blocker_count": streaming_runner_blocker_count,
                "streaming_runner_blocker_names": streaming_runner_blocker_names,
                "streaming_runner_blocker_names_sha256": streaming_runner_blocker_names_sha256,
                "computed_streaming_runner_blocker_names_sha256": computed_streaming_runner_blocker_names_sha256,
                "streaming_runner_blocker_names_sha256_match": streaming_runner_blocker_names_sha256_match,
                "first_streaming_runner_blocker": first_streaming_runner_blocker,
                "first_streaming_runner_blocker_sha256": first_streaming_runner_blocker_sha256,
                "streaming_runner_blockers": streaming_runner_blockers,
                "launch_schedule_streaming_runner_blocker_count": launch_schedule.get(
                    "streaming_runner_blocker_count"
                ),
                "run_plan_streaming_runner_blocker_count": run_plan.get(
                    "streaming_runner_blocker_count"
                ),
            },
        },
        {
            "name": "kv_kernel_launch_trace",
            "ready": bool(run_plan.get("kv_kernel_trace_artifact")),
            "required_evidence": (
                "A retained rocprofv3 or equivalent trace must show the prompt KV write, "
                "decode KV write, and gated decode-attention kernels launching for the canonical prompt"
            ),
            "current": run_plan.get("kv_kernel_trace_artifact"),
        },
        {
            "name": "kv_backed_next_token_artifact",
            "ready": bool(run_plan.get("kv_backed_next_token_artifact")),
            "required_evidence": (
                "A KV-backed one-token decode artifact must record the generated token/logit path "
                "without host-composed layer-prefix outputs"
            ),
            "current": run_plan.get("kv_backed_next_token_artifact"),
        },
    ]
    missing_preconditions = [
        str(item["name"]) for item in preconditions if item.get("ready") is not True
    ]
    missing_evidence = [
        str(item["name"]) for item in remaining_evidence if item.get("ready") is not True
    ]
    streaming_decode_loop_blueprint_summary = {
        "recorded": kv_decode_dispatch_progress.get(
            "streaming_decode_loop_blueprint_recorded"
        ),
        "matches_launch_schedule": kv_decode_dispatch_progress.get(
            "streaming_decode_loop_blueprint_matches_launch_schedule"
        ),
        "upload_order_matches": kv_decode_dispatch_progress.get(
            "streaming_decode_loop_blueprint_upload_order_matches"
        ),
        "blocker_matches": kv_decode_dispatch_progress.get(
            "streaming_decode_loop_blueprint_blocker_matches"
        ),
        "executable": streaming_decode_loop_blueprint.get("executable"),
        "blocked_by": streaming_decode_loop_blueprint.get("blocked_by"),
        "blocked_by_sha256": streaming_decode_loop_blueprint.get("blocked_by_sha256"),
        "operation_count": streaming_decode_loop_blueprint.get("operation_count"),
        "operation_sequence_sha256": streaming_decode_loop_blueprint.get(
            "operation_sequence_sha256"
        ),
        "stage_count": streaming_decode_loop_blueprint.get("stage_count"),
        "pre_run_upload_checks_passed": streaming_decode_loop_blueprint.get(
            "pre_run_upload_checks_passed"
        ),
    }
    streaming_decode_loop_blueprint_summary_sha256 = _stable_json_sha256(
        streaming_decode_loop_blueprint_summary
    )
    streaming_decode_loop_status_summary = {
        "recorded": kv_decode_dispatch_progress.get(
            "streaming_decode_loop_status_recorded"
        ),
        "matches_blueprint": kv_decode_dispatch_progress.get(
            "streaming_decode_loop_status_matches_blueprint"
        ),
        "blocker_names_match": kv_decode_dispatch_progress.get(
            "streaming_decode_loop_status_blocker_names_match"
        ),
        "ready": streaming_decode_loop_status.get("ready"),
        "executable": streaming_decode_loop_status.get("executable"),
        "blocked_by": streaming_decode_loop_status.get("blocked_by"),
        "blocked_by_sha256": streaming_decode_loop_status.get("blocked_by_sha256"),
        "blocker_count": streaming_decode_loop_status.get("blocker_count"),
        "blocker_names_sha256": streaming_decode_loop_status.get(
            "blocker_names_sha256"
        ),
        "blueprint_operation_count": streaming_decode_loop_status.get(
            "blueprint_operation_count"
        ),
        "blueprint_stage_count": streaming_decode_loop_status.get(
            "blueprint_stage_count"
        ),
        "blueprint_sha256": streaming_decode_loop_status.get("blueprint_sha256"),
        "next_action": streaming_decode_loop_status.get("next_action"),
        "next_action_sha256": _stable_json_sha256(
            streaming_decode_loop_status.get("next_action")
        ),
    }
    streaming_decode_loop_status_summary_sha256 = _stable_json_sha256(
        streaming_decode_loop_status_summary
    )
    return {
        "source": "kv_decode_dispatch_progress",
        "status": "ready" if not missing_preconditions and not missing_evidence else "blocked",
        "precondition_count": len(preconditions),
        "validated_precondition_count": sum(1 for item in preconditions if item.get("ready") is True),
        "validated_preconditions": [
            str(item["name"]) for item in preconditions if item.get("ready") is True
        ],
        "missing_preconditions": missing_preconditions,
        "missing_precondition_count": len(missing_preconditions),
        "first_missing_precondition": missing_preconditions[0] if missing_preconditions else None,
        "missing_evidence": missing_evidence,
        "missing_evidence_count": len(missing_evidence),
        "first_missing_evidence": missing_evidence[0] if missing_evidence else None,
        "preconditions": preconditions,
        "remaining_evidence": remaining_evidence,
        "operation_count": launch_schedule.get("operation_count"),
        "streaming_decode_loop_blueprint": streaming_decode_loop_blueprint_summary,
        "streaming_decode_loop_blueprint_sha256": streaming_decode_loop_blueprint_summary_sha256,
        "streaming_decode_loop_status": streaming_decode_loop_status_summary,
        "streaming_decode_loop_status_sha256": streaming_decode_loop_status_summary_sha256,
        "streaming_runner_blocker_count": streaming_runner_blocker_count,
        "streaming_runner_blocker_names": streaming_runner_blocker_names,
        "streaming_runner_blocker_names_sha256": streaming_runner_blocker_names_sha256,
        "computed_streaming_runner_blocker_names_sha256": computed_streaming_runner_blocker_names_sha256,
        "streaming_runner_blocker_names_sha256_match": streaming_runner_blocker_names_sha256_match,
        "first_streaming_runner_blocker": first_streaming_runner_blocker,
        "first_streaming_runner_blocker_sha256": first_streaming_runner_blocker_sha256,
        "streaming_runner_blockers": streaming_runner_blockers,
        "upload_entry_count": decode_input_upload_plan.get("entry_count"),
        "upload_total_nbytes": decode_input_upload_plan.get("total_nbytes"),
        "note": (
            "This separates validated metadata/input prerequisites from the evidence still needed "
            "before kv_backed_decode_ready can become true. It is not a performance or parity claim."
        ),
    }


def _status_refresh_command(
    *,
    prompt_artifact: Path,
    oracle_artifact: Path,
    resource_artifact: Path,
    docs_path: Path,
    output_artifact: Path = DEFAULT_STATUS_ARTIFACT,
) -> str:
    return (
        "python3 scripts/stepfun_correctness_status.py "
        f"--prompt-artifact {prompt_artifact} "
        f"--oracle-artifact {oracle_artifact} "
        f"--resource-artifact {resource_artifact} "
        f"--docs {docs_path} "
        f"--output {output_artifact} --pretty"
    )


def _source_artifacts_verify_command(
    *,
    status_artifact: Path = DEFAULT_STATUS_ARTIFACT,
    extra_args: tuple[str, ...] = (),
) -> str:
    args = " ".join(extra_args)
    suffix = f" {args}" if args else ""
    return (
        "python3 scripts/stepfun_correctness_status.py "
        f"--verify-source-artifacts {status_artifact}{suffix} --pretty"
    )


def _resource_plan_refresh_command(
    *,
    output_artifact: Path = DEFAULT_RESOURCE_ARTIFACT,
) -> str:
    return (
        "python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan "
        "--kv-context-pages 1 --kv-page-size 512 --pretty "
        f"> {output_artifact}"
    )


def _command_length_hash(prefix: str, command: str) -> dict[str, object]:
    return {
        f"{prefix}_nchars": len(command),
        f"{prefix}_sha256": hashlib.sha256(command.encode()).hexdigest(),
    }



def _oracle_helper_refresh_command(
    *,
    oracle_progress: dict[str, object],
    prompt_artifact: Path,
    oracle_artifact: Path,
    timeout_s_override: float | None = None,
) -> str:
    command = [
        "python3",
        "scripts/stepfun_llamacpp_oracle.py",
        "--artifact",
        str(prompt_artifact),
        "--llama-cli",
        str(oracle_progress.get("llama_cli")),
        "--model",
        str(oracle_progress.get("model")),
        "--n-predict",
        str(oracle_progress.get("n_predict") or 1),
    ]
    timeout_s = (
        timeout_s_override
        if timeout_s_override is not None
        else oracle_progress.get("timeout_s")
    )
    if timeout_s is not None:
        command.extend(["--timeout-s", str(timeout_s)])
    if oracle_progress.get("diagnostic_logs") is True:
        command.append("--diagnostic-logs")
    for extra_arg in oracle_progress.get("extra_llama_args", []):
        command.append(f"--llama-arg={extra_arg}")
    command.extend(["--execute", "--pretty", "--output", str(oracle_artifact)])
    return shlex.join(command)



def _next_action_commands(
    *,
    oracle_progress: dict[str, object],
    oracle_gap_report: dict[str, object],
    kv_backed_decode_gap_report: dict[str, object],
    prompt_artifact: Path,
    oracle_artifact: Path,
    resource_artifact: Path,
    docs_path: Path,
) -> dict[str, object]:
    """Return copy/pasteable blocker reproduction and refresh commands."""

    status_refresh = _status_refresh_command(
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
        resource_artifact=resource_artifact,
        docs_path=docs_path,
    )
    oracle_missing_preconditions = list(oracle_gap_report.get("missing_preconditions", []))
    oracle_missing_evidence = list(oracle_gap_report.get("missing_evidence", []))
    kv_missing_evidence = list(kv_backed_decode_gap_report.get("missing_evidence", []))
    oracle_helper_refresh = _oracle_helper_refresh_command(
        oracle_progress=oracle_progress,
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
    )
    oracle_helper_long_timeout = _oracle_helper_refresh_command(
        oracle_progress=oracle_progress,
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
        timeout_s_override=DEFAULT_ORACLE_LONG_TIMEOUT_S,
    )
    resource_plan_refresh = _resource_plan_refresh_command(output_artifact=resource_artifact)
    source_artifacts_verify = _source_artifacts_verify_command()
    verification_status = _source_artifacts_verify_command(
        extra_args=("--verification-status-only",)
    )
    verification_exit_code = _source_artifacts_verify_command(
        extra_args=("--verification-exit-code-only",)
    )
    verification_failures = _source_artifacts_verify_command(
        extra_args=("--verification-failures-only",)
    )
    verification_failures_sha = _source_artifacts_verify_command(
        extra_args=("--verification-failures-sha-only",)
    )
    return {
        "handoff_integrity": {
            "source_artifacts_verify_command": source_artifacts_verify,
            **_command_length_hash(
                "source_artifacts_verify_command",
                source_artifacts_verify,
            ),
            "verification_status_command": verification_status,
            **_command_length_hash(
                "verification_status_command",
                verification_status,
            ),
            "verification_exit_code_command": verification_exit_code,
            **_command_length_hash(
                "verification_exit_code_command",
                verification_exit_code,
            ),
            "verification_failures_command": verification_failures,
            **_command_length_hash(
                "verification_failures_command",
                verification_failures,
            ),
            "verification_failures_sha_command": verification_failures_sha,
            **_command_length_hash(
                "verification_failures_sha_command",
                verification_failures_sha,
            ),
            "success_criteria": [
                "source artifact verification exits 0",
                "source artifact verification reports status=match",
                "source artifact verification reports all_match=true",
            ],
        },
        "oracle_parity_blocked": {
            "rerun_command_shell": oracle_progress.get("command_shell"),
            "oracle_helper_refresh_command": oracle_helper_refresh,
            "oracle_helper_refresh_command_nchars": len(oracle_helper_refresh),
            "oracle_helper_refresh_command_sha256": hashlib.sha256(
                oracle_helper_refresh.encode()
            ).hexdigest(),
            "oracle_helper_long_timeout_command": oracle_helper_long_timeout,
            "oracle_helper_long_timeout_s": DEFAULT_ORACLE_LONG_TIMEOUT_S,
            **_command_length_hash(
                "oracle_helper_long_timeout_command",
                oracle_helper_long_timeout,
            ),
            "status_refresh_command": status_refresh,
            **_command_length_hash("status_refresh_command", status_refresh),
            "gap_report_status": oracle_gap_report.get("status"),
            "missing_preconditions": oracle_missing_preconditions,
            "first_missing_precondition": oracle_gap_report.get("first_missing_precondition"),
            "missing_evidence": oracle_missing_evidence,
            "first_missing_evidence": oracle_gap_report.get("first_missing_evidence"),
            "success_criteria": [
                "oracle_gap_report.status is ready",
                "oracle_gap_report.missing_preconditions is empty",
                "oracle_gap_report.missing_evidence is empty",
                "oracle_parity is true",
                "readiness_gates.oracle_parity.ready is true",
            ],
        },
        "kv_backed_decode_not_wired": {
            "resource_plan_refresh_command": resource_plan_refresh,
            **_command_length_hash("resource_plan_refresh_command", resource_plan_refresh),
            "status_refresh_command": status_refresh,
            **_command_length_hash("status_refresh_command", status_refresh),
            "gap_report_status": kv_backed_decode_gap_report.get("status"),
            "missing_evidence": kv_missing_evidence,
            "first_missing_evidence": kv_backed_decode_gap_report.get("first_missing_evidence"),
            "streaming_runner_blocker_count": kv_backed_decode_gap_report.get(
                "streaming_runner_blocker_count"
            ),
            "streaming_runner_blocker_names": kv_backed_decode_gap_report.get(
                "streaming_runner_blocker_names"
            ),
            "streaming_runner_blocker_names_sha256": kv_backed_decode_gap_report.get(
                "streaming_runner_blocker_names_sha256"
            ),
            "streaming_runner_blocker_names_sha256_match": kv_backed_decode_gap_report.get(
                "streaming_runner_blocker_names_sha256_match"
            ),
            "first_streaming_runner_blocker": kv_backed_decode_gap_report.get(
                "first_streaming_runner_blocker"
            ),
            "first_streaming_runner_blocker_sha256": kv_backed_decode_gap_report.get(
                "first_streaming_runner_blocker_sha256"
            ),
            "streaming_decode_loop_blueprint": kv_backed_decode_gap_report.get(
                "streaming_decode_loop_blueprint"
            ),
            "streaming_decode_loop_blueprint_sha256": kv_backed_decode_gap_report.get(
                "streaming_decode_loop_blueprint_sha256"
            ),
            "streaming_decode_loop_status": kv_backed_decode_gap_report.get(
                "streaming_decode_loop_status"
            ),
            "streaming_decode_loop_status_sha256": kv_backed_decode_gap_report.get(
                "streaming_decode_loop_status_sha256"
            ),
            "success_criteria": [
                "kv_backed_decode_gap_report.status is ready",
                "kv_backed_decode_gap_report.missing_evidence is empty",
                "kv_backed_decode_ready is true",
                "readiness_gates.kv_backed_decode.ready is true",
                "e2e_inference_ready is true only after oracle_parity is also true",
            ],
        },
    }


def _remaining_blockers_report(
    *,
    docs_status: dict[str, object],
    blockers: Sequence[dict[str, object]],
    readiness_gates: dict[str, object],
    handoff_summary: dict[str, object],
    next_action_commands: dict[str, object],
) -> dict[str, object]:
    """Return a compact report joining P11 partial items to evidence gaps."""

    checklist_labels = {
        "oracle_parity_blocked": "P11 llama.cpp greedy next-token/logit comparison",
        "kv_backed_decode_not_wired": "P11 text-only c=1 KV-backed decode runner",
    }
    gate_by_kind = {
        "oracle_parity_blocked": "oracle_parity",
        "kv_backed_decode_not_wired": "kv_backed_decode",
    }
    work_items_by_kind = {
        str(item.get("blocker_kind")): item
        for item in handoff_summary.get("blocker_work_queue", [])
        if isinstance(item, dict)
    }
    items: list[dict[str, object]] = []
    for index, blocker in enumerate(blockers):
        kind = str(blocker.get("kind"))
        gate_name = gate_by_kind.get(kind)
        gate = readiness_gates.get(gate_name, {}) if gate_name is not None else {}
        gate_record = gate if isinstance(gate, dict) else {}
        work_item = work_items_by_kind.get(kind, {})
        work_record = work_item if isinstance(work_item, dict) else {}
        next_action = next_action_commands.get(kind, {})
        next_action_record = next_action if isinstance(next_action, dict) else {}
        items.append(
            {
                "blocker_kind": kind,
                "checklist_item": checklist_labels.get(kind),
                "queue_index": work_record.get("queue_index", index),
                "readiness_gate": gate_name,
                "gate_ready": gate_record.get("ready") is True,
                "gate_blocked_by": gate_record.get("blocked_by"),
                "required_evidence": gate_record.get("required_evidence"),
                "gap_report_status": blocker.get("gap_report_status"),
                "first_missing_precondition": blocker.get("first_missing_precondition"),
                "first_missing_evidence": blocker.get("first_missing_evidence"),
                "missing_evidence": list(blocker.get("missing_evidence", [])),
                "recommended_command_kind": work_record.get("recommended_command_kind"),
                "recommended_command": work_record.get("recommended_command"),
                "recommended_command_sha256": work_record.get("recommended_command_sha256"),
                "recommended_command_reason": work_record.get("recommended_command_reason"),
                "success_criteria": list(next_action_record.get("success_criteria", [])),
                "artifact": blocker.get("artifact"),
                "resource_artifact": blocker.get("resource_artifact"),
                "streaming_runner_blocker_names": blocker.get(
                    "streaming_runner_blocker_names"
                ),
                "first_streaming_runner_blocker": blocker.get(
                    "first_streaming_runner_blocker"
                ),
            }
        )
    return {
        "schema_version": 1,
        "status": "blocked" if items else "ready",
        "open_or_partial_items_p0_p12": docs_status.get("open_or_partial_count_p0_p12"),
        "remaining_blocker_count": len(items),
        "remaining_blocker_kinds": [item["blocker_kind"] for item in items],
        "items": items,
        "blocked_gates": list(handoff_summary.get("blocked_gates", [])),
        "next_commands_available_for": list(
            handoff_summary.get("next_commands_available_for", [])
        ),
        "no_claim_policy": dict(handoff_summary.get("no_claim_policy", {})),
    }


def _first_remaining_blocker_report(
    remaining_blockers_report: dict[str, object]
) -> dict[str, object]:
    """Return compact routing metadata for the front remaining blocker."""

    items = remaining_blockers_report.get("items", [])
    first_item = items[0] if isinstance(items, list) and items else None
    first_record = first_item if isinstance(first_item, dict) else None
    return {
        "schema_version": 1,
        "status": "blocked" if first_record is not None else "ready",
        "open_or_partial_items_p0_p12": remaining_blockers_report.get(
            "open_or_partial_items_p0_p12"
        ),
        "remaining_blocker_count": remaining_blockers_report.get(
            "remaining_blocker_count", 0
        ),
        "remaining_blocker_kinds": list(
            remaining_blockers_report.get("remaining_blocker_kinds", [])
        ),
        "blocked_gates": list(remaining_blockers_report.get("blocked_gates", [])),
        "next_commands_available_for": list(
            remaining_blockers_report.get("next_commands_available_for", [])
        ),
        "blocker_kind": first_record.get("blocker_kind") if first_record else None,
        "queue_index": first_record.get("queue_index") if first_record else None,
        "readiness_gate": first_record.get("readiness_gate") if first_record else None,
        "gate_ready": first_record.get("gate_ready") if first_record else None,
        "first_missing_evidence": first_record.get("first_missing_evidence")
        if first_record
        else None,
        "recommended_command_kind": first_record.get("recommended_command_kind")
        if first_record
        else None,
        "recommended_command": first_record.get("recommended_command")
        if first_record
        else None,
        "recommended_command_sha256": first_record.get("recommended_command_sha256")
        if first_record
        else None,
        "success_criteria": list(first_record.get("success_criteria", []))
        if first_record
        else [],
        "item": first_record,
        "no_claim_policy": dict(remaining_blockers_report.get("no_claim_policy", {})),
    }


def _primary_command_metadata(kind: str | None, command: object) -> dict[str, object]:
    """Return stable metadata for a blocker primary command."""

    command_text = command if isinstance(command, str) and command else None
    return {
        "primary_command_kind": kind,
        "primary_command": command_text,
        "primary_command_nchars": len(command_text) if command_text is not None else 0,
        "primary_command_sha256": (
            hashlib.sha256(command_text.encode()).hexdigest()
            if command_text is not None
            else None
        ),
    }


def _recommended_command_metadata(
    kind: str | None,
    command: object,
    *,
    reason: str | None,
) -> dict[str, object]:
    """Return stable metadata for the command automation should run next."""

    command_text = command if isinstance(command, str) and command else None
    return {
        "recommended_command_kind": kind,
        "recommended_command": command_text,
        "recommended_command_nchars": len(command_text) if command_text is not None else 0,
        "recommended_command_sha256": (
            hashlib.sha256(command_text.encode()).hexdigest()
            if command_text is not None
            else None
        ),
        "recommended_command_reason": reason if command_text is not None else None,
    }


def _stable_json_sha256(value: object) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible metadata."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _readiness_gates(
    *,
    oracle_parity: bool,
    kv_decode_dispatch_ready: bool,
    kv_backed_decode_ready: bool,
    oracle_progress: dict[str, object],
    oracle_gap_report: dict[str, object],
    kv_decode_dispatch_progress: dict[str, object],
    kv_backed_decode_gap_report: dict[str, object],
) -> dict[str, object]:
    """Return explicit readiness gates for the remaining StepFun blockers."""

    return {
        "oracle_parity": {
            "ready": oracle_parity,
            "blocked_by": None if oracle_parity else oracle_progress.get("oracle_blocker_kind"),
            "required_evidence": (
                "A StepFun/step35-capable oracle must generate a comparable token and match the "
                "expected next-token text/logit policy recorded in oracle_progress."
            ),
            "expected_next_token_id": oracle_progress.get("expected_next_token_id"),
            "expected_next_token_text": oracle_progress.get("expected_next_token_text"),
            "current_oracle_status": oracle_progress.get("status"),
            "current_oracle_returncode": oracle_progress.get("returncode"),
            "gap_report": oracle_gap_report,
        },
        "kv_backed_decode": {
            "ready": kv_backed_decode_ready,
            "blocked_by": None if kv_backed_decode_ready else "kv_backed_decode_not_wired",
            "dispatch_ready": kv_decode_dispatch_ready,
            "required_evidence": (
                "A streaming runner must materialize prompt KV rows, launch one-token decode KV "
                "write and gated paged attention from the resident cache, then feed final logits "
                "without host-composed layer outputs."
            ),
            "current_evidence": {
                "dispatch_ready": kv_decode_dispatch_ready,
                "decode_span_shape_compatible": kv_decode_dispatch_progress.get(
                    "decode_span_shape_compatible"
                ),
                "prompt_span_shape_compatible": kv_decode_dispatch_progress.get(
                    "prompt_span_shape_compatible"
                ),
                "launch_schedule_operation_count": dict(
                    kv_decode_dispatch_progress.get("launch_schedule", {})
                ).get("operation_count"),
                "launch_schedule_streaming_ready": dict(
                    kv_decode_dispatch_progress.get("launch_schedule", {})
                ).get("streaming_runner_ready"),
                "run_plan_prompt_fits_resource_plan": kv_decode_dispatch_progress.get(
                    "run_plan_prompt_fits_resource_plan"
                ),
                "run_plan_context_fits_resource_plan": kv_decode_dispatch_progress.get(
                    "run_plan_context_fits_resource_plan"
                ),
                "run_plan_input_id_count": dict(
                    kv_decode_dispatch_progress.get("run_plan", {})
                ).get("input_id_count"),
                "run_plan_input_ids_nbytes": dict(
                    kv_decode_dispatch_progress.get("run_plan", {})
                ).get("input_ids_nbytes"),
                "run_plan_input_ids_sha256": dict(
                    kv_decode_dispatch_progress.get("run_plan", {})
                ).get("input_ids_sha256"),
                "run_plan_rendered_prompt_sha256": dict(
                    kv_decode_dispatch_progress.get("run_plan", {})
                ).get("rendered_prompt_sha256"),
                "run_plan_prompt_span_base_offsets_len": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "prompt_span_inputs", {}
                    )
                ).get("base_offsets_len"),
                "run_plan_decode_span_base_offsets_len": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "decode_span_inputs", {}
                    )
                ).get("base_offsets_len"),
                "run_plan_span_input_total_nbytes": dict(
                    kv_decode_dispatch_progress.get("run_plan", {})
                ).get("span_input_total_nbytes"),
                "run_plan_upload_manifest_total_nbytes": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "span_input_upload_manifest", {}
                    )
                ).get("total_nbytes"),
                "run_plan_upload_manifest_entry_count": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "span_input_upload_manifest", {}
                    )
                ).get("entry_count"),
                "run_plan_decode_input_upload_entry_count": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "decode_input_upload_plan", {}
                    )
                ).get("entry_count"),
                "run_plan_decode_input_upload_total_nbytes": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "decode_input_upload_plan", {}
                    )
                ).get("total_nbytes"),
                "run_plan_decode_input_upload_checks_passed": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "decode_input_upload_plan", {}
                    )
                ).get("all_consistency_checks_passed")
                is True,
                "run_plan_host_payload_total_nbytes": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "span_input_host_payloads", {}
                    )
                ).get("total_nbytes"),
                "run_plan_host_payload_entry_count": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "span_input_host_payloads", {}
                    )
                ).get("entry_count"),
                "run_plan_decode_input_upload_entry_count": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "decode_input_upload_plan", {}
                    )
                ).get("entry_count"),
                "run_plan_decode_input_upload_total_nbytes": dict(
                    dict(kv_decode_dispatch_progress.get("run_plan", {})).get(
                        "decode_input_upload_plan", {}
                    )
                ).get("total_nbytes"),
                "run_plan_streaming_ready": dict(
                    kv_decode_dispatch_progress.get("run_plan", {})
                ).get("streaming_runner_ready"),
                "resident_prompt_smoke": "host_composed_layer_prefix",
            },
            "gap_report": kv_backed_decode_gap_report,
        },
        "e2e_inference": {
            "ready": oracle_parity and kv_backed_decode_ready,
            "blocked_by": [
                name
                for name, ready in (
                    ("oracle_parity", oracle_parity),
                    ("kv_backed_decode", kv_backed_decode_ready),
                )
                if not ready
            ],
            "required_evidence": (
                "Both oracle parity and KV-backed decode readiness must be true before StepFun "
                "text-only GGUF inference is marked ready."
            ),
        },
    }


def _handoff_summary(
    *,
    docs_status: dict[str, object],
    blockers: Sequence[dict[str, object]],
    readiness_gates: dict[str, object],
    next_action_commands: dict[str, object],
    all_layer_prompt_smoke: bool,
    oracle_progress: dict[str, object],
    oracle_gap_report: dict[str, object],
    kv_decode_dispatch_ready: bool,
    kv_decode_dispatch_progress: dict[str, object],
    kv_backed_decode_gap_report: dict[str, object],
) -> dict[str, object]:
    """Return a compact status summary for cross-session handoff."""

    blocker_kinds = [str(blocker.get("kind")) for blocker in blockers]
    ready_gates: list[str] = []
    blocked_gates: list[str] = []
    for name, gate_obj in readiness_gates.items():
        gate = dict(gate_obj) if isinstance(gate_obj, dict) else {}
        if gate.get("ready") is True:
            ready_gates.append(str(name))
        else:
            blocked_gates.append(str(name))
    launch_schedule = dict(kv_decode_dispatch_progress.get("launch_schedule", {}))
    run_plan = dict(kv_decode_dispatch_progress.get("run_plan", {}))
    decode_input_upload_plan = dict(run_plan.get("decode_input_upload_plan", {}))
    blocker_work_queue: list[dict[str, object]] = []
    for queue_index, blocker_kind in enumerate(blocker_kinds):
        if blocker_kind == "oracle_parity_blocked":
            oracle_action = dict(next_action_commands.get(blocker_kind, {}))
            oracle_helper_command = oracle_action.get("oracle_helper_refresh_command")
            oracle_helper_command_text = (
                oracle_helper_command if isinstance(oracle_helper_command, str) else None
            )
            oracle_helper_long_timeout_command = oracle_action.get(
                "oracle_helper_long_timeout_command"
            )
            oracle_helper_long_timeout_command_text = (
                oracle_helper_long_timeout_command
                if isinstance(oracle_helper_long_timeout_command, str)
                else None
            )
            oracle_recommended_command_kind = "oracle_helper_long_timeout_command"
            oracle_recommended_command = oracle_helper_long_timeout_command_text
            oracle_recommended_reason = "oracle_timeout_retry_with_longer_timeout"
            if oracle_recommended_command is None:
                oracle_recommended_command_kind = "oracle_helper_refresh_command"
                oracle_recommended_command = oracle_helper_command_text
                oracle_recommended_reason = "regenerate_oracle_artifact"
            if oracle_recommended_command is None:
                oracle_recommended_command_kind = "rerun_command_shell"
                oracle_recommended_command = oracle_action.get("rerun_command_shell")
                oracle_recommended_reason = "rerun_recorded_oracle_command"
            blocker_work_queue.append(
                {
                    "blocker_kind": blocker_kind,
                    "work_item_schema_version": 1,
                    "queue_index": queue_index,
                    "is_first": queue_index == 0,
                    "gate": "oracle_parity",
                    "command_available": blocker_kind in next_action_commands,
                    **_primary_command_metadata(
                        "rerun_command_shell",
                        oracle_action.get("rerun_command_shell"),
                    ),
                    **_recommended_command_metadata(
                        oracle_recommended_command_kind,
                        oracle_recommended_command,
                        reason=oracle_recommended_reason,
                    ),
                    "helper_command_kind": "oracle_helper_refresh_command",
                    "helper_command": oracle_helper_command_text,
                    "helper_command_nchars": (
                        len(oracle_helper_command_text)
                        if oracle_helper_command_text is not None
                        else 0
                    ),
                    "helper_command_sha256": (
                        hashlib.sha256(oracle_helper_command_text.encode()).hexdigest()
                        if oracle_helper_command_text is not None
                        else None
                    ),
                    "long_timeout_helper_command_kind": "oracle_helper_long_timeout_command",
                    "long_timeout_helper_command": oracle_helper_long_timeout_command_text,
                    "long_timeout_helper_command_nchars": (
                        len(oracle_helper_long_timeout_command_text)
                        if oracle_helper_long_timeout_command_text is not None
                        else 0
                    ),
                    "long_timeout_helper_command_sha256": (
                        hashlib.sha256(
                            oracle_helper_long_timeout_command_text.encode()
                        ).hexdigest()
                        if oracle_helper_long_timeout_command_text is not None
                        else None
                    ),
                    "long_timeout_s": oracle_action.get("oracle_helper_long_timeout_s"),
                    "gap_report_status": oracle_gap_report.get("status"),
                    "current_status": oracle_gap_report.get("oracle_status"),
                    "current_returncode": oracle_gap_report.get("returncode"),
                    "elapsed_s": oracle_gap_report.get("elapsed_s"),
                    "timeout_s": oracle_gap_report.get("timeout_s"),
                    "diagnostic_logs": oracle_progress.get("diagnostic_logs") is True,
                    "first_missing_precondition": oracle_gap_report.get(
                        "first_missing_precondition"
                    ),
                    "first_missing_evidence": oracle_gap_report.get("first_missing_evidence"),
                    "oracle_blocker_kind": oracle_gap_report.get("oracle_blocker_kind"),
                }
            )
        elif blocker_kind == "kv_backed_decode_not_wired":
            kv_action = dict(next_action_commands.get(blocker_kind, {}))
            kv_resource_command = kv_action.get("resource_plan_refresh_command")
            blocker_work_queue.append(
                {
                    "blocker_kind": blocker_kind,
                    "work_item_schema_version": 1,
                    "queue_index": queue_index,
                    "is_first": queue_index == 0,
                    "gate": "kv_backed_decode",
                    "command_available": blocker_kind in next_action_commands,
                    **_primary_command_metadata(
                        "resource_plan_refresh_command",
                        kv_resource_command,
                    ),
                    **_recommended_command_metadata(
                        "resource_plan_refresh_command",
                        kv_resource_command,
                        reason="refresh_kv_resource_and_run_plan_artifact",
                    ),
                    "gap_report_status": kv_backed_decode_gap_report.get("status"),
                    "operation_count": kv_backed_decode_gap_report.get("operation_count"),
                    "streaming_runner_blocker_count": kv_backed_decode_gap_report.get(
                        "streaming_runner_blocker_count"
                    ),
                    "streaming_runner_blocker_names": kv_backed_decode_gap_report.get(
                        "streaming_runner_blocker_names"
                    ),
                    "streaming_runner_blocker_names_sha256": kv_backed_decode_gap_report.get(
                        "streaming_runner_blocker_names_sha256"
                    ),
                    "streaming_runner_blocker_names_sha256_match": kv_backed_decode_gap_report.get(
                        "streaming_runner_blocker_names_sha256_match"
                    ),
                    "first_missing_evidence": kv_backed_decode_gap_report.get(
                        "first_missing_evidence"
                    ),
                    "first_streaming_runner_blocker": kv_backed_decode_gap_report.get(
                        "first_streaming_runner_blocker"
                    ),
                    "first_streaming_runner_blocker_sha256": kv_backed_decode_gap_report.get(
                        "first_streaming_runner_blocker_sha256"
                    ),
                    "streaming_decode_loop_blueprint": kv_backed_decode_gap_report.get(
                        "streaming_decode_loop_blueprint"
                    ),
                    "streaming_decode_loop_blueprint_sha256": kv_backed_decode_gap_report.get(
                        "streaming_decode_loop_blueprint_sha256"
                    ),
                    "streaming_decode_loop_status": kv_backed_decode_gap_report.get(
                        "streaming_decode_loop_status"
                    ),
                    "streaming_decode_loop_status_sha256": kv_backed_decode_gap_report.get(
                        "streaming_decode_loop_status_sha256"
                    ),
                }
            )
        else:
            blocker_work_queue.append(
                {
                    "blocker_kind": blocker_kind,
                    "work_item_schema_version": 1,
                    "queue_index": queue_index,
                    "is_first": queue_index == 0,
                    "gate": None,
                    "command_available": blocker_kind in next_action_commands,
                    **_primary_command_metadata(None, None),
                    **_recommended_command_metadata(None, None, reason=None),
                    "gap_report_status": None,
                }
            )
    blocker_work_queue_sha256 = _stable_json_sha256(blocker_work_queue)
    blocker_recommended_commands = [
        {
            "blocker_kind": item.get("blocker_kind"),
            "queue_index": item.get("queue_index"),
            "recommended_command_kind": item.get("recommended_command_kind"),
            "recommended_command": item.get("recommended_command"),
            "recommended_command_nchars": item.get("recommended_command_nchars"),
            "recommended_command_sha256": item.get("recommended_command_sha256"),
            "recommended_command_reason": item.get("recommended_command_reason"),
        }
        for item in blocker_work_queue
    ]
    blocker_recommended_commands_sha256 = _stable_json_sha256(
        blocker_recommended_commands
    )
    first_blocker_work_item = blocker_work_queue[0] if blocker_work_queue else None
    first_blocker_work_item_sha256 = (
        _stable_json_sha256(first_blocker_work_item)
        if first_blocker_work_item is not None
        else None
    )
    blocker_work_queue_meta = {
        "schema_version": 1,
        "count": len(blocker_work_queue),
        "sha256": blocker_work_queue_sha256,
        "first_blocker_kind": first_blocker_work_item["blocker_kind"] if first_blocker_work_item else None,
        "first_work_item_schema_version": (
            first_blocker_work_item["work_item_schema_version"] if first_blocker_work_item else None
        ),
        "first_work_item_sha256": first_blocker_work_item_sha256,
        "first_recommended_command_kind": (
            first_blocker_work_item.get("recommended_command_kind")
            if first_blocker_work_item
            else None
        ),
        "first_recommended_command_sha256": (
            first_blocker_work_item.get("recommended_command_sha256")
            if first_blocker_work_item
            else None
        ),
        "recommended_commands_sha256": blocker_recommended_commands_sha256,
    }
    return {
        "schema_version": HANDOFF_SUMMARY_SCHEMA_VERSION,
        "status": "blocked" if blocker_kinds else "ready",
        "open_or_partial_items_p0_p12": docs_status.get("open_or_partial_count_p0_p12"),
        "open_blocker_count": len(blocker_kinds),
        "open_blockers": blocker_kinds,
        "blocker_work_queue_schema_version": blocker_work_queue_meta["schema_version"],
        "blocker_work_queue_count": blocker_work_queue_meta["count"],
        "blocker_work_queue_sha256": blocker_work_queue_meta["sha256"],
        "blocker_work_queue_meta": blocker_work_queue_meta,
        "blocker_work_queue": blocker_work_queue,
        "blocker_recommended_commands": blocker_recommended_commands,
        "blocker_recommended_commands_sha256": blocker_recommended_commands_sha256,
        "first_blocker_work_item": first_blocker_work_item,
        "first_blocker_work_item_sha256": first_blocker_work_item_sha256,
        "exit_codes": {
            "ready": READY_EXIT_CODE,
            "source_artifact_mismatch": SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
            "blocked_when_fail_on_blocked": BLOCKED_EXIT_CODE,
            "current_with_fail_on_blocked": (
                READY_EXIT_CODE if not blocker_kinds else BLOCKED_EXIT_CODE
            ),
        },
        "compact_output_modes": {
            "summary_only": "handoff_summary",
            "handoff_summary_sha_only": "handoff_summary_sha256",
            "schema_versions_only": "schema_versions",
            "status_integrity_only": "status_integrity",
            "status_integrity_sha_only": "status_integrity_sha256",
            "status_integrity_failures_only": "status_integrity.failed_checks",
            "persisted_status_integrity_only": "persisted_status_integrity",
            "persisted_status_integrity_failures_only": (
                "persisted_status_integrity.failed_checks"
            ),
            "readiness_summary_only": "readiness_summary",
            "readiness_summary_sha_only": "readiness_summary_sha256",
            "readiness_gates_only": "readiness_gates",
            "readiness_gates_sha_only": "readiness_gates_sha256",
            "blocked_gates_only": "blocked_gates",
            "blocked_gates_sha_only": "blocked_gates_sha256",
            "remaining_blockers_report_only": "remaining_blockers_report",
            "remaining_blockers_report_sha_only": "remaining_blockers_report_sha256",
            "first_remaining_blocker_report_only": "first_remaining_blocker_report",
            "first_remaining_blocker_report_sha_only": "first_remaining_blocker_report_sha256",
            "source_artifacts_sha_only": "source_artifacts_sha256",
            "oracle_wrapper_timeout_source_only": "source_artifacts.oracle_wrapper_timeout",
            "oracle_wrapper_timeout_source_sha_only": (
                "source_artifacts.oracle_wrapper_timeout.sha256"
            ),
            "text_resource_source_only": "source_artifacts.text_resource",
            "text_resource_source_sha_only": "source_artifacts.text_resource.sha256",
            "next_action_commands_only": "next_action_commands",
            "next_action_commands_sha_only": "next_action_commands_sha256",
            "blocker_kinds_only": "blocker_kinds",
            "blocker_kinds_sha_only": "blocker_kinds_sha256",
            "kv_streaming_blockers_only": "kv_streaming_runner_blocker_names",
            "kv_streaming_blockers_sha_only": "kv_streaming_runner_blocker_names_sha256",
            "kv_first_streaming_blocker_only": (
                "kv_backed_decode_gap_report.first_streaming_runner_blocker"
            ),
            "kv_first_streaming_blocker_sha_only": (
                "kv_backed_decode_gap_report.first_streaming_runner_blocker_sha256"
            ),
            "kv_streaming_blueprint_only": (
                "kv_backed_decode_gap_report.streaming_decode_loop_blueprint"
            ),
            "kv_streaming_blueprint_sha_only": (
                "kv_backed_decode_gap_report.streaming_decode_loop_blueprint_sha256"
            ),
            "kv_streaming_loop_status_only": (
                "kv_backed_decode_gap_report.streaming_decode_loop_status"
            ),
            "kv_streaming_loop_status_sha_only": (
                "kv_backed_decode_gap_report.streaming_decode_loop_status_sha256"
            ),
            "kv_streaming_loop_next_action_only": (
                "kv_backed_decode_gap_report.streaming_decode_loop_status.next_action"
            ),
            "kv_streaming_loop_next_action_sha_only": (
                "kv_backed_decode_gap_report.streaming_decode_loop_status.next_action_sha256"
            ),
            "status_refresh_command_only": (
                "next_action_commands.oracle_parity_blocked.status_refresh_command"
            ),
            "status_refresh_command_sha_only": (
                "next_action_commands.oracle_parity_blocked.status_refresh_command_sha256"
            ),
            "source_verify_command_only": (
                "next_action_commands.handoff_integrity.source_artifacts_verify_command"
            ),
            "source_verify_command_sha_only": (
                "next_action_commands.handoff_integrity.source_artifacts_verify_command_sha256"
            ),
            "verification_status_command_only": (
                "next_action_commands.handoff_integrity.verification_status_command"
            ),
            "verification_status_command_sha_only": (
                "next_action_commands.handoff_integrity.verification_status_command_sha256"
            ),
            "verification_exit_code_command_only": (
                "next_action_commands.handoff_integrity.verification_exit_code_command"
            ),
            "verification_exit_code_command_sha_only": (
                "next_action_commands.handoff_integrity.verification_exit_code_command_sha256"
            ),
            "verification_failures_command_only": (
                "next_action_commands.handoff_integrity.verification_failures_command"
            ),
            "verification_failures_command_sha_only": (
                "next_action_commands.handoff_integrity.verification_failures_command_sha256"
            ),
            "verification_failures_sha_command_only": (
                "next_action_commands.handoff_integrity.verification_failures_sha_command"
            ),
            "verification_failures_sha_command_sha_only": (
                "next_action_commands.handoff_integrity.verification_failures_sha_command_sha256"
            ),
            "kv_resource_command_only": (
                "next_action_commands.kv_backed_decode_not_wired.resource_plan_refresh_command"
            ),
            "kv_resource_command_sha_only": (
                "next_action_commands.kv_backed_decode_not_wired.resource_plan_refresh_command_sha256"
            ),
            "oracle_helper_command_only": (
                "next_action_commands.oracle_parity_blocked.oracle_helper_refresh_command"
            ),
            "oracle_helper_command_sha_only": (
                "next_action_commands.oracle_parity_blocked.oracle_helper_refresh_command_sha256"
            ),
            "oracle_helper_long_timeout_command_only": (
                "next_action_commands.oracle_parity_blocked.oracle_helper_long_timeout_command"
            ),
            "oracle_helper_long_timeout_command_sha_only": (
                "next_action_commands.oracle_parity_blocked.oracle_helper_long_timeout_command_sha256"
            ),
            "blocker_work_queue_only": "handoff_summary.blocker_work_queue",
            "blocker_work_queue_meta_only": "handoff_summary.blocker_work_queue_meta",
            "blocker_work_queue_sha_only": "handoff_summary.blocker_work_queue_sha256",
            "blocker_recommended_commands_only": "handoff_summary.blocker_recommended_commands",
            "blocker_recommended_commands_sha_only": "handoff_summary.blocker_recommended_commands_sha256",
            "first_blocker_sha_only": "handoff_summary.first_blocker_work_item_sha256",
            "first_blocker_only": "handoff_summary.first_blocker_work_item",
            "first_blocker_recommended_command_only": (
                "handoff_summary.first_blocker_work_item.recommended_command"
            ),
            "first_blocker_recommended_command_sha_only": (
                "handoff_summary.first_blocker_work_item.recommended_command_sha256"
            ),
            "fail_on_blocked_preserves_payload": True,
        },
        "ready_gates": ready_gates,
        "blocked_gates": blocked_gates,
        "ready_signals": {
            "all_layer_prompt_smoke": all_layer_prompt_smoke,
            "oracle_target_recorded": oracle_progress.get("expected_next_token_id") is not None,
            "kv_decode_dispatch_ready": kv_decode_dispatch_ready,
            "kv_launch_schedule_recorded": bool(launch_schedule.get("operation_count")),
            "kv_decode_run_plan_recorded": bool(run_plan.get("prompt_length")),
            "kv_decode_input_upload_plan_recorded": bool(
                decode_input_upload_plan.get("entry_count")
            ),
            "kv_streaming_decode_loop_blueprint_recorded": bool(
                dict(kv_backed_decode_gap_report.get("streaming_decode_loop_blueprint", {})).get(
                    "recorded"
                )
            ),
            "kv_streaming_decode_loop_status_recorded": bool(
                dict(kv_backed_decode_gap_report.get("streaming_decode_loop_status", {})).get(
                    "recorded"
                )
            ),
        },
        "oracle_gap_report": {
            "status": oracle_gap_report.get("status"),
            "precondition_count": oracle_gap_report.get("precondition_count"),
            "validated_precondition_count": oracle_gap_report.get(
                "validated_precondition_count"
            ),
            "missing_precondition_count": oracle_gap_report.get(
                "missing_precondition_count"
            ),
            "first_missing_precondition": oracle_gap_report.get("first_missing_precondition"),
            "missing_evidence_count": oracle_gap_report.get("missing_evidence_count"),
            "first_missing_evidence": oracle_gap_report.get("first_missing_evidence"),
            "missing_evidence": list(oracle_gap_report.get("missing_evidence", [])),
            "oracle_blocker_kind": oracle_gap_report.get("oracle_blocker_kind"),
            "oracle_status": oracle_gap_report.get("oracle_status"),
            "elapsed_s": oracle_gap_report.get("elapsed_s"),
            "timeout_s": oracle_gap_report.get("timeout_s"),
            "expected_next_token_id": oracle_gap_report.get("expected_next_token_id"),
            "expected_next_token_text": oracle_gap_report.get("expected_next_token_text"),
        },
        "kv_decode_input_upload_plan": {
            "entry_count": decode_input_upload_plan.get("entry_count"),
            "total_nbytes": decode_input_upload_plan.get("total_nbytes"),
            "upload_order": list(decode_input_upload_plan.get("upload_order", [])),
            "cleanup_order": list(decode_input_upload_plan.get("cleanup_order", [])),
            "all_consistency_checks_passed": decode_input_upload_plan.get(
                "all_consistency_checks_passed"
            )
            is True,
        },
        "kv_backed_decode_gap_report": {
            "status": kv_backed_decode_gap_report.get("status"),
            "precondition_count": kv_backed_decode_gap_report.get("precondition_count"),
            "validated_precondition_count": kv_backed_decode_gap_report.get(
                "validated_precondition_count"
            ),
            "missing_precondition_count": kv_backed_decode_gap_report.get(
                "missing_precondition_count"
            ),
            "missing_evidence_count": kv_backed_decode_gap_report.get(
                "missing_evidence_count"
            ),
            "first_missing_evidence": kv_backed_decode_gap_report.get("first_missing_evidence"),
            "missing_evidence": list(kv_backed_decode_gap_report.get("missing_evidence", [])),
            "operation_count": kv_backed_decode_gap_report.get("operation_count"),
            "streaming_runner_blocker_count": kv_backed_decode_gap_report.get(
                "streaming_runner_blocker_count"
            ),
            "streaming_runner_blocker_names": kv_backed_decode_gap_report.get(
                "streaming_runner_blocker_names"
            ),
            "streaming_runner_blocker_names_sha256": kv_backed_decode_gap_report.get(
                "streaming_runner_blocker_names_sha256"
            ),
            "streaming_runner_blocker_names_sha256_match": kv_backed_decode_gap_report.get(
                "streaming_runner_blocker_names_sha256_match"
            ),
            "first_streaming_runner_blocker": kv_backed_decode_gap_report.get(
                "first_streaming_runner_blocker"
            ),
            "first_streaming_runner_blocker_sha256": kv_backed_decode_gap_report.get(
                "first_streaming_runner_blocker_sha256"
            ),
            "streaming_decode_loop_blueprint": kv_backed_decode_gap_report.get(
                "streaming_decode_loop_blueprint"
            ),
            "streaming_decode_loop_blueprint_sha256": kv_backed_decode_gap_report.get(
                "streaming_decode_loop_blueprint_sha256"
            ),
            "streaming_decode_loop_status": kv_backed_decode_gap_report.get(
                "streaming_decode_loop_status"
            ),
            "streaming_decode_loop_status_sha256": kv_backed_decode_gap_report.get(
                "streaming_decode_loop_status_sha256"
            ),
            "upload_total_nbytes": kv_backed_decode_gap_report.get("upload_total_nbytes"),
        },
        "blocked_signals": {
            "oracle_parity": "oracle_parity" in blocked_gates,
            "kv_backed_decode": "kv_backed_decode" in blocked_gates,
            "e2e_inference": "e2e_inference" in blocked_gates,
        },
        "next_commands_available_for": [
            kind for kind in blocker_kinds if kind in next_action_commands
        ],
        "no_claim_policy": {
            "performance_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "reason": (
                "StepFun performance or e2e readiness claims require oracle_parity and "
                "kv_backed_decode readiness gates to pass first."
            ),
        },
    }


def build_status(
    prompt_artifact: Path,
    oracle_artifact: Path,
    docs_path: Path = DEFAULT_DOCS_PATH,
    resource_artifact: Path = DEFAULT_RESOURCE_ARTIFACT,
) -> dict[str, object]:
    prompt = _load(prompt_artifact)
    oracle = _load(oracle_artifact)
    resource = _load(resource_artifact)
    docs_status = _docs_checklist_status(docs_path)
    all_layer_prompt_smoke = (
        prompt.get("status") == "partial_prompt_smoke"
        and prompt.get("execution_mode") == "chunked"
        and prompt.get("layer_count") == 45
        and prompt.get("skipped_layers") == []
        and prompt.get("no_vision_projector_mtp_slots") is True
        and prompt.get("memory_stats_after_free", {}).get("active_allocations") == 0
        and prompt.get("memory_stats_after_free", {}).get("current_allocated_bytes") == 0
    )
    oracle_progress = _oracle_progress(oracle)
    oracle_gap_report = _oracle_gap_report(oracle_progress)
    oracle_parity = oracle_gap_report["status"] == "ready"
    kv_decode_dispatch_progress = _kv_decode_dispatch_progress(resource)
    kv_decode_dispatch_ready = kv_decode_dispatch_progress["all_registered"] is True
    kv_backed_decode_gap_report = _kv_backed_decode_gap_report(kv_decode_dispatch_progress)
    kv_backed_decode_ready = kv_backed_decode_gap_report["status"] == "ready"
    e2e_inference_ready = oracle_parity and kv_backed_decode_ready
    readiness_gates = _readiness_gates(
        oracle_parity=oracle_parity,
        kv_decode_dispatch_ready=kv_decode_dispatch_ready,
        kv_backed_decode_ready=kv_backed_decode_ready,
        oracle_progress=oracle_progress,
        oracle_gap_report=oracle_gap_report,
        kv_decode_dispatch_progress=kv_decode_dispatch_progress,
        kv_backed_decode_gap_report=kv_backed_decode_gap_report,
    )
    next_action_commands = _next_action_commands(
        oracle_progress=oracle_progress,
        oracle_gap_report=oracle_gap_report,
        kv_backed_decode_gap_report=kv_backed_decode_gap_report,
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
        resource_artifact=resource_artifact,
        docs_path=docs_path,
    )
    next_action_commands_sha256 = _stable_json_sha256(next_action_commands)
    readiness_gates_sha256 = _stable_json_sha256(readiness_gates)
    blockers: list[dict[str, object]] = []
    if not oracle_parity:
        blockers.append(
            {
                "kind": "oracle_parity_blocked",
                "detail": oracle.get("oracle_blocker_detail")
                or "llama.cpp/CPU oracle result has not matched the StepFun artifact yet",
                "artifact": str(oracle_artifact),
                "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
                "expected_next_token_id": oracle_progress.get("expected_next_token_id"),
                "expected_next_token_text": oracle_progress.get("expected_next_token_text"),
                "elapsed_s": oracle_progress.get("elapsed_s"),
                "timeout_s": oracle_progress.get("timeout_s"),
                "gap_report_status": oracle_gap_report.get("status"),
                "first_missing_precondition": oracle_gap_report.get("first_missing_precondition"),
                "missing_evidence": list(oracle_gap_report.get("missing_evidence", [])),
                "first_missing_evidence": oracle_gap_report.get("first_missing_evidence"),
            }
        )
    if not kv_backed_decode_ready:
        blockers.append(
            {
                "kind": "kv_backed_decode_not_wired",
                "detail": (
                    "KV dispatch registry keys are recorded as ready in the text resource plan; "
                    "current all-layer prompt smoke is still host-composed prefill/logits, so the "
                    "final KV-backed one-token decode runner remains open."
                ),
                "artifact": str(prompt_artifact),
                "resource_artifact": str(resource_artifact),
                "kv_decode_dispatch_ready": kv_decode_dispatch_ready,
                "gap_report_status": kv_backed_decode_gap_report.get("status"),
                "missing_evidence": list(kv_backed_decode_gap_report.get("missing_evidence", [])),
                "streaming_runner_blocker_count": kv_backed_decode_gap_report.get(
                    "streaming_runner_blocker_count"
                ),
                "streaming_runner_blocker_names": kv_backed_decode_gap_report.get(
                    "streaming_runner_blocker_names"
                ),
                "streaming_runner_blocker_names_sha256": kv_backed_decode_gap_report.get(
                    "streaming_runner_blocker_names_sha256"
                ),
                "streaming_runner_blocker_names_sha256_match": kv_backed_decode_gap_report.get(
                    "streaming_runner_blocker_names_sha256_match"
                ),
                "first_streaming_runner_blocker": kv_backed_decode_gap_report.get(
                    "first_streaming_runner_blocker"
                ),
                "first_streaming_runner_blocker_sha256": kv_backed_decode_gap_report.get(
                    "first_streaming_runner_blocker_sha256"
                ),
                "streaming_decode_loop_blueprint": kv_backed_decode_gap_report.get(
                    "streaming_decode_loop_blueprint"
                ),
                "streaming_decode_loop_blueprint_sha256": kv_backed_decode_gap_report.get(
                    "streaming_decode_loop_blueprint_sha256"
                ),
            }
        )
    next_actions = []
    if not oracle_parity:
        next_actions.append(
            {
                "blocker_kind": "oracle_parity_blocked",
                "action": (
                    "Build or locate a StepFun/step35-capable llama.cpp or CPU oracle, rerun "
                    "scripts/stepfun_llamacpp_oracle.py --execute, and record exact token/logit comparison."
                ),
            }
        )
    if not kv_backed_decode_ready:
        next_actions.append(
            {
                "blocker_kind": "kv_backed_decode_not_wired",
                "action": (
                    "Replace the host-composed layer-prefix prompt smoke with a KV-backed one-token decode runner "
                    "using StepFunResidentSession weight/KV ownership and the validated layer probes."
                ),
            }
        )
    handoff_summary = _handoff_summary(
        docs_status=docs_status,
        blockers=blockers,
        readiness_gates=readiness_gates,
        next_action_commands=next_action_commands,
        all_layer_prompt_smoke=all_layer_prompt_smoke,
        oracle_progress=oracle_progress,
        oracle_gap_report=oracle_gap_report,
        kv_decode_dispatch_ready=kv_decode_dispatch_ready,
        kv_decode_dispatch_progress=kv_decode_dispatch_progress,
        kv_backed_decode_gap_report=kv_backed_decode_gap_report,
    )
    handoff_summary_sha256 = _stable_json_sha256(handoff_summary)
    remaining_blockers_report = _remaining_blockers_report(
        docs_status=docs_status,
        blockers=blockers,
        readiness_gates=readiness_gates,
        handoff_summary=handoff_summary,
        next_action_commands=next_action_commands,
    )
    remaining_blockers_report_sha256 = _stable_json_sha256(remaining_blockers_report)
    first_remaining_blocker_report = _first_remaining_blocker_report(
        remaining_blockers_report
    )
    first_remaining_blocker_report_sha256 = _stable_json_sha256(
        first_remaining_blocker_report
    )
    blocker_kinds = list(handoff_summary["open_blockers"])
    blocker_kinds_sha256 = _stable_json_sha256(blocker_kinds)
    blocked_gates = list(handoff_summary["blocked_gates"])
    blocked_gates_sha256 = _stable_json_sha256(blocked_gates)
    source_artifacts = _source_artifacts(
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
        resource_artifact=resource_artifact,
        docs_path=docs_path,
    )
    source_artifacts_sha256 = _stable_json_sha256(source_artifacts)
    schema_versions = {
        "status": STATUS_SCHEMA_VERSION,
        "readiness_summary": READINESS_SUMMARY_SCHEMA_VERSION,
        "handoff_summary": HANDOFF_SUMMARY_SCHEMA_VERSION,
        "blocker_work_queue": handoff_summary["blocker_work_queue_schema_version"],
        "first_blocker_work_item": handoff_summary["blocker_work_queue_meta"][
            "first_work_item_schema_version"
        ],
    }
    readiness_summary = {
        "schema_version": READINESS_SUMMARY_SCHEMA_VERSION,
        "status": "blocked" if blockers else "ready",
        "oracle_parity": oracle_parity,
        "kv_decode_dispatch_ready": kv_decode_dispatch_ready,
        "kv_backed_decode_ready": kv_backed_decode_ready,
        "e2e_inference_ready": e2e_inference_ready,
        "open_or_partial_items_p0_p12": docs_status.get("open_or_partial_count_p0_p12"),
        "open_blocker_count": handoff_summary["open_blocker_count"],
        "handoff_summary_sha256": handoff_summary_sha256,
        "source_artifacts_sha256": source_artifacts_sha256,
        "readiness_gates_sha256": readiness_gates_sha256,
        "next_action_commands_sha256": next_action_commands_sha256,
        "blocker_kinds_sha256": blocker_kinds_sha256,
        "blocked_gates_sha256": blocked_gates_sha256,
        "first_blocker_kind": handoff_summary["blocker_work_queue_meta"]["first_blocker_kind"],
        "first_blocker_work_item_sha256": handoff_summary[
            "first_blocker_work_item_sha256"
        ],
        "blocker_work_queue_count": handoff_summary["blocker_work_queue_count"],
        "blocker_work_queue_sha256": handoff_summary["blocker_work_queue_sha256"],
        "fail_on_blocked_exit_code": handoff_summary["exit_codes"][
            "current_with_fail_on_blocked"
        ],
        "performance_claim_allowed": handoff_summary["no_claim_policy"][
            "performance_claim_allowed"
        ],
        "e2e_inference_claim_allowed": handoff_summary["no_claim_policy"][
            "e2e_inference_claim_allowed"
        ],
    }
    readiness_summary_sha256 = _stable_json_sha256(readiness_summary)
    status = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "schema_versions": schema_versions,
        "status": "blocked" if blockers else "ready",
        "model": "Step-3.7-flash-Q3_K_L",
        "source_artifacts": source_artifacts,
        "source_artifacts_sha256": source_artifacts_sha256,
        "backend": prompt.get("backend", "hip_gfx1151"),
        "prompt_artifact": str(prompt_artifact),
        "oracle_artifact": str(oracle_artifact),
        "text_resource_artifact": str(resource_artifact),
        "all_layer_prompt_smoke": all_layer_prompt_smoke,
        "all_layer_prompt_next_token_id": prompt.get("next_token_id"),
        "all_layer_prompt_next_token_text": prompt.get("next_token_text"),
        "all_layer_prompt_peak_resident_weight_nbytes": prompt.get("peak_resident_weight_nbytes"),
        "oracle_parity": oracle_parity,
        "oracle_status": oracle.get("status"),
        "oracle_elapsed_s": oracle.get("elapsed_s"),
        "oracle_llama_cpp_version": oracle.get("llama_cpp_version"),
        "oracle_stdout_len": len(str(oracle.get("stdout", ""))),
        "oracle_stderr_len": len(str(oracle.get("stderr", ""))),
        "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
        "step35_supported_by_local_llama_cpp": oracle.get("step35_supported"),
        "oracle_progress": oracle_progress,
        "oracle_gap_report": oracle_gap_report,
        "linear_projection_progress": _linear_projection_progress(prompt),
        "kv_decode_dispatch_progress": kv_decode_dispatch_progress,
        "kv_decode_dispatch_ready": kv_decode_dispatch_ready,
        "kv_backed_decode_gap_report": kv_backed_decode_gap_report,
        "kv_backed_decode_ready": kv_backed_decode_ready,
        "e2e_inference_ready": e2e_inference_ready,
        "readiness_gates": readiness_gates,
        "readiness_gates_sha256": readiness_gates_sha256,
        "readiness_summary": readiness_summary,
        "readiness_summary_sha256": readiness_summary_sha256,
        "handoff_summary": handoff_summary,
        "handoff_summary_sha256": handoff_summary_sha256,
        "remaining_blockers_report": remaining_blockers_report,
        "remaining_blockers_report_sha256": remaining_blockers_report_sha256,
        "first_remaining_blocker_report": first_remaining_blocker_report,
        "first_remaining_blocker_report_sha256": first_remaining_blocker_report_sha256,
        "blocker_kinds": blocker_kinds,
        "blocker_kinds_sha256": blocker_kinds_sha256,
        "blocked_gates": blocked_gates,
        "blocked_gates_sha256": blocked_gates_sha256,
        "blockers": blockers,
        "next_actions": next_actions,
        "next_action_commands": next_action_commands,
        "next_action_commands_sha256": next_action_commands_sha256,
        "docs_checklist": docs_status,
        "note": (
            "Host-composed all-layer prompt smoke is present; true e2e inference still needs "
            "oracle parity and KV-backed decode."
        ),
    }
    status_integrity = _status_integrity(status)
    status["status_integrity"] = status_integrity
    status["status_integrity_sha256"] = _stable_json_sha256(status_integrity)
    return status


def _emit_json(result: dict[str, object], *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(result, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_source_artifacts is not None:
        verification = _verify_source_artifacts(args.verify_source_artifacts)
        if args.verification_exit_code_only:
            result = verification["verification_exit_code"]
        elif args.verification_status_only:
            result = verification["status"]
        elif args.verification_failures_sha_only:
            result = verification["verification_failures_sha256"]
        elif args.verification_failures_only:
            result = verification["verification_failures"]
        elif args.oracle_wrapper_timeout_source_sha_only:
            oracle_wrapper_timeout_record = verification["records"].get(
                "oracle_wrapper_timeout", {}
            )
            result = dict(oracle_wrapper_timeout_record.get("current", {})).get("sha256")
        elif args.oracle_wrapper_timeout_source_only:
            result = verification["records"].get("oracle_wrapper_timeout")
        elif args.text_resource_source_sha_only:
            text_resource_record = verification["records"].get("text_resource", {})
            result = dict(text_resource_record.get("current", {})).get("sha256")
        elif args.text_resource_source_only:
            result = verification["records"].get("text_resource")
        elif args.source_artifact_failures_only:
            result = verification["source_artifact_failed_records"]
        elif args.persisted_status_integrity_failures_only:
            result = verification["persisted_status_integrity"]["failed_checks"]
        elif args.persisted_status_integrity_only:
            result = verification["persisted_status_integrity"]
        elif args.status_integrity_failures_only:
            result = verification["status_integrity"]["failed_checks"]
        elif args.status_integrity_sha_only:
            result = _stable_json_sha256(verification["status_integrity"])
        elif args.status_integrity_only:
            result = verification["status_integrity"]
        else:
            result = verification
        _emit_json(
            result,
            pretty=args.pretty,
            output=args.output,
        )
        return READY_EXIT_CODE if verification["all_match"] is True else SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    status = build_status(
        args.prompt_artifact,
        args.oracle_artifact,
        args.docs,
        resource_artifact=args.resource_artifact,
    )
    if args.first_blocker_recommended_command_sha_only:
        first_blocker_work_item = status["handoff_summary"].get("first_blocker_work_item")
        result = (
            first_blocker_work_item.get("recommended_command_sha256")
            if isinstance(first_blocker_work_item, dict)
            else None
        )
    elif args.first_blocker_recommended_command_only:
        first_blocker_work_item = status["handoff_summary"].get("first_blocker_work_item")
        result = (
            first_blocker_work_item.get("recommended_command")
            if isinstance(first_blocker_work_item, dict)
            else None
        )
    elif args.first_blocker_only:
        result = status["handoff_summary"]["first_blocker_work_item"]
    elif args.first_blocker_sha_only:
        result = status["handoff_summary"]["first_blocker_work_item_sha256"]
    elif args.oracle_helper_command_sha_only:
        result = status["next_action_commands"]["oracle_parity_blocked"].get(
            "oracle_helper_refresh_command_sha256"
        )
    elif args.oracle_helper_command_only:
        result = status["next_action_commands"]["oracle_parity_blocked"].get(
            "oracle_helper_refresh_command"
        )
    elif args.oracle_helper_long_timeout_command_sha_only:
        result = status["next_action_commands"]["oracle_parity_blocked"].get(
            "oracle_helper_long_timeout_command_sha256"
        )
    elif args.oracle_helper_long_timeout_command_only:
        result = status["next_action_commands"]["oracle_parity_blocked"].get(
            "oracle_helper_long_timeout_command"
        )
    elif args.next_action_commands_sha_only:
        result = status["next_action_commands_sha256"]
    elif args.next_action_commands_only:
        result = status["next_action_commands"]
    elif args.readiness_gates_sha_only:
        result = status["readiness_gates_sha256"]
    elif args.readiness_gates_only:
        result = status["readiness_gates"]
    elif args.blocker_kinds_sha_only:
        result = status["blocker_kinds_sha256"]
    elif args.blocker_kinds_only:
        result = status["blocker_kinds"]
    elif args.first_remaining_blocker_report_sha_only:
        result = status["first_remaining_blocker_report_sha256"]
    elif args.first_remaining_blocker_report_only:
        result = status["first_remaining_blocker_report"]
    elif args.remaining_blockers_report_sha_only:
        result = status["remaining_blockers_report_sha256"]
    elif args.remaining_blockers_report_only:
        result = status["remaining_blockers_report"]
    elif args.blocked_gates_sha_only:
        result = status["blocked_gates_sha256"]
    elif args.blocked_gates_only:
        result = status["blocked_gates"]
    elif args.kv_streaming_loop_next_action_sha_only:
        loop_status = status["kv_backed_decode_gap_report"].get(
            "streaming_decode_loop_status", {}
        )
        result = (
            loop_status.get("next_action_sha256")
            if isinstance(loop_status, dict)
            else None
        )
    elif args.kv_streaming_loop_next_action_only:
        loop_status = status["kv_backed_decode_gap_report"].get(
            "streaming_decode_loop_status", {}
        )
        result = loop_status.get("next_action") if isinstance(loop_status, dict) else None
    elif args.kv_streaming_loop_status_sha_only:
        result = status["kv_backed_decode_gap_report"].get(
            "streaming_decode_loop_status_sha256"
        )
    elif args.kv_streaming_loop_status_only:
        result = status["kv_backed_decode_gap_report"].get(
            "streaming_decode_loop_status"
        )
    elif args.kv_streaming_blueprint_sha_only:
        result = status["kv_backed_decode_gap_report"].get(
            "streaming_decode_loop_blueprint_sha256"
        )
    elif args.kv_streaming_blueprint_only:
        result = status["kv_backed_decode_gap_report"].get(
            "streaming_decode_loop_blueprint"
        )
    elif args.kv_first_streaming_blocker_sha_only:
        result = status["kv_backed_decode_gap_report"].get(
            "first_streaming_runner_blocker_sha256"
        )
    elif args.kv_first_streaming_blocker_only:
        result = status["kv_backed_decode_gap_report"].get(
            "first_streaming_runner_blocker"
        )
    elif args.kv_streaming_blockers_sha_only:
        result = status["kv_backed_decode_gap_report"].get(
            "streaming_runner_blocker_names_sha256"
        )
    elif args.kv_streaming_blockers_only:
        result = status["kv_backed_decode_gap_report"].get("streaming_runner_blocker_names")
    elif args.oracle_wrapper_timeout_source_sha_only:
        result = status["source_artifacts"].get("oracle_wrapper_timeout", {}).get("sha256")
    elif args.oracle_wrapper_timeout_source_only:
        result = status["source_artifacts"].get("oracle_wrapper_timeout")
    elif args.text_resource_source_sha_only:
        result = status["source_artifacts"].get("text_resource", {}).get("sha256")
    elif args.text_resource_source_only:
        result = status["source_artifacts"].get("text_resource")
    elif args.source_artifacts_sha_only:
        result = status["source_artifacts_sha256"]
    elif args.verification_status_command_sha_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "verification_status_command_sha256"
        )
    elif args.verification_status_command_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "verification_status_command"
        )
    elif args.verification_exit_code_command_sha_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "verification_exit_code_command_sha256"
        )
    elif args.verification_exit_code_command_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "verification_exit_code_command"
        )
    elif args.verification_failures_command_sha_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "verification_failures_command_sha256"
        )
    elif args.verification_failures_command_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "verification_failures_command"
        )
    elif args.verification_failures_sha_command_sha_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "verification_failures_sha_command_sha256"
        )
    elif args.verification_failures_sha_command_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "verification_failures_sha_command"
        )
    elif args.source_verify_command_sha_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "source_artifacts_verify_command_sha256"
        )
    elif args.source_verify_command_only:
        result = status["next_action_commands"]["handoff_integrity"].get(
            "source_artifacts_verify_command"
        )
    elif args.status_refresh_command_sha_only:
        result = status["next_action_commands"]["oracle_parity_blocked"].get(
            "status_refresh_command_sha256"
        )
    elif args.status_refresh_command_only:
        result = status["next_action_commands"]["oracle_parity_blocked"].get(
            "status_refresh_command"
        )
    elif args.kv_resource_command_sha_only:
        result = status["next_action_commands"]["kv_backed_decode_not_wired"].get(
            "resource_plan_refresh_command_sha256"
        )
    elif args.kv_resource_command_only:
        result = status["next_action_commands"]["kv_backed_decode_not_wired"].get(
            "resource_plan_refresh_command"
        )
    elif args.persisted_status_integrity_failures_only:
        result = _persisted_status_integrity(status, status["status_integrity"])[
            "failed_checks"
        ]
    elif args.persisted_status_integrity_only:
        result = _persisted_status_integrity(status, status["status_integrity"])
    elif args.status_integrity_failures_only:
        result = status["status_integrity"]["failed_checks"]
    elif args.status_integrity_sha_only:
        result = status["status_integrity_sha256"]
    elif args.status_integrity_only:
        result = status["status_integrity"]
    elif args.schema_versions_only:
        result = status["schema_versions"]
    elif args.handoff_summary_sha_only:
        result = status["handoff_summary_sha256"]
    elif args.readiness_summary_sha_only:
        result = status["readiness_summary_sha256"]
    elif args.readiness_summary_only:
        result = status["readiness_summary"]
    elif args.blocker_recommended_commands_sha_only:
        result = status["handoff_summary"]["blocker_recommended_commands_sha256"]
    elif args.blocker_recommended_commands_only:
        result = status["handoff_summary"]["blocker_recommended_commands"]
    elif args.blocker_work_queue_sha_only:
        result = status["handoff_summary"]["blocker_work_queue_sha256"]
    elif args.blocker_work_queue_meta_only:
        result = status["handoff_summary"]["blocker_work_queue_meta"]
    elif args.blocker_work_queue_only:
        result = status["handoff_summary"]["blocker_work_queue"]
    elif args.summary_only:
        result = status["handoff_summary"]
    else:
        result = status
    _emit_json(
        result,
        pretty=args.pretty,
        output=args.output,
    )
    if args.fail_on_blocked and status["status"] != "ready":
        return BLOCKED_EXIT_CODE
    return READY_EXIT_CODE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
