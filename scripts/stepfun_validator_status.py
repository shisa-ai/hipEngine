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
ORACLE_EVIDENCE_GAP_NAMES = (
    "oracle_success_status",
    "oracle_returncode_zero",
    "no_timeout_or_oracle_blocker",
    "generated_text_nonempty",
    "generated_text_matches_target",
)


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
        "--blocked-evidence-summary-only",
        action="store_true",
        help="Emit only the compact evidence summary for all failed/missing validators.",
    )
    parser.add_argument(
        "--blocked-evidence-summary-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the blocked evidence summary.",
    )
    parser.add_argument(
        "--blocked-evidence-by-gate-only",
        action="store_true",
        help="Emit only the blocked evidence summary grouped by readiness gate.",
    )
    parser.add_argument(
        "--blocked-evidence-by-gate-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the gate-level blocked evidence summary.",
    )
    parser.add_argument(
        "--blocked-readiness-gates-only",
        action="store_true",
        help="Emit only the ordered readiness-gate names with blocked validators.",
    )
    parser.add_argument(
        "--blocked-readiness-gates-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the blocked readiness-gate name list.",
    )
    parser.add_argument(
        "--next-blocked-gate-only",
        action="store_true",
        help="Emit only the first blocked readiness-gate evidence summary, or null.",
    )
    parser.add_argument(
        "--next-blocked-gate-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the first blocked gate summary.",
    )
    parser.add_argument(
        "--blocked-evidence-gate",
        help="Readiness gate name used by the selected blocked-evidence gate modes.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-only",
        action="store_true",
        help="Emit only the blocked evidence summary for --blocked-evidence-gate, or null.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-found-only",
        action="store_true",
        help="Emit only whether --blocked-evidence-gate matched a blocked readiness gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-artifacts-only",
        action="store_true",
        help="Emit only the artifact-name list for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-artifacts-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of --blocked-evidence-gate artifact names.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-artifact-count-only",
        action="store_true",
        help="Emit only the artifact count for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-blocked-count-only",
        action="store_true",
        help="Emit only the blocked artifact count for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-status-counts-only",
        action="store_true",
        help="Emit only the status-count map for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-status-counts-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of --blocked-evidence-gate status counts.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-producer-commands-only",
        action="store_true",
        help="Emit only the producer command list for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-producer-commands-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of --blocked-evidence-gate producer commands.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-producer-command-count-only",
        action="store_true",
        help="Emit only the producer command count for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-validator-commands-only",
        action="store_true",
        help="Emit only the validator command list for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-validator-commands-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of --blocked-evidence-gate validator commands.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-validator-command-count-only",
        action="store_true",
        help="Emit only the validator command count for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-missing-evidence-only",
        action="store_true",
        help="Emit only the missing-evidence list for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-missing-evidence-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of --blocked-evidence-gate missing evidence.",
    )
    parser.add_argument(
        "--blocked-evidence-gate-missing-evidence-count-only",
        action="store_true",
        help="Emit only the missing-evidence item count for --blocked-evidence-gate.",
    )
    parser.add_argument(
        "--next-blocker-only",
        action="store_true",
        help="Emit only the first failed or missing validator record, or null.",
    )
    parser.add_argument(
        "--next-blocker-artifact-name-only",
        action="store_true",
        help="Emit only the artifact name for the first failed/missing validator record, or null.",
    )
    parser.add_argument(
        "--next-blocker-readiness-gate-only",
        action="store_true",
        help="Emit only the readiness gate for the first failed/missing validator record, or null.",
    )
    parser.add_argument(
        "--next-blocker-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the first failed/missing validator record.",
    )
    parser.add_argument(
        "--next-blocker-status-only",
        action="store_true",
        help="Emit only the status of the first failed/missing validator record, or null.",
    )
    parser.add_argument(
        "--next-blocker-reason-only",
        action="store_true",
        help="Emit only the reason for the first failed/missing validator record, or null.",
    )
    parser.add_argument(
        "--next-command-only",
        action="store_true",
        help="Emit only the concrete validator command for the first failed/missing record, or null.",
    )
    parser.add_argument(
        "--next-command-kind-only",
        action="store_true",
        help="Emit only the validator command kind for the first failed/missing record, or null.",
    )
    parser.add_argument(
        "--next-command-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the next concrete validator command.",
    )
    parser.add_argument(
        "--next-producer-command-only",
        action="store_true",
        help="Emit only the recommended producer/rerun command for the first failed/missing record, or null.",
    )
    parser.add_argument(
        "--next-producer-command-kind-only",
        action="store_true",
        help="Emit only the recommended producer/rerun command kind for the first failed/missing record, or null.",
    )
    parser.add_argument(
        "--next-producer-command-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the next producer/rerun command.",
    )
    parser.add_argument(
        "--next-action-only",
        action="store_true",
        help="Emit a compact action bundle for the first failed/missing record, or null.",
    )
    parser.add_argument(
        "--next-action-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the next action bundle.",
    )
    parser.add_argument(
        "--next-action-available-only",
        action="store_true",
        help="Emit only whether a compact next-action payload is available.",
    )
    parser.add_argument(
        "--next-action-artifact-name-only",
        action="store_true",
        help="Emit only the artifact name from the next-action payload, or null.",
    )
    parser.add_argument(
        "--next-action-readiness-gate-only",
        action="store_true",
        help="Emit only the readiness gate from the next-action payload, or null.",
    )
    parser.add_argument(
        "--next-action-status-only",
        action="store_true",
        help="Emit only the status from the next-action payload, or null.",
    )
    parser.add_argument(
        "--next-action-reason-only",
        action="store_true",
        help="Emit only the reason from the next-action payload, or null.",
    )
    parser.add_argument(
        "--next-action-validator-command-kind-only",
        action="store_true",
        help="Emit only the validator command kind from the next-action payload, or null.",
    )
    parser.add_argument(
        "--next-action-validator-command-only",
        action="store_true",
        help="Emit only the validator command from the next-action payload, or null.",
    )
    parser.add_argument(
        "--next-action-validator-command-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the validator command from the next-action payload.",
    )
    parser.add_argument(
        "--next-action-producer-command-kind-only",
        action="store_true",
        help="Emit only the producer command kind from the next-action payload, or null.",
    )
    parser.add_argument(
        "--next-action-producer-command-only",
        action="store_true",
        help="Emit only the producer command from the next-action payload, or null.",
    )
    parser.add_argument(
        "--next-action-producer-command-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the producer command from the next-action payload.",
    )
    parser.add_argument(
        "--next-action-partial-output-handoff-only",
        action="store_true",
        help="Print only the partial-output handoff bundle from the next-action payload.",
    )
    parser.add_argument(
        "--next-action-partial-output-handoff-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the next-action partial-output handoff bundle.",
    )
    parser.add_argument(
        "--next-action-partial-output-path-only",
        action="store_true",
        help="Print only the producer partial-output path from the next-action payload.",
    )
    parser.add_argument(
        "--next-action-partial-output-status-only",
        action="store_true",
        help="Print only the producer partial-output status from the next-action payload.",
    )
    parser.add_argument(
        "--next-action-validator-summary-only",
        action="store_true",
        help=(
            "Print only the validator summary embedded in the next-action payload."
        ),
    )
    parser.add_argument(
        "--next-action-validator-summary-sha-only",
        action="store_true",
        help=(
            "Print only the SHA-256 digest of the validator summary embedded "
            "in the next-action payload."
        ),
    )
    parser.add_argument(
        "--next-action-validator-summary-status-only",
        action="store_true",
        help="Print only the status field from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-validator-summary-ready-only",
        action="store_true",
        help="Print only the ready boolean from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-validator-summary-oracle-status-only",
        action="store_true",
        help="Print only the oracle_status field from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-validator-summary-oracle-blocker-kind-only",
        action="store_true",
        help="Print only the oracle_blocker_kind field from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-oracle-expected-token-only",
        action="store_true",
        help="Print only the oracle expected-token bundle from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-oracle-expected-token-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the next-action oracle expected-token bundle.",
    )
    parser.add_argument(
        "--next-action-expected-next-token-id-only",
        action="store_true",
        help="Print only the expected next-token id from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-expected-next-token-text-only",
        action="store_true",
        help="Print only the expected next-token text from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-expected-next-token-logit-only",
        action="store_true",
        help="Print only the expected next-token logit from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-oracle-generated-text-only",
        action="store_true",
        help="Print only the oracle generated-text bundle from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-oracle-generated-text-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the next-action oracle generated-text bundle.",
    )
    parser.add_argument(
        "--next-action-generated-text-only",
        action="store_true",
        help="Print only the generated text from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-generated-text-len-only",
        action="store_true",
        help="Print only the generated text length from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-generated-text-matches-expected-exact-only",
        action="store_true",
        help="Print only the exact generated-text match flag from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-generated-text-matches-expected-stripped-only",
        action="store_true",
        help="Print only the stripped generated-text match flag from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-oracle-artifact-provenance-only",
        action="store_true",
        help="Print only the oracle artifact provenance bundle from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-oracle-artifact-provenance-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the next-action oracle artifact provenance bundle.",
    )
    parser.add_argument(
        "--next-action-oracle-artifact-path-only",
        action="store_true",
        help="Print only the oracle artifact path from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-oracle-artifact-sha-only",
        action="store_true",
        help="Print only the oracle artifact SHA-256 from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-prompt-artifact-path-only",
        action="store_true",
        help="Print only the prompt artifact path from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-prompt-artifact-sha-only",
        action="store_true",
        help="Print only the prompt artifact SHA-256 from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-evidence-checks-sha-only",
        action="store_true",
        help="Print only the evidence-checks SHA-256 from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-oracle-artifact-presence-only",
        action="store_true",
        help="Print only the oracle/prompt artifact presence bundle from the next-action provenance.",
    )
    parser.add_argument(
        "--next-action-oracle-artifact-presence-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the next-action artifact presence bundle.",
    )
    parser.add_argument(
        "--next-action-oracle-artifact-present-only",
        action="store_true",
        help="Print only whether the next-action oracle artifact path exists.",
    )
    parser.add_argument(
        "--next-action-prompt-artifact-present-only",
        action="store_true",
        help="Print only whether the next-action prompt artifact path exists.",
    )
    parser.add_argument(
        "--next-action-no-claim-policy-only",
        action="store_true",
        help="Print only the no-claim policy from the next-action validator summary.",
    )
    parser.add_argument(
        "--next-action-no-claim-policy-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the next-action no-claim policy.",
    )
    parser.add_argument(
        "--next-action-oracle-parity-claim-allowed-only",
        action="store_true",
        help="Print only the next-action oracle-parity claim-allowed boolean.",
    )
    parser.add_argument(
        "--next-action-kv-backed-decode-claim-allowed-only",
        action="store_true",
        help="Print only the next-action KV-backed-decode claim-allowed boolean.",
    )
    parser.add_argument(
        "--next-action-e2e-inference-claim-allowed-only",
        action="store_true",
        help="Print only the next-action e2e-inference claim-allowed boolean.",
    )
    parser.add_argument(
        "--next-action-performance-claim-allowed-only",
        action="store_true",
        help="Print only the next-action performance-claim allowed boolean.",
    )
    parser.add_argument(
        "--next-action-missing-evidence-only",
        action="store_true",
        help="Print only the next-action validator missing-evidence list.",
    )
    parser.add_argument(
        "--next-action-missing-evidence-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the next-action missing-evidence list.",
    )
    parser.add_argument(
        "--next-action-missing-evidence-count-only",
        action="store_true",
        help="Print only the next-action missing-evidence item count.",
    )
    parser.add_argument(
        "--next-action-oracle-evidence-gap-count-only",
        action="store_true",
        help="Print only the next-action oracle evidence-gap item count.",
    )
    parser.add_argument(
        "--next-action-oracle-evidence-gaps-only",
        action="store_true",
        help="Print only the next-action oracle evidence-gap list.",
    )
    parser.add_argument(
        "--next-action-oracle-evidence-gaps-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the oracle evidence-gap list.",
    )
    parser.add_argument(
        "--next-action-missing-evidence-present-only",
        action="store_true",
        help="Print only whether the next-action missing-evidence list is non-empty.",
    )
    parser.add_argument(
        "--next-action-missing-evidence-joined-only",
        action="store_true",
        help="Print only the pipe-joined next-action missing-evidence sequence.",
    )
    parser.add_argument(
        "--next-action-missing-evidence-sorted-only",
        action="store_true",
        help="Print only the sorted next-action missing-evidence list.",
    )
    parser.add_argument(
        "--next-action-missing-evidence-sorted-sha-only",
        action="store_true",
        help="Print only the stable SHA-256 digest of the sorted missing-evidence list.",
    )
    parser.add_argument(
        "--next-action-missing-evidence-sorted-joined-only",
        action="store_true",
        help="Print only the pipe-joined sorted next-action missing-evidence sequence.",
    )
    parser.add_argument(
        "--next-action-first-missing-evidence-only",
        action="store_true",
        help="Print only the first next-action missing-evidence item.",
    )
    parser.add_argument(
        "--next-action-last-missing-evidence-only",
        action="store_true",
        help="Print only the last next-action missing-evidence item.",
    )
    parser.add_argument(
        "--next-action-artifact-file-present-missing-only",
        action="store_true",
        help="Print only whether artifact_file_present is missing evidence.",
    )
    parser.add_argument(
        "--next-action-oracle-success-status-missing-only",
        action="store_true",
        help="Print only whether oracle_success_status is missing evidence.",
    )
    parser.add_argument(
        "--next-action-oracle-returncode-zero-missing-only",
        action="store_true",
        help="Print only whether oracle_returncode_zero is missing evidence.",
    )
    parser.add_argument(
        "--next-action-no-timeout-or-oracle-blocker-missing-only",
        action="store_true",
        help="Print only whether no_timeout_or_oracle_blocker is missing evidence.",
    )
    parser.add_argument(
        "--next-action-generated-text-matches-target-missing-only",
        action="store_true",
        help="Print only whether generated_text_matches_target is missing evidence.",
    )
    parser.add_argument(
        "--next-action-generated-text-nonempty-missing-only",
        action="store_true",
        help="Print only whether generated_text_nonempty is missing evidence.",
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
        missing_evidence = ["validator_artifact_path_present"]
        return {
            **base,
            "status": "missing",
            "ready": False,
            "reason": "validator_artifact_path_missing",
            "validator_missing_evidence": missing_evidence,
            "validator_missing_evidence_count": len(missing_evidence),
        }
    if not artifact_path.exists():
        missing_evidence = ["artifact_file_present"]
        return {
            **base,
            "status": "missing",
            "ready": False,
            "reason": "artifact_file_missing",
            "validator_missing_evidence": missing_evidence,
            "validator_missing_evidence_count": len(missing_evidence),
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
        "validator_summary": summary,
        "validator_summary_sha256": status_mod._stable_json_sha256(summary),
        "validator_missing_evidence": summary.get("missing_evidence"),
        "validator_missing_evidence_count": summary.get("missing_evidence_count"),
    }


def _recommended_commands_by_gate(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return recommended producer/rerun commands keyed by readiness gate."""

    commands = manifest.get("recommended_commands_handoff", [])
    if not isinstance(commands, list):
        return {}
    by_gate: dict[str, dict[str, object]] = {}
    for item in commands:
        if not isinstance(item, dict):
            continue
        gate = item.get("readiness_gate")
        if gate not in (None, ""):
            by_gate[str(gate)] = item
    return by_gate


def _artifacts_by_name(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return final-blocker artifact handoff records keyed by artifact name."""

    artifacts = manifest.get("artifacts_to_collect", [])
    if not isinstance(artifacts, list):
        return {}
    by_name: dict[str, dict[str, object]] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("artifact_name")
        if name not in (None, ""):
            by_name[str(name)] = item
    return by_name


def _maybe_copy(
    target: dict[str, object],
    source: dict[str, object] | None,
    mapping: dict[str, str],
) -> None:
    if not source:
        return
    for source_key, target_key in mapping.items():
        if source_key in source:
            target[target_key] = source.get(source_key)


def _attach_producer_command(
    result: dict[str, object],
    recommended_by_gate: dict[str, dict[str, object]],
    artifacts_by_name: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Attach the producer/rerun command and handoff metadata for a result."""

    gate = result.get("readiness_gate")
    command_record = recommended_by_gate.get(str(gate)) if gate not in (None, "") else None
    artifact_name = result.get("artifact_name")
    artifact_record = (
        artifacts_by_name.get(str(artifact_name))
        if artifact_name not in (None, "")
        else None
    )
    attached = {
        **result,
        "producer_command_kind": command_record.get("recommended_command_kind")
        if command_record
        else None,
        "producer_command": command_record.get("recommended_command")
        if command_record
        else None,
        "producer_command_sha256": command_record.get("recommended_command_sha256")
        if command_record
        else None,
        "producer_command_reason": command_record.get("recommended_command_reason")
        if command_record
        else None,
    }
    _maybe_copy(
        attached,
        command_record,
        {
            "writes_partial_output_before_launch": "producer_writes_partial_output_before_launch",
            "partial_output_path": "producer_partial_output_path",
            "partial_output_status": "producer_partial_output_status",
            "partial_output_overwrite_policy": "producer_partial_output_overwrite_policy",
            "partial_output_supervisor_signal_handoff_safe": "producer_partial_output_supervisor_signal_handoff_safe",
        },
    )
    _maybe_copy(
        attached,
        artifact_record,
        {
            "partial_output_handoff_safe": "artifact_partial_output_handoff_safe",
            "partial_output_supervisor_signal_handoff_safe": "artifact_partial_output_supervisor_signal_handoff_safe",
            "partial_output_supervisor_signal_contract": "artifact_partial_output_supervisor_signal_contract",
        },
    )
    return attached


def _result_missing_evidence(record: dict[str, object]) -> list[object]:
    missing = record.get("validator_missing_evidence")
    summary = record.get("validator_summary")
    if missing is None and isinstance(summary, dict):
        missing = summary.get("missing_evidence")
    return list(missing) if isinstance(missing, list) else []


def _summary_field(summary: object, key: str) -> object:
    if isinstance(summary, dict):
        return summary.get(key)
    return None


def _next_action_partial_output_handoff(
    next_action: object,
) -> dict[str, object] | None:
    if not isinstance(next_action, dict):
        return None
    keys = (
        "producer_writes_partial_output_before_launch",
        "producer_partial_output_path",
        "producer_partial_output_status",
        "producer_partial_output_overwrite_policy",
        "producer_partial_output_supervisor_signal_handoff_safe",
        "artifact_partial_output_handoff_safe",
        "artifact_partial_output_supervisor_signal_handoff_safe",
        "artifact_partial_output_supervisor_signal_contract",
    )
    handoff = {key: next_action.get(key) for key in keys}
    if not any(value is not None and value != "" for value in handoff.values()):
        return None
    return handoff


def _next_action_oracle_expected_token(
    validator_summary: object,
) -> dict[str, object] | None:
    if not isinstance(validator_summary, dict):
        return None
    expected_token = {
        "expected_next_token_id": validator_summary.get("expected_next_token_id"),
        "expected_next_token_text": validator_summary.get("expected_next_token_text"),
        "expected_next_token_logit": validator_summary.get("expected_next_token_logit"),
    }
    if not any(value is not None and value != "" for value in expected_token.values()):
        return None
    return expected_token


def _next_action_oracle_generated_text(
    validator_summary: object,
) -> dict[str, object] | None:
    if not isinstance(validator_summary, dict):
        return None
    generated_text = {
        "generated_text": validator_summary.get("generated_text"),
        "generated_text_len": validator_summary.get("generated_text_len"),
        "text_matches_expected_exact": validator_summary.get(
            "text_matches_expected_exact"
        ),
        "text_matches_expected_stripped": validator_summary.get(
            "text_matches_expected_stripped"
        ),
    }
    if not any(value is not None and value != "" for value in generated_text.values()):
        return None
    return generated_text


def _next_action_oracle_artifact_provenance(
    validator_summary: object,
) -> dict[str, object] | None:
    if not isinstance(validator_summary, dict):
        return None
    provenance = {
        "artifact": validator_summary.get("artifact"),
        "artifact_sha256": validator_summary.get("artifact_sha256"),
        "prompt_artifact": validator_summary.get("prompt_artifact"),
        "prompt_artifact_sha256": validator_summary.get("prompt_artifact_sha256"),
        "evidence_checks_sha256": validator_summary.get("evidence_checks_sha256"),
    }
    if not any(value is not None and value != "" for value in provenance.values()):
        return None
    return provenance


def _path_value_present(path_value: object) -> bool | None:
    if path_value in (None, ""):
        return None
    return Path(str(path_value)).exists()


def _artifact_presence_bundle(
    *,
    oracle_artifact_present: bool | None,
    prompt_artifact_present: bool | None,
) -> dict[str, object] | None:
    presence = {
        "oracle_artifact_present": oracle_artifact_present,
        "prompt_artifact_present": prompt_artifact_present,
    }
    if not any(value is not None for value in presence.values()):
        return None
    return presence


def _unique_preserving_order(values: Sequence[object]) -> list[object]:
    seen: set[str] = set()
    result: list[object] = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _blocked_evidence_summary(
    blocked_results: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return compact blocker evidence gaps without full checker payloads."""

    summary: list[dict[str, object]] = []
    for record in blocked_results:
        missing = _result_missing_evidence(record)
        validator_summary = record.get("validator_summary")
        summary.append(
            {
                "artifact_name": record.get("artifact_name"),
                "readiness_gate": record.get("readiness_gate"),
                "status": record.get("status"),
                "reason": record.get("reason"),
                "validator_artifact_path": record.get("validator_artifact_path"),
                "validator_command_kind": record.get("validator_command_kind"),
                "validator_command_sha256": record.get(
                    "validator_command_concrete_sha256"
                ),
                "producer_command_kind": record.get("producer_command_kind"),
                "producer_command_sha256": record.get("producer_command_sha256"),
                "missing_evidence": missing,
                "missing_evidence_count": len(missing),
                "validator_summary_sha256": status_mod._stable_json_sha256(
                    validator_summary
                )
                if isinstance(validator_summary, dict)
                else None,
            }
        )
    return summary


def _blocked_evidence_by_gate(
    blocked_evidence_summary: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Group compact blocked evidence by readiness gate."""

    grouped: dict[str, dict[str, object]] = {}
    for record in blocked_evidence_summary:
        gate = str(record.get("readiness_gate") or "unknown")
        gate_record = grouped.setdefault(
            gate,
            {
                "readiness_gate": gate,
                "artifact_names": [],
                "blocked_count": 0,
                "status_counts": {},
                "missing_evidence": [],
                "producer_command_kinds": [],
                "producer_command_sha256s": [],
                "validator_command_kinds": [],
                "validator_command_sha256s": [],
            },
        )
        gate_record["blocked_count"] = int(gate_record["blocked_count"]) + 1
        status = str(record.get("status") or "unknown")
        status_counts = gate_record["status_counts"]
        assert isinstance(status_counts, dict)
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        for record_key, gate_key in (
            ("artifact_name", "artifact_names"),
            ("producer_command_kind", "producer_command_kinds"),
            ("producer_command_sha256", "producer_command_sha256s"),
            ("validator_command_kind", "validator_command_kinds"),
            ("validator_command_sha256", "validator_command_sha256s"),
        ):
            value = record.get(record_key)
            if value in (None, ""):
                continue
            values = gate_record[gate_key]
            assert isinstance(values, list)
            values.append(value)
        missing = record.get("missing_evidence")
        if isinstance(missing, list):
            evidence_values = gate_record["missing_evidence"]
            assert isinstance(evidence_values, list)
            evidence_values.extend(missing)

    gate_summary: list[dict[str, object]] = []
    for gate_record in grouped.values():
        missing = _unique_preserving_order(gate_record["missing_evidence"])
        gate_record["artifact_names"] = _unique_preserving_order(
            gate_record["artifact_names"]
        )
        gate_record["producer_command_kinds"] = _unique_preserving_order(
            gate_record["producer_command_kinds"]
        )
        gate_record["producer_command_sha256s"] = _unique_preserving_order(
            gate_record["producer_command_sha256s"]
        )
        gate_record["validator_command_kinds"] = _unique_preserving_order(
            gate_record["validator_command_kinds"]
        )
        gate_record["validator_command_sha256s"] = _unique_preserving_order(
            gate_record["validator_command_sha256s"]
        )
        gate_record["missing_evidence"] = missing
        gate_record["missing_evidence_count"] = len(missing)
        gate_summary.append(gate_record)
    return gate_summary


def _blocked_gate_by_name(
    blocked_evidence_by_gate: Sequence[dict[str, object]],
    readiness_gate: object,
) -> dict[str, object] | None:
    if readiness_gate in (None, ""):
        return None
    for record in blocked_evidence_by_gate:
        if record.get("readiness_gate") == readiness_gate:
            return record
    return None


def build_validator_status_report(
    manifest: dict[str, object],
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    resource_artifact: Path = status_mod.DEFAULT_RESOURCE_ARTIFACT,
    selected_blocked_gate_name: str | None = None,
) -> dict[str, object]:
    """Return aggregate status for concrete final-blocker validators."""

    validator_commands = manifest.get("validator_commands_handoff", [])
    if not isinstance(validator_commands, list):
        validator_commands = []
    recommended_by_gate = _recommended_commands_by_gate(manifest)
    artifacts_by_name = _artifacts_by_name(manifest)
    results = [
        _attach_producer_command(
            _run_validator(
                dict(record) if isinstance(record, dict) else {},
                prompt_artifact=prompt_artifact,
                resource_artifact=resource_artifact,
            ),
            recommended_by_gate,
            artifacts_by_name,
        )
        for record in validator_commands
    ]
    blocked_results = [
        record for record in results if record.get("status") in {"missing", "failed"}
    ]
    blocked_evidence_summary = _blocked_evidence_summary(blocked_results)
    blocked_evidence_by_gate = _blocked_evidence_by_gate(blocked_evidence_summary)
    blocked_readiness_gates = [
        str(record.get("readiness_gate")) for record in blocked_evidence_by_gate
    ]
    next_blocked_gate = blocked_evidence_by_gate[0] if blocked_evidence_by_gate else None
    next_blocker = blocked_results[0] if blocked_results else None
    next_blocker_command = (
        next_blocker.get("validator_command_concrete")
        if isinstance(next_blocker, dict)
        else None
    )
    next_producer_command = (
        next_blocker.get("producer_command")
        if isinstance(next_blocker, dict)
        else None
    )
    next_action = None
    if isinstance(next_blocker, dict):
        next_action = {
            "artifact_name": next_blocker.get("artifact_name"),
            "readiness_gate": next_blocker.get("readiness_gate"),
            "status": next_blocker.get("status"),
            "reason": next_blocker.get("reason"),
            "validator_command_kind": next_blocker.get("validator_command_kind"),
            "validator_command": next_blocker_command,
            "validator_command_sha256": status_mod._stable_json_sha256(
                next_blocker_command
            ),
            "producer_command_kind": next_blocker.get("producer_command_kind"),
            "producer_command": next_producer_command,
            "producer_command_sha256": status_mod._stable_json_sha256(
                next_producer_command
            ),
            "validator_artifact_path": next_blocker.get("validator_artifact_path"),
            "validator_missing_evidence": next_blocker.get(
                "validator_missing_evidence"
            ),
            "validator_missing_evidence_count": next_blocker.get(
                "validator_missing_evidence_count"
            ),
        }
        for key in (
            "validator_summary",
            "producer_writes_partial_output_before_launch",
            "producer_partial_output_path",
            "producer_partial_output_status",
            "producer_partial_output_overwrite_policy",
            "producer_partial_output_supervisor_signal_handoff_safe",
            "artifact_partial_output_handoff_safe",
            "artifact_partial_output_supervisor_signal_handoff_safe",
            "artifact_partial_output_supervisor_signal_contract",
        ):
            if key in next_blocker:
                next_action[key] = next_blocker.get(key)
    next_action_available = isinstance(next_action, dict)
    next_action_validator_summary = (
        next_action.get("validator_summary") if isinstance(next_action, dict) else None
    )
    next_action_validator_summary_status = _summary_field(
        next_action_validator_summary,
        "status",
    )
    next_action_validator_summary_ready = _summary_field(
        next_action_validator_summary,
        "ready",
    )
    next_action_validator_summary_oracle_status = _summary_field(
        next_action_validator_summary,
        "oracle_status",
    )
    next_action_validator_summary_oracle_blocker_kind = _summary_field(
        next_action_validator_summary,
        "oracle_blocker_kind",
    )
    next_action_no_claim_policy = _summary_field(
        next_action_validator_summary,
        "no_claim_policy",
    )
    next_action_oracle_parity_claim_allowed = _summary_field(
        next_action_no_claim_policy,
        "oracle_parity_claim_allowed",
    )
    next_action_kv_backed_decode_claim_allowed = _summary_field(
        next_action_no_claim_policy,
        "kv_backed_decode_claim_allowed",
    )
    next_action_e2e_inference_claim_allowed = _summary_field(
        next_action_no_claim_policy,
        "e2e_inference_claim_allowed",
    )
    next_action_performance_claim_allowed = _summary_field(
        next_action_no_claim_policy,
        "performance_claim_allowed",
    )
    next_action_oracle_expected_token = _next_action_oracle_expected_token(
        next_action_validator_summary
    )
    next_action_expected_next_token_id = _summary_field(
        next_action_oracle_expected_token,
        "expected_next_token_id",
    )
    next_action_expected_next_token_text = _summary_field(
        next_action_oracle_expected_token,
        "expected_next_token_text",
    )
    next_action_expected_next_token_logit = _summary_field(
        next_action_oracle_expected_token,
        "expected_next_token_logit",
    )
    next_action_oracle_generated_text = _next_action_oracle_generated_text(
        next_action_validator_summary
    )
    next_action_generated_text = _summary_field(
        next_action_oracle_generated_text,
        "generated_text",
    )
    next_action_generated_text_len = _summary_field(
        next_action_oracle_generated_text,
        "generated_text_len",
    )
    next_action_generated_text_matches_expected_exact = _summary_field(
        next_action_oracle_generated_text,
        "text_matches_expected_exact",
    )
    next_action_generated_text_matches_expected_stripped = _summary_field(
        next_action_oracle_generated_text,
        "text_matches_expected_stripped",
    )
    next_action_oracle_artifact_provenance = (
        _next_action_oracle_artifact_provenance(next_action_validator_summary)
    )
    next_action_oracle_artifact_path = _summary_field(
        next_action_oracle_artifact_provenance,
        "artifact",
    )
    next_action_oracle_artifact_sha256 = _summary_field(
        next_action_oracle_artifact_provenance,
        "artifact_sha256",
    )
    next_action_prompt_artifact_path = _summary_field(
        next_action_oracle_artifact_provenance,
        "prompt_artifact",
    )
    next_action_prompt_artifact_sha256 = _summary_field(
        next_action_oracle_artifact_provenance,
        "prompt_artifact_sha256",
    )
    next_action_evidence_checks_sha256 = _summary_field(
        next_action_oracle_artifact_provenance,
        "evidence_checks_sha256",
    )
    next_action_oracle_artifact_present = _path_value_present(
        next_action_oracle_artifact_path
    )
    next_action_prompt_artifact_present = _path_value_present(
        next_action_prompt_artifact_path
    )
    next_action_oracle_artifact_presence = _artifact_presence_bundle(
        oracle_artifact_present=next_action_oracle_artifact_present,
        prompt_artifact_present=next_action_prompt_artifact_present,
    )
    next_action_missing_evidence = None
    if isinstance(next_action, dict):
        next_action_missing_evidence = next_action.get("validator_missing_evidence")
        if next_action_missing_evidence is None and isinstance(
            next_action_validator_summary, dict
        ):
            next_action_missing_evidence = next_action_validator_summary.get(
                "missing_evidence"
            )
    next_action_missing_evidence_count = (
        len(next_action_missing_evidence)
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_missing_evidence_present = (
        bool(next_action_missing_evidence)
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_missing_evidence_joined = (
        "|".join(str(item) for item in next_action_missing_evidence)
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_missing_evidence_sorted = (
        sorted(str(item) for item in next_action_missing_evidence)
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_missing_evidence_sorted_joined = (
        "|".join(next_action_missing_evidence_sorted)
        if isinstance(next_action_missing_evidence_sorted, list)
        else None
    )
    next_action_first_missing_evidence = (
        next_action_missing_evidence[0]
        if (
            isinstance(next_action_missing_evidence, list)
            and next_action_missing_evidence
        )
        else None
    )
    next_action_last_missing_evidence = (
        next_action_missing_evidence[-1]
        if (
            isinstance(next_action_missing_evidence, list)
            and next_action_missing_evidence
        )
        else None
    )
    next_action_artifact_file_present_missing = (
        "artifact_file_present" in next_action_missing_evidence
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_oracle_success_status_missing = (
        "oracle_success_status" in next_action_missing_evidence
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_oracle_returncode_zero_missing = (
        "oracle_returncode_zero" in next_action_missing_evidence
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_no_timeout_or_oracle_blocker_missing = (
        "no_timeout_or_oracle_blocker" in next_action_missing_evidence
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_generated_text_matches_target_missing = (
        "generated_text_matches_target" in next_action_missing_evidence
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_generated_text_nonempty_missing = (
        "generated_text_nonempty" in next_action_missing_evidence
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_oracle_evidence_gaps = (
        [
            name
            for name in ORACLE_EVIDENCE_GAP_NAMES
            if name in next_action_missing_evidence
        ]
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    next_action_oracle_evidence_gap_count = (
        len(next_action_oracle_evidence_gaps)
        if isinstance(next_action_oracle_evidence_gaps, list)
        else None
    )
    next_action_partial_output_handoff = _next_action_partial_output_handoff(
        next_action
    )
    next_action_partial_output_path = (
        next_action_partial_output_handoff.get("producer_partial_output_path")
        if isinstance(next_action_partial_output_handoff, dict)
        else None
    )
    next_action_partial_output_status = (
        next_action_partial_output_handoff.get("producer_partial_output_status")
        if isinstance(next_action_partial_output_handoff, dict)
        else None
    )
    selected_blocked_gate = _blocked_gate_by_name(
        blocked_evidence_by_gate,
        selected_blocked_gate_name,
    )
    selected_blocked_gate_records = [
        record
        for record in blocked_results
        if selected_blocked_gate_name not in (None, "")
        and record.get("readiness_gate") == selected_blocked_gate_name
    ]
    selected_blocked_gate_producer_commands = (
        _unique_preserving_order(
            [
                record.get("producer_command")
                for record in selected_blocked_gate_records
                if record.get("producer_command") not in (None, "")
            ]
        )
        if isinstance(selected_blocked_gate, dict)
        else None
    )
    selected_blocked_gate_producer_command_count = (
        len(selected_blocked_gate_producer_commands)
        if isinstance(selected_blocked_gate_producer_commands, list)
        else None
    )
    selected_blocked_gate_validator_commands = (
        _unique_preserving_order(
            [
                record.get("validator_command_concrete")
                for record in selected_blocked_gate_records
                if record.get("validator_command_concrete") not in (None, "")
            ]
        )
        if isinstance(selected_blocked_gate, dict)
        else None
    )
    selected_blocked_gate_validator_command_count = (
        len(selected_blocked_gate_validator_commands)
        if isinstance(selected_blocked_gate_validator_commands, list)
        else None
    )
    selected_blocked_gate_artifact_names = (
        selected_blocked_gate.get("artifact_names")
        if isinstance(selected_blocked_gate, dict)
        else None
    )
    selected_blocked_gate_artifact_count = (
        len(selected_blocked_gate_artifact_names)
        if isinstance(selected_blocked_gate_artifact_names, list)
        else None
    )
    selected_blocked_gate_blocked_count = (
        selected_blocked_gate.get("blocked_count")
        if isinstance(selected_blocked_gate, dict)
        else None
    )
    selected_blocked_gate_status_counts = (
        selected_blocked_gate.get("status_counts")
        if isinstance(selected_blocked_gate, dict)
        else None
    )
    selected_blocked_gate_missing_evidence = (
        selected_blocked_gate.get("missing_evidence")
        if isinstance(selected_blocked_gate, dict)
        else None
    )
    selected_blocked_gate_missing_evidence_count = (
        len(selected_blocked_gate_missing_evidence)
        if isinstance(selected_blocked_gate_missing_evidence, list)
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
        "blocked_evidence_summary": blocked_evidence_summary,
        "blocked_evidence_summary_sha256": status_mod._stable_json_sha256(
            blocked_evidence_summary
        ),
        "blocked_evidence_by_gate": blocked_evidence_by_gate,
        "blocked_evidence_by_gate_sha256": status_mod._stable_json_sha256(
            blocked_evidence_by_gate
        ),
        "blocked_readiness_gates": blocked_readiness_gates,
        "blocked_readiness_gates_sha256": status_mod._stable_json_sha256(
            blocked_readiness_gates
        ),
        "next_blocked_gate": next_blocked_gate,
        "next_blocked_gate_readiness_gate": next_blocked_gate.get("readiness_gate")
        if isinstance(next_blocked_gate, dict)
        else None,
        "next_blocked_gate_sha256": status_mod._stable_json_sha256(
            next_blocked_gate
        ),
        "selected_blocked_gate_readiness_gate": selected_blocked_gate_name,
        "selected_blocked_gate_found": selected_blocked_gate is not None,
        "selected_blocked_gate": selected_blocked_gate,
        "selected_blocked_gate_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate
        ),
        "selected_blocked_gate_artifact_names": selected_blocked_gate_artifact_names,
        "selected_blocked_gate_artifact_count": selected_blocked_gate_artifact_count,
        "selected_blocked_gate_artifact_names_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_artifact_names
        ),
        "selected_blocked_gate_blocked_count": selected_blocked_gate_blocked_count,
        "selected_blocked_gate_status_counts": selected_blocked_gate_status_counts,
        "selected_blocked_gate_status_counts_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_status_counts
        ),
        "selected_blocked_gate_producer_commands": selected_blocked_gate_producer_commands,
        "selected_blocked_gate_producer_command_count": selected_blocked_gate_producer_command_count,
        "selected_blocked_gate_producer_commands_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_producer_commands
        ),
        "selected_blocked_gate_validator_commands": selected_blocked_gate_validator_commands,
        "selected_blocked_gate_validator_command_count": selected_blocked_gate_validator_command_count,
        "selected_blocked_gate_validator_commands_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_validator_commands
        ),
        "selected_blocked_gate_missing_evidence": selected_blocked_gate_missing_evidence,
        "selected_blocked_gate_missing_evidence_count": selected_blocked_gate_missing_evidence_count,
        "selected_blocked_gate_missing_evidence_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_missing_evidence
        ),
        "next_blocker_artifact_name": next_blocker.get("artifact_name")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_readiness_gate": next_blocker.get("readiness_gate")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_status": next_blocker.get("status")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_reason": next_blocker.get("reason")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_command_kind": next_blocker.get("validator_command_kind")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_command": next_blocker_command,
        "next_blocker_command_sha256": status_mod._stable_json_sha256(
            next_blocker_command
        ),
        "next_producer_command_kind": next_blocker.get("producer_command_kind")
        if isinstance(next_blocker, dict)
        else None,
        "next_producer_command": next_producer_command,
        "next_producer_command_sha256": status_mod._stable_json_sha256(
            next_producer_command
        ),
        "next_action_available": next_action_available,
        "next_action_artifact_name": next_action.get("artifact_name")
        if isinstance(next_action, dict)
        else None,
        "next_action_readiness_gate": next_action.get("readiness_gate")
        if isinstance(next_action, dict)
        else None,
        "next_action_status": next_action.get("status")
        if isinstance(next_action, dict)
        else None,
        "next_action_reason": next_action.get("reason")
        if isinstance(next_action, dict)
        else None,
        "next_action_validator_command_kind": next_action.get("validator_command_kind")
        if isinstance(next_action, dict)
        else None,
        "next_action_validator_command": next_action.get("validator_command")
        if isinstance(next_action, dict)
        else None,
        "next_action_validator_command_sha256": status_mod._stable_json_sha256(
            next_action.get("validator_command") if isinstance(next_action, dict) else None
        ),
        "next_action_producer_command_kind": next_action.get("producer_command_kind")
        if isinstance(next_action, dict)
        else None,
        "next_action_producer_command": next_action.get("producer_command")
        if isinstance(next_action, dict)
        else None,
        "next_action_producer_command_sha256": status_mod._stable_json_sha256(
            next_action.get("producer_command") if isinstance(next_action, dict) else None
        ),
        "next_action_partial_output_handoff": next_action_partial_output_handoff,
        "next_action_partial_output_handoff_sha256": status_mod._stable_json_sha256(
            next_action_partial_output_handoff
        ),
        "next_action_partial_output_path": next_action_partial_output_path,
        "next_action_partial_output_status": next_action_partial_output_status,
        "next_action_sha256": status_mod._stable_json_sha256(next_action),
        "next_action_validator_summary_sha256": status_mod._stable_json_sha256(
            next_action_validator_summary
        ),
        "next_action_validator_summary_status": next_action_validator_summary_status,
        "next_action_validator_summary_ready": next_action_validator_summary_ready,
        "next_action_validator_summary_oracle_status": (
            next_action_validator_summary_oracle_status
        ),
        "next_action_validator_summary_oracle_blocker_kind": (
            next_action_validator_summary_oracle_blocker_kind
        ),
        "next_action_oracle_expected_token": next_action_oracle_expected_token,
        "next_action_oracle_expected_token_sha256": status_mod._stable_json_sha256(
            next_action_oracle_expected_token
        ),
        "next_action_expected_next_token_id": next_action_expected_next_token_id,
        "next_action_expected_next_token_text": next_action_expected_next_token_text,
        "next_action_expected_next_token_logit": next_action_expected_next_token_logit,
        "next_action_oracle_generated_text": next_action_oracle_generated_text,
        "next_action_oracle_generated_text_sha256": status_mod._stable_json_sha256(
            next_action_oracle_generated_text
        ),
        "next_action_generated_text": next_action_generated_text,
        "next_action_generated_text_len": next_action_generated_text_len,
        "next_action_generated_text_matches_expected_exact": (
            next_action_generated_text_matches_expected_exact
        ),
        "next_action_generated_text_matches_expected_stripped": (
            next_action_generated_text_matches_expected_stripped
        ),
        "next_action_oracle_artifact_provenance": (
            next_action_oracle_artifact_provenance
        ),
        "next_action_oracle_artifact_provenance_sha256": (
            status_mod._stable_json_sha256(next_action_oracle_artifact_provenance)
        ),
        "next_action_oracle_artifact_path": next_action_oracle_artifact_path,
        "next_action_oracle_artifact_sha256": next_action_oracle_artifact_sha256,
        "next_action_prompt_artifact_path": next_action_prompt_artifact_path,
        "next_action_prompt_artifact_sha256": next_action_prompt_artifact_sha256,
        "next_action_evidence_checks_sha256": next_action_evidence_checks_sha256,
        "next_action_oracle_artifact_presence": next_action_oracle_artifact_presence,
        "next_action_oracle_artifact_presence_sha256": (
            status_mod._stable_json_sha256(next_action_oracle_artifact_presence)
        ),
        "next_action_oracle_artifact_present": next_action_oracle_artifact_present,
        "next_action_prompt_artifact_present": next_action_prompt_artifact_present,
        "next_action_no_claim_policy": next_action_no_claim_policy,
        "next_action_no_claim_policy_sha256": status_mod._stable_json_sha256(
            next_action_no_claim_policy
        ),
        "next_action_oracle_parity_claim_allowed": (
            next_action_oracle_parity_claim_allowed
        ),
        "next_action_kv_backed_decode_claim_allowed": (
            next_action_kv_backed_decode_claim_allowed
        ),
        "next_action_e2e_inference_claim_allowed": (
            next_action_e2e_inference_claim_allowed
        ),
        "next_action_performance_claim_allowed": (
            next_action_performance_claim_allowed
        ),
        "next_action_missing_evidence": next_action_missing_evidence,
        "next_action_missing_evidence_count": next_action_missing_evidence_count,
        "next_action_oracle_evidence_gaps": next_action_oracle_evidence_gaps,
        "next_action_oracle_evidence_gaps_sha256": status_mod._stable_json_sha256(
            next_action_oracle_evidence_gaps
        ),
        "next_action_oracle_evidence_gap_count": (
            next_action_oracle_evidence_gap_count
        ),
        "next_action_missing_evidence_present": next_action_missing_evidence_present,
        "next_action_missing_evidence_joined": next_action_missing_evidence_joined,
        "next_action_missing_evidence_sorted": next_action_missing_evidence_sorted,
        "next_action_missing_evidence_sorted_sha256": status_mod._stable_json_sha256(
            next_action_missing_evidence_sorted
        ),
        "next_action_missing_evidence_sorted_joined": (
            next_action_missing_evidence_sorted_joined
        ),
        "next_action_first_missing_evidence": next_action_first_missing_evidence,
        "next_action_last_missing_evidence": next_action_last_missing_evidence,
        "next_action_artifact_file_present_missing": (
            next_action_artifact_file_present_missing
        ),
        "next_action_oracle_success_status_missing": (
            next_action_oracle_success_status_missing
        ),
        "next_action_oracle_returncode_zero_missing": (
            next_action_oracle_returncode_zero_missing
        ),
        "next_action_no_timeout_or_oracle_blocker_missing": (
            next_action_no_timeout_or_oracle_blocker_missing
        ),
        "next_action_generated_text_matches_target_missing": (
            next_action_generated_text_matches_target_missing
        ),
        "next_action_generated_text_nonempty_missing": (
            next_action_generated_text_nonempty_missing
        ),
        "next_action_missing_evidence_sha256": status_mod._stable_json_sha256(
            next_action_missing_evidence
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
        "blocked_evidence_summary": blocked_evidence_summary,
        "blocked_evidence_summary_sha256": status_mod._stable_json_sha256(
            blocked_evidence_summary
        ),
        "blocked_evidence_by_gate": blocked_evidence_by_gate,
        "blocked_evidence_by_gate_sha256": status_mod._stable_json_sha256(
            blocked_evidence_by_gate
        ),
        "blocked_readiness_gates": blocked_readiness_gates,
        "blocked_readiness_gates_sha256": status_mod._stable_json_sha256(
            blocked_readiness_gates
        ),
        "next_blocked_gate": next_blocked_gate,
        "next_blocked_gate_sha256": status_mod._stable_json_sha256(
            next_blocked_gate
        ),
        "selected_blocked_gate_readiness_gate": selected_blocked_gate_name,
        "selected_blocked_gate_found": selected_blocked_gate is not None,
        "selected_blocked_gate": selected_blocked_gate,
        "selected_blocked_gate_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate
        ),
        "selected_blocked_gate_artifact_names": selected_blocked_gate_artifact_names,
        "selected_blocked_gate_artifact_count": selected_blocked_gate_artifact_count,
        "selected_blocked_gate_artifact_names_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_artifact_names
        ),
        "selected_blocked_gate_blocked_count": selected_blocked_gate_blocked_count,
        "selected_blocked_gate_status_counts": selected_blocked_gate_status_counts,
        "selected_blocked_gate_status_counts_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_status_counts
        ),
        "selected_blocked_gate_producer_commands": selected_blocked_gate_producer_commands,
        "selected_blocked_gate_producer_command_count": selected_blocked_gate_producer_command_count,
        "selected_blocked_gate_producer_commands_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_producer_commands
        ),
        "selected_blocked_gate_validator_commands": selected_blocked_gate_validator_commands,
        "selected_blocked_gate_validator_command_count": selected_blocked_gate_validator_command_count,
        "selected_blocked_gate_validator_commands_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_validator_commands
        ),
        "selected_blocked_gate_missing_evidence": selected_blocked_gate_missing_evidence,
        "selected_blocked_gate_missing_evidence_count": selected_blocked_gate_missing_evidence_count,
        "selected_blocked_gate_missing_evidence_sha256": status_mod._stable_json_sha256(
            selected_blocked_gate_missing_evidence
        ),
        "next_blocker": next_blocker,
        "next_blocker_sha256": status_mod._stable_json_sha256(next_blocker),
        "next_blocker_artifact_name": next_blocker.get("artifact_name")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_readiness_gate": next_blocker.get("readiness_gate")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_status": next_blocker.get("status")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_reason": next_blocker.get("reason")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_command": next_blocker_command,
        "next_blocker_command_kind": next_blocker.get("validator_command_kind")
        if isinstance(next_blocker, dict)
        else None,
        "next_blocker_command_sha256": status_mod._stable_json_sha256(
            next_blocker_command
        ),
        "next_producer_command_kind": next_blocker.get("producer_command_kind")
        if isinstance(next_blocker, dict)
        else None,
        "next_producer_command": next_producer_command,
        "next_producer_command_sha256": status_mod._stable_json_sha256(
            next_producer_command
        ),
        "next_action": next_action,
        "next_action_available": summary["next_action_available"],
        "next_action_validator_summary_sha256": summary[
            "next_action_validator_summary_sha256"
        ],
        "next_action_validator_summary_status": summary[
            "next_action_validator_summary_status"
        ],
        "next_action_validator_summary_ready": summary[
            "next_action_validator_summary_ready"
        ],
        "next_action_validator_summary_oracle_status": summary[
            "next_action_validator_summary_oracle_status"
        ],
        "next_action_validator_summary_oracle_blocker_kind": summary[
            "next_action_validator_summary_oracle_blocker_kind"
        ],
        "next_action_oracle_expected_token": summary[
            "next_action_oracle_expected_token"
        ],
        "next_action_oracle_expected_token_sha256": summary[
            "next_action_oracle_expected_token_sha256"
        ],
        "next_action_expected_next_token_id": summary[
            "next_action_expected_next_token_id"
        ],
        "next_action_expected_next_token_text": summary[
            "next_action_expected_next_token_text"
        ],
        "next_action_expected_next_token_logit": summary[
            "next_action_expected_next_token_logit"
        ],
        "next_action_oracle_generated_text": summary[
            "next_action_oracle_generated_text"
        ],
        "next_action_oracle_generated_text_sha256": summary[
            "next_action_oracle_generated_text_sha256"
        ],
        "next_action_generated_text": summary["next_action_generated_text"],
        "next_action_generated_text_len": summary[
            "next_action_generated_text_len"
        ],
        "next_action_generated_text_matches_expected_exact": summary[
            "next_action_generated_text_matches_expected_exact"
        ],
        "next_action_generated_text_matches_expected_stripped": summary[
            "next_action_generated_text_matches_expected_stripped"
        ],
        "next_action_oracle_artifact_provenance": summary[
            "next_action_oracle_artifact_provenance"
        ],
        "next_action_oracle_artifact_provenance_sha256": summary[
            "next_action_oracle_artifact_provenance_sha256"
        ],
        "next_action_oracle_artifact_path": summary[
            "next_action_oracle_artifact_path"
        ],
        "next_action_oracle_artifact_sha256": summary[
            "next_action_oracle_artifact_sha256"
        ],
        "next_action_prompt_artifact_path": summary[
            "next_action_prompt_artifact_path"
        ],
        "next_action_prompt_artifact_sha256": summary[
            "next_action_prompt_artifact_sha256"
        ],
        "next_action_evidence_checks_sha256": summary[
            "next_action_evidence_checks_sha256"
        ],
        "next_action_oracle_artifact_presence": summary[
            "next_action_oracle_artifact_presence"
        ],
        "next_action_oracle_artifact_presence_sha256": summary[
            "next_action_oracle_artifact_presence_sha256"
        ],
        "next_action_oracle_artifact_present": summary[
            "next_action_oracle_artifact_present"
        ],
        "next_action_prompt_artifact_present": summary[
            "next_action_prompt_artifact_present"
        ],
        "next_action_no_claim_policy": summary["next_action_no_claim_policy"],
        "next_action_no_claim_policy_sha256": summary[
            "next_action_no_claim_policy_sha256"
        ],
        "next_action_oracle_parity_claim_allowed": summary[
            "next_action_oracle_parity_claim_allowed"
        ],
        "next_action_kv_backed_decode_claim_allowed": summary[
            "next_action_kv_backed_decode_claim_allowed"
        ],
        "next_action_e2e_inference_claim_allowed": summary[
            "next_action_e2e_inference_claim_allowed"
        ],
        "next_action_performance_claim_allowed": summary[
            "next_action_performance_claim_allowed"
        ],
        "next_action_artifact_name": summary["next_action_artifact_name"],
        "next_action_readiness_gate": summary["next_action_readiness_gate"],
        "next_action_status": summary["next_action_status"],
        "next_action_reason": summary["next_action_reason"],
        "next_action_missing_evidence_count": summary[
            "next_action_missing_evidence_count"
        ],
        "next_action_oracle_evidence_gaps": summary[
            "next_action_oracle_evidence_gaps"
        ],
        "next_action_oracle_evidence_gaps_sha256": summary[
            "next_action_oracle_evidence_gaps_sha256"
        ],
        "next_action_oracle_evidence_gap_count": summary[
            "next_action_oracle_evidence_gap_count"
        ],
        "next_action_missing_evidence_present": summary[
            "next_action_missing_evidence_present"
        ],
        "next_action_missing_evidence_joined": summary[
            "next_action_missing_evidence_joined"
        ],
        "next_action_missing_evidence_sorted": summary[
            "next_action_missing_evidence_sorted"
        ],
        "next_action_missing_evidence_sorted_sha256": summary[
            "next_action_missing_evidence_sorted_sha256"
        ],
        "next_action_missing_evidence_sorted_joined": summary[
            "next_action_missing_evidence_sorted_joined"
        ],
        "next_action_first_missing_evidence": summary[
            "next_action_first_missing_evidence"
        ],
        "next_action_last_missing_evidence": summary[
            "next_action_last_missing_evidence"
        ],
        "next_action_artifact_file_present_missing": summary[
            "next_action_artifact_file_present_missing"
        ],
        "next_action_oracle_success_status_missing": summary[
            "next_action_oracle_success_status_missing"
        ],
        "next_action_oracle_returncode_zero_missing": summary[
            "next_action_oracle_returncode_zero_missing"
        ],
        "next_action_no_timeout_or_oracle_blocker_missing": summary[
            "next_action_no_timeout_or_oracle_blocker_missing"
        ],
        "next_action_generated_text_matches_target_missing": summary[
            "next_action_generated_text_matches_target_missing"
        ],
        "next_action_generated_text_nonempty_missing": summary[
            "next_action_generated_text_nonempty_missing"
        ],
        "next_action_missing_evidence_sha256": summary[
            "next_action_missing_evidence_sha256"
        ],
        "next_action_validator_command_kind": summary[
            "next_action_validator_command_kind"
        ],
        "next_action_validator_command": summary["next_action_validator_command"],
        "next_action_validator_command_sha256": summary[
            "next_action_validator_command_sha256"
        ],
        "next_action_producer_command_kind": summary[
            "next_action_producer_command_kind"
        ],
        "next_action_producer_command": summary["next_action_producer_command"],
        "next_action_producer_command_sha256": summary[
            "next_action_producer_command_sha256"
        ],
        "next_action_partial_output_handoff": summary[
            "next_action_partial_output_handoff"
        ],
        "next_action_partial_output_handoff_sha256": summary[
            "next_action_partial_output_handoff_sha256"
        ],
        "next_action_partial_output_path": summary[
            "next_action_partial_output_path"
        ],
        "next_action_partial_output_status": summary[
            "next_action_partial_output_status"
        ],
        "next_action_sha256": status_mod._stable_json_sha256(next_action),
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
        selected_blocked_gate_name=args.blocked_evidence_gate,
    )
    next_action = report.get("next_action")
    next_action_validator_summary = (
        next_action.get("validator_summary") if isinstance(next_action, dict) else None
    )
    next_action_missing_evidence = None
    if isinstance(next_action, dict):
        next_action_missing_evidence = next_action.get("validator_missing_evidence")
        if next_action_missing_evidence is None and isinstance(
            next_action_validator_summary, dict
        ):
            next_action_missing_evidence = next_action_validator_summary.get(
                "missing_evidence"
            )
    next_action_missing_evidence_count = (
        len(next_action_missing_evidence)
        if isinstance(next_action_missing_evidence, list)
        else None
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.next_action_available_only:
        payload = report["next_action_available"]
    elif args.next_action_validator_summary_sha_only:
        payload = report["next_action_validator_summary_sha256"]
    elif args.next_action_validator_summary_status_only:
        payload = report["next_action_validator_summary_status"]
    elif args.next_action_validator_summary_ready_only:
        payload = report["next_action_validator_summary_ready"]
    elif args.next_action_validator_summary_oracle_status_only:
        payload = report["next_action_validator_summary_oracle_status"]
    elif args.next_action_validator_summary_oracle_blocker_kind_only:
        payload = report["next_action_validator_summary_oracle_blocker_kind"]
    elif args.next_action_oracle_expected_token_sha_only:
        payload = report["next_action_oracle_expected_token_sha256"]
    elif args.next_action_oracle_expected_token_only:
        payload = report["next_action_oracle_expected_token"]
    elif args.next_action_expected_next_token_id_only:
        payload = report["next_action_expected_next_token_id"]
    elif args.next_action_expected_next_token_text_only:
        payload = report["next_action_expected_next_token_text"]
    elif args.next_action_expected_next_token_logit_only:
        payload = report["next_action_expected_next_token_logit"]
    elif args.next_action_oracle_generated_text_sha_only:
        payload = report["next_action_oracle_generated_text_sha256"]
    elif args.next_action_oracle_generated_text_only:
        payload = report["next_action_oracle_generated_text"]
    elif args.next_action_generated_text_only:
        payload = report["next_action_generated_text"]
    elif args.next_action_generated_text_len_only:
        payload = report["next_action_generated_text_len"]
    elif args.next_action_generated_text_matches_expected_exact_only:
        payload = report["next_action_generated_text_matches_expected_exact"]
    elif args.next_action_generated_text_matches_expected_stripped_only:
        payload = report["next_action_generated_text_matches_expected_stripped"]
    elif args.next_action_oracle_artifact_provenance_sha_only:
        payload = report["next_action_oracle_artifact_provenance_sha256"]
    elif args.next_action_oracle_artifact_provenance_only:
        payload = report["next_action_oracle_artifact_provenance"]
    elif args.next_action_oracle_artifact_path_only:
        payload = report["next_action_oracle_artifact_path"]
    elif args.next_action_oracle_artifact_sha_only:
        payload = report["next_action_oracle_artifact_sha256"]
    elif args.next_action_prompt_artifact_path_only:
        payload = report["next_action_prompt_artifact_path"]
    elif args.next_action_prompt_artifact_sha_only:
        payload = report["next_action_prompt_artifact_sha256"]
    elif args.next_action_evidence_checks_sha_only:
        payload = report["next_action_evidence_checks_sha256"]
    elif args.next_action_oracle_artifact_presence_sha_only:
        payload = report["next_action_oracle_artifact_presence_sha256"]
    elif args.next_action_oracle_artifact_presence_only:
        payload = report["next_action_oracle_artifact_presence"]
    elif args.next_action_oracle_artifact_present_only:
        payload = report["next_action_oracle_artifact_present"]
    elif args.next_action_prompt_artifact_present_only:
        payload = report["next_action_prompt_artifact_present"]
    elif args.next_action_no_claim_policy_sha_only:
        payload = report["next_action_no_claim_policy_sha256"]
    elif args.next_action_no_claim_policy_only:
        payload = report["next_action_no_claim_policy"]
    elif args.next_action_oracle_parity_claim_allowed_only:
        payload = report["next_action_oracle_parity_claim_allowed"]
    elif args.next_action_kv_backed_decode_claim_allowed_only:
        payload = report["next_action_kv_backed_decode_claim_allowed"]
    elif args.next_action_e2e_inference_claim_allowed_only:
        payload = report["next_action_e2e_inference_claim_allowed"]
    elif args.next_action_performance_claim_allowed_only:
        payload = report["next_action_performance_claim_allowed"]
    elif args.next_action_validator_summary_only:
        payload = next_action_validator_summary
    elif args.next_action_missing_evidence_count_only:
        payload = report["next_action_missing_evidence_count"]
    elif args.next_action_oracle_evidence_gap_count_only:
        payload = report["next_action_oracle_evidence_gap_count"]
    elif args.next_action_oracle_evidence_gaps_sha_only:
        payload = report["next_action_oracle_evidence_gaps_sha256"]
    elif args.next_action_oracle_evidence_gaps_only:
        payload = report["next_action_oracle_evidence_gaps"]
    elif args.next_action_missing_evidence_present_only:
        payload = report["next_action_missing_evidence_present"]
    elif args.next_action_missing_evidence_joined_only:
        payload = report["next_action_missing_evidence_joined"]
    elif args.next_action_missing_evidence_sorted_sha_only:
        payload = report["next_action_missing_evidence_sorted_sha256"]
    elif args.next_action_missing_evidence_sorted_only:
        payload = report["next_action_missing_evidence_sorted"]
    elif args.next_action_missing_evidence_sorted_joined_only:
        payload = report["next_action_missing_evidence_sorted_joined"]
    elif args.next_action_first_missing_evidence_only:
        payload = report["next_action_first_missing_evidence"]
    elif args.next_action_last_missing_evidence_only:
        payload = report["next_action_last_missing_evidence"]
    elif args.next_action_artifact_file_present_missing_only:
        payload = report["next_action_artifact_file_present_missing"]
    elif args.next_action_oracle_success_status_missing_only:
        payload = report["next_action_oracle_success_status_missing"]
    elif args.next_action_oracle_returncode_zero_missing_only:
        payload = report["next_action_oracle_returncode_zero_missing"]
    elif args.next_action_no_timeout_or_oracle_blocker_missing_only:
        payload = report["next_action_no_timeout_or_oracle_blocker_missing"]
    elif args.next_action_generated_text_matches_target_missing_only:
        payload = report["next_action_generated_text_matches_target_missing"]
    elif args.next_action_generated_text_nonempty_missing_only:
        payload = report["next_action_generated_text_nonempty_missing"]
    elif args.next_action_missing_evidence_sha_only:
        payload = report["next_action_missing_evidence_sha256"]
    elif args.next_action_missing_evidence_only:
        payload = next_action_missing_evidence
    elif args.next_action_sha_only:
        payload = report["next_action_sha256"]
    elif args.next_action_artifact_name_only:
        payload = report["next_action_artifact_name"]
    elif args.next_action_readiness_gate_only:
        payload = report["next_action_readiness_gate"]
    elif args.next_action_status_only:
        payload = report["next_action_status"]
    elif args.next_action_reason_only:
        payload = report["next_action_reason"]
    elif args.next_action_validator_command_kind_only:
        payload = report["next_action_validator_command_kind"]
    elif args.next_action_validator_command_only:
        payload = report["next_action_validator_command"]
    elif args.next_action_validator_command_sha_only:
        payload = report["next_action_validator_command_sha256"]
    elif args.next_action_producer_command_kind_only:
        payload = report["next_action_producer_command_kind"]
    elif args.next_action_producer_command_only:
        payload = report["next_action_producer_command"]
    elif args.next_action_producer_command_sha_only:
        payload = report["next_action_producer_command_sha256"]
    elif args.next_action_partial_output_handoff_sha_only:
        payload = report["next_action_partial_output_handoff_sha256"]
    elif args.next_action_partial_output_handoff_only:
        payload = report["next_action_partial_output_handoff"]
    elif args.next_action_partial_output_path_only:
        payload = report["next_action_partial_output_path"]
    elif args.next_action_partial_output_status_only:
        payload = report["next_action_partial_output_status"]
    elif args.next_action_only:
        payload = report["next_action"]
    elif args.next_producer_command_kind_only:
        payload = report["next_producer_command_kind"]
    elif args.next_producer_command_sha_only:
        payload = report["next_producer_command_sha256"]
    elif args.next_producer_command_only:
        payload = report["next_producer_command"]
    elif args.next_command_kind_only:
        payload = report["next_blocker_command_kind"]
    elif args.next_command_sha_only:
        payload = report["next_blocker_command_sha256"]
    elif args.next_command_only:
        payload = report["next_blocker_command"]
    elif args.next_blocker_reason_only:
        payload = report["next_blocker_reason"]
    elif args.next_blocker_status_only:
        payload = report["next_blocker_status"]
    elif args.next_blocker_readiness_gate_only:
        payload = report["next_blocker_readiness_gate"]
    elif args.next_blocker_artifact_name_only:
        payload = report["next_blocker_artifact_name"]
    elif args.next_blocker_sha_only:
        payload = report["next_blocker_sha256"]
    elif args.next_blocker_only:
        payload = report["next_blocker"]
    elif args.next_blocked_gate_sha_only:
        payload = report["next_blocked_gate_sha256"]
    elif args.next_blocked_gate_only:
        payload = report["next_blocked_gate"]
    elif args.blocked_readiness_gates_sha_only:
        payload = report["blocked_readiness_gates_sha256"]
    elif args.blocked_readiness_gates_only:
        payload = report["blocked_readiness_gates"]
    elif args.blocked_evidence_gate_found_only:
        payload = report["selected_blocked_gate_found"]
    elif args.blocked_evidence_gate_artifact_count_only:
        payload = report["selected_blocked_gate_artifact_count"]
    elif args.blocked_evidence_gate_blocked_count_only:
        payload = report["selected_blocked_gate_blocked_count"]
    elif args.blocked_evidence_gate_status_counts_sha_only:
        payload = report["selected_blocked_gate_status_counts_sha256"]
    elif args.blocked_evidence_gate_status_counts_only:
        payload = report["selected_blocked_gate_status_counts"]
    elif args.blocked_evidence_gate_producer_command_count_only:
        payload = report["selected_blocked_gate_producer_command_count"]
    elif args.blocked_evidence_gate_producer_commands_sha_only:
        payload = report["selected_blocked_gate_producer_commands_sha256"]
    elif args.blocked_evidence_gate_producer_commands_only:
        payload = report["selected_blocked_gate_producer_commands"]
    elif args.blocked_evidence_gate_validator_command_count_only:
        payload = report["selected_blocked_gate_validator_command_count"]
    elif args.blocked_evidence_gate_validator_commands_sha_only:
        payload = report["selected_blocked_gate_validator_commands_sha256"]
    elif args.blocked_evidence_gate_validator_commands_only:
        payload = report["selected_blocked_gate_validator_commands"]
    elif args.blocked_evidence_gate_artifacts_sha_only:
        payload = report["selected_blocked_gate_artifact_names_sha256"]
    elif args.blocked_evidence_gate_artifacts_only:
        payload = report["selected_blocked_gate_artifact_names"]
    elif args.blocked_evidence_gate_missing_evidence_count_only:
        payload = report["selected_blocked_gate_missing_evidence_count"]
    elif args.blocked_evidence_gate_missing_evidence_sha_only:
        payload = report["selected_blocked_gate_missing_evidence_sha256"]
    elif args.blocked_evidence_gate_missing_evidence_only:
        payload = report["selected_blocked_gate_missing_evidence"]
    elif args.blocked_evidence_gate_sha_only:
        payload = report["selected_blocked_gate_sha256"]
    elif args.blocked_evidence_gate_only:
        payload = report["selected_blocked_gate"]
    elif args.blocked_evidence_by_gate_sha_only:
        payload = report["blocked_evidence_by_gate_sha256"]
    elif args.blocked_evidence_by_gate_only:
        payload = report["blocked_evidence_by_gate"]
    elif args.blocked_evidence_summary_sha_only:
        payload = report["blocked_evidence_summary_sha256"]
    elif args.blocked_evidence_summary_only:
        payload = report["blocked_evidence_summary"]
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
