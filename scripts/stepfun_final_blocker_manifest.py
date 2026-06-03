#!/usr/bin/env python3
"""Emit a compact manifest for the remaining StepFun GGUF correctness blockers."""

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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Prompt/layer-prefix artifact to summarize.",
    )
    parser.add_argument(
        "--oracle-artifact",
        type=Path,
        default=status_mod.DEFAULT_ORACLE_ARTIFACT,
        help="llama.cpp oracle artifact to summarize.",
    )
    parser.add_argument(
        "--resource-artifact",
        type=Path,
        default=status_mod.DEFAULT_RESOURCE_ARTIFACT,
        help="StepFun text-resource dry-run artifact to summarize.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=status_mod.DEFAULT_DOCS_PATH,
        help="docs/STEPFUN.md checklist source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON output to this path instead of stdout.",
    )
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="Compare a persisted final-blocker manifest with the current inputs.",
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-manifest, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-manifest, emit only verification_failures.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the manifest.",
    )
    parser.add_argument(
        "--entries-only",
        action="store_true",
        help="Emit only the manifest entries for compact blocker polling.",
    )
    parser.add_argument(
        "--entries-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of manifest entries.",
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Emit only artifacts_to_collect for compact evidence polling.",
    )
    parser.add_argument(
        "--artifacts-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of artifacts_to_collect.",
    )
    parser.add_argument(
        "--success-criteria-only",
        action="store_true",
        help="Emit only the compact success criteria for each remaining blocker.",
    )
    parser.add_argument(
        "--success-criteria-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the success-criteria handoff.",
    )
    parser.add_argument(
        "--no-claim-policy-only",
        action="store_true",
        help="Emit only the no-claim policy for compact claim-gate polling.",
    )
    parser.add_argument(
        "--no-claim-policy-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the no-claim policy.",
    )
    parser.add_argument(
        "--gate-status-only",
        action="store_true",
        help="Emit only compact readiness-gate status for remaining blockers.",
    )
    parser.add_argument(
        "--gate-status-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the gate-status handoff.",
    )
    parser.add_argument(
        "--status-provenance-only",
        action="store_true",
        help="Emit only status/source provenance for compact manifest drift polling.",
    )
    parser.add_argument(
        "--status-provenance-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of status/source provenance.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args(argv)


def build_final_blocker_manifest(status: dict[str, object]) -> dict[str, object]:
    """Return the compact evidence manifest for the two final P11 blockers."""

    remaining = dict(status.get("remaining_blockers_report", {}))
    remaining_items = [
        item for item in remaining.get("items", []) if isinstance(item, dict)
    ]
    readiness_gates = dict(status.get("readiness_gates", {}))
    source_artifacts = dict(status.get("source_artifacts", {}))
    oracle_source = dict(source_artifacts.get("oracle", {}))
    oracle_progress = dict(status.get("oracle_progress", {}))
    oracle_gap_report = dict(status.get("oracle_gap_report", {}))
    oracle_partial_handoff = dict(status.get("oracle_partial_output_handoff", {}))
    kv_gap_report = dict(status.get("kv_backed_decode_gap_report", {}))
    kv_blocker_summary = dict(kv_gap_report.get("kv_decode_blocker_summary", {}))
    kv_launch_trace_summary = dict(
        kv_gap_report.get("streaming_decode_launch_trace_summary", {})
    )
    status_provenance = {
        "source_artifacts_sha256": status.get("source_artifacts_sha256"),
        "status_integrity_sha256": status.get("status_integrity_sha256"),
        "handoff_summary_sha256": status.get("handoff_summary_sha256"),
        "readiness_summary_sha256": status.get("readiness_summary_sha256"),
        "next_action_commands_sha256": status.get("next_action_commands_sha256"),
        "oracle_progress_sha256": status.get("oracle_progress_sha256"),
        "oracle_partial_output_handoff_sha256": status.get(
            "oracle_partial_output_handoff_sha256"
        ),
        "source_artifacts": source_artifacts,
    }

    entries: list[dict[str, object]] = []
    artifacts_to_collect: list[dict[str, object]] = []
    for item in remaining_items:
        blocker_kind = str(item.get("blocker_kind"))
        gate_name = item.get("readiness_gate")
        gate = readiness_gates.get(gate_name, {}) if isinstance(gate_name, str) else {}
        gate_record = gate if isinstance(gate, dict) else {}
        missing_evidence = list(item.get("missing_evidence", []))
        entry: dict[str, object] = {
            "blocker_kind": blocker_kind,
            "checklist_item": item.get("checklist_item"),
            "queue_index": item.get("queue_index"),
            "readiness_gate": gate_name,
            "gate_ready": item.get("gate_ready") is True,
            "gate_blocked_by": item.get("gate_blocked_by"),
            "required_evidence": gate_record.get("required_evidence"),
            "first_missing_evidence": item.get("first_missing_evidence")
            or (missing_evidence[0] if missing_evidence else None),
            "missing_evidence": missing_evidence,
            "success_criteria": list(item.get("success_criteria", [])),
            "recommended_command_kind": item.get("recommended_command_kind"),
            "recommended_command_sha256": item.get("recommended_command_sha256"),
        }
        if blocker_kind == "oracle_parity_blocked":
            artifact = {
                "name": "llama_cpp_oracle_success_artifact",
                "required_for": blocker_kind,
                "readiness_gate": "oracle_parity",
                "path": oracle_source.get("path"),
                "source_sha256": oracle_source.get("sha256"),
                "current_status": oracle_progress.get("status"),
                "current_blocker_kind": oracle_progress.get("oracle_blocker_kind"),
                "expected_next_token_id": oracle_progress.get("expected_next_token_id"),
                "expected_next_token_text": oracle_progress.get("expected_next_token_text"),
                "first_missing_evidence": oracle_gap_report.get(
                    "first_missing_evidence"
                ),
                "recommended_command_kind": item.get("recommended_command_kind"),
                "recommended_command_sha256": item.get("recommended_command_sha256"),
                "partial_output_handoff_safe": oracle_partial_handoff.get(
                    "all_partial_output_contracts_safe"
                )
                is True,
            }
            entry["artifact_handoff"] = artifact
            artifacts_to_collect.append(artifact)
        elif blocker_kind == "kv_backed_decode_not_wired":
            kv_required_artifacts = [
                dict(record)
                for record in kv_blocker_summary.get("artifacts_needed", [])
                if isinstance(record, dict)
            ]
            artifact = {
                "name": "kv_backed_decode_runtime_artifacts",
                "required_for": blocker_kind,
                "readiness_gate": "kv_backed_decode",
                "first_streaming_runner_blocker": kv_gap_report.get(
                    "first_streaming_runner_blocker"
                ),
                "streaming_runner_blocker_names": list(
                    kv_gap_report.get("streaming_runner_blocker_names", [])
                ),
                "streaming_decode_launch_trace_sha256": kv_gap_report.get(
                    "streaming_decode_launch_trace_sha256"
                ),
                "streaming_decode_launch_trace_summary_sha256": kv_gap_report.get(
                    "streaming_decode_launch_trace_summary_sha256"
                ),
                "launch_trace_operation_count": kv_launch_trace_summary.get(
                    "operation_count"
                ),
                "launch_trace_non_executable": kv_launch_trace_summary.get(
                    "non_executable"
                )
                is True,
                "required_artifacts": kv_required_artifacts,
                "required_artifact_names": [
                    str(record.get("name")) for record in kv_required_artifacts
                ],
                "required_artifacts_sha256": kv_blocker_summary.get(
                    "artifacts_needed_sha256"
                ),
            }
            entry["artifact_handoff"] = artifact
            artifacts_to_collect.extend(kv_required_artifacts)
        entries.append(entry)

    success_criteria_handoff = [
        {
            "blocker_kind": entry.get("blocker_kind"),
            "readiness_gate": entry.get("readiness_gate"),
            "gate_ready": entry.get("gate_ready"),
            "first_missing_evidence": entry.get("first_missing_evidence"),
            "success_criteria": list(entry.get("success_criteria", [])),
            "recommended_command_kind": entry.get("recommended_command_kind"),
            "recommended_command_sha256": entry.get("recommended_command_sha256"),
        }
        for entry in entries
    ]
    entry_by_gate = {
        str(entry.get("readiness_gate")): entry
        for entry in entries
        if entry.get("readiness_gate") is not None
    }
    gate_status_handoff = []
    for gate_name in remaining.get("blocked_gates", []):
        gate_record = readiness_gates.get(gate_name, {})
        gate = gate_record if isinstance(gate_record, dict) else {}
        entry = entry_by_gate.get(str(gate_name), {})
        gate_status_handoff.append(
            {
                "readiness_gate": gate_name,
                "ready": gate.get("ready"),
                "blocked_by": gate.get("blocked_by"),
                "required_evidence": gate.get("required_evidence"),
                "blocker_kind": entry.get("blocker_kind"),
                "first_missing_evidence": entry.get("first_missing_evidence"),
                "success_criteria": list(entry.get("success_criteria", [])),
            }
        )
    no_claim_policy = dict(remaining.get("no_claim_policy", {}))
    entries_sha256 = status_mod._stable_json_sha256(entries)
    artifacts_to_collect_sha256 = status_mod._stable_json_sha256(artifacts_to_collect)
    success_criteria_handoff_sha256 = status_mod._stable_json_sha256(
        success_criteria_handoff
    )
    no_claim_policy_sha256 = status_mod._stable_json_sha256(no_claim_policy)
    gate_status_handoff_sha256 = status_mod._stable_json_sha256(gate_status_handoff)
    status_provenance_sha256 = status_mod._stable_json_sha256(status_provenance)
    compact_output_modes = {
        "sha_only": "manifest_sha256",
        "entries_only": "entries",
        "entries_sha_only": "entries_sha256",
        "artifacts_only": "artifacts_to_collect",
        "artifacts_sha_only": "artifacts_to_collect_sha256",
        "success_criteria_only": "success_criteria_handoff",
        "success_criteria_sha_only": "success_criteria_handoff_sha256",
        "no_claim_policy_only": "no_claim_policy",
        "no_claim_policy_sha_only": "no_claim_policy_sha256",
        "gate_status_only": "gate_status_handoff",
        "gate_status_sha_only": "gate_status_handoff_sha256",
        "status_provenance_only": "status_provenance",
        "status_provenance_sha_only": "status_provenance_sha256",
        "verification_status_only": "verification.status",
        "verification_failures_only": "verification.verification_failures",
    }

    return {
        "schema_version": 1,
        "status": remaining.get("status", "blocked"),
        "status_provenance": status_provenance,
        "status_provenance_sha256": status_provenance_sha256,
        "open_or_partial_items_p0_p12": remaining.get(
            "open_or_partial_items_p0_p12"
        ),
        "remaining_blocker_count": remaining.get("remaining_blocker_count"),
        "remaining_blocker_kinds": list(remaining.get("remaining_blocker_kinds", [])),
        "blocked_gates": list(remaining.get("blocked_gates", [])),
        "entries": entries,
        "entries_sha256": entries_sha256,
        "artifacts_to_collect": artifacts_to_collect,
        "artifacts_to_collect_sha256": artifacts_to_collect_sha256,
        "success_criteria_handoff": success_criteria_handoff,
        "success_criteria_handoff_sha256": success_criteria_handoff_sha256,
        "gate_status_handoff": gate_status_handoff,
        "gate_status_handoff_sha256": gate_status_handoff_sha256,
        "compact_output_modes": compact_output_modes,
        "artifact_count": len(artifacts_to_collect),
        "entry_count": len(entries),
        "all_entries_have_success_criteria": all(
            bool(entry.get("success_criteria")) for entry in entries
        ),
        "all_entries_have_recommended_commands": all(
            bool(entry.get("recommended_command_sha256")) for entry in entries
        ),
        "no_claim_policy": no_claim_policy,
        "no_claim_policy_sha256": no_claim_policy_sha256,
    }


def verify_final_blocker_manifest(
    manifest_path: Path,
    *,
    current_manifest: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted final-blocker manifest with the current one."""

    persisted = json.loads(manifest_path.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_manifest)
    failures: list[dict[str, object]] = []
    if persisted != current_manifest:
        failures.append(
            {
                "name": "final_blocker_manifest_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": "Persisted final-blocker manifest differs from current prompt/oracle/resource/docs inputs.",
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_manifest_sha256": persisted_sha256,
        "current_manifest_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failure_count": len(failures),
        "current_status_provenance": current_manifest.get("status_provenance"),
        "persisted_status_provenance": persisted.get("status_provenance")
        if isinstance(persisted, dict)
        else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    status = status_mod.build_status(
        args.prompt_artifact,
        args.oracle_artifact,
        args.docs,
        resource_artifact=args.resource_artifact,
    )
    manifest = build_final_blocker_manifest(status)
    if args.verify_manifest is not None:
        verification = verify_final_blocker_manifest(
            args.verify_manifest,
            current_manifest=manifest,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        else:
            payload = verification
        status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
        return (
            status_mod.READY_EXIT_CODE
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    if args.entries_sha_only:
        payload = manifest["entries_sha256"]
    elif args.entries_only:
        payload = manifest["entries"]
    elif args.artifacts_sha_only:
        payload = manifest["artifacts_to_collect_sha256"]
    elif args.artifacts_only:
        payload = manifest["artifacts_to_collect"]
    elif args.success_criteria_sha_only:
        payload = manifest["success_criteria_handoff_sha256"]
    elif args.success_criteria_only:
        payload = manifest["success_criteria_handoff"]
    elif args.no_claim_policy_sha_only:
        payload = manifest["no_claim_policy_sha256"]
    elif args.no_claim_policy_only:
        payload = manifest["no_claim_policy"]
    elif args.gate_status_sha_only:
        payload = manifest["gate_status_handoff_sha256"]
    elif args.gate_status_only:
        payload = manifest["gate_status_handoff"]
    elif args.status_provenance_sha_only:
        payload = manifest["status_provenance_sha256"]
    elif args.status_provenance_only:
        payload = manifest["status_provenance"]
    else:
        payload = status_mod._stable_json_sha256(manifest) if args.sha_only else manifest
    status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
