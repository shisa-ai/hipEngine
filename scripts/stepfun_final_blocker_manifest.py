#!/usr/bin/env python3
"""Emit a compact manifest for the remaining StepFun GGUF correctness blockers."""

from __future__ import annotations

import argparse
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
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the manifest.",
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

    return {
        "schema_version": 1,
        "status": remaining.get("status", "blocked"),
        "open_or_partial_items_p0_p12": remaining.get(
            "open_or_partial_items_p0_p12"
        ),
        "remaining_blocker_count": remaining.get("remaining_blocker_count"),
        "remaining_blocker_kinds": list(remaining.get("remaining_blocker_kinds", [])),
        "blocked_gates": list(remaining.get("blocked_gates", [])),
        "entries": entries,
        "artifacts_to_collect": artifacts_to_collect,
        "artifact_count": len(artifacts_to_collect),
        "entry_count": len(entries),
        "all_entries_have_success_criteria": all(
            bool(entry.get("success_criteria")) for entry in entries
        ),
        "all_entries_have_recommended_commands": all(
            bool(entry.get("recommended_command_sha256")) for entry in entries
        ),
        "no_claim_policy": dict(remaining.get("no_claim_policy", {})),
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
    payload: object = status_mod._stable_json_sha256(manifest) if args.sha_only else manifest
    status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
