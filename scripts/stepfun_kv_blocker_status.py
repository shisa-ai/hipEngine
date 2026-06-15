#!/usr/bin/env python3
"""Emit a compact StepFun KV-backed decode blocker-status artifact.

This generator is intentionally diagnostic-only: it records whether the current
final-blocker validator view still lacks the KV kernel-trace and KV-backed
next-token artifacts. Passing this script is not a KV, e2e, or performance
claim.
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
from scripts import stepfun_validator_status as validator_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-kv-backed-blocker-status.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"
DEFAULT_READINESS_GATE = "kv_backed_decode"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Canonical prompt artifact used by final-blocker validators.",
    )
    parser.add_argument(
        "--oracle-artifact",
        type=Path,
        default=status_mod.DEFAULT_ORACLE_ARTIFACT,
        help="Canonical llama.cpp oracle artifact used by the status builder.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=status_mod.DEFAULT_DOCS_PATH,
        help="docs/STEPFUN.md checklist source used by the status builder.",
    )
    parser.add_argument(
        "--resource-artifact",
        type=Path,
        default=status_mod.DEFAULT_RESOURCE_ARTIFACT,
        help="Text-resource artifact used by the KV trace validator.",
    )
    parser.add_argument(
        "--readiness-gate",
        default=DEFAULT_READINESS_GATE,
        help="Readiness gate to summarize; defaults to kv_backed_decode.",
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
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Emit only the artifact status string.",
    )
    parser.add_argument(
        "--blocked-count-only",
        action="store_true",
        help="Emit only the blocked record count.",
    )
    parser.add_argument(
        "--missing-paths-only",
        action="store_true",
        help="Emit only the list of missing artifact paths.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable JSON SHA-256 digest of the artifact payload.",
    )
    return parser.parse_args(argv)


def _build_status(
    *,
    prompt_artifact: Path,
    oracle_artifact: Path,
    docs: Path,
    resource_artifact: Path,
) -> dict[str, object]:
    return status_mod.build_status(
        prompt_artifact,
        oracle_artifact,
        docs,
        resource_artifact=resource_artifact,
    )


def _build_validator_report(
    status: dict[str, object],
    *,
    prompt_artifact: Path,
    resource_artifact: Path,
    readiness_gate: str,
) -> dict[str, object]:
    manifest = manifest_mod.build_final_blocker_manifest(status)
    return validator_mod.build_validator_status_report(
        manifest,
        prompt_artifact=prompt_artifact,
        resource_artifact=resource_artifact,
        selected_blocked_gate_name=readiness_gate,
    )


def _blocked_records_for_gate(
    report: dict[str, object],
    readiness_gate: str,
) -> list[dict[str, object]]:
    results = report.get("validator_results", [])
    if not isinstance(results, list):
        return []
    return [
        record
        for record in results
        if isinstance(record, dict)
        and record.get("readiness_gate") == readiness_gate
        and record.get("status") in {"missing", "failed"}
    ]


def _load_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _dict_or_empty(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _streaming_runner_source_status(resource_artifact: Path) -> dict[str, object]:
    resource = _load_json_object(resource_artifact)
    run_plan = _dict_or_empty(resource.get("kv_decode_run_plan"))
    loop_status = _dict_or_empty(run_plan.get("streaming_decode_loop_status"))
    blueprint = _dict_or_empty(run_plan.get("streaming_decode_loop_blueprint"))
    blocker_summary = _dict_or_empty(run_plan.get("kv_decode_blocker_summary"))
    artifacts_needed = blocker_summary.get("artifacts_needed", [])
    if not isinstance(artifacts_needed, list):
        artifacts_needed = []
    return {
        "schema_version": 1,
        "resource_artifact": str(resource_artifact),
        "kv_decode_run_plan_present": bool(run_plan),
        "source": loop_status.get("source") or blueprint.get("source"),
        "ready": loop_status.get("ready"),
        "executable": loop_status.get("executable") or blueprint.get("executable"),
        "blocked_by": loop_status.get("blocked_by") or blueprint.get("blocked_by"),
        "blocked_by_sha256": loop_status.get("blocked_by_sha256")
        or blueprint.get("blocked_by_sha256"),
        "next_action": loop_status.get("next_action"),
        "blocker_count": loop_status.get("blocker_count"),
        "blocker_names": loop_status.get("blocker_names"),
        "blueprint_operation_count": loop_status.get("blueprint_operation_count")
        or blueprint.get("operation_count"),
        "blueprint_stage_count": loop_status.get("blueprint_stage_count"),
        "blueprint_sha256": loop_status.get("blueprint_sha256"),
        "required_artifact_names": [
            item.get("name") for item in artifacts_needed if isinstance(item, dict)
        ],
        "kernel_trace_blocker_name": blocker_summary.get("kernel_trace_blocker_name"),
        "last_blocker_name": blocker_summary.get("last_blocker_name"),
        "no_kernel_launches": _dict_or_empty(
            run_plan.get("streaming_decode_launch_trace")
        ).get("no_kernel_launches"),
    }


def build_kv_blocker_status(
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    oracle_artifact: Path = status_mod.DEFAULT_ORACLE_ARTIFACT,
    docs: Path = status_mod.DEFAULT_DOCS_PATH,
    resource_artifact: Path = status_mod.DEFAULT_RESOURCE_ARTIFACT,
    readiness_gate: str = DEFAULT_READINESS_GATE,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the compact blocker status for the selected KV readiness gate."""

    status = _build_status(
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
        docs=docs,
        resource_artifact=resource_artifact,
    )
    report = _build_validator_report(
        status,
        prompt_artifact=prompt_artifact,
        resource_artifact=resource_artifact,
        readiness_gate=readiness_gate,
    )
    blocked = _blocked_records_for_gate(report, readiness_gate)
    missing_paths = [
        record.get("validator_artifact_path")
        for record in blocked
        if record.get("reason") == "artifact_file_missing"
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_kv_backed_decode_blocker_status",
        "date": artifact_date,
        "status": "blocked" if blocked else "ready",
        "readiness_gate": readiness_gate,
        "source_validator_status_sha256": report["validator_status_summary_sha256"],
        "source_correctness_status_sha256": status_mod._stable_json_sha256(status),
        "kv_decode_dispatch_ready": status.get("kv_decode_dispatch_ready"),
        "kv_backed_decode_ready": status.get("kv_backed_decode_ready"),
        "e2e_inference_ready": status.get("e2e_inference_ready"),
        "blocked_count": len(blocked),
        "blocked_artifact_names": [record.get("artifact_name") for record in blocked],
        "missing_artifact_paths": missing_paths,
        "validator_command_kinds": [
            record.get("validator_command_kind") for record in blocked
        ],
        "producer_command_kinds": sorted(
            {
                record.get("producer_command_kind")
                for record in blocked
                if record.get("producer_command_kind")
            }
        ),
        "streaming_runner_source_status": _streaming_runner_source_status(resource_artifact),
        "blocked_records": blocked,
        "no_claim_policy": {
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This artifact records missing KV evidence only; it does not "
                "contain a kernel trace or a KV-backed next-token result."
            ),
        },
        "next_required_evidence": [
            (
                "produce benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json "
                "and pass scripts/stepfun_kv_trace_check.py"
            ),
            (
                "produce benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json "
                "and pass scripts/stepfun_kv_next_token_check.py"
            ),
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    artifact = build_kv_blocker_status(
        prompt_artifact=args.prompt_artifact,
        oracle_artifact=args.oracle_artifact,
        docs=args.docs,
        resource_artifact=args.resource_artifact,
        readiness_gate=args.readiness_gate,
        artifact_date=args.artifact_date,
    )
    if args.status_only:
        payload: object = artifact["status"]
    elif args.blocked_count_only:
        payload = artifact["blocked_count"]
    elif args.missing_paths_only:
        payload = artifact["missing_artifact_paths"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(artifact)
    else:
        payload = artifact
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
