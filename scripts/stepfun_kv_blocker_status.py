#!/usr/bin/env python3
"""Emit a compact StepFun KV-backed decode blocker-status artifact.

This generator is intentionally diagnostic-only: it records whether the current
final-blocker validator view still lacks the KV kernel-trace and KV-backed
next-token artifacts. Passing this script is not a KV, e2e, or performance
claim.
"""

from __future__ import annotations

import argparse
import ast
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


def _runtime_wiring_map() -> dict[str, object]:
    """Return the current in-tree entrypoint map for wiring KV-backed decode."""

    runner_file = "hipengine/runtime/stepfun_gguf_runner.py"
    return {
        "schema_version": 1,
        "source": "static_runtime_symbol_map",
        "runner_file": runner_file,
        "planner_entrypoint": {
            "symbol": "StepFunShortContextDecodePlanner.plan_kv_decode_chat",
            "file": runner_file,
            "role": "Bind the rendered StepFun chat prompt to a StepFunKVDecodeRunPlan.",
        },
        "resource_plan_entrypoint": {
            "symbol": "StepFunTextDecodeResourcePlan.kv_decode_launch_schedule",
            "file": runner_file,
            "role": "Define the planned per-layer order: prompt_kv_write, decode_kv_write, decode_attention.",
        },
        "device_input_entrypoints": [
            {
                "symbol": "StepFunKVDecodeRunPlan.upload_decode_inputs",
                "file": runner_file,
                "role": "Allocate/copy input IDs and KVLiveSpans-compatible span metadata before launches.",
            },
            {
                "symbol": "StepFunKVDecodeRunPlan.decode_input_upload_plan",
                "file": runner_file,
                "role": "Metadata-only manifest for pre-run upload order and cleanup order.",
            },
        ],
        "metadata_only_trace_entrypoints": [
            {
                "symbol": "StepFunKVDecodeRunPlan.streaming_decode_loop_blueprint",
                "file": runner_file,
                "role": "Records upload and launch contract for the future streaming loop.",
            },
            {
                "symbol": "StepFunKVDecodeRunPlan.streaming_decode_launch_trace",
                "file": runner_file,
                "role": "Records the 45-layer × 3-operation launch trace without launching kernels.",
            },
            {
                "symbol": "StepFunKVDecodeRunPlan.streaming_decode_loop_status",
                "file": runner_file,
                "role": "Reports blocked_by=streaming_decode_loop_not_wired until an executable loop exists.",
            },
        ],
        "current_host_composed_prompt_smoke": {
            "symbol": "StepFunResidentSession.layer_prefix_prompt_logits_probe_bf16",
            "file": runner_file,
            "role": "Current all-layer prompt smoke path; it is host-composed and not KV-backed decode.",
        },
        "session_contract_entrypoint": {
            "symbol": "StepFunResidentSession.kv_streaming_decode_contract",
            "file": runner_file,
            "role": "Validate a StepFunKVDecodeRunPlan against the resident session and expose the future executable loop contract without launching kernels.",
        },
        "missing_execution_entrypoint": {
            "owner": "StepFunResidentSession",
            "expected_role": (
                "Launch resident prompt KV writes, one-token decode KV writes, and gated paged attention "
                "from the uploaded StepFunKVDecodeRunPlan inputs, then emit the KV-backed next-token artifact."
            ),
            "required_artifacts": [
                "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
                "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
            ],
        },
        "next_action": "wire_streaming_decode_loop",
        "no_claim_policy": {
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": "This is an entrypoint map, not an executable KV-backed decode run.",
        },
    }


def _ast_class_methods(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_methods: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {
            child.name
            for child in node.body
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        class_methods[node.name] = methods
    return class_methods


def _symbol_validation_for_runtime_wiring_map(
    wiring_map: dict[str, object],
) -> dict[str, object]:
    runner_file = str(wiring_map.get("runner_file") or "")
    runner_path = REPO_ROOT / runner_file
    class_methods = _ast_class_methods(runner_path)

    symbol_records: list[dict[str, object]] = []

    def add_symbol(symbol: object, *, role: str) -> None:
        symbol_text = str(symbol)
        class_name, sep, method_name = symbol_text.partition(".")
        class_present = class_name in class_methods
        method_present = bool(sep) and method_name in class_methods.get(class_name, set())
        symbol_records.append(
            {
                "symbol": symbol_text,
                "role": role,
                "class_name": class_name,
                "method_name": method_name if sep else None,
                "class_present": class_present,
                "method_present": method_present,
                "present": class_present and method_present,
            }
        )

    planner = _dict_or_empty(wiring_map.get("planner_entrypoint"))
    add_symbol(planner.get("symbol"), role="planner_entrypoint")
    resource_plan = _dict_or_empty(wiring_map.get("resource_plan_entrypoint"))
    add_symbol(resource_plan.get("symbol"), role="resource_plan_entrypoint")
    for entry in wiring_map.get("device_input_entrypoints", []):
        if isinstance(entry, dict):
            add_symbol(entry.get("symbol"), role="device_input_entrypoint")
    for entry in wiring_map.get("metadata_only_trace_entrypoints", []):
        if isinstance(entry, dict):
            add_symbol(entry.get("symbol"), role="metadata_only_trace_entrypoint")
    prompt_smoke = _dict_or_empty(wiring_map.get("current_host_composed_prompt_smoke"))
    add_symbol(prompt_smoke.get("symbol"), role="current_host_composed_prompt_smoke")
    session_contract = _dict_or_empty(wiring_map.get("session_contract_entrypoint"))
    add_symbol(session_contract.get("symbol"), role="session_contract_entrypoint")

    missing_execution = _dict_or_empty(wiring_map.get("missing_execution_entrypoint"))
    owner = str(missing_execution.get("owner") or "")
    owner_record = {
        "owner": owner,
        "class_present": owner in class_methods,
        "present": owner in class_methods,
    }
    missing_symbols = [
        record["symbol"] for record in symbol_records if record.get("present") is not True
    ]
    if owner_record["present"] is not True:
        missing_symbols.append(owner)
    return {
        "schema_version": 1,
        "source": "ast_symbol_validation",
        "file": runner_file,
        "all_symbols_present": not missing_symbols,
        "missing_symbols": missing_symbols,
        "symbol_count": len(symbol_records),
        "symbols": symbol_records,
        "missing_execution_owner": owner_record,
    }


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
    wiring_map = _runtime_wiring_map()
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
        "runtime_wiring_map": wiring_map,
        "runtime_wiring_symbol_validation": _symbol_validation_for_runtime_wiring_map(
            wiring_map
        ),
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
