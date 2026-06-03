#!/usr/bin/env python3
"""Validate a StepFun KV-backed decode kernel trace artifact.

This checker is intentionally mechanical and conservative: it accepts a retained
rocprofv3 CSV (or a compact JSON with kernel names) only when the trace contains
all kernel-symbol families required by the planned StepFun prompt-KV,
decode-KV, and gated decode-attention path. It does not launch kernels and does
not claim that StepFun e2e inference is ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod

TRACE_CHECK_SCHEMA_VERSION = 1
DEFAULT_EXPECTED_LAYER_COUNT = 45
PASSED_EXIT_CODE = 0
FAILED_EXIT_CODE = 2

REQUIRED_KERNEL_FAMILIES: tuple[dict[str, object], ...] = (
    {
        "name": "prompt_kv_write",
        "operation": "prompt_kv_write",
        "required_for": "kv_kernel_trace_artifact",
        "symbols": (
            "hipengine_qwen35_write_paged_kv_mixed_value_bf16_prompt_spans",
        ),
        "evidence": "prompt prefill KV write kernel launched for every StepFun layer",
    },
    {
        "name": "decode_kv_write",
        "operation": "decode_kv_write",
        "required_for": "kv_kernel_trace_artifact",
        "symbols": (
            "hipengine_qwen35_write_paged_kv_mixed_value_bf16_spans",
        ),
        "evidence": "one-token decode KV write kernel launched for every StepFun layer",
    },
    {
        "name": "decode_attention_context",
        "operation": "decode_attention",
        "required_for": "kv_kernel_trace_artifact",
        "symbols": (
            "hipengine_qwen35_paged_full_attn_decode_split_k_context_bf16_spans",
        ),
        "evidence": "split-K decode-attention context kernel launched for every StepFun layer",
    },
    {
        "name": "decode_attention_gate_reduce",
        "operation": "decode_attention",
        "required_for": "kv_kernel_trace_artifact",
        "symbols": (
            "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_f32",
        ),
        "evidence": "gated FP32 decode-attention reduce kernel launched for every StepFun layer",
    },
)

_KERNEL_FIELD_NAMES = {
    "kernel",
    "kernel name",
    "kernel_name",
    "kernelname",
    "kernel symbol",
    "kernel_symbol",
    "kernel function",
    "kernel_function",
    "name",
    "symbol",
}


def _write_text_atomic(output: Path, text: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, output)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _emit_json(payload: object, *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(payload, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    _write_text_atomic(output, text)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="rocprofv3 CSV or compact JSON trace artifact to validate.",
    )
    parser.add_argument(
        "--resource-artifact",
        type=Path,
        default=status_mod.DEFAULT_RESOURCE_ARTIFACT,
        help="StepFun text-resource dry-run artifact used for layer/operation expectations.",
    )
    parser.add_argument(
        "--expected-layer-count",
        type=int,
        default=None,
        help="Override expected StepFun layer count instead of deriving it from the resource artifact.",
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
        help="Emit only the compact trace_summary payload.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the full report or compact summary.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Emit only passed/failed status.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return exit code 2 when any required kernel family is missing or under-counted.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args(argv)


def _is_kernel_field_name(key: object) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _KERNEL_FIELD_NAMES or "kernel" in normalized


def _extract_kernel_names_from_json_value(value: object) -> list[str]:
    if isinstance(value, dict):
        if isinstance(value.get("kernel_names"), list):
            return [str(item) for item in value["kernel_names"] if item not in (None, "")]
        names: list[str] = []
        for key, item in value.items():
            if _is_kernel_field_name(key):
                if isinstance(item, list):
                    names.extend(str(entry) for entry in item if entry not in (None, ""))
                elif item not in (None, ""):
                    names.append(str(item))
            elif isinstance(item, (dict, list)):
                names.extend(_extract_kernel_names_from_json_value(item))
        return names
    if isinstance(value, list):
        names = []
        for item in value:
            names.extend(_extract_kernel_names_from_json_value(item))
        return names
    return []


def _load_json_kernel_names(path: Path) -> list[str]:
    return _extract_kernel_names_from_json_value(json.loads(path.read_text()))


def _load_csv_kernel_names(path: Path) -> list[str]:
    text = path.read_text()
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    rows = csv.DictReader(text.splitlines(), dialect=dialect)
    names: list[str] = []
    for row in rows:
        for key, value in row.items():
            if value in (None, ""):
                continue
            if _is_kernel_field_name(key):
                names.append(str(value))
                break
    return names


def load_kernel_names(path: Path) -> tuple[str, list[str]]:
    """Return trace format and extracted kernel names from JSON or CSV."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json", _load_json_kernel_names(path)
    if suffix in {".csv", ".txt"}:
        return "csv", _load_csv_kernel_names(path)
    try:
        return "json", _load_json_kernel_names(path)
    except json.JSONDecodeError:
        return "csv", _load_csv_kernel_names(path)


def _resource_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _expected_layer_count(resource: dict[str, object], override: int | None) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("--expected-layer-count must be positive")
        return int(override)
    run_plan = resource.get("kv_decode_run_plan", {})
    if isinstance(run_plan, dict):
        trace = run_plan.get("streaming_decode_launch_trace", {})
        if isinstance(trace, dict) and trace.get("layer_count"):
            return int(trace["layer_count"])
    text_plan = resource.get("text_decode_resource_plan", {})
    if isinstance(text_plan, dict):
        schedule = text_plan.get("kv_decode_launch_schedule", {})
        if isinstance(schedule, dict) and schedule.get("layer_count"):
            return int(schedule["layer_count"])
    return DEFAULT_EXPECTED_LAYER_COUNT


def _planned_operation_count(resource: dict[str, object]) -> int | None:
    run_plan = resource.get("kv_decode_run_plan", {})
    if isinstance(run_plan, dict) and run_plan.get("kv_decode_launch_operation_count"):
        return int(run_plan["kv_decode_launch_operation_count"])
    text_plan = resource.get("text_decode_resource_plan", {})
    if isinstance(text_plan, dict):
        schedule = text_plan.get("kv_decode_launch_schedule", {})
        if isinstance(schedule, dict) and schedule.get("operation_count"):
            return int(schedule["operation_count"])
    return None


def _family_record(
    family: dict[str, object],
    *,
    kernel_names: Sequence[str],
    expected_min_count: int,
) -> dict[str, object]:
    symbols = tuple(str(symbol) for symbol in family.get("symbols", ()))
    matches = [name for name in kernel_names if any(symbol in name for symbol in symbols)]
    unique_matches = sorted(set(matches))
    return {
        "name": family["name"],
        "operation": family["operation"],
        "required_for": family["required_for"],
        "symbols": list(symbols),
        "evidence": family["evidence"],
        "expected_min_count": expected_min_count,
        "observed_count": len(matches),
        "ready": len(matches) >= expected_min_count,
        "matched_kernel_names": unique_matches,
        "matched_kernel_names_sha256": status_mod._stable_json_sha256(unique_matches),
    }


def build_trace_check_report(
    trace_path: Path,
    *,
    resource_artifact: Path = status_mod.DEFAULT_RESOURCE_ARTIFACT,
    expected_layer_count: int | None = None,
) -> dict[str, object]:
    """Return a mechanical validation report for a StepFun KV kernel trace."""

    trace_format, kernel_names = load_kernel_names(trace_path)
    kernel_counts = Counter(kernel_names)
    resource = _resource_payload(resource_artifact)
    layer_count = _expected_layer_count(resource, expected_layer_count)
    family_records = [
        _family_record(
            family,
            kernel_names=kernel_names,
            expected_min_count=layer_count,
        )
        for family in REQUIRED_KERNEL_FAMILIES
    ]
    missing_families = [
        record["name"] for record in family_records if record.get("ready") is not True
    ]
    observed_unique_kernel_names = sorted(kernel_counts)
    trace_summary = {
        "schema_version": TRACE_CHECK_SCHEMA_VERSION,
        "status": "passed" if not missing_families else "failed",
        "ready": not missing_families,
        "trace_path": str(trace_path),
        "trace_format": trace_format,
        "kernel_record_count": len(kernel_names),
        "unique_kernel_count": len(observed_unique_kernel_names),
        "kernel_names_sha256": status_mod._stable_json_sha256(kernel_names),
        "expected_layer_count": layer_count,
        "planned_operation_count": _planned_operation_count(resource),
        "expected_min_kernel_launch_count": layer_count * len(REQUIRED_KERNEL_FAMILIES),
        "required_family_count": len(REQUIRED_KERNEL_FAMILIES),
        "missing_family_count": len(missing_families),
        "missing_families": missing_families,
        "required_family_names": [str(family["name"]) for family in REQUIRED_KERNEL_FAMILIES],
        "no_claim_policy": {
            "kv_kernel_trace_artifact_claim_allowed": not missing_families,
            "kv_backed_decode_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "KV trace validation only checks required kernel-family presence; "
                "token/logit correctness and benchmark gates remain separate."
            ),
        },
    }
    report = {
        "schema_version": TRACE_CHECK_SCHEMA_VERSION,
        "status": trace_summary["status"],
        "trace_summary": trace_summary,
        "trace_summary_sha256": status_mod._stable_json_sha256(trace_summary),
        "resource_artifact": str(resource_artifact),
        "resource_artifact_present": resource_artifact.exists(),
        "required_kernel_families": family_records,
        "required_kernel_families_sha256": status_mod._stable_json_sha256(family_records),
        "observed_unique_kernel_names": observed_unique_kernel_names,
        "observed_unique_kernel_names_sha256": status_mod._stable_json_sha256(
            observed_unique_kernel_names
        ),
        "kernel_counts": dict(sorted(kernel_counts.items())),
        "readiness_impact": {
            "kv_kernel_trace_artifact": not missing_families,
            "kv_backed_decode_ready": False,
            "e2e_inference_ready": False,
            "reason": (
                "This report can satisfy only the retained KV kernel trace artifact; "
                "the streaming decode loop and KV-backed next-token artifact must also pass."
            ),
        },
    }
    report["report_sha256"] = status_mod._stable_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_trace_check_report(
        args.trace,
        resource_artifact=args.resource_artifact,
        expected_layer_count=args.expected_layer_count,
    )
    payload: object
    if args.status_only:
        payload = report["status"]
    elif args.sha_only:
        payload = report["trace_summary_sha256"] if args.summary_only else report["report_sha256"]
    elif args.summary_only:
        payload = report["trace_summary"]
    else:
        payload = report
    _emit_json(payload, pretty=args.pretty, output=args.output)
    if args.fail_on_missing and report["status"] != "passed":
        return FAILED_EXIT_CODE
    return PASSED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
