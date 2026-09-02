#!/usr/bin/env python3
"""Summarize cached SPECDEC2 profile-child traces by physical stage and kernel family.

The input child is produced by ``specdec2_perf_bridge.py --profile-child``.  Its
ROCTX marker names identify the measured arm and nested cycle/proposal/target
windows.  This tool joins those names to rocprofv3 CSVs, reports summed kernel
time separately from distinct device-busy interval coverage, and deduplicates
per-group timing counters copied onto every request row.

This is attribution, not a throughput benchmark.  HIP API time overlaps device
work, target readback is a dependency wait rather than pure transfer time, and
accept/commit/readback fields are a decomposition of the accept window rather
than additional wall.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


_REQUIRED_PHASES = ("cycle_total", "proposal", "target_accept_commit_provider")
_GROUP_COUNTERS = (
    "specdec2_mtp2_cycles",
    "specdec2_mtp2_proposal_ms",
    "specdec2_mtp2_target_ms",
    "specdec2_mtp2_provider_update_ms",
    "specdec2_mtp2_accept_ms",
    "specdec2_mtp2_accept_enqueue_ms",
    "specdec2_mtp2_accept_upload_ms",
    "specdec2_mtp2_accept_tail_ms",
    "specdec2_mtp2_selected_commit_ms",
    "specdec2_mtp2_target_readback_ms",
    "specdec2_mtp2_candidate_readback_ms",
    "specdec2_mtp2_proposal_physical_rows",
    "specdec2_mtp2_target_physical_rows",
    "specdec2_mtp2_recoverable_failures",
    "specdec2_mtp2_candidate_d2h_after_target",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_file(root: Path, pattern: str, *, required: bool = True) -> Path | None:
    matches = sorted(root.rglob(pattern))
    if len(matches) > 1 or (required and len(matches) != 1):
        raise ValueError(f"expected {'one' if required else 'at most one'} {pattern} under {root}; found {matches}")
    return matches[0] if matches else None


def _read_rows(path: Path | None, *, name_fields: Sequence[str]) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                start = int(float(raw["Start_Timestamp"]))
                end = int(float(raw["End_Timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            if end < start:
                continue
            name = next((str(raw.get(field) or "").strip() for field in name_fields if raw.get(field)), "")
            rows.append(
                {
                    "name": name,
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "raw": raw,
                }
            )
    return rows


def _marker_windows(path: Path) -> dict[str, list[tuple[int, int]]]:
    rows = _read_rows(path, name_fields=("Function", "Name", "Message"))
    windows: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for row in rows:
        name = str(row["name"])
        if name.startswith("specdec2_perf_"):
            windows[name].append((int(row["start_ns"]), int(row["end_ns"])))
    return dict(windows)


def _select(rows: Sequence[Mapping[str, Any]], windows: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if any(
            int(row["start_ns"]) >= start and int(row["end_ns"]) <= end
            for start, end in windows
        )
    ]


def _interval_union_ns(rows: Sequence[Mapping[str, Any]], windows: Sequence[tuple[int, int]]) -> int:
    total = 0
    for window_start, window_end in windows:
        intervals = sorted(
            (
                max(window_start, int(row["start_ns"])),
                min(window_end, int(row["end_ns"])),
            )
            for row in rows
            if int(row["end_ns"]) >= window_start and int(row["start_ns"]) <= window_end
        )
        current_start: int | None = None
        current_end: int | None = None
        for start, end in intervals:
            if end < start:
                continue
            if current_start is None:
                current_start, current_end = start, end
            elif start <= int(current_end):
                current_end = max(int(current_end), end)
            else:
                total += int(current_end) - current_start
                current_start, current_end = start, end
        if current_start is not None:
            total += int(current_end) - current_start
    return total


def _short_kernel(name: str) -> str:
    value = re.sub(r"^void\s+", "", str(name).strip())
    value = value.replace("(anonymous namespace)::", "")
    return value.split("(", 1)[0].strip()


def classify_kernel(name: str) -> str:
    value = str(name).lower()
    if any(token in value for token in ("q4_k_t16", "gguf_q4_k", "gguf_q4_t16")):
        return "q4_t16"
    if any(
        token in value
        for token in (
            "q8_1_d4s4_f32_quantize_bf16_kernel",
            "q8_1_d4s4_f32_quantize_bf16_kmajor_kernel",
            "q8_1_ds4_quantize_bf16_kmajor_kernel",
        )
    ):
        return "q5_activation_quant"
    if (
        "q5_k_t16" in value
        or "gguf_q5_k" in value
        or ("gguf_k_raw_mmq32" in value and "kernel<5," in value)
    ):
        return "q5_t16"
    if "q6_k_t16" in value or "gguf_q6_k" in value:
        return "q6_t16"
    if "q8_0_t16" in value or "q8_0_dp4a" in value:
        return "q8_t16"
    if "accept_from_top1" in value:
        return "accept"
    if any(token in value for token in ("commit", "scatter", "gather_rows", "copy_rows")):
        return "commit_repair"
    if any(token in value for token in ("linear_attn", "gdn", "ssm", "_conv_")):
        return "gdn_linear_attention"
    if any(token in value for token in ("paged_full_attn", "paged_kv", "flash_attn", "softmax")):
        return "attention"
    if any(token in value for token in ("rmsnorm", "rotary")):
        return "norm_rope"
    if any(token in value for token in ("silu", "weighted_sum", "combine_residual")):
        return "combine_silu"
    if any(token in value for token in ("argmax", "top1")):
        return "sampler"
    if "embedding" in value:
        return "embedding"
    if any(token in value for token in ("copybuffer", "fillbuffer", "memcpy", "memset")):
        return "copy_fill"
    return "other"


def _kernel_table(rows: Sequence[Mapping[str, Any]], *, key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"calls": 0, "duration_ns": 0, "geometries": set()}
    )
    for row in rows:
        raw = row["raw"]
        name = classify_kernel(str(row["name"])) if key == "family" else _short_kernel(str(row["name"]))
        entry = grouped[name]
        entry["calls"] += 1
        entry["duration_ns"] += int(row["duration_ns"])
        if key == "kernel":
            geometry = tuple(
                int(float(raw.get(field) or 0))
                for field in (
                    "Grid_Size_X",
                    "Grid_Size_Y",
                    "Grid_Size_Z",
                    "Workgroup_Size_X",
                    "VGPR_Count",
                    "Scratch_Size",
                    "LDS_Block_Size",
                )
            )
            entry["geometries"].add(geometry)
    result: list[dict[str, Any]] = []
    for name, entry in sorted(grouped.items(), key=lambda item: (-item[1]["duration_ns"], item[0])):
        row = {
            "name": name,
            "calls": int(entry["calls"]),
            "total_ms": float(entry["duration_ns"]) / 1.0e6,
            "us_per_call": float(entry["duration_ns"]) / max(int(entry["calls"]), 1) / 1.0e3,
        }
        if key == "kernel":
            row["geometries"] = [list(value) for value in sorted(entry["geometries"])]
        result.append(row)
    return result


def _named_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[str(row["name"])][0] += 1
        grouped[str(row["name"])][1] += int(row["duration_ns"])
    return [
        {"name": name, "calls": values[0], "total_ms": values[1] / 1.0e6}
        for name, values in sorted(grouped.items(), key=lambda item: (-item[1][1], item[0]))
    ]


def summarize_phase(
    *,
    marker_names: Sequence[str],
    all_markers: Mapping[str, Sequence[tuple[int, int]]],
    kernels: Sequence[Mapping[str, Any]],
    hip_api: Sequence[Mapping[str, Any]],
    copies: Sequence[Mapping[str, Any]],
    allocations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    missing = [name for name in marker_names if name not in all_markers]
    if missing:
        raise ValueError(f"trace is missing child marker names: {missing}")
    # The bridge intentionally warms the same arm before measuring it, and its
    # per-arm marker counter restarts. The child marker list describes the final
    # measured arm, so the final trace occurrence of each marker name is the
    # corresponding measured window.
    windows = [all_markers[name][-1] for name in marker_names]
    selected_kernels = _select(kernels, windows)
    selected_api = _select(hip_api, windows)
    selected_copies = _select(copies, windows)
    selected_allocations = _select(allocations, windows)
    wall_ns = sum(end - start for start, end in windows)
    kernel_sum_ns = sum(int(row["duration_ns"]) for row in selected_kernels)
    kernel_union_ns = _interval_union_ns(selected_kernels, windows)
    allocation_bytes = sum(
        int(float(row["raw"].get("Allocation_Size") or 0))
        for row in selected_allocations
        if str(row["raw"].get("Operation") or "").endswith("ALLOCATE")
    )
    calls = max(len(windows), 1)
    kernel_families = _kernel_table(selected_kernels, key="family")
    top_kernels = _kernel_table(selected_kernels, key="kernel")[:32]
    for row in (*kernel_families, *top_kernels):
        row["calls_per_phase_call"] = float(row["calls"]) / calls
        row["ms_per_phase_call"] = float(row["total_ms"]) / calls
    hip_api_sum_ns = sum(int(row["duration_ns"]) for row in selected_api)
    copy_sum_ns = sum(int(row["duration_ns"]) for row in selected_copies)
    return {
        "calls": len(windows),
        "host_marker_wall_ms": wall_ns / 1.0e6,
        "host_marker_ms_per_call": wall_ns / calls / 1.0e6,
        "kernel_calls": len(selected_kernels),
        "kernel_calls_per_call": len(selected_kernels) / calls,
        "kernel_sum_ms": kernel_sum_ns / 1.0e6,
        "kernel_sum_ms_per_call": kernel_sum_ns / calls / 1.0e6,
        "kernel_interval_union_ms": kernel_union_ns / 1.0e6,
        "kernel_interval_union_ms_per_call": kernel_union_ns / calls / 1.0e6,
        "uncovered_marker_wall_ms": (wall_ns - kernel_union_ns) / 1.0e6,
        "uncovered_marker_wall_ms_per_call": (wall_ns - kernel_union_ns) / calls / 1.0e6,
        "hip_api_calls": len(selected_api),
        "hip_api_calls_per_call": len(selected_api) / calls,
        "hip_api_sum_ms": hip_api_sum_ns / 1.0e6,
        "hip_api_sum_ms_per_call": hip_api_sum_ns / calls / 1.0e6,
        "memory_copy_ops": len(selected_copies),
        "memory_copy_ops_per_call": len(selected_copies) / calls,
        "memory_copy_sum_ms": copy_sum_ns / 1.0e6,
        "memory_copy_sum_ms_per_call": copy_sum_ns / calls / 1.0e6,
        "allocation_ops": len(selected_allocations),
        "allocation_ops_per_call": len(selected_allocations) / calls,
        "allocated_bytes": allocation_bytes,
        "kernel_families": kernel_families,
        "top_kernels": top_kernels,
        "top_hip_api": _named_table(selected_api)[:24],
        "memory_copy_directions": _named_table(
            [dict(row, name=str(row["raw"].get("Direction") or row["name"])) for row in selected_copies]
        ),
    }


def extract_group_telemetry(child: Mapping[str, Any], *, expected_concurrency: int) -> dict[str, Any]:
    cells = child.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise ValueError("profile child must contain exactly one cell")
    cell = cells[0]
    if int(cell.get("concurrency", 0)) != int(expected_concurrency):
        raise ValueError("profile child concurrency does not match lane")
    arm = cell.get("arms", {}).get("specdec2", {})
    if arm.get("status") != "complete" or arm.get("realized_route") != "specdec2":
        raise ValueError("profile child did not realize SPECDEC2")
    routes = arm.get("recent_routes")
    if not isinstance(routes, list) or len(routes) != int(expected_concurrency):
        raise ValueError("profile child does not contain one route row per request")
    if not all(bool(row.get("specdec2_mtp2_used")) for row in routes):
        raise ValueError("not every request engaged physical MTP")
    first = routes[0]
    for key in _GROUP_COUNTERS:
        expected = first.get(key)
        if any(row.get(key) != expected for row in routes[1:]):
            raise ValueError(f"copied physical-group counter differs by request: {key}")
    cycles = int(first["specdec2_mtp2_cycles"])
    if cycles <= 0:
        raise ValueError("physical group has no cycles")
    stage_keys = {
        "proposal_ms": "specdec2_mtp2_proposal_ms",
        "target_submit_ms": "specdec2_mtp2_target_ms",
        "provider_update_ms": "specdec2_mtp2_provider_update_ms",
        "accept_window_ms": "specdec2_mtp2_accept_ms",
        "accept_enqueue_ms": "specdec2_mtp2_accept_enqueue_ms",
        "accept_upload_ms": "specdec2_mtp2_accept_upload_ms",
        "accept_tail_ms": "specdec2_mtp2_accept_tail_ms",
        "selected_commit_ms": "specdec2_mtp2_selected_commit_ms",
        "target_readback_ms": "specdec2_mtp2_target_readback_ms",
        "candidate_readback_ms": "specdec2_mtp2_candidate_readback_ms",
    }
    per_cycle = {
        name: float(first[source]) / cycles for name, source in stage_keys.items()
    }
    per_cycle["nonoverlap_named_sum_ms"] = sum(
        per_cycle[name]
        for name in ("proposal_ms", "target_submit_ms", "provider_update_ms", "accept_window_ms")
    )
    ledger = arm.get("stage_ledger", {})
    return {
        "request_rows": len(routes),
        "physical_cycles": cycles,
        "service_capacity": int(child.get("workload", {}).get("service_capacity", 0)),
        "proposal_physical_rows": list(first["specdec2_mtp2_proposal_physical_rows"]),
        "target_physical_rows": list(first["specdec2_mtp2_target_physical_rows"]),
        "recoverable_failures": int(first["specdec2_mtp2_recoverable_failures"]),
        "candidate_d2h_after_target": int(first["specdec2_mtp2_candidate_d2h_after_target"]),
        "per_cycle_ms": per_cycle,
        "operation_ledger": {
            "totals_ms": {
                key: float(value) * 1000.0
                for key, value in ledger.get("totals_seconds", {}).items()
            },
            "call_counts": dict(ledger.get("call_counts", {})),
        },
        "generated_tokens": int(arm.get("generated_tokens", 0)),
        "complete_wall_seconds": float(arm.get("complete_wall_seconds", 0.0)),
    }


def summarize_lane(concurrency: int, child_path: Path, trace_dir: Path) -> dict[str, Any]:
    child = json.loads(child_path.read_text(encoding="utf-8"))
    if child.get("status") != "complete":
        raise ValueError(f"incomplete child: {child_path}")
    markers_path = _single_file(trace_dir, "*_marker_api_trace.csv")
    kernels_path = _single_file(trace_dir, "*_kernel_trace.csv")
    hip_api_path = _single_file(trace_dir, "*_hip_api_trace.csv", required=False)
    copies_path = _single_file(trace_dir, "*_memory_copy_trace.csv", required=False)
    allocations_path = _single_file(trace_dir, "*_memory_allocation_trace.csv", required=False)
    assert markers_path is not None and kernels_path is not None
    all_markers = _marker_windows(markers_path)
    kernels = _read_rows(kernels_path, name_fields=("Kernel_Name", "KernelName", "Name"))
    hip_api = _read_rows(hip_api_path, name_fields=("Function", "Name"))
    copies = _read_rows(copies_path, name_fields=("Direction", "Kind"))
    allocations = _read_rows(allocations_path, name_fields=("Operation", "Kind"))
    spec_arm = child["cells"][0]["arms"]["specdec2"]
    marker_names = spec_arm["stage_ledger"]["marker_names"]
    phases = {
        phase: summarize_phase(
            marker_names=marker_names[phase],
            all_markers=all_markers,
            kernels=kernels,
            hip_api=hip_api,
            copies=copies,
            allocations=allocations,
        )
        for phase in _REQUIRED_PHASES
    }
    provenance = child.get("canonical_provenance", {})
    return {
        "concurrency": int(concurrency),
        "group": extract_group_telemetry(child, expected_concurrency=int(concurrency)),
        "phases": phases,
        "source": {
            "child": str(child_path),
            "child_sha256": _sha256(child_path),
            "trace_dir": str(trace_dir),
            "marker_csv": str(markers_path),
            "marker_csv_sha256": _sha256(markers_path),
            "kernel_csv": str(kernels_path),
            "kernel_csv_sha256": _sha256(kernels_path),
            "hip_api_csv": None if hip_api_path is None else str(hip_api_path),
            "memory_copy_csv": None if copies_path is None else str(copies_path),
            "memory_allocation_csv": None if allocations_path is None else str(allocations_path),
            "child_command": provenance.get("command"),
            "hipengine_commit": provenance.get("hipengine_commit"),
            "tracked_clean": provenance.get("dirty") is False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        action="append",
        nargs=3,
        metavar=("C", "CHILD_JSON", "TRACE_DIR"),
        required=True,
        help="Add one physical concurrency lane; repeat for C6/C7/C8.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    lanes = [
        summarize_lane(int(c), Path(child), Path(trace))
        for c, child, trace in args.lane
    ]
    widths = [int(lane["concurrency"]) for lane in lanes]
    if len(widths) != len(set(widths)):
        raise ValueError("lane concurrency values must be unique")
    payload = {
        "schema": 1,
        "kind": "specdec2_rocprof_stage_summary",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "diagnostic",
        "performance_claim": False,
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "lanes": {str(lane["concurrency"]): lane for lane in lanes},
        "limitations": [
            "Single canonical prompt and profiler instrumentation: attribution only, never a retained throughput row.",
            "Summed kernel duration may exceed wall under stream overlap; interval-union time is the non-overlapping device-busy lower bound.",
            "HIP API durations overlap device work and are not additive savings.",
            "Accept commit/readback counters decompose the accept dependency window and are not added to it.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for lane in lanes:
        c = lane["concurrency"]
        group = lane["group"]
        cycle = lane["phases"]["cycle_total"]
        print(
            f"C{c}: cycles={group['physical_cycles']} "
            f"host={cycle['host_marker_ms_per_call']:.3f}ms "
            f"kernel={cycle['kernel_sum_ms_per_call']:.3f}ms "
            f"busy={cycle['kernel_interval_union_ms_per_call']:.3f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
