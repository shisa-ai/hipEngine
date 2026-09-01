#!/usr/bin/env python3
"""Summarize rocprofv3 kernel, HIP API, marker, and memory-copy traces."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


def opt_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def timestamps(row: dict[str, str]) -> tuple[int, int] | None:
    try:
        start = int(float(row["Start_Timestamp"]))
        end = int(float(row["End_Timestamp"]))
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if end >= start else None


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"{pattern}: expected one file, got {matches}")
    return matches[0]


def marker_windows(rows: Iterable[dict[str, str]], prefix: str) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    for row in rows:
        name = (
            row.get("Function")
            or row.get("Marker_Name")
            or row.get("Marker_Text")
            or row.get("Name")
            or ""
        ).strip()
        if not name.startswith(prefix):
            continue
        bounds = timestamps(row)
        if bounds is not None:
            windows.append((bounds[0], bounds[1], name))
    return sorted(windows)


def inside(bounds: tuple[int, int], windows: list[tuple[int, int, str]]) -> bool:
    return any(bounds[0] >= start and bounds[1] <= end for start, end, _name in windows)


def hipengine_family(name: str) -> str:
    lowered = name.lower()
    if any(part in lowered for part in ("copybuffer", "fillbuffer", "memcpy", "rocclr")):
        return "copy_fill_kernel"
    if "q4_k_selected" in lowered or "q4k_selected" in lowered:
        return "moe_gate_up_q4"
    if "q5_1" in lowered and any(part in lowered for part in ("selected", "wmma", "mmq")):
        return "moe_down_q5"
    if any(part in lowered for part in ("q8_0_raw_mmq", "q8_0_mmq", "gguf_k_pack8", "q8_1_mmq", "dense_q8")):
        return "dense_quant_q8"
    if any(part in lowered for part in ("rocblas", "cijk_", "gemm")):
        return "dense_gemm_library"
    if "dense_gemv" in lowered or "prefill_out_coltile" in lowered or "selected_prefill_out" in lowered:
        return "dense_other"
    if any(part in lowered for part in ("qwen4_exp_gdn", "gdn_prefill", "gdn_decode", "linear_attn_conv")):
        return "gdn"
    if any(part in lowered for part in ("qwen4_exp_qsa_flash", "paged_full_attn", "sparse_attention")):
        return "qsa_attention"
    if any(part in lowered for part in ("qsa_", "pool_norm", "index_", "topk")):
        return "qsa_index_selection"
    if "ple" in lowered or "conv1d" in lowered:
        return "ple"
    if any(part in lowered for part in ("grouped_rmsnorm", "gr_write", "gated_mean", "scaled_silu", "sigmoid_f32")):
        return "gr_norm_inject"
    if any(part in lowered for part in ("router", "moe_group", "scatter_gather", "weighted_lanes")):
        return "moe_routing"
    if any(part in lowered for part in ("shared_gate_combine", "silu_mul", "weighted_sum", "f32_to_bf16", "bf16_to_f32")):
        return "cast_combine"
    if any(part in lowered for part in ("rope", "rmsnorm", "norm", "embedding", "repeat_bf16")):
        return "elementwise_norm_rope"
    return "other"


def llama_family(name: str) -> str:
    lowered = name.lower()
    if any(part in lowered for part in ("copybuffer", "fillbuffer", "memcpy", "rocclr")):
        return "copy_fill_kernel"
    q4 = any(part in lowered for part in ("type12", "type 12", "(ggml_type)12"))
    q5 = any(part in lowered for part in ("type7", "type 7", "(ggml_type)7"))
    q8 = any(part in lowered for part in ("type8", "type 8", "(ggml_type)8"))
    if "mul_mat_q" in lowered and q4:
        return "moe_gate_up_q4"
    if "mul_mat_q" in lowered and q5:
        return "moe_down_q5"
    if "mul_mat_vec_q" in lowered and q4:
        return "moe_gate_up_q4"
    if "mul_mat_vec_q" in lowered and q5:
        return "moe_down_q5"
    if ("mul_mat_q" in lowered or "mul_mat_vec_q" in lowered) and q8:
        return "dense_quant_q8"
    if any(part in lowered for part in ("cijk_", "rocblas", "gemm")):
        return "dense_gemm_library"
    if "mul_mat_vec_f" in lowered:
        return "dense_other"
    if any(part in lowered for part in ("gated_delta_net", "ssm_conv")):
        return "gdn"
    if "flash_attn" in lowered or "fattn" in lowered:
        return "qsa_attention"
    if any(part in lowered for part in ("topk", "argsort", "mm_ids")):
        return "moe_routing"
    if any(part in lowered for part in ("quantize", "convert")):
        return "cast_combine"
    if any(part in lowered for part in ("rope", "rms_norm", "norm", "unary", "bin_bcast", "scale_f32", "repeat")):
        return "elementwise_norm_rope"
    return "other"


def api_family(function: str) -> str:
    lowered = function.lower()
    if "graphlaunch" in lowered:
        return "graph_launch"
    if any(part in lowered for part in ("launchkernel", "extlaunch", "modulelaunch", "cooperativelaunch")):
        return "direct_kernel_launch"
    if "memcpy" in lowered:
        return "memcpy_api"
    if "memset" in lowered:
        return "memset_api"
    if "synchronize" in lowered or "waitevent" in lowered:
        return "synchronization"
    if "event" in lowered:
        return "event"
    if "malloc" in lowered or "free" in lowered:
        return "allocation"
    return "other"


def _family_table(values: dict[str, list[int]], total_ns: int, count_name: str) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "ms": totals[0] / 1e6,
            count_name: totals[1],
            "share_pct": (100 * totals[0] / total_ns) if total_ns else 0,
        }
        for name, totals in sorted(values.items(), key=lambda item: -item[1][0])
    ]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.trace_dir
    kernel_path = one(root, "*_kernel_trace.csv")
    hip_api_path = one(root, "*_hip_api_trace.csv")
    marker_path = one(root, "*_marker_api_trace.csv")
    copy_path = one(root, "*_memory_copy_trace.csv")
    allocation_path = one(root, "*_memory_allocation_trace.csv")
    kernels = read_csv(kernel_path)
    apis = read_csv(hip_api_path)
    markers = read_csv(marker_path)
    copies = read_csv(copy_path)
    allocations = read_csv(allocation_path)
    windows = marker_windows(markers, args.marker_prefix) if args.marker_prefix else []
    family_fn: Callable[[str], str] = hipengine_family if args.engine == "hipengine" else llama_family

    if not windows:
        compute: list[tuple[int, int]] = []
        for row in kernels:
            bounds = timestamps(row)
            name = row.get("Kernel_Name") or row.get("KernelName") or row.get("Name") or ""
            if bounds is not None and family_fn(name) != "copy_fill_kernel":
                compute.append(bounds)
        if not compute:
            raise RuntimeError("no compute kernel window")
        windows = [(min(item[0] for item in compute), max(item[1] for item in compute), "derived_compute_extent")]

    selected_kernels: list[dict[str, Any]] = []
    for row in kernels:
        bounds = timestamps(row)
        if bounds is None or not inside(bounds, windows):
            continue
        name = row.get("Kernel_Name") or row.get("KernelName") or row.get("Name") or ""
        selected_kernels.append(
            {
                "name": name,
                "start": bounds[0],
                "end": bounds[1],
                "ns": bounds[1] - bounds[0],
                "corr": opt_int(row.get("Correlation_Id")),
                "family": family_fn(name),
            }
        )

    selected_apis: list[dict[str, Any]] = []
    for row in apis:
        bounds = timestamps(row)
        if bounds is None or not inside(bounds, windows):
            continue
        function = row.get("Function") or row.get("Name") or ""
        selected_apis.append(
            {
                "function": function,
                "start": bounds[0],
                "end": bounds[1],
                "ns": bounds[1] - bounds[0],
                "corr": opt_int(row.get("Correlation_Id")),
                "family": api_family(function),
            }
        )

    selected_copies: list[dict[str, Any]] = []
    for row in copies:
        bounds = timestamps(row)
        if bounds is not None and inside(bounds, windows):
            selected_copies.append({"row": row, "ns": bounds[1] - bounds[0]})

    selected_allocations: list[dict[str, Any]] = []
    parsed_allocations: list[dict[str, Any]] = []
    for row in allocations:
        bounds = timestamps(row)
        if bounds is None:
            continue
        parsed = {
            "row": row,
            "start": bounds[0],
            "end": bounds[1],
            "agent": str(row.get("Agent_Id") or "unknown"),
            "operation": str(row.get("Operation") or "unknown"),
            "bytes": opt_int(row.get("Allocation_Size")) or 0,
        }
        parsed_allocations.append(parsed)
        if inside(bounds, windows):
            selected_allocations.append(parsed)

    graph_starts = [
        bounds[0]
        for row in apis
        if api_family(str(row.get("Function") or row.get("Name") or ""))
        == "graph_launch"
        if (bounds := timestamps(row)) is not None
    ]
    first_graph_launch = min(graph_starts) if graph_starts else None
    def is_allocate(row: dict[str, Any]) -> bool:
        return row["operation"] == "MEMORY_ALLOCATION_ALLOCATE"

    selected_allocate_rows = [row for row in selected_allocations if is_allocate(row)]
    allocations_after_first_graph = [
        row
        for row in parsed_allocations
        if first_graph_launch is not None
        and row["start"] >= first_graph_launch
        and is_allocate(row)
    ]

    def allocated_bytes(rows: Iterable[dict[str, Any]]) -> int:
        return sum(int(row["bytes"]) for row in rows)

    kernel_families: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kernel_symbols: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in selected_kernels:
        kernel_families[row["family"]][0] += row["ns"]
        kernel_families[row["family"]][1] += 1
        kernel_symbols[row["name"]][0] += row["ns"]
        kernel_symbols[row["name"]][1] += 1

    api_families: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    api_functions: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in selected_apis:
        api_families[row["family"]][0] += row["ns"]
        api_families[row["family"]][1] += 1
        api_functions[row["function"]][0] += row["ns"]
        api_functions[row["function"]][1] += 1

    launch_correlations = {
        row["corr"]
        for row in selected_apis
        if row["family"] == "direct_kernel_launch" and row["corr"] is not None
    }
    kernel_correlations = {row["corr"] for row in selected_kernels if row["corr"] is not None}
    total_kernel_ns = sum(row["ns"] for row in selected_kernels)

    return {
        "engine": args.engine,
        "kernel_csv": str(kernel_path),
        "hip_api_csv": str(hip_api_path) if hip_api_path else None,
        "marker_csv": str(marker_path) if marker_path else None,
        "memory_copy_csv": str(copy_path) if copy_path else None,
        "windows": [
            {"name": name, "start_ns": start, "end_ns": end, "wall_ms": (end - start) / 1e6}
            for start, end, name in windows
        ],
        "kernel": {
            "rows": len(selected_kernels),
            "sum_ms": total_kernel_ns / 1e6,
            "span_ms": (
                (max(row["end"] for row in selected_kernels) - min(row["start"] for row in selected_kernels)) / 1e6
                if selected_kernels
                else 0
            ),
            "families": _family_table(kernel_families, total_kernel_ns, "rows"),
            "top_symbols": [
                {"name": name, "ms": totals[0] / 1e6, "rows": totals[1]}
                for name, totals in sorted(kernel_symbols.items(), key=lambda item: -item[1][0])[:50]
            ],
        },
        "hip_api": {
            "rows": len(selected_apis),
            "sum_ms": sum(row["ns"] for row in selected_apis) / 1e6,
            "families": _family_table(api_families, sum(row["ns"] for row in selected_apis), "calls"),
            "top_functions": [
                {"name": name, "ms": totals[0] / 1e6, "calls": totals[1]}
                for name, totals in sorted(api_functions.items(), key=lambda item: -item[1][0])[:30]
            ],
            "direct_launch_correlations": len(launch_correlations),
            "direct_launch_corr_without_kernel_row": len(launch_correlations - kernel_correlations),
            "kernel_corr_without_direct_launch": len(kernel_correlations - launch_correlations),
        },
        "memory_copy": {
            "rows": len(selected_copies),
            "sum_ms": sum(row["ns"] for row in selected_copies) / 1e6,
            "sample_keys": list(selected_copies[0]["row"].keys()) if selected_copies else [],
        },
        "memory_allocation": {
            "csv": str(allocation_path) if allocation_path else None,
            "events_in_window": len(selected_allocations),
            "rows_in_window": len(selected_allocate_rows),
            "allocated_bytes_in_window": allocated_bytes(selected_allocate_rows),
            "agents_in_window": dict(
                sorted(
                    (
                        agent,
                        sum(row["agent"] == agent for row in selected_allocate_rows),
                    )
                    for agent in {row["agent"] for row in selected_allocate_rows}
                )
            ),
            "first_graph_launch_ns": first_graph_launch,
            "rows_at_or_after_first_graph_launch": len(allocations_after_first_graph),
            "allocated_bytes_at_or_after_first_graph_launch": allocated_bytes(
                allocations_after_first_graph
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=("hipengine", "llama"), required=True)
    parser.add_argument("--marker-prefix", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "windows": output["windows"],
                "kernel": {key: output["kernel"][key] for key in ("rows", "sum_ms", "span_ms")},
                "hip_api": output["hip_api"],
                "memory_copy": output["memory_copy"],
                "memory_allocation": output["memory_allocation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
