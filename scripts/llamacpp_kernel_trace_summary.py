#!/usr/bin/env python3
"""Summarize llama.cpp HIP rocprofv3 kernel traces into parity buckets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "hipengine.llamacpp_kernel_trace_summary.v1"


@dataclass(frozen=True)
class KernelRow:
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    vgpr: int | None
    scratch: int | None
    lds: int | None
    workgroup_size: tuple[int | None, int | None, int | None]
    grid_size: tuple[int | None, int | None, int | None]


@dataclass(frozen=True)
class MarkerRange:
    name: str
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def read_kernel_trace(path: Path) -> list[KernelRow]:
    rows: list[KernelRow] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            name = (row.get("Kernel_Name") or row.get("KernelName") or row.get("Name") or "").strip()
            if not name:
                continue
            rows.append(
                KernelRow(
                    name=name,
                    start_ns=start,
                    end_ns=end,
                    duration_ns=end - start,
                    vgpr=_int_or_none(row.get("VGPR_Count")),
                    scratch=_int_or_none(row.get("Scratch_Size")),
                    lds=_int_or_none(row.get("LDS_Block_Size")),
                    workgroup_size=(
                        _int_or_none(row.get("Workgroup_Size_X")),
                        _int_or_none(row.get("Workgroup_Size_Y")),
                        _int_or_none(row.get("Workgroup_Size_Z")),
                    ),
                    grid_size=(
                        _int_or_none(row.get("Grid_Size_X")),
                        _int_or_none(row.get("Grid_Size_Y")),
                        _int_or_none(row.get("Grid_Size_Z")),
                    ),
                )
            )
    return rows


def read_marker_trace(path: Path) -> list[MarkerRange]:
    rows: list[MarkerRange] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            name = (row.get("Function") or row.get("Name") or "").strip()
            if not name:
                continue
            rows.append(MarkerRange(name=name, start_ns=start, end_ns=end))
    return rows


def classify_kernel(name: str) -> str:
    lower = name.lower()
    mmvq_quant_types = {
        "(ggml_type)8,": "q8_0",
        "(ggml_type)12,": "q4_k",
        "(ggml_type)13,": "q5_k",
        "(ggml_type)14,": "q6_k",
    }
    if "mul_mat_vec_q_moe" in lower:
        quant = next((quant for signature, quant in mmvq_quant_types.items() if signature in lower), None)
        return f"llama_mmvq_moe_{quant}" if quant is not None else "llama_mmvq_moe"
    if "mul_mat_vec_q" in lower:
        quant = next((quant for signature, quant in mmvq_quant_types.items() if signature in lower), None)
        return f"llama_mmvq_{quant}" if quant is not None else "llama_mmvq"
    if "mul_mat_vec_f" in lower or "mul_mat_f" in lower:
        return "llama_mmvf"
    if "mul_mat_q<" in lower:
        quant_type_buckets = {
            "(ggml_type)12,": "llama_mmq_q4_k",
            "(ggml_type)13,": "llama_mmq_q5_k",
            "(ggml_type)8,": "llama_mmq_q8_0",
            "(ggml_type)14,": "llama_mmq_q6_k",
        }
        return next(
            (bucket for signature, bucket in quant_type_buckets.items() if signature in lower),
            "llama_mmq_other",
        )
    if "quantize_q8_1" in lower or "quantize_mmq_q8_1" in lower:
        return "llama_quantize_q8_1"
    if "mm_ids_helper" in lower:
        return "llama_moe_scheduler"
    if "ssm_conv" in lower:
        return "llama_linear_attn_conv"
    if lower.startswith("cijk_"):
        return "llama_rocblas_gemm"
    if "k_argsort" in lower or "top_k" in lower or "argsort" in lower:
        return "llama_topk_argsort"
    if "gated_delta_net" in lower:
        return "llama_gdn"
    if "flash_attn" in lower:
        return "llama_flash_attn"
    if "rope" in lower or "rope_" in lower:
        return "llama_rope"
    if "rms_norm" in lower or "l2_norm" in lower or "norm_f32" in lower:
        return "llama_norm"
    if (
        "__amd_rocclr_copy" in lower
        or "__amd_rocclr_fill" in lower
        or "cpy_" in lower
        or "copy" in lower
        or "get_rows" in lower
        or "set_rows" in lower
        or "concat" in lower
        or "cont_" in lower
    ):
        return "llama_copy_layout"
    if "k_bin_bcast" in lower or "scale_f32" in lower or "op_add" in lower or "op_mul" in lower:
        return "llama_elementwise"
    if "soft_max" in lower or "softmax" in lower:
        return "llama_softmax"
    return "other"


def _kernel_stats(rows: list[KernelRow], *, total_ns: int, include_details: bool = True) -> dict[str, Any]:
    duration_ns = sum(row.duration_ns for row in rows)
    stats = {
        "total_ms": duration_ns / 1.0e6,
        "dispatches": len(rows),
        "avg_dispatch_ms": duration_ns / 1.0e6 / len(rows) if rows else 0.0,
        "share_of_total": duration_ns / total_ns if total_ns else 0.0,
    }
    if include_details:
        stats.update(
            {
                "kernel_names": sorted({row.name for row in rows}),
                "vgpr_values": sorted({row.vgpr for row in rows if row.vgpr is not None}),
                "scratch_values": sorted({row.scratch for row in rows if row.scratch is not None}),
                "lds_values": sorted({row.lds for row in rows if row.lds is not None}),
            }
        )
    return stats


def _range_kernel_summary(marker: MarkerRange, rows: list[KernelRow], *, top: int) -> dict[str, Any]:
    total_ns = sum(row.duration_ns for row in rows)
    by_bucket: dict[str, list[KernelRow]] = defaultdict(list)
    by_name: dict[str, list[KernelRow]] = defaultdict(list)
    for row in rows:
        by_bucket[classify_kernel(row.name)].append(row)
        by_name[row.name].append(row)

    buckets = [
        {"bucket": bucket, **_kernel_stats(bucket_rows, total_ns=total_ns, include_details=False)}
        for bucket, bucket_rows in sorted(
            by_bucket.items(),
            key=lambda item: sum(row.duration_ns for row in item[1]),
            reverse=True,
        )
    ]
    kernels = [
        {"kernel": name, **_kernel_stats(name_rows, total_ns=total_ns, include_details=False)}
        for name, name_rows in sorted(
            by_name.items(),
            key=lambda item: sum(row.duration_ns for row in item[1]),
            reverse=True,
        )
    ][:top]

    return {
        "range": marker.name,
        "range_start_ns": marker.start_ns,
        "range_end_ns": marker.end_ns,
        "range_duration_ms": marker.duration_ns / 1.0e6,
        "kernel_ms": total_ns / 1.0e6,
        "kernel_dispatches": len(rows),
        "kernel_share_of_range": total_ns / marker.duration_ns if marker.duration_ns else 0.0,
        "buckets": buckets,
        "top_kernels": kernels,
    }


def _range_summaries(markers: list[MarkerRange], rows: list[KernelRow], *, top: int) -> list[dict[str, Any]]:
    summaries = []
    for marker in markers:
        marker_rows = [
            row
            for row in rows
            if marker.start_ns <= ((row.start_ns + row.end_ns) // 2) <= marker.end_ns
        ]
        summaries.append(_range_kernel_summary(marker, marker_rows, top=top))
    return sorted(summaries, key=lambda item: item["kernel_ms"], reverse=True)


def _range_name_summaries(markers: list[MarkerRange], rows: list[KernelRow], *, top: int) -> list[dict[str, Any]]:
    by_name: dict[str, list[MarkerRange]] = defaultdict(list)
    for marker in markers:
        by_name[marker.name].append(marker)

    summaries = []
    for name, name_markers in by_name.items():
        marker_rows: list[KernelRow] = []
        range_ns = 0
        for marker in name_markers:
            range_ns += marker.duration_ns
            marker_rows.extend(
                row
                for row in rows
                if marker.start_ns <= ((row.start_ns + row.end_ns) // 2) <= marker.end_ns
            )

        total_ns = sum(row.duration_ns for row in marker_rows)
        by_bucket: dict[str, list[KernelRow]] = defaultdict(list)
        by_kernel: dict[str, list[KernelRow]] = defaultdict(list)
        for row in marker_rows:
            by_bucket[classify_kernel(row.name)].append(row)
            by_kernel[row.name].append(row)

        buckets = [
            {"bucket": bucket, **_kernel_stats(bucket_rows, total_ns=total_ns, include_details=False)}
            for bucket, bucket_rows in sorted(
                by_bucket.items(),
                key=lambda item: sum(row.duration_ns for row in item[1]),
                reverse=True,
            )
        ]
        kernels = [
            {"kernel": kernel, **_kernel_stats(kernel_rows, total_ns=total_ns, include_details=False)}
            for kernel, kernel_rows in sorted(
                by_kernel.items(),
                key=lambda item: sum(row.duration_ns for row in item[1]),
                reverse=True,
            )
        ][:top]

        summaries.append(
            {
                "range": name,
                "range_calls": len(name_markers),
                "range_duration_ms": range_ns / 1.0e6,
                "kernel_ms": total_ns / 1.0e6,
                "kernel_dispatches": len(marker_rows),
                "kernel_share_of_range": total_ns / range_ns if range_ns else 0.0,
                "buckets": buckets,
                "top_kernels": kernels,
            }
        )

    return sorted(summaries, key=lambda item: item["kernel_ms"], reverse=True)


def build_summary(
    csv_path: Path,
    *,
    label: str,
    command: str | None,
    top: int,
    marker_csv: Path | None = None,
) -> dict[str, Any]:
    rows = read_kernel_trace(csv_path)
    markers = read_marker_trace(marker_csv) if marker_csv is not None else []
    total_ns = sum(row.duration_ns for row in rows)

    by_bucket: dict[str, list[KernelRow]] = defaultdict(list)
    by_name: dict[str, list[KernelRow]] = defaultdict(list)
    for row in rows:
        by_bucket[classify_kernel(row.name)].append(row)
        by_name[row.name].append(row)

    buckets = [
        {"bucket": bucket, **_kernel_stats(bucket_rows, total_ns=total_ns)}
        for bucket, bucket_rows in sorted(
            by_bucket.items(),
            key=lambda item: sum(row.duration_ns for row in item[1]),
            reverse=True,
        )
    ]
    kernels = [
        {"kernel": name, **_kernel_stats(name_rows, total_ns=total_ns)}
        for name, name_rows in sorted(
            by_name.items(),
            key=lambda item: sum(row.duration_ns for row in item[1]),
            reverse=True,
        )
    ][:top]

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "performance_claim": False,
        "inputs": {
            "kernel_trace_csv": str(csv_path),
            "marker_trace_csv": str(marker_csv) if marker_csv is not None else None,
            "command": command,
        },
        "total_kernel_ms": total_ns / 1.0e6,
        "total_dispatches": len(rows),
        "buckets": buckets,
        "top_kernels": kernels,
        "range_name_summaries": _range_name_summaries(markers, rows, top=top) if markers else [],
        "range_summaries": _range_summaries(markers, rows, top=top) if markers else [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="rocprofv3 *_kernel_trace.csv file")
    parser.add_argument("--marker-csv", type=Path, default=None, help="Optional rocprofv3 *_marker_api_trace.csv file")
    parser.add_argument("--json", type=Path, required=True, help="Output summary JSON")
    parser.add_argument("--label", default="llamacpp-hip-trace")
    parser.add_argument("--command", default=None, help="Command that produced the trace")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    summary = build_summary(args.csv, label=args.label, command=args.command, top=args.top, marker_csv=args.marker_csv)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
