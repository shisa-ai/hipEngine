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
    duration_ns: int
    vgpr: int | None
    scratch: int | None
    lds: int | None
    workgroup_size: tuple[int | None, int | None, int | None]
    grid_size: tuple[int | None, int | None, int | None]


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


def classify_kernel(name: str) -> str:
    lower = name.lower()
    if "mul_mat_vec_q_moe" in lower:
        return "llama_mmvq_moe"
    if "mul_mat_vec_q" in lower:
        return "llama_mmvq"
    if "mul_mat_vec_f" in lower or "mul_mat_f" in lower:
        return "llama_mmvf"
    if "quantize_q8_1" in lower:
        return "llama_quantize_q8_1"
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


def _kernel_stats(rows: list[KernelRow], *, total_ns: int) -> dict[str, Any]:
    duration_ns = sum(row.duration_ns for row in rows)
    names = sorted({row.name for row in rows})
    vgprs = sorted({row.vgpr for row in rows if row.vgpr is not None})
    scratches = sorted({row.scratch for row in rows if row.scratch is not None})
    lds_values = sorted({row.lds for row in rows if row.lds is not None})
    return {
        "total_ms": duration_ns / 1.0e6,
        "dispatches": len(rows),
        "avg_dispatch_ms": duration_ns / 1.0e6 / len(rows) if rows else 0.0,
        "share_of_total": duration_ns / total_ns if total_ns else 0.0,
        "kernel_names": names,
        "vgpr_values": vgprs,
        "scratch_values": scratches,
        "lds_values": lds_values,
    }


def build_summary(
    csv_path: Path,
    *,
    label: str,
    command: str | None,
    top: int,
) -> dict[str, Any]:
    rows = read_kernel_trace(csv_path)
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
            "command": command,
        },
        "total_kernel_ms": total_ns / 1.0e6,
        "total_dispatches": len(rows),
        "buckets": buckets,
        "top_kernels": kernels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="rocprofv3 *_kernel_trace.csv file")
    parser.add_argument("--json", type=Path, required=True, help="Output summary JSON")
    parser.add_argument("--label", default="llamacpp-hip-trace")
    parser.add_argument("--command", default=None, help="Command that produced the trace")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    summary = build_summary(args.csv, label=args.label, command=args.command, top=args.top)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
