#!/usr/bin/env python3
"""Summarize a Qwen3.8 prefill row sweep from raw timings and rocprofv3 CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

FAMILIES = ("q4", "q5", "q6", "gdn", "other")


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    for family in ("q4", "q5", "q6"):
        if family in lowered:
            return family
    if name.startswith("Cijk_") or "fp16_to_bf16_strided_rows" in lowered:
        return "q5"
    if any(marker in lowered for marker in ("gdn", "conv1d", "linear_attn")):
        return "gdn"
    return "other"


def _kernel_csv(pattern: str) -> Path:
    matches = tuple(Path(item) for item in glob.glob(pattern, recursive=True))
    if len(matches) != 1:
        raise ValueError(f"--kernel-csv must resolve to exactly one file, found {len(matches)}")
    return matches[0]


def _load_dispatches(path: Path) -> list[dict[str, Any]]:
    dispatches: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = int(row["Start_Timestamp"])
            stop = int(row["End_Timestamp"])
            if stop < start:
                raise ValueError("kernel trace contains a negative dispatch duration")
            dispatches.append(
                {
                    "name": row["Kernel_Name"],
                    "family": classify_kernel(row["Kernel_Name"]),
                    "start_ns": start,
                    "stop_ns": stop,
                    "grid_y": int(row.get("Grid_Size_Y") or 1),
                }
            )
    return dispatches


def _family_summary(
    family: str,
    dispatches: Sequence[dict[str, Any]],
    *,
    ledger: dict[str, Any] | None,
) -> dict[str, Any]:
    durations_ns = [item["stop_ns"] - item["start_ns"] for item in dispatches]
    duration_ns = sum(durations_ns)
    result: dict[str, Any] = {
        "dispatches": len(dispatches),
        "gpu_ms": duration_ns / 1e6,
        "gpu_share": None,
        "grid_size_y_min": min((item["grid_y"] for item in dispatches), default=None),
        "grid_size_y_median": (
            statistics.median(item["grid_y"] for item in dispatches) if dispatches else None
        ),
        "grid_size_y_max": max((item["grid_y"] for item in dispatches), default=None),
    }
    if family not in {"q4", "q5", "q6"} or not dispatches:
        result.update(
            {
                "active_weight_bytes": None,
                "sweep_multiplicity": None,
                "swept_weight_bytes": None,
                "per_tile_gb_s": None,
                "effective_tf_s": None,
            }
        )
        return result
    if ledger is None:
        raise ValueError(f"raw capture is missing the {family} launch weight ledger")
    entries = list(ledger.get("entries", ()))
    if family == "q5":
        dequant_dispatches = [
            item for item in dispatches if "dequantize_f16" in item["name"].lower()
        ]
        if dequant_dispatches:
            if not entries or len(dequant_dispatches) % len(entries):
                raise ValueError(
                    "q5 dequant dispatches must be a uniform integer multiple of launch groups"
                )
            factors = [len(dequant_dispatches) // len(entries)] * len(entries)
        else:
            weight_dispatches = [
                item for item in dispatches if "wmma_prefill" in item["name"].lower()
            ]
            factors = [item["grid_y"] for item in weight_dispatches]
    else:
        weight_dispatches = [
            item
            for item in dispatches
            if "embedding" not in item["name"].lower()
            and ("prefill" in item["name"].lower() or "gemv" in item["name"].lower())
        ]
        factors = [item["grid_y"] for item in weight_dispatches]
    if len(entries) != len(factors):
        raise ValueError(
            f"{family} launch ledger/weight-dispatch mismatch: {len(entries)} != {len(factors)}"
        )
    active_bytes = sum(int(entry["active_weight_bytes"]) for entry in entries)
    swept_bytes = sum(
        int(entry["active_weight_bytes"]) * int(factor)
        for entry, factor in zip(entries, factors, strict=True)
    )
    logical_row_elements = int(ledger["logical_row_elements"])
    multiplicity = swept_bytes / active_bytes if active_bytes else 0.0
    result.update(
        {
            "active_weight_bytes": active_bytes,
            "sweep_multiplicity": multiplicity,
            "sweep_basis": "byte-weighted rocprof Grid_Size_Y for one matched weight-bearing dispatch per launch group; Q5 rocBLAS dequant routes count one source sweep",
            "swept_weight_bytes": swept_bytes,
            "per_tile_gb_s": swept_bytes / duration_ns if duration_ns else None,
            "effective_tf_s": (
                (2 * logical_row_elements) / duration_ns / 1e3 if duration_ns else None
            ),
        }
    )
    return result


def analyze(raw: dict[str, Any], dispatches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    inventory = raw["model_family_inventory"]
    points: list[dict[str, Any]] = []
    for record in raw["records"]:
        start = int(record["start_monotonic_ns"])
        stop = int(record["stop_monotonic_ns"])
        selected = [
            item for item in dispatches if item["start_ns"] >= start and item["stop_ns"] <= stop
        ]
        by_family = {
            family: [item for item in selected if item["family"] == family]
            for family in FAMILIES
        }
        weight_ledger = record.get("weight_ledger", {})
        family_summaries = {
            family: _family_summary(
                family,
                items,
                ledger=weight_ledger.get(family),
            )
            for family, items in by_family.items()
        }
        kernel_ms = sum(value["gpu_ms"] for value in family_summaries.values())
        for value in family_summaries.values():
            value["gpu_share"] = value["gpu_ms"] / kernel_ms if kernel_ms else None
        q4 = family_summaries["q4"]
        multiplicity = float(q4["sweep_multiplicity"] or 1.0)
        q4_single_sweep_savings_ms = q4["gpu_ms"] * (1.0 - 1.0 / multiplicity)
        quant_single_sweep_savings = {
            family: family_summaries[family]["gpu_ms"]
            * (1.0 - 1.0 / float(family_summaries[family]["sweep_multiplicity"] or 1.0))
            for family in ("q4", "q5", "q6")
        }
        points.append(
            {
                "rows": int(record["rows"]),
                "tick_wall_ms": float(record["wall_ms"]),
                "tick_hip_event_gpu_span_ms": float(record["gpu_span_ms"]),
                "tick_wall_minus_gpu_ms": float(record["wall_minus_gpu_ms"]),
                "traced_kernel_ms": kernel_ms,
                "trace_coverage_of_hip_span": kernel_ms / float(record["gpu_span_ms"]),
                "families": family_summaries,
                "single_sweep_upper_bounds": {
                    "by_quant_family_savings_ms": quant_single_sweep_savings,
                    "all_quant_savings_ms": sum(quant_single_sweep_savings.values()),
                    "all_quant_share_of_tick_wall": sum(quant_single_sweep_savings.values())
                    / float(record["wall_ms"]),
                },
                "y1_q4_single_sweep_upper_bound": {
                    "savings_ms": q4_single_sweep_savings_ms,
                    "share_of_tick_wall": q4_single_sweep_savings_ms / float(record["wall_ms"]),
                    "projected_tick_wall_ms": float(record["wall_ms"])
                    - q4_single_sweep_savings_ms,
                    "assumption": "all Q4 family time above one measured sweep is removable",
                },
            }
        )
    return {
        "schema": 1,
        "kind": "gfx1151-qwen38-prefill-y0-sweep-multiplicity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "measurement_only_no_performance_claim",
        "source": {
            "raw_kind": raw["kind"],
            "model": raw["model"],
            "model_sha256": raw["model_sha256"],
            "prompts": raw["prompts"],
            "host": raw["host"],
            "backend": raw["backend"],
            "token_source": raw["token_source"],
        },
        "definitions": {
            "sweep_multiplicity": "byte-weighted M-tile passes across active family weights",
            "swept_weight_bytes": "sum of active source-weight bytes times the matched weight-bearing dispatch's M-grid sweep count",
            "per_tile_gb_s": "decimal GB/s; swept bytes divided by traced family duration",
            "effective_tf_s": "2*logical family elements*rows divided by traced family duration",
            "limitation": "swept bytes are derived from active GGUF source-weight bytes and matched launch M-grid geometry; rocprofv3 does not report physical DRAM bytes",
        },
        "model_family_inventory": inventory,
        "points": points,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--kernel-csv", required=True, help="Path or glob for one kernel trace CSV")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    payload = analyze(raw, _load_dispatches(_kernel_csv(args.kernel_csv)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
