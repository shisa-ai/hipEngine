#!/usr/bin/env python3
"""Packet 1 attribution: per-stage ROCTX marker ranges from profiled bridge runs.

Reads the profile child JSONs + rocprofv3 marker/hip-api/kernel/copy traces for
the four cells (C8/K3, C2/K2, C5/K3, C3/K3), attributes wall seconds to named
stages inside the last (measured) arm-complete window of each arm, and emits a
compact attribution artifact. Raw trace CSVs stay outside the repo.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

RAW = Path("/tmp/he-bettermtp-raw/packet1")
CELLS = ("c8k3", "c2k2", "c5k3", "c3k3")
CELL_META = {
    "c8k3": {"concurrency": 8, "capacity": 8, "budget": 3, "route": "product"},
    "c2k2": {"concurrency": 2, "capacity": 2, "budget": 2, "route": "product"},
    "c5k3": {"concurrency": 5, "capacity": 8, "budget": 3, "route": "screening"},
    "c3k3": {"concurrency": 3, "capacity": 8, "budget": 3, "route": "screening"},
}


def _stage_name(function: str) -> str:
    """'specdec2_perf_specdec2_cycle_total_000003' -> 'specdec2_cycle_total'."""
    text = function.strip('"')
    if not text.startswith("specdec2_perf_"):
        return text
    body = text[len("specdec2_perf_"):]
    stem = body.rsplit("_", 1)[0] if body.rsplit("_", 1)[-1].isdigit() else body
    return stem


def _read_markers(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "name": row["Function"],
                    "start": int(row["Start_Timestamp"]),
                    "end": int(row["End_Timestamp"]),
                }
            )
    return rows


def _arm_windows(markers: list[dict]) -> dict[str, tuple[int, int]]:
    """Last arm_complete marker per arm prefix -> (start, end) ns window."""

    windows: dict[str, tuple[int, int]] = {}
    for marker in markers:
        stage = _stage_name(marker["name"])
        if not stage.endswith("_arm_complete"):
            continue
        arm = stage[: -len("_arm_complete")]
        prev = windows.get(arm)
        if prev is None or marker["start"] > prev[0]:
            windows[arm] = (marker["start"], marker["end"])
    return windows


def _stage_sums(markers: list[dict], window: tuple[int, int]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for marker in markers:
        stage = _stage_name(marker["name"])
        if stage.endswith("_arm_complete"):
            continue
        if window[0] <= marker["start"] and marker["end"] <= window[1]:
            sums[stage] += (marker["end"] - marker["start"]) / 1e9
    return dict(sorted(sums.items()))


def _hip_api_window_stats(trace_dir: Path, window: tuple[int, int]) -> dict:
    """HIP API call counts + copy byte/period totals inside the window."""

    files = list(trace_dir.glob("*_hip_api_trace.csv"))
    if not files:
        return {}
    calls: dict[str, int] = defaultdict(int)
    copy_bytes = 0
    copy_count = 0
    with files[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            start = int(row.get("Start_Timestamp") or 0)
            end = int(row.get("End_Timestamp") or 0)
            if not (window[0] <= start and end <= window[1]):
                continue
            name = row.get("Name") or row.get("Function") or "?"
            calls[name] += 1
            if "Memcpy" in name:
                copy_count += 1
                try:
                    copy_bytes += int(row.get("Size") or 0)
                except (TypeError, ValueError):
                    pass
    top = dict(
        sorted(calls.items(), key=lambda item: -item[1])[:12]
    )
    return {"hip_api_calls_top": top, "memcpy_count": copy_count, "memcpy_bytes": copy_bytes}


def _kernel_window_stats(trace_dir: Path, window: tuple[int, int]) -> dict:
    """Kernel launch count + busy sums inside the window, by kernel name."""

    files = list(trace_dir.glob("*_kernel_trace.csv"))
    if not files:
        return {}
    kernels: dict[str, dict] = defaultdict(lambda: {"count": 0, "busy_s": 0.0})
    with files[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            start = int(row.get("Start_Timestamp") or 0)
            end = int(row.get("End_Timestamp") or 0)
            if not (window[0] <= start and end <= window[1]):
                continue
            name = row.get("Kernel_Name") or row.get("Name") or "?"
            entry = kernels[name]
            entry["count"] += 1
            entry["busy_s"] += (end - start) / 1e9
    top = {
        name: {"count": stats["count"], "busy_s": round(stats["busy_s"], 6)}
        for name, stats in sorted(
            kernels.items(), key=lambda item: -item[1]["busy_s"]
        )[:16]
    }
    total_count = sum(stats["count"] for stats in kernels.values())
    total_busy = sum(stats["busy_s"] for stats in kernels.values())
    return {
        "kernel_count": total_count,
        "kernel_busy_s": round(total_busy, 6),
        "kernels_top_by_busy": top,
    }


def _cycle_window_kernel_stats(
    trace_dir: Path, markers: list[dict], window: tuple[int, int]
) -> dict:
    """Kernel stats inside the measured arm's cycle_total ranges."""

    files = list(trace_dir.glob("*_kernel_trace.csv"))
    if not files:
        return {}
    ranges = sorted(
        (marker["start"], marker["end"])
        for marker in markers
        if _stage_name(marker["name"]) == "specdec2_cycle_total"
        and window[0] <= marker["start"]
        and marker["end"] <= window[1]
    )
    if not ranges:
        return {}
    kernels: dict[str, dict] = defaultdict(lambda: {"count": 0, "busy_s": 0.0})
    with files[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            start = int(row.get("Start_Timestamp") or 0)
            end = int(row.get("End_Timestamp") or 0)
            if not any(lo <= start and end <= hi for lo, hi in ranges):
                continue
            name = row.get("Kernel_Name") or row.get("Name") or "?"
            entry = kernels[name]
            entry["count"] += 1
            entry["busy_s"] += (end - start) / 1e9
    top = {
        name: {"count": stats["count"], "busy_s": round(stats["busy_s"], 6)}
        for name, stats in sorted(
            kernels.items(), key=lambda item: -item[1]["busy_s"]
        )[:16]
    }
    total_count = sum(stats["count"] for stats in kernels.values())
    total_busy = sum(stats["busy_s"] for stats in kernels.values())
    return {
        "cycle_count": len(ranges),
        "cycle_span_s": round(sum(hi - lo for lo, hi in ranges) / 1e9, 6),
        "kernel_count": total_count,
        "kernel_busy_s": round(total_busy, 6),
        "kernels_top_by_busy": top,
    }


def analyze_cell(tag: str) -> dict:
    meta = CELL_META[tag]
    child = json.loads((RAW / f"profile-{tag}-child.json").read_text())
    trace_dir = RAW / f"trace-{tag}" / "epyc"
    markers = _read_markers(next(trace_dir.glob("*_marker_api_trace.csv")))
    windows = _arm_windows(markers)
    arms: dict[str, dict] = {}
    for arm, window in windows.items():
        child_arm = (child.get("cells") or [{}])[0].get("arms", {}).get(arm, {})
        entry = {
            "window_s": round((window[1] - window[0]) / 1e9, 6),
            "complete_wall_seconds": child_arm.get("complete_wall_seconds"),
            "decode_only_seconds": child_arm.get("decode_only_seconds"),
            "stages": _stage_sums(markers, window),
            "kernels": _kernel_window_stats(trace_dir, window),
            "hip_api": _hip_api_window_stats(trace_dir, window),
        }
        if arm == "specdec2":
            entry["cycle_kernels"] = _cycle_window_kernel_stats(
                trace_dir, markers, window
            )
        arms[arm] = entry
    base = json.loads((RAW / f"base-{tag}.json").read_text())
    base_arm_walls = {
        arm: data.get("complete_wall_seconds")
        for arm, data in (base.get("cells") or [{}])[0].get("arms", {}).items()
        if isinstance(data, dict)
    }
    return {
        "cell": tag,
        "meta": meta,
        "profiled_status": child.get("status"),
        "unprofiled_arm_walls": base_arm_walls,
        "arms": arms,
    }


def main() -> int:
    cells = [analyze_cell(tag) for tag in CELLS]
    artifact = {
        "kind": "mtp-packet1-attribution",
        "created": "2026-09-06",
        "hardware": "AMD Radeon Pro W7900 (gfx1100)",
        "host": "epyc",
        "model": "Qwen3.8-27B Q4_K_M GGUF (D24 outputs, greedy)",
        "protocol": (
            "ROCm staged-cycle MTP (verify_chain) vs no-MTP true AR arm, same "
            "prompt, 12 output tokens, runs=1, per-arm warmup; rocprofv3 "
            "kernel+marker+hip-runtime+copy+allocation traces on the final-leaf "
            "bridge child; stage walls are marker-range sums inside the last "
            "(measured) arm-complete window"
        ),
        "commands": [
            "bash campaign-artifacts/packet1/run_p1_baselines.sh",
            "bash campaign-artifacts/packet1/run_p1_profiles.sh",
            "bash campaign-artifacts/packet1/run_p1_stage3_retry.sh",
            ".venv/bin/python campaign-artifacts/packet1/analyze_p1_attribution.py",
        ],
        "raw_traces": "/tmp/he-bettermtp-raw/packet1 (outside repo)",
        "cells": cells,
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "benchmarks/results/2026-09-06-w7900-q4km-mtp-packet1-attribution.json"
    )
    out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"wrote {out}")
    for cell in cells:
        print(f"\n== {cell['cell']} {cell['meta']}")
        for arm, data in cell["arms"].items():
            stages = " ".join(
                f"{name}={value:.4f}" for name, value in data["stages"].items()
            )
            k = data["kernels"]
            print(f"  {arm}: window={data['window_s']:.4f}s kernels={k.get('kernel_count')} busy={k.get('kernel_busy_s'):.4f}s")
            print(f"    stages: {stages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
