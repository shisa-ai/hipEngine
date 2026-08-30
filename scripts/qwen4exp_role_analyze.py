#!/usr/bin/env python3
"""Attribute rocprofv3 kernels to profiler-only Qwen4Exp role ranges."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"{pattern}: expected one file, got {matches}")
    return matches[0]


def _timestamp(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def normalized_role(role: str) -> str:
    return re.sub(r"layers\.\d+\.", "layers.*.", role)


def analyze(trace_dir: Path, measure_prefix: str) -> dict[str, Any]:
    marker_path = _one(trace_dir, "*_marker_api_trace.csv")
    hip_api_path = _one(trace_dir, "*_hip_api_trace.csv")
    kernel_path = _one(trace_dir, "*_kernel_trace.csv")

    windows: list[dict[str, Any]] = []
    measured: list[tuple[int, int]] = []
    for row in _read_csv(marker_path):
        name = row["Function"]
        start = _timestamp(row, "Start_Timestamp")
        end = _timestamp(row, "End_Timestamp")
        thread_id = int(float(row["Thread_Id"]))
        if name.startswith("qwen4exp_role:"):
            windows.append(
                {
                    "role": name.removeprefix("qwen4exp_role:"),
                    "start": start,
                    "end": end,
                    "tid": thread_id,
                }
            )
        elif name.startswith(measure_prefix):
            measured.append((start, end))

    if len(measured) != 1:
        raise RuntimeError(f"expected exactly one {measure_prefix!r} marker, got {measured}")
    measure_start, measure_end = measured[0]

    launches: list[tuple[int, str, str]] = []
    for row in _read_csv(hip_api_path):
        start = _timestamp(row, "Start_Timestamp")
        end = _timestamp(row, "End_Timestamp")
        if start < measure_start or end > measure_end:
            continue
        try:
            correlation_id = int(float(row["Correlation_Id"]))
            thread_id = int(float(row["Thread_Id"]))
        except (KeyError, TypeError, ValueError):
            continue
        candidates = [
            window
            for window in windows
            if window["tid"] == thread_id and window["start"] <= start and end <= window["end"]
        ]
        role = (
            min(candidates, key=lambda window: (window["end"] - window["start"], -window["start"]))["role"]
            if candidates
            else "unattributed"
        )
        launches.append((correlation_id, role, row["Function"]))

    role_by_correlation = {correlation_id: role for correlation_id, role, _fn in launches}
    function_by_correlation = {correlation_id: fn for correlation_id, _role, fn in launches}

    rows: list[dict[str, Any]] = []
    for row in _read_csv(kernel_path):
        start = _timestamp(row, "Start_Timestamp")
        end = _timestamp(row, "End_Timestamp")
        if start < measure_start or end > measure_end:
            continue
        correlation_id = int(float(row["Correlation_Id"]))
        exact_role = role_by_correlation.get(correlation_id, "unattributed")
        rows.append(
            {
                "role": normalized_role(exact_role),
                "exact_role": exact_role,
                "kernel": row["Kernel_Name"],
                "ns": end - start,
                "api": function_by_correlation.get(correlation_id, ""),
            }
        )

    grouped: dict[str, list[Any]] = defaultdict(lambda: [0, 0, set()])
    exact: dict[str, list[Any]] = defaultdict(lambda: [0, 0, set()])
    for row in rows:
        for table, key in ((grouped, row["role"]), (exact, row["exact_role"])):
            table[key][0] += row["ns"]
            table[key][1] += 1
            table[key][2].add(row["kernel"])

    unattributed: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["role"] == "unattributed":
            unattributed[row["kernel"]][0] += row["ns"]
            unattributed[row["kernel"]][1] += 1

    return {
        "window_ms": (measure_end - measure_start) / 1e6,
        "role_ranges": len(windows),
        "kernel_rows": len(rows),
        "attributed_rows": sum(row["role"] != "unattributed" for row in rows),
        "roles": [
            {"name": name, "ms": values[0] / 1e6, "rows": values[1], "symbols": len(values[2])}
            for name, values in sorted(grouped.items(), key=lambda item: -item[1][0])
        ],
        "exact_roles": [
            {"name": name, "ms": values[0] / 1e6, "rows": values[1], "symbols": len(values[2])}
            for name, values in sorted(exact.items(), key=lambda item: -item[1][0])
        ],
        "unattributed_top": [
            {"name": name, "ms": values[0] / 1e6, "rows": values[1]}
            for name, values in sorted(unattributed.items(), key=lambda item: -item[1][0])[:20]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure-prefix", default="qwen4exp_prefill_p508_")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = analyze(args.trace_dir, args.measure_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
