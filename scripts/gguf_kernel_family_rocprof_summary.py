#!/usr/bin/env python3
"""Summarize one rocprofv3 kernel-trace output directory by kernel family.

Reads the kernel-dispatch CSV that ``rocprofv3 --kernel-trace --output-format
csv -d DIR`` writes and aggregates per kernel name (optionally restricted to a
ROCTx region prefix produced by ``--kernel-rename``). Used to answer whether two
workload shapes run the same kernel with worse efficiency or select different
kernels at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def _columns(header: list[str]) -> dict[str, str]:
    return {name.lower(): name for name in header}


def _pick(keys: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in keys:
            return keys[candidate]
    return None


def load_kernels(csv_path: Path, region_pattern: re.Pattern | None) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        keys = _columns(header)
        name_col = _pick(keys, "kernel_name", "kernel name", "kernname", "kernel")
        if name_col is None:
            return rows
        start_col = _pick(keys, "start", "startts", "start_ts", "start_timestamp", "timestamp")
        end_col = _pick(keys, "end", "endts", "end_ts", "end_timestamp")
        dur_col = _pick(keys, "dur", "duration", "duration_ns", "durationns")
        grid = {axis: _pick(keys, f"grd/{axis}", f"grid/{axis}", f"grid_{axis}", f"grid_size_{axis}") for axis in "xyz"}
        block = {axis: _pick(keys, f"blk/{axis}", f"block/{axis}", f"block_{axis}", f"workgroup_size_{axis}") for axis in "xyz"}
        for row in reader:
            name = (row.get(name_col) or "").strip()
            if not name:
                continue
            if region_pattern is not None and not region_pattern.search(name):
                continue
            duration = row.get(dur_col) if dur_col else None
            start = row.get(start_col) if start_col else None
            end = row.get(end_col) if end_col else None
            try:
                duration_ns = float(duration) if duration not in (None, "") else None
            except ValueError:
                duration_ns = None
            try:
                start_ns = float(start) if start not in (None, "") else None
            except ValueError:
                start_ns = None
            try:
                end_ns = float(end) if end not in (None, "") else None
            except ValueError:
                end_ns = None
            if duration_ns is None and start_ns is not None and end_ns is not None:
                duration_ns = end_ns - start_ns
            dims = {}
            for axis, col in {**{f"grid_{k}": v for k, v in grid.items()}, **{f"block_{k}": v for k, v in block.items()}}.items():
                value = row.get(col) if col else None
                try:
                    dims[axis] = int(value) if value not in (None, "") else None
                except ValueError:
                    dims[axis] = None
            rows.append(
                {
                    "name": re.sub(r"^\[[^\]]*\]\s*", "", name),
                    "raw_name": name,
                    "duration_ns": duration_ns,
                    "start_ns": start_ns,
                    **dims,
                }
            )
    return rows


def summarize(rows: list[dict], *, tokens: int | None) -> dict:
    families: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_ns": 0.0, "durations": [], "grids": set()})
    for row in rows:
        family = families[row["name"]]
        family["count"] += 1
        if row["duration_ns"] is not None:
            family["total_ns"] += row["duration_ns"]
            family["durations"].append(row["duration_ns"])
        grid = tuple(row.get(f"grid_{axis}") for axis in "xyz")
        if any(value is not None for value in grid):
            family["grids"].add(grid)
    out = []
    for name, family in families.items():
        durations = family["durations"]
        entry = {
            "kernel": name,
            "launches": family["count"],
            "total_ms": round(family["total_ns"] / 1e6, 3),
            "mean_us": round(statistics.mean(durations) / 1e3, 2) if durations else None,
            "median_us": round(statistics.median(durations) / 1e3, 2) if durations else None,
            "grids": sorted({str(g) for g in family["grids"]})[:4],
        }
        if tokens:
            entry["ms_per_1k_tokens"] = round(entry["total_ms"] / tokens * 1000.0, 2)
        out.append(entry)
    out.sort(key=lambda item: -item["total_ms"])
    total = sum(item["total_ms"] for item in out)
    return {"kernels": out, "total_ms": round(total, 3), "launches": sum(item["launches"] for item in out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=None, help="Workload token count for per-token normalization.")
    parser.add_argument("--region", default=None, help="Regex matched against the --kernel-rename region prefix.")
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    csv_files = sorted(args.trace_dir.rglob("*.csv"))
    if not csv_files:
        raise SystemExit(f"no CSV under {args.trace_dir}")
    pattern = re.compile(args.region) if args.region else None
    merged: list[dict] = []
    used: list[str] = []
    for path in csv_files:
        rows = load_kernels(path, pattern)
        if rows:
            merged.extend(rows)
            used.append(str(path))
    if not merged:
        raise SystemExit(f"no kernel rows parsed from {len(csv_files)} CSV file(s) under {args.trace_dir}")
    report = summarize(merged, tokens=args.tokens)
    report["csv_files"] = used
    print(f"total device {report['total_ms']:.2f} ms over {report['launches']} launches ({len(used)} csv)")
    for entry in report["kernels"][: args.top]:
        per_token = f"  {entry['ms_per_1k_tokens']:8.2f} ms/1k tok" if args.tokens else ""
        print(f"{entry['total_ms']:9.2f} ms  n={entry['launches']:5d}  med={entry['median_us']} us  grid={entry['grids'][:2]}  {entry['kernel'][:72]}{per_token}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
