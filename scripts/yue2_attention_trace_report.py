#!/usr/bin/env python3
"""Summarize a ``rocprofv3 --kernel-trace`` CSV into per-kernel totals.

M7's profile of the YuE2 e2e path is driven from these summaries: which kernel
owns the GPU time, and how the NAR attention's per-call cost moves between
candidates. Raw traces stay out of the tree (they are large and machine
specific); this is the reproducible reduction of them.

Usage:
    rocprofv3 --kernel-trace --output-format csv -d /tmp/trace -- \
        python3 scripts/yue2_e2e_gate.py --only <case> --steps 2 --skip-live
    python3 scripts/yue2_attention_trace_report.py /tmp/trace --json out.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def summarize(directory: Path) -> dict:
    rows = []
    for path in sorted(directory.rglob("*_kernel_trace.csv")):
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    totals: dict[str, dict] = collections.defaultdict(
        lambda: {"dispatches": 0, "total_ns": 0, "vgpr": set(), "grid": set()}
    )
    for row in rows:
        name = row["Kernel_Name"].split("(")[0]
        entry = totals[name]
        entry["dispatches"] += 1
        entry["total_ns"] += int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
        entry["vgpr"].add(int(row["VGPR_Count"]))
        entry["grid"].add((int(row["Grid_Size_X"]), int(row["Grid_Size_Y"])))
    out = {}
    for name, entry in sorted(totals.items(), key=lambda kv: -kv[1]["total_ns"]):
        out[name] = {
            "dispatches": entry["dispatches"],
            "total_ms": round(entry["total_ns"] / 1e6, 3),
            "avg_us": round(entry["total_ns"] / entry["dispatches"] / 1e3, 2),
            "vgpr": sorted(entry["vgpr"]),
            "grid_xy": sorted(entry["grid"])[:4],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    summary = summarize(args.trace)
    print(f"{'kernel':48s} {'n':>6s} {'total_ms':>10s} {'avg_us':>10s} {'vgpr':>6s}")
    for name, entry in list(summary.items())[: args.top]:
        print(
            f"{name[:48]:48s} {entry['dispatches']:6d} {entry['total_ms']:10.3f} "
            f"{entry['avg_us']:10.2f} {entry['vgpr']}"
        )
    if args.json:
        payload = {
            "provenance": {
                "command_line": " ".join(sys.argv),
                "trace_dir": str(args.trace),
                "revision": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=REPO,
                    capture_output=True, text=True, check=False,
                ).stdout.strip(),
            },
            "kernels": summary,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"[trace] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
