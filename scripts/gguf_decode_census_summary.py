#!/usr/bin/env python3
"""Per-kernel decode census from a rocprofv3 kernel trace.

The decode-graph driver runs prefill, warmup, capture, then a 0.5 s GPU
idle gap followed by 32 measured decode steps; the measured window is
the trailing kernel burst after the last long idle interval. This tool
slices that window and reports per-kernel pure us/token over the 32
steps, matching the 2026-09-11 four-arm attribution protocol.

Usage: gguf_decode_census_summary.py TRACE.csv [--steps 32]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    with args.trace.open() as fh:
        for r in csv.DictReader(fh):
            try:
                start = int(r["Start_Timestamp"])
                end = int(r["End_Timestamp"])
            except (TypeError, ValueError):
                continue
            rows.append((start, end, r))
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise SystemExit("no kernel records")

    steps = max(args.steps, 1)
    # The measured window is the trailing 32 decode steps. The driver's
    # 0.5 s idle gap is the intended boundary, but a plain-arm trace can
    # carry longer idle gaps inside the decode burst (host-side graph
    # replay stalls), which mis-slices a pure-gap heuristic and can even
    # make "pure" exceed wall. The per-step advance_decode_position
    # kernel fires exactly once per decode step, so the window is sliced
    # on its 32 trailing occurrences instead.
    advance_idx = [
        i for i, (s, e, r) in enumerate(rows)
        if "advance_decode_position" in r["Kernel_Name"]
    ]
    if len(advance_idx) >= steps:
        window_rows = rows[advance_idx[-steps]:]
    else:  # fall back to the trailing-gap heuristic
        gap_start = rows[0][0]
        for i in range(1, len(rows)):
            idle = rows[i][0] - rows[i - 1][1]
            if idle >= 300_000_000:
                gap_start = rows[i][0]
        window_rows = [x for x in rows if x[0] >= gap_start]
    window = [r for (s, e, r) in window_rows]

    per = defaultdict(lambda: [0, 0.0])
    meta = {}
    total = 0.0
    for r in window:
        name = r["Kernel_Name"]
        dur = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        per[name][0] += 1
        per[name][1] += dur
        total += dur
        meta[name] = (r.get("VGPR_Count", ""), r.get("LDS_Block_Size", ""),
                      r.get("Workgroup_Size_X", ""))
    out = {
        "window_kernels": len(window),
        "steps": steps,
        "launches_per_token": round(len(window) / steps, 2),
        "pure_us_per_token": round(total / steps / 1000.0, 2),
        "by_kernel": {
            name: {
                "us_per_token": round(d / steps / 1000.0, 2),
                "launches_per_token": round(c / steps, 2),
                "vgpr": meta[name][0], "lds": meta[name][1],
                "workgroup": meta[name][2],
            }
            for name, (c, d) in sorted(per.items(), key=lambda x: -x[1][1])
        },
    }
    print(json.dumps({k: out[k] for k in
                      ("window_kernels", "launches_per_token",
                       "pure_us_per_token")}, indent=1))
    for name, v in list(out["by_kernel"].items())[:12]:
        print(f"  {v['us_per_token']:9.2f} us/tok {v['launches_per_token']:7.1f}/tok"
              f"  {name[:70]}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
