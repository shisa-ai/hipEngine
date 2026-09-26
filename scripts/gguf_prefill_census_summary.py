#!/usr/bin/env python3
"""Rank per-kernel time from a prefill-mode rocprofv3 kernel trace.

Companion to ``scripts/gguf_decode_graph_rocprof_driver.py`` mode=prefill
(campaign UD-GFX1151-OPTIMIZE2 E1). The driver places a 0.5 s GPU idle gap
between session construction and the measured fresh prefill, so the measured
window is every kernel dispatch after the last >=0.4 s inter-kernel gap --
the same trailing-burst slicing the decode census uses, no marker trace
needed. Everything before the gap (warm-up session, copies, construction) is
excluded.

Usage::

    python3 scripts/gguf_prefill_census_summary.py TRACE.csv [--json OUT.json]

Output: ranked per-kernel table (launch count, total/mean duration, share of
window) plus window geometry, all with provenance-friendly raw numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict

# The driver sleeps 0.5 s before the measured prefill; 0.4 s clears it with
# margin while staying far above any within-prefill kernel gap.
MIN_GAP_NS = 400_000_000


def load(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("Kind") == "KERNEL_DISPATCH"]
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="rocprofv3 --kernel-trace CSV")
    parser.add_argument("--json", default=None, help="write ranked table here")
    parser.add_argument("--top", type=int, default=25, help="rows to print")
    args = parser.parse_args()

    rows = load(args.trace)
    if not rows:
        print("no kernel dispatches in trace", file=sys.stderr)
        return 1

    # Trailing window: everything after the last >= MIN_GAP_NS idle gap.
    cut = None
    for prev, cur in zip(rows, rows[1:]):
        if int(cur["Start_Timestamp"]) - int(prev["End_Timestamp"]) >= MIN_GAP_NS:
            cut = cur
    if cut is None:
        print(f"no >= {MIN_GAP_NS} ns gap found; cannot isolate measured window",
              file=sys.stderr)
        return 1
    window = [r for r in rows if int(r["Start_Timestamp"]) >= int(cut["Start_Timestamp"])]

    total_ns = sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in window)
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_ns": 0, "first": None})
    for r in window:
        name = r["Kernel_Name"]
        dur = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        a = agg[name]
        a["count"] += 1
        a["total_ns"] += dur
        if a["first"] is None:
            a["first"] = r

    ranked = sorted(agg.items(), key=lambda kv: -kv[1]["total_ns"])
    table = []
    for name, a in ranked:
        f = a["first"]
        grid = (int(f["Grid_Size_X"]), int(f["Grid_Size_Y"]), int(f["Grid_Size_Z"]))
        wg = (int(f["Workgroup_Size_X"]), int(f["Workgroup_Size_Y"]),
              int(f["Workgroup_Size_Z"]))
        table.append({
            "kernel": name,
            "count": a["count"],
            "total_ms": a["total_ns"] / 1e6,
            "mean_us": a["total_ns"] / a["count"] / 1e3,
            "share": a["total_ns"] / total_ns,
            "grid_first": grid,
            "workgroup_first": wg,
            "vgpr_first": int(f["VGPR_Count"]),
            "sgpr_first": int(f["SGPR_Count"]),
            "lds_first": int(f["LDS_Block_Size"]),
            "scratch_first": int(f["Scratch_Size"]),
        })

    span_ns = int(window[-1]["End_Timestamp"]) - int(window[0]["Start_Timestamp"])
    summary = {
        "trace": args.trace,
        "dispatches_total": len(rows),
        "window_dispatches": len(window),
        "window_total_ms": total_ns / 1e6,
        "window_span_ms": span_ns / 1e6,
        "window_first_start": int(window[0]["Start_Timestamp"]),
        "kernels_distinct": len(ranked),
        "ranked": table,
    }

    print(f"trace: {args.trace}")
    print(f"dispatches: {len(rows)} total, {len(window)} in window "
          f"(after last >= {MIN_GAP_NS // 1_000_000} ms gap)")
    print(f"window: {total_ns / 1e6:.2f} ms kernel time over {span_ns / 1e6:.2f} ms span, "
          f"{len(ranked)} distinct kernels")
    print(f"{'rank':>4} {'kernel':<62} {'count':>7} {'total_ms':>10} {'share':>7} "
          f"{'mean_us':>9}")
    for i, e in enumerate(table[:args.top], 1):
        print(f"{i:>4} {e['kernel'][:62]:<62} {e['count']:>7} {e['total_ms']:>10.3f} "
              f"{e['share']:>6.1%} {e['mean_us']:>9.2f}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())