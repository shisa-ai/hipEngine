"""Roll up a rocprofv3 --kernel-trace CSV into kernel-family totals.

Reports each kernel by aggregate device time, launch count, mean duration, and
launch geometry (grid/workgroup/VGPR/SGPR), which is what identifies *which
owner* actually ran for a stage. Optionally filters kernel names by regex, so a
stage question ("who serves ssm_out at 45 rows?") becomes one command.

Usage: gguf_kernel_trace_rollup.py TRACE_DIR [--regex RE] [--top N] [--unit us|ns|ms]
"""

from __future__ import annotations

import argparse
import csv
import glob
import pathlib
import re
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_dir")
    ap.add_argument("--regex", default=None, help="keep kernel names matching this regex")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--unit", default="us", choices=("ns", "us", "ms"))
    ap.add_argument("--geom", action="store_true", help="show grid/workgroup/VGPR/SGPR")
    args = ap.parse_args()

    scale = {"ns": 1.0, "us": 1e-3, "ms": 1e-6}[args.unit]
    paths = sorted(glob.glob(str(pathlib.Path(args.trace_dir) / "**" / "*.csv"), recursive=True))
    if not paths:
        raise SystemExit(f"no CSV under {args.trace_dir}")

    total = 0.0
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"ns": 0.0, "n": 0})
    geom: dict[str, tuple[str, str, str, str]] = {}
    seen = 0
    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            name_key = next((f for f in fields if "Kernel_Name" in f), None)
            dur_key = next((f for f in fields if f.strip().lower() in ("duration", "durationns")), None)
            start_key = next((f for f in fields if "Start_Timestamp" in f), None)
            end_key = next((f for f in fields if "End_Timestamp" in f), None)
            if name_key is None or (dur_key is None and not (start_key and end_key)):
                continue
            grid = next((f for f in fields if "Grid_Size_X" in f), None)
            wg = next((f for f in fields if "Workgroup_Size_X" in f), None)
            vgpr = next((f for f in fields if "VGPR_Count" in f or f == "VGPR"), None)
            sgpr = next((f for f in fields if "SGPR_Count" in f or f == "SGPR"), None)
            for row in reader:
                name = (row.get(name_key) or "").strip()
                if not name:
                    continue
                seen += 1
                if args.regex and not re.search(args.regex, name):
                    continue
                try:
                    if dur_key is not None:
                        ns = float(row[dur_key])
                    else:
                        ns = float(row[end_key]) - float(row[start_key])
                except (TypeError, ValueError):
                    continue
                if ns <= 0:
                    continue
                stats[name]["ns"] += ns
                stats[name]["n"] += 1
                total += ns
                if args.geom and name not in geom:
                    geom[name] = (
                        row.get(grid) or "?",
                        row.get(wg) or "?",
                        row.get(vgpr) or "?",
                        row.get(sgpr) or "?",
                    )

    rows = sorted(stats.items(), key=lambda kv: -kv[1]["ns"])[: max(1, args.top)]
    print(f"records={seen} kernels={len(stats)} total={total * scale:.1f} {args.unit}")
    print(f"{'share':>6}  {'total_' + args.unit:>10} {'n':>6} {'mean_' + args.unit:>9}  kernel")
    for name, st in rows:
        share = st["ns"] / total * 100 if total else 0.0
        mean = st["ns"] / max(1, st["n"]) * scale
        print(f"{share:5.1f}% {st['ns'] * scale:10.1f} {int(st['n']):6d} {mean:9.2f}  {name[:78]}")
        if args.geom:
            g, w, v, s = geom.get(name, ("?", "?", "?", "?"))
            print(f"{'':7}  grid={g} workgroup={w} VGPR={v} SGPR={s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
