#!/usr/bin/env python3
"""Sum repair-pass kernel time from a rocprofv3 kernel trace.

Iteration 5 of ``let-s-qwen-3-8-flash`` asks whether the sparse exact repair
pass is really the 1370 ms share a 2026-09-16 capture reports, or something much
smaller - which decides whether the launch-geometry unit is worth building.  The
answer is the repair kernels' own duration summed across the prefill window, at
two risk multipliers.

``rocprofv3 --kernel-trace`` emits ``Start_Timestamp``/``End_Timestamp`` rather
than a duration column, so the duration is computed here.  Timestamps are
nanoseconds.

Usage: ``python3 scripts/qwen4exp_repair_trace_sum.py /tmp/repair-trace/raw-4.0 ...``
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

# Every family that exists to repair iu8 escapes, and the risk owners that
# produce the escapes, so the repair share can be read against its cause.
REPAIR_MARKERS = ("sparse_exact_repair", "sparse_repair_kernel")
RISK_MARKERS = ("iu8_risk_prefill_kernel",)


def classify(name: str) -> str | None:
    if any(marker in name for marker in REPAIR_MARKERS):
        return "repair"
    if any(marker in name for marker in RISK_MARKERS):
        return "risk_owner"
    return None


def short(name: str) -> str:
    """Trim a mangled kernel name down to its family."""
    for marker in ("gguf_q4_k_selected_dual_", "q5_1_selected_", "gguf_q5_k_selected_dual_",
                   "q8_0_selected_"):
        index = name.find(marker)
        if index >= 0:
            tail = name[index + len(marker):]
            for stop in ("_kernel", "(", "E"):
                cut = tail.find(stop)
                if cut > 0:
                    tail = tail[:cut]
            return f"{marker}{tail}".rstrip("_")
    return name[:80]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    for directory in argv[1:]:
        root = Path(directory)
        csvs = sorted(root.glob("*kernel_trace*.csv")) or sorted(root.glob("*.csv"))
        if not csvs:
            print(f"{root}: no CSV found")
            continue
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        per_family: dict[tuple[str, str], float] = defaultdict(float)
        per_family_count: dict[tuple[str, str], int] = defaultdict(int)
        for path in csvs:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    name = row.get("Kernel_Name") or ""
                    label = classify(name)
                    if label is None:
                        continue
                    try:
                        start = float(row["Start_Timestamp"])
                        end = float(row["End_Timestamp"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    duration = end - start
                    if duration <= 0:
                        continue
                    totals[label] += duration
                    counts[label] += 1
                    key = (label, short(name))
                    per_family[key] += duration
                    per_family_count[key] += 1
        print(f"== {root}")
        if not totals:
            print("   no matching kernels")
            continue
        grand = totals.get("repair", 0.0) + totals.get("risk_owner", 0.0)
        for label in ("repair", "risk_owner"):
            if label in totals:
                ms = totals[label] / 1e6
                share = 100.0 * totals[label] / grand if grand else 0.0
                print(f"   {label:11s} {counts[label]:6d} launches  {ms:10.3f} ms"
                      f"  {share:5.1f}% of repair+risk")
        for (label, family), value in sorted(per_family.items(),
                                             key=lambda kv: -kv[1]):
            print(f"      {label:10s} {per_family_count[(label, family)]:6d}"
                  f"  {value / 1e6:10.3f} ms  {family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
