"""Split a rocprofv3 counter census by execution phase.

`scripts/gemma4_campaign_bench.py` runs a prefill and then a decode loop, so a
profiled run contains both phases in one dispatch stream. Summing a counter over
the whole run mixes them, and at 512/128 the decode phase dominates: it is 94.6
percent of the read traffic. Two worklog entries on this branch reported a
"prefill" figure that was mostly decode before the split was applied by hand.

This script makes the split structural. It reads the counter-collection CSVs
rocprofv3 writes, finds the first dispatch of a marker kernel -- by default
`gemma4_attention_decode_class_kernel`, the first thing the decode loop launches
that the prefill does not -- and reports per-kernel totals either side of it.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/rocprof_phase_split.py /tmp/prof -o f
  PYTHONPATH=. .venv/bin/python scripts/rocprof_phase_split.py /tmp/prof -o f \\
      --marker gemma4_attention_decode_class_kernel --top 8

`-d <dir> -o <name>` is what `rocprofv3 -f csv -d <dir> -o <name> -- ...` writes,
either directly under the directory or under `pass_N/` subdirectories; both
layouts are found.

The counter's absolute unit is the profiler's, not this script's. What is safe
to compare is one kernel against another within one phase, because any constant
factor cancels.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

DEFAULT_MARKER = "gemma4_attention_decode_class_kernel"

# "void (anonymous namespace)::gemma4_attention_prefill_kernel<unsigned short>(...)"
_KERNEL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*_kernel)")


def kernel_name(mangled: str) -> str:
    """Return the kernel's own name, without the namespace or template args."""
    match = _KERNEL_RE.search(mangled)
    if match is not None:
        return match.group(1)
    match = re.search(r"::([A-Za-z_][A-Za-z0-9_]*)", mangled)
    if match is not None:
        return match.group(1)
    return mangled[:48]


def load_rows(directory: Path, stem: str | None) -> list[tuple[int, str, float]]:
    """Return (dispatch_id, kernel, counter_value) for every collected row."""
    pattern = f"{stem}*counter_collection.csv" if stem else "*counter_collection.csv"
    paths = sorted(directory.glob(pattern)) + sorted(
        directory.glob(f"pass_*/{pattern}")
    )
    if not paths:
        raise SystemExit(f"no counter-collection CSV matching {pattern} under {directory}")
    rows: list[tuple[int, str, float]] = []
    for path in paths:
        with path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                try:
                    dispatch = int(record["Dispatch_Id"])
                    value = float(record["Counter_Value"])
                except (KeyError, ValueError):
                    continue
                rows.append((dispatch, kernel_name(record["Kernel_Name"]), value))
    if not rows:
        raise SystemExit(f"{len(paths)} CSV file(s) under {directory} held no counter rows")
    return rows


def report(label: str, totals: collections.Counter, dispatches: collections.Counter,
           top: int) -> None:
    total = sum(totals.values())
    print(f"=== {label}: {total:,.0f} counter units over "
          f"{sum(dispatches.values()):,} dispatches")
    if total == 0:
        print("    (empty)")
        return
    for name, value in totals.most_common(top):
        print(f"    {name[:56]:<56} {value:>14,.0f} {100 * value / total:>6.1f}%"
              f"  n={dispatches[name]:,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="rocprofv3 -d directory")
    parser.add_argument("-o", "--stem", default=None,
                        help="rocprofv3 -o name; omit to read every CSV present")
    parser.add_argument("--marker", default=DEFAULT_MARKER,
                        help=f"kernel whose first dispatch starts phase 2 "
                             f"(default {DEFAULT_MARKER})")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--label-1", default="phase 1")
    parser.add_argument("--label-2", default="phase 2")
    args = parser.parse_args()

    rows = load_rows(args.directory, args.stem)
    markers = [dispatch for dispatch, name, _ in rows if name == args.marker]
    if not markers:
        seen = sorted({name for _, name, _ in rows})
        print(f"marker {args.marker!r} never dispatched; {len(seen)} kernel(s) present:",
              file=sys.stderr)
        for name in seen[:20]:
            print(f"    {name}", file=sys.stderr)
        return 2
    boundary = min(markers)

    totals: dict[str, collections.Counter] = {
        args.label_1: collections.Counter(), args.label_2: collections.Counter()}
    dispatches: dict[str, collections.Counter] = {
        args.label_1: collections.Counter(), args.label_2: collections.Counter()}
    for dispatch, name, value in rows:
        phase = args.label_1 if dispatch < boundary else args.label_2
        totals[phase][name] += value
        dispatches[phase][name] += 1

    print(f"{len(rows):,} counter rows; {args.label_2} begins at dispatch {boundary} "
          f"(of {max(d for d, _, _ in rows):,})")
    print()
    report(args.label_1, totals[args.label_1], dispatches[args.label_1], args.top)
    print()
    report(args.label_2, totals[args.label_2], dispatches[args.label_2], args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
