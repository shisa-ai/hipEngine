#!/usr/bin/env python3
"""Per-marker-range kernel breakdown of a rocprofv3 CSV trace.

Attributes every kernel dispatch whose start falls inside a ROCTx range to that
range's label, then prints, per label: calls, wall per call, kernel-sum per
call, launches per call, host/launch gap per call, and the top kernel families.
Use it with any driver that wraps phases in ``roctxRangePushA``/``Pop`` (for
example ``scripts/muse_glimmer_context_profile.py``) and was traced with
``rocprofv3 --kernel-trace --marker-trace --output-format csv``.

Ranges must not nest (a nested range double-counts its kernels). Labels that
start with ``--skip-prefix`` (default ``warm``) are ignored.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import re
import sys
from pathlib import Path


def family(name: str) -> str:
    """Kernel name without return type, namespace noise, template args or params."""
    name = name.replace("(anonymous namespace)::", "")
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"[<(].*", "", name)
    return name.rsplit("::", 1)[-1] or name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernel_trace", type=Path)
    parser.add_argument("marker_trace", type=Path)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--skip-prefix", default="warm")
    args = parser.parse_args(argv)

    kernels: list[tuple[int, int, str]] = []
    with args.kernel_trace.open(newline="") as fh:
        for row in csv.DictReader(fh):
            kernels.append((int(row["Start_Timestamp"]), int(row["End_Timestamp"]), row["Kernel_Name"]))
    kernels.sort()
    starts = [k[0] for k in kernels]

    calls: collections.Counter[str] = collections.Counter()
    wall: collections.Counter[str] = collections.Counter()
    launches: collections.Counter[str] = collections.Counter()
    by_family: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    with args.marker_trace.open(newline="") as fh:
        for row in csv.DictReader(fh):
            label = row["Function"]
            if args.skip_prefix and label.startswith(args.skip_prefix):
                continue
            begin, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            calls[label] += 1
            wall[label] += end - begin
            lo, hi = bisect.bisect_left(starts, begin), bisect.bisect_left(starts, end)
            launches[label] += hi - lo
            for start, stop, name in kernels[lo:hi]:
                by_family[label][family(name)] += stop - start

    for label in sorted(calls):
        n = calls[label]
        ksum = sum(by_family[label].values()) / n / 1e6
        print(f"\n== {label}  calls={n}  wall/call={wall[label] / n / 1e6:.2f} ms  "
              f"kernels/call={ksum:.2f} ms  launches/call={launches[label] / n:.0f}  "
              f"gap/call={wall[label] / n / 1e6 - ksum:.2f} ms")
        for name, ns in by_family[label].most_common(args.top):
            ms = ns / n / 1e6
            print(f"   {name:56s} {ms:9.2f} ms  {100 * ms / ksum if ksum else 0:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
