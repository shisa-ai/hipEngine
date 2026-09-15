#!/usr/bin/env python3
"""Run-to-run repeatability of a role-marked prefill profile.

Before a per-component gap can be attributed between two engines, the noise floor
of the measurement has to be known. Otherwise a difference between two single
runs gets reported as a finding when it is inside the scatter of the same
configuration measured twice.

This takes two or more role-analysis artifacts from runs of the *same*
configuration and reports, for each pair, how much the per-(role, kernel) times
move. Two properties matter:

* the **net** delta, which is what a family-level comparison sees, and
* the **scatter**, ``sum |delta|`` and the per-row standard deviation, which is
  what a per-component comparison sees.

A configuration whose net delta is small while the scatter is large supports
family-level attribution only. The script reports both so that call can be made
explicitly rather than assumed.

The pairs are also broken down by row size, because a difference that grows with
kernel size is not averaging noise — averaging noise shrinks as rows get bigger.

Example:
    python3 scripts/qwen4exp_profile_repeatability.py \
        --analysis cold_a.json cold_b.json warm.json \
        --labels "cold A" "cold B" "warm PLE" \
        --output benchmarks/results/<dir>/repeatability.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any

# Rows below this are mostly launch overhead and dominate the count without
# carrying the cost, so they are excluded from the relative-scatter statistics.
MIN_ROW_MS = 20.0

SIZE_BANDS = ((20.0, 50.0), (50.0, 100.0), (100.0, 300.0), (300.0, 1e9))

_LAYER = re.compile(r"layers\.(\d+)\.")


def load(path: Path) -> dict[tuple[str, str], float]:
    data = json.loads(path.read_text())
    return {
        (row["role"], row["kernel"]): float(row["ms"])
        for row in data["exact_role_kernels"]
    }


def summarize(a: dict[tuple[str, str], float], b: dict[tuple[str, str], float]) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("the two analyses share no (role, kernel) rows")
    deltas = [b[k] - a[k] for k in common]
    net = sum(deltas)
    scatter = sum(abs(d) for d in deltas)
    large = [b[k] - a[k] for k in common if a[k] >= MIN_ROW_MS]
    large_pct = [100.0 * (b[k] - a[k]) / a[k] for k in common if a[k] >= MIN_ROW_MS]

    bands: list[dict[str, Any]] = []
    for low, high in SIZE_BANDS:
        pct = [
            100.0 * (b[k] - a[k]) / a[k]
            for k in common
            if low <= a[k] < high
        ]
        if len(pct) < 2:
            continue
        bands.append({
            "low_ms": low,
            "high_ms": None if high > 1e8 else high,
            "rows": len(pct),
            "median_pct": round(statistics.median(pct), 2),
            "stdev_pct": round(statistics.pstdev(pct), 2),
            "min_pct": round(min(pct), 2),
            "max_pct": round(max(pct), 2),
        })

    by_layer: list[dict[str, Any]] = []
    for start in range(0, 48, 8):
        pct = [
            100.0 * (b[k] - a[k]) / a[k]
            for k in common
            if a[k] >= MIN_ROW_MS
            and (m := _LAYER.search(k[0])) is not None
            and start <= int(m.group(1)) < start + 8
        ]
        if len(pct) < 2:
            continue
        by_layer.append({
            "layer_start": start,
            "layer_end": start + 7,
            "rows": len(pct),
            "median_pct": round(statistics.median(pct), 2),
            "stdev_pct": round(statistics.pstdev(pct), 2),
        })

    return {
        "rows_compared": len(common),
        "rows_at_or_above_20ms": len(large),
        "net_ms": round(net, 1),
        "scatter_ms": round(scatter, 1),
        "scatter_over_net": round(scatter / abs(net), 2) if net else None,
        "median_pct": round(statistics.median(large_pct), 2) if large_pct else None,
        "stdev_pct": round(statistics.pstdev(large_pct), 2) if large_pct else None,
        "min_pct": round(min(large_pct), 2) if large_pct else None,
        "max_pct": round(max(large_pct), 2) if large_pct else None,
        "total_a_ms": round(sum(a.values()), 1),
        "total_b_ms": round(sum(b.values()), 1),
        "by_size_band": bands,
        "by_layer_octet": by_layer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", default="qwen4exp_profile_repeatability")
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    labels = args.labels or [p.stem for p in args.analysis]
    if len(labels) != len(args.analysis):
        raise SystemExit("--labels must match --analysis one for one")
    loaded = {label: load(path) for label, path in zip(labels, args.analysis)}

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(labels, 2):
        summary = summarize(loaded[left], loaded[right])
        summary["left"] = left
        summary["right"] = right
        pairs.append(summary)

    header = (
        f"{'comparison':34s} {'net ms':>9s} {'scatter':>9s} {'x net':>6s} "
        f"{'median':>8s} {'stdev':>7s} {'min':>8s} {'max':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in pairs:
        print(
            f"{row['left'] + ' vs ' + row['right']:34s} {row['net_ms']:+9.1f} "
            f"{row['scatter_ms']:9.1f} {str(row['scatter_over_net']):>6s} "
            f"{row['median_pct']:+8.2f} {row['stdev_pct']:7.2f} "
            f"{row['min_pct']:+8.2f} {row['max_pct']:+8.2f}"
        )

    print("\nscatter by row size (a stdev that grows with row size is not averaging noise)")
    for row in pairs:
        print(f"  {row['left']} vs {row['right']}")
        for band in row["by_size_band"]:
            high = "inf" if band["high_ms"] is None else str(int(band["high_ms"]))
            print(
                f"    {int(band['low_ms']):5d}-{high:>5s} ms  n={band['rows']:4d}  "
                f"median={band['median_pct']:+6.2f}%  stdev={band['stdev_pct']:5.2f}%"
            )

    totals = {label: round(sum(m.values()), 1) for label, m in loaded.items()}
    print("\ntotals: " + "  ".join(f"{k}={v} ms" for k, v in totals.items()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": 1,
        "kind": args.kind,
        "performance_claim": False,
        "status": "diagnostic",
        "note": args.note,
        "inputs": [str(p) for p in args.analysis],
        "labels": labels,
        "totals_ms": totals,
        "min_row_ms": MIN_ROW_MS,
        "pairs": pairs,
        "notes": [
            "Per-(role, kernel) times from role-marked rocprofv3 captures of the same configuration.",
            "net_ms is what a family-level comparison sees; scatter_ms is what a per-component comparison sees.",
            "by_size_band exists because averaging noise shrinks with row size; a stdev that grows with row size is systematic.",
        ],
    }, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
