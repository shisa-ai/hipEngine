#!/usr/bin/env python3
"""Compare mixed-length ITL A/B run artifacts (before vs after, N reps each).

Usage:
  python scripts/analyze_mixed_length_ab.py \
      --before rep1.json rep2.json --after rep1.json rep2.json \
      --json-out comparison.json

Primary endpoint: mixed-phase window-average ITL of the long request (the
phase where short rows are co-resident). Alone/post phases are controls and
must stay flat; short-request walls are secondary.
"""
from __future__ import annotations

import argparse
import json
import statistics


def load_runs(paths):
    runs = [json.load(open(p)) for p in paths]
    for r in runs:
        assert r["phase_chunk_counts"]["alone"] >= 8, f"too few alone chunks: {r['label']}"
        assert r["phase_chunk_counts"]["mixed"] >= 8, f"too few mixed chunks: {r['label']}"
    return runs


def arm_stats(runs):
    def vals(key):
        out = []
        for r in runs:
            v = r["phase_window_avg_ms"].get(key)
            if v is not None:
                out.append(v)
        return out

    stats = {}
    for phase in ("alone", "mixed", "post"):
        v = vals(phase)
        stats[phase] = {
            "per_rep_ms": v,
            "median_ms": statistics.median(v) if v else None,
        }
    stats["short_walls_s"] = [
        [s["wall_s"] for s in r["short_requests"]] for r in runs
    ]
    stats["labels"] = [r["label"] for r in runs]
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    before = arm_stats(load_runs(args.before))
    after = arm_stats(load_runs(args.after))
    deltas = {}
    for phase in ("alone", "mixed", "post"):
        b, a = before[phase]["median_ms"], after[phase]["median_ms"]
        deltas[phase] = {
            "before_median_ms": b,
            "after_median_ms": a,
            "delta_pct": round((a / b - 1) * 100, 2) if b and a else None,
        }
    result = {
        "endpoint": "mixed-phase window-average ITL of the long request",
        "controls": ["alone", "post"],
        "before": before,
        "after": after,
        "deltas": deltas,
    }
    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(deltas, indent=2))
    print("artifact:", args.json_out)


if __name__ == "__main__":
    main()