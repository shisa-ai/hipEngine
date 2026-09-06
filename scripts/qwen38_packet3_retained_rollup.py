#!/usr/bin/env python3
"""Packet 3 rollup: ratio table from the retained-cell re-measurement JSONs.

Reads the p3-* bench JSONs and prints the per-cell MTP-vs-AR ratio next to
the pre-change retained floor, plus the token-exactness gates. Exit code 1
if any gate fails: ar_exact token contract broken, route expectation broken,
or any retained-cell ratio below its floor minus noise tolerance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Pre-change retained floors from the packet0/packet2 evidence rows.
FLOORS = {
    "c1k3": 1.4734,
    "c1k2": 1.5590,
    "c2k2": 0.917,
    "c8k3": 0.987,
    "c5k3": 0.800,
    "c2k1": 0.721,
    "c7k3": 0.882,
}
NOISE = 0.02

KEY_BY_CELL = {
    "c1k3": "1",
    "c1k2": "1",
    "c2k2": "2",
    "c8k3": "8",
    "c5k3": "5",
    "c2k1": "2",
    "c7k3": "7",
}


def main() -> int:
    out_dir = Path("/tmp/he-bettermtp-raw/packet3")
    failures: list[str] = []
    rows = []
    for cell, width_key in KEY_BY_CELL.items():
        path = out_dir / f"p3-{cell}-{'retained' if cell in ('c1k3', 'c1k2', 'c2k2', 'c8k3') else 'screen'}.json"
        if not path.exists():
            rows.append((cell, None, FLOORS[cell], "MISSING"))
            failures.append(f"{cell}: result JSON missing")
            continue
        d = json.loads(path.read_text())
        summary = d["summary"][width_key]
        ar = summary["ar"]["tok_s"]
        mtp = summary["mtp"]["tok_s"]
        ratio = summary["mtp_vs_ar_ratio"]
        tokens_exact = summary["exact_cells"] == summary["cells"]
        engaged = summary.get("engaged_cells")
        conformed = summary.get("budget_conformed_cells")
        route_ok = bool(summary.get("route_expectation_passed", True))
        floor = FLOORS[cell]
        ok = tokens_exact and route_ok and ratio >= floor - NOISE
        rows.append((cell, ratio, floor, "OK" if ok else "FAIL"))
        if not tokens_exact:
            failures.append(f"{cell}: exact_cells {summary['exact_cells']}/{summary['cells']}")
        if not route_ok:
            failures.append(f"{cell}: route expectation failed")
        if ratio < floor - NOISE:
            failures.append(f"{cell}: ratio {ratio:.4f} < floor {floor} - {NOISE}")
        print(
            f"{cell:>5}: mtp {mtp:6.2f} tok/s vs ar {ar:6.2f} tok/s  "
            f"ratio {ratio:.4f} (floor {floor}, {ratio - floor:+.4f})  "
            f"exact {summary['exact_cells']}/{summary['cells']}"
            + (f"  engaged {engaged}/{summary['cells']} conformed {conformed}" if engaged is not None else "")
            + f"  [{rows[-1][3]}]"
        )
    for name in ("c2", "c8"):
        path = out_dir / f"p3-{name}-automatic-k0.json"
        if path.exists():
            d = json.loads(path.read_text())
            summary = d["summary"][KEY_BY_CELL[name]]
            print(
                f"{name} k0 : mtp engaged expected none -> "
                f"engaged {summary.get('engaged_cells')}  "
                f"[{'OK' if summary.get('engaged_cells') == 0 else 'FAIL'}]"
            )
            if summary.get("engaged_cells") != 0:
                failures.append(f"{name} k0: engaged {summary.get('engaged_cells')} != 0")
        else:
            failures.append(f"{name} k0: result JSON missing")
    if failures:
        print("\nGATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll packet 3 retention gates pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
