"""Aggregate the Packet 6 K1-K7 x C1-C8 diagnostic grid sweep.

Reads the per-cell probe JSON artifacts under /tmp/he-bettermtp-raw/packet6
(written by scripts/qwen38_packet5_k4_watchdog_probe.py) and emits:

* the full 56-cell grid table (AR tok/s, MTP tok/s, ratio, engaged/exact/
  budget-conformed counts per cell),
* per-width best-depth selection with losing depths recorded explicitly,
* capacity-8 realized-width observations from the per-prompt rows.

Selection is reported from measured cells only; the qualified-depth policy
decision is recorded separately in the worklog. Diagnostic evidence only —
the retained economics reproduction at the selected cell goes through the
canonical retained harness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = Path("/tmp/he-bettermtp-raw/packet6")
WIDTHS = range(1, 9)
BUDGETS = range(1, 8)


def main() -> int:
    rows: list[dict] = []
    missing: list[str] = []
    for width in WIDTHS:
        for budget in BUDGETS:
            path = OUT_DIR / f"k4-w{width}-b{budget}-probe.json"
            summary_path = OUT_DIR / f"k4-w{width}-b{budget}-probe-summary.json"
            if not path.exists() or not summary_path.exists():
                missing.append(f"w{width}-b{budget}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "complete":
                missing.append(f"w{width}-b{budget}({summary.get('status')})")
                continue
            probe = json.loads(path.read_text(encoding="utf-8"))
            cell = probe["summary"].get(str(width))
            if cell is None:
                missing.append(f"w{width}-b{budget}(no summary row)")
                continue
            rows.append(
                {
                    "width": width,
                    "budget": budget,
                    "ar_tok_s": cell["ar"]["tok_s"],
                    "mtp_tok_s": cell["mtp"]["tok_s"],
                    "ratio": cell["mtp_vs_ar_ratio"],
                    "engaged": cell["engaged_cells"],
                    "exact": cell["exact_cells"],
                    "budget_conformed": cell["budget_conformed_cells"],
                    "cells": cell["cells"],
                }
            )
    if missing:
        print("MISSING/INCOMPLETE CELLS:", ", ".join(missing), file=sys.stderr)
        return 1

    # Full grid table.
    print(
        f"{'W':>2} {'K':>2} {'AR tok/s':>9} {'MTP tok/s':>9} "
        f"{'ratio':>6} {'eng':>4} {'exact':>5} {'bconf':>5}"
    )
    for r in sorted(rows, key=lambda r: (r["width"], r["budget"])):
        print(
            f"{r['width']:>2} {r['budget']:>2} {r['ar_tok_s']:>9.2f} "
            f"{r['mtp_tok_s']:>9.2f} {r['ratio']:>6.3f} {r['engaged']:>4} "
            f"{r['exact']:>5} {r['budget_conformed']:>5}"
        )

    # Per-width best depth by MTP tok/s (engaged+exact+budget-conformed=cells).
    print("\nPer-width best depth (by MTP tok/s, all cells engaged+exact+bconf):")
    best: dict[int, dict] = {}
    for width in WIDTHS:
        candidates = [
            r
            for r in rows
            if r["width"] == width
            and r["engaged"] == r["cells"]
            and r["exact"] == r["cells"]
            and r["budget_conformed"] == r["cells"]
        ]
        if not candidates:
            print(f"  W{width}: no fully-engaged cell")
            continue
        winner = max(candidates, key=lambda r: r["mtp_tok_s"])
        best[width] = winner
        losers = sorted(
            (r for r in candidates if r is not winner),
            key=lambda r: -r["mtp_tok_s"],
        )
        loser_txt = ", ".join(
            f"K{r['budget']}={r['mtp_tok_s']:.1f}({r['ratio']:.2f})"
            for r in losers[:3]
        )
        print(
            f"  W{width}: K{winner['budget']} {winner['mtp_tok_s']:.1f} tok/s "
            f"(ratio {winner['ratio']:.3f}, AR {winner['ar_tok_s']:.1f})"
            + (f" | next: {loser_txt}" if losers else "")
        )

    # Capacity-8 realized-width note: per-prompt widths from the width-8 rows.
    realized: dict[int, int] = {}
    for budget in BUDGETS:
        path = OUT_DIR / f"k4-w8-b{budget}-probe.json"
        if not path.exists():
            continue
        probe = json.loads(path.read_text(encoding="utf-8"))
        for cell in probe["cells"]:
            for row in cell["mtp"].get("rows", ()):  # realized group rows
                width_seen = row.get("width")
                if width_seen is not None:
                    realized[width_seen] = realized.get(width_seen, 0) + 1
    if realized:
        print("\nCapacity-8 realized group widths (across K1-K7 MTP rows):")
        for width in sorted(realized):
            print(f"  rows={width}: {realized[width]}")

    artifact = {
        "kind": "packet6-grid-sweep-aggregate",
        "diagnostic_only": True,
        "cells": rows,
        "per_width_best": {
            str(w): b for w, b in best.items()
        },
    }
    out = OUT_DIR / "grid-aggregate.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nartifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
