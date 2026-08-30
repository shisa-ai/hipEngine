#!/usr/bin/env python3
"""Roll hipEngine's prefill row into the C1-C8 matrix convention.

The published peer prefill rows are the mean(prompt_tokens x width) / mean(wall)
over a width's ten one-token cells -- validated by reproducing llama.cpp
current's published row exactly at all eight widths. This tool applies the same
arithmetic to hipEngine's one-token packet and prints a README-ready row.

A one-token packet cannot exercise K3 (nothing to verify), so it is normally run
without ``--expected-mtp-widths``; the tool refuses any packet that reports
``status != complete`` and reports the drift against an optional prior packet so
a published row is never quietly swapped between sessions.

Usage:
    python3 scripts/gguf_prefill_row_rollup.py HE_D1_PACKET.json \
        [--prior HE_D1_PACKET_PREV.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

# Prompt token counts per prompt id, read from the C1-C8 packet's per-row usage.
# Verified against packet 0740a9497 (`usage.prompt_tokens`).
PUBLISHED_PEER_PREFILL = {
    1: (200.946, 195.803),
    2: (239.658, 231.307),
    3: (259.036, 252.893),
    4: (281.828, 274.520),
    5: (323.043, 316.169),
    6: (366.213, 358.053),
    7: (374.207, 368.136),
    8: (424.072, 424.202),
}


def _rows(packet: dict) -> dict[int, float]:
    cells: dict[int, list[dict]] = {}
    for cell in packet.get("cells", []):
        cells.setdefault(int(cell["width"]), []).append(cell)
    out: dict[int, float] = {}
    for width, group in cells.items():
        numerator: list[float] = []
        walls: list[float] = []
        for cell in group:
            arm = cell.get("ar") or {}
            tok_s = arm.get("tok_s")
            usage = None
            for row in arm.get("rows") or []:
                usage = row.get("usage")
                if usage:
                    break
            if not tok_s or not usage or not usage.get("prompt_tokens"):
                raise ValueError(f"width {width}: missing ar rate or prompt_tokens")
            # One token per lane: cell rate = width / wall.
            numerator.append(float(usage["prompt_tokens"]) * width)
            walls.append(width / float(tok_s))
        out[width] = st.mean(numerator) / st.mean(walls)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packet", type=Path)
    ap.add_argument("--prior", type=Path, default=None)
    args = ap.parse_args(argv)

    packet = json.loads(Path(args.packet).read_text(encoding="utf-8"))
    status, passed = packet.get("status"), packet.get("passed")
    if status != "complete" or not passed:
        print(
            f"REFUSING: packet reports status={status!r} passed={passed!r}; "
            "run the one-token arm without --expected-mtp-widths",
            file=sys.stderr,
        )
        return 2

    current = _rows(packet)
    prior = _rows(json.loads(Path(args.prior).read_text(encoding="utf-8"))) if args.prior else {}
    cells = ["| hipEngine prefill "]
    for width in range(1, 9):
        value = current.get(width)
        if value is None:
            print(f"REFUSING: width {width} missing", file=sys.stderr)
            return 2
        cells.append(f"**{value:.3f}** |" if value >= max(PUBLISHED_PEER_PREFILL[width]) else f"{value:.3f} |")
    print(" ".join(cells))
    print()
    print(f"{'C':>2} {'hipEngine':>9} {'current':>8} {'laurent':>8} {'vs cur':>7} "
          f"{'vs prior':>9}")
    for width in range(1, 9):
        value = current[width]
        peer_cur, peer_lau = PUBLISHED_PEER_PREFILL[width]
        drift = f"{100 * (value / prior[width] - 1):+8.2f}%" if width in prior else "        -"
        print(f"{width:>2} {value:9.3f} {peer_cur:8.3f} {peer_lau:8.3f} "
              f"{value / peer_cur:6.2f}x {drift:>9}")
    if prior:
        spread = max(abs(current[w] / prior[w] - 1) for w in current if w in prior) * 100
        print(f"\nmax cross-session spread vs prior packet: {spread:.2f}%")
        if spread > 2.0:
            print("WARNING: spread exceeds the ~1% same-protocol band; do not publish "
                  "the row from this pair without re-measuring.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
