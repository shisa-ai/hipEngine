#!/usr/bin/env python3
"""Compare MTP draft acceptance across C1-C8 packets, per prompt and per width.

The published matrices report throughput only, so a change in *acceptance* - how many
drafted tokens the verifier keeps - is invisible in them. Acceptance is the lever that
decides whether speculative decoding beats AR at all, and it can regress while every
cell stays content-exact: a rejected draft just falls back to the verified token, so
output equality cannot see it. Only acceptance telemetry can.

Reads two or more ``gguf_mtp_c1c8_server_bench.py`` packets (read-only) and prints, per
prompt, the accepted/drafted counts at each width, plus the per-width aggregate. Use it
to A/B a verify or draft change: ``--baseline`` is the reference packet and every other
label is diffed against it.

Usage:
    python3 scripts/gguf_mtp_acceptance_compare.py \
        --baseline native=/tmp/he-c1c8-k3/c1c8-k3-packet.json \
        serial_exact=/tmp/he-k3-serialexact/k3-serialexact.json \
        [--widths 1,2,4,8] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _acceptance(payload: Mapping[str, Any], arm: str) -> dict[tuple[str, int], dict[str, int]]:
    """Return {(prompt_id, width): accepted/drafted/cycles/tokens} for one arm."""
    out: dict[tuple[str, int], dict[str, int]] = {}
    for cell in payload.get("cells", ()):
        arm_block = cell.get(arm)
        if not isinstance(arm_block, dict):
            continue
        rows = arm_block.get("rows") or ()
        stats = {
            "accepted": 0,
            "drafted": 0,
            "cycles": 0,
            "lanes": 0,
            "tokens": 0,
        }
        for row in rows:
            mtp = row.get("mtp")
            if not isinstance(mtp, dict) or not mtp.get("used"):
                continue
            stats["accepted"] += int(mtp.get("accepted_draft_tokens") or 0)
            stats["drafted"] += int(mtp.get("draft_tokens") or 0)
            stats["cycles"] += int(mtp.get("draft_cycles") or 0)
            stats["tokens"] += int(mtp.get("accepted_draft_tokens") or 0)
            stats["lanes"] += 1
        if stats["lanes"]:
            out[(str(cell.get("prompt_id")), int(cell["width"]))] = stats
        arm_block_tokens = int(arm_block.get("generated_tokens") or 0)
        if arm_block_tokens:
            stats["tokens"] = arm_block_tokens
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, metavar="LABEL=PACKET.json")
    ap.add_argument("comparisons", nargs="*", metavar="LABEL=PACKET.json")
    ap.add_argument("--arm", default="mtp")
    ap.add_argument("--widths", default="", help="comma list, default: all present")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    def parse(spec: str) -> tuple[str, dict[str, Any]]:
        label, sep, path = spec.partition("=")
        if not sep:
            ap.error(f"--{label}: expected LABEL=PACKET.json")
        return label, json.loads(Path(path).read_text(encoding="utf-8"))

    base_label, base_payload = parse(args.baseline)
    others = [parse(spec) for spec in args.comparisons]
    base = _acceptance(base_payload, args.arm)
    if not base:
        print(f"baseline {base_label}: no {args.arm} acceptance rows", file=sys.stderr)
        return 2
    widths = (
        sorted({w for _, w in base})
        if not args.widths
        else sorted(int(w) for w in args.widths.split(",") if w.strip())
    )
    prompts = sorted({p for p, _ in base})

    for label, payload in [(base_label, base_payload)] + others:
        stats = _acceptance(payload, args.arm)
        acc = sum(s["accepted"] for (p, w), s in stats.items() if w in widths)
        dft = sum(s["drafted"] for (p, w), s in stats.items() if w in widths)
        rate = acc / dft if dft else 0.0
        per_lane = acc / max(sum(s["cycles"] for (p, w), s in stats.items() if w in widths), 1)
        print(f"{label}: accepted {acc}/{dft} = {rate:.4f}  (accepted per cycle {per_lane:.3f})")
    print()

    header = " ".join(f"C{w:<12}" for w in widths)
    print(f"{'prompt':26} {header}")
    for prompt in prompts:
        cells = []
        for w in widths:
            s = base.get((prompt, w))
            if not s or not s["drafted"]:
                cells.append("     -/----    ")
                continue
            rate = s["accepted"] / s["drafted"]
            cells.append(f"{s['accepted']:>4}/{s['drafted']:<4}={rate:.2f} ")
        print(f"{prompt[:26]:26} {' '.join(cells)}")

    if others:
        print("\nper-width aggregate rate vs baseline (positive = comparison accepts more):")
        for label, payload in others:
            stats = _acceptance(payload, args.arm)
            parts = []
            for w in widths:
                b = [s for (p, ww), s in base.items() if ww == w]
                c = [s for (p, ww), s in stats.items() if ww == w]
                if not b or not c or not sum(s["drafted"] for s in b) or not sum(s["drafted"] for s in c):
                    parts.append(f"C{w}=   n/a ")
                    continue
                rb = sum(s["accepted"] for s in b) / sum(s["drafted"] for s in b)
                rc = sum(s["accepted"] for s in c) / sum(s["drafted"] for s in c)
                parts.append(f"C{w}={rc - rb:+.3f} ")
            print(f"  {label:16} {' '.join(parts)}")

    if args.json_out:
        out = {
            "schema": "hipengine.mtp_acceptance_compare.v1",
            "arm": args.arm,
            "baseline": base_label,
            "widths": widths,
            "per_prompt_per_width": {
                f"{p}|{w}": base.get((p, w)) for p in prompts for w in widths if (p, w) in base
            },
            "packets": {label: str(Path(spec.partition('=')[2])) for label, spec in
                        [args.baseline, *args.comparisons]},
        }
        Path(args.json_out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
