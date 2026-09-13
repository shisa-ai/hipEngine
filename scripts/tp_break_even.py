#!/usr/bin/env python3
"""TP2 break-even projection from measured collective latency and TP1 baselines.

This is the Packet 0 go/no-go input for Packet 3. It does not measure an engine;
it answers one question with measured numbers on both sides:

    how much per-token collective time can TP2 afford before it stops winning?

The model is deliberately small and its assumption is explicit:

    token_time = fixed + weights / bandwidth

``fixed`` covers attention over the KV cache, the GDN recurrence, launch
overhead and everything else that does not shrink when weights are halved. Only
its *share* of the TP1 token time is assumed, and the projection is reported for
a range of shares, so no single guess decides the verdict. ``bandwidth`` is not
assumed either: it is implied by the TP1 measurement itself
(``(1 - fixed_share) * token_time = weights / bandwidth``).

Break-even collective budget:

    C* = (1 - fixed_share) * T1 * (1 - rank_weight_fraction)

A measured collective cost below ``C*`` means TP2 is still ahead. The measured
cost is the same-host chain-ladder number from ``tp2_graph_capture_probe.json``.

Usage:
    python3 scripts/tp_break_even.py \
        --tp1 "W7900=27.9:15.652:8.646" --tp1 "XTX=29.82:15.652:7.009" \
        --collective-ms 1.3,1.5 --json benchmarks/results/tp2_break_even.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXED_SHARES = (0.0, 0.1, 0.2, 0.3)


def parse_tp1(spec: str) -> dict[str, Any]:
    """Parse ``NAME=TOK_S:TP1_GIB:RANK_GIB``.

    ``TP1_GIB`` is the whole-model weight bytes one GPU reads per token at
    TP1 (what TP2 halves), and ``RANK_GIB`` is the rank-local shard this GPU
    would read at TP2.
    """

    if "=" not in spec or spec.count(":") != 2:
        raise ValueError(f"expected NAME=TOK_S:TP1_GIB:RANK_GIB, got {spec!r}")
    name, rest = spec.split("=", 1)
    tok_s_text, tp1_text, rank_text = rest.split(":")
    name = name.strip()
    if not name:
        raise ValueError(f"missing device name in {spec!r}")
    tok_s = float(tok_s_text)
    tp1_gib = float(tp1_text)
    rank_gib = float(rank_text)
    if tok_s <= 0 or tp1_gib <= 0 or rank_gib <= 0:
        raise ValueError(f"values must be positive in {spec!r}")
    if rank_gib > tp1_gib:
        raise ValueError(f"rank shard {rank_gib} exceeds TP1 weights {tp1_gib} in {spec!r}")
    return {"name": name, "tok_s": tok_s, "tp1_gib": tp1_gib, "rank_gib": rank_gib}


def project(device: dict[str, Any], *, collective_ms: float, fixed_share: float) -> dict[str, Any]:
    """Project TP2 for one device with one fixed-cost assumption."""

    tp1_ms = 1000.0 / float(device["tok_s"])
    weight_share = 1.0 - float(fixed_share)
    fixed_ms = tp1_ms * float(fixed_share)
    implied_bandwidth_gbs = (device["tp1_gib"] * weight_share) / (tp1_ms / 1000.0)
    rank_weight_fraction = float(device["rank_gib"]) / float(device["tp1_gib"])
    rank_weight_ms = tp1_ms * weight_share * rank_weight_fraction
    tp2_ms = fixed_ms + rank_weight_ms + float(collective_ms)
    break_even_ms = tp1_ms * weight_share * (1.0 - rank_weight_fraction)
    return {
        "device": device["name"],
        "fixed_share": float(fixed_share),
        "tp1_ms_per_token": tp1_ms,
        "fixed_ms_per_token": fixed_ms,
        "implied_bandwidth_gbs": implied_bandwidth_gbs,
        "rank_weight_fraction": rank_weight_fraction,
        "rank_weight_ms_per_token": rank_weight_ms,
        "collective_ms_per_token": float(collective_ms),
        "tp2_ms_per_token": tp2_ms,
        "projected_speedup": tp1_ms / tp2_ms,
        "break_even_collective_ms": break_even_ms,
        "headroom_factor": break_even_ms / float(collective_ms) if collective_ms > 0 else None,
        "tp2_tok_s": 1000.0 / tp2_ms,
    }


def build_report(
    devices: list[dict[str, Any]],
    *,
    collective_ms: tuple[float, ...],
    fixed_shares: tuple[float, ...] = DEFAULT_FIXED_SHARES,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "tp2_break_even",
        "model": "token_time = fixed + weights / bandwidth; bandwidth implied by the TP1 row",
        "assumption": (
            "fixed cost (attention, GDN recurrence, launch overhead) does not shrink with "
            "tensor parallelism; only its share of the TP1 token time is assumed and the "
            "projection is reported across a range of shares"
        ),
        "collective_ms_measured": list(collective_ms),
        "devices": [],
        "errors": [],
    }
    for device in devices:
        entry: dict[str, Any] = {
            "device": device["name"],
            "tp1_tok_s": device["tok_s"],
            "tp1_weights_gib": device["tp1_gib"],
            "tp2_rank_weights_gib": device["rank_gib"],
            "rows": [],
        }
        for share in fixed_shares:
            for collective in collective_ms:
                entry["rows"].append(project(device, collective_ms=collective, fixed_share=share))
        entry["worst_case_speedup"] = min(row["projected_speedup"] for row in entry["rows"])
        entry["best_case_speedup"] = max(row["projected_speedup"] for row in entry["rows"])
        entry["minimum_headroom_factor"] = min(row["headroom_factor"] for row in entry["rows"])
        report["devices"].append(entry)
    report["verdict"] = {
        "target_speedup": 1.3,
        "passes_target_in_every_row": all(
            entry["worst_case_speedup"] >= 1.3 for entry in report["devices"]
        ),
        "collective_would_have_to_be_worse_by": min(
            entry["minimum_headroom_factor"] for entry in report["devices"]
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tp1",
        action="append",
        required=True,
        metavar="NAME=TOK_S:TP1_GIB:RANK_GIB",
        help="same-host TP1 measurement and its TP2 rank shard; repeatable",
    )
    parser.add_argument(
        "--collective-ms",
        default="1.3,1.5",
        help="measured per-token collective cost, comma list (default: the chain-ladder range)",
    )
    parser.add_argument(
        "--fixed-share",
        default=",".join(str(share) for share in DEFAULT_FIXED_SHARES),
        help="assumed fixed-cost share of the TP1 token time, comma list",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the JSON artifact here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    devices = [parse_tp1(spec) for spec in args.tp1]
    collective_ms = tuple(float(chunk) for chunk in args.collective_ms.split(",") if chunk.strip())
    fixed_shares = tuple(float(chunk) for chunk in args.fixed_share.split(",") if chunk.strip())
    report = build_report(devices, collective_ms=collective_ms, fixed_shares=fixed_shares)

    if not args.quiet:
        print(f"{'device':8s} {'fixed':>6s} {'coll_ms':>8s} {'tp2_ms':>8s} {'speedup':>8s} {'break_even':>10s} {'headroom':>9s}")
        for entry in report["devices"]:
            for row in entry["rows"]:
                print(
                    f"{row['device']:8s} {row['fixed_share'] * 100:5.0f}% {row['collective_ms_per_token']:8.2f} "
                    f"{row['tp2_ms_per_token']:8.2f} {row['projected_speedup']:8.2f}x "
                    f"{row['break_even_collective_ms']:10.2f} {row['headroom_factor']:8.1f}x"
                )
        verdict = report["verdict"]
        print(
            f"verdict: every row >= {verdict['target_speedup']}x = {verdict['passes_target_in_every_row']}; "
            f"collectives would have to be {verdict['collective_would_have_to_be_worse_by']:.1f}x worse to break even"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
