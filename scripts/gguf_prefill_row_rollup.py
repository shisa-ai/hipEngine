#!/usr/bin/env python3
"""Roll up the wave-admission prefill row from a GGUF C1-C8 server packet.

Protocol (normative for this row): the 10-prompt K3/LL-320 suite at
``--max-tokens 1``, so a wave's wall is its admission plus one decode step. The row is a
**prompt-token admission rate**: per cell it contributes ``prompt_tokens x width`` tokens
against a wall of ``width / ar.tok_s`` seconds (the cell rate is lanes-per-second, one
completion token per lane, so inverting it and multiplying by width recovers the wave wall).
The per-width figure is ``mean(tokens) / mean(wall)`` over that width's cells and the
aggregate is ``sum(tokens) / sum(wall)`` over all cells, which weights wide waves by their
lane count. A C1 wave admits its prompt once, so ``requests`` counts cells (80 = 8 widths x
10 prompts), not completions.

Two properties are asserted rather than eyeballed, because both have been
reported carelessly here before:

* **Cross-session drift** is only meaningful between two runs of the *same*
  protocol on the *same* model. The prior packet is therefore compared on the
  protocol hash and the model path, and an unequal prior is reported as a
  session-to-session change rather than drift unless ``--strict-prior`` makes it
  an error.
* **A prior whose packet status is not ``ok`` is not comparable in both
  directions.** The published d1 prior fails only because of an MTP-engagement
  expectation the one-token protocol cannot satisfy, so its AR admission walls
  are valid while its aggregate status is not. Such a prior yields a one-sided
  check, and ``--strict-prior`` refuses it outright.

The functions here are importable so tools do not re-implement the row and the
guards separately; ``main`` prints them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

MAX_ADMISSION_SECONDS = 30.0
STATUS_OK = "ok"


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        packet = json.load(handle)
    if "protocol" not in packet or "cells" not in packet:
        raise ValueError(f"{path}: not a C1-C8 server packet")
    return packet


def protocol_key(packet: dict) -> tuple[str, str]:
    """Identity of a packet for comparability: protocol hash and model path."""
    return (str(packet["protocol"].get("protocol_sha256")), str(packet["protocol"].get("model")))


def aggregate(packet: dict) -> dict:
    """Return the aggregate prompt-admission rate, per-width rates, and refusals.

    A cell whose observed width differs from its expected width is excluded: its
    wall measures a wave the protocol did not ask for. A cell with no ``ar.tok_s``
    rate or no ``prompt_tokens`` is refused and named rather than guessed at, and
    ``requests`` counts accepted cells.
    """
    per_width: dict[str, list[dict[str, float]]] = {}
    all_tokens: list[float] = []
    all_walls: list[float] = []
    refused: list[str] = []
    for cell in packet.get("cells", []):
        width = int(cell["width"])
        ar = cell.get("ar") or {}
        if int(ar.get("observed_width", width)) != width:
            refused.append(f"C{width}:{cell.get('prompt_id')}:observed_width")
            continue
        rate = ar.get("tok_s")
        usage = next((row.get("usage") for row in ar.get("rows") or [] if row.get("usage")), None)
        prompt_tokens = (usage or {}).get("prompt_tokens")
        if not rate or float(rate) <= 0.0 or not prompt_tokens or int(prompt_tokens) <= 0:
            refused.append(f"C{width}:{cell.get('prompt_id')}"
                           f":rate={rate},prompt_tokens={prompt_tokens}")
            continue
        # cell rate = lanes / wall with one completion token per lane.
        tokens = float(prompt_tokens) * width
        wall = width / float(rate)
        per_width.setdefault(f"C{width}", []).append({"tokens": tokens, "wall": wall})
        all_tokens.append(tokens)
        all_walls.append(wall)
    groups = {
        width: {
            "tok_per_s": sum(item["tokens"] for item in rows) / sum(item["wall"] for item in rows),
            "seconds": sum(item["wall"] for item in rows) / len(rows),
            "cells": len(rows),
        }
        for width, rows in per_width.items()
    }
    total_wall = sum(all_walls)
    return {
        "protocol_sha256": str(packet["protocol"].get("protocol_sha256")),
        "model": str(packet["protocol"].get("model")),
        "packet_status": packet.get("status"),
        "aggregate_tok_per_s": sum(all_tokens) / total_wall if total_wall else 0.0,
        "seconds": total_wall / len(all_walls) if all_walls else 0.0,
        "requests": len(all_walls),
        "per_width": groups,
        "excluded_cells": refused,
    }


def identity(row: dict) -> tuple[str, str]:
    """Comparability identity of a rolled-up packet: protocol hash and model path."""
    return (row["protocol_sha256"], row["model"])


def drift(current: dict, prior: dict) -> tuple[float | None, bool, bool]:
    """Return ``(drift_fraction, comparable_both_ways, prior_status_ok)``.

    ``comparable_both_ways`` requires equal protocol identity *and* an ``ok``
    status on the prior; without the latter the mismatch direction is missing,
    which is a one-sided check and must be labelled as one.
    """
    same_protocol = identity(current) == identity(prior)
    prior_ok = prior.get("packet_status") == STATUS_OK
    if not (same_protocol and prior_ok):
        return None, same_protocol and prior_ok, prior_ok
    before = prior["aggregate_tok_per_s"]
    if not before:
        return None, False, prior_ok
    return abs(current["aggregate_tok_per_s"] - before) / before, True, prior_ok


def spread(current: dict, prior: dict) -> tuple[float | None, dict[str, float]]:
    """Return the max per-width delta and the per-width deltas.

    The published repeatability figure is a *max per-width* spread, not an aggregate delta:
    two packets can agree on the aggregate to 0.1% while one width moves 0.4%, and the band
    is meant to catch the latter. Reporting only the aggregate hides exactly the case the
    band exists for.
    """
    deltas = {}
    for width, row in current["per_width"].items():
        before = prior["per_width"].get(width, {}).get("tok_per_s")
        if before:
            deltas[width] = 100.0 * (row["tok_per_s"] / before - 1.0)
    if not deltas:
        return None, {}
    return max(abs(value) for value in deltas.values()), deltas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packet", help="packet JSON produced by scripts/gguf_mtp_c1c8_server_bench.py")
    ap.add_argument("--prior", help="prior packet for the cross-session drift check")
    ap.add_argument(
        "--strict-prior",
        action="store_true",
        help="fail if the --prior packet is not comparable in both directions",
    )
    ap.add_argument(
        "--prior-config-changed",
        action="store_true",
        help="the prior packet differs by configuration (an A/B pair), so report the delta as "
             "an effect size and skip the same-protocol drift band",
    )
    args = ap.parse_args(argv)

    agg = aggregate(load(args.packet))
    if not agg["requests"]:
        print("no comparable AR rows in packet", file=sys.stderr)
        return 2

    line = (f"lane-weighted {agg['aggregate_tok_per_s']:8.3f} tok/s over mean wave wall "
            f"{agg['seconds']:7.3f}s across {agg['requests']} cells")
    if agg["excluded_cells"]:
        shown = ", ".join(agg["excluded_cells"][:3])
        line += f"  [refused {len(agg['excluded_cells'])} cells: {shown}]"
    print(line + "  (lane-weighted is not the published row)")
    widths = sorted(agg["per_width"], key=lambda key: int(key[1:]))
    rates = [agg["per_width"][key]["tok_per_s"] for key in widths]
    for width in sorted(agg["per_width"], key=lambda key: int(key[1:])):
        print(f"  {width:>3} {agg['per_width'][width]['tok_per_s']:9.3f} tok/s")
    if len(rates) >= 2:
        print(f"flatness C1->Cmax: {100.0 * (rates[-1] / rates[0] - 1.0):+.2f}%")

    if args.prior:
        prior = aggregate(load(args.prior))
        prior_ok = prior.get("packet_status") == STATUS_OK
        same_protocol = identity(agg) == identity(prior)
        max_spread, deltas = spread(agg, prior)
        if args.prior_config_changed:
            if max_spread is None:
                print("no overlapping widths with --prior; cannot report an effect size",
                      file=sys.stderr)
                return 3
            print(f"max delta vs prior packet: {max_spread:.2f}% (different configuration; this "
                  f"is an effect size, not drift)")
            print("  per-width delta: " + " ".join(
                f"{k}:{v:+.1f}%" for k, v in sorted(
                    deltas.items(), key=lambda item: int(item[0][1:]))[:8]))
            if not same_protocol:
                print("  note: protocol hash and/or model differ, so only the declared config "
                      "delta may be cited")
            return 0
        frac, two_sided, _ = drift(agg, prior)
        if frac is not None:
            print(f"cross-session drift vs prior: {max_spread:.2f}% max per-width "
                  f"({frac * 100:.2f}% aggregate)")
        elif args.strict_prior:
            why = "packet status is not ok" if not prior_ok else "protocol or model differs"
            print(f"--strict-prior: prior packet is not comparable ({why})", file=sys.stderr)
            return 3
        else:
            print(f"cross-session drift vs prior: {max_spread:.2f}% max per-width, but NOT "
                  f"two-sided (prior packet status is not ok or protocol differs); AR admission "
                  f"walls may still be compared, and a configuration change must be reported "
                  f"with --prior-config-changed rather than as drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
