#!/usr/bin/env python3
"""Decompose C1-C8 complete-wall engine packets into prefill and decode submodules.

The published cross-engine matrices time one boundary: barrier-to-last complete
wall for a whole width wave. That single number mixes prompt processing, first
token, and steady-state decode, so it cannot say *which* submodule is slow.

This collector joins two complete-wall packets per engine -- a one-token arm and
a long arm over the same prompt suite and widths -- and reports, per width:

* ``first_token_wall_ms``: the one-token arm wall per request lane, which owns
  admission + prompt processing + first visible token;
* ``decode_only_tok_s``: the marginal decode rate
  ``(D_long - 1) / (wall_long - wall_d1)`` at the same width, which removes the
  prefill/first-token submodule from the decode measurement;
* ``complete_wall_tok_s`` for both arms, so the published headline is reproduced
  rather than replaced.

Inputs are read-only artifacts. Rates are aggregate generated tokens per second
across the wave, matching the matrix protocol.

Usage:
    python3 scripts/gguf_engine_submodule_decomposition.py \
      --engine hipengine=hipengine:/tmp/he-d1.json:/tmp/he-d24.json \
      --engine current=llamacpp:/tmp/ll-d1.json:/tmp/ll-d24.json \
      --output /tmp/submodule-decomposition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "hipengine.gguf_engine_submodule_decomposition.v1"
KINDS = ("hipengine", "llamacpp")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _provenance(command: Sequence[str]) -> dict[str, Any]:
    """Describe where and from what code this packet was produced.

    The packet this script writes had no provenance at all: no host, no commit, no argv. That is how
    a pre-grouped-prefill decomposition kept being quoted as current - nothing in the file said when
    it was measured, and the correction had to be reconstructed from commit dates. Use the normative
    collector; fall back to cheap, GPU-free facts rather than writing nothing.
    """

    try:
        from hipengine.benchmark.provenance import collect_artifact_provenance

        return collect_artifact_provenance(repo_root=REPO_ROOT, command=list(command))
    except Exception as exc:  # pragma: no cover - no-HIP runners must still get a stamped packet
        import socket

        from hipengine.benchmark.provenance import collect_repo_state

        state = collect_repo_state(REPO_ROOT)
        return {
            "host_name": socket.gethostname(),
            "hipengine_commit": state.get("hipengine_commit"),
            "git_branch": state.get("git_branch"),
            "dirty": state.get("dirty"),
            "command": list(command),
            "provenance_error": f"{type(exc).__name__}: {exc}",
        }



def _lanes_and_wall(cell: Mapping[str, Any]) -> tuple[int, float]:
    rows = cell.get("rows")
    if isinstance(rows, Sequence) and rows:
        lanes = len(rows)
    else:
        lanes = 1
    wall = float(cell.get("wall_seconds") or cell.get("wall") or 0.0)
    return lanes, wall


def _arm_names(arm: str) -> set[str]:
    """Accepted arm names.

    A comma list is accepted because the short (one-token) arm is recorded as
    ``prefill`` by the llama.cpp matrix and as ``ar`` by the hipEngine server
    bench when it is run with ``--max-tokens 1``.
    """

    return {name.strip() for name in str(arm).split(",") if name.strip()}


def _llamacpp_arms(payload: Mapping[str, Any], arm: str) -> dict[int, dict[str, float]]:
    names = _arm_names(arm)
    out: dict[int, dict[str, float]] = {}
    for cell in payload.get("cells", ()):
        if str(cell.get("arm")) not in names:
            continue
        width = int(cell["width"])
        lanes, wall = _lanes_and_wall(cell)
        row = out.setdefault(
            width, {"wall": 0.0, "lanes": 0, "generated": 0, "prompt": 0, "cells": 0}
        )
        row["wall"] += wall
        row["lanes"] += lanes
        row["cells"] += 1
        row["generated"] += sum(int(r.get("tokens_predicted", 0)) for r in cell.get("rows", ()))
        row["prompt"] += sum(int(r.get("tokens_evaluated", 0)) for r in cell.get("rows", ()))
    return out


def _hipengine_arms(payload: Mapping[str, Any], arm: str) -> dict[int, dict[str, float]]:
    names = _arm_names(arm)
    out: dict[int, dict[str, float]] = {}
    for cell in payload.get("cells", ()):
        body = next(
            (
                value
                for key, value in cell.items()
                if isinstance(value, Mapping) and str(value.get("arm")) in names
            ),
            None,
        )
        if body is None:
            continue
        width = int(body["width"])
        lanes, wall = _lanes_and_wall(body)
        row = out.setdefault(
            width, {"wall": 0.0, "lanes": 0, "generated": 0, "prompt": 0, "cells": 0}
        )
        row["wall"] += wall
        row["lanes"] += lanes
        row["cells"] += 1
        row["generated"] += sum(
            int(r.get("usage", {}).get("completion_tokens", 0)) for r in body.get("rows", ())
        )
        row["prompt"] += sum(
            int(r.get("usage", {}).get("prompt_tokens", 0)) for r in body.get("rows", ())
        )
    return out


def _widths(rows: Mapping[int, Mapping[str, float]]) -> list[int]:
    return sorted(int(key) for key in rows)


def _decompose(
    label: str,
    kind: str,
    d1: Mapping[str, Any],
    dlong: Mapping[str, Any],
    *,
    arm: str,
    arm_d1: str | None = None,
) -> dict[str, Any]:
    """Split one engine's wave wall into admission and steady-state decode.

    Two arms of the same workload identify the two terms of the wave model
    ``wall(D, W) ~= admission(W) + (D - 1) * step(W)``: the long-minus-short
    slope gives the cost of one co-scheduled step at that batch, and the short
    arm minus its own interior steps gives what the wave pays before decode
    starts. The published complete rate hides which term is losing, which is why
    the C1-C8 table could show hipEngine behind while its decode was ahead.
    """

    loader = _llamacpp_arms if kind == "llamacpp" else _hipengine_arms
    short = loader(d1, str(arm_d1 or arm))
    long_ = loader(dlong, arm)
    widths = [w for w in _widths(long_) if w in short]
    if not widths:
        raise ValueError(f"engine {label}: no overlapping widths between arms")
    per_width: dict[str, Any] = {}
    for width in widths:
        s, lng = short[width], long_[width]
        if s.get("cells") or lng.get("cells"):
            # Loaders sum the prompt suite; every derived quantity is per wave.
            for row in (s, lng):
                cells = max(int(row.get("cells") or 1), 1)
                row["wall"] /= cells
                row["generated"] /= cells
                row["prompt"] /= cells
                row["lanes"] /= cells
        if s["lanes"] <= 0 or lng["lanes"] <= 0 or s["wall"] <= 0 or lng["wall"] <= 0:
            continue
        width = int(round(s["lanes"]))
        tokens_short = s["generated"] / s["lanes"]
        tokens_long = lng["generated"] / lng["lanes"]
        delta_tokens = lng["generated"] - s["generated"]
        delta_wall = lng["wall"] - s["wall"]
        step_s = delta_wall / max(delta_tokens / width, 1e-9)
        admission_s = s["wall"] - max(tokens_short - 1.0, 0.0) * step_s
        per_width[str(width)] = {
            "lanes": width,
            "prompts_per_arm": int(short.get("cells") or 0),
            "prompt_tokens_per_lane": round(s["prompt"] / s["lanes"], 3),
            "short_tokens_per_lane": round(tokens_short, 3),
            "short_wall_ms": round(1000.0 * s["wall"], 4),
            "long_tokens_per_lane": round(tokens_long, 3),
            "long_wall_ms": round(1000.0 * lng["wall"], 4),
            "published_short_tok_s": round(s["generated"] / s["wall"], 6),
            "published_long_tok_s": round(lng["generated"] / lng["wall"], 6),
            "decode_step_ms": round(1000.0 * step_s, 6),
            "decode_tok_s_aggregate": (
                round(delta_tokens / delta_wall, 6) if delta_wall > 0 else None
            ),
            "admission_ms": round(1000.0 * admission_s, 4),
            "admission_ms_per_lane": round(1000.0 * admission_s / width, 4),
            "long_tok_s_admission_removed": (
                round(lng["generated"] / (lng["wall"] - admission_s), 6)
                if lng["wall"] > admission_s
                else None
            ),
        }
    return {"label": label, "kind": kind, "arm": arm, "widths": per_width}


def _winners(blocks: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    widths = sorted({int(w) for block in blocks for w in block["widths"]}, key=int)
    out: dict[str, Any] = {}
    for width in widths:
        values: dict[str, float | None] = {}
        for block in blocks:
            row = block["widths"].get(str(width))
            values[str(block["label"])] = None if row is None else row.get(metric)
        numeric = {k: v for k, v in values.items() if isinstance(v, (int, float))}
        winner = None
        if numeric:
            higher_is_better = "tok_s" in metric
            key = max if higher_is_better else min
            winner = str(key(numeric, key=lambda k: numeric[k]))
        out[str(width)] = {"values": values, "winner": winner}
    return {"metric": metric, "widths": out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        help="label=kind:d1.json:long.json where kind is hipengine or llamacpp",
    )
    parser.add_argument("--arm", default="ar", help="cell arm key for the long arm (ar or mtp)")
    parser.add_argument(
        "--arm-d1",
        default=None,
        help="cell arm key for the one-token arm; defaults to --arm "
        "(llama.cpp packets label the one-token arm 'prefill')",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    blocks: list[dict[str, Any]] = []
    arm_d1 = args.arm_d1 or args.arm
    for spec in args.engine:
        label, _, rest = spec.partition("=")
        kind, _, paths = rest.partition(":")
        first, _, second = paths.partition(":")
        if kind not in KINDS or not first or not second:
            raise ValueError(f"bad --engine {spec!r}; expected label=kind:d1:long")
        d1 = json.loads(Path(first).read_text(encoding="utf-8"))
        dlong = json.loads(Path(second).read_text(encoding="utf-8"))
        blocks.append(_decompose(label, kind, d1, dlong, arm=args.arm, arm_d1=arm_d1))

    metrics = (
        "published_long_tok_s",
        "decode_step_ms",
        "decode_tok_s_aggregate",
        "admission_ms",
        "admission_ms_per_lane",
        "long_tok_s_admission_removed",
    )
    # Cross-engine what-if: each engine's published long rate with its own
    # admission cost replaced by the cheapest admission measured at that width.
    # This is the decision-relevant number, because admission is the only term
    # the losing engine can copy from the winner without touching arithmetic.
    by_width: dict[str, dict[str, Mapping[str, Any]]] = {}
    for block in blocks:
        for width, row in block["widths"].items():
            by_width.setdefault(width, {})[str(block["label"])] = row
    admission_what_if: dict[str, dict[str, Any]] = {}
    for width, rows in by_width.items():
        admissions = {
            name: float(row["admission_ms"])
            for name, row in rows.items()
            if isinstance(row.get("admission_ms"), (int, float))
        }
        if not admissions:
            continue
        best_name = min(admissions, key=lambda name: admissions[name])
        per_engine: dict[str, Any] = {}
        for name, row in rows.items():
            own = row.get("admission_ms")
            wall = row.get("long_wall_ms")
            generated = row.get("long_tokens_per_lane", 0.0) * int(width) * float(
                row.get("prompts_per_arm") or 1.0
            )
            if not isinstance(own, (int, float)) or not isinstance(wall, (int, float)):
                per_engine[name] = None
                continue
            swapped = float(wall) - float(own) + admissions[best_name]
            per_engine[name] = (
                round(1000.0 * generated / swapped, 6) if swapped > 0 else None
            )
        admission_what_if[width] = {
            "cheapest_admission_engine": best_name,
            "cheapest_admission_ms": admissions[best_name],
            "long_tok_s_if_admission_best": per_engine,
        }

    payload = {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(),
        "provenance": _provenance([sys.executable, *[str(item) for item in sys.argv[1:]]]),
        "arm": args.arm,
        "note": args.note,
        "method": {
            "decode_step_ms": "(long_wall - short_wall) / ((long_gen - short_gen) / width); one co-scheduled step at that batch",
            "admission_ms": "short_wall - (short_tokens_per_lane - 1) * decode_step; what the wave pays before steady decode",
            "admission_ms_per_lane": "admission_ms / width; a flat value across widths means batched admission, a rising one means serialized admission",
            "long_tok_s_admission_removed": "published long rate recomputed with this engine's own admission cost removed; an upper bound on what admission alone could buy",
            "assumption": "admission cost is token-count independent over the arm pair",
            "rate_unit": "aggregate generated tokens per second across the width wave",
        },
        "engines": blocks,
        "admission_what_if": admission_what_if,
        "submodule_winners": {metric: _winners(blocks, metric) for metric in metrics},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("== long_tok_s_if_admission_best (each engine with the cheapest admission at that width)")
    for width in sorted(admission_what_if, key=int):
        row = admission_what_if[width]
        values = "  ".join(
            f"{k}={'-' if v is None else round(v, 3)}"
            for k, v in sorted(row["long_tok_s_if_admission_best"].items())
        )
        print(f"  C{width}: {values}  (cheapest admission: {row['cheapest_admission_engine']})")
    for metric, table in payload["submodule_winners"].items():
        print(f"== {metric}")
        for width, row in table["widths"].items():
            values = "  ".join(
                f"{k}={'-' if v is None else round(v, 3)}" for k, v in sorted(row["values"].items())
            )
            print(f"  C{width}: {values}  -> {row['winner']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
