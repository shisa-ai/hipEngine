#!/usr/bin/env python3
"""Per-role cost table for one Qwen4Exp prefill, joined to real tensor shapes.

A kernel-level profile says ``gguf_k_prefill_out_coltile_rowbatch_kernel``
dominates. It cannot say whether that is an attention projection, a
hyper-connection projection or a shared-expert FFN, because the same variant
serves all of them. This joins two records taken from the same process:

* a role-marked ``rocprofv3`` trace, which gives milliseconds per owner and
  tensor slot path (``qwen4exp_role_analyze.py``);
* the GGUF launch census, which gives the quant type, ``K``, ``N``, row count
  and launch count behind each of those roles.

The join is on the normalized role name (``layers.7.attn_q`` ->
``layers.*.attn_q``), so the output is one row per *kind* of projection rather
than one row per layer.

Reported per role: milliseconds, share of the window, the real shape, achieved
GFLOP/s, and achieved bytes/s for the quantized weight stream. Achieved GFLOP/s
is a measured rate against a stated peak, not an efficiency claim: for a
quantized kernel the dequantization work is not in the FLOP count, so a low
share of peak does not by itself prove an inefficient kernel. The tile geometry
that drives the reuse per decoded weight is reported alongside so the mechanism
is visible rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# Weight bytes per stored element, from the GGUF block layouts.
BYTES_PER_ELEMENT = {
    "gguf_q8_0": 34 / 32,
    "gguf_q4_k": 144 / 256,
    "gguf_q5_k": 176 / 256,
    "gguf_q6_k": 210 / 256,
    "gguf_q5_1": 24 / 32,
}

# W7900-class FP32 peak is irrelevant here; this is the gfx1151 host. Stated so
# the share-of-peak column has an explicit denominator rather than a silent one.
DEFAULT_PEAK_GFLOPS = 14_800.0

_NORMALIZE = re.compile(r"layers\.\d+\.")


def normalize(role: str) -> str:
    return _NORMALIZE.sub("layers.*.", role)


def load(role_analysis: Path, census: Path) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    analysis = json.loads(role_analysis.read_text())
    roles: dict[str, float] = defaultdict(float)
    for row in analysis["exact_roles"]:
        roles[normalize(row["name"])] += float(row["ms"])

    shapes: dict[str, dict[str, Any]] = {}
    for row in json.loads(census.read_text())["rows"]:
        key = normalize(row["role"])
        entry = shapes.setdefault(key, {
            "quant": row["quant"],
            "in_features": row["in_features"],
            "out_features": row["out_features"],
            "launches": 0,
            "rows_per_launch": row["rows"],
        })
        entry["launches"] += int(row["launches"])
    return dict(roles), shapes


def build_table(
    roles: dict[str, float], shapes: dict[str, dict[str, Any]], peak_gflops: float
) -> list[dict[str, Any]]:
    total_ms = sum(roles.values())
    table = []
    for role, ms in roles.items():
        shape = shapes.get(role)
        row: dict[str, Any] = {
            "role": role,
            "ms": round(ms, 1),
            "share_of_window_pct": round(100.0 * ms / total_ms, 2),
        }
        if shape is None:
            # The role exists in the trace but issues no GGUF matmul through the
            # census hook, so its shape is not recoverable from these two files.
            row["shape_source"] = "unavailable"
            table.append(row)
            continue
        k, n = shape["in_features"], shape["out_features"]
        launches = shape["launches"]
        rows_per = shape["rows_per_launch"]
        flops = 2.0 * rows_per * k * n * launches
        bytes_moved = BYTES_PER_ELEMENT.get(shape["quant"], 0.0) * k * n * launches
        row.update({
            "shape_source": "launch_census",
            "quant": shape["quant"],
            "in_features": k,
            "out_features": n,
            "launches": launches,
            "rows_per_launch": rows_per,
            "gflop": round(flops / 1e9, 1),
            "achieved_gflops": round(flops / (ms / 1000.0) / 1e9, 1) if ms > 0 else 0.0,
            "share_of_peak_pct": (
                round(100.0 * flops / (ms / 1000.0) / 1e9 / peak_gflops, 1)
                if ms > 0 and peak_gflops > 0
                else 0.0
            ),
            "weight_traffic_gbytes": round(bytes_moved / 1e9, 2),
            # Counts every re-read, so this can exceed DRAM bandwidth when a
            # layer's weights stay resident in cache across prefill chunks. It
            # is a traffic figure, not a measured memory-bandwidth utilisation.
            "weight_traffic_gbytes_per_s": (
                round(bytes_moved / (ms / 1000.0) / 1e9, 1) if ms > 0 else 0.0
            ),
        })
        table.append(row)
    table.sort(key=lambda r: -r["ms"])
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-analysis", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-gflops", type=float, default=DEFAULT_PEAK_GFLOPS)
    parser.add_argument("--min-ms", type=float, default=20.0)
    args = parser.parse_args()

    roles, shapes = load(args.role_analysis, args.census)
    table = build_table(roles, shapes, args.peak_gflops)
    total_ms = sum(roles.values())

    width = 34
    print(f"{'role':{width}s} {'ms':>8s} {'%':>6s} {'K':>7s} {'N':>7s} {'n':>5s} "
          f"{'GFLOP':>9s} {'GFLOP/s':>9s} {'%peak':>6s} {'TB/s':>7s}")
    print("-" * (width + 68))
    for row in table:
        if row["ms"] < args.min_ms:
            continue
        if row["shape_source"] == "unavailable":
            print(f"{row['role']:{width}s} {row['ms']:8.1f} {row['share_of_window_pct']:5.1f}%"
                  f"{'shape not in census':>46s}")
            continue
        print(f"{row['role']:{width}s} {row['ms']:8.1f} {row['share_of_window_pct']:5.1f}% "
              f"{row['in_features']:7d} {row['out_features']:7d} {row['launches']:5d} "
              f"{row['gflop']:9.1f} {row['achieved_gflops']:9.1f} "
              f"{row['share_of_peak_pct']:5.1f}% {row['weight_traffic_gbytes_per_s']:7.2f}")
    print("-" * (width + 68))
    print(f"{'TOTAL':{width}s} {total_ms:8.1f} 100.0%")

    args.output.write_text(json.dumps({
        "schema": 1,
        "kind": "qwen4exp_per_role_cost",
        "performance_claim": False,
        "notes": [
            "Milliseconds come from a role-marked rocprofv3 trace with 100% kernel attribution.",
            "Shapes come from the launch census taken in the same process.",
            "A role with no census entry issues no matmul through the census hook; its shape is not recoverable here.",
            "Achieved GFLOP/s excludes dequantization work, so share-of-peak is not an efficiency verdict on its own.",
            "Weight traffic counts every prefill-chunk re-read, so it can exceed DRAM bandwidth when a layer stays cache-resident; it is not a bandwidth measurement.",
        ],
        "peak_gflops": args.peak_gflops,
        "total_ms": round(total_ms, 1),
        "roles": table,
    }, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
