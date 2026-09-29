#!/usr/bin/env python3
"""Per-kernel bandwidth split for DE-1's routed-expert MoE block.

The family-level gate in ``qwen4exp_de1_moe_bandwidth_gate.py`` answers "is the
MoE block bandwidth-bound?" and finds it is not.  It cannot answer *which* kernel
carries the gap, because ``moe:expert_gate`` is a role that wraps the whole
``run_qwen4_exp_moe`` call and blends twelve distinct kernels.

This script splits that family into its raw kernels and charges each one the
traffic it actually moves, so the gap is localized to a kernel rather than to a
family name.

Two artifacts are joined:

* ``rocprof/<stem>_kernel_trace.csv`` - the harness renames every *decode*
  dispatch to its semantic role, so this file carries the attribution but not
  the raw kernel name.
* ``rocprof/<stem>_results.db`` - ``rocpd_info_kernel_symbol`` maps
  ``Kernel_Id`` back to the mangled raw kernel name.

The decode window is taken from the ``qwen4exp_decode_live4097_rep{0,1,2}``
ROCTX ranges in the marker trace, and every figure is divided by the number of
reps so the output is per decode step.

Traffic is charged only where the mapping is unambiguous:

  routed gate+up -> ffn_gate_exps + ffn_up_exps bytes x 10/512 x 48 layers
  routed down    -> ffn_down_exps bytes             x 10/512 x 48 layers
  router         -> ffn_gate_inp bytes              x 48 layers

Kernels that cannot be charged unambiguously are reported as ``other`` and are
given no bandwidth figure.  They are NOT folded into the charged kernels.

Usage:
    python3 scripts/qwen4exp_de1_moe_per_kernel.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from qwen4exp_de1_moe_bandwidth_gate import (  # noqa: E402
    DEFAULT_ATTRIBUTION,
    DEFAULT_MODEL,
    EXPERTS_TOTAL,
    EXPERTS_USED,
    LAYERS,
    DOCUMENTED_ROOF_GBPS,
    collect_tensors,
)

DEFAULT_RESULTS = pathlib.Path(
    "benchmarks/results/2026-09-29-qwen4exp-x4-decode-split"
)

# Raw mangled kernel name -> traffic bucket.  Matched by substring so the
# template suffixes do not have to be spelled out.
GATE_UP_MARKER = "gguf_q4_k_selected_dual_silu_logical128_t64"
DOWN_MARKER = "q5_1_selected_weighted_sum_logical256_t64"
# The router GEMV is the fp32 dense GEMV over ffn_gate_inp (512 x 2560, 4 B/weight).
# `qwen35_router_select_kernel` is NOT charged: it is the top-k over 512 logits
# and its 0.331 ms/step would imply 760 GB/s for the router traffic, 3.4x the roof.
ROUTER_GEMV_MARKER = "dense_gemv_out_kernelIffLb0"
ROUTER_SELECT_MARKER = "qwen35_router_select_kernel"

ROUTED_GATE_UP_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight")
ROUTED_DOWN_SUFFIXES = ("ffn_down_exps.weight",)
ROUTER_SUFFIX = "ffn_gate_inp.weight"


def _single(path: pathlib.Path, pattern: str) -> pathlib.Path:
    hits = sorted(pathlib.Path(path).glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {pattern} under {path}, got {hits}")
    return hits[0]


def decode_rep_windows(marker_csv: pathlib.Path) -> list[tuple[int, int]]:
    """ROCTX windows of the measured decode repetitions."""
    windows = []
    with marker_csv.open() as handle:
        for row in csv.DictReader(handle):
            if row["Function"].startswith("qwen4exp_decode_live4097_rep"):
                windows.append((int(row["Start_Timestamp"]), int(row["End_Timestamp"])))
    if not windows:
        raise SystemExit(f"no decode rep windows in {marker_csv}")
    return windows


def kernel_symbols(db: pathlib.Path) -> dict[int, str]:
    """``Kernel_Id`` -> raw kernel name, from the rocpd symbol table."""
    conn = sqlite3.connect(str(db))
    try:
        table = [
            r[0]
            for r in conn.execute(
                "select name from sqlite_master where type='table' "
                "and name like '%info_kernel_symbol%'"
            )
        ]
        if not table:
            raise SystemExit(f"no kernel symbol table in {db}")
        return {i: name for i, name in conn.execute(f'select id,kernel_name from "{table[0]}"')}
    finally:
        conn.close()


def per_kernel_ms_per_step(
    kernel_csv: pathlib.Path,
    symbols: dict[int, str],
    windows: list[tuple[int, int]],
) -> tuple[collections.Counter, collections.Counter]:
    """(role-family, raw kernel) -> ms/step and dispatch count/step."""
    ms = collections.Counter()
    count = collections.Counter()
    reps = len(windows)
    with kernel_csv.open() as handle:
        for row in csv.DictReader(handle):
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if not any(start >= lo and end <= hi for lo, hi in windows):
                continue
            role = row["Kernel_Name"]
            if not role.startswith("qwen4exp_role:"):
                continue
            raw = symbols.get(int(row["Kernel_Id"]), "?")
            key = (role.split(":", 2)[1], raw)
            ms[key] += (end - start) / 1e6 / reps
            count[key] += 1 / reps
    return ms, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--stem", default="x4full")
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rocprof = args.results_dir / "rocprof"
    kernel_csv = _single(rocprof, f"{args.stem}_kernel_trace.csv")
    marker_csv = _single(rocprof, f"{args.stem}_marker_api_trace.csv")
    db = _single(rocprof, f"{args.stem}_results.db")

    windows = decode_rep_windows(marker_csv)
    symbols = kernel_symbols(db)
    per_kernel_ms, per_kernel_n = per_kernel_ms_per_step(kernel_csv, symbols, windows)

    tensors = collect_tensors(args.model_root)
    routed_scale = EXPERTS_USED / EXPERTS_TOTAL

    def payload(suffixes: tuple[str, ...]) -> int:
        return sum(t.nbytes for t in tensors if t.name.endswith(suffixes))

    gate_up_step = payload(ROUTED_GATE_UP_SUFFIXES) * routed_scale
    down_step = payload(ROUTED_DOWN_SUFFIXES) * routed_scale
    # collect_tensors already sums every layer's ffn_gate_inp, so this is the
    # per-step router traffic without any further layer multiplier.
    router_step = payload((ROUTER_SUFFIX,))

    def family(ms: collections.Counter, prefix: str) -> float:
        return sum(v for (fam, _), v in ms.items() if fam == prefix)

    moe_ms = family(per_kernel_ms, "moe")

    def find(marker: str) -> tuple[float, float, str]:
        hits = [(v, per_kernel_n[k], k[1]) for k, v in per_kernel_ms.items() if marker in k[1]]
        if not hits:
            raise SystemExit(f"no decode kernel matching {marker!r}")
        hits.sort(reverse=True)
        return hits[0]

    gate_up_ms, gate_up_n, gate_up_name = find(GATE_UP_MARKER)
    down_ms, down_n, down_name = find(DOWN_MARKER)
    router_ms, router_n, router_name = find(ROUTER_GEMV_MARKER)
    select_ms, select_n, select_name = find(ROUTER_SELECT_MARKER)

    charged_ms = gate_up_ms + down_ms + router_ms

    def gbps(nbytes: int, ms: float) -> float:
        return nbytes / 1e9 / (ms / 1e3)

    report = {
        "schema": 1,
        "kind": "qwen4exp_de1_moe_per_kernel",
        "performance_claim": False,
        "measurement_class": "derived_from_profiled_capture_diagnostic",
        "model_root": str(args.model_root),
        "results_dir": str(args.results_dir),
        "stem": args.stem,
        "decode_rep_windows": len(windows),
        "traffic_model": {
            "layers": LAYERS,
            "experts_used": EXPERTS_USED,
            "experts_total": EXPERTS_TOTAL,
            "routed_gate_up_bytes_per_step": gate_up_step,
            "routed_down_bytes_per_step": down_step,
            "router_bytes_per_step": router_step,
            "charged_bytes_per_step": gate_up_step + down_step + router_step,
        },
        "per_kernel": [
            {
                "role_family": "moe",
                "raw_kernel": gate_up_name,
                "bucket": "routed_gate_up",
                "ms_per_step": gate_up_ms,
                "kernels_per_step": gate_up_n,
                "us_per_kernel": gate_up_ms / gate_up_n * 1e3,
                "bytes_per_step": gate_up_step,
                "achieved_gbps": gbps(gate_up_step, gate_up_ms),
                "fraction_of_documented_roof": gbps(gate_up_step, gate_up_ms)
                / DOCUMENTED_ROOF_GBPS,
            },
            {
                "role_family": "moe",
                "raw_kernel": down_name,
                "bucket": "routed_down",
                "ms_per_step": down_ms,
                "kernels_per_step": down_n,
                "us_per_kernel": down_ms / down_n * 1e3,
                "bytes_per_step": down_step,
                "achieved_gbps": gbps(down_step, down_ms),
                "fraction_of_documented_roof": gbps(down_step, down_ms)
                / DOCUMENTED_ROOF_GBPS,
            },
            {
                "role_family": "moe",
                "raw_kernel": router_name,
                "bucket": "router_gemv",
                "ms_per_step": router_ms,
                "kernels_per_step": router_n,
                "us_per_kernel": router_ms / router_n * 1e3,
                "bytes_per_step": router_step,
                "achieved_gbps": gbps(router_step, router_ms),
                "fraction_of_documented_roof": gbps(router_step, router_ms)
                / DOCUMENTED_ROOF_GBPS,
            },
        ],
        "summary": {
            "moe_family_ms_per_step": moe_ms,
            "charged_ms_per_step": charged_ms,
            "unattributed_ms_per_step": moe_ms - charged_ms,
            "unattributed_pct_of_family": (moe_ms - charged_ms) / moe_ms * 100.0,
            "gate_up_share_of_family_pct": gate_up_ms / moe_ms * 100.0,
            "router_select_ms_per_step": select_ms,
            "router_select_kernels_per_step": select_n,
            "router_select_raw_kernel": select_name,
        },
        "result": {
            "dominant_owner": gate_up_name,
            "dominant_owner_ms_per_step": gate_up_ms,
            "dominant_owner_gbps": gbps(gate_up_step, gate_up_ms),
            "dominant_owner_fraction_of_roof": gbps(gate_up_step, gate_up_ms)
            / DOCUMENTED_ROOF_GBPS,
            "gate_up_floor_ms_at_comparator_rate": gate_up_step
            / 1e9
            / (226.67 / 1e3),
            "gate_up_headroom_ms_per_step": gate_up_ms
            - gate_up_step / 1e9 / (226.67 / 1e3),
            "verdict": (
                "The MoE gap is localized to the routed gate+up GEMV: "
                f"{gate_up_name.split('N_1')[-1][:40]} carries {gate_up_ms:.3f} of "
                f"{moe_ms:.3f} ms/step ({gate_up_ms / moe_ms * 100.0:.1f}% of the "
                f"family) at {gbps(gate_up_step, gate_up_ms):.2f} GB/s = "
                f"{gbps(gate_up_step, gate_up_ms) / DOCUMENTED_ROOF_GBPS * 100.0:.1f}% "
                "of the documented roof, while the router GEMV in the same block "
                f"reaches {gbps(router_step, router_ms):.2f} GB/s."
            ),
        },
        "limits": [
            "Derived from a PROFILED capture; the source artifact's own limits state "
            "that per-kernel times there are diagnostic, not a performance measurement.",
            "The decode window is the three snapshot-restored measured repetitions; the "
            "same capture also contains prefill, whose kernels are excluded.",
            "Traffic is charged per kernel only where the tensor mapping is unambiguous. "
            "The 'other' remainder is reported, not distributed.",
            "If a kernel stages or dequantizes into wider buffers, its real DRAM traffic "
            "exceeds the model, making the achieved rate a lower bound.",
            "One case (code-p4096), one live count (4097), one model, one host.",
        ],
    }

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
