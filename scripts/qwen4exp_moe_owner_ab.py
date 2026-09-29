#!/usr/bin/env python3
"""Time the registered owners of a routed-MoE route against each other.

Two routes have more than one registered owner, and in both cases the owners have never
been measured against each other:

``gate-up`` (``moe_linear`` / ``prefill_rows_ge64_exact_grouped_q4_gate_up``):

* production default ``selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out``, one output
  column per lane, each lane reading its own weight row;
* strict default and production fallback
  ``selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out``, cooperative LDS
  metadata staging with the weight row reused across an expert's rows.

``down`` (``moe_linear`` / ``prefill_rows_ge512_k640_exact_grouped_q5_1_down``):

* production default ``selected_grouped_prefill_pair2_row_publish_bf16_bf16_out``;
* strict default and production fallback
  ``selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out``;
* ``selected_wmma_iu8_risk_prefill_bf16_bf16_out``, the family the 2026-09-29 ED-1
  artifacts screened for LDS staging. ED-1 compared staged prototypes against this owner
  but never against the owner production actually selects for this shape.

This is a main-kernel timing comparison. It sets ``max_risks=0`` on wmma arms so the risk
queue is never written, matching ``qwen4exp_moe_main_kernel_cost.py``: it times the
arithmetic, not the risk bookkeeping, and it does not time a repair pass or engine work. It
makes no correctness claim and changes no default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "_mkc", REPO / "scripts" / "qwen4exp_moe_main_kernel_cost.py"
)
_mkc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mkc)

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_host_to_device,
    host_array_ptr,
    malloc,
)

EXPERTS = _mkc.EXPERTS
HIDDEN = _mkc.HIDDEN
FFN = _mkc.FFN
QK_K = _mkc.QK_K
Q4_K_BLOCK_BYTES = _mkc.Q4_K_BLOCK_BYTES
WMMA_TILE_ROWS = _mkc.WMMA_TILE_ROWS

# Per-route geometry. Gate/up consumes the hidden state and produces two FFN-wide tensors;
# down consumes the FFN-width expert output and produces the hidden state.
ROUTES = {
    "gate-up": {
        "in_features": HIDDEN,
        "out_features": FFN,
        "dual": True,
        "quant": "gguf_q4_k",
        "row_bytes": (HIDDEN // QK_K) * Q4_K_BLOCK_BYTES,
        "owners": [
            "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
        ],
    },
    "down": {
        "in_features": FFN,
        "out_features": HIDDEN,
        "dual": False,
        "quant": "gguf_q5_1",
        "row_bytes": (FFN // 32) * _mkc.Q5_1_BLOCK_BYTES,
        "owners": [
            "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_wmma_iu8_risk_prefill_bf16_bf16_out",
        ],
    },
}


def _down_launchers(gu: Any, dn: Any, route: dict[str, Any]) -> dict[str, Any]:
    """Bind the down-route owners. All share one Python signature."""

    def grouped(module: Any, name: str) -> Any:
        return getattr(module, f"qwen4_exp_q5_1_{name}")

    return {
        "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out": (
            lambda *a: grouped(dn, "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out")(*a)
        ),
        "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out": (
            lambda *a: grouped(
                dn, "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out"
            )(*a)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=sorted(ROUTES), default="gate-up")
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--rows", default="512,1024,4096,10240")
    parser.add_argument("--distribution", default="balanced,sparse")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--tile-rows", type=int, default=WMMA_TILE_ROWS)
    args = parser.parse_args()

    import numpy as np

    route = ROUTES[args.route]
    in_features = route["in_features"]
    out_features = route["out_features"]
    row_bytes = route["row_bytes"]
    weight_bytes = EXPERTS * out_features * row_bytes

    runtime = get_hip_runtime()

    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as gu
    from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as dn

    if route["quant"] == "gguf_q4_k":
        gu.build_gguf_q4_k_selected_prefill(load=True)
    dn.build_qwen4_exp_q5_1(load=True)

    rows_list = [int(v) for v in args.rows.split(",") if v.strip()]
    distributions = [v for v in args.distribution.split(",") if v.strip()]

    results: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_moe_owner_ab",
        "performance_claim": False,
        "route": args.route,
        "quant": route["quant"],
        "owners": route["owners"],
        "question": (
            f"Among the registered owners of the {args.route} route, which is fastest on "
            "the main kernel at the real geometry, and is that comparison already archived?"
        ),
        "protocol": {
            "experts": EXPERTS,
            "in_features": in_features,
            "out_features": out_features,
            "row_bytes": row_bytes,
            "dual": route["dual"],
            "wmma_tile_rows": args.tile_rows,
            "repetitions": args.repetitions,
            "max_risks": 0,
            "timing": "median of per-launch HIP event pairs, 2 primed launches",
            "weights": (
                "synthesized at the real shapes and byte layout; the access pattern and "
                "byte counts are representative, the values are not"
            ),
            "scope": (
                "main kernel only; no risk queue writes, no repair pass, no packing, no "
                "engine work, no correctness claim"
            ),
        },
        "rows": {},
    }

    max_rows = max(rows_list)
    max_padded = max_rows + args.tile_rows
    x = malloc(max_padded * in_features * 2)
    out_dual = malloc(max_padded * 2 * out_features * 2)
    out_a = malloc(max_rows * out_features * 2)
    out_b = malloc(max_rows * out_features * 2)
    w_a = malloc(weight_bytes)
    w_b = malloc(weight_bytes) if route["dual"] else None
    for buffer, pattern in (
        (x, 0x2D),
        (out_dual, 0x00),
        (out_a, 0x00),
        (out_b, 0x00),
        (w_a, 0x3C),
    ):
        runtime.memset(buffer.ptr, pattern, min(int(buffer.nbytes), 64 << 20))
    if w_b is not None:
        runtime.memset(w_b.ptr, 0x5A, min(int(w_b.nbytes), 64 << 20))

    def measure(rows: int, distribution: str) -> dict[str, Any]:
        counts = _mkc._counts_for(rows, EXPERTS, distribution, seed=args.seed)
        if sum(counts) != rows:
            raise SystemExit("count synthesis failed")
        compact, padded, tile_expert, padded_rows = _mkc._maps(counts, args.tile_rows)

        compact_dev = malloc(np.asarray(compact, dtype=np.int64).nbytes)
        padded_dev = malloc(np.asarray(padded, dtype=np.int64).nbytes)
        tile_dev = malloc(max(8, np.asarray(tile_expert, dtype=np.int64).nbytes))
        for device, values in (
            (compact_dev, compact),
            (padded_dev, padded),
            (tile_dev, tile_expert or [0]),
        ):
            host = np.asarray(values, dtype=np.int64)
            copy_host_to_device(device, host_array_ptr(host), host.nbytes)
        risk_count = malloc(4)
        risk_indices = malloc(4)
        runtime.memset(risk_count.ptr, 0, 4)

        if route["quant"] == "gguf_q4_k":

            def wmma() -> None:
                gu.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out(
                    x.ptr, compact_dev.ptr, padded_dev.ptr, tile_dev.ptr,
                    w_a.ptr, w_b.ptr, out_dual.ptr,
                    risk_count.ptr, risk_indices.ptr, 0, _mkc.RISK_MULTIPLIER,
                    rows, in_features, out_features, out_features, EXPERTS, padded_rows,
                )

            def grouped() -> None:
                gu.gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out(
                    x.ptr, compact_dev.ptr, w_a.ptr, w_b.ptr,
                    out_a.ptr, out_b.ptr, rows, EXPERTS, in_features, out_features,
                )

            arms = {
                "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out": wmma,
                "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out": grouped,
            }
            primary = "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"
        else:

            def pair2() -> None:
                dn.qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out(
                    x.ptr, compact_dev.ptr, w_a.ptr, out_a.ptr,
                    rows, EXPERTS, in_features, out_features,
                )

            def compact8() -> None:
                dn.qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out(
                    x.ptr, compact_dev.ptr, w_a.ptr, out_a.ptr,
                    rows, EXPERTS, in_features, out_features,
                )

            def wmma_down() -> None:
                dn.qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
                    x.ptr, compact_dev.ptr, padded_dev.ptr, tile_dev.ptr,
                    w_a.ptr, out_a.ptr,
                    risk_count.ptr, risk_indices.ptr, 0, _mkc.RISK_MULTIPLIER,
                    rows, in_features, out_features, EXPERTS, padded_rows,
                )

            arms = {
                "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out": pair2,
                "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out": compact8,
                "selected_wmma_iu8_risk_prefill_bf16_bf16_out": wmma_down,
            }
            primary = "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"

        # Fixed order, with the primary arm repeated last to expose drift across the sweep.
        order = [name for name in route["owners"] if name in arms]
        medians: dict[str, float] = {}
        for name in order:
            medians[name] = _mkc._time_launch(arms[name], runtime, args.repetitions)
        medians[f"{primary}__repeated_last"] = _mkc._time_launch(
            arms[primary], runtime, args.repetitions
        )

        primary_ms = medians[primary]
        return {
            "distribution": distribution,
            "rows": rows,
            "padded_rows": padded_rows,
            "active_experts": sum(1 for c in counts if c),
            "max_expert_rows": max(counts),
            "medians_ms": {k: round(v, 4) for k, v in medians.items()},
            "vs_primary_pct": {
                k: round(100.0 * (v - primary_ms) / primary_ms, 4)
                for k, v in medians.items()
                if not k.endswith("__repeated_last")
            },
            "primary_repeat_drift_pct": round(
                100.0 * (medians[f"{primary}__repeated_last"] - primary_ms) / primary_ms, 4
            ),
        }

    for distribution in distributions:
        for rows in rows_list:
            key = f"{distribution}-{rows}"
            print(f"  {key} ...", flush=True)
            results["rows"][key] = measure(rows, distribution)

    primary = route["owners"][0]
    others = [o for o in route["owners"] if o != primary]
    summary: dict[str, Any] = {
        "cases": len(results["rows"]),
        "primary": primary,
        "max_abs_repeat_drift_pct": round(
            max(abs(r["primary_repeat_drift_pct"]) for r in results["rows"].values()), 4
        ),
        "reading": (
            "vs_primary_pct is positive when that owner is SLOWER than the primary owner"
        ),
    }
    for other in others:
        deltas = [r["vs_primary_pct"][other] for r in results["rows"].values()]
        summary[other] = {
            "slower_cases": sum(1 for d in deltas if d > 0),
            "faster_cases": sum(1 for d in deltas if d < 0),
            "median_pct": round(statistics.median(deltas), 4),
            "min_pct": round(min(deltas), 4),
            "max_pct": round(max(deltas), 4),
        }
    results["summary"] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
