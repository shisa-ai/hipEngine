#!/usr/bin/env python3
"""Time the two registered owners of the Q4_K routed gate/up route against each other.

The manifest has two designs for ``moe_linear`` /
``prefill_rows_ge64_exact_grouped_q4_gate_up``:

* production default ``selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out``
  (``gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_kernel``), which assigns one output
  column per lane and has each lane read its own weight row;
* strict default and production fallback
  ``selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out``
  (``gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel<8,4,true>``), which stages the
  Q4_K metadata cooperatively into LDS and reuses a weight row across the rows of one
  expert.

The archived promotion evidence for the production default
(``2026-09-07-framework-qwen4exp-q4-iu8-exact-production.json``) compares it against
``selected_gemv_bf16_bf16_out``, not against the grouped owner. This script measures the
missing pair directly, on the same weights, activations, populations and timing method.

This is a main-kernel timing comparison. It sets ``max_risks=0`` on the wmma arm so the
risk queue is never written, matching ``qwen4exp_moe_main_kernel_cost.py``: it times the
arithmetic, not the risk bookkeeping, and it does not time the repair pass or any engine
work. It makes no correctness claim and changes no default.
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

from hipengine.core.memory import (  # noqa: E402
    copy_host_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.core.hip import get_hip_runtime  # noqa: E402

EXPERTS = _mkc.EXPERTS
HIDDEN = _mkc.HIDDEN
FFN = _mkc.FFN
QK_K = _mkc.QK_K
Q4_K_BLOCK_BYTES = _mkc.Q4_K_BLOCK_BYTES
WMMA_TILE_ROWS = _mkc.WMMA_TILE_ROWS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--rows", default="512,1024,4096,10240")
    parser.add_argument("--distribution", default="balanced,skewed")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--tile-rows", type=int, default=WMMA_TILE_ROWS)
    args = parser.parse_args()

    import numpy as np

    runtime = get_hip_runtime()

    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as gu

    gu.build_gguf_q4_k_selected_prefill(load=True)

    row_bytes = (HIDDEN // QK_K) * Q4_K_BLOCK_BYTES
    weight_bytes = EXPERTS * FFN * row_bytes

    rows_list = [int(v) for v in args.rows.split(",") if v.strip()]
    distributions = [v for v in args.distribution.split(",") if v.strip()]

    results: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_gate_up_owner_ab",
        "performance_claim": False,
        "question": (
            "Between the two registered owners of the Q4_K routed gate/up route - the "
            "production default (dual wmma iu8-risk, one output column per lane) and the "
            "strict default / production fallback (dual grouped rowbatch8, cooperative "
            "LDS metadata staging) - which is faster on the main kernel at the real "
            "geometry, and is that comparison already archived?"
        ),
        "protocol": {
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "ffn": FFN,
            "row_bytes": row_bytes,
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
    gate_x = malloc(max_padded * HIDDEN * 2)
    gate_out = malloc(max_padded * 2 * FFN * 2)
    out_a = malloc(max_rows * FFN * 2)
    out_b = malloc(max_rows * FFN * 2)
    gate_a = malloc(weight_bytes)
    gate_b = malloc(weight_bytes)
    for buffer, pattern in (
        (gate_x, 0x2D),
        (gate_out, 0x00),
        (out_a, 0x00),
        (out_b, 0x00),
        (gate_a, 0x3C),
        (gate_b, 0x5A),
    ):
        runtime.memset(buffer.ptr, pattern, min(int(buffer.nbytes), 64 << 20))

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

        def wmma_launch() -> None:
            gu.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out(
                gate_x.ptr,
                compact_dev.ptr,
                padded_dev.ptr,
                tile_dev.ptr,
                gate_a.ptr,
                gate_b.ptr,
                gate_out.ptr,
                risk_count.ptr,
                risk_indices.ptr,
                0,
                _mkc.RISK_MULTIPLIER,
                rows,
                HIDDEN,
                FFN,
                FFN,
                EXPERTS,
                padded_rows,
            )

        def grouped_launch() -> None:
            gu.gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out(
                gate_x.ptr,
                compact_dev.ptr,
                gate_a.ptr,
                gate_b.ptr,
                out_a.ptr,
                out_b.ptr,
                rows,
                EXPERTS,
                HIDDEN,
                FFN,
            )

        wmma_ms = _mkc._time_launch(wmma_launch, runtime, args.repetitions)
        grouped_ms = _mkc._time_launch(grouped_launch, runtime, args.repetitions)
        # Interleave a repeat of the first arm last, to catch drift across the sweep.
        wmma_repeat_ms = _mkc._time_launch(wmma_launch, runtime, args.repetitions)

        return {
            "distribution": distribution,
            "rows": rows,
            "padded_rows": padded_rows,
            "active_experts": sum(1 for c in counts if c),
            "max_expert_rows": max(counts),
            "arms": {
                "dual_wmma_iu8_risk_prefill": {
                    "median_ms": round(wmma_ms, 4),
                    "repeated_last_ms": round(wmma_repeat_ms, 4),
                },
                "dual_grouped_rowbatch8_out4_expertgrid64": {
                    "median_ms": round(grouped_ms, 4),
                },
            },
            "grouped_vs_wmma_pct": round(
                100.0 * (grouped_ms - wmma_ms) / wmma_ms, 4
            ),
            "wmma_repeat_drift_pct": round(
                100.0 * (wmma_repeat_ms - wmma_ms) / wmma_ms, 4
            ),
        }

    for distribution in distributions:
        for rows in rows_list:
            key = f"{distribution}-{rows}"
            print(f"  {key} ...", flush=True)
            results["rows"][key] = measure(rows, distribution)

    # Summary: which arm wins, and by how much, over every measured case.
    deltas = [r["grouped_vs_wmma_pct"] for r in results["rows"].values()]
    drifts = [abs(r["wmma_repeat_drift_pct"]) for r in results["rows"].values()]
    results["summary"] = {
        "cases": len(deltas),
        "grouped_slower_cases": sum(1 for d in deltas if d > 0),
        "grouped_faster_cases": sum(1 for d in deltas if d < 0),
        "grouped_vs_wmma_pct_median": round(statistics.median(deltas), 4),
        "grouped_vs_wmma_pct_min": round(min(deltas), 4),
        "grouped_vs_wmma_pct_max": round(max(deltas), 4),
        "max_abs_repeat_drift_pct": round(max(drifts), 4),
        "reading": (
            "positive grouped_vs_wmma_pct means the grouped owner is slower than the "
            "production default"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print()
    print(json.dumps(results["summary"], indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
