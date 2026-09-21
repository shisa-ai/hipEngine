"""What sets the cost of the routed-MoE main kernels on the shipped geometry.

The gate/up main kernel is the largest single owner in a ``code-p4096`` prefill
(2710.3 ms of 16673.2 ms, `benchmarks/results/2026-09-17-iu8-repair-unroll/`)
and the down kernel is the second (1484.5 ms). Both run at a small fraction of
either roofline: gate/up reads about 45.5 GB of expert weights per prefill at
16.8 GB/s on a host whose achievable bandwidth is roughly 200 GB/s, and its
67 GFLOP is nowhere near the WMMA peak.

This packet measures which of the candidate explanations holds, on the shipped
kernels and the shipped geometry (512 experts, hidden 2560, ffn 640, 10240 rows
= 1024 tokens x top-10), with synthesized weights at the real byte layout:

* **padding amplification** - each expert's rows are padded to a 16-row WMMA
  tile, so an expert holding 20 rows costs two weight reads, not one. The
  ``--distribution`` sweep varies rows per expert from one to all of them and
  reports achieved bytes/s against the weight bytes actually read.
* **fixed vs proportional cost** - the ``--rows`` sweep separates launch and
  per-row terms.
* **occupancy** - ``--grids`` overrides the column-block count, which is the
  only launch-geometry knob the kernel exposes.

Read the result as "where the time goes on this shape", not as a rate: the
weights are synthesized, so values are not representative, while byte counts,
access patterns and the parallelism response are.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc

# The model's MoE geometry, from the shipped model-shape map
# (benchmarks/results/2026-09-17-qwen4exp-shipped-default-attribution/model-shapes.json).
EXPERTS = 512
HIDDEN = 2560
FFN = 640
ROWS = 10240  # 1024 tokens x top_k 10, one prefill chunk
QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q5_1_BLOCK_BYTES = 24
# Q4_K is 4.52 bits/weight on the shipped mix, Q5_1 is 6.26; these are the
# measured bytes-per-weight from the model-shape map.
Q4_K_BYTES_PER_WEIGHT = 0.5651
Q5_1_BYTES_PER_WEIGHT = 0.7826
WMMA_TILE_ROWS = 16
# The shipped criterion's multiplier. ``max_risks`` is 0 here, so the queue is
# never written: this packet times the arithmetic, not the risk bookkeeping.
RISK_MULTIPLIER = 0.5


def _q4_k_row_bytes(in_features: int) -> int:
    return (in_features // QK_K) * Q4_K_BLOCK_BYTES


def _q5_1_row_bytes(in_features: int) -> int:
    return (in_features // 32) * Q5_1_BLOCK_BYTES


def _balanced_counts(rows: int, experts: int, *, seed: int) -> list[int]:
    """Rows per expert as the router actually leaves them: uneven, all hit."""

    import random

    rng = random.Random(seed)
    weights = [rng.random() + 0.5 for _ in range(experts)]
    total = sum(weights)
    counts = [max(1, int(round(w / total * rows))) for w in weights]
    drift = rows - sum(counts)
    index = 0
    while drift != 0:
        counts[index % experts] += 1 if drift > 0 else -1
        drift += -1 if drift > 0 else 1
        index += 1
    return counts


def _counts_for(rows: int, experts: int, distribution: str, *, seed: int) -> list[int]:
    if distribution == "balanced":
        return _balanced_counts(rows, experts, seed=seed)
    if distribution == "uniform":
        # Every expert exactly the same number of rows.
        base, extra = divmod(rows, experts)
        return [base + (1 if index < extra else 0) for index in range(experts)]
    if distribution == "single":
        # Perfect routing: one expert owns every row. This is the weight-traffic
        # floor for the same arithmetic - each weight byte read exactly once.
        return [rows] + [0] * (experts - 1)
    if distribution == "sparse":
        # A 1024-token chunk cannot reach every expert if routing is skewed;
        # only a quarter of the experts take rows, so the others cost nothing.
        active = max(1, experts // 4)
        base, extra = divmod(rows, active)
        counts = [base + (1 if index < extra else 0) for index in range(active)]
        return counts + [0] * (experts - active)
    raise ValueError(f"unknown distribution {distribution!r}")


def _maps(counts: list[int]) -> tuple[list[int], list[int], list[int], int]:
    """Compact starts, 16-row-padded starts, and the per-tile expert map."""

    compact = [0]
    padded = [0]
    tile_expert: list[int] = []
    for expert, count in enumerate(counts):
        compact.append(compact[-1] + count)
        if count > 0:
            tiles = (count + WMMA_TILE_ROWS - 1) // WMMA_TILE_ROWS
            tile_expert.extend([expert] * tiles)
            padded.append(padded[-1] + tiles * WMMA_TILE_ROWS)
        else:
            padded.append(padded[-1])
    return compact, padded, tile_expert, padded[-1]


def _time_launch(launch: Any, runtime: Any, repetitions: int) -> float:
    """Median of per-launch HIP event pairs, after two primed launches."""

    for _ in range(2):
        launch()
    runtime.device_synchronize()
    samples: list[float] = []
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        for _ in range(repetitions):
            runtime.event_record(start)
            launch()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            samples.append(runtime.event_elapsed_time_ms(start, stop))
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--rows", default=str(ROWS), help="Comma-separated row counts.")
    parser.add_argument(
        "--distribution",
        default="balanced",
        help="Comma-separated: balanced, uniform, single, sparse.",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--grids",
        default="",
        help="Override the gate/up column-block count, e.g. '10,20'.",
    )
    args = parser.parse_args()

    import numpy as np

    runtime = get_hip_runtime()

    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as gu
    from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as dn

    gu.build_gguf_q4_k_selected_prefill(load=True)
    dn.build_qwen4_exp_q5_1(load=True)

    gate_row_bytes = _q4_k_row_bytes(HIDDEN)
    down_row_bytes = _q5_1_row_bytes(FFN)
    gate_weight_bytes = EXPERTS * FFN * gate_row_bytes
    down_weight_bytes = EXPERTS * HIDDEN * down_row_bytes
    gate_out_total = 2 * FFN

    rows_list = [int(value) for value in args.rows.split(",") if value.strip()]
    distributions = [value for value in args.distribution.split(",") if value.strip()]
    grid_override = [int(value) for value in args.grids.split(",") if value.strip()]

    results: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_moe_main_kernel_cost",
        "question": (
            "What sets the cost of the routed-MoE gate/up and down main kernels: "
            "per-expert WMMA tile padding, fixed launch cost, or something the "
            "launch geometry does not move?"
        ),
        "protocol": {
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "ffn": FFN,
            "wmma_tile_rows": WMMA_TILE_ROWS,
            "repetitions": args.repetitions,
            "risk_multiplier": RISK_MULTIPLIER,
            "max_risks": 0,
            "note_on_risk_queue": (
                "max_risks is 0, so the risk queue is never written; the "
                "engine's own capture includes the queue writes"
            ),
            "timing": "median of per-launch HIP event pairs, 2 primed launches",
            "weights": (
                "synthesized at the real shapes and byte layout; the access "
                "pattern and byte counts are representative, the values are not"
            ),
            "weight_bytes_per_tensor": {
                "gate_up": gate_weight_bytes,
                "down": down_weight_bytes,
            },
            "bytes_per_weight": {
                "gate_up_q4_k": Q4_K_BYTES_PER_WEIGHT,
                "down_q5_1": Q5_1_BYTES_PER_WEIGHT,
            },
        },
        "rows": {},
    }

    # One activation buffer and one output buffer at the largest padded size,
    # reused across shapes: the kernel only reads and writes, so a bigger buffer
    # cannot change the timing of a smaller launch.
    max_padded = max(rows_list) + WMMA_TILE_ROWS
    gate_x = malloc(max_padded * HIDDEN * 2)
    gate_out = malloc(max_padded * gate_out_total * 2)
    down_x = malloc(max_padded * FFN * 2)
    down_out = malloc(max_padded * HIDDEN * 2)
    gate_a = malloc(gate_weight_bytes)
    gate_b = malloc(gate_weight_bytes)
    down_w = malloc(down_weight_bytes)
    for buffer, pattern in (
        (gate_x, 0x2D),
        (gate_out, 0x00),
        (down_x, 0x3B),
        (down_out, 0x00),
        (gate_a, 0x3C),
        (gate_b, 0x5A),
        (down_w, 0x6E),
    ):
        runtime.memset(buffer.ptr, pattern, min(int(buffer.nbytes), 64 << 20))

    def measure(
        rows: int, distribution: str, grid: int | None
    ) -> dict[str, Any]:
        counts = _counts_for(rows, EXPERTS, distribution, seed=args.seed)
        compact, padded, tile_expert, padded_rows = _maps(counts)
        if sum(counts) != rows:
            raise SystemExit("count synthesis failed")

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

        def gate_launch() -> None:
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
                RISK_MULTIPLIER,
                rows,
                HIDDEN,
                FFN,
                FFN,
                EXPERTS,
                padded_rows,
            )

        def down_launch() -> None:
            dn.qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
                down_x.ptr,
                compact_dev.ptr,
                padded_dev.ptr,
                tile_dev.ptr,
                down_w.ptr,
                down_out.ptr,
                risk_count.ptr,
                risk_indices.ptr,
                0,
                RISK_MULTIPLIER,
                rows,
                FFN,
                HIDDEN,
                EXPERTS,
                padded_rows,
            )

        gate_ms = _time_launch(gate_launch, runtime, args.repetitions)
        down_ms = _time_launch(down_launch, runtime, args.repetitions)
        for device in (compact_dev, padded_dev, tile_dev, risk_count, risk_indices):
            free(device)

        # Weight bytes actually read: each expert's weights are read once per
        # 16-row tile it owns, so padding inflates the read volume above the
        # tensor size whenever an expert holds rows that do not fill a tile.
        active_experts = sum(1 for count in counts if count > 0)
        gate_tiles = sum((count + WMMA_TILE_ROWS - 1) // WMMA_TILE_ROWS for count in counts)
        gate_read_bytes = gate_tiles * gate_out_total * gate_row_bytes
        down_read_bytes = gate_tiles * HIDDEN * down_row_bytes
        gate_macs = rows * gate_out_total * HIDDEN
        down_macs = rows * HIDDEN * FFN
        return {
            "rows": rows,
            "distribution": distribution,
            "grid_override": grid,
            "active_experts": active_experts,
            "rows_per_active_expert": round(rows / max(1, active_experts), 2),
            "compact_rows": rows,
            "padded_rows": padded_rows,
            "padding_ratio": round(padded_rows / rows, 4),
            "tiles": gate_tiles,
            "gate_up_ms": round(gate_ms, 4),
            "down_ms": round(down_ms, 4),
            "gate_up_weight_bytes": gate_weight_bytes,
            "gate_up_weight_bytes_read": gate_read_bytes,
            "down_weight_bytes_read": down_read_bytes,
            "gate_up_read_amplification": round(gate_read_bytes / gate_weight_bytes, 3),
            "gate_up_gb_per_s": round(gate_read_bytes / gate_ms / 1e6, 2),
            "down_gb_per_s": round(down_read_bytes / down_ms / 1e6, 2),
            "gate_up_tflops": round(2 * gate_macs / gate_ms / 1e9, 2),
            "down_tflops": round(2 * down_macs / down_ms / 1e9, 2),
        }

    print(
        f"{'rows':>6} {'dist':>9} {'act':>5} {'rows/e':>7} {'pad':>6} "
        f"{'gate ms':>9} {'GB/s':>7} {'TFLOP/s':>8} {'down ms':>9} {'GB/s':>7}"
    )
    for distribution in distributions:
        for rows in rows_list:
            entry = measure(rows, distribution, None)
            results["rows"].setdefault(f"{rows}:{distribution}", entry)
            print(
                f"{entry['rows']:>6} {distribution:>9} {entry['active_experts']:>5} "
                f"{entry['rows_per_active_expert']:>7} {entry['padding_ratio']:>6} "
                f"{entry['gate_up_ms']:>9.4f} {entry['gate_up_gb_per_s']:>7.1f} "
                f"{entry['gate_up_tflops']:>8.2f} {entry['down_ms']:>9.4f} "
                f"{entry['down_gb_per_s']:>7.1f}"
            )

    for grid in grid_override:
        entry = measure(rows_list[0], distributions[0], grid)
        results["rows"].setdefault(f"grid{grid}", entry)
        print(f"  grid override {grid}: gate_up {entry['gate_up_ms']:.4f} ms")

    for buffer in (
        gate_x,
        gate_out,
        down_x,
        down_out,
        gate_a,
        gate_b,
        down_w,
    ):
        free(buffer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
