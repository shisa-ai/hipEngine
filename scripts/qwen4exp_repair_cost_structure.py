#!/usr/bin/env python3
"""Where does the iu8 exact-repair time go?

The shipped default spends 1420.9 ms of a 16733.4 ms ``code-p4096`` prefill in
the sparse exact-repair passes, to correct 0.71% of output elements. This
packet runs the two shipped repair kernels on the real admitted shapes and the
measured incidence, and sweeps the two knobs that separate the candidate
explanations:

* risk count (0, 1k, 10k, and the measured 0.712% of outputs), which separates
  a fixed launch cost from a per-element cost;
* grid size (the shipped ``static_tiles * 16``, and larger), which separates a
  latency-bound kernel from a bandwidth-bound one.

Weights are synthesized, not loaded from the model: this packet measures the
*access pattern* (one weight row and one full activation row per repaired
element), so the byte counts and the parallelism response are representative
while the values are not. Outputs are compared bitwise across grid sizes to
show the sweep cannot change a repaired value.

The kernels themselves are the shipped ones; nothing here is a prototype.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)

# The model's MoE geometry, from the shipped model-shape map.
EXPERTS = 512
HIDDEN = 2560
FFN = 640
ROWS = 10240  # 1024 tokens x top_k 10, one chunk

# Measured aggregate repair rate for q4_iu8_exact:expert_gate_up
# (worklog 20260915T111206.049450Z-lhl-journey-risk-repair-instrumentation).
INCIDENCE = 0.007121483417267495

Q4_K_BLOCK_BYTES = 144
Q5_1_BLOCK_BYTES = 24


def _q4_k_row_bytes(in_features: int) -> int:
    return (in_features // 256) * Q4_K_BLOCK_BYTES


def _q5_1_row_bytes(in_features: int) -> int:
    return (in_features // 32) * Q5_1_BLOCK_BYTES


def _expert_start(rows: int, experts: int, *, seed: int) -> list[int]:
    """A realistic compact-row -> expert map: every expert owns a few rows.

    The router sends each token to ten experts, so the tile map hands each
    expert a small block of consecutive compact rows. The exact counts do not
    affect the repair: it binary-searches ``expert_start`` for the owning
    expert whatever the counts are.
    """

    import random

    if rows < experts:
        raise ValueError("rows must be at least one per expert")
    rng = random.Random(seed)
    weights = [rng.random() + 0.5 for _ in range(experts)]
    total = sum(weights)
    counts = [max(1, int(round(w / total * rows))) for w in weights]
    # Reconcile the rounding so the starts cover exactly ``rows`` rows.
    drift = rows - sum(counts)
    index = 0
    while drift != 0:
        counts[index % experts] += 1 if drift > 0 else -1
        drift += -1 if drift > 0 else 1
        index += 1
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    return starts


def _risk_indices(
    rows: int,
    out_features_total: int,
    count: int,
    *,
    seed: int,
    clustered: bool,
) -> list[int]:
    """Risk slots at the measured incidence, uniform or row-clustered."""

    import random

    rng = random.Random(seed)
    if count <= 0:
        return []
    if not clustered:
        return [
            rng.randrange(rows) * out_features_total + rng.randrange(out_features_total)
            for _ in range(count)
        ]
    # Clustered: a minority of rows carry most of the risk, which is what a
    # per-row bound would produce if the activation magnitude varies by row.
    indices: list[int] = []
    hot_rows = max(1, rows // 8)
    while len(indices) < count:
        row = rng.randrange(hot_rows) if rng.random() < 0.85 else rng.randrange(rows)
        indices.append(row * out_features_total + rng.randrange(out_features_total))
    return indices


def _time_launch(
    launch: Any,
    runtime: Any,
    repetitions: int,
    *,
    prime: int = 2,
) -> float:
    samples: list[float] = []
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        for _ in range(prime):
            launch()
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


def _bits(buffer: Any, count: int) -> Any:
    import numpy as np

    host = np.zeros(count, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(host), buffer, count * 2)
    return host


def _fill_device(buffer: Any, value: int, runtime: Any) -> None:
    """Write ``value`` into every element of a uint16 device buffer.

    The host array is a named local on purpose. ``host_array_ptr(np.full(...))``
    inside the copy call passes the address of a temporary whose owner can be
    released before the memcpy reads it, which segfaults once the allocation is
    large enough for numpy to mmap it.
    """

    import numpy as np

    host = np.full(buffer.nbytes // 2, value, dtype=np.uint16)
    copy_host_to_device(buffer, host_array_ptr(host), buffer.nbytes, runtime=runtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument(
        "--grids",
        default="",
        help="Override the gate/up and down grid lists, e.g. '17920,65536'.",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="Draw the risk set from a small row window instead of uniformly, "
        "so the activation rows a repair re-reads are L2-resident. If the "
        "per-element cost falls, the kernel is bandwidth/L2-bound and a "
        "row-sorted queue would help; if it does not, it is memory-level "
        "parallelism that sets the cost and sorting cannot help.",
    )
    parser.add_argument(
        "--counts",
        default="",
        help="Explicit risk counts; defaults to 0, 1k, 10k and the measured "
        "incidence for each shape.",
    )
    args = parser.parse_args()

    import numpy as np

    runtime = get_hip_runtime()

    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as gu
    from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as dn

    gu.build_gguf_q4_k_selected_prefill(load=True)
    dn.build_qwen4_exp_q5_1(load=True)

    rows = int(args.rows)
    gate_out = FFN
    gate_total = 2 * FFN
    down_out = HIDDEN

    gate_row_bytes = _q4_k_row_bytes(HIDDEN)
    down_row_bytes = _q5_1_row_bytes(FFN)

    gate_weight_bytes = EXPERTS * gate_out * gate_row_bytes
    down_weight_bytes = EXPERTS * down_out * down_row_bytes

    shapes = {
        "gate_up": {
            "in_features": HIDDEN,
            "out_features": gate_out,
            "out_features_total": gate_total,
            "row_bytes": gate_row_bytes,
            "weight_bytes_per_tensor": gate_weight_bytes,
            "weight_bytes_per_element": gate_row_bytes,
            "activation_bytes_per_element": HIDDEN * 2,
            "out_features": gate_total,
        },
        "down": {
            "in_features": FFN,
            "out_features": down_out,
            "out_features_total": down_out,
            "row_bytes": down_row_bytes,
            "weight_bytes_per_tensor": down_weight_bytes,
            "weight_bytes_per_element": down_row_bytes,
            "activation_bytes_per_element": FFN * 2,
            "out_features": down_out,
        },
    }

    starts = np.asarray(_expert_start(rows, EXPERTS, seed=args.seed), dtype=np.int64)
    if starts.size != EXPERTS + 1 or int(starts[-1]) != rows:
        raise SystemExit("expert_start synthesis failed")

    expert_start = malloc(starts.nbytes)
    copy_host_to_device(expert_start, host_array_ptr(starts), starts.nbytes)

    # The runner's repair grid is ``static_tiles * 16`` with
    # ``static_tiles = active + (compact - active) // 16`` and
    # ``active = min(compact, experts)``.
    active = min(rows, EXPERTS)
    gate_grid = 16 * (active + (rows - active) // 16)
    grid_override = [int(p) for p in args.grids.split(",") if p.strip()]
    if args.counts:
        counts_override = [int(p) for p in args.counts.split(",") if p.strip()]
    else:
        counts_override = []

    results: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_repair_cost_structure",
        "question": (
            "Is the iu8 exact repair a fixed launch cost, a per-element cost, or "
            "a bandwidth cost, and does grid size move it?"
        ),
        "protocol": {
            "rows": rows,
            "experts": EXPERTS,
            "in_features": {"gate_up": HIDDEN, "down": FFN},
            "out_features": {"gate_up": gate_total, "down": down_out},
            "incidence": INCIDENCE,
            "repetitions": args.repetitions,
            "timing": "median of per-launch HIP event pairs, 2 primed launches",
            "weights": (
                "synthesized at the real shapes and byte layout; the access "
                "pattern and byte counts are representative, the values are not"
            ),
            "shipped_grid": {
                "gate_up": gate_grid,
                "down": gate_grid,
                "note": "static_tiles * 16, the runner's value for a 1024-token chunk",
            },
        },
        "shapes": shapes,
        "grids": {},
        "risk_set": "clustered" if args.clustered else "uniform",
    }

    def report(name: str, entries: list[dict[str, Any]]) -> None:
        results["grids"][name] = entries
        print(f"\n=== {name} ===")
        for e in entries:
            print(
                f"  risks={e['risks']:>7d} grid={e['grid']:>7d} "
                f"{e['ms']:8.4f} ms  {e['ms_per_1k_risks']:8.4f} ms/1k  "
                f"{e['gb_per_s']:8.1f} GB/s"
            )

    # ---- gate/up dual repair ------------------------------------------------
    gate_a = malloc(gate_weight_bytes)
    gate_b = malloc(gate_weight_bytes)
    gate_x = malloc(rows * HIDDEN * 2)
    gate_out_buf = malloc(rows * gate_total * 2)
    gate_risk_count = malloc(4)
    gate_capacity = rows * gate_total
    gate_indices = malloc(gate_capacity * 4)
    runtime.memset(gate_a.ptr, 0x3C, min(gate_weight_bytes, 64 << 20))
    runtime.memset(gate_b.ptr, 0x5A, min(gate_weight_bytes, 64 << 20))
    runtime.memset(gate_x.ptr, 0x2D, min(rows * HIDDEN * 2, 64 << 20))
    runtime.memset(gate_risk_count.ptr, 0, 4)

    def gate_launch(grid: int, count: int) -> Any:
        def run() -> None:
            gu.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
                gate_x.ptr,
                expert_start.ptr,
                gate_a.ptr,
                gate_b.ptr,
                gate_out_buf.ptr,
                gate_risk_count.ptr,
                gate_indices.ptr,
                gate_capacity,
                rows,
                HIDDEN,
                gate_out,
                gate_out,
                EXPERTS,
                grid_blocks=grid,
                stream=0,
                runtime=runtime,
            )

        return run

    gate_entries: list[dict[str, Any]] = []
    gate_bits: dict[int, Any] = {}
    for grid in ([gate_grid] if not grid_override else grid_override):
        for count in (counts_override or [0, 1000, 10000, int(rows * gate_total * INCIDENCE)]):
            if count > gate_capacity:
                raise SystemExit("risk count exceeds capacity")
            if count:
                indices = np.asarray(
                    _risk_indices(
                        rows,
                        gate_total,
                        count,
                        seed=args.seed,
                        clustered=args.clustered,
                    ),
                    dtype=np.int32,
                )
                copy_host_to_device(
                    gate_indices, host_array_ptr(indices), indices.nbytes
                )
            runtime.memset(gate_risk_count.ptr, 0, 4)
            counter = np.asarray([count], dtype=np.int32)
            copy_host_to_device(gate_risk_count, host_array_ptr(counter), 4)
            ms = _time_launch(gate_launch(grid, count), runtime, args.repetitions)
            moved = count * (
                gate_row_bytes + HIDDEN * 2
            )
            gate_entries.append(
                {
                    "risks": count,
                    "grid": grid,
                    "ms": ms,
                    "ms_per_1k_risks": 1000 * ms / count if count else 0.0,
                    "bytes_moved": moved,
                    "gb_per_s": moved / (ms * 1e6) if ms else 0.0,
                }
            )
            if count and grid in (gate_grid, 17920, 65536):
                gate_bits.setdefault(count, {})[grid] = _bits(
                    gate_out_buf, rows * gate_total
                )
    report("gate_up", gate_entries)

    gate_identity = {}
    for count, by_grid in gate_bits.items():
        grids = sorted(by_grid)
        base = by_grid[grids[0]]
        gate_identity[str(count)] = {
            f"grid_{grids[0]}_vs_{g}": int(np.count_nonzero(base != by_grid[g]))
            for g in grids[1:]
        }
    results["bitwise_identity"] = gate_identity

    # ---- down row-publish repair -------------------------------------------
    down_w = malloc(down_weight_bytes)
    down_x = malloc(rows * FFN * 2)
    down_out_buf = malloc(rows * down_out * 2)
    down_risk_count = malloc(4)
    down_capacity = rows * down_out
    down_indices = malloc(down_capacity * 4)
    runtime.memset(down_w.ptr, 0x71, min(down_weight_bytes, 64 << 20))
    runtime.memset(down_x.ptr, 0x1B, min(rows * FFN * 2, 64 << 20))
    runtime.memset(down_risk_count.ptr, 0, 4)

    def down_launch(grid: int, count: int) -> Any:
        def run() -> None:
            dn.qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16(
                down_x.ptr,
                expert_start.ptr,
                down_w.ptr,
                down_out_buf.ptr,
                down_risk_count.ptr,
                down_indices.ptr,
                down_capacity,
                rows,
                FFN,
                down_out,
                EXPERTS,
                grid_blocks=grid,
                stream=0,
                runtime=runtime,
            )

        return run

    down_entries: list[dict[str, Any]] = []
    down_bits: dict[int, Any] = {}
    for grid in ([gate_grid] if not grid_override else grid_override):
        for count in (counts_override or [0, 1000, 10000, int(rows * down_out * INCIDENCE)]):
            if count > down_capacity:
                raise SystemExit("risk count exceeds capacity")
            if count:
                indices = np.asarray(
                    _risk_indices(
                        rows, down_out, count, seed=args.seed + 1, clustered=False
                    ),
                    dtype=np.int32,
                )
                copy_host_to_device(
                    down_indices, host_array_ptr(indices), indices.nbytes
                )
            runtime.memset(down_risk_count.ptr, 0, 4)
            counter = np.asarray([count], dtype=np.int32)
            copy_host_to_device(down_risk_count, host_array_ptr(counter), 4)
            ms = _time_launch(down_launch(grid, count), runtime, args.repetitions)
            moved = count * (down_row_bytes + FFN * 2)
            down_entries.append(
                {
                    "risks": count,
                    "grid": grid,
                    "ms": ms,
                    "ms_per_1k_risks": 1000 * ms / count if count else 0.0,
                    "bytes_moved": moved,
                    "gb_per_s": moved / (ms * 1e6) if ms else 0.0,
                }
            )
            if count and grid in (gate_grid, 17920, 65536):
                down_bits.setdefault(count, {})[grid] = _bits(
                    down_out_buf, rows * down_out
                )
    report("down", down_entries)

    down_identity = {}
    for count, by_grid in down_bits.items():
        grids = sorted(by_grid)
        base = by_grid[grids[0]]
        down_identity[str(count)] = {
            f"grid_{grids[0]}_vs_{g}": int(np.count_nonzero(base != by_grid[g]))
            for g in grids[1:]
        }
    results["down_bitwise_identity"] = down_identity

    # ---- row-bitmap repair (gate/up) ---------------------------------------
    # The same exact repair, with the flat queue folded into a per-row column
    # bitmap and one LDS-staged activation row per row. Timed as the composite
    # the engine would launch: bitmap pass, then repair.
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        risk_bitmap_bytes,
        risk_bitmap_words,
    )

    gate_bitmap_buf = malloc(risk_bitmap_bytes(rows, gate_total))
    words = risk_bitmap_words(gate_total)
    bitmap_entries: list[dict[str, Any]] = []
    bitmap_identity: dict[str, Any] = {}
    for count in (counts_override or [0, 1000, 10000, int(rows * gate_total * INCIDENCE)]):
        if count > gate_capacity:
            raise SystemExit("risk count exceeds capacity")
        if count:
            indices = np.asarray(
                _risk_indices(rows, gate_total, count, seed=args.seed, clustered=False),
                dtype=np.int32,
            )
            copy_host_to_device(gate_indices, host_array_ptr(indices), indices.nbytes)
        counter = np.asarray([count], dtype=np.int32)
        copy_host_to_device(gate_risk_count, host_array_ptr(counter), 4)
        runtime.memset(gate_bitmap_buf.ptr, 0, gate_bitmap_buf.nbytes)
        _fill_device(gate_out_buf, 0x7FFF, runtime)

        def bitmap_launch() -> None:
            gu.gguf_q4_k_selected_dual_risk_bitmap(
                gate_risk_count.ptr,
                gate_indices.ptr,
                gate_bitmap_buf.ptr,
                gate_capacity,
                rows,
                gate_total,
                stream=0,
                runtime=runtime,
            )
            gu.gguf_q4_k_selected_dual_row_bitmap_repair_bf16(
                gate_x.ptr,
                expert_start.ptr,
                gate_a.ptr,
                gate_b.ptr,
                gate_out_buf.ptr,
                gate_bitmap_buf.ptr,
                rows,
                HIDDEN,
                gate_out,
                gate_out,
                EXPERTS,
                stream=0,
                runtime=runtime,
            )

        ms = _time_launch(bitmap_launch, runtime, args.repetitions)
        moved = count * gate_row_bytes + rows * HIDDEN * 2
        bitmap_entries.append(
            {
                "risks": count,
                "ms": ms,
                "ms_per_1k_risks": 1000 * ms / count if count else 0.0,
                "bytes_moved": moved,
                "gb_per_s": moved / (ms * 1e6) if ms else 0.0,
                "note": "bitmap pass + row repair, composite",
            }
        )
        if count:
            # Ownership and value identity against the per-slot kernel, whose
            # output the sweep above left in gate_out_buf... which the bitmap
            # arm overwrote, so re-run the per-slot kernel with the sentinel
            # priming and the same queue.
            _fill_device(gate_out_buf, 0x7FFF, runtime)
            gu.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
                gate_x.ptr,
                expert_start.ptr,
                gate_a.ptr,
                gate_b.ptr,
                gate_out_buf.ptr,
                gate_risk_count.ptr,
                gate_indices.ptr,
                gate_capacity,
                rows,
                HIDDEN,
                gate_out,
                gate_out,
                EXPERTS,
                grid_blocks=gate_grid,
                stream=0,
                runtime=runtime,
            )
            runtime.device_synchronize()
            per_slot = _bits(gate_out_buf, rows * gate_total)
            bitmap_launch()
            runtime.device_synchronize()
            bucketed = _bits(gate_out_buf, rows * gate_total)
            targets = np.unique(indices.astype(np.int64))
            untouched = np.setdiff1d(
                np.arange(rows * gate_total), targets, assume_unique=False
            )
            bitmap_identity[str(count)] = {
                "targets": int(targets.size),
                "value_mismatches": int(
                    np.count_nonzero(per_slot[targets] != bucketed[targets])
                ),
                "per_slot_wrote_unqueued": int(
                    np.count_nonzero(per_slot[untouched] != 0x7FFF)
                ),
                "bitmap_wrote_unqueued": int(
                    np.count_nonzero(bucketed[untouched] != 0x7FFF)
                ),
                "bitmap_words": words,
            }

    results["row_bitmap_gate_up"] = bitmap_entries
    results["row_bitmap_identity"] = bitmap_identity
    print("\n=== gate_up row-bitmap repair (composite) ===")
    for e in bitmap_entries:
        print(
            f"  risks={e['risks']:>7d} {e['ms']:8.4f} ms  "
            f"{e['ms_per_1k_risks']:8.4f} ms/1k  {e['gb_per_s']:8.1f} GB/s"
        )
    for count, ident in bitmap_identity.items():
        print(
            f"  identity risks={count}: value mismatches "
            f"{ident['value_mismatches']}, unqueued writes "
            f"{ident['per_slot_wrote_unqueued']}/{ident['bitmap_wrote_unqueued']}"
        )

    for buffer in (
        gate_a, gate_b, gate_x, gate_out_buf, gate_risk_count, gate_indices,
        down_w, down_x, down_out_buf, down_risk_count, down_indices,
        expert_start, gate_bitmap_buf,
    ):
        free(buffer, runtime=runtime)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
