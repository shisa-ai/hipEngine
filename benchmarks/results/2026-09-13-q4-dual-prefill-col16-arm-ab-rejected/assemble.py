#!/usr/bin/env python3
"""Assemble the rejected 16-column dual-SiLU prefill A/B artifact.

Regenerates ``artifact.json`` byte-identically from the three committed harness
runs (``run{1,2,3}.json`` beside this script) plus the fixed protocol facts of
this unit (commit, model hash, host).  No number in the artifact is transcribed
by hand: every timing, ratio, spread, and bit-exactness flag is derived here,
and the script fails closed if a cell is not bit-exact.

Usage:
    python3 benchmarks/results/2026-09-13-q4-dual-prefill-col16-arm-ab-rejected/assemble.py \
        --out benchmarks/results/2026-09-13-q4-dual-prefill-col16-arm-ab-rejected/artifact.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ARMS = ("col16_row256", "col16_row512")
CONTROL = "parent"
SHAPES = (256, 512, 1024, 2048, 4096)
MEASUREMENT_COMMIT = "4620b9cf5712f99c1e03dee9cbf92ab26e8e8cc8"
MODEL_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

CANDIDATE = {
    "id": "q4-dual-prefill-col16",
    "kernel": (
        "gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel<false,false,rt,4,1>"
    ),
    "layer": "linear_pair_silu",
    "quant": "gguf_q4_k_t16_v1",
    "shape": {"in_features": 5120, "out_features": 17408},
    "arms": {
        "col16_row256": {
            "registered_variant": "dense_dual_wmma_prefill_col16_row256_bf16_bf16_out",
            "instantiation": "<false,false,4,4,1>",
            "geometry": "16 columns x 256 rows per block, 4 waves, 128 threads",
            "vgpr": 224,
            "lds_bytes": 16384,
            "grid_x_at_5120x17408": 1088,
        },
        "col16_row512": {
            "registered_variant": "dense_dual_wmma_prefill_col16_row512_bf16_bf16_out",
            "instantiation": "<false,false,8,4,1>",
            "geometry": "16 columns x 512 rows per block, 4 waves, 128 threads",
            "vgpr": 248,
            "lds_bytes": 32768,
            "grid_x_at_5120x17408": 1088,
        },
    },
}

CONTROL_OWNER = {
    "registered_variant": "dense_dual_wmma_prefill_bf16_bf16_out",
    "kernel": "gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel<false,false,4,4,2>",
    "geometry": "32 columns x 256 rows per block, 4 waves, 128 threads",
    "vgpr": 248,
    "lds_bytes": 32768,
    "grid_x_at_5120x17408": 544,
}

PROTOCOL = {
    "harness": "scripts/qwen38_packet4_q4_dual_silu_row_leaf.py",
    "harness_note": (
        "Same harness and flags as the 2026-09-13 tile-knob screen, so the control "
        "column here is comparable with that screen's control column."
    ),
    "weights": "real blk.0.ffn_gate.weight / blk.0.ffn_up.weight Q4_K tiles from the model",
    "command": (
        "PYTHONPATH=. .venv/bin/python scripts/qwen38_packet4_q4_dual_silu_row_leaf.py "
        "--model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf "
        "--rows 256 512 1024 2048 4096 --burst 4 --repetitions 3 --warmups 2 "
        "--output /tmp/q4-col16-ab/run<N>.json"
    ),
    "invocations": 3,
    "burst": 4,
    "repetitions": 3,
    "warmups": 2,
    "timing": (
        "HIP events around a burst of 4 launches; each cell is the median of 3 "
        "burst samples; the reported value is the median across the 3 harness "
        "invocations; spread is (max-min)/min across those invocations"
    ),
    "arm_order": (
        "fixed dict order (parent first, then the row arms, then the col16 arms); "
        "arm position is not counterbalanced within an invocation"
    ),
    "ratio_convention": "control/candidate, so a value above 1.0 means the candidate is faster",
}

MECHANISM = {
    "kind": "unverified_hypothesis",
    "finding": (
        "At 16 columns only 64 of 128 threads participate in weight decode. "
        "This does not determine block duration or total issue cost. At 512+ rows "
        "col16_row256 doubles block count, but col16_row512 has the same block "
        "count and output elements per full block as the parent. The cause of "
        "the measured slowdown is not isolated."
    ),
    "prediction": (
        "Withdrawn: source-level operation counts are not measured issue time "
        "and cannot predict a 1.4x slowdown."
    ),
    "corrects": (
        "The tile-knob screen's first-ranked candidate assumed that halving columns "
        "improves runtime. Neither the original ranking nor the later doubled-"
        "decode-time explanation is established by the available measurements."
    ),
}

DECISION = {
    "verdict": "rejected_and_removed",
    "rule": "retain only if production-correct and non-regressive in the declared scope",
    "result": (
        "bit-exact at every cell, regressive by 29-59% at every shape and both arms, "
        "so both arms, their exports, wrappers, registry entries, tests, and the "
        "out_tiles_per_block template parameter they were built on were removed in "
        "the same unit"
    ),
    "end_to_end_note": (
        "No resident-session 512/128, 1K/128, 4K/128 run was spent on the candidate. "
        "The owner carries 33-36% of prefill kernel time (companion kernel profile) "
        "and the control column here reproduces the prior microbenchmark within "
        "0.1%. Under unchanged owner share, no overlap changes and transferable "
        "microbenchmark ratios, a 1.40-1.65x slower owner projects roughly 13-23% "
        "extra wall. This is conditional, not proof of an end-to-end loss. The candidate also "
        "has no dispatch path, so an end-to-end arm would have required a new "
        "temporary selector for a losing kernel."
    ),
    "next_directions": [
        {
            "id": "decode_lane_remap",
            "why": "investigate actual decode loads, reuse and instruction stalls",
            "prerequisite": "ISA/resource audit and production numerical gate; unchanged geometry does not guarantee unchanged resources",
        },
        {
            "id": "wider_column_tile_48",
            "why": "the sibling measurement says wider column tiles win (48 columns 30.82 ms vs 32 columns 39.11-42.97 ms at 4096 rows)",
            "prerequisite": "the decode must reach 192 threads or change lane mapping; 128 threads cannot cover 48 decode pairs",
        },
        {
            "id": "rows_512_at_32_columns",
            "why": "doubles the rows the saturated decode is amortized over, at the measured-good column width",
            "prerequisite": "the epilogue union becomes 64 KiB of LDS; measure mode-specific residency, or test a register-local-SiLU epilogue",
        },
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def launch_geometry(rows: int, *, columns: int, tile_rows: int) -> dict:
    return {
        "blocks": ((17408 + columns - 1) // columns) * ((rows + tile_rows - 1) // tile_rows),
        "outputs_per_full_block": columns * tile_rows,
    }


def build(run_dir: Path, model: Path, *, generated_at: str) -> dict:
    runs = [json.loads((run_dir / f"run{i}.json").read_text()) for i in (1, 2, 3)]
    kinds = {run["kind"] for run in runs}
    devices = {run["device"] for run in runs}
    models = {run["model"] for run in runs}
    if len(kinds) != 1 or len(devices) != 1 or len(models) != 1:
        raise SystemExit("the three runs disagree on kind, device, or model")
    if models != {str(model)}:
        raise SystemExit("model does not match the recorded measurement")
    for run in runs:
        if (run["burst"], run["repetitions"], run["warmups"]) != (4, 3, 2):
            raise SystemExit("a run did not use the declared burst/repetitions/warmups")
        if [row["rows"] for row in run["rows"]] != list(SHAPES):
            raise SystemExit("a run did not cover the declared shapes")

    cells = []
    for shape in SHAPES:
        rows = [next(row for row in run["rows"] if row["rows"] == shape) for run in runs]
        control = [row[f"{CONTROL}_ms"] for row in rows]
        control_median = statistics.median(control)
        cell = {
            "rows": shape,
            "control_ms": [round(value, 4) for value in control],
            "control_median_ms": round(control_median, 4),
            "control_spread": round(max(control) / min(control) - 1.0, 5),
            "arms": {},
        }
        for arm in ARMS:
            values = [row[f"{arm}_ms"] for row in rows]
            median = statistics.median(values)
            exact = [row[f"{arm}_bit_equal"] is True for row in rows]
            if not all(exact):
                raise SystemExit(f"{arm} at rows {shape} is not bit-exact")
            cell["arms"][arm] = {
                "ms": [round(value, 4) for value in values],
                "median_ms": round(median, 4),
                "ratio_vs_control": round(control_median / median, 3),
                "spread": round(max(values) / min(values) - 1.0, 5),
                "bit_equal": exact,
            }
        cells.append(cell)

    worst_spread = max(
        max(cell["control_spread"], *(arm["spread"] for arm in cell["arms"].values()))
        for cell in cells
    )
    best_ratio = max(
        arm["ratio_vs_control"] for cell in cells for arm in cell["arms"].values()
    )
    exact_cells = sum(
        1 for cell in cells for arm in cell["arms"].values() if all(arm["bit_equal"])
    )
    return {
        "schema": 1,
        "date": "2026-09-13",
        "status": "rejected",
        "kind": "hipengine_gfx1151_q4_dual_silu_prefill_tile_ab",
        "generated_at": generated_at,
        "performance_claim": False,
        "candidate": CANDIDATE,
        "control_owner": CONTROL_OWNER,
        "model": str(model),
        "model_sha256": MODEL_SHA256,
        "quant": "Q4_K_M",
        "kv": "BF16",
        "backend": "hip_gfx1151",
        "hardware": {
            "host_name": "gfx1151",
            "device_name": devices.pop(),
            "cu_count": 40,
            "arch": "gfx1151",
        },
        "protocol": PROTOCOL,
        "cells": cells,
        "summary": {
            "best_candidate_ratio": best_ratio,
            "worst_candidate_ratio": min(
                arm["ratio_vs_control"]
                for cell in cells
                for arm in cell["arms"].values()
            ),
            "worst_cell_spread": round(worst_spread, 5),
            "exact_cells": exact_cells,
            "total_cells": len(cells) * len(ARMS),
            "control_matches_prior_screen": {
                "rows": 512,
                "this_unit_ms": next(
                    cell["control_median_ms"] for cell in cells if cell["rows"] == 512
                ),
                "prior_screen_ms": 6.629,
                "delta": round(
                    next(
                        cell["control_median_ms"]
                        for cell in cells
                        if cell["rows"] == 512
                    )
                    / 6.629
                    - 1.0,
                    5,
                ),
                "prior_screen": (
                    "benchmarks/results/2026-09-13-q4-dual-prefill-tile-knob-screen/"
                    "rowtile_prefill.json"
                ),
            },
        },
        "mechanism": MECHANISM,
        "launch_geometry": {
            str(rows): {
                "parent": launch_geometry(rows, columns=32, tile_rows=256),
                "col16_row256": launch_geometry(rows, columns=16, tile_rows=256),
                "col16_row512": launch_geometry(rows, columns=16, tile_rows=512),
            } for rows in SHAPES
        },
        "decision": DECISION,
        "provenance": {
            "rejected_commit": MEASUREMENT_COMMIT,
            "revert_target": "88e0e6b38",
            "revert_note": (
                "the four reverted paths are byte-identical to this commit; the "
                "removal itself lands as a later commit"
            ),
            "git_head_at_measurement": MEASUREMENT_COMMIT,
            "model_hash_source": "recorded_measurement",
            "harness_run_dir": "benchmarks/results/2026-09-13-q4-dual-prefill-col16-arm-ab-rejected",
            "input_sha256": {f"run{i}.json": _sha256(run_dir / f"run{i}.json") for i in (1, 2, 3)},
        },
        "limitations": [
            "Single host, single model, single quant, one session per invocation.",
            "Arm order inside an invocation is fixed, so a position or thermal effect "
            "is not counterbalanced. The 3-invocation spread measures repeatability, "
            "not a bound on systematic ordering bias.",
            "Prefill owner only: no resident-session wall, no decode column, and no "
            "multi-prompt category suite was run, because the candidate has no "
            "dispatch path and the owner-level loss did not warrant further validation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generated-at", default="2026-09-13T22:05:00+00:00",
                        help="original assembly timestamp, not a measurement timestamp")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    artifact = build(args.run_dir, args.model, generated_at=args.generated_at)
    encoded = json.dumps(artifact, indent=1) + "\n"
    if args.check:
        if args.out.read_text() != encoded:
            raise SystemExit("artifact differs from deterministic assembly")
        print("artifact verified")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
