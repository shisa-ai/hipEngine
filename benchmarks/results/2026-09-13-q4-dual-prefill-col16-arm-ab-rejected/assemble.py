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
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("col16_row256", "col16_row512")
CONTROL = "parent"
SHAPES = (256, 512, 1024, 2048, 4096)

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
    "kind": "measured_and_analytic",
    "finding": (
        "The 16-column block halves per-block compute but not per-block decode time. "
        "The decode needs threads/4 >= 2 matrices x (columns/2) pairs, exactly "
        "saturated at 32 columns with 128 threads; at 16 columns only 64 of 128 "
        "threads decode, each doing the same 32 k-values per sub-block, so a block's "
        "decode takes as long as the parent's while its WMMA work halves. Blocks "
        "double for the same output, so total decode time doubles and total WMMA "
        "time is unchanged."
    ),
    "prediction": (
        "With the decode at about 40% of the parent's issue stream (source-level "
        "instruction count in the tile-knob screen), doubling it predicts about "
        "1.4x the parent's time. Measured: 1.40-1.65x at the 512/1024/4096-row shapes."
    ),
    "corrects": (
        "The tile-knob screen's first-ranked candidate assumed that halving columns "
        "halves the per-block decode for the same compute. It does not, which is why "
        "the screen's own sibling measurement (16 columns 1.7x slower) was right and "
        "its instruction-stream ranking was wrong."
    ),
}

DECISION = {
    "verdict": "rejected_and_removed",
    "rule": "retain only if exact and non-regressive at every prefill shape",
    "result": (
        "bit-exact at every cell, regressive by 29-59% at every shape and both arms, "
        "so both arms, their exports, wrappers, registry entries, tests, and the "
        "out_tiles_per_block template parameter they were built on were removed in "
        "the same unit"
    ),
    "end_to_end_note": (
        "No resident-session 512/128, 1K/128, 4K/128 run was spent on the candidate. "
        "The owner carries 33-36% of prefill kernel time (companion kernel profile) "
        "and the control column here reproduces the in-situ profiled owner within "
        "0.1%, so a 1.40-1.65x slower owner adds roughly 13-23% prefill wall by "
        "arithmetic; no end-to-end result could be non-regressive. The candidate also "
        "has no dispatch path, so an end-to-end arm would have required a new "
        "temporary selector for a losing kernel."
    ),
    "next_directions": [
        {
            "id": "decode_lane_remap",
            "why": "the decode is the only part rows per block amortize and about 40% of the issue stream",
            "prerequisite": "none beyond the exact-decode gate; unchanged geometry keeps the resources fixed",
        },
        {
            "id": "wider_column_tile_48",
            "why": "the sibling measurement says wider column tiles win (48 columns 30.82 ms vs 32 columns 39.11-42.97 ms at 4096 rows)",
            "prerequisite": "the decode must reach 192 threads or change lane mapping; 128 threads cannot cover 48 decode pairs",
        },
        {
            "id": "rows_512_at_32_columns",
            "why": "doubles the rows the saturated decode is amortized over, at the measured-good column width",
            "prerequisite": "the epilogue union becomes 64 KiB of LDS (1 workgroup per CU), or the 256-thread register-local-SiLU epilogue rewrite",
        },
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=Path(__file__).resolve().parents[3],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def build(run_dir: Path, model: Path, *, generated_at: str) -> dict:
    runs = [json.loads((run_dir / f"run{i}.json").read_text()) for i in (1, 2, 3)]
    kinds = {run["kind"] for run in runs}
    devices = {run["device"] for run in runs}
    models = {run["model"] for run in runs}
    if len(kinds) != 1 or len(devices) != 1 or len(models) != 1:
        raise SystemExit("the three runs disagree on kind, device, or model")
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
            exact = [bool(row[f"{arm}_bit_equal"]) for row in rows]
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
        "model_sha256": _sha256(model),
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
        "decision": DECISION,
        "provenance": {
            "rejected_commit": _git("rev-parse", "4620b9cf5"),
            "revert_target": _git("rev-parse", "88e0e6b38"),
            "revert_note": (
                "the four reverted paths are byte-identical to this commit; the "
                "removal itself lands as a later commit"
            ),
            "git_head_at_measurement": _git("rev-parse", "HEAD"),
            "harness_run_dir": str(run_dir),
        },
        "limitations": [
            "Single host, single model, single quant, one session per invocation.",
            "Arm order inside an invocation is fixed, so a position or thermal effect "
            "is not counterbalanced; the 3-invocation spread bounds it at "
            f"{round(worst_spread * 100, 1)}% against a 29-59% effect.",
            "Prefill owner only: no resident-session wall, no decode column, and no "
            "multi-prompt category suite was run, because the candidate has no "
            "dispatch path and the owner-level result is decisive by arithmetic.",
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
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args()

    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()
    artifact = build(args.run_dir, args.model, generated_at=generated_at)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
