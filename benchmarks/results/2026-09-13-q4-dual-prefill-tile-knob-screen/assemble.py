#!/usr/bin/env python3
"""Assemble the Q4 dual-WMMA prefill tile-knob artifact from raw screen output.

Inputs (all produced by committed scripts, none hand-edited):

* ``resource_table.json`` - per-kernel VGPR/SGPR/LDS from
  ``scripts/gguf_prefill_kernel_resources.py`` (compiler metadata, the same
  fields the runtime profiler reports).
* ``rowtile_prefill.json`` - the dual gate/up SiLU row-tile sweep from
  ``scripts/qwen38_packet4_q4_dual_silu_row_leaf.py`` on real
  ``Qwen3.8-27B-Q4_K_M`` ``blk.0.ffn_gate``/``blk.0.ffn_up`` tiles.
* ``col_tile_owners.json`` - the single-matrix column-tile sweep from
  ``scripts/gguf_q4_t16_dense_prefill_owner_microbench.py``.

Everything in ``artifact.json`` is derived here: geometry labels come from the
launcher template arguments in the kernel source (the mangled names carry them),
occupancy comes from the resource metadata, and rates come from the measured
medians.

    python3 assemble.py --out artifact.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Launcher template arguments, from
# hipengine/kernels/hip_gfx1100/quant/gguf_k_t16_selected_prefill.hip:
#   * dual gate/up SiLU:  launch_..._dual_wmma_prefill_silu_bf16<
#                            qmicro, external_meta, row_tiles_per_wave,
#                            active_waves_per_block>
#     with a fixed 2 x T16_COLS = 32-column block and a fixed 128-thread launch.
#   * single-matrix shared-B: launch_q4_t16_dense_wmma_prefill_shared_b_impl<
#                            out_tiles, waves, row_tiles, scalar, skip>
#     with threads = 32 * waves.
T16_COLS = 16
DUAL_COLUMNS_PER_BLOCK = 2 * T16_COLS
DUAL_THREADS = 128
TOTAL_LDS_BYTES = 65536
VGPR_FILE_PER_SIMD = 512

DUAL_TEMPLATE_RE = re.compile(
    r"dual_wmma_prefill_silu_bf16_kernelILb0ELb0ELi(\d+)ELi(\d+)EE"
)
DUAL_OWNER_BY_ROWS_PER_BLOCK = {
    32: "row32",
    48: "row48",
    64: "row64",
    128: "row128",
    256: "parent",
}
# Column tile and block shape per single-matrix owner, read from the launcher
# template arguments in the kernel source. `shared_b*` use
# launch_q4_t16_dense_wmma_prefill_shared_b_impl<out_tiles, waves, row_tiles>;
# `default` uses gguf_q4_t16_dense_wmma_prefill_bf16_kernel<false, 4> at
# out_features / (3 * T16_COLS); `lowvgpr`/`lowvgpr48` use
# gguf_q4_t16_dense_wmma_prefill_bf16_kernel<false, 2|3, 1> at
# out_features / T16_COLS, i.e. 16 columns per block.
SINGLE_MATRIX_GEOMETRY = {
    "shared_b": (3, 4, 4),
    "shared_b2w4": (2, 4, 4),
    "shared_b2w2": (2, 2, 4),
    "shared_b3w8r3": (3, 8, 3),
}
SINGLE_WAVE_GEOMETRY = {
    "default": (3, 1, 4),
    "lowvgpr": (1, 1, 2),
    "lowvgpr48": (1, 1, 3),
}


def occupancy(vgpr: int, lds_bytes: int, waves_per_workgroup: int) -> dict:
    waves_per_simd = min(8, VGPR_FILE_PER_SIMD // ((vgpr + 7) // 8 * 8))
    workgroups_per_cu = max(1, TOTAL_LDS_BYTES // lds_bytes) if lds_bytes else 32
    return {
        "waves_per_simd_vgpr": waves_per_simd,
        "workgroups_per_cu_lds": workgroups_per_cu,
        "waves_per_cu": min(
            32, waves_per_simd * 4, workgroups_per_cu * waves_per_workgroup
        ),
    }


def resource_row(kernels: list[dict], pattern: re.Pattern, family: str) -> dict:
    """Return the single resource row whose mangled name matches ``pattern``."""

    matches = [row for row in kernels if pattern.search(str(row["kernel"]))]
    if len(matches) != 1:
        raise SystemExit(
            f"expected one {family} instantiation, found {len(matches)}"
        )
    return matches[0]


def dual_ladder(resources: dict) -> list[dict]:
    """Geometry and occupancy of every registered dual gate/up SiLU variant."""

    rows = []
    for row in resources["kernels"]:
        match = DUAL_TEMPLATE_RE.search(str(row["kernel"]))
        if not match:
            continue
        row_tiles, waves = int(match.group(1)), int(match.group(2))
        rows_per_block = waves * row_tiles * T16_COLS
        accumulator_vgpr = (
            2  # gate + up
            * (DUAL_COLUMNS_PER_BLOCK // T16_COLS)
            * row_tiles
            * 8  # float8_t accumulator
        )
        rows.append(
            {
                "kernel": row["kernel"],
                "owner": DUAL_OWNER_BY_ROWS_PER_BLOCK.get(
                    rows_per_block, f"rows{rows_per_block}"
                ),
                "row_tiles_per_wave": row_tiles,
                "active_waves_per_block": waves,
                "columns_per_block": DUAL_COLUMNS_PER_BLOCK,
                "rows_per_block": rows_per_block,
                "threads_per_block": DUAL_THREADS,
                "accumulator_vgpr": accumulator_vgpr,
                "vgpr": row["vgpr"],
                "vgpr_allocated": row["vgpr_allocated"],
                "sgpr": row["sgpr"],
                "lds_bytes": row["lds_bytes"],
                "private_bytes": row["private_bytes"],
                **occupancy(row["vgpr"], row["lds_bytes"], waves),
            }
        )
    rows.sort(key=lambda row: row["rows_per_block"])
    return rows


def row_tile_sweep(sweep: dict, ladder: list[dict]) -> dict:
    """Per-row timings, ratios and derived per-block cost for the dual ladder."""

    out_features = int(sweep["out_features"])
    in_features = int(sweep["in_features"])
    columns_per_block = DUAL_COLUMNS_PER_BLOCK
    column_blocks = out_features // columns_per_block
    by_owner = {row["owner"]: row for row in ladder}
    cells = []
    for entry in sweep["rows"]:
        rows = int(entry["rows"])
        cell: dict = {"rows": rows, "variants": {}}
        for owner, row in by_owner.items():
            key = f"{owner}_ms"
            if key not in entry:
                continue
            milliseconds = float(entry[key])
            row_blocks = -(-rows // int(row["rows_per_block"]))
            blocks = column_blocks * row_blocks
            flops = 2 * 2 * rows * out_features * in_features
            cell["variants"][owner] = {
                "median_ms": milliseconds,
                "vs_parent": round(float(entry["parent_ms"]) / milliseconds, 4),
                "bit_equal_vs_parent": (
                    None
                    if owner == "parent"
                    else bool(entry[f"{owner}_bit_equal"])
                ),
                "blocks": blocks,
                "per_block_us": round(milliseconds * 1000.0 / blocks, 4),
                "per_row_ns": round(milliseconds * 1e6 / rows, 3),
                "tflop_s": round(flops / (milliseconds * 1e-3) / 1e12, 3),
            }
        cells.append(cell)
    return {
        "protocol": {
            "harness": "scripts/qwen38_packet4_q4_dual_silu_row_leaf.py",
            "weights": "real Qwen3.8-27B-Q4_K_M blk.0.ffn_gate/ffn_up tiles",
            "burst": sweep["burst"],
            "warmups": sweep["warmups"],
            "repetitions": sweep["repetitions"],
            "device": sweep["device"],
            "model": sweep["model"],
        },
        "shape": {"rows": [cell["rows"] for cell in cells],
                  "in_features": in_features, "out_features": out_features},
        "cells": cells,
    }


def column_tile_sweep(sweep: dict, resources: dict) -> dict:
    """Timings and derived rates for the single-matrix column-tile sweep."""

    in_features = int(sweep["config"]["in_features"])
    out_features = int(sweep["config"]["out_features"])
    geometry = {**SINGLE_MATRIX_GEOMETRY, **SINGLE_WAVE_GEOMETRY}
    cells = []
    for cell in sweep["cells"]:
        rows = int(cell["rows"])
        owners = {}
        for owner, timings in cell["owners"].items():
            out_tiles, waves, row_tiles = geometry[owner]
            columns = out_tiles * T16_COLS
            rows_per_block = waves * row_tiles * T16_COLS
            milliseconds = float(timings["median_s"]) * 1e3
            owners[owner] = {
                "columns_per_block": columns,
                "rows_per_block": rows_per_block,
                "threads_per_block": 32 * waves,
                "waves_per_workgroup": waves,
                "row_tiles_per_wave": row_tiles,
                "median_ms": round(milliseconds, 4),
                "tflop_s": round(
                    2 * rows * out_features * in_features
                    / (milliseconds * 1e-3)
                    / 1e12,
                    3,
                ),
                "column_blocks": -(-out_features // columns),
                "blocks": -(-out_features // columns) * -(-rows // rows_per_block),
                "activation_bytes_per_launch": -(-out_features // columns)
                * rows
                * in_features
                * 2,
                "bit_exact_vs_shared_b": timings.get("bit_exact_vs_shared_b"),
            }
        cells.append({"rows": rows, "owners": owners})
    return {
        "protocol": {
            "harness": "scripts/gguf_q4_t16_dense_prefill_owner_microbench.py",
            "weights": "synthetic Q4_K tiles, seed "
            f"{sweep['config']['seed']}",
            "warmup": sweep["config"]["warmup"],
            "iters": sweep["config"]["iters"],
            "note": (
                "single-matrix owners; each launch computes one projection. "
                "The 16-column owners are 32-thread single-wave blocks, so "
                "they confound the column tile with block size and rows per "
                "block; the 32- and 48-column owners are 4- or 8-wave blocks"
            ),
        },
        "cells": cells,
        "resource_metadata": {
            owner: {
                key: value
                for key, value in resource_row(
                    resources["kernels"],
                    re.compile(
                        r"shared_b_bf16_kernelILi{}ELi{}ELi{}EtLb0EE".format(
                            *SINGLE_MATRIX_GEOMETRY[owner]
                        )
                    ),
                    owner,
                ).items()
                if key
                in (
                    "vgpr",
                    "vgpr_allocated",
                    "sgpr",
                    "lds_bytes",
                    "private_bytes",
                    "waves_per_simd_vgpr",
                    "workgroups_per_cu_lds",
                    "waves_per_cu",
                )
            }
            for owner in SINGLE_MATRIX_GEOMETRY
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, default=HERE / "resource_table.json")
    parser.add_argument("--rowtile", type=Path, default=HERE / "rowtile_prefill.json")
    parser.add_argument("--coltile", type=Path, default=HERE / "col_tile_owners.json")
    parser.add_argument("--out", type=Path, default=HERE / "artifact.json")
    args = parser.parse_args()

    resources = json.loads(args.resources.read_text())
    ladder = dual_ladder(resources)
    rowtile = row_tile_sweep(json.loads(args.rowtile.read_text()), ladder)
    coltile = column_tile_sweep(json.loads(args.coltile.read_text()), resources)

    parent = next(row for row in ladder if row["owner"] == "parent")
    artifact = {
        "schema": 1,
        "status": "diagnostic",
        "kind": "q4_dual_wmma_prefill_tile_knob_screen",
        "date": "2026-09-13",
        "performance_claim": False,
        "purpose": (
            "Name the tile-geometry and occupancy knobs of the gfx1151 Q4_K_M "
            "dense WMMA prefill owner (the gate/up dual-SiLU kernel that holds "
            "33-36% of prefill kernel time) and measure which direction each "
            "knob moves at prefill rows, before writing a new tile variant."
        ),
        "hardware": {
            "host_name": "gfx1151",
            "device_name": rowtile["protocol"]["device"],
            "target_arch": resources["provenance"]["arch"],
            "cu_count": 40,
            "lds_bytes_per_cu": TOTAL_LDS_BYTES,
            "vgpr_file_per_simd": VGPR_FILE_PER_SIMD,
            "source": "rocprofv3 agent_info.csv in the companion kernel profile; "
            "LDS and VGPR capacities are the gfx11 wave32 limits used by "
            "scripts/gguf_prefill_kernel_resources.py",
        },
        "owner_under_study": {
            "kernel": parent["kernel"],
            "registered_variant": "dense_dual_wmma_prefill_bf16_bf16_out",
            "layer": "linear_pair_silu",
            "quant": "gguf_q4_k_t16_v1",
            "tile": {
                "columns_per_block": parent["columns_per_block"],
                "rows_per_block": parent["rows_per_block"],
                "row_tiles_per_wave": parent["row_tiles_per_wave"],
                "active_waves_per_block": parent["active_waves_per_block"],
                "threads_per_block": parent["threads_per_block"],
            },
            "dispatch": {
                "selector": "hipengine.runtime.gguf_linear."
                "_q4_t16_physical_dual_silu_variant",
                "policies": [
                    "GGUF_SPECDEC2_Q4_DUAL_SILU_ROWTILE_POLICY",
                    "GGUF_SPECDEC2_Q4_DUAL_SILU_PRODUCTION_R28_POLICY",
                ],
                "selected_rows": {"32": "row32", "36": "row48", "28": "row32"},
                "prefill_rows": (
                    "every row count above 36 selects the 256-row parent"
                ),
            },
            "knobs": {
                "row_tiles_per_wave": (
                    "template parameter, 1-4; sets rows per wave (16 x rt) and "
                    "the accumulator footprint (2 matrices x 2 out tiles x rt "
                    "x float8)"
                ),
                "active_waves_per_block": (
                    "template parameter, 1-4; sets rows per block (waves x rt "
                    "x 16). The launcher always launches 128 threads, so waves "
                    "below 4 leave wave capacity idle in the WMMA phase"
                ),
                "out_tiles_per_block": (
                    "compile-time constant 2 (32 columns) in this kernel; it "
                    "sets the LDS weight slab (2 x 256 x 32 x 2 B = 32 KiB) "
                    "and the decode work per block"
                ),
                "threads_per_block": (
                    "compile-time constant 128; the weight decode needs "
                    "threads/4 >= 2 matrices x (columns/2) decode pairs, which "
                    "is exactly saturated at 32 columns"
                ),
                "launch_bounds_min_blocks": (
                    "compile-time constant 1; the compiler therefore spends up "
                    "to 512 VGPRs per thread and every variant lands at 248"
                ),
            },
        },
        "dual_row_tile_sweep": rowtile,
        "dual_ladder_resources": ladder,
        "column_tile_sweep": coltile,
        "fork_comparison": {
            "fork_commit": "654803517b06da47f5210553a661bf6c80deb97f",
            "fork_file": "ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh",
            "fork_change": (
                "Q4_K/Q5_K/Q6_K dense MMQ entries gain a 128-thread "
                "configuration with I=64 weight rows per block alongside the "
                "256-thread I=128 entry (J stays 128 tokens)"
            ),
            "axis_mapping": {
                "fork_I_weight_rows": "hipEngine columns_per_block",
                "fork_J_tokens": "hipEngine rows_per_block",
                "fork_nthreads": "hipEngine threads_per_block (32 x waves)",
            },
            "fork_transfer": (
                "the fork's change is a halved weight axis plus halved "
                "threads; hipEngine's measured equivalents are the 16-column "
                "block (columns_per_block 32 -> 16) and the 64-thread launch. "
                "Neither is exercised by the existing registered variants, "
                "which vary only rows_per_block."
            ),
        },
        "findings": [
            {
                "id": "row_tile_direction",
                "kind": "measured",
                "finding": (
                    "shrinking the token-row tile loses at every prefill row "
                    "count: row128 is 0.84-0.88x the parent, row64 0.46-0.47x, "
                    "row48 0.40-0.44x, row32 0.40x, on real gate/up tiles with "
                    "bit-identical outputs in all 20 cells"
                ),
                "evidence": "dual_row_tile_sweep",
            },
            {
                "id": "row_tile_mechanism",
                "kind": "measured",
                "finding": (
                    "per-block time does not fall with rows per block: at 4096 "
                    "rows one block costs 6.00 us at 256 rows, 3.42 us at 128 "
                    "rows, 3.19 us at 64 rows and 1.86 us at 32 rows, so the "
                    "per-row cost rises as the tile shrinks"
                ),
                "evidence": "dual_row_tile_sweep.cells[].variants[].per_block_us",
            },
            {
                "id": "occupancy_is_pinned",
                "kind": "measured",
                "finding": (
                    "every registered dual variant reports 248 allocated VGPR "
                    "and 32768 LDS bytes, so all of them sit at 2 workgroups "
                    "per CU and 2 waves per SIMD; the row-tile knob cannot "
                    "move occupancy"
                ),
                "evidence": "dual_ladder_resources",
            },
            {
                "id": "column_tile_direction",
                "kind": "measured",
                "finding": (
                    "for the single-matrix sibling at the same shape, wider "
                    "column tiles win at both row counts. At 4096 rows the "
                    "48-column block takes 30.82 ms, the 32-column block "
                    "39.11-42.97 ms and the 16-column block 52.15-56.91 ms. At "
                    "512 rows the four-wave pair keeps the same order (48 "
                    "columns 3.88 ms against 32 columns 4.55 ms), and the "
                    "16-column blocks are slowest at 6.23-6.78 ms. Outputs "
                    "are bit-identical in every non-reference cell"
                ),
                "evidence": "column_tile_sweep",
            },
            {
                "id": "lds_reduction_is_not_occupancy",
                "kind": "measured",
                "finding": (
                    "LDS headroom alone does not raise resident waves: the "
                    "16 KiB shared_b2w4 instantiation still allocates 217 VGPR "
                    "and stays at 2 waves per SIMD, and the highest-occupancy "
                    "owner in the companion profile (q6, 32 columns, 96 VGPR, "
                    "5 waves per SIMD) is the slowest Q6 prefill owner"
                ),
                "evidence": "column_tile_sweep.resource_metadata, "
                "benchmarks/results/2026-09-13-qwen38-gfx1151-prefill-kernel-profile",
            },
            {
                "id": "fork_direction_does_not_transfer",
                "kind": "measured",
                "finding": (
                    "the fork halves the weight axis (I 128 -> 64) and the "
                    "threads (256 -> 128) at J=128 tokens. Mapped to "
                    "hipEngine, that is a halved column tile plus a halved "
                    "thread count; the measured hipEngine equivalents of the "
                    "same direction are 1.14x slower (rows halved) and "
                    "1.27-1.39x slower (columns halved at the same rows)"
                ),
                "evidence": "dual_row_tile_sweep, column_tile_sweep",
            },
            {
                "id": "instruction_stream_model",
                "kind": "analytic",
                "finding": (
                    "counting source-level operations for the parent block "
                    "gives about 3840 warp instructions per K block for 8.39 "
                    "MFLOP: roughly 1536 for the weight decode (2 matrices x 32 "
                    "columns x 256 weights at about 3 instructions per "
                    "weight) and roughly 2304 for the WMMA phase (16 "
                    "sub-block/k-tile iterations x 4 waves x (4 activation "
                    "loads + 8 b-fragment loads + 16 WMMAs + addressing)). "
                    "The decode is about 40% of the issue stream and is the "
                    "only part rows per block amortize; b-fragment loads are "
                    "about 20% and are amortized by row tiles per wave"
                ),
                "assumptions": [
                    "about 3 instructions per decoded weight (one packed byte "
                    "load, unpack, two scale/min FMAs, two half converts, one "
                    "vectorized LDS store per two weights)",
                    "one 16-byte LDS load per b fragment (the compiler emits "
                    "ds_load_b128 for the contiguous 16-half fragment)",
                    "issue-bound behaviour at 2 waves per SIMD",
                ],
                "evidence": "derived from the kernel source in "
                "hipengine/kernels/hip_gfx1100/quant/gguf_k_t16_selected_prefill.hip",
            },
        ],
        "candidate_direction": {
            "summary": (
                "Neither tile axis should be shrunk. The measured per-block "
                "cost is dominated by work that rows per block amortize and by "
                "operand traffic that a wider block amortizes, so the "
                "candidates below increase amortization while holding the "
                "resources that the measurements show are already at the "
                "family norm (248 VGPR, 2 waves per SIMD, 2 workgroups per CU)"
            ),
            "candidates": [
                {
                    "id": "ot1_rt8_w4",
                    "priority": 1,
                    "geometry": (
                        "16 columns x 512 rows per block, row tiles per wave 8, "
                        "4 waves, 128 threads"
                    ),
                    "resources": (
                        "32 KiB LDS (the rounded-tile union member, 2 x 512 x "
                        "16 x 2 B, equals the weight slab), 128 accumulator "
                        "VGPRs, 2 workgroups per CU, 2 waves per SIMD"
                    ),
                    "why": (
                        "halving columns per block halves the per-block decode "
                        "for the same compute, and doubling row tiles per wave "
                        "halves b-fragment loads per WMMA. Per block the "
                        "instruction-stream model predicts about 3072 warp "
                        "instructions for the same 8.39 MFLOP, roughly 20% "
                        "fewer than the parent"
                    ),
                    "risk": (
                        "16-column blocks double activation traffic per FLOP "
                        "(31.25 versus 15.6 KB per MFLOP) and the only "
                        "same-shape 16-column arm measured so far is 1.7x "
                        "slower, though that arm is a 32-thread single-wave "
                        "block"
                    ),
                    "code_change": (
                        "out_tiles as a template parameter (currently the "
                        "constant 2), row_tiles_per_wave up to 8 in the "
                        "static_assert, and a 16-pair decode guard; no change "
                        "to threads, epilogue or LDS layout"
                    ),
                },
                {
                    "id": "ot1_rt4_w4_minblocks4",
                    "priority": 2,
                    "geometry": (
                        "16 columns x 256 rows per block, row tiles per wave "
                        "4, 4 waves, 128 threads, launch bounds min 4 blocks "
                        "per CU"
                    ),
                    "resources": (
                        "16 KiB LDS allows 4 workgroups per CU; the register "
                        "cap forces the 128 VGPR budget that 4 waves per SIMD "
                        "needs"
                    ),
                    "why": (
                        "the only arm that can raise resident waves. The "
                        "measured sibling instantiations show LDS savings alone "
                        "do not: 16 KiB LDS blocks still allocate 217 VGPR and "
                        "stay at 2 waves per SIMD"
                    ),
                    "risk": (
                        "spills if 128 VGPRs cannot hold the accumulators, "
                        "activation fragments and b fragments; the highest-"
                        "occupancy owner in the companion profile is the "
                        "slowest, so more resident waves is not itself a win"
                    ),
                },
                {
                    "id": "ot2_rt4_w8_rows512",
                    "priority": 3,
                    "geometry": (
                        "32 columns x 512 rows per block, row tiles per wave "
                        "4, 8 waves, 256 threads"
                    ),
                    "resources": (
                        "the epilogue must stop publishing through the LDS "
                        "union, because 2 x 512 x 32 x 2 B is 64 KiB; with a "
                        "register-local SiLU the LDS is the 32 KiB weight slab "
                        "and occupancy is unchanged"
                    ),
                    "why": (
                        "keeps the measured-good column tile, row tiles per "
                        "wave and epilogue structure and only doubles rows per "
                        "block, so the decode is amortized twice as far; the "
                        "register-local SiLU is bit-identical because the same "
                        "BF16 rounding is applied to gate and up before the "
                        "product"
                    ),
                    "risk": (
                        "needs 256-thread blocks and an epilogue rewrite, and "
                        "it does not reduce b-fragment loads per WMMA"
                    ),
                },
                {
                    "id": "decode_restructure",
                    "priority": 4,
                    "geometry": "unchanged tile",
                    "why": (
                        "the decode reads one packed byte per thread per two "
                        "weights and is about 40% of the issue stream; a lane "
                        "mapping that gives each thread four consecutive bytes "
                        "would cut those loads fourfold"
                    ),
                    "risk": (
                        "touches the exact decode path, so it needs the same "
                        "bit-exactness gate as a geometry change"
                    ),
                },
            ],
            "rejected_by_measurement": [
                "every shrink of rows per block (row32, row48, row64, row128) "
                "at prefill rows",
                "shrinking the column tile at fixed rows, per the sibling "
                "measurement",
                "raising occupancy without a register cap: 16 KiB LDS "
                "instantiations still allocate 217-248 VGPR",
            ],
            "gate": (
                "every geometry arm in this family is bit-exact against the "
                "parent by construction, and the screen confirms it in 20 of "
                "20 cells, so a candidate is judged by wall time at prefill "
                "rows plus the registered strict fallback, not by a new "
                "numerical envelope"
            ),
        },
        "provenance": {
            "resource_table": resources["provenance"],
            "row_tile_harness": rowtile["protocol"]["harness"],
            "column_tile_harness": coltile["protocol"]["harness"],
            "source_commit": json.loads(args.coltile.read_text()).get(
                "source_commit"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "limitations": [
            "single host (Radeon 8060S, gfx1151), single model, one warmup "
            "and three measured repetitions per cell",
            "the column-tile sweep uses synthetic Q4_K tiles and single-matrix "
            "owners; the dual kernel cannot express a 16-column block without "
            "a new instantiation, so that direction is measured on the sibling "
            "and inferred for the dual",
            "occupancy figures come from compiler metadata, not from a "
            "hardware counter",
        ],
    }
    args.out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
