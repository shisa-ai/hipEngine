#!/usr/bin/env python3
"""Achieved-bandwidth microbench for the dense GGUF Q8_0 T16 decode GEMV.

The 2026-06-29 ``scripts/gguf_decode_rocprof.py`` breakdown showed dense Q8_0
GEMV (attention projections + shared-expert FFN, all Q8_0) is ~47% of the
~18 ms/token GGUF decode wall on gfx1151 — the single largest family, far above
the selected-MoE GEMV (~25%, separately measured at 70-80% of peak).  This probe
isolates the dense Q8_0 single-output decode kernel and reports the **achieved
weight-read bandwidth** (matrix bytes / kernel time) across decode row counts and
the model's real projection shapes.

It is the diagnostic backing task #10: at rows=1 (the decode case) the dense Q8_0
GEMV reads weights at only ~20-28% of the ~256 GB/s LPDDR5X peak, vs the
selected-MoE GEMV's 70-80%.  That gap — not the MoE GEMV, not dp4a compute — is
the bandwidth-efficiency lever toward llama.cpp's ~1.9x.

Diagnostic only; no perf claim retained from a single run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Q8_0 T16 tile-block = 16 fp16 col scales (32 B) + 16 cols * 32 int8 weights (512 B).
# One T16 block spans Q8_0_BLOCK = 32 contraction (k) values, so
# blocks_per_row = in_features / 32 (NOT /256 -- that is the K-quant super-block).
Q8_0_BLOCK = 32
Q8_0_T16_BLOCK_BYTES = 16 * 2 + 32 * 16


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(arr, np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument(
        "--shapes",
        nargs="+",
        default=["2048x2048", "2048x6144", "768x2048"],
        help="in_features x out_features pairs (real Qwen3.6 dense projection shapes)",
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--peak-gbs", type=float, default=256.0)
    ap.add_argument(
        "--mall-bytes",
        type=int,
        default=32 * 1024 * 1024,
        help="Strix Halo MALL/Infinity Cache size; the weight pool is sized >2x this "
        "and cycled per-iter so each launch reads cold DRAM, not cache.",
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
        build_gguf_q8_0_t16_gemv,
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16
    from tests._gguf_synthetic_weights import make_q8_0_weight

    rt = get_hip_runtime()
    lib = build_gguf_q8_0_t16_gemv(load=True, require_cached=args.require_cached_build)
    rng = np.random.default_rng(629)

    def bench(rows: int, in_f: int, out_f: int) -> dict:
        x = _f32_to_bf16(rng.standard_normal((rows, in_f)) * 0.1)
        tiles = repack_gguf_q8_0_tile16(make_q8_0_weight(out_f, in_f)).tiles
        matrix_bytes = (out_f // 16) * (in_f // Q8_0_BLOCK) * Q8_0_T16_BLOCK_BYTES
        # Defeat the 32 MB MALL/Infinity Cache: a pool of distinct weight copies
        # sized >2x MALL, cycled per launch so every launch reads cold DRAM.
        pool = max(2, (2 * args.mall_bytes) // max(matrix_bytes, 1) + 1)
        xb = malloc(x.nbytes, runtime=rt)
        copy_host_to_device(xb, host_array_ptr(x), runtime=rt)
        tbs = []
        for _ in range(pool):
            tb = malloc(tiles.nbytes, runtime=rt)
            copy_host_to_device(tb, host_array_ptr(tiles), runtime=rt)
            tbs.append(tb)
        ob = malloc(rows * out_f * 2, runtime=rt)
        try:
            def go(i: int) -> None:
                gguf_q8_0_t16_gemv_decode_bf16_bf16_out(
                    xb.ptr, tbs[i % pool].ptr, ob.ptr, rows, in_f, out_f, library=lib, runtime=rt
                )

            for i in range(args.warmup):
                go(i)
            rt.device_synchronize()
            t0 = time.perf_counter()
            for i in range(args.iters):
                go(i)
            rt.device_synchronize()
            ms = (time.perf_counter() - t0) / args.iters * 1000.0
        finally:
            for b in (xb, ob, *tbs):
                free(b, runtime=rt)
        # Decode rereads the full weight matrix for every row (no reuse in this kernel).
        read_bytes = matrix_bytes * rows
        bw = read_bytes / (ms / 1000.0) / 1e9
        return {
            "rows": rows,
            "in_features": in_f,
            "out_features": out_f,
            "us": round(ms * 1000.0, 2),
            "matrix_MB": round(matrix_bytes / 1e6, 3),
            "pool_copies": pool,
            "pool_MB": round(pool * matrix_bytes / 1e6, 1),
            "achieved_read_bw_gbs": round(bw, 1),
            "pct_peak": round(bw / args.peak_gbs * 100.0, 1),
        }

    results = []
    for shape in args.shapes:
        in_f, out_f = (int(v) for v in shape.lower().split("x"))
        for rows in args.rows:
            r = bench(rows, in_f, out_f)
            results.append(r)
            print(
                f"in={in_f:5d} out={out_f:5d} rows={rows} "
                f"{r['us']:8.2f}us matMB={r['matrix_MB']:6.3f} pool={r['pool_MB']:6.0f}MB "
                f"BW={r['achieved_read_bw_gbs']:6.1f}GB/s ({r['pct_peak']:4.1f}% peak)"
            )

    out = {
        "schema": "hipengine.gguf_q8_0_dense_bw_microbench.v1",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "peak_gbs": args.peak_gbs,
        "iters": args.iters,
        "warmup": args.warmup,
        "results": results,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
