#!/usr/bin/env python3
"""Microbench Q8_0 T16 dual-split GEMV variants.

This targets the llama-compat verifier hot leaf:
``attn_qkv + attn_gate`` with T16 Q8_0 weights. It compares the exact BF16
activation path against the llama.cpp-like q8_1+dp4a diagnostic path while
cycling weight copies to avoid measuring only cache-resident matrix reads.
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

Q8_0_BLOCK = 32
Q8_0_T16_BLOCK_BYTES = 16 * 2 + 32 * 16
Q8_1_BLOCK_BYTES = 36


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(arr, np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_shape(raw: str) -> tuple[int, int, int]:
    parts = [int(part) for part in raw.lower().split("x")]
    if len(parts) != 3:
        raise ValueError("--shape must be formatted as in_features x out_a x out_b")
    return parts[0], parts[1], parts[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--rows", default="1,2,3,4")
    ap.add_argument("--threads", default="64,128")
    ap.add_argument(
        "--modes",
        default="exact,q8_1_dp4a,prequant_q8_1_dp4a",
        help="Comma-separated: exact, rowtile2, rowtile4, q8_1_dp4a, prequant_q8_1_dp4a",
    )
    ap.add_argument(
        "--shape",
        default="2048x8192x4096",
        help="in_features x out_features_a x out_features_b; default is qwen35 linear-attn qkv+gate",
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--peak-gbs", type=float, default=256.0)
    ap.add_argument(
        "--mall-bytes",
        type=int,
        default=32 * 1024 * 1024,
        help="Weight pool is sized >2x this cache estimate and cycled per launch.",
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    rows_list = _parse_csv_ints(args.rows)
    threads_list = _parse_csv_ints(args.threads)
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    allowed_modes = {"exact", "rowtile2", "rowtile4", "q8_1_dp4a", "prequant_q8_1_dp4a"}
    unknown_modes = sorted(set(modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"unknown modes: {', '.join(unknown_modes)}")
    in_f, out_a, out_b = _parse_shape(args.shape)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
        build_gguf_q8_0_t16_gemv,
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
        gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out,
        gguf_q8_0_t16_dual_gemv_decode_rowtile2_bf16_bf16_out,
        gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16
    from tests._gguf_synthetic_weights import make_q8_0_weight

    rt = get_hip_runtime()
    lib = build_gguf_q8_0_t16_gemv(load=True, require_cached=args.require_cached_build)
    q4_lib = build_gguf_q4_k_gemv(load=True, require_cached=args.require_cached_build)
    rng = np.random.default_rng(701)

    def matrix_bytes(out_features: int) -> int:
        return (out_features // 16) * (in_f // Q8_0_BLOCK) * Q8_0_T16_BLOCK_BYTES

    matrix_a = matrix_bytes(out_a)
    matrix_b = matrix_bytes(out_b)
    matrix_total = matrix_a + matrix_b
    pool = max(2, (2 * args.mall_bytes) // max(matrix_total, 1) + 1)

    qa = make_q8_0_weight(out_a, in_f)
    qb = make_q8_0_weight(out_b, in_f)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles

    def bench(rows: int, threads: int, mode: str) -> dict:
        x = _f32_to_bf16(rng.standard_normal((rows, in_f)) * 0.1)
        xb = malloc(x.nbytes, runtime=rt)
        copy_host_to_device(xb, host_array_ptr(x), runtime=rt)
        xq_buf = malloc(rows * (in_f // Q8_0_BLOCK) * Q8_1_BLOCK_BYTES, runtime=rt)
        a_pool = []
        b_pool = []
        for _ in range(pool):
            ab = malloc(ta.nbytes, runtime=rt)
            bb = malloc(tb.nbytes, runtime=rt)
            copy_host_to_device(ab, host_array_ptr(ta), runtime=rt)
            copy_host_to_device(bb, host_array_ptr(tb), runtime=rt)
            a_pool.append(ab)
            b_pool.append(bb)
        out_a_buf = malloc(rows * out_a * 2, runtime=rt)
        out_b_buf = malloc(rows * out_b * 2, runtime=rt)
        try:
            def go(i: int) -> None:
                if mode in {"exact", "rowtile2", "rowtile4"}:
                    if mode == "rowtile2":
                        exact_fn = gguf_q8_0_t16_dual_gemv_decode_rowtile2_bf16_bf16_out
                    elif mode == "rowtile4":
                        exact_fn = gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out
                    else:
                        exact_fn = gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out
                    exact_fn(
                        xb.ptr,
                        a_pool[i % pool].ptr,
                        b_pool[i % pool].ptr,
                        out_a_buf.ptr,
                        out_b_buf.ptr,
                        rows,
                        in_f,
                        out_a,
                        out_b,
                        threads=threads,
                        library=lib,
                        runtime=rt,
                    )
                    return
                if mode == "q8_1_dp4a":
                    gguf_q4_k_quantize_bf16_q8_1(xb.ptr, xq_buf.ptr, rows, in_f, library=q4_lib, runtime=rt)
                gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out(
                    xq_buf.ptr,
                    a_pool[i % pool].ptr,
                    b_pool[i % pool].ptr,
                    out_a_buf.ptr,
                    out_b_buf.ptr,
                    rows,
                    in_f,
                    out_a,
                    out_b,
                    threads=threads,
                    library=lib,
                    runtime=rt,
                )

            if mode == "prequant_q8_1_dp4a":
                gguf_q4_k_quantize_bf16_q8_1(xb.ptr, xq_buf.ptr, rows, in_f, library=q4_lib, runtime=rt)

            for i in range(args.warmup):
                go(i)
            rt.device_synchronize()
            t0 = time.perf_counter()
            for i in range(args.iters):
                go(i)
            rt.device_synchronize()
            ms = (time.perf_counter() - t0) / args.iters * 1000.0
        finally:
            for buf in (xb, xq_buf, out_a_buf, out_b_buf, *a_pool, *b_pool):
                free(buf, runtime=rt)

        if mode == "rowtile2":
            matrix_read_groups = (rows + 1) // 2
        elif mode == "rowtile4":
            matrix_read_groups = (rows + 3) // 4
        else:
            matrix_read_groups = rows
        read_bytes = matrix_total * matrix_read_groups
        bw = read_bytes / (ms / 1000.0) / 1e9
        return {
            "mode": mode,
            "rows": rows,
            "threads": threads,
            "in_features": in_f,
            "out_features_a": out_a,
            "out_features_b": out_b,
            "us": round(ms * 1000.0, 2),
            "matrix_MB": round(matrix_total / 1e6, 3),
            "matrix_read_groups": matrix_read_groups,
            "pool_copies": pool,
            "pool_MB": round(pool * matrix_total / 1e6, 1),
            "achieved_read_bw_gbs": round(bw, 1),
            "pct_peak": round(bw / args.peak_gbs * 100.0, 1),
        }

    results = []
    for rows in rows_list:
        for threads in threads_list:
            for mode in modes:
                result = bench(rows, threads, mode)
                results.append(result)
                print(
                    f"mode={mode:21s} rows={rows} threads={threads:3d} in={in_f} out=({out_a},{out_b}) "
                    f"{result['us']:8.2f}us matMB={result['matrix_MB']:6.3f} "
                    f"groups={result['matrix_read_groups']:2d} pool={result['pool_MB']:6.1f}MB "
                    f"BW={result['achieved_read_bw_gbs']:6.1f}GB/s "
                    f"({result['pct_peak']:4.1f}% peak)"
                )

    out = {
        "schema": "hipengine.gguf_q8_0_t16_pair_microbench.v3",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "peak_gbs": args.peak_gbs,
        "modes": modes,
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
