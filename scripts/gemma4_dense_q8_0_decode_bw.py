#!/usr/bin/env python3
"""Achieved-bandwidth probe for the raw-GGUF dense Q8_0 decode GEMV.

Gemma 4 26B-A4B keeps every non-expert projection at Q8_0: the attention
q/k/v/o, the dense FFN gate/up/down and the tied 262144-wide lm head. At
decode those are 2.35 GB of the 3.50 GB the model reads per token, and at
``rows == 1`` they all resolve to ``gguf_q8_0/gemv_bf16_bf16_out`` -- the
raw-layout kernel whose launch is one 128-thread block per output row.

This probe times that kernel alone at the artifact's own shapes and reports
achieved weight-read bandwidth, so the decode gap has a measured per-shape
starting point instead of an assumed one. It is diagnostic only.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \
        .venv/bin/python scripts/gemma4_dense_q8_0_decode_bw.py --json out.json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34

# in_features, out_features -- the artifact's own dense projection shapes.
DEFAULT_SHAPES = [
    ("attn_q", 2816, 2816),
    ("attn_k", 2816, 2816),
    ("attn_v", 2816, 2816),
    ("attn_output", 2816, 2816),
    ("ffn_gate", 2816, 2112),
    ("ffn_up", 2816, 2112),
    ("ffn_down", 2112, 2816),
    ("lm_head", 2816, 262144),
]


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(arr, np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1])
    ap.add_argument("--shapes", nargs="+", default=None)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument(
        "--kernels",
        nargs="+",
        default=["gemv", "pack8"],
        help="which raw Q8_0 GEMV owners to time",
    )
    ap.add_argument("--peak-gbs", type=float, default=256.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        build_gguf_k_gemv,
        gguf_q8_0_gemv_bf16_bf16_out,
        gguf_q8_0_pack8_gemv_bf16_bf16_out,
    )
    from tests._gguf_synthetic_weights import make_q8_0_weight

    owners = {
        "gemv": gguf_q8_0_gemv_bf16_bf16_out,
        "pack8": gguf_q8_0_pack8_gemv_bf16_bf16_out,
    }
    for k in args.kernels:
        if k not in owners:
            raise SystemExit(f"unknown kernel {k!r}; known: {sorted(owners)}")

    rt = get_hip_runtime()
    lib = build_gguf_k_gemv(load=True)
    rng = np.random.default_rng(1151)

    shapes = DEFAULT_SHAPES
    if args.shapes:
        shapes = []
        for spec in args.shapes:
            name, rest = spec.split("=", 1)
            i, o = (int(v) for v in rest.lower().split("x"))
            shapes.append((name, i, o))

    results = []
    for name, in_f, out_f in shapes:
        weights = make_q8_0_weight(out_f, in_f)
        matrix_bytes = out_f * (in_f // Q8_0_BLOCK) * Q8_0_BLOCK_BYTES
        wb = malloc(weights.nbytes, runtime=rt)
        copy_host_to_device(wb, host_array_ptr(weights), runtime=rt)
        try:
            for rows in args.rows:
                x = _f32_to_bf16(rng.standard_normal((rows, in_f)) * 0.1)
                xb = malloc(x.nbytes, runtime=rt)
                copy_host_to_device(xb, host_array_ptr(x), runtime=rt)
                ob = malloc(rows * out_f * 2, runtime=rt)
                try:
                    for kernel_name in args.kernels:
                        fn = owners[kernel_name]

                        def go() -> None:
                            fn(
                                xb.ptr, wb.ptr, ob.ptr, rows, in_f, out_f,
                                threads=args.threads, library=lib, runtime=rt,
                            )

                        for _ in range(args.warmup):
                            go()
                        rt.device_synchronize()
                        t0 = time.perf_counter()
                        for _ in range(args.iters):
                            go()
                        rt.device_synchronize()
                        ms = (time.perf_counter() - t0) / args.iters * 1000.0
                        read_bytes = matrix_bytes * rows
                        bw = read_bytes / (ms / 1000.0) / 1e9
                        grid_blocks = (
                            out_f * rows
                            if kernel_name == "gemv"
                            else (out_f // 8) * rows
                        )
                        row = {
                            "name": name,
                            "kernel": kernel_name,
                            "rows": rows,
                            "in_features": in_f,
                            "out_features": out_f,
                            "us": round(ms * 1000.0, 2),
                            "matrix_MB": round(matrix_bytes / 1e6, 3),
                            "grid_blocks": grid_blocks,
                            "achieved_read_bw_gbs": round(bw, 1),
                            "pct_peak": round(bw / args.peak_gbs * 100.0, 1),
                        }
                        results.append(row)
                        print(
                            f"{kernel_name:6s} {name:12s} in={in_f:5d} out={out_f:6d} "
                            f"rows={rows} {row['us']:9.2f}us "
                            f"matMB={row['matrix_MB']:8.3f} "
                            f"blocks={row['grid_blocks']:7d} "
                            f"BW={row['achieved_read_bw_gbs']:6.1f}GB/s "
                            f"({row['pct_peak']:4.1f}% peak)"
                        )
                finally:
                    for b in (xb, ob):
                        free(b, runtime=rt)
        finally:
            free(wb, runtime=rt)

    out = {
        "schema": "hipengine.gemma4_dense_q8_0_decode_bw.v1",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "peak_gbs": args.peak_gbs,
        "threads": args.threads,
        "iters": args.iters,
        "warmup": args.warmup,
        "kernel": "hipengine_gguf_q8_0_gemv_bf16_bf16_out vs pack8",
        "results": results,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
