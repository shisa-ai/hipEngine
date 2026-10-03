#!/usr/bin/env python3
"""Achieved-bandwidth probe for the Gemma 4 decode MoE selected GEMVs.

The route ablation puts 14.67 ms of the 44.84 ms Gemma 4 gfx1151 decode step in
the routed-expert FFN, and the route census shows 59 of 60 projections per
forward take the ``selected_gemv`` family: Q4_K for the gate/up projection and
Q5_1 for the down projection. This probe times each of those two kernels at the
artifact's own shapes so the expert share has a per-kernel starting point.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python3 scripts/gemma4_expert_selected_bw.py --json out.json
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

# The artifact's routed-expert projections: (label, quant, in, out, bytes/elem).
DEFAULT_SHAPES = [
    ("gate_up_q4_k", "q4_k", 2816, 1408),
    ("down_q5_1", "q5_1", 704, 2816),
]


def _f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(arr, np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _expert_matrix(quant: str, out_f: int, in_f: int, experts: int) -> np.ndarray:
    from tests._gguf_synthetic_weights import make_q4_k_weight, make_q5_1_weight

    maker = {"q4_k": make_q4_k_weight, "q5_1": make_q5_1_weight}[quant]
    one = maker(out_f, in_f)
    return np.ascontiguousarray(np.tile(one, (experts, 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[8])
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="0 = each kernel's production default (Q5_1 256, Q4_K 128)",
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--peak-gbs", type=float, default=256.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_selected_gemv_bf16_bf16_out,
        gguf_q4_k_selected_pack8_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        build_qwen4_exp_q5_1,
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
    )

    rt = get_hip_runtime()
    q4_lib = build_gguf_q4_k_gemv(load=True)
    q5_lib = build_qwen4_exp_q5_1(load=True)
    rng = np.random.default_rng(704)

    results = []
    for label, quant, in_f, out_f in DEFAULT_SHAPES:
        if quant == "q4_k" and in_f % 256:
            raise SystemExit(f"{label}: the Q4_K fixture needs in_features % 256 == 0")
        matrix = _expert_matrix(quant, out_f, in_f, args.experts)
        row_bytes = matrix.shape[1]
        matrix_bytes = matrix.nbytes
        wb = malloc(matrix.nbytes, runtime=rt)
        copy_host_to_device(wb, host_array_ptr(matrix), runtime=rt)
        try:
            for rows in args.rows:
                x = _f32_to_bf16(rng.standard_normal((rows, in_f)) * 0.1)
                selected = np.array(
                    [(i * 7) % args.experts for i in range(rows)], dtype=np.int64
                )
                xb = malloc(x.nbytes, runtime=rt)
                copy_host_to_device(xb, host_array_ptr(x), runtime=rt)
                sb = malloc(selected.nbytes, runtime=rt)
                copy_host_to_device(sb, host_array_ptr(selected), runtime=rt)
                ob = malloc(rows * out_f * 2, runtime=rt)
                try:
                    launch_threads = args.threads or (256 if quant == "q5_1" else 128)
                    variants: list[tuple[str, object, object]] = []
                    if quant == "q4_k":
                        variants = [
                            ("selected", gguf_q4_k_selected_gemv_bf16_bf16_out, q4_lib),
                            (
                                "selected_pack8",
                                gguf_q4_k_selected_pack8_gemv_bf16_bf16_out,
                                q4_lib,
                            ),
                        ]
                    else:
                        variants = [
                            (
                                "selected",
                                qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
                                q5_lib,
                            ),
                        ]
                    for variant_name, fn, lib in variants:
                        def go() -> None:
                            fn(
                                xb.ptr, sb.ptr, wb.ptr, ob.ptr, rows, rows,
                                args.experts, in_f, out_f,
                                threads=launch_threads, library=lib, runtime=rt,
                            )

                        for _ in range(args.warmup):
                            go()
                        rt.device_synchronize()
                        t0 = time.perf_counter()
                        for _ in range(args.iters):
                            go()
                        rt.device_synchronize()
                        ms = (time.perf_counter() - t0) / args.iters * 1000.0

                        # Every launch reads one row of each of the `rows` experts.
                        read_bytes = matrix_bytes // args.experts * rows
                        bw = read_bytes / (ms / 1000.0) / 1e9
                        grid_blocks = rows * (
                            out_f if variant_name == "selected" else out_f // 8
                        )
                        row = {
                            "label": label,
                            "variant": variant_name,
                            "quant": quant,
                            "rows": rows,
                            "experts": args.experts,
                            "threads": launch_threads,
                            "in_features": in_f,
                            "out_features": out_f,
                            "row_bytes": row_bytes,
                            "us": round(ms * 1000.0, 2),
                            "read_MB": round(read_bytes / 1e6, 3),
                            "grid_blocks": grid_blocks,
                            "achieved_read_bw_gbs": round(bw, 1),
                            "pct_peak": round(bw / args.peak_gbs * 100.0, 1),
                        }
                        results.append(row)
                        print(
                            f"{label:14s} {variant_name:16s} rows={rows} "
                            f"t={launch_threads:3d} in={in_f:5d} out={out_f:5d} "
                            f"{row['us']:9.2f}us readMB={row['read_MB']:8.3f} "
                            f"blocks={row['grid_blocks']:6d} "
                            f"BW={row['achieved_read_bw_gbs']:6.1f}GB/s "
                            f"({row['pct_peak']:4.1f}% peak)"
                        )
                finally:
                    for b in (xb, sb, ob):
                        free(b, runtime=rt)
        finally:
            free(wb, runtime=rt)

    out = {
        "schema": "hipengine.gemma4_expert_selected_bw.v1",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "peak_gbs": args.peak_gbs,
        "threads": args.threads,
        "iters": args.iters,
        "warmup": args.warmup,
        "results": results,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
