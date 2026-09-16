#!/usr/bin/env python3
"""Measure hipBLASLt's achieved fp16 rate on the YuE2 NAR projection shapes.

The solver spends most of its time in the per-layer projections, and they run well
below the device's fp16 peak. This probe reports the best algorithm's rate for each
real shape and for a large square GEMM, so a projection rate can be compared against
what the library itself can reach on this device.

Usage:
    python3 scripts/yue2_gemm_rate.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rows", type=int, default=1299)
    args = parser.parse_args()

    from hipengine.core.hip import HipRuntime
    from hipengine.core.hipblaslt import HipblasLt, HIP_R_16F

    runtime = HipRuntime.load()
    lt = HipblasLt()

    # (label, in_features, out_features) for the NAR projections at hidden 2048,
    # kv width 1024, q width 2048 and MLP intermediate 6144.
    hidden, kv_width, q_width, inter = 2048, 1024, 2048, 6144
    shapes = [
        ("vae2llm  (64->2048)", 64, hidden),
        ("k / v    (2048->1024)", hidden, kv_width),
        ("q        (2048->2048)", hidden, q_width),
        ("o        (2048->2048)", q_width, hidden),
        ("gate/up  (2048->6144)", hidden, inter),
        ("down     (6144->2048)", inter, hidden),
        ("square   (4096->4096)", 4096, 4096),
    ]

    rows = int(args.rows)
    print(f"rows={rows}, fp16 in / fp32 out, best of the library's own algorithm list")
    print(f"{'shape':26s} {'GFLOP':>8s} {'best algo ms':>13s} {'TFLOP/s':>9s}  algorithm")
    total_flops = 0.0
    total_seconds = 0.0
    for label, in_features, out_features in shapes:
        r = 4096 if label.startswith("square") else rows
        problem = lt.problem(r, in_features, out_features)
        workspace = 64 * 1024 * 1024
        x = runtime.malloc(in_features * r * 2)
        w = runtime.malloc(in_features * out_features * 2)
        out = runtime.malloc(out_features * r * 4)
        ws = runtime.malloc(workspace)
        flops = 2.0 * r * in_features * out_features
        best = None
        best_zero = None
        candidates = problem.algorithms(maximum=32)
        for index, candidate in enumerate(candidates):
            try:
                problem.launch(candidate, x, w, out, ws)
                runtime.device_synchronize()
            except Exception:
                continue
            started = time.perf_counter()
            for _ in range(int(args.repeat)):
                problem.launch(candidate, x, w, out, ws)
            runtime.device_synchronize()
            seconds = (time.perf_counter() - started) / int(args.repeat)
            record = (seconds, index, int(candidate.workspace_size))
            if best is None or seconds < best[0]:
                best = record
            if candidate.workspace_size == 0 and (best_zero is None or seconds < best_zero[0]):
                best_zero = record
        if best is None:
            print(f"{label:26s} {flops / 1e9:8.1f}  no launchable algorithm")
            continue
        seconds, index, workspace = best
        note = f"index {index}, ws {workspace >> 10} KiB"
        if best_zero is not None:
            note += f"; best zero-workspace {best_zero[0] * 1e3:.3f} ms ({flops / best_zero[0] / 1e12:.2f} TFLOP/s)"
        print(f"{label:26s} {flops / 1e9:8.1f} {seconds * 1e3:13.3f} {flops / seconds / 1e12:9.2f}  {note}")
        total_flops += flops
        total_seconds += seconds
        for buffer in (x, w, out, ws):
            runtime.free(buffer)
        problem.close()
    print(f"\nbest-algorithm total for one eval of this shape mix: {total_seconds * 1e3:.1f} ms "
          f"at {total_flops / total_seconds / 1e12:.2f} TFLOP/s")
    lt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
