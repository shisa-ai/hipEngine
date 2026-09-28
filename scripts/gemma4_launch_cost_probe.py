#!/usr/bin/env python3
"""Measure the cost of one elementwise launch on this host.

The decode launch census says 448 launches per step move zero weight bytes and
cost 10.37 ms -- 23.1 us each. That number decides the fix: if it is launch
overhead, fuse launches; if it is real small-kernel work, make the kernels
faster.

This probe launches one production elementwise kernel at the production decode
shape (rows=1, hidden=2816) in a tight back-to-back loop and reports wall time
per launch, with and without a synchronize between launches. The difference
separates submission cost from execution cost.

Usage:
    env -u ROCR_VISIBLE_DEVICES PYTHONPATH=. python3 scripts/gemma4_launch_cost_probe.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, malloc
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import gemma4_rmsnorm_f32w_bf16
from hipengine.loading.materialize import float_array_to_bf16_bits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2816)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--sync-every", type=int, default=0,
                    help="0 = never sync inside the loop; N = sync every N launches")
    args = ap.parse_args()

    runtime = get_hip_runtime()
    h = args.hidden

    x = malloc(h * args.rows * 2)
    w = malloc(h * 4)
    out = malloc(h * args.rows * 2)
    try:
        copy_host_to_device(x, float_array_to_bf16_bits(
            np.ones((args.rows, h), dtype=np.float32)).ctypes.data, h * args.rows * 2)
        copy_host_to_device(w, np.ones(h, dtype=np.float32).ctypes.data, h * 4)

        def run(n: int) -> float:
            t0 = time.perf_counter()
            for i in range(n):
                gemma4_rmsnorm_f32w_bf16(x.ptr, w.ptr, out.ptr, args.rows, h, 1e-6, stream=0)
                if args.sync_every and (i + 1) % args.sync_every == 0:
                    runtime.device_synchronize()
            runtime.device_synchronize()
            return (time.perf_counter() - t0) / n

        run(50)  # warm the JIT cache and the queue

        print(f"shape rows={args.rows} hidden={h}   iters={args.iters}")
        for label, sync in (("no sync in loop", 0), ("sync every launch", 1),
                            ("sync every 32", 32)):
            args.sync_every = sync
            per = run(args.iters)
            print(f"  {label:20s} {per * 1e6:8.2f} us/launch")
        print()
        print("  census's elementwise average:    23.10 us/launch (448 launches, 10.37 ms)")
    finally:
        free(x)
        free(w)
        free(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
