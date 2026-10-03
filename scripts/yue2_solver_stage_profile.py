#!/usr/bin/env python3
"""Break down one YuE2 NAR velocity evaluation into its stages.

The e2e gate reports one number for a whole solve, and a kernel trace inflates every
dispatch (its per-dispatch overhead is comparable to the elementwise kernels it is
measuring). This script loads the same product configuration and reports, per
velocity evaluation, the device time (HIP events), the host enqueue time, and the
stage split by ablation:

    --skip none       full evaluation
    --skip attention  the NAR attention launch is suppressed
    --skip gemm       the projection launches are suppressed

Differences between the three runs give the attention and projection shares without
inserting a synchronize between kernels, which would destroy the pipeline.

Usage:
    python3 scripts/yue2_solver_stage_profile.py --case mandarin-off-s1234 --evals 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _resolve(env: str, pattern: str) -> Path:
    import os

    value = os.environ.get(env)
    if value:
        return Path(value)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    matches = sorted(cache.glob(pattern))
    if not matches:
        raise SystemExit(f"no model matching {pattern}; set {env}")
    return matches[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="mandarin-off-s1234")
    parser.add_argument("--cases", default=str(REPO / "tests" / "fixtures" / "yue2" / "cases"))
    parser.add_argument("--evals", type=int, default=3)
    parser.add_argument("--skip", default="none", choices=("none", "attention", "gemm"))
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    import numpy as np

    from hipengine.loading.yue2 import load_yue2_weights
    from hipengine.kernels.hip_gfx1100.yue2 import nar as nar_kernels
    from hipengine.runtime.yue2_ar import Yue2ArRuntime
    from hipengine.runtime.yue2_nar import Yue2NarRuntime, solver_schedule, song_chunks

    cases_dir = Path(args.cases)
    arrays = np.load(cases_dir / f"{args.case}.npz")
    prefix = [int(v) for v in arrays["prefix"]]
    codec = [int(v) for v in arrays["semantic"]]

    weights = load_yue2_weights(_resolve("YUE2_MODEL_DIR", "models--m-a-p--YuE2-3B/snapshots/*"))
    ar = Yue2ArRuntime(weights, branches=2)
    nar = Yue2NarRuntime(weights, ar)
    runtime = nar.runtime
    skip = str(args.skip)

    def gpu_ms(fn):
        start = runtime.event_create()
        stop = runtime.event_create()
        runtime.event_record(start)
        fn()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        elapsed = runtime.event_elapsed_time_ms(start, stop)
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
        return elapsed

    # --- stage instrumentation ----------------------------------------------------
    attention_calls: list[tuple[int, int]] = []
    gemm_calls: list[tuple[int, int, int]] = []
    attention_name = "nar_attention_wmma" if nar._wmma_attention else "nar_attention_f32"
    attention = getattr(nar_kernels, attention_name)

    def timed_attention(q, nk, nv, ak, av, out, rows, keys, heads, kv_heads, head_dim, scale, **kw):
        attention_calls.append((rows, keys))
        if skip == "attention":
            return 0
        return attention(q, nk, nv, ak, av, out, rows, keys, heads, kv_heads, head_dim, scale, **kw)

    setattr(nar_kernels, attention_name, timed_attention)

    original_gemm = type(nar)._gemm

    def timed_gemm(self, x16, weight16, out, rows, inputs, outputs):
        gemm_calls.append((rows, inputs, outputs))
        if skip == "gemm":
            return
        original_gemm(self, x16, weight16, out, rows, inputs, outputs)

    type(nar)._gemm = timed_gemm

    # --- conditioning -------------------------------------------------------------
    chunk = song_chunks(prefix, codec, int(args.case.rsplit("-s", 1)[-1]))[0]
    ar.reset(branch=0)
    started = time.perf_counter()
    nar.condition(chunk)
    condition_wall = time.perf_counter() - started
    schedule = solver_schedule(32)

    # --- evaluations --------------------------------------------------------------
    evals: list[dict] = []
    for index in range(int(args.evals)):
        raw = schedule[index % len(schedule)][0]
        attention_calls.clear()
        gemm_calls.clear()
        runtime.device_synchronize()
        started = time.perf_counter()
        nar._velocity(nar._state, raw, nar._first_velocity)
        host_seconds = time.perf_counter() - started
        runtime.device_synchronize()
        wall_seconds = time.perf_counter() - started
        device_ms = gpu_ms(lambda: nar._velocity(nar._state, raw, nar._first_velocity))
        evals.append({
            "wall_seconds": wall_seconds,
            "host_enqueue_seconds": host_seconds,
            "device_seconds": device_ms / 1e3,
            "attention_calls": list(attention_calls),
            "gemm_calls": list(gemm_calls),
        })

    med_wall = float(np.median([e["wall_seconds"] for e in evals]))
    med_device = float(np.median([e["device_seconds"] for e in evals]))
    med_host = float(np.median([e["host_enqueue_seconds"] for e in evals]))
    gemm_flops = sum(2.0 * r * i * o for r, i, o in evals[0]["gemm_calls"]) if evals else 0.0

    print(f"case {args.case}: ar_tokens={len(prefix) + len(codec)} nar_frames={len(codec)}"
          f"  skip={skip}")
    print(f"  condition() wall                  {condition_wall * 1e3:9.2f} ms")
    print(f"  per eval (median of {len(evals)}):")
    print(f"    device (HIP events)             {med_device * 1e3:9.2f} ms")
    print(f"    wall (enqueue + drain)          {med_wall * 1e3:9.2f} ms")
    print(f"    host enqueue only               {med_host * 1e3:9.2f} ms"
          f"  (throttled by queue depth once the device is behind)")
    print(f"    NAR attention launches          {len(evals[0]['attention_calls'])}"
          f"  {evals[0]['attention_calls'][:1]}")
    print(f"    projection launches             {len(evals[0]['gemm_calls'])}"
          f"  {gemm_flops / 1e9:.1f} GFLOP")
    shapes: dict[tuple[int, int, int], int] = {}
    for rows, inputs, outputs in evals[0]["gemm_calls"]:
        shapes[(rows, inputs, outputs)] = shapes.get((rows, inputs, outputs), 0) + 1
    for (rows, inputs, outputs), count in sorted(shapes.items(), key=lambda kv: -kv[1]):
        print(f"      rows={rows:5d} K={inputs:5d} N={outputs:5d}  n={count:3d}"
              f"  {2.0 * rows * inputs * outputs * count / 1e9:8.2f} GFLOP")
    print(f"  32-step projection from these numbers:"
          f" {med_device * 64 + condition_wall:.2f} s device"
          f" / {med_wall * 64 + condition_wall:.2f} s wall")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "case": args.case,
            "skip": skip,
            "nar_frames": len(codec),
            "ar_tokens": len(prefix) + len(codec),
            "condition_seconds": condition_wall,
            "eval_median_device_seconds": med_device,
            "eval_median_wall_seconds": med_wall,
            "eval_median_host_enqueue_seconds": med_host,
            "attention_calls_per_eval": len(evals[0]["attention_calls"]),
            "attention_shape": evals[0]["attention_calls"][:1],
            "projection_calls_per_eval": len(evals[0]["gemm_calls"]),
            "projection_gflop_per_eval": gemm_flops / 1e9,
            "projection_shapes": {f"{r}x{i}x{o}": c for (r, i, o), c in shapes.items()},
            "projected_32_step_device_seconds": med_device * 64 + condition_wall,
            "evals": evals,
        }, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
