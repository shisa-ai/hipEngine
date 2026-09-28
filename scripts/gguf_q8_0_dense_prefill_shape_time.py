#!/usr/bin/env python3
"""Isolated Q8_0 dense prefill timing: mean against best-of-N for one shape.

``scripts/gemma4_prefill_dense_probe.py`` reports a per-call mean over a real
prefill step and the sweep recorded in ``gguf_q8_0_prefill.py``'s docstring
reports a best of 8.  Comparing the two gave every shape a 10 to 31 percent
in-situ penalty, but a best-of-N estimator is optimistic by construction, so the
comparison mixes two effects.

This probe isolates the estimator: one shape, one tile, one weight matrix, no
other kernel resident, and it reports the mean, the median, the best and the
worst over the same samples.  Whatever gap survives the estimator change is the
environment rather than the statistic.

Diagnostic only; no perf claim retained from a single run.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from _gguf_synthetic_weights import make_q8_0_weight  # noqa: E402
from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_host_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (  # noqa: E402
    gguf_q8_0_wmma_prefill_bf16_bf16_out as prefill,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--in-features", type=int, nargs="+", default=[4096, 2816, 2816])
    ap.add_argument("--out-features", type=int, nargs="+", default=[2816, 4096, 2112])
    ap.add_argument("--tile-m", type=int, default=32)
    ap.add_argument("--tile-n", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    args = ap.parse_args()

    if len(args.in_features) != len(args.out_features):
        raise SystemExit("--in-features and --out-features must have equal length")

    runtime = get_hip_runtime()
    stream = 0

    print(f"rows={args.rows} tile=({args.tile_m}, {args.tile_n}) "
          f"iters={args.iters} warmup={args.warmup}")
    print(f"{'in':>6} {'out':>6} {'mean':>9} {'median':>9} {'best':>9} "
          f"{'worst':>9} {'mean/best':>9} {'TFLOP/s':>8}")

    for in_features, out_features in zip(args.in_features, args.out_features):
        qweight = make_q8_0_weight(out_features, in_features)
        x = np.random.default_rng(3).standard_normal(
            (args.rows, in_features)).astype(np.float32).astype(np.float16)
        x = (x.astype(np.float32) * 0.05).astype(np.float16)

        w_dev = malloc(qweight.nbytes)
        x_dev = malloc(x.nbytes)
        out_dev = malloc(args.rows * out_features * 2)
        copy_host_to_device(w_dev, host_array_ptr(qweight), qweight.nbytes)
        copy_host_to_device(x_dev, host_array_ptr(x), x.nbytes)

        for _ in range(args.warmup):
            prefill(x_dev.ptr, w_dev.ptr, out_dev.ptr, args.rows, in_features,
                    out_features, tile_m=args.tile_m, tile_n=args.tile_n, stream=stream)
        runtime.stream_synchronize(stream)

        start_ev = runtime.event_create()
        stop_ev = runtime.event_create()
        times = []
        for _ in range(args.iters):
            runtime.event_record(start_ev, stream)
            prefill(x_dev.ptr, w_dev.ptr, out_dev.ptr, args.rows, in_features,
                    out_features, tile_m=args.tile_m, tile_n=args.tile_n, stream=stream)
            runtime.event_record(stop_ev, stream)
            runtime.event_synchronize(stop_ev)
            times.append(runtime.event_elapsed_time_ms(start_ev, stop_ev))
        runtime.event_destroy(start_ev)
        runtime.event_destroy(stop_ev)

        t = np.array(times)
        flops = 2.0 * args.rows * in_features * out_features
        q = max(1, args.iters // 4)
        first, last = t[:q].mean(), t[-q:].mean()
        print(f"{in_features:>6} {out_features:>6} {t.mean():>9.3f} "
              f"{np.median(t):>9.3f} {t.min():>9.3f} {t.max():>9.3f} "
              f"{t.mean() / t.min():>9.3f} "
              f"{flops / 1e9 / t.mean():>8.2f}")
        print(f"{'':>6} {'':>6} first q{args.iters//4} {first:.3f}  last q{args.iters//4} "
              f"{last:.3f}  drift {last / first:>6.3f}")

        # Host-side dispatch cost: launch the same kernel back to back with no
        # intervening synchronize and time the wall clock of the loop. If the
        # host takes longer than the GPU work it queues, the engine is
        # host-bound; if it is much faster, the in-situ gap is elsewhere.
        runtime.stream_synchronize(stream)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            prefill(x_dev.ptr, w_dev.ptr, out_dev.ptr, args.rows, in_features,
                    out_features, tile_m=args.tile_m, tile_n=args.tile_n, stream=stream)
        host_ms = (time.perf_counter() - t0) * 1e3 / args.iters
        runtime.stream_synchronize(stream)
        print(f"{'':>6} {'':>6} host dispatch {host_ms:.4f} ms/launch  "
              f"gpu {t.mean():.4f} ms/launch  host/gpu {host_ms / t.mean():.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
