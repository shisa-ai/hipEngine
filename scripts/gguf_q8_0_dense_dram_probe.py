#!/usr/bin/env python3
"""One Q8_0 dense prefill GEMM shape, launched a fixed number of times.

Throwaway probe for measuring DRAM read traffic with ``rocprofv3 --pmc``.  It
does no counter collection itself: it exists so the kernel under test can be
driven with a known launch count and a kernel filter, and so the weight size and
the *predicted* re-read multiplier are printed next to whatever the counters
report.

The tile convention comes from the kernel itself
(``hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_prefill.hip``): ``tile_m``
tiles ``out_features`` (the grid's x axis) and ``tile_n`` tiles ``rows`` (the
grid's y axis).  A weight element is therefore re-read once per row tile, i.e.
``ceil(rows / tile_n)`` times, not ``rows / tile_m``.

Diagnostic only; no perf claim retained from a single run.
"""

from __future__ import annotations

import argparse
import sys

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
    ap.add_argument("--in-features", type=int, default=4096)
    ap.add_argument("--out-features", type=int, default=2816)
    ap.add_argument("--tile-m", type=int, default=32)
    ap.add_argument("--tile-n", type=int, default=64)
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    runtime = get_hip_runtime()
    stream = 0

    qweight = make_q8_0_weight(args.out_features, args.in_features)
    x = np.random.default_rng(3).standard_normal(
        (args.rows, args.in_features)).astype(np.float32).astype(np.float16)
    x = (x.astype(np.float32) * 0.05).astype(np.float16)

    w_dev = malloc(qweight.nbytes)
    x_dev = malloc(x.nbytes)
    out_dev = malloc(args.rows * args.out_features * 2)
    copy_host_to_device(w_dev, host_array_ptr(qweight), qweight.nbytes)
    copy_host_to_device(x_dev, host_array_ptr(x), x.nbytes)

    row_tiles = (args.rows + args.tile_n - 1) // args.tile_n
    out_tiles = (args.out_features + args.tile_m - 1) // args.tile_m
    w_mb = qweight.nbytes / 1e6
    act_bytes = args.rows * args.in_features * 2
    out_bytes = args.rows * args.out_features * 2

    print(f"shape rows={args.rows} in={args.in_features} out={args.out_features} "
          f"tile=({args.tile_m}, {args.tile_n})")
    print(f"grid=({out_tiles}, {row_tiles}) blocks={out_tiles * row_tiles}")
    print(f"weight_bytes={qweight.nbytes} ({w_mb:.3f} MB) "
          f"act_bytes={act_bytes} out_bytes={out_bytes}")
    print(f"predicted_weight_traffic: rows/tile_n = {row_tiles}x -> "
          f"{row_tiles * qweight.nbytes / 1e6:.3f} MB")
    print(f"predicted_weight_traffic: rows/tile_m = "
          f"{args.rows / args.tile_m:.0f}x -> "
          f"{args.rows / args.tile_m * qweight.nbytes / 1e6:.3f} MB")

    for _ in range(args.warmup):
        prefill(x_dev.ptr, w_dev.ptr, out_dev.ptr, args.rows, args.in_features,
                args.out_features, tile_m=args.tile_m, tile_n=args.tile_n,
                stream=stream)
    runtime.stream_synchronize(stream)

    start_ev = runtime.event_create()
    stop_ev = runtime.event_create()
    times = []
    for _ in range(args.launches):
        runtime.event_record(start_ev, stream)
        prefill(x_dev.ptr, w_dev.ptr, out_dev.ptr, args.rows, args.in_features,
                args.out_features, tile_m=args.tile_m, tile_n=args.tile_n,
                stream=stream)
        runtime.event_record(stop_ev, stream)
        runtime.event_synchronize(stop_ev)
        times.append(runtime.event_elapsed_time_ms(start_ev, stop_ev))
    runtime.event_destroy(start_ev)
    runtime.event_destroy(stop_ev)

    t = float(np.mean(times))
    flops = 2.0 * args.rows * args.in_features * args.out_features
    print(f"launches={args.launches} warmup={args.warmup} "
          f"mean_ms={t:.4f} best_ms={min(times):.4f} "
          f"TFLOP/s={flops / 1e9 / t:.2f}")
    for mult in (row_tiles, args.rows / args.tile_m, 1.0):
        print(f"implied_rate_at_{mult:g}x_weight = "
              f"{mult * qweight.nbytes / 1e9 / (t / 1e3):.1f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
