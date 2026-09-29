#!/usr/bin/env python3
"""DE-3 candidate screen: the three launch-starved decode shapes.

The decode linear owner ``gguf_k_pack8_prefill_out_kernel<float, float, 8>``
runs 24.005 ms/step.  All eight of its shapes are now byte-exact, and the
measured rate depends on grid size rather than arithmetic intensity: every
shape at 320 or more workgroups reaches 90.8-102.9% of the 221 GB/s roof,
while every shape below 100 workgroups reaches only 42.4-58.8%.

Three shapes are starved, and they are exactly the ones with
``out_features <= 640``:

    out=320  in=10240   97 tensors/step   3.601 ms/step    93.8 GB/s  42.4%
    out=640  in=2560    96 tensors/step   1.286 ms/step   130.0 GB/s  58.8%
    out=512  in=2560    24 tensors/step   0.307 ms/step   108.9 GB/s  49.3%

The grid for this kernel is ``(out_features / 8, rows)`` with a 128-wide
block, so ``out=320`` launches only 40 workgroups.  Two levers are cheap
enough to screen before writing any kernel: the ``threads`` argument the
launcher already takes, and the variants already registered for this quant.

**Working-set discipline.**  A first version of this screen timed a single
3.48 MB weight across 35 re-reads and reported 326.9 GB/s -- above the
221 GB/s roof, therefore invalid, because the whole weight is L3-resident.
The real decode step streams 4477 MB and caches nothing.  This version
therefore allocates one weight per *real tensor in the step* (97 / 96 / 24)
and sweeps them all, so the working set is the same 337.7 / 167.1 / 33.4 MB
the engine actually touches.  The sweep is timed as a whole and divided by
the sweep's byte total, which is the quantity comparable to the trace.

out=512 is retained for completeness even though its 33.4 MB sweep still
fits in L3; its row is marked and its absolute rate must not be quoted.

Correctness is reported, not gated: whether a faster variant is admissible
is a question for the production numerical envelope, not for this script.

Usage:
    python3 scripts/qwen4exp_de3_starved_shape_screen.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hipengine.core.memory import (  # noqa: E402
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (  # noqa: E402
    gguf_q8_0_gemv_f32_f32_out,
    gguf_q8_0_pack8_gemv_f32_f32_out,
)

# (out_features, in_features, tensors/step, ms/step, MB/step measured in the trace)
# The first three are the starved shapes (grid < 100 workgroups); the rest are
# the saturated ones and are carried so the threads lever is tested on both.
STARVED = (
    (320, 10240, 97, 3.601, 337.7),
    (640, 2560, 96, 1.286, 167.1),
    (512, 2560, 24, 0.307, 33.4),
)

# Sweep of the gated range itself, so the rule's boundary is supported across
# every width it claims rather than only at the three shapes the engine
# happened to exercise.  out=48 and out=128 appear in the Qwen3.6-35B model
# (96 and 0 tensors respectively) and out=48 was not covered by the original
# three; the in_features values are the ones that model actually uses.
NARROW_RANGE = (
    (48, 512, 96, 0.0, 0.0),
    (128, 512, 96, 0.0, 0.0),
    (256, 512, 96, 0.0, 0.0),
    (320, 10240, 97, 3.601, 337.7),
    (384, 512, 96, 0.0, 0.0),
    (512, 2560, 24, 0.307, 33.4),
    (640, 2560, 96, 1.286, 167.1),
    (768, 512, 96, 0.0, 0.0),
)

# (out_features, in_features, tensors/step, ms/step, MB/step) for the shapes
# already at 90.8-102.9% of roof.  out=10240 and out=2560 mix two or three
# in_features; the dominant one is used, and the resulting byte figure is
# within 0.5% of the trace's.
SATURATED = (
    (10240, 2560, 134, 6.822, 1368.3),
    (2560, 6144, 97, 4.333, 892.7),
    (248320, 2560, 1, 2.969, 675.4),
    (6144, 2560, 36, 2.867, 601.6),
    (12288, 2560, 12, 1.820, 401.1),
)

ROOF_GBPS = 221.0
Q8_0_BPW = 34 / 32  # 1.0625 bytes per weight
L3_BYTES = 64 * 1024 * 1024


def make_q8_0_weight(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Synthetic Q8_0 payload: per 32-weight block, f16 scale then 32 int8."""
    rng = np.random.default_rng(seed)
    blocks = (rows * cols) // 32
    scales = (rng.random(blocks).astype(np.float32) * 0.02 + 0.005).astype(np.float16)
    quants = rng.integers(-127, 128, size=(blocks, 32), dtype=np.int8)
    out = np.zeros((blocks, 34), dtype=np.uint8)
    out[:, :2] = scales.view(np.uint8).reshape(blocks, 2)
    out[:, 2:] = quants.view(np.uint8)
    return out.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    ap.add_argument(
        "--shapes",
        choices=("starved", "saturated", "range", "all"),
        default="starved",
        help="which shapes to sweep; default is the three starved ones",
    )
    args = ap.parse_args()

    if args.shapes == "starved":
        shapes = STARVED
    elif args.shapes == "saturated":
        shapes = SATURATED
    elif args.shapes == "range":
        shapes = NARROW_RANGE
    else:
        shapes = STARVED + SATURATED

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        print("SKIP: libamdhip64.so not available")
        return 0

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()

    variants = (
        ("pack8_gemv_f32_f32_out", gguf_q8_0_pack8_gemv_f32_f32_out),
        ("gemv_f32_f32_out", gguf_q8_0_gemv_f32_f32_out),
    )

    results = []
    print(
        f"{'shape':>14} {'variant':>24} {'thr':>4} {'tensors':>7} {'us/sweep':>9} "
        f"{'GB/s':>7} {'%roof':>7} {'L3?':>4}"
    )
    for out_features, in_features, n_tensors, ms_measured, mb_measured in shapes:
        payload = make_q8_0_weight(out_features, in_features)
        w_bytes = payload.nbytes
        # One distinct device weight per real tensor in the step: the sweep then
        # streams exactly the bytes the engine streams, with no L3 reuse.
        w_devs = []
        for i in range(n_tensors):
            d = malloc(w_bytes, runtime=runtime)
            copy_host_to_device(d, host_array_ptr(payload), w_bytes, runtime=runtime)
            w_devs.append(d)
        x = np.ones(in_features, dtype=np.float32)
        x_dev = malloc(x.nbytes, runtime=runtime)
        copy_host_to_device(x_dev, host_array_ptr(x), x.nbytes, runtime=runtime)
        out_buf = np.zeros(out_features, dtype=np.float32)
        out_dev = malloc(out_buf.nbytes, runtime=runtime)
        sweep_bytes = w_bytes * n_tensors
        l3_resident = sweep_bytes <= L3_BYTES
        try:
            for name, fn in variants:
                for thr in args.threads:
                    def sweep():
                        for d in w_devs:
                            fn(
                                x_dev.ptr, d.ptr, out_dev.ptr, 1,
                                in_features, out_features, threads=thr,
                            )

                    for _ in range(args.warmup):
                        sweep()
                    runtime.device_synchronize()
                    best = float("inf")
                    for _ in range(args.reps):
                        runtime.device_synchronize()
                        t0 = time.perf_counter()
                        sweep()
                        runtime.device_synchronize()
                        best = min(best, time.perf_counter() - t0)
                    us = best * 1e6
                    gbps = sweep_bytes / (us * 1e-6) / 1e9
                    print(
                        f"{out_features:>6}x{in_features:<7} {name:>24} {thr:>4} "
                        f"{n_tensors:>7} {us:>9.1f} {gbps:>7.1f} "
                        f"{100 * gbps / ROOF_GBPS:>6.1f}% {'yes' if l3_resident else 'no':>4}"
                    )
                    results.append(
                        dict(
                            out_features=out_features,
                            in_features=in_features,
                            variant=name,
                            threads=thr,
                            tensors_per_sweep=n_tensors,
                            us_per_sweep=us,
                            gbps=gbps,
                            pct_roof=100 * gbps / ROOF_GBPS,
                            sweep_bytes=sweep_bytes,
                            l3_resident=l3_resident,
                            trace_ms_per_step=ms_measured,
                            trace_mb_per_step=mb_measured,
                        )
                    )
        finally:
            for d in w_devs:
                free(d, runtime=runtime)
            for d in (x_dev, out_dev):
                free(d, runtime=runtime)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
