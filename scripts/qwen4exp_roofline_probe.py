#!/usr/bin/env python3
"""Measure the practical roofline of this GPU and compare it to the spec sheet.

The dense GGUF projection analysis concludes that the kernel design is capped at
22.1% of FP32 peak by its instruction mix, and reaches 8.9%. Both figures use the
datasheet peak (40 CU x 128 lanes x 2 FLOP x 2.9 GHz = 29696 GFLOP/s) as the
denominator. If this part cannot reach that rate in practice — an APU shares a
power budget with its CPU — then the honest denominator is lower and the kernel
is closer to its ceiling than the analysis claims.

This script measures the three bounds the analysis depends on:

* the FP32 FMA rate a register-resident kernel actually sustains,
* the rate a kernel with the projection's 32-FMA/105-integer mix sustains, which
  tests the issue-mix ceiling directly rather than inferring it, and
* achievable stream bandwidth.

Diagnostic only. Nothing here is a production kernel.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import numpy as np
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.build import BuildArtifact, build_hip  # noqa: E402
from hipengine.core.hip import HIP_SUCCESS, get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)

SOURCE = REPO_ROOT / "hipengine/kernels/hip_gfx1100/smoke/roofline_probe.hip"

FMA_SYMBOL = "hipengine_roofline_fma_peak"
MIXED_SYMBOL = "hipengine_roofline_mixed_fma40_int105"
STREAM_SYMBOL = "hipengine_roofline_stream"

FMA_PER_ITERATION = 16
MIXED_FMA_PER_ITERATION = 40


def build() -> ctypes.CDLL:
    artifact = build_hip(
        sources=[SOURCE],
        family="smoke",
        profile="baseline",
        output_name="roofline_probe.so",
        load=True,
    )
    if isinstance(artifact, BuildArtifact):
        raise SystemExit("build_hip returned a plan, not a loaded library")
    return artifact


def _bind(library, symbol: str, n_args: int):
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p] * (n_args - 1) + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def timed(runtime, launch, repetitions: int) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(repetitions):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return runtime.event_elapsed_time_ms(start, stop) / repetitions
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--stream-mib", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--compute-units", type=int, default=40)
    parser.add_argument("--clock-ghz", type=float, default=2.9)
    parser.add_argument("--dram-bytes-per-second", type=float, default=256e9)
    args = parser.parse_args()

    runtime = get_hip_runtime()
    library = build()

    threads_total = args.blocks * args.threads
    out = malloc(threads_total * 4)
    side = malloc(threads_total * 4)

    fma_fn = _bind(library, FMA_SYMBOL, 5)
    mixed_fn = _bind(library, MIXED_SYMBOL, 6)
    stream_fn = _bind(library, STREAM_SYMBOL, 6)

    results: dict[str, object] = {}

    # --- FP32 FMA peak -----------------------------------------------------
    fma_flops = threads_total * args.iters * FMA_PER_ITERATION * 2
    fma_fn(out.ptr, args.iters, args.blocks, args.threads, 0)
    runtime.stream_synchronize(0)
    fma_ms = timed(
        runtime,
        lambda: fma_fn(out.ptr, args.iters, args.blocks, args.threads, 0),
        args.repetitions,
    )
    fma_gflops = fma_flops / (fma_ms / 1000.0) / 1e9
    results["fma_peak"] = {
        "ms": round(fma_ms, 3),
        "gflops": round(fma_gflops, 1),
        "flops": fma_flops,
    }

    # --- the projection's issue mix ---------------------------------------
    mixed_flops = threads_total * args.iters * MIXED_FMA_PER_ITERATION * 2
    mixed_fn(out.ptr, side.ptr, args.iters, args.blocks, args.threads, 0)
    runtime.stream_synchronize(0)
    mixed_ms = timed(
        runtime,
        lambda: mixed_fn(out.ptr, side.ptr, args.iters, args.blocks, args.threads, 0),
        args.repetitions,
    )
    mixed_gflops = mixed_flops / (mixed_ms / 1000.0) / 1e9
    results["mixed_fma40_int105"] = {
        "ms": round(mixed_ms, 3),
        "gflops": round(mixed_gflops, 1),
        "flops": mixed_flops,
        "fma_per_iteration": MIXED_FMA_PER_ITERATION,
        "int_per_iteration": 105,
    }

    # --- stream bandwidth --------------------------------------------------
    n_float4 = args.stream_mib * 1024 * 1024 // 16
    src = malloc(n_float4 * 16)
    dst = malloc(n_float4 * 16)
    try:
        # host_array_ptr needs a ctypes.data pointer, so the staging buffer is
        # a NumPy array rather than a bytearray.
        host = np.zeros(n_float4 * 16, dtype=np.uint8)
        copy_host_to_device(src, host_array_ptr(host))
        stream_fn(dst.ptr, src.ptr, n_float4, args.blocks, args.threads, 0)
        runtime.stream_synchronize(0)
        stream_ms = timed(
            runtime,
            lambda: stream_fn(dst.ptr, src.ptr, n_float4, args.blocks, args.threads, 0),
            args.repetitions,
        )
        # A copy reads and writes, so the moved bytes are twice the buffer.
        moved = n_float4 * 16 * 2
        results["stream_copy"] = {
            "ms": round(stream_ms, 3),
            "bytes_moved": moved,
            "gb_per_second": round(moved / (stream_ms / 1000.0) / 1e9, 1),
            "buffer_mib": args.stream_mib,
        }
    finally:
        free(src)
        free(dst)

    theoretical_fp32 = args.compute_units * 128 * 2 * args.clock_ghz
    payload = {
        "schema": 1,
        "kind": "roofline_probe",
        "performance_claim": False,
        "status": "diagnostic",
        "hardware": {
            "compute_units": args.compute_units,
            "clock_ghz": args.clock_ghz,
            "theoretical_fp32_gflops": theoretical_fp32,
            "theoretical_dram_bytes_per_second": args.dram_bytes_per_second,
        },
        "config": {
            "blocks": args.blocks,
            "threads": args.threads,
            "iterations": args.iters,
            "repetitions": args.repetitions,
        },
        "measured": results,
        "derived": {
            "fp32_practical_share_of_theoretical_pct": round(
                100.0 * results["fma_peak"]["gflops"] / theoretical_fp32, 1
            ),
            "mixed_share_of_measured_fma_peak_pct": round(
                100.0
                * results["mixed_fma40_int105"]["gflops"]
                / results["fma_peak"]["gflops"],
                1,
            ),
            "mixed_share_of_theoretical_fp32_pct": round(
                100.0 * results["mixed_fma40_int105"]["gflops"] / theoretical_fp32, 1
            ),
            "stream_share_of_theoretical_dram_pct": round(
                100.0
                * results["stream_copy"]["gb_per_second"]
                * 1e9
                / args.dram_bytes_per_second,
                1,
            ),
        },
        "notes": [
            "Diagnostic only; no production path is exercised.",
            "The mixed kernel holds the dense GGUF projection's measured 40-FMA to 105-integer issue ratio, so its rate is the issue-mix ceiling measured rather than inferred from disassembly.",
            "Stream bandwidth counts read plus write bytes.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"FP32 FMA peak        : {fma_gflops:9.0f} GFLOP/s  "
          f"({100.0 * fma_gflops / theoretical_fp32:.1f}% of "
          f"{theoretical_fp32:.0f} theoretical)")
    print(f"40 FMA + 105 integer : {mixed_gflops:9.0f} GFLOP/s  "
          f"({100.0 * mixed_gflops / fma_gflops:.1f}% of measured FMA peak)")
    print(f"stream copy          : {results['stream_copy']['gb_per_second']:9.1f} GB/s    "
          f"({100.0 * results['stream_copy']['gb_per_second'] * 1e9 / args.dram_bytes_per_second:.1f}% "
          f"of {args.dram_bytes_per_second / 1e9:.0f} GB/s theoretical)")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
