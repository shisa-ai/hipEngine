#!/usr/bin/env python3
"""Measure the HIP side of the v2 narrow dispatch/grid-floor contract.

Serial latency replays a pre-recorded graph on one ordered stream. Independent
throughput distributes disjoint outputs across nonblocking streams. Native HIP
events and host clocks cover matching single and burst controls, and both exact
output shapes are validated. The optional wide-argument probe remains a separate
HIP-only direct-versus-graph diagnostic.

Usage:

  HIPENGINE_HIP_ARCH=gfx1100 HIP_VISIBLE_DEVICES=0 \
    python3 scripts/graph_node_microbench.py \
      --counts 1,50,200,941 --timing-mode serial_latency \
      --reps 50 --warmup 10 \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt \
      --json /tmp/graph_node_microbench.json
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, get_hip_runtime

_SOURCE = Path(__file__).with_name("microbench") / "graph_node_microbench.hip"
_HIP_TIMING_HEADER = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "micro"
    / "runners"
    / "micro_timing_hip.hpp"
)
_LAUNCH_SYMBOL = "gmb_launch_n"
_LAUNCHERS = {
    "tiny": "gmb_launch_n",        # 2-arg trivial kernel (the M16.1 default)
    "wide": "gmb_launch_wide_n",   # 16-arg kernel: realistic hot-kernel marshaling cost
}
_MEASURE_GRAPH_SYMBOL = "gmb_measure_graph_contract"
_MEASURE_INDEPENDENT_SYMBOL = "gmb_measure_independent_contract"

# Steady-state replay should accumulate roughly this many kernel nodes per timed
# batch so the single sync + the handful of graph_launch CPU calls are amortized
# to negligible against the GPU-side dispatch work being measured.
_TARGET_NODES_PER_BATCH = 5000


def _bind_launcher(library: ctypes.CDLL, symbol: str = _LAUNCH_SYMBOL):
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_int,     # n
        ctypes.c_int,     # count
        ctypes.c_int,     # grid_blocks (<=0 = auto)
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    return fn


def _bind_graph_measurement(library: ctypes.CDLL):
    fn = getattr(library, _MEASURE_GRAPH_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    fn.restype = ctypes.c_int
    return fn


def _bind_independent_measurement(library: ctypes.CDLL):
    fn = getattr(library, _MEASURE_INDEPENDENT_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    fn.restype = ctypes.c_int
    return fn


def _launch_n(fn, out_ptr: int, n: int, count: int, stream: int, runtime, *, grid_blocks: int = 0) -> None:
    err = fn(
        ctypes.c_void_p(out_ptr),
        ctypes.c_int(n),
        ctypes.c_int(count),
        ctypes.c_int(grid_blocks),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _median_us(samples: list[float]) -> float:
    return statistics.median(samples) * 1e6


def _measure_graph_contract(
    fn,
    out_ptr: int,
    n: int,
    count: int,
    runtime,
    *,
    samples: int,
    warmup: int,
    grid_blocks: int = 0,
) -> dict:
    sample_array = ctypes.c_double * samples
    single_gpu = sample_array()
    single_host = sample_array()
    burst_gpu = sample_array()
    burst_host = sample_array()
    single_mismatches = ctypes.c_uint()
    burst_mismatches = ctypes.c_uint()
    err = fn(
        ctypes.c_void_p(out_ptr),
        ctypes.c_int(n),
        ctypes.c_int(count),
        ctypes.c_int(grid_blocks),
        ctypes.c_int(samples),
        ctypes.c_int(warmup),
        single_gpu,
        single_host,
        burst_gpu,
        burst_host,
        ctypes.byref(single_mismatches),
        ctypes.byref(burst_mismatches),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))
    return {
        "node_count": count,
        "grid_blocks": grid_blocks if grid_blocks > 0 else max(1, math.ceil(n / 256)),
        "timing_mode": "serial_latency",
        "submission_strategy": "hip_graph",
        "stream_count": 1,
        "single_gpu_samples_us": list(single_gpu),
        "single_host_samples_us": list(single_host),
        "burst_gpu_samples_us": list(burst_gpu),
        "burst_host_samples_us": list(burst_host),
        "single_correctness_pass": single_mismatches.value == 0,
        "burst_correctness_pass": burst_mismatches.value == 0,
        "correctness_mismatches": single_mismatches.value + burst_mismatches.value,
    }


def _measure_independent_contract(
    fn,
    out_ptr: int,
    n: int,
    count: int,
    independent_streams: int,
    runtime,
    *,
    samples: int,
    warmup: int,
    grid_blocks: int = 0,
) -> dict:
    stream_count = min(count, independent_streams)
    sample_array = ctypes.c_double * samples
    single_gpu = sample_array()
    single_host = sample_array()
    burst_gpu = sample_array()
    burst_host = sample_array()
    single_mismatches = ctypes.c_uint()
    burst_mismatches = ctypes.c_uint()
    err = fn(
        ctypes.c_void_p(out_ptr),
        ctypes.c_int(n),
        ctypes.c_int(count),
        ctypes.c_int(grid_blocks),
        ctypes.c_int(stream_count),
        ctypes.c_int(samples),
        ctypes.c_int(warmup),
        single_gpu,
        single_host,
        burst_gpu,
        burst_host,
        ctypes.byref(single_mismatches),
        ctypes.byref(burst_mismatches),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))
    return {
        "node_count": count,
        "grid_blocks": grid_blocks if grid_blocks > 0 else max(1, math.ceil(n / 256)),
        "timing_mode": "independent_throughput",
        "submission_strategy": "multi_stream",
        "stream_count": stream_count,
        "single_gpu_samples_us": list(single_gpu),
        "single_host_samples_us": list(single_host),
        "burst_gpu_samples_us": list(burst_gpu),
        "burst_host_samples_us": list(burst_host),
        "single_correctness_pass": single_mismatches.value == 0,
        "burst_correctness_pass": burst_mismatches.value == 0,
        "correctness_mismatches": single_mismatches.value + burst_mismatches.value,
    }


def _measure_direct(fn, out_ptr, n, count, stream, runtime, *, reps, warmup, grid_blocks=0) -> dict:
    # Warmup
    for _ in range(warmup):
        _launch_n(fn, out_ptr, n, count, stream, runtime, grid_blocks=grid_blocks)
        runtime.stream_synchronize(stream)
    per_burst: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _launch_n(fn, out_ptr, n, count, stream, runtime, grid_blocks=grid_blocks)
        runtime.stream_synchronize(stream)
        per_burst.append(time.perf_counter() - t0)
    burst_us = _median_us(per_burst)
    return {
        "burst_us_median": burst_us,
        "us_per_node": burst_us / count,
        "burst_us_min": min(per_burst) * 1e6,
        "reps": reps,
    }


def _measure_graph(fn, out_ptr, n, count, stream, runtime, *, reps, warmup, grid_blocks=0) -> dict:
    # Capture the same C burst into ONE graph (fixed args, fixed stream).
    runtime.stream_begin_capture(stream)
    try:
        _launch_n(fn, out_ptr, n, count, stream, runtime, grid_blocks=grid_blocks)
    except Exception:
        # Abort the capture cleanly before re-raising.
        try:
            runtime.stream_end_capture(stream)
        except Exception:
            pass
        raise
    graph = runtime.stream_end_capture(stream)
    graph_exec = runtime.graph_instantiate(graph)
    try:
        # Warmup replays
        for _ in range(warmup):
            runtime.graph_launch(graph_exec, stream)
            runtime.stream_synchronize(stream)

        # (a) Steady-state throughput: M replays accumulated per timed batch.
        inner = max(1, math.ceil(_TARGET_NODES_PER_BATCH / count))
        per_batch: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                runtime.graph_launch(graph_exec, stream)
            runtime.stream_synchronize(stream)
            per_batch.append(time.perf_counter() - t0)
        batch_us = _median_us(per_batch)
        steady_us_per_node = batch_us / (inner * count)

        # (b) Per-replay latency: one replay + one sync (mirrors one verify cycle).
        per_replay: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            runtime.graph_launch(graph_exec, stream)
            runtime.stream_synchronize(stream)
            per_replay.append(time.perf_counter() - t0)
        replay_us = _median_us(per_replay)
    finally:
        runtime.graph_exec_destroy(graph_exec)
        runtime.graph_destroy(graph)

    return {
        "inner_replays_per_batch": inner,
        "batch_us_median": batch_us,
        "steady_us_per_node": steady_us_per_node,
        "replay_latency_us_median": replay_us,
        "replay_latency_us_per_node": replay_us / count,
        "reps": reps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", default="1,50,200,941", help="Comma-separated node counts per burst/graph")
    parser.add_argument(
        "--kernels",
        default="tiny",
        help="Comma-separated profiles: matched tiny and optional HIP-only wide",
    )
    parser.add_argument("--n", type=int, default=256, help="Trivial-kernel element count (one block by default)")
    parser.add_argument(
        "--grid-sweep",
        default="",
        help="Optional narrow-kernel grid block counts at a fixed dispatch count",
    )
    parser.add_argument("--grid-sweep-count", type=int, default=941, help="Node count (launches per burst) used for the --grid-sweep measurement")
    parser.add_argument("--reps", type=int, default=50, help="Timed repetitions (median reported)")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup iterations")
    parser.add_argument(
        "--timing-mode",
        choices=("serial_latency", "independent_throughput"),
        default="serial_latency",
    )
    parser.add_argument("--independent-streams", type=int, default=4)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--hardware-gpu", default=None, help="Human-readable GPU name for the artifact")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    counts = [int(x) for x in args.counts.split(",") if x.strip()]
    if not counts or any(c <= 0 for c in counts):
        raise ValueError("--counts must be positive integers")
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.reps <= 0 or args.independent_streams <= 0:
        raise ValueError("--reps and --independent-streams must be positive")
    kernel_profiles = [k.strip() for k in args.kernels.split(",") if k.strip()]
    if not kernel_profiles or any(k not in _LAUNCHERS for k in kernel_profiles):
        raise ValueError(f"--kernels must be a comma list from {sorted(_LAUNCHERS)}")
    grid_sweep = [int(x) for x in args.grid_sweep.split(",") if x.strip()]
    if any(g <= 0 for g in grid_sweep):
        raise ValueError("--grid-sweep values must be positive integers")

    compiler_version = (
        args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    )

    library = build_hip(
        sources=[_SOURCE],
        family="graph_node_microbench",
        profile="baseline",
        compiler_version=compiler_version,
        extra_flags=[
            "-DHIPENGINE_MICRO_TIMING_HEADER_HASH=0x"
            + hashlib.sha256(_HIP_TIMING_HEADER.read_bytes()).hexdigest()[:16]
            + "ULL"
        ],
        output_name="graph_node_microbench.so",
        load=True,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    max_count = max([*counts, args.grid_sweep_count if grid_sweep else 1, args.warmup, 1])
    nbytes = args.n * max_count * ctypes.sizeof(ctypes.c_float)
    out_ptr = runtime.malloc(nbytes)
    runtime.memset(out_ptr, 0, nbytes)
    streams = [runtime.stream_create(nonblocking=True)]

    narrow_rows: list[dict] = []
    wide_diagnostics: list[dict] = []
    grid_rows: list = []
    try:
        graph_measurement = _bind_graph_measurement(library)
        independent_measurement = _bind_independent_measurement(library)
        for count in counts:
            if args.timing_mode == "serial_latency":
                row = _measure_graph_contract(
                    graph_measurement,
                    out_ptr,
                    args.n,
                    count,
                    runtime,
                    samples=args.reps,
                    warmup=args.warmup,
                )
            else:
                row = _measure_independent_contract(
                    independent_measurement,
                    out_ptr,
                    args.n,
                    count,
                    args.independent_streams,
                    runtime,
                    samples=args.reps,
                    warmup=args.warmup,
                )
            narrow_rows.append(row)
            burst_median = statistics.median(row["burst_gpu_samples_us"])
            print(
                f"[tiny] N={count:>4} mode={args.timing_mode} "
                f"gpu={burst_median / count:6.2f} us/node"
            )

        if "wide" in kernel_profiles:
            wide_fn = _bind_launcher(library, _LAUNCHERS["wide"])
            for count in counts:
                direct = _measure_direct(
                    wide_fn,
                    out_ptr,
                    args.n,
                    count,
                    streams[0],
                    runtime,
                    reps=args.reps,
                    warmup=args.warmup,
                )
                graph = _measure_graph(
                    wide_fn,
                    out_ptr,
                    args.n,
                    count,
                    streams[0],
                    runtime,
                    reps=args.reps,
                    warmup=args.warmup,
                )
                wide_diagnostics.append(
                    {
                        "node_count": count,
                        "direct": direct,
                        "graph": graph,
                        "graph_speedup_vs_direct": (
                            direct["us_per_node"] / graph["steady_us_per_node"]
                            if graph["steady_us_per_node"] > 0
                            else None
                        ),
                    }
                )
                print(
                    f"[wide] N={count:>4} direct={direct['us_per_node']:6.2f} us/node "
                    f"graph_steady={graph['steady_us_per_node']:6.2f} us/node  "
                    f"speedup={wide_diagnostics[-1]['graph_speedup_vs_direct'] or float('nan'):.2f}x"
                )

        if grid_sweep:
            sweep_count = args.grid_sweep_count
            for g in grid_sweep:
                if args.timing_mode == "serial_latency":
                    row = _measure_graph_contract(
                        graph_measurement,
                        out_ptr,
                        args.n,
                        sweep_count,
                        runtime,
                        samples=args.reps,
                        warmup=args.warmup,
                        grid_blocks=g,
                    )
                else:
                    row = _measure_independent_contract(
                        independent_measurement,
                        out_ptr,
                        args.n,
                        sweep_count,
                        args.independent_streams,
                        runtime,
                        samples=args.reps,
                        warmup=args.warmup,
                        grid_blocks=g,
                    )
                grid_rows.append(row)
                gpu_per_node = statistics.median(row["burst_gpu_samples_us"]) / sweep_count
                print(
                    f"[grid] blocks={g:>6} mode={args.timing_mode} "
                    f"gpu={gpu_per_node:6.2f} us/launch"
                )
    finally:
        for stream in streams:
            runtime.stream_destroy(stream)
        runtime.free(out_ptr)

    artifact = {
        "run_tag": "hip-dispatch-floor-v2-raw",
        "summary": (
            "Narrow dispatch-floor benchmark with explicit serialized-latency or "
            "independent-throughput semantics, GPU-event and host-wall controls, "
            "and exact validation of the measured burst shape."
        ),
        "status": "diagnostic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": args.hardware_gpu,
            "arch": os.environ.get("HIPENGINE_HIP_ARCH", "unknown"),
        },
        "software": {"python": platform.python_version(), "platform": platform.platform(), "hipcc_version": compiler_version},
        "config": {
            "counts": counts,
            "n_elements": args.n,
            "local_size_x": 256,
            "reps": args.reps,
            "warmup": args.warmup,
            "kernels": kernel_profiles,
            "timing_mode": args.timing_mode,
            "independent_streams": args.independent_streams,
            "grid_sweep": grid_sweep,
            "grid_sweep_count": args.grid_sweep_count,
            "target_nodes_per_batch": _TARGET_NODES_PER_BATCH,
            "method": (
                "serial_latency uses a pre-recorded HIP graph on one ordered stream; "
                "independent_throughput partitions disjoint outputs over nonblocking HIP streams. "
                "Both modes record HIP-event and submit-to-completion host-wall single/burst samples."
            ),
        },
        "rows": narrow_rows,
        "rows_by_kernel": {"tiny": narrow_rows},
        "grid_sweep": grid_rows or None,
        "hip_only_wide_diagnostics": wide_diagnostics or None,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
