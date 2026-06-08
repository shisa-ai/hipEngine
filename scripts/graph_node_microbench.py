#!/usr/bin/env python3
"""M16.1 diagnostic: isolate the ROCm HIP graph-replay per-node cost on W7900/gfx1100.

This is the cheapest diagnostic that decides the whole C_B strategy. It measures,
in isolation (trivial kernels, fixed args, no validation, no bucket churn, single
fixed stream), two numbers:

  * direct_us_per_node    -- N trivial kernels issued back-to-back from a C loop
                             (zero Python/ctypes per-node overhead), one sync.
                             This is the pure HIP host-issue + GPU-dispatch cost
                             that produces the verify-window launch residual.
  * graph_us_per_node     -- the SAME C burst captured into ONE HIP graph and
                             replayed in steady state. Pure GPU graph-scheduler
                             per-node cost; ~one graph_launch CPU call per replay.

Decision (see docs/MTP.md "M16 -- the C_B <= 2 program"):
  * If graph replay is ~2-6 us/node (CUDA-class), the ~19.4 ms launch residual
    collapses toward ~4 ms WITHOUT touching kernels -> program is
    "native loop + one clean graph" (M16.2 + M16.5).
  * If graph replay is genuinely ~15-20 us/node, graphs do not collapse the
    dispatch floor -> program is "fewer larger kernels" (M16.3 + M16.4).

The earlier "graph neutral" verdict (M12.1 / M13.D) was measured on the full
verify path with per-cycle bucket churn + validation, NOT a clean steady-state
replay, so it is re-tested here in isolation.

Usage (prebuild the .so outside any profiler; this script will build on first run):

  HIPENGINE_HIP_ARCH=gfx1100 HIP_VISIBLE_DEVICES=0 \
    python3 scripts/graph_node_microbench.py \
      --counts 1,50,200,941 --reps 50 --warmup 10 \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt \
      --json /tmp/graph_node_microbench.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, get_hip_runtime

_SOURCE = Path(__file__).with_name("microbench") / "graph_node_microbench.hip"
_LAUNCH_SYMBOL = "gmb_launch_n"

# Steady-state replay should accumulate roughly this many kernel nodes per timed
# batch so the single sync + the handful of graph_launch CPU calls are amortized
# to negligible against the GPU-side dispatch work being measured.
_TARGET_NODES_PER_BATCH = 5000


def _bind_launcher(library: ctypes.CDLL):
    fn = getattr(library, _LAUNCH_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_int,     # n
        ctypes.c_int,     # count
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    return fn


def _launch_n(fn, out_ptr: int, n: int, count: int, stream: int, runtime) -> None:
    err = fn(ctypes.c_void_p(out_ptr), ctypes.c_int(n), ctypes.c_int(count), ctypes.c_void_p(stream))
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _median_us(samples: list[float]) -> float:
    return statistics.median(samples) * 1e6


def _measure_direct(fn, out_ptr, n, count, stream, runtime, *, reps, warmup) -> dict:
    # Warmup
    for _ in range(warmup):
        _launch_n(fn, out_ptr, n, count, stream, runtime)
        runtime.stream_synchronize(stream)
    per_burst: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _launch_n(fn, out_ptr, n, count, stream, runtime)
        runtime.stream_synchronize(stream)
        per_burst.append(time.perf_counter() - t0)
    burst_us = _median_us(per_burst)
    return {
        "burst_us_median": burst_us,
        "us_per_node": burst_us / count,
        "burst_us_min": min(per_burst) * 1e6,
        "reps": reps,
    }


def _measure_graph(fn, out_ptr, n, count, stream, runtime, *, reps, warmup) -> dict:
    # Capture the same C burst into ONE graph (fixed args, fixed stream).
    runtime.stream_begin_capture(stream)
    try:
        _launch_n(fn, out_ptr, n, count, stream, runtime)
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


def _verdict(
    steady_us_per_node_941: float,
    direct_us_per_node_941: float,
    *,
    observed_residual_us_per_op: float,
) -> dict:
    # The measured HIP floor is the *minimum* per-node dispatch cost (host issue +
    # GPU dispatch) with zero Python per-op overhead. The graph-vs-direct ratio
    # says whether a HIP graph beats a tight native launch loop at all.
    hip_floor = min(steady_us_per_node_941, direct_us_per_node_941)
    graph_beats_direct = (
        direct_us_per_node_941 / steady_us_per_node_941 if steady_us_per_node_941 > 0 else float("nan")
    )
    # Decompose the observed full verify-path residual (~20.6 us/op = 19.4 ms / 941)
    # into the unavoidable HIP floor vs Python/ctypes per-op orchestration.
    orchestration_us_per_op = max(0.0, observed_residual_us_per_op - hip_floor)
    orchestration_share = (
        orchestration_us_per_op / observed_residual_us_per_op if observed_residual_us_per_op > 0 else 0.0
    )

    graph_neutral = graph_beats_direct < 1.15
    floor_is_low = hip_floor <= 6.0

    if graph_neutral and floor_is_low:
        program = "native-loop FIRST (M16.2); fewer-larger-kernels for the floor (M16.3+M16.4); graphs NEUTRAL (de-prioritize M16.5)"
        rationale = (
            f"HIP floor is {hip_floor:.2f} us/node and a HIP graph does NOT beat a native launch loop "
            f"({graph_beats_direct:.2f}x). So the ~{observed_residual_us_per_op:.1f} us/op full-path residual is "
            f"~{orchestration_share*100:.0f}% host-side per-op overhead ({orchestration_us_per_op:.1f} us/op: Python/ctypes "
            "marshaling + per-op pointer/bucket setup above the trivial-kernel floor), not HIP dispatch. "
            f"A micro-bench-tight native C++ verify loop (M16.2: fixed args, baked launch sequence, persistent scratch) "
            f"collapses the residual toward the {hip_floor*941/1000:.2f} ms HIP floor at 941 nodes; going below that needs "
            "FEWER nodes (M16.3) + lower kernel time (M16.4). A HIP graph buys ~nothing over the native loop (M16.5 de-prioritized). "
            "Caveat: M14.dispatch.1's partial MoE dispatcher was parity, so the native loop must match this burst's tightness to realize the floor."
        )
    elif (not graph_neutral) and floor_is_low:
        program = "native-loop-plus-one-clean-graph (M16.2 + M16.5)"
        rationale = (
            f"HIP floor is {hip_floor:.2f} us/node and a HIP graph beats a native launch loop "
            f"({graph_beats_direct:.2f}x), so one clean graph collapses the host-issue residual; M16.5 is load-bearing."
        )
    else:
        program = "fewer-larger-kernels (M16.3 + M16.4)"
        rationale = (
            f"HIP floor is {hip_floor:.2f} us/node (~hardware dispatch floor) and graphs do not help "
            f"({graph_beats_direct:.2f}x); reduce node COUNT (structural megakernels) + weight amortization."
        )
    return {
        "steady_us_per_node_941": steady_us_per_node_941,
        "direct_us_per_node_941": direct_us_per_node_941,
        "hip_floor_us_per_node": hip_floor,
        "graph_speedup_vs_direct_941": graph_beats_direct,
        "observed_full_path_residual_us_per_op": observed_residual_us_per_op,
        "implied_orchestration_us_per_op": orchestration_us_per_op,
        "implied_orchestration_share_of_residual": orchestration_share,
        "implied_residual_ms_direct_941": direct_us_per_node_941 * 941 / 1000.0,
        "implied_residual_ms_graph_941": steady_us_per_node_941 * 941 / 1000.0,
        "implied_hip_floor_ms_941": hip_floor * 941 / 1000.0,
        "program": program,
        "rationale": rationale,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", default="1,50,200,941", help="Comma-separated node counts per burst/graph")
    parser.add_argument("--n", type=int, default=256, help="Trivial-kernel element count (one block by default)")
    parser.add_argument("--reps", type=int, default=50, help="Timed repetitions (median reported)")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup iterations")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--hardware-gpu", default=None, help="Human-readable GPU name for the artifact")
    parser.add_argument(
        "--observed-residual-us-per-op",
        type=float,
        default=20.6,
        help="Observed full verify-path launch residual per op (default 20.6 = 19.4 ms / 941 launches, B=3 from docs/MTP.md M16) used to decompose HIP floor vs Python orchestration",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    counts = [int(x) for x in args.counts.split(",") if x.strip()]
    if not counts or any(c <= 0 for c in counts):
        raise ValueError("--counts must be positive integers")
    if args.n <= 0:
        raise ValueError("--n must be positive")

    compiler_version = (
        args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    )

    library = build_hip(
        sources=[_SOURCE],
        family="graph_node_microbench",
        profile="baseline",
        compiler_version=compiler_version,
        output_name="graph_node_microbench.so",
        load=True,
        require_cached=args.require_cached_build,
    )
    fn = _bind_launcher(library)
    runtime = get_hip_runtime()

    nbytes = args.n * 4
    out_ptr = runtime.malloc(nbytes)
    runtime.memset(out_ptr, 0, nbytes)
    stream = runtime.stream_create(nonblocking=True)

    rows = []
    try:
        for count in counts:
            direct = _measure_direct(fn, out_ptr, args.n, count, stream, runtime, reps=args.reps, warmup=args.warmup)
            graph = _measure_graph(fn, out_ptr, args.n, count, stream, runtime, reps=args.reps, warmup=args.warmup)
            row = {
                "node_count": count,
                "direct": direct,
                "graph": graph,
                "graph_speedup_vs_direct": (
                    direct["us_per_node"] / graph["steady_us_per_node"]
                    if graph["steady_us_per_node"] > 0
                    else None
                ),
            }
            rows.append(row)
            print(
                f"N={count:>4}  direct={direct['us_per_node']:6.2f} us/node  "
                f"graph_steady={graph['steady_us_per_node']:6.2f} us/node  "
                f"graph_replay_latency={graph['replay_latency_us_median']:8.2f} us "
                f"({graph['replay_latency_us_per_node']:6.2f}/node)  "
                f"speedup={row['graph_speedup_vs_direct'] or float('nan'):.2f}x"
            )
    finally:
        runtime.stream_destroy(stream)
        runtime.free(out_ptr)

    by_count = {r["node_count"]: r for r in rows}
    ref = by_count.get(941) or rows[-1]
    verdict = _verdict(
        ref["graph"]["steady_us_per_node"],
        ref["direct"]["us_per_node"],
        observed_residual_us_per_op=args.observed_residual_us_per_op,
    )
    verdict["reference_node_count"] = ref["node_count"]

    print()
    print(f"VERDICT: {verdict['program']}")
    print(f"  {verdict['rationale']}")
    print(
        f"  per-node @{ref['node_count']}: HIP floor {verdict['hip_floor_us_per_node']:.2f} us  "
        f"(graph {verdict['steady_us_per_node_941']:.2f} vs direct {verdict['direct_us_per_node_941']:.2f}, "
        f"{verdict['graph_speedup_vs_direct_941']:.2f}x)"
    )
    print(
        f"  residual decomposition: full-path ~{verdict['observed_full_path_residual_us_per_op']:.1f} us/op = "
        f"{verdict['hip_floor_us_per_node']:.1f} HIP floor + {verdict['implied_orchestration_us_per_op']:.1f} Python orchestration "
        f"({verdict['implied_orchestration_share_of_residual']*100:.0f}% orchestration)"
    )

    artifact = {
        "run_tag": "m16.1-graph-node-replay-microbench",
        "summary": (
            "Isolated HIP graph-replay per-node cost vs direct hipLaunchKernelGGL on W7900/gfx1100. "
            "Trivial one-block kernels, fixed args, single fixed stream, steady-state replay; "
            "decides the C_B program (native-loop+graph vs fewer-larger-kernels)."
        ),
        "status": "diagnostic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {"gpu": args.hardware_gpu, "arch": "gfx1100"},
        "software": {"python": platform.python_version(), "platform": platform.platform(), "hipcc_version": compiler_version},
        "config": {
            "counts": counts,
            "n_elements": args.n,
            "reps": args.reps,
            "warmup": args.warmup,
            "target_nodes_per_batch": _TARGET_NODES_PER_BATCH,
            "method": (
                "N back-to-back hipLaunchKernelGGL issued from a single C call (no Python per-node "
                "overhead); same C burst captured into one graph and replayed; wall via perf_counter "
                "around stream sync; median over reps."
            ),
        },
        "rows": rows,
        "verdict": verdict,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
