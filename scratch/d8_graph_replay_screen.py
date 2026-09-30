#!/usr/bin/env python3
"""X7 screen: host cost of launching 784 ops vs replaying one captured graph.

The D8/X7 map says a decode step spends ~2.6 ms of host time in 784 kernel
launches (probe) / 13.1 ms across 6752 HIP-API calls (rocprof, inflated by its
own instrumentation). Graph replay replaces all of it with one hipGraphLaunch.
This screen bounds the HOST-side ceiling before anyone wires capture into the
Gemma engine: enqueue 784 stream ops the normal way, then capture the same
chain once and replay it, timing both per-step host costs.

It uses hipMemsetAsync as the op (capture-compatible, no device dependency on
content) because the question here is host submission cost, not device work.

Run: .venv/bin/python scratch/d8_graph_replay_screen.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime  # noqa: E402

N_OPS = 784          # launches per decode step (probe)
N_REPLAYS = 50       # timed graph replays
N_LAUNCH_RUNS = 20   # timed direct-launch steps
BYTES = 1 << 20      # 1 MB per memset (device work is irrelevant here)


def main() -> int:
    rt = get_hip_runtime()
    # The legacy default stream cannot be captured (HIP error 900), and the
    # engine's production stream question is separate from this host-cost
    # screen -- a dedicated capture stream carries both paths here.
    stream = rt.stream_create(nonblocking=True)
    buf = rt.malloc(BYTES)

    def chain_once():
        for _ in range(N_OPS):
            rt.memset_async(buf, 0, BYTES, stream)

    # Warm.
    chain_once()
    rt.stream_synchronize(stream)

    # A: direct launches, as the engine issues them today.
    direct = []
    for _ in range(N_LAUNCH_RUNS):
        t0 = time.perf_counter_ns()
        chain_once()
        direct.append(time.perf_counter_ns() - t0)
        rt.stream_synchronize(stream)

    # B: capture the chain once, instantiate, then replay.
    rt.stream_begin_capture(stream, 2)  # 2 = hipStreamCaptureModeGlobal
    chain_once()
    graph = rt.stream_end_capture(stream)
    rt.stream_synchronize(stream)  # leave capture mode
    exec_graph = rt.graph_instantiate(graph)

    # Warm replays, then timed ones.
    for _ in range(5):
        rt.graph_launch(exec_graph, stream)
    rt.stream_synchronize(stream)
    replay = []
    for _ in range(N_REPLAYS):
        # Drain first: without this, replay N+1 blocks in hipGraphLaunch
        # waiting for replay N's 784 MB of device work, and the timed region
        # measures queue backpressure instead of submission cost.
        rt.stream_synchronize(stream)
        t0 = time.perf_counter_ns()
        rt.graph_launch(exec_graph, stream)
        replay.append(time.perf_counter_ns() - t0)
    rt.stream_synchronize(stream)

    # Device-side sanity: both paths must complete without error.
    err = rt.get_last_error() if hasattr(rt, "get_last_error") else 0

    out = {
        "ops_per_step": N_OPS,
        "direct_launch_host_us": {
            "median": round(statistics.median(direct) / 1e3, 1),
            "min": round(min(direct) / 1e3, 1),
            "max": round(max(direct) / 1e3, 1),
        },
        "graph_replay_host_us": {
            "median": round(statistics.median(replay) / 1e3, 1),
            "min": round(min(replay) / 1e3, 1),
            "max": round(max(replay) / 1e3, 1),
        },
        "host_time_saved_ms_per_step": round(
            (statistics.median(direct) - statistics.median(replay)) / 1e6, 3
        ),
        "capture_instantiated": bool(graph) and bool(exec_graph),
        "last_error": int(err),
        "note": (
            "Host submission cost only, synthetic memset chain. The WALL "
            "ceiling is smaller: launches overlap device work, and the "
            "committed wall-minus-busy gap is ~1.7 ms per step."
        ),
    }
    import json

    print(json.dumps(out, indent=2))
    rt.graph_exec_destroy(exec_graph)
    rt.graph_destroy(graph)
    rt.free(buf)
    if hasattr(rt, "stream_destroy"):
        rt.stream_destroy(stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())