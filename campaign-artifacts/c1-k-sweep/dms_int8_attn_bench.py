#!/usr/bin/env python3
"""Microbenchmark for the compact DMS INT8 decode attention kernel.

Mirrors scripts/dms_compact_attn_long_bench.py with INT8 payloads + FP32
per-token scales, timing hipengine_dms_compact_attn_decode_splitk_int8 at a
given live count. Diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.attention.dms_compact_int8 import (
    build_dms_compact_int8,
    dms_compact_attn_decode_splitk_int8,
)


def _timed_ms(runtime, launch, *, burst: int) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(int(burst)):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return float(runtime.event_elapsed_time_ms(start, stop)) / int(burst)
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=int, default=8192)
    parser.add_argument("--q-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--burst", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    live = int(args.live)
    q_heads = int(args.q_heads)
    kv_heads = int(args.kv_heads)
    dim = int(args.head_dim)
    chunk = 256
    splits = (live + chunk - 1) // chunk

    library = build_dms_compact_int8(load=True)
    runtime = get_hip_runtime()
    q = np.linspace(-0.125, 0.125, q_heads * dim, dtype=np.float32).reshape(1, q_heads, dim)
    slots = kv_heads * live
    k_int8 = np.full((slots, dim), 11, dtype=np.int8)
    v_int8 = np.full((slots, dim), 13, dtype=np.int8)
    k_scales = np.full(slots, 0.01, dtype=np.float32)
    v_scales = np.full(slots, 0.01, dtype=np.float32)
    base = np.arange(kv_heads, dtype=np.int32).reshape(1, -1) * live
    live_counts = np.full((1, kv_heads), live, dtype=np.int32)
    out = np.empty((1, q_heads, dim), dtype=np.float32)

    buffers = {}
    host_inputs = {
        "q": q, "k": k_int8, "v": v_int8, "ks": k_scales, "vs": v_scales,
        "base": base, "live": live_counts,
    }
    try:
        for name, array in host_inputs.items():
            buffer = malloc(array.nbytes, runtime=runtime)
            buffers[name] = buffer
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes,
                                runtime=runtime)
        buffers["out"] = malloc(out.nbytes, runtime=runtime)
        buffers["po"] = malloc(q_heads * splits * dim * 4, runtime=runtime)
        buffers["pm"] = malloc(q_heads * splits * 4, runtime=runtime)
        buffers["pl"] = malloc(q_heads * splits * 4, runtime=runtime)

        def launch() -> None:
            dms_compact_attn_decode_splitk_int8(
                buffers["q"].ptr, buffers["k"].ptr, buffers["v"].ptr,
                buffers["base"].ptr, buffers["live"].ptr,
                buffers["po"].ptr, buffers["pm"].ptr, buffers["pl"].ptr,
                buffers["out"].ptr,
                1, q_heads, kv_heads, dim, dim ** -0.5, chunk, splits,
                k_scale_ptr=buffers["ks"].ptr, v_scale_ptr=buffers["vs"].ptr,
                library=library, runtime=runtime,
            )

        for _ in range(int(args.warmup)):
            launch()
        runtime.device_synchronize()
        samples = [
            _timed_ms(runtime, launch, burst=int(args.burst))
            for _ in range(int(args.repeats))
        ]
        copy_device_to_host(host_array_ptr(out), buffers["out"], out.nbytes,
                            runtime=runtime)
    finally:
        for buffer in reversed(tuple(buffers.values())):
            free(buffer, runtime=runtime)

    payload_bytes = 2 * slots * dim  # int8 K+V
    result = {
        "kind": "hipengine_dms_compact_attn_int8_bench",
        "geometry": {"live_per_kv_head": live, "q_heads": q_heads,
                     "kv_heads": kv_heads, "head_dim": dim,
                     "chunk_size": chunk, "num_splits": splits, "rows": 1},
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "finite_output": bool(np.isfinite(out).all()),
        "payload_bytes_read_logical": payload_bytes,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
