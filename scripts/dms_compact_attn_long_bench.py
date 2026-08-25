#!/usr/bin/env python3
"""Cached-capable long-context microbenchmark for compact DMS decode attention."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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
from hipengine.kernels.hip_gfx1100.attention.dms_compact import (
    build_dms_compact,
    dms_compact_attn_decode_splitk_bf16,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=int, default=69641)
    parser.add_argument("--q-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--burst", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    live = int(args.live)
    q_heads = int(args.q_heads)
    kv_heads = int(args.kv_heads)
    dim = int(args.head_dim)
    if live <= 0 or q_heads <= 0 or kv_heads <= 0 or dim <= 0:
        raise ValueError("DMS benchmark geometry must be positive")
    if q_heads % kv_heads:
        raise ValueError("DMS benchmark q_heads must be divisible by kv_heads")
    chunk = 256
    splits = (live + chunk - 1) // chunk
    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8").strip()
    )
    library = build_dms_compact(
        load=True,
        compiler_version=compiler_version,
        require_cached=bool(args.require_cached_build),
    )
    runtime = get_hip_runtime()
    q = np.linspace(-0.125, 0.125, q_heads * dim, dtype=np.float32).reshape(
        1, q_heads, dim
    )
    # Finite deterministic BF16 values avoid host-side FP32 staging of the
    # production-size K/V payload while preserving the exact memory traffic.
    slots = kv_heads * live
    k_bits = np.full((slots, dim), np.uint16(0x3D00), dtype=np.uint16)
    v_bits = np.full((slots, dim), np.uint16(0x3E00), dtype=np.uint16)
    base = np.arange(kv_heads, dtype=np.int32).reshape(1, -1) * live
    live_counts = np.full((1, kv_heads), live, dtype=np.int32)
    out = np.empty((1, q_heads, dim), dtype=np.float32)
    partial_out_nbytes = q_heads * splits * dim * np.dtype(np.float32).itemsize
    partial_stat_nbytes = q_heads * splits * np.dtype(np.float32).itemsize
    host_inputs = {
        "q": q,
        "k": k_bits,
        "v": v_bits,
        "base": base,
        "live": live_counts,
    }
    buffers = {}
    try:
        for name, array in host_inputs.items():
            buffer = malloc(array.nbytes, runtime=runtime)
            buffers[name] = buffer
            copy_host_to_device(
                buffer,
                host_array_ptr(array),
                array.nbytes,
                runtime=runtime,
            )
        buffers["out"] = malloc(out.nbytes, runtime=runtime)
        buffers["partial_out"] = malloc(partial_out_nbytes, runtime=runtime)
        buffers["partial_m"] = malloc(partial_stat_nbytes, runtime=runtime)
        buffers["partial_l"] = malloc(partial_stat_nbytes, runtime=runtime)

        def launch() -> None:
            dms_compact_attn_decode_splitk_bf16(
                buffers["q"].ptr,
                buffers["k"].ptr,
                buffers["v"].ptr,
                buffers["base"].ptr,
                buffers["live"].ptr,
                buffers["partial_out"].ptr,
                buffers["partial_m"].ptr,
                buffers["partial_l"].ptr,
                buffers["out"].ptr,
                1,
                q_heads,
                kv_heads,
                dim,
                dim**-0.5,
                chunk,
                splits,
                library=library,
                runtime=runtime,
            )

        for _ in range(int(args.warmup)):
            launch()
        runtime.device_synchronize()
        samples = [
            _timed_ms(runtime, launch, burst=int(args.burst))
            for _ in range(int(args.repeats))
        ]
        copy_device_to_host(
            host_array_ptr(out),
            buffers["out"],
            out.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(tuple(buffers.values())):
            free(buffer, runtime=runtime)

    payload_bytes = 2 * slots * dim * np.dtype(np.uint16).itemsize
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "hipengine_dms_compact_attn_long_bench",
        "geometry": {
            "rows": 1,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": dim,
            "live_per_kv_head": live,
            "chunk_size": chunk,
            "num_splits": splits,
        },
        "payload_bytes_read_logical": payload_bytes,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "finite_output": bool(np.isfinite(out).all()),
        "output_abs_max": float(np.max(np.abs(out))),
        "require_cached_build": bool(args.require_cached_build),
    }
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
