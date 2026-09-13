#!/usr/bin/env python3
"""Packet 4 L1: screen the dense Q4T16 gate/up dual-SiLU family at frontier rows.

Launches the registered sibling owners on real Qwen3.8-27B gate/up tensors at
the staged verify-frontier row counts and compares each against the one-wave
256-row parent (the current C8/K3 cycle owner): bit-exactness of the fused
SiLU output plus event-timed medians. Same methodology as the packet3 q6
row screen (scripts/qwen38_packet3_q6_planar_wmma_row_leaf.py).
"""

from __future__ import annotations

import argparse
import statistics
from datetime import datetime, timezone

import numpy as np


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounded = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
    )
    parser.add_argument("--burst", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--rows", type=int, nargs="+",
        default=[4, 6, 8, 12, 16, 20, 24, 28, 32, 33, 36, 48, 64, 65, 128],
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.loading.gguf import GGUFReader
    from hipengine.benchmark.provenance import detect_device_name
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        build_gguf_k_t16_selected_prefill,
        gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
        gguf_q4_k_t16_dense_dual_wmma_prefill_row32_silu_bf16_bf16_out,
        gguf_q4_k_t16_dense_dual_wmma_prefill_row48_silu_bf16_bf16_out,
        gguf_q4_k_t16_dense_dual_wmma_prefill_row64_silu_bf16_bf16_out,
        gguf_q4_k_t16_dense_dual_wmma_prefill_row128_silu_bf16_bf16_out,
        gguf_q4_k_t16_dense_dual_wmma_smallm_silu_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    gate_name = "blk.0.ffn_gate.weight"
    up_name = "blk.0.ffn_up.weight"
    gate_info = reader.tensor_info(gate_name)
    up_info = reader.tensor_info(up_name)
    in_features = int(gate_info.shape[1])
    out_features = int(gate_info.shape[0])
    assert int(up_info.shape[1]) == in_features and int(up_info.shape[0]) == out_features

    library = build_gguf_k_t16_selected_prefill(load=True)

    gate_tiles = np.ascontiguousarray(
        repack_gguf_q4_k_tile16(
            np.ascontiguousarray(reader.tensor_data(gate_name))[None, ...]
        ).tiles
    )
    up_tiles = np.ascontiguousarray(
        repack_gguf_q4_k_tile16(
            np.ascontiguousarray(reader.tensor_data(up_name))[None, ...]
        ).tiles
    )

    def upload(host: np.ndarray):
        host = np.ascontiguousarray(host)
        buffer = malloc(host.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
        return buffer

    def download(buffer, shape):
        host = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), buffer, runtime=runtime)
        return host

    def event_ms(function) -> float:
        start = runtime.event_create()
        stop = runtime.event_create()
        try:
            runtime.event_record(start)
            for _ in range(args.burst):
                function()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            return float(runtime.event_elapsed_time_ms(start, stop)) / args.burst
        finally:
            runtime.event_destroy(stop)
            runtime.event_destroy(start)

    candidates = {
        "parent": gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
        "row32": gguf_q4_k_t16_dense_dual_wmma_prefill_row32_silu_bf16_bf16_out,
        "row48": gguf_q4_k_t16_dense_dual_wmma_prefill_row48_silu_bf16_bf16_out,
        "row64": gguf_q4_k_t16_dense_dual_wmma_prefill_row64_silu_bf16_bf16_out,
        "row128": gguf_q4_k_t16_dense_dual_wmma_prefill_row128_silu_bf16_bf16_out,
        "smallm": gguf_q4_k_t16_dense_dual_wmma_smallm_silu_bf16_bf16_out,
    }

    gate_device = upload(gate_tiles)
    up_device = upload(up_tiles)
    results = {
        "kind": "packet4-q4-dual-silu-row-screen",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": detect_device_name(),
        "model": args.model,
        "in_features": in_features,
        "out_features": out_features,
        "burst": args.burst,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "rows": [],
    }
    try:
        for rows in args.rows:
            rng = np.random.default_rng(2_026_090_604 + rows)
            x = _bf16_bits(
                rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
            )
            x_device = upload(x)
            outs = {
                name: malloc(rows * out_features * 2, runtime=runtime)
                for name in candidates
            }

            def run(name, fn):
                fn(
                    x_device.ptr,
                    gate_device.ptr,
                    up_device.ptr,
                    outs[name].ptr,
                    rows,
                    in_features,
                    out_features,
                    library=library,
                    runtime=runtime,
                )

            try:
                active = {
                    name: fn
                    for name, fn in candidates.items()
                    if name != "smallm" or rows <= 16
                }
                for _ in range(args.warmups):
                    for name, fn in active.items():
                        run(name, fn)
                timings = {
                    name: statistics.median(
                        [event_ms(lambda fn=fn: run(name, fn)) for _ in range(args.repetitions)]
                    )
                    for name, fn in active.items()
                }
                host_parent = download(outs["parent"], (rows, out_features))
                row_entry = {
                    "rows": rows,
                    "parent_ms": round(timings["parent"], 4),
                }
                for name in ("row32", "row48", "row64", "row128", "smallm"):
                    if name not in timings:
                        continue
                    host_cand = download(outs[name], (rows, out_features))
                    bit_equal = bool(np.array_equal(host_parent, host_cand))
                    row_entry[f"{name}_ms"] = round(timings[name], 4)
                    row_entry[f"{name}_bit_equal"] = bit_equal
                    row_entry[f"{name}_vs_parent"] = round(
                        timings["parent"] / timings[name], 3
                    )
                results["rows"].append(row_entry)
                printable = {
                    k: v for k, v in row_entry.items() if k.endswith("_ms") or k == "rows"
                }
                print(f"rows={rows:4d}: " + "  ".join(f"{k}={v}" for k, v in printable.items()))
                for name in ("row32", "row48", "row64", "row128", "smallm"):
                    if f"{name}_bit_equal" not in row_entry:
                        continue
                    if not row_entry[f"{name}_bit_equal"]:
                        print(f"    {name}: BIT-EXACTNESS FAIL")
            finally:
                for buffer in outs.values():
                    free(buffer, runtime=runtime)
                free(x_device, runtime=runtime)
    finally:
        free(gate_device, runtime=runtime)
        free(up_device, runtime=runtime)

    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=1)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    import json

    raise SystemExit(main())
