#!/usr/bin/env python3
"""Packet 3 L1 leaf: plain planar-Q6 WMMA prefill vs the shared4_row64 sibling.

The staged-cycle traces (packet1 attribution) show the q6-planar
wmma_prefill kernel at 64 launches/cycle, width-flat, ~0.8 ms/call = 9-11x
the 84.6 us one-stream floor for the 73 MB planar-Q6 down tensor. The
qualified cooperative sibling band starts at rows 33 ("rows<=32 keep
verifier ownership unchanged"). This probe measures both owners at the
verify-frontier row counts (4-32) plus the sibling band, on real Qwen3.8
planar-Q6 tensors, and compares outputs bit-exactly.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone

import numpy as np


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
    )
    parser.add_argument("--burst", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[4, 6, 12, 16, 24, 32, 33, 48]
    )
    parser.add_argument("--shapes", type=str, nargs="*", default=None,
                        help="optional subset of role names to screen")
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
    from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    cases = []
    for role, name in (
        ("ffn_down", "blk.0.ffn_down.weight"),
        ("recurrent_qkv", "blk.0.attn_qkv.weight"),
        ("attention_v", "blk.3.attn_v.weight"),
    ):
        if args.shapes and role not in args.shapes:
            continue
        info = reader.tensor_info(name)
        cases.append((role, name, int(info.shape[0]), int(info.shape[1])))

    library = build_gguf_q6_k_t16_gemv(load=True)

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

    plain_fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out
    row64_fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_bf16_out
    shared4_fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out

    results = {
        "kind": "packet3-q6-planar-wmma-prefill-row-screen",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": detect_device_name(),
        "model": args.model,
        "burst": args.burst,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cases": [],
    }
    for index, (role, name, out_features, in_features) in enumerate(cases):
        raw = np.ascontiguousarray(reader.tensor_data(name))
        planar = np.ascontiguousarray(
            repack_gguf_q6_k_tile16_qmicro_planar(raw[None, ...]).tiles
        )
        planar_device = upload(planar)
        case_entry = {
            "role": role,
            "tensor": name,
            "in_features": in_features,
            "out_features": out_features,
            "rows": [],
        }
        case_results = []
        try:
            for rows in args.rows:
                rng = np.random.default_rng(2_026_090_601 + index * 1000 + rows)
                x = _bf16_bits(
                    rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
                )
                x_device = upload(x)
                out_plain = malloc(rows * out_features * 2, runtime=runtime)
                out_row64 = malloc(rows * out_features * 2, runtime=runtime)
                out_shared4 = malloc(rows * out_features * 2, runtime=runtime)

                def plain() -> None:
                    plain_fn(
                        x_device.ptr,
                        planar_device.ptr,
                        out_plain.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )

                def row64() -> None:
                    row64_fn(
                        x_device.ptr,
                        planar_device.ptr,
                        out_row64.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )

                def shared4() -> None:
                    shared4_fn(
                        x_device.ptr,
                        planar_device.ptr,
                        out_shared4.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )

                for _ in range(args.warmups):
                    plain()
                    row64()
                plain_ms = statistics.median(
                    [event_ms(plain) for _ in range(args.repetitions)]
                )
                row64_ms = statistics.median(
                    [event_ms(row64) for _ in range(args.repetitions)]
                )
                shared4_ms = None
                if rows >= 129:
                    shared4_ms = statistics.median(
                        [event_ms(shared4) for _ in range(args.repetitions)]
                    )
                host_plain = download(out_plain, (rows, out_features))
                host_row64 = download(out_row64, (rows, out_features))
                bit_equal_row64 = bool(np.array_equal(host_plain, host_row64))
                mismatch = int(np.count_nonzero(host_plain != host_row64))
                max_delta = float(
                    np.abs(
                        host_plain.astype(np.float32)
                        - host_row64.astype(np.float32)
                    ).max()
                )
                entry = {
                    "rows": rows,
                    "plain_ms": round(plain_ms, 4),
                    "shared4_row64_ms": round(row64_ms, 4),
                    "shared4_ms": (
                        round(shared4_ms, 4) if shared4_ms is not None else None
                    ),
                    "bit_equal": bit_equal_row64,
                    "mismatched_elements": mismatch,
                    "max_abs_delta": max_delta,
                    "row64_vs_plain": round(plain_ms / row64_ms, 4),
                }
                case_entry["rows"].append(entry)
                case_results.append(entry)
                free(x_device, runtime=runtime)
                free(out_plain, runtime=runtime)
                free(out_row64, runtime=runtime)
                free(out_shared4, runtime=runtime)
        finally:
            free(planar_device, runtime=runtime)
        results["cases"].append(case_entry)
        for entry in case_results:
            speed = (
                f"{entry['row64_vs_plain']:.2f}x"
                if entry["row64_vs_plain"] >= 1
                else f"{1 / entry['row64_vs_plain']:.2f}x slower"
            )
            print(
                f"{role:>14} rows={entry['rows']:>3}: "
                f"plain {entry['plain_ms']:.3f} ms  "
                f"row64 {entry['shared4_row64_ms']:.3f} ms  "
                f"({speed})  bit_equal={entry['bit_equal']} "
                f"mismatch={entry['mismatched_elements']}"
            )

    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
