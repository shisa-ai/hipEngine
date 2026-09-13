#!/usr/bin/env python3
"""Screen the planar-qmicro Q6T16 WMMA prefill geometry ladder at prefill rows.

The gfx1151 package policy routes the wide planar-Q6 down shape
``(K, N) = (17408, 5120)`` to the historical ``shared4`` owner (2 waves x 1
row tile = 32 rows/block, 64 threads) from row 256 upward.  That band was set
when the screened rows were the verify frontier (4-255); the gfx1100 package
policy for the same shape instead hands rows 512+ back to the plain owner.

This probe measures the whole registered shared-weight ladder on real Qwen3.8
planar-Q6 tensors at prefill rows (256-4096), so a large-row band is chosen
from measurement instead of by analogy.  Siblings share the K16 WMMA
association but are not guaranteed BF16-bit exact, so the probe reports both
bit-equality and max absolute deviation against the plain parent.

Usage:
  PYTHONPATH=. python3 scripts/qwen38_q6_planar_prefill_large_row_screen.py \\
      --rows 256 512 1024 2048 4096 --output /tmp/q6-large-row-screen.json
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


def _bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--burst", type=int, default=8)
    parser.add_argument("--rows", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="*",
        default=["ffn_down"],
        help="subset of role names to screen (ffn_down, attn_qkv, attn_v)",
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
    from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_gfx1100_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r9_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    cases = []
    for role, name in (
        ("ffn_down", "blk.0.ffn_down.weight"),
        ("attn_qkv", "blk.0.attn_qkv.weight"),
        ("attn_v", "blk.3.attn_v.weight"),
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

    def event_ms(function, burst: int) -> float:
        start = runtime.event_create()
        stop = runtime.event_create()
        try:
            runtime.event_record(start)
            for _ in range(burst):
                function()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            return float(runtime.event_elapsed_time_ms(start, stop)) / burst
        finally:
            runtime.event_destroy(stop)
            runtime.event_destroy(start)

    # (name, launcher, block rows x cols) -- geometries read from the exported
    # symbols in hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_t16_gemv.hip.
    candidates = {
        "plain": (
            gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
            "waves2/rows?/cols32",
        ),
        "shared4": (gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out, "2w x 1r x 32c (32 rows, 64 thr)"),
        "shared3r1": (gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out, "3w x 1r x 32c (48 rows, 96 thr)"),
        "shared4_row64": (gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_bf16_out, "4w x 1r x 48c (64 rows, 128 thr)"),
        "shared4r3": (gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out, "4w x 3r x 32c (192 rows, 128 thr)"),
        "shared4r4": (gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out, "4w x 4r x 32c (256 rows, 128 thr)"),
        "shared4r6": (gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out, "4w x 6r x 32c (384 rows, 128 thr)"),
        "shared4r9": (gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r9_bf16_bf16_out, "4w x 9r x 32c (576 rows, 128 thr)"),
        "shared4_gfx1100": (gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_gfx1100_bf16_bf16_out, "4w x 4r x 48c (256 rows, 128 thr)"),
    }

    results = {
        "kind": "q6-planar-prefill-large-row-screen",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": detect_device_name(),
        "model": args.model,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "burst": args.burst,
        "geometries": {name: geo for name, (_, geo) in candidates.items()},
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
        try:
            for rows in args.rows:
                rng = np.random.default_rng(2_026_091_300 + index * 1000 + rows)
                x = _bf16_bits(
                    rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
                )
                x_device = upload(x)
                outs = {
                    cname: malloc(rows * out_features * 2, runtime=runtime)
                    for cname in candidates
                }

                def run(cname, fn):
                    fn(
                        x_device.ptr,
                        planar_device.ptr,
                        outs[cname].ptr,
                        rows,
                        in_features,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )

                burst = max(1, min(args.burst, max(1, 65536 // rows)))
                try:
                    for _ in range(args.warmups):
                        for cname, (fn, _) in candidates.items():
                            run(cname, fn)
                    timings = {
                        cname: statistics.median(
                            [
                                event_ms(lambda c=cname, f=fn: run(c, f), burst)
                                for _ in range(args.repetitions)
                            ]
                        )
                        for cname, (fn, _) in candidates.items()
                    }
                    host_plain = _bits_to_f32(
                        download(outs["plain"], (rows, out_features))
                    )
                    row_entry = {
                        "rows": rows,
                        "burst": burst,
                        "plain_ms": round(timings["plain"], 4),
                    }
                    for cname in candidates:
                        if cname == "plain":
                            continue
                        host_cand = _bits_to_f32(
                            download(outs[cname], (rows, out_features))
                        )
                        row_entry[f"{cname}_ms"] = round(timings[cname], 4)
                        row_entry[f"{cname}_bit_equal"] = bool(
                            np.array_equal(host_plain, host_cand)
                        )
                        row_entry[f"{cname}_max_abs"] = float(
                            np.max(np.abs(host_plain - host_cand))
                        )
                        row_entry[f"{cname}_vs_plain"] = round(
                            timings["plain"] / timings[cname], 3
                        )
                    case_entry["rows"].append(row_entry)
                    best = min(
                        (c for c in candidates), key=lambda c: timings[c]
                    )
                    printable = "  ".join(
                        f"{c}={row_entry[f'{c}_ms']:8.3f}"
                        for c in candidates
                    )
                    print(f"rows={rows:5d}: {printable}   best={best}")
                finally:
                    for buffer in outs.values():
                        free(buffer, runtime=runtime)
                    free(x_device, runtime=runtime)
        finally:
            free(planar_device, runtime=runtime)
            results["cases"].append(case_entry)

    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=1)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
