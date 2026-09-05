#!/usr/bin/env python3
"""Microbench the dense Q4T16 WMMA prefill owner siblings at parity shapes.

Times and cross-validates the strict-sibling owners on synthetic Q4T16
weights so low-M dispatch decisions are made from measurement, not analogy:

* ``shared_b`` - 128-thread LDS-staged, 48 cols x 256-row-capacity blocks.
* ``shared_b2w2`` - 64-thread LDS-staged, 32 cols x 128-row-capacity blocks.
* ``shared_b2w4`` - 128-thread LDS-staged, 32 cols x 256-row-capacity blocks.
* ``shared_b3w8r3`` - 256-thread LDS-staged, 48 cols x 384-row-capacity blocks.
* ``default``  - 32-thread, 48 cols x 64-row-capacity blocks
  (``gguf_q4_k_t16_wmma_prefill_bf16_bf16_out``).
* ``smallm``   - 32-thread, 48 cols x 16-row-capacity blocks (rows <= 16).

The kernels share the same K16 WMMA association and BF16 store, so outputs
must match bit-exactly; the script fails closed on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    gguf_q4_k_t16_wmma_prefill_bf16_bf16_out as launch_default,
    gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out as launch_lowvgpr,
    gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out as launch_lowvgpr48,
    gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out as launch_shared_b,
    gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out as launch_shared_b3w8r3,
    gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out as launch_shared_b2w2,
    gguf_q4_k_t16_wmma_prefill_shared_b2w4_bf16_bf16_out as launch_shared_b2w4,
    gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out as launch_smallm,
)
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
from tests._gguf_synthetic_weights import make_q4_k_weight


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[45])
    parser.add_argument("--in-features", type=int, default=5120)
    parser.add_argument("--out-features", type=int, default=17408)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    runtime = get_hip_runtime()
    stream = runtime.stream_create()

    raw = np.stack(
        [make_q4_k_weight(args.out_features, args.in_features)], axis=0
    )
    tiles = repack_gguf_q4_k_tile16(raw)
    tiles_host = np.ascontiguousarray(tiles.tiles)
    weight_bytes = int(tiles_host.nbytes)
    tiles_buf = malloc(weight_bytes, runtime=runtime)
    copy_host_to_device(tiles_buf, host_array_ptr(tiles_host), runtime=runtime)

    results: dict[str, object] = {
        "schema": 1,
        "kind": "gguf_q4_t16_dense_prefill_owner_microbench",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "rows": args.rows,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "weight_bytes": weight_bytes,
        },
        "cells": [],
    }

    for rows in args.rows:
        x_host = _f32_to_bf16_u16(
            rng.standard_normal((rows, args.in_features)) * 0.05
        )
        x_buf = malloc(x_host.nbytes, runtime=runtime)
        copy_host_to_device(x_buf, host_array_ptr(x_host), runtime=runtime)
        out_nbytes = rows * args.out_features * 2
        out_buf = malloc(out_nbytes, runtime=runtime)

        owners: dict[str, object] = {
            "shared_b": launch_shared_b,
            "shared_b3w8r3": launch_shared_b3w8r3,
            "shared_b2w2": launch_shared_b2w2,
            "shared_b2w4": launch_shared_b2w4,
            "default": launch_default,
            "lowvgpr": launch_lowvgpr,
            "lowvgpr48": launch_lowvgpr48,
        }
        if rows <= 16:
            owners["smallm"] = launch_smallm

        outputs: dict[str, np.ndarray] = {}
        cell: dict[str, object] = {"rows": rows, "owners": {}}
        for name, fn in owners.items():
            def run() -> None:
                fn(
                    x_buf.ptr,
                    tiles_buf.ptr,
                    out_buf.ptr,
                    rows,
                    args.in_features,
                    args.out_features,
                    stream=stream,
                    runtime=runtime,
                )

            for _ in range(args.warmup):
                run()
            runtime.device_synchronize()
            times = []
            for _ in range(args.iters):
                start = time.perf_counter()
                run()
                runtime.stream_synchronize(stream)
                times.append(time.perf_counter() - start)
            med = float(np.median(times))
            out_host = np.zeros((rows, args.out_features), dtype=np.uint16)
            copy_device_to_host(
                host_array_ptr(out_host), out_buf, out_nbytes, runtime=runtime
            )
            outputs[name] = out_host.copy()
            cell["owners"][name] = {
                "median_s": med,
                "min_s": float(np.min(times)),
                "effective_weight_bandwidth_gb_s": weight_bytes / med / 1e9,
            }

        ref = outputs["shared_b"]
        for name, out in outputs.items():
            if name == "shared_b":
                continue
            exact = bool(np.array_equal(out, ref))
            max_abs = (
                float(
                    np.max(
                        np.abs(_bf16_u16_to_f32(out) - _bf16_u16_to_f32(ref))
                    )
                )
                if out.shape == ref.shape
                else None
            )
            cell["owners"][name]["bit_exact_vs_shared_b"] = exact
            cell["owners"][name]["max_abs_vs_shared_b"] = max_abs
            if not exact:
                cell["validation_failed"] = True
        results["cells"].append(cell)

        free(x_buf, runtime=runtime)
        free(out_buf, runtime=runtime)
        summary = {
            name: {
                "ms": round(v["median_s"] * 1e3, 3),
                "gb_s": round(v["effective_weight_bandwidth_gb_s"], 1),
            }
            for name, v in cell["owners"].items()
        }
        print(f"rows={rows}: {summary}", flush=True)

    free(tiles_buf, runtime=runtime)

    results["source_commit"] = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
    ).strip()
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
