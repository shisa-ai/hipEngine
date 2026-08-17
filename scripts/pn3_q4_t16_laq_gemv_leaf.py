#!/usr/bin/env python3
"""PN3 leaf baseline: exact local32 Q4T16 GEMV for Qwen3.6-35B linear attention.

Measures the ``dense_single_local32_bf16_bf16_out`` leaf that serves the c1
linear-attention ``attn_qkv`` (2048->8192) and ``attn_gate`` (2048->4096)
projections on gfx1151. Synthetic Q4_K weights are repacked to the T16 tile
layout so the byte access pattern / traffic is identical to production; values
are deterministic random bytes (timing-only, no correctness binding).

Outputs per-shape median launch ms, effective GB/s, and a projected 30-layer
per-token stage cost so the PN3 declaration can bind a measured complete-wall
ceiling for the #1 device stage (decode_linear_attn_qkv_gate, 7.74 ms/token on
the PN2 baseline).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
)
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

LAYERS = 30
SHAPES = {
    "attn_qkv": (2048, 8192),
    "attn_gate": (2048, 4096),
}
QK_K = 256
GGUF_Q4_K_BLOCK_BYTES = 144


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray):
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _event_ms(runtime, fn: Callable[[], None], *, burst: int) -> float:
    """Wall-clock burst timing with device sync, in milliseconds (robust on
    this gfx1151/ROCm combo where HIP event elapsed can report 0)."""
    import time

    fn()  # ensure in-flight enqueue settles before the timed burst
    runtime.device_synchronize()
    start = time.perf_counter()
    for _ in range(burst):
        fn()
    runtime.device_synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / burst * 1e3


def _make_raw_q4_k(rng: np.random.Generator, in_features: int, out_features: int) -> np.ndarray:
    blocks_per_row = in_features // QK_K
    bytes_per_row = blocks_per_row * GGUF_Q4_K_BLOCK_BYTES
    return rng.integers(0, 256, size=(1, out_features, bytes_per_row), dtype=np.uint8)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--burst", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/pn3-laq-gemv-leaf.json"))
    args = parser.parse_args()

    runtime = get_hip_runtime()
    library = build_gguf_t16_selected_gemv(require_cached=args.require_cached_build)
    rng = np.random.default_rng(args.seed)

    results = {}
    buffers = []
    try:
        x_host = _bf16_bits(rng.standard_normal(2048))
        x_dev = _upload(runtime, x_host)
        buffers.append(x_dev)
        for name, (in_features, out_features) in SHAPES.items():
            raw = _make_raw_q4_k(rng, in_features, out_features)
            tile16 = repack_gguf_q4_k_tile16(raw)
            tiles_host = np.ascontiguousarray(tile16.tiles)
            out_host = np.zeros((1, out_features), dtype=np.uint16)
            tiles_dev = _upload(runtime, tiles_host)
            out_dev = _upload(runtime, out_host)
            buffers.extend((tiles_dev, out_dev))
            tile_bytes = int(tiles_host.nbytes)

            def launcher(tiles_dev=tiles_dev, out_dev=out_dev, in_features=in_features, out_features=out_features):
                gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
                    x_dev.ptr,
                    tiles_dev.ptr,
                    out_dev.ptr,
                    1,
                    in_features,
                    out_features,
                    library=library,
                    runtime=runtime,
                )

            for _ in range(args.warmups):
                launcher()
            runtime.device_synchronize()
            timings = [
                _event_ms(runtime, launcher, burst=args.burst)
                for _ in range(args.samples)
            ]
            median_ms = statistics.median(timings)
            gbps = tile_bytes / (median_ms * 1e-3) / 1e9
            results[name] = {
                "shape": [in_features, out_features],
                "tile_bytes": tile_bytes,
                "median_ms": median_ms,
                "samples_ms": timings,
                "effective_gbps": gbps,
            }
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    pair_ms = results["attn_qkv"]["median_ms"] + results["attn_gate"]["median_ms"]
    stage_proj_ms = pair_ms * LAYERS
    results["_summary"] = {
        "layers": LAYERS,
        "pair_ms_per_layer": pair_ms,
        "stage_projected_ms_per_token": stage_proj_ms,
        "pn2_stage_measured_ms_per_token": 7.74,
        "projected_vs_pn2_ratio": stage_proj_ms / 7.74,
        "effective_gbps_pair": (
            (results["attn_qkv"]["tile_bytes"] + results["attn_gate"]["tile_bytes"])
            / (pair_ms * 1e-3)
            / 1e9
        ),
    }

    print(json.dumps(results, indent=2))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
