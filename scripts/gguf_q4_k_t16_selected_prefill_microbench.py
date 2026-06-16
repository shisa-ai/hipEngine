#!/usr/bin/env python3
"""Microbench the current GGUF Q4_K T16 selected-dual WMMA prefill kernel.

This is a diagnostic baseline for the llama.cpp MMQ/Q8_1 prefill detour.  It
builds a synthetic compact-selected MoE fixture with Q4_K gate/up expert weights
repacked to the resident T16 layout, launches the current hipEngine selected
WMMA prefill kernel repeatedly, and emits a compact JSON artifact.

The script does not load a full GGUF model and does not validate model quality;
it is a same-shape kernel baseline to beat with a future Q8_1-activation/MMQ
prototype.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_GIB = 1 << 30


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return (((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)).reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _make_activation(rows: int, hidden: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Keep magnitudes modest so output finiteness is a useful sanity check and
    # the kernel is not measuring overflow behavior.
    return _f32_to_bf16_u16((rng.standard_normal((rows, hidden)) * 0.02).astype(np.float32))


def _make_uniform_compact_metadata(experts: int, rows_per_expert: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    counts = np.full(experts, rows_per_expert, dtype=np.int64)
    expert_start_compact = np.zeros(experts + 1, dtype=np.int64)
    expert_start_compact[1:] = np.cumsum(counts)

    padded_counts = ((counts + 15) // 16) * 16
    expert_start_wmma = np.zeros(experts + 1, dtype=np.int64)
    expert_start_wmma[1:] = np.cumsum(padded_counts)
    tile_expert = np.asarray(
        [expert for expert, padded in enumerate(padded_counts) for _ in range(int(padded) // 16)],
        dtype=np.int64,
    )
    compact_rows = int(expert_start_compact[-1])
    wmma_total_rows = int(expert_start_wmma[-1])
    return expert_start_compact, expert_start_wmma, tile_expert, compact_rows, wmma_total_rows


def _copy_to_device(arr: np.ndarray, *, runtime: Any):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    contiguous = np.ascontiguousarray(arr)
    dev = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(dev, host_array_ptr(contiguous), runtime=runtime)
    return dev, contiguous


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _git_dirty() -> list[str]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except Exception:
        return []
    return [line for line in out.splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--out-features-a", type=int, default=4096)
    parser.add_argument("--out-features-b", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--rows-per-expert", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", type=Path, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.hidden % 256:
        raise SystemExit("--hidden must be divisible by 256 for Q4_K")
    if args.out_features_a % 16 or args.out_features_b % 16:
        raise SystemExit("--out-features-a/b must be divisible by 16")
    if args.experts <= 0 or args.rows_per_expert <= 0:
        raise SystemExit("--experts and --rows-per-expert must be positive")
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be positive and --warmup non-negative")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, free, host_array_ptr, malloc, memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
        build_gguf_q4_k_t16_selected_prefill,
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from tests._gguf_synthetic_weights import make_q4_k_weight

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_t16_selected_prefill(
        load=True,
        require_cached=args.require_cached_build,
    )
    reset_memory_stats()

    expert_start_compact, expert_start_wmma, tile_expert, compact_rows, wmma_total_rows = _make_uniform_compact_metadata(
        args.experts, args.rows_per_expert
    )
    x_host = _make_activation(compact_rows, args.hidden, seed=args.seed)

    base_a = make_q4_k_weight(args.out_features_a, args.hidden)
    base_b = make_q4_k_weight(args.out_features_b, args.hidden)
    qweight_a = np.ascontiguousarray(
        np.stack([np.roll(base_a, shift=expert, axis=0) for expert in range(args.experts)], axis=0)
    )
    qweight_b = np.ascontiguousarray(
        np.stack([np.roll(base_b, shift=expert + 3, axis=0) for expert in range(args.experts)], axis=0)
    )
    tiles_a = repack_gguf_q4_k_tile16(qweight_a).tiles
    tiles_b = repack_gguf_q4_k_tile16(qweight_b).tiles
    out_host = np.zeros((compact_rows, args.out_features_a + args.out_features_b), dtype=np.uint16)

    bufs = []
    stream = runtime.stream_create()
    try:
        for arr in (x_host, expert_start_compact, expert_start_wmma, tile_expert, tiles_a, tiles_b):
            dev, _ = _copy_to_device(arr, runtime=runtime)
            bufs.append(dev)
        x_dev, start_compact_dev, start_wmma_dev, tile_expert_dev, tiles_a_dev, tiles_b_dev = bufs
        out_dev = malloc(out_host.nbytes, runtime=runtime)
        bufs.append(out_dev)

        def launch() -> None:
            gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
                x_dev.ptr,
                start_compact_dev.ptr,
                start_wmma_dev.ptr,
                tile_expert_dev.ptr,
                tiles_a_dev.ptr,
                tiles_b_dev.ptr,
                out_dev.ptr,
                compact_rows,
                args.hidden,
                args.out_features_a,
                args.out_features_b,
                args.experts,
                wmma_total_rows,
                stream=stream,
                library=library,
                runtime=runtime,
            )

        for _ in range(args.warmup):
            launch()
        runtime.stream_synchronize(stream)
        start = time.perf_counter()
        for _ in range(args.iters):
            launch()
        runtime.stream_synchronize(stream)
        elapsed_s = time.perf_counter() - start
        ms_per_call = elapsed_s * 1e3 / args.iters

        launch()
        runtime.stream_synchronize(stream)
        copy_device_to_host(host_array_ptr(out_host), out_dev, runtime=runtime)
        out_f32 = _bf16_u16_to_f32(out_host)
        finite = bool(np.isfinite(out_f32).all())
        checksum = float(out_f32.astype(np.float64).sum())
        max_abs = float(np.max(np.abs(out_f32))) if out_f32.size else 0.0

        out_features_total = args.out_features_a + args.out_features_b
        logical_fma = int(compact_rows * out_features_total * args.hidden)
        logical_tflops = (2.0 * logical_fma) / (ms_per_call / 1e3) / 1e12
        result: dict[str, Any] = {
            "schema": 1,
            "status": "diagnostic_retained",
            "performance_claim": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_tag": "gguf_q4_k_t16_selected_prefill_microbench",
            "reason_not_promoted": "Synthetic kernel microbench baseline only; no runtime dispatch/default change.",
            "software": {
                "hipengine_commit": _git_commit(),
                "hipengine_dirty_files": _git_dirty(),
                "compiler_version_file": str(args.compiler_version_file) if args.compiler_version_file else None,
                "python": sys.version.split()[0],
            },
            "shape": {
                "hidden": args.hidden,
                "out_features_a": args.out_features_a,
                "out_features_b": args.out_features_b,
                "out_features_total": out_features_total,
                "experts": args.experts,
                "rows_per_expert": args.rows_per_expert,
                "compact_rows": compact_rows,
                "wmma_total_rows": wmma_total_rows,
                "topology_note": "Uniform selected rows across a reduced expert set; use to compare kernel designs, not as full-model routing evidence.",
            },
            "timing": {
                "warmup": args.warmup,
                "iters": args.iters,
                "elapsed_s": elapsed_s,
                "ms_per_call": ms_per_call,
                "calls_per_s": 1000.0 / ms_per_call,
                "logical_fma": logical_fma,
                "logical_tflops": logical_tflops,
            },
            "memory": {
                "host_input_mib": x_host.nbytes / (1 << 20),
                "host_tiles_a_mib": tiles_a.nbytes / (1 << 20),
                "host_tiles_b_mib": tiles_b.nbytes / (1 << 20),
                "host_output_mib": out_host.nbytes / (1 << 20),
                "tracked_peak_allocated_gib": memory_stats()["peak_allocated_bytes"] / _GIB,
            },
            "sanity": {
                "finite_output": finite,
                "output_checksum_f64": checksum,
                "output_max_abs": max_abs,
            },
        }
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
        runtime.stream_destroy(stream)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["sanity"]["finite_output"]:
        raise SystemExit("kernel output contained non-finite values")


if __name__ == "__main__":
    main()
