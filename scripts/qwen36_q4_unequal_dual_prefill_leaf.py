#!/usr/bin/env python3
"""Screen exact unequal-output Q4T16 dual prefill on actual Qwen3.6-27B weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Callable

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
    build_gguf_k_t16_selected_prefill,
    gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("/tmp/qwen36-q4-unequal-dual-prefill-leaf.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--rows", default="512,1024,4096")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0x36D00)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _tracked_status() -> list[str]:
    output = subprocess.check_output(
        ("git", "status", "--short", "--untracked-files=no"),
        cwd=ROOT,
        text=True,
    )
    return [line for line in output.splitlines() if line]


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
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            fn()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return float(runtime.event_elapsed_time_ms(start, end)) / burst
    finally:
        runtime.event_destroy(end)
        runtime.event_destroy(start)


def _screen_row(
    *,
    runtime,
    library,
    tiles_a: np.ndarray,
    tiles_b: np.ndarray,
    rows: int,
    warmups: int,
    samples: int,
    burst: int,
    seed: int,
) -> dict[str, object]:
    in_features = 5_120
    out_features_a = 10_240
    out_features_b = 6_144
    rng = np.random.default_rng(seed + rows)
    x = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        tiles_a_dev = _upload(runtime, tiles_a)
        tiles_b_dev = _upload(runtime, tiles_b)
        bytes_a = rows * out_features_a * 2
        bytes_b = rows * out_features_b * 2
        control_a_dev = malloc(bytes_a, runtime=runtime)
        control_b_dev = malloc(bytes_b, runtime=runtime)
        candidate_a_dev = malloc(bytes_a, runtime=runtime)
        candidate_b_dev = malloc(bytes_b, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                tiles_a_dev,
                tiles_b_dev,
                control_a_dev,
                control_b_dev,
                candidate_a_dev,
                candidate_b_dev,
            )
        )

        def control() -> None:
            gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
                x_dev.ptr,
                tiles_a_dev.ptr,
                control_a_dev.ptr,
                rows,
                in_features,
                out_features_a,
                library=library,
                runtime=runtime,
            )
            gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
                x_dev.ptr,
                tiles_b_dev.ptr,
                control_b_dev.ptr,
                rows,
                in_features,
                out_features_b,
                library=library,
                runtime=runtime,
            )

        def candidate() -> None:
            gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out(
                x_dev.ptr,
                tiles_a_dev.ptr,
                tiles_b_dev.ptr,
                candidate_a_dev.ptr,
                candidate_b_dev.ptr,
                rows,
                in_features,
                out_features_a,
                out_features_b,
                library=library,
                runtime=runtime,
            )

        for _ in range(warmups):
            control()
            candidate()
        runtime.device_synchronize()
        timings = {"control": [], "candidate": []}
        wins = 0
        for sample in range(samples):
            order = (
                ("control", control, "candidate", candidate)
                if sample % 2 == 0
                else ("candidate", candidate, "control", control)
            )
            pair: dict[str, float] = {}
            for index in (0, 2):
                name = order[index]
                pair[name] = _event_ms(runtime, order[index + 1], burst=burst)
                timings[name].append(pair[name])
            wins += int(pair["candidate"] < pair["control"])

        control()
        candidate()
        runtime.device_synchronize()
        control_a = np.empty((rows, out_features_a), dtype=np.uint16)
        control_b = np.empty((rows, out_features_b), dtype=np.uint16)
        candidate_a = np.empty_like(control_a)
        candidate_b = np.empty_like(control_b)
        for host, device in (
            (control_a, control_a_dev),
            (control_b, control_b_dev),
            (candidate_a, candidate_a_dev),
            (candidate_b, candidate_b_dev),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    medians = {
        name: statistics.median(values) for name, values in timings.items()
    }
    return {
        "samples_ms": timings,
        "median_ms": medians,
        "speedup": medians["control"] / medians["candidate"],
        "candidate_delta_percent":
            (medians["candidate"] / medians["control"] - 1.0) * 100.0,
        "candidate_wins": wins,
        "pair_count": samples,
        "bf16_mismatches_a": int(np.count_nonzero(candidate_a != control_a)),
        "bf16_mismatches_b": int(np.count_nonzero(candidate_b != control_b)),
        "control_a_sha256": hashlib.sha256(control_a.astype("<u2").tobytes()).hexdigest(),
        "candidate_a_sha256": hashlib.sha256(candidate_a.astype("<u2").tobytes()).hexdigest(),
        "control_b_sha256": hashlib.sha256(control_b.astype("<u2").tobytes()).hexdigest(),
        "candidate_b_sha256": hashlib.sha256(candidate_b.astype("<u2").tobytes()).hexdigest(),
    }


def main() -> int:
    args = _parse_args()
    dirty = _tracked_status()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked worktree must be clean; pass --allow-dirty for a screen")
    rows = tuple(int(value) for value in args.rows.split(","))
    if not rows or any(value <= 0 for value in rows) or len(set(rows)) != len(rows):
        raise ValueError("rows must contain unique positive integers")
    if args.warmups < 0 or min(args.samples, args.burst) <= 0:
        raise ValueError("warmups must be non-negative; samples/burst positive")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    reader = GGUFReader(args.model)
    names = (
        f"blk.{args.layer}.attn_qkv.weight",
        f"blk.{args.layer}.attn_gate.weight",
    )
    infos = tuple(reader.tensor_info(name) for name in names)
    if any(info.ggml_type_name != "Q4_K" for info in infos):
        raise ValueError(f"layer {args.layer} is not a Q4/Q4 pair")
    raw_a, raw_b = (np.asarray(reader.tensor_data(name)) for name in names)
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    runtime = get_hip_runtime()
    library = build_gguf_k_t16_selected_prefill(
        load=True, require_cached=args.require_cached_build
    )
    results = {
        str(value): _screen_row(
            runtime=runtime,
            library=library,
            tiles_a=tiles_a,
            tiles_b=tiles_b,
            rows=value,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
            seed=args.seed,
        )
        for value in rows
    }
    exact = all(
        result["bf16_mismatches_a"] == result["bf16_mismatches_b"] == 0
        for result in results.values()
    )
    all_positive = all(
        result["candidate_wins"] == result["pair_count"]
        and result["speedup"] > 1.0
        for result in results.values()
    )
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "leaf_admitted" if exact and all_positive else "leaf_rejected",
        "performance_claim": False,
        "repo_revision": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
        "hardware": {
            "selector": os.environ.get("HIP_VISIBLE_DEVICES"),
            "arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        },
        "model": str(args.model.resolve()),
        "layer": args.layer,
        "weights": list(names),
        "shape": {"k": 5_120, "n_a": 10_240, "n_b": 6_144},
        "protocol": {
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counterbalanced HIP events",
            "control": "two retained shared-B Q4T16 singleton projections",
            "candidate": "paired 6144-column prefix plus singleton-geometry QKV tail",
        },
        "results": results,
        "correctness_pass": exact,
        "all_required_rows_positive": all_positive,
        "tracked_dirty_paths": dirty,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
