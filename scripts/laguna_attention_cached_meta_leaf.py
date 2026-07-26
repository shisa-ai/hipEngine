#!/usr/bin/env python3
"""Benchmark exact cached-metadata qrow4 attention on pp512 tile positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
    build_laguna_kv_attention,
    laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans,
    laguna_global_attention_prefill_qrow4_cached_online_bf16_spans,
    laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans,
    laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans,
)
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache


ROWS = 128
CONTEXT = 512
KV_HEADS = 8
HEAD_DIM = 128
GLOBAL_HEADS = 48
SWA_HEADS = 72
STARTS = (0, 128, 256, 384)
MODES = ("global_cached", "global_cached_meta", "swa_cached", "swa_cached_meta")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--burst", type=int, default=25)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _tracked_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _time_ms(runtime, fn, burst: int) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            fn()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return float(runtime.event_elapsed_time_ms(start, stop)) / burst
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _copy_f32(runtime, buffer, shape: tuple[int, ...]) -> np.ndarray:
    host = np.empty(shape, dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(host),
        buffer,
        host.nbytes,
        runtime=runtime,
    )
    return host


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.samples <= 0 or args.warmups < 0 or args.burst <= 0:
        raise ValueError("samples/burst must be positive and warmups non-negative")
    tracked = _tracked_status()
    if tracked and not args.allow_dirty:
        raise RuntimeError("tracked worktree must be clean; use --allow-dirty for a candidate")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=args.require_cached_build,
    )
    config = SimpleNamespace(
        block_count=2,
        layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(GLOBAL_HEADS, SWA_HEADS),
        head_count_kv=KV_HEADS,
        key_length=HEAD_DIM,
        value_length=HEAD_DIM,
        sliding_window=CONTEXT,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=CONTEXT,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(20260726)
    keys = rng.normal(0.0, 0.12, size=(CONTEXT, KV_HEADS, HEAD_DIM)).astype(
        np.float32
    )
    values = rng.normal(0.0, 0.12, size=(CONTEXT, KV_HEADS, HEAD_DIM)).astype(
        np.float32
    )
    query_global = rng.normal(
        0.0, 0.12, size=(CONTEXT, GLOBAL_HEADS, HEAD_DIM)
    ).astype(np.float32)
    query_swa = rng.normal(
        0.0, 0.12, size=(CONTEXT, SWA_HEADS, HEAD_DIM)
    ).astype(np.float32)
    allocations = []
    before = memory_stats()
    results: list[dict[str, object]] = []
    try:
        key_device = malloc(keys.nbytes, runtime=runtime)
        value_device = malloc(values.nbytes, runtime=runtime)
        global_query_device = malloc(query_global.nbytes, runtime=runtime)
        swa_query_device = malloc(query_swa.nbytes, runtime=runtime)
        global_baseline_out = malloc(
            ROWS * GLOBAL_HEADS * HEAD_DIM * 4, runtime=runtime
        )
        global_candidate_out = malloc(
            ROWS * GLOBAL_HEADS * HEAD_DIM * 4, runtime=runtime
        )
        swa_baseline_out = malloc(ROWS * SWA_HEADS * HEAD_DIM * 4, runtime=runtime)
        swa_candidate_out = malloc(ROWS * SWA_HEADS * HEAD_DIM * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                global_query_device,
                swa_query_device,
                global_baseline_out,
                global_candidate_out,
                swa_baseline_out,
                swa_candidate_out,
            )
        )
        for device, host in (
            (key_device, keys),
            (value_device, values),
            (global_query_device, query_global),
            (swa_query_device, query_swa),
        ):
            copy_host_to_device(
                device,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )

        scale = HEAD_DIM**-0.5
        kv_row_bytes = KV_HEADS * HEAD_DIM * 4
        global_q_row_bytes = GLOBAL_HEADS * HEAD_DIM * 4
        swa_q_row_bytes = SWA_HEADS * HEAD_DIM * 4
        for start_position in STARTS:
            positions = tuple(range(start_position, start_position + ROWS))
            cache.prepare_rows(positions)
            kv_offset = start_position * kv_row_bytes
            for layer_id in range(2):
                cache.append_rows(
                    layer_id,
                    key_device.ptr + kv_offset,
                    value_device.ptr + kv_offset,
                    ROWS,
                    library=library,
                )
            global_layer = cache.layer(0)
            swa_layer = cache.layer(1)
            global_q_ptr = (
                global_query_device.ptr + start_position * global_q_row_bytes
            )
            swa_q_ptr = swa_query_device.ptr + start_position * swa_q_row_bytes

            def global_cached() -> None:
                laguna_global_attention_prefill_qrow4_cached_online_bf16_spans(
                    global_q_ptr,
                    key_device.ptr + kv_offset,
                    value_device.ptr + kv_offset,
                    global_layer.key_cache.ptr,
                    global_layer.value_cache.ptr,
                    global_baseline_out.ptr,
                    global_layer.spans,
                    ROWS,
                    global_layer.capacity,
                    GLOBAL_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    scale,
                    library=library,
                    runtime=runtime,
                )

            def global_cached_meta() -> None:
                laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans(
                    global_q_ptr,
                    key_device.ptr + kv_offset,
                    value_device.ptr + kv_offset,
                    global_layer.key_cache.ptr,
                    global_layer.value_cache.ptr,
                    global_candidate_out.ptr,
                    global_layer.spans,
                    ROWS,
                    global_layer.capacity,
                    GLOBAL_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    scale,
                    library=library,
                    runtime=runtime,
                )

            def swa_cached() -> None:
                laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans(
                    swa_q_ptr,
                    key_device.ptr + kv_offset,
                    value_device.ptr + kv_offset,
                    swa_layer.key_cache.ptr,
                    swa_layer.value_cache.ptr,
                    swa_baseline_out.ptr,
                    swa_layer.spans,
                    ROWS,
                    SWA_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    scale,
                    sliding_window=CONTEXT,
                    start_position=start_position,
                    library=library,
                    runtime=runtime,
                )

            def swa_cached_meta() -> None:
                laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans(
                    swa_q_ptr,
                    key_device.ptr + kv_offset,
                    value_device.ptr + kv_offset,
                    swa_layer.key_cache.ptr,
                    swa_layer.value_cache.ptr,
                    swa_candidate_out.ptr,
                    swa_layer.spans,
                    ROWS,
                    SWA_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    scale,
                    sliding_window=CONTEXT,
                    start_position=start_position,
                    library=library,
                    runtime=runtime,
                )

            functions = {
                "global_cached": global_cached,
                "global_cached_meta": global_cached_meta,
                "swa_cached": swa_cached,
                "swa_cached_meta": swa_cached_meta,
            }
            for _ in range(args.warmups):
                for mode in MODES:
                    functions[mode]()
            runtime.device_synchronize()

            samples: dict[str, list[float]] = {mode: [] for mode in MODES}
            for sample in range(args.samples):
                offset = sample % len(MODES)
                order = MODES[offset:] + MODES[:offset]
                for mode in order:
                    samples[mode].append(
                        _time_ms(runtime, functions[mode], args.burst)
                    )

            for mode in MODES:
                functions[mode]()
            runtime.device_synchronize()
            expected_global = _copy_f32(
                runtime,
                global_baseline_out,
                (ROWS, GLOBAL_HEADS, HEAD_DIM),
            )
            actual_global = _copy_f32(
                runtime,
                global_candidate_out,
                (ROWS, GLOBAL_HEADS, HEAD_DIM),
            )
            expected_swa = _copy_f32(
                runtime,
                swa_baseline_out,
                (ROWS, SWA_HEADS, HEAD_DIM),
            )
            actual_swa = _copy_f32(
                runtime,
                swa_candidate_out,
                (ROWS, SWA_HEADS, HEAD_DIM),
            )
            medians = {
                mode: statistics.median(values_ms)
                for mode, values_ms in samples.items()
            }
            results.append(
                {
                    "start_position": start_position,
                    "samples_ms": samples,
                    "medians_ms": medians,
                    "global_speedup": medians["global_cached"]
                    / medians["global_cached_meta"],
                    "swa_speedup": medians["swa_cached"]
                    / medians["swa_cached_meta"],
                    "global_f32_bit_mismatches": int(
                        np.count_nonzero(
                            expected_global.view(np.uint32)
                            != actual_global.view(np.uint32)
                        )
                    ),
                    "swa_f32_bit_mismatches": int(
                        np.count_nonzero(
                            expected_swa.view(np.uint32)
                            != actual_swa.view(np.uint32)
                        )
                    ),
                }
            )
            cache.commit_rows()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
    after = memory_stats()
    total_baseline = sum(
        float(row["medians_ms"]["global_cached"])
        + 3.0 * float(row["medians_ms"]["swa_cached"])
        for row in results
    )
    total_candidate = sum(
        float(row["medians_ms"]["global_cached_meta"])
        + 3.0 * float(row["medians_ms"]["swa_cached_meta"])
        for row in results
    )
    return {
        "schema_version": 1,
        "kind": "laguna_attention_cached_meta_leaf",
        "repo": {
            "revision": _revision(),
            "tracked_status": tracked,
        },
        "protocol": {
            "starts": STARTS,
            "rows": ROWS,
            "samples": args.samples,
            "warmups": args.warmups,
            "burst": args.burst,
            "order": "four-mode counter-rotation",
        },
        "results": results,
        "weighted_pp512": {
            "formula": "sum(global + 3*swa) across four M128 positions",
            "baseline_ms": total_baseline,
            "candidate_ms": total_candidate,
            "speedup": total_baseline / total_candidate,
        },
        "correctness": {
            "pass": all(
                int(row["global_f32_bit_mismatches"]) == 0
                and int(row["swa_f32_bit_mismatches"]) == 0
                for row in results
            ),
            "tracked_current_bytes_before": before["current_allocated_bytes"],
            "tracked_current_bytes_after": after["current_allocated_bytes"],
            "tracked_active_allocations_after": after["active_allocations"],
        },
    }


def main() -> None:
    args = _parse_args()
    artifact = run(args)
    encoded = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")
    print("sha256=" + hashlib.sha256(encoded.encode()).hexdigest())


if __name__ == "__main__":
    main()
