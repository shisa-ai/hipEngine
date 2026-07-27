#!/usr/bin/env python3
"""Measure an inclusive packed-F32 hipBLASLt ceiling for Laguna global attention."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
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
    laguna_dense_initial_cache_block_bf16_to_f32_spans,
    laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans,
    laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans,
)
from hipengine.loading.laguna_gguf import FULL_ATTENTION
from hipengine.runtime.laguna_attention_hipblaslt import LagunaAttentionHipblasLt
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache


CONTEXTS = (512, 4096, 16384, 65536)
CONTEXT_128K = 131072
ROWS = 128
FILL_ROWS = 512
KV_HEADS = 8
Q_HEADS = 48
HEAD_DIM = 128
MODES = ("qrow6", "packed_f32_hipblaslt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--rows",
        type=int,
        choices=(128, 256, 512, 1024, 2048),
        default=128,
    )
    parser.add_argument("--screen-algorithms", action="store_true")
    parser.add_argument("--block-context", type=int)
    parser.add_argument("--dense-contiguous-cache", action="store_true")
    parser.add_argument("--only-128k", action="store_true")
    parser.add_argument(
        "--contexts",
        help="comma-separated context lengths; overrides the default sweep",
    )
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


def _time_ms(runtime, fn) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        fn()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return float(runtime.event_elapsed_time_ms(start, stop))
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _copy_f32(runtime, buffer, rows: int) -> np.ndarray:
    host = np.empty((rows, Q_HEADS, HEAD_DIM), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(host),
        buffer,
        host.nbytes,
        runtime=runtime,
    )
    return host


def _append_rows(
    *,
    cache,
    library,
    runtime,
    rng: np.random.Generator,
    key_device,
    value_device,
    start: int,
    rows: int,
) -> None:
    keys = rng.normal(0.0, 0.12, size=(rows, KV_HEADS, HEAD_DIM)).astype(
        np.float32
    )
    values = rng.normal(0.0, 0.12, size=(rows, KV_HEADS, HEAD_DIM)).astype(
        np.float32
    )
    copy_host_to_device(
        key_device,
        host_array_ptr(keys),
        keys.nbytes,
        runtime=runtime,
    )
    copy_host_to_device(
        value_device,
        host_array_ptr(values),
        values.nbytes,
        runtime=runtime,
    )
    cache.prepare_rows(tuple(range(start, start + rows)))
    cache.append_rows(
        0,
        key_device.ptr,
        value_device.ptr,
        rows,
        library=library,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.samples <= 0 or args.warmups < 0:
        raise ValueError("samples must be positive and warmups non-negative")
    if args.only_128k and args.contexts:
        raise ValueError("--only-128k and --contexts are mutually exclusive")
    query_rows = int(args.rows)
    if args.block_context is not None and (
        args.block_context < ROWS
        or args.block_context % ROWS != 0
    ):
        raise ValueError(
            "block-context must be an M128 multiple"
        )
    tracked = _tracked_status()
    if tracked and not args.allow_dirty:
        raise RuntimeError("tracked worktree must be clean; use --allow-dirty")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )
    if args.contexts:
        contexts = tuple(int(value) for value in args.contexts.split(","))
        if (
            not contexts
            or any(context < query_rows for context in contexts)
            or tuple(sorted(set(contexts))) != contexts
        ):
            raise ValueError(
                "contexts must be unique ascending integers >= rows"
            )
    else:
        contexts = (CONTEXT_128K,) if args.only_128k else CONTEXTS

    before = memory_stats()
    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=args.require_cached_build,
    )
    config = SimpleNamespace(
        block_count=1,
        layer_types=(FULL_ATTENTION,),
        head_counts=(Q_HEADS,),
        head_count_kv=KV_HEADS,
        key_length=HEAD_DIM,
        value_length=HEAD_DIM,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=contexts[-1],
        backend="hip_gfx1151",
        runtime=runtime,
    )
    route = LagunaAttentionHipblasLt(
        runtime=runtime,
        packed_queries=True,
        wave_rows_softmax=True,
        max_context=max(contexts[-1], args.block_context or 0),
        max_q_heads=Q_HEADS,
        block_context=args.block_context,
        query_rows=query_rows,
    )
    scratch_nbytes = route.scratch_nbytes
    allocations = []
    results: list[dict[str, object]] = []
    rng = np.random.default_rng(20260727)
    try:
        kv_bytes = max(FILL_ROWS, query_rows) * KV_HEADS * HEAD_DIM * 4
        output_bytes = query_rows * Q_HEADS * HEAD_DIM * 4
        key_device = malloc(kv_bytes, runtime=runtime)
        value_device = malloc(kv_bytes, runtime=runtime)
        query_device = malloc(output_bytes, runtime=runtime)
        qrow6_out = malloc(output_bytes, runtime=runtime)
        blas_out = malloc(output_bytes, runtime=runtime)
        allocations.extend(
            (key_device, value_device, query_device, qrow6_out, blas_out)
        )

        for context in contexts:
            final_start = context - query_rows
            while cache.position + 1 < final_start:
                start = cache.position + 1
                rows = min(FILL_ROWS, final_start - start)
                _append_rows(
                    cache=cache,
                    library=library,
                    runtime=runtime,
                    rng=rng,
                    key_device=key_device,
                    value_device=value_device,
                    start=start,
                    rows=rows,
                )
                cache.commit_rows()
            for row_offset in range(0, query_rows, FILL_ROWS):
                current_rows = min(FILL_ROWS, query_rows - row_offset)
                _append_rows(
                    cache=cache,
                    library=library,
                    runtime=runtime,
                    rng=rng,
                    key_device=key_device,
                    value_device=value_device,
                    start=final_start + row_offset,
                    rows=current_rows,
                )
                cache.commit_rows()
            row_position = ctypes.c_int64(final_start)
            runtime.memcpy(
                cache.layer(0).spans.row_positions.ptr,
                ctypes.addressof(row_position),
                ctypes.sizeof(row_position),
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            query = rng.normal(
                0.0, 0.12, size=(query_rows, Q_HEADS, HEAD_DIM)
            ).astype(np.float32)
            copy_host_to_device(
                query_device,
                host_array_ptr(query),
                query.nbytes,
                runtime=runtime,
            )
            layer = cache.layer(0)
            scale = HEAD_DIM**-0.5
            qk_algorithm_index: int | None = None
            pv_algorithm_index: int | None = None

            def qrow6() -> None:
                laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans(
                    query_device.ptr,
                    key_device.ptr,
                    value_device.ptr,
                    layer.key_cache.ptr,
                    layer.value_cache.ptr,
                    qrow6_out.ptr,
                    layer.spans,
                    query_rows,
                    layer.capacity,
                    Q_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    scale,
                    library=library,
                    runtime=runtime,
                )

            def packed_f32_hipblaslt() -> None:
                route.launch(
                    query_device.ptr,
                    layer.key_cache.ptr,
                    layer.value_cache.ptr,
                    blas_out.ptr,
                    layer.spans,
                    rows=query_rows,
                    start_position=final_start,
                    num_q_heads=Q_HEADS,
                    num_kv_heads=KV_HEADS,
                    head_dim=HEAD_DIM,
                    scale=scale,
                    kv_library=library,
                    qk_algorithm_index=qk_algorithm_index,
                    pv_algorithm_index=pv_algorithm_index,
                    dense_contiguous_cache=args.dense_contiguous_cache,
                )

            functions = {
                "qrow6": qrow6,
                "packed_f32_hipblaslt": packed_f32_hipblaslt,
            }
            cache_widen_samples: dict[str, list[float]] = {}
            cache_widen_medians: dict[str, float] = {}
            if args.block_context is not None:
                tile_count = min(int(args.block_context), context)

                def span_cache_widen() -> None:
                    laguna_dense_initial_cache_block_bf16_to_f32_spans(
                        layer.key_cache.ptr,
                        layer.value_cache.ptr,
                        route.key_f32.ptr,
                        route.value_f32.ptr,
                        layer.spans,
                        0,
                        tile_count,
                        context,
                        KV_HEADS,
                        HEAD_DIM,
                        library=library,
                        runtime=runtime,
                    )

                def contiguous_cache_widen() -> None:
                    laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans(
                        layer.key_cache.ptr,
                        layer.value_cache.ptr,
                        route.key_f32.ptr,
                        route.value_f32.ptr,
                        layer.spans,
                        0,
                        tile_count,
                        context,
                        KV_HEADS,
                        HEAD_DIM,
                        library=library,
                        runtime=runtime,
                    )

                cache_functions = {
                    "spans": span_cache_widen,
                    "contiguous": contiguous_cache_widen,
                }
                for _ in range(args.warmups):
                    for cache_function in cache_functions.values():
                        cache_function()
                runtime.device_synchronize()
                cache_widen_samples = {
                    mode: [] for mode in cache_functions
                }
                for sample in range(args.samples):
                    order = (
                        tuple(cache_functions)
                        if sample % 2 == 0
                        else tuple(reversed(cache_functions))
                    )
                    for mode in order:
                        cache_widen_samples[mode].append(
                            _time_ms(runtime, cache_functions[mode])
                        )
                cache_widen_medians = {
                    mode: statistics.median(values)
                    for mode, values in cache_widen_samples.items()
                }
            qk_screen: list[dict[str, object]] = []
            pv_screen: list[dict[str, object]] = []
            algorithm_context = args.block_context or context
            qk_count, pv_count = route.algorithm_counts(
                num_q_heads=Q_HEADS,
                context=algorithm_context,
            )
            if args.screen_algorithms and context > 512:
                for index in range(qk_count):
                    qk_algorithm_index = index
                    pv_algorithm_index = 0
                    try:
                        elapsed = _time_ms(runtime, packed_f32_hipblaslt)
                    except Exception as error:
                        qk_screen.append(
                            {
                                "index": index,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    else:
                        qk_screen.append({"index": index, "ms": elapsed})
                qk_algorithm_index = int(
                    min(
                        (row for row in qk_screen if "ms" in row),
                        key=lambda row: float(row["ms"]),
                    )["index"]
                )
                for index in range(pv_count):
                    pv_algorithm_index = index
                    try:
                        elapsed = _time_ms(runtime, packed_f32_hipblaslt)
                    except Exception as error:
                        pv_screen.append(
                            {
                                "index": index,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    else:
                        pv_screen.append({"index": index, "ms": elapsed})
                pv_algorithm_index = int(
                    min(
                        (row for row in pv_screen if "ms" in row),
                        key=lambda row: float(row["ms"]),
                    )["index"]
                )
            for _ in range(args.warmups):
                for mode in MODES:
                    functions[mode]()
            runtime.device_synchronize()
            samples = {mode: [] for mode in MODES}
            for sample in range(args.samples):
                order = MODES if sample % 2 == 0 else tuple(reversed(MODES))
                for mode in order:
                    samples[mode].append(_time_ms(runtime, functions[mode]))

            for mode in MODES:
                functions[mode]()
            runtime.device_synchronize()
            expected = _copy_f32(runtime, qrow6_out, query_rows)
            actual = _copy_f32(runtime, blas_out, query_rows)
            delta = np.abs(expected - actual)
            medians = {
                mode: statistics.median(values)
                for mode, values in samples.items()
            }
            results.append(
                {
                    "context": context,
                    "samples_ms": samples,
                    "medians_ms": medians,
                    "candidate_speedup": (
                        medians["qrow6"] / medians["packed_f32_hipblaslt"]
                    ),
                    "algorithm_counts": {
                        "qk": qk_count,
                        "pv": pv_count,
                    },
                    "selected_algorithms": {
                        "qk": qk_algorithm_index,
                        "pv": pv_algorithm_index,
                    },
                    "cache_widen_samples_ms": cache_widen_samples,
                    "cache_widen_medians_ms": cache_widen_medians,
                    "qk_algorithm_screen": qk_screen,
                    "pv_algorithm_screen": pv_screen,
                    "max_abs": float(np.max(delta)),
                    "mean_abs": float(np.mean(delta)),
                    "rmse": float(np.sqrt(np.mean(np.square(delta)))),
                    "finite": bool(np.isfinite(actual).all()),
                }
            )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        route.close()
        cache.free()
    after = memory_stats()
    return {
        "schema_version": 1,
        "kind": "laguna_attention_long_context_hipblaslt_ceiling",
        "repo": {
            "revision": _revision(),
            "tracked_status": tracked,
        },
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "protocol": {
            "contexts": contexts,
            "rows": query_rows,
            "samples": args.samples,
            "warmups": args.warmups,
            "order": "two-mode alternating counter-rotation",
            "cache_capacity": contexts[-1],
            "score_scratch": (
                f"F32 [48,{query_rows},block_context]"
                if args.block_context is not None
                else f"F32 [48,{query_rows},max_context]"
            ),
            "block_context": args.block_context,
            "dense_contiguous_cache": args.dense_contiguous_cache,
            "inclusive_candidate": "BF16 cache widen + query transpose + F32 QK + wave-row softmax + F32 PV + output transpose",
        },
        "scratch_nbytes": scratch_nbytes,
        "results": results,
        "correctness": {
            "finite": all(bool(row["finite"]) for row in results),
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
