#!/usr/bin/env python3
"""Measure the fixed M128/512-window tensorized Laguna SWA route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
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
)
from hipengine.loading.laguna_gguf import SLIDING_ATTENTION
from hipengine.runtime.laguna_attention_hipblaslt import (
    LagunaSwaAttentionHipblasLt,
)
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache


ROWS = 128
WINDOW = 512
Q_HEADS = 72
KV_HEADS = 8
HEAD_DIM = 128
MODES = ("qrow2_online", "tensorized_union")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--screen-algorithms", action="store_true")
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


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.samples <= 0 or args.warmups < 0:
        raise ValueError("samples must be positive and warmups non-negative")
    tracked = _tracked_status()
    if tracked and not args.allow_dirty:
        raise RuntimeError("tracked worktree must be clean; use --allow-dirty")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    before = memory_stats()
    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=args.require_cached_build,
    )
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(Q_HEADS,),
        head_count_kv=KV_HEADS,
        key_length=HEAD_DIM,
        value_length=HEAD_DIM,
        sliding_window=WINDOW,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=WINDOW + ROWS,
        backend="hip_gfx1151",
        runtime=runtime,
        swa_prefill_variant="swa_context_rows_qrow2_online_spans",
    )
    route = LagunaSwaAttentionHipblasLt(runtime=runtime)
    scratch_nbytes = route.scratch_nbytes
    allocations = []
    rng = np.random.default_rng(20260728)
    qk_index: int | None = None
    pv_index: int | None = None
    try:
        total_rows = WINDOW + ROWS
        keys = rng.normal(
            0.0,
            0.12,
            size=(total_rows, KV_HEADS, HEAD_DIM),
        ).astype(np.float32)
        values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
        queries = rng.normal(
            0.0,
            0.12,
            size=(ROWS, Q_HEADS, HEAD_DIM),
        ).astype(np.float32)
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        baseline_out = malloc(queries.nbytes, runtime=runtime)
        candidate_out = malloc(queries.nbytes, runtime=runtime)
        allocations.extend(
            (key_rows, value_rows, query_rows, baseline_out, candidate_out)
        )
        for buffer, host in (
            (key_rows, keys),
            (value_rows, values),
            (query_rows, queries),
        ):
            copy_host_to_device(
                buffer,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )
        cache.prepare_rows(tuple(range(WINDOW)))
        cache.append_rows(
            0,
            key_rows.ptr,
            value_rows.ptr,
            WINDOW,
            library=library,
        )
        cache.commit_rows()
        cache.prepare_rows(tuple(range(WINDOW, total_rows)))
        layer = cache.layer(0)
        row_nbytes = KV_HEADS * HEAD_DIM * np.dtype(np.float32).itemsize
        current_key_ptr = key_rows.ptr + WINDOW * row_nbytes
        current_value_ptr = value_rows.ptr + WINDOW * row_nbytes
        scale = HEAD_DIM**-0.5

        def baseline() -> None:
            cache.attend_prefill(
                0,
                query_rows.ptr,
                current_key_ptr,
                current_value_ptr,
                baseline_out.ptr,
                ROWS,
                library=library,
            )

        def candidate() -> None:
            route.launch(
                query_rows.ptr,
                current_key_ptr,
                current_value_ptr,
                layer.key_cache.ptr,
                layer.value_cache.ptr,
                candidate_out.ptr,
                layer.spans,
                rows=ROWS,
                start_position=WINDOW,
                num_q_heads=Q_HEADS,
                num_kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                sliding_window=WINDOW,
                scale=scale,
                kv_library=library,
                qk_algorithm_index=qk_index,
                pv_algorithm_index=pv_index,
            )

        qk_count, pv_count = route.algorithm_counts(
            num_q_heads=Q_HEADS,
            context=WINDOW + ROWS - 1,
        )
        qk_screen: list[dict[str, object]] = []
        pv_screen: list[dict[str, object]] = []
        if args.screen_algorithms:
            for index in range(qk_count):
                qk_index = index
                pv_index = 3
                qk_screen.append({"index": index, "ms": _time_ms(runtime, candidate)})
            qk_index = int(min(qk_screen, key=lambda row: float(row["ms"]))["index"])
            for index in range(pv_count):
                pv_index = index
                pv_screen.append({"index": index, "ms": _time_ms(runtime, candidate)})
            pv_index = int(min(pv_screen, key=lambda row: float(row["ms"]))["index"])

        functions = {
            "qrow2_online": baseline,
            "tensorized_union": candidate,
        }
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
        expected = np.empty_like(queries)
        actual = np.empty_like(queries)
        for host, buffer in (
            (expected, baseline_out),
            (actual, candidate_out),
        ):
            copy_device_to_host(
                host_array_ptr(host),
                buffer,
                host.nbytes,
                runtime=runtime,
            )
        delta = np.abs(expected - actual)
        medians = {
            mode: statistics.median(values)
            for mode, values in samples.items()
        }
        cache.discard_rows()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        route.close()
        cache.free()
    after = memory_stats()
    return {
        "schema_version": 1,
        "kind": "laguna_swa_attention_hipblaslt_ceiling",
        "status": "diagnostic",
        "repo_revision": _revision(),
        "tracked_dirty": tracked,
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "shape": {
            "rows": ROWS,
            "window": WINDOW,
            "union_context": WINDOW + ROWS - 1,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
        },
        "samples_ms": samples,
        "medians_ms": medians,
        "candidate_speedup": medians["qrow2_online"] / medians["tensorized_union"],
        "algorithm_counts": {"qk": qk_count, "pv": pv_count},
        "selected_algorithms": {"qk": qk_index, "pv": pv_index},
        "qk_algorithm_screen": qk_screen,
        "pv_algorithm_screen": pv_screen,
        "scratch_nbytes": scratch_nbytes,
        "max_abs": float(np.max(delta)),
        "mean_abs": float(np.mean(delta)),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "finite": bool(np.isfinite(actual).all()),
        "lifecycle": {
            "current_bytes_before": before["current_allocated_bytes"],
            "current_bytes_after": after["current_allocated_bytes"],
            "active_before": before["active_allocations"],
            "active_after": after["active_allocations"],
        },
    }


def main() -> None:
    args = _parse_args()
    artifact = run(args)
    payload = json.dumps(artifact, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
