#!/usr/bin/env python3
"""Gate exact natural-shape Laguna global-attention reduction."""

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
)
from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
    build_laguna_kv_attention,
    laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans,
)
from hipengine.loading.laguna_gguf import FULL_ATTENTION
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache


CAPACITY = 4096
Q_HEADS = 48
KV_HEADS = 8
HEAD_DIM = 128
LIVE_COUNTS = (513, 576, 639)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--burst", type=int, default=50)
    parser.add_argument(
        "--candidate",
        choices=("fixedshape", "fused-gqa1"),
        default="fixedshape",
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


def _download(runtime, buffer, dtype) -> np.ndarray:
    host = np.empty(Q_HEADS * HEAD_DIM, dtype=dtype)
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
        raise RuntimeError("tracked worktree must be clean; use --allow-dirty")
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
        context_length=CAPACITY,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    max_live = max(LIVE_COUNTS)
    rng = np.random.default_rng(20260728)
    keys = rng.normal(
        0.0, 0.12, size=(max_live, KV_HEADS, HEAD_DIM)
    ).astype(np.float32)
    values = rng.normal(
        0.0, 0.12, size=(max_live, KV_HEADS, HEAD_DIM)
    ).astype(np.float32)
    query = rng.normal(0.0, 0.12, size=(Q_HEADS, HEAD_DIM)).astype(np.float32)
    gate = rng.normal(0.0, 0.4, size=Q_HEADS).astype(np.float32)
    allocations = []
    try:
        key_device = malloc(keys.nbytes, runtime=runtime)
        value_device = malloc(values.nbytes, runtime=runtime)
        query_device = malloc(query.nbytes, runtime=runtime)
        gate_device = malloc(gate.nbytes, runtime=runtime)
        control_context = malloc(query.nbytes, runtime=runtime)
        candidate_context = malloc(query.nbytes, runtime=runtime)
        control_gated = malloc(query.size * 2, runtime=runtime)
        candidate_gated = malloc(query.size * 2, runtime=runtime)
        score_scratch = malloc(Q_HEADS * CAPACITY * 4, runtime=runtime)
        physical_scratch = malloc(Q_HEADS * CAPACITY * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                gate_device,
                control_context,
                candidate_context,
                control_gated,
                candidate_gated,
                score_scratch,
                physical_scratch,
            )
        )
        for device, host in (
            (key_device, keys),
            (value_device, values),
            (query_device, query),
            (gate_device, gate),
        ):
            copy_host_to_device(
                device,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )
        cache.prepare_rows(tuple(range(max_live)))
        cache.append_rows(
            0,
            key_device.ptr,
            value_device.ptr,
            max_live,
            library=library,
        )
        cache.commit_rows()
        cache.prepare_position(max_live)
        state = cache.layer(0)
        common = (
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
        )
        results = []
        for live_count in LIVE_COUNTS:
            tail = (
                score_scratch.ptr,
                physical_scratch.ptr,
                state.spans,
                live_count,
                CAPACITY,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                HEAD_DIM**-0.5,
            )
            control_kernel = (
                laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans
                if args.candidate.startswith("fused-")
                else laguna_global_attention_decode_split_exact_gated_bf16_spans
            )
            candidate_kernel = {
                "fixedshape": laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans,
                "fused-gqa1": laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans,
            }[args.candidate]

            def control() -> None:
                control_kernel(
                    *common,
                    control_context.ptr,
                    gate_device.ptr,
                    control_gated.ptr,
                    *tail,
                    library=library,
                    runtime=runtime,
                )

            def candidate() -> None:
                candidate_kernel(
                    *common,
                    candidate_context.ptr,
                    gate_device.ptr,
                    candidate_gated.ptr,
                    *tail,
                    library=library,
                    runtime=runtime,
                )

            control()
            candidate()
            runtime.device_synchronize()
            control_context_host = _download(runtime, control_context, np.float32)
            candidate_context_host = _download(
                runtime, candidate_context, np.float32
            )
            control_gated_host = _download(runtime, control_gated, np.uint16)
            candidate_gated_host = _download(
                runtime, candidate_gated, np.uint16
            )
            context_exact = np.array_equal(
                control_context_host, candidate_context_host
            )
            gated_exact = np.array_equal(control_gated_host, candidate_gated_host)
            if not context_exact or not gated_exact:
                raise AssertionError(
                    f"{args.candidate} is not byte-exact at {live_count=}"
                )

            for _ in range(args.warmups):
                control()
                candidate()
            runtime.device_synchronize()
            control_ms = []
            candidate_ms = []
            for sample in range(args.samples):
                order = (
                    (("control", control), ("candidate", candidate))
                    if sample % 2 == 0
                    else (("candidate", candidate), ("control", control))
                )
                row = {}
                for name, fn in order:
                    row[name] = _time_ms(runtime, fn, args.burst)
                control_ms.append(row["control"])
                candidate_ms.append(row["candidate"])
            control_median = statistics.median(control_ms)
            candidate_median = statistics.median(candidate_ms)
            results.append(
                {
                    "live_count": live_count,
                    "correctness": {
                        "context_f32_byte_exact": context_exact,
                        "gated_bf16_byte_exact": gated_exact,
                        "context_sha256": hashlib.sha256(
                            candidate_context_host.tobytes()
                        ).hexdigest(),
                        "gated_sha256": hashlib.sha256(
                            candidate_gated_host.tobytes()
                        ).hexdigest(),
                    },
                    "control": {
                        "samples_ms": control_ms,
                        "median_ms": control_median,
                    },
                    "candidate": {
                        "samples_ms": candidate_ms,
                        "median_ms": candidate_median,
                        "latency_change_percent": (
                            candidate_median / control_median - 1.0
                        )
                        * 100.0,
                    },
                }
            )
        return {
            "schema": 1,
            "kind": "hipengine_laguna_global_fixedshape_reduce_leaf",
            "candidate_kind": args.candidate,
            "status": "directional_candidate",
            "performance_claim": False,
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "tracked_changes": tracked,
            "hardware": {
                "device": "AMD Radeon 8060S Graphics",
                "arch": os.environ.get("HIPENGINE_HIP_ARCH"),
            },
            "shape": {
                "capacity": CAPACITY,
                "query_heads": Q_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "live_counts": LIVE_COUNTS,
            },
            "protocol": {
                "samples": args.samples,
                "warmups": args.warmups,
                "burst": args.burst,
            },
            "rows": results,
        }
    finally:
        cache.free()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
