#!/usr/bin/env python3
"""Screen exact Laguna grouped weighted-reduction plus shared-add fusion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    build_gguf_ops,
    gguf_bf16_add,
)
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    build_paro_combine,
    weighted_lanes_sum_out_bf16_f32w,
    weighted_lanes_sum_shared_add_out_bf16_f32w,
)
from scripts.laguna_target_ar_bench import _compiler_version, _repo_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (32, 55, 64, 122, 128)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-laguna-grouped-combine-micro.json"
)


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not rows or any(item <= 0 for item in rows):
        raise argparse.ArgumentTypeError("rows must be distinct positive integers")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--rows", type=_parse_rows, default=DEFAULT_ROWS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--features", type=int, default=3072)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    return ((u32 + np.uint32(0x7FFF) + lsb) >> 16).astype(np.uint16)


def _device_copy(values: np.ndarray):
    buffer = malloc(values.nbytes)
    copy_host_to_device(buffer, host_array_ptr(values), values.nbytes)
    return buffer


def _summarize(
    rows: Sequence[int],
    samples: Mapping[int, Mapping[str, Sequence[float]]],
    gpu_samples: Mapping[int, Mapping[str, Sequence[float]]],
    exact: Mapping[int, bool],
) -> dict[str, Any]:
    parsed_rows = tuple(int(value) for value in rows)
    if not parsed_rows or tuple(sorted(set(parsed_rows))) != parsed_rows:
        raise ValueError("micro rows must be sorted and distinct")
    shapes: dict[str, Any] = {}
    failed: list[str] = []
    baseline_gpu_total = 0.0
    candidate_gpu_total = 0.0
    for value in parsed_rows:
        baseline = [float(item) for item in samples[value]["baseline"]]
        candidate = [float(item) for item in samples[value]["candidate"]]
        baseline_gpu = [float(item) for item in gpu_samples[value]["baseline"]]
        candidate_gpu = [float(item) for item in gpu_samples[value]["candidate"]]
        counts = {len(baseline), len(candidate), len(baseline_gpu), len(candidate_gpu)}
        if counts == {0} or len(counts) != 1:
            raise ValueError(f"rows={value} requires equal non-empty sample counts")
        baseline_median = statistics.median(baseline)
        candidate_median = statistics.median(candidate)
        baseline_gpu_median = statistics.median(baseline_gpu)
        candidate_gpu_median = statistics.median(candidate_gpu)
        gpu_speedup = baseline_gpu_median / candidate_gpu_median
        if not bool(exact[value]):
            failed.append(f"rows_{value}_not_bit_exact")
        if not math.isfinite(gpu_speedup) or gpu_speedup <= 1.0:
            failed.append(f"rows_{value}_gpu_span_not_faster")
        baseline_gpu_total += baseline_gpu_median
        candidate_gpu_total += candidate_gpu_median
        shapes[str(value)] = {
            "rows": value,
            "bit_exact": bool(exact[value]),
            "wall_us": {
                "baseline_samples": baseline,
                "candidate_samples": candidate,
                "baseline_median": baseline_median,
                "candidate_median": candidate_median,
                "speedup": baseline_median / candidate_median,
            },
            "gpu_span_us": {
                "baseline_samples": baseline_gpu,
                "candidate_samples": candidate_gpu,
                "baseline_median": baseline_gpu_median,
                "candidate_median": candidate_gpu_median,
                "speedup": gpu_speedup,
            },
        }
    effective_speedup = baseline_gpu_total / candidate_gpu_total
    if not math.isfinite(effective_speedup) or effective_speedup <= 1.0:
        failed.append("aggregate_gpu_span_not_faster")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "shapes": shapes,
        "aggregate_gpu_span_speedup": effective_speedup,
        "policy": "bit-exact and candidate median GPU span faster at every shape",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tuple(int(value) for value in args.rows)
    if rows != DEFAULT_ROWS:
        raise ValueError(f"retained grouped-combine micro requires rows {DEFAULT_ROWS}")
    if args.top_k != 10 or args.features != 3072:
        raise ValueError("retained grouped-combine micro requires top-k 10/features 3072")
    if args.iterations <= 0 or args.repetitions < 3 or args.warmups < 0:
        raise ValueError("iterations must be positive, repetitions >=3, warmups non-negative")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained grouped-combine micro requires a clean tracked worktree")

    compiler_version = _compiler_version(args.compiler_version_file)
    combine_library = build_paro_combine(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    ops_library = build_gguf_ops(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    samples = {value: {"baseline": [], "candidate": []} for value in rows}
    gpu_samples = {value: {"baseline": [], "candidate": []} for value in rows}
    exact: dict[int, bool] = {}
    rng = np.random.default_rng(args.seed)

    for tokens in rows:
        lane_rows = tokens * args.top_k
        values = _bf16_bits(
            rng.normal(size=(lane_rows, args.features)).astype(np.float32)
        )
        weights = rng.normal(size=lane_rows).astype(np.float32)
        sorted_lanes = rng.permutation(lane_rows).astype(np.int64)
        lane_to_row = np.zeros(lane_rows, dtype=np.int64)
        shared = _bf16_bits(
            rng.normal(size=(tokens, args.features)).astype(np.float32)
        )
        selected = np.empty((tokens, args.features), dtype=np.uint16)
        baseline = np.empty_like(selected)
        candidate = np.empty_like(selected)
        buffers = []
        try:
            values_d = _device_copy(values)
            weights_d = _device_copy(weights)
            sorted_lanes_d = _device_copy(sorted_lanes)
            lane_to_row_d = _device_copy(lane_to_row)
            shared_d = _device_copy(shared)
            selected_d = _device_copy(selected)
            baseline_d = _device_copy(baseline)
            candidate_d = _device_copy(candidate)
            buffers.extend(
                (
                    values_d,
                    weights_d,
                    sorted_lanes_d,
                    lane_to_row_d,
                    shared_d,
                    selected_d,
                    baseline_d,
                    candidate_d,
                )
            )

            def launch_baseline() -> None:
                weighted_lanes_sum_out_bf16_f32w(
                    values_d.ptr,
                    weights_d.ptr,
                    sorted_lanes_d.ptr,
                    lane_to_row_d.ptr,
                    selected_d.ptr,
                    tokens,
                    args.top_k,
                    args.features,
                    library=combine_library,
                    runtime=runtime,
                )
                gguf_bf16_add(
                    selected_d.ptr,
                    shared_d.ptr,
                    baseline_d.ptr,
                    tokens * args.features,
                    library=ops_library,
                    runtime=runtime,
                )

            def launch_candidate() -> None:
                weighted_lanes_sum_shared_add_out_bf16_f32w(
                    values_d.ptr,
                    weights_d.ptr,
                    sorted_lanes_d.ptr,
                    lane_to_row_d.ptr,
                    shared_d.ptr,
                    candidate_d.ptr,
                    tokens,
                    args.top_k,
                    args.features,
                    library=combine_library,
                    runtime=runtime,
                )

            launch_baseline()
            launch_candidate()
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(baseline), baseline_d, baseline.nbytes)
            copy_device_to_host(host_array_ptr(candidate), candidate_d, candidate.nbytes)
            exact[tokens] = bool(np.array_equal(baseline, candidate))
            for _ in range(args.warmups):
                launch_baseline()
                launch_candidate()
            runtime.device_synchronize()
            for repetition in range(args.repetitions):
                order = (
                    ("baseline", "candidate")
                    if repetition % 2 == 0
                    else ("candidate", "baseline")
                )
                for mode in order:
                    start_event = runtime.event_create()
                    stop_event = runtime.event_create()
                    try:
                        runtime.event_record(start_event)
                        started = time.perf_counter()
                        launch = launch_baseline if mode == "baseline" else launch_candidate
                        for _ in range(args.iterations):
                            launch()
                        runtime.event_record(stop_event)
                        runtime.event_synchronize(stop_event)
                        elapsed = time.perf_counter() - started
                        gpu_ms = runtime.event_elapsed_time_ms(start_event, stop_event)
                    finally:
                        runtime.event_destroy(stop_event)
                        runtime.event_destroy(start_event)
                    samples[tokens][mode].append(
                        elapsed * 1.0e6 / args.iterations
                    )
                    gpu_samples[tokens][mode].append(
                        gpu_ms * 1.0e3 / args.iterations
                    )
        finally:
            for buffer in reversed(buffers):
                free(buffer)

    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during grouped-combine micro")
    summary = _summarize(rows, samples, gpu_samples, exact)
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    if not recovered:
        summary["failed_checks"].append("tracked_ownership_not_recovered")
        summary["pass"] = False
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        quant="bf16_grouped_combine",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_grouped_combine_micro",
        timing_protocol="counterbalanced_event_and_wall_batches",
        warmups=args.warmups,
        repetitions=args.repetitions,
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_grouped_combine_micro",
        "status": "screen_passed" if summary["pass"] else "rejected",
        "pass": bool(summary["pass"]),
        "performance_claim": False,
        "performance_claim_scope": (
            "synthetic production-shape combine sub-window only; full-model wall and "
            "category gates are separate"
        ),
        "provenance": provenance,
        "repo": repo,
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "rows": list(rows),
            "top_k": args.top_k,
            "features": args.features,
            "iterations_per_sample": args.iterations,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "seed": args.seed,
            "baseline": "weighted_lanes_sum_out_bf16_f32w + gguf_bf16_add",
            "candidate": "weighted_lanes_sum_shared_add_out_bf16_f32w",
            "timed_order": "counterbalanced by repetition",
        },
        "summary": summary,
        "correctness": {
            "pass": all(exact.values()) and recovered,
            "bit_exact_by_shape": {str(key): value for key, value in exact.items()},
            "tracked_returned_to_baseline": recovered,
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "The baseline and candidate share identical sorted BF16 values, F32 weights, "
            "lane map scratch, shared BF16 output, and one default stream.",
            "HIP events report stream span; synchronized host wall is recorded separately.",
            "A pass is candidate-selection evidence, not a runtime promotion claim.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
