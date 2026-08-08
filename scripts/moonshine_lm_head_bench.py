#!/usr/bin/env python3
"""Qualify the exact gfx1151 Moonshine wave8 LM-head/top-1 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.cpu_reference.moonshine import moonshine_projection
from hipengine.kernels.hip_gfx1100.fused.moonshine_glue import (
    build_moonshine_glue,
    moonshine_argmax_fp16,
)
from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
    build_moonshine_projection,
    moonshine_f16_lm_head_projection_wave8,
    moonshine_f16_lm_head_projection_wave8_top1,
    moonshine_lm_head_partial_count,
)

HIDDEN_SIZE = 416
VOCAB_SIZE = 36_864
MODEL_ID = "shisa-ai/shisa-realtime-asr-0.92b"
MODEL_REVISION = "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--prebuild-only", action="store_true")
    parser.add_argument("--trace-smoke", action="store_true")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--burst", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0x1151)
    parser.add_argument("--minimum-improvement-percent", type=float, default=1.0)
    parser.add_argument("--profiler-summary-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-accept", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upload(
    host: np.ndarray,
    runtime: HipRuntime,
    allocations: list[DeviceBuffer],
) -> DeviceBuffer:
    contiguous = np.ascontiguousarray(host)
    device = malloc(contiguous.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(contiguous), runtime=runtime)
    return device


def _allocate(
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
    runtime: HipRuntime,
    allocations: list[DeviceBuffer],
) -> DeviceBuffer:
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(
    device: DeviceBuffer,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
    runtime: HipRuntime,
) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _measure(
    runtime: HipRuntime,
    launch: Callable[[], None],
    burst: int,
) -> tuple[float, float]:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        wall_start = time.perf_counter_ns()
        for _ in range(burst):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        wall_us = (time.perf_counter_ns() - wall_start) / 1000.0 / burst
        event_us = runtime.event_elapsed_time_ms(start, stop) * 1000.0 / burst
        return float(event_us), float(wall_us)
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    if not all(np.isfinite(samples)):
        raise ValueError("samples must be finite")
    ordered = sorted(float(value) for value in samples)
    p95_index = max(0, min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1))
    return {
        "samples_us": [float(value) for value in samples],
        "median_us": float(statistics.median(samples)),
        "mean_us": float(statistics.fmean(samples)),
        "p95_us": ordered[p95_index],
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "stdev_us": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
    }


def improvement_percent(baseline_us: float, candidate_us: float) -> float:
    if not np.isfinite(baseline_us) or not np.isfinite(candidate_us) or baseline_us <= 0:
        raise ValueError("timings must be finite and baseline must be positive")
    return 100.0 * (baseline_us - candidate_us) / baseline_us


def _load_profiler_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_attached",
            "expected_kernels": [
                "moonshine_f16_lm_head_projection_wave8_top1_kernel",
                "moonshine_lm_head_top1_reduce_kernel",
            ],
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("profiler summary must be a JSON object")
    names = payload.get("observed_kernels")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("profiler summary observed_kernels must be a string list")
    required = {
        "moonshine_f16_lm_head_projection_wave8_top1_kernel",
        "moonshine_lm_head_top1_reduce_kernel",
    }
    if not all(any(expected in name for name in names) for expected in required):
        raise ValueError("profiler summary is missing an expected candidate kernel")
    return dict(payload)


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repetitions <= 0 or args.burst <= 0:
        raise ValueError("warmups must be nonnegative; repetitions and burst must be positive")
    if (
        not np.isfinite(args.minimum_improvement_percent)
        or args.minimum_improvement_percent < 0
    ):
        raise ValueError("minimum improvement must be finite and nonnegative")
    if args.prebuild_only and args.trace_smoke:
        raise ValueError("prebuild-only and trace-smoke are mutually exclusive")

    compiler_version = args.compiler_version_file.read_text()
    build_args = {
        "compiler_version": compiler_version,
        "load": not args.prebuild_only,
        "require_cached": args.require_cached_build,
    }
    projection_library = build_moonshine_projection(**build_args)
    glue_library = build_moonshine_glue(**build_args)
    if args.prebuild_only:
        print(projection_library.output_path)
        print(glue_library.output_path)
        return 0

    runtime = get_hip_runtime()
    baseline_stats = memory_stats()
    rng = np.random.default_rng(args.seed)
    hidden = rng.normal(0.0, 0.05, size=(1, HIDDEN_SIZE)).astype(np.float16)
    weights = rng.normal(0.0, 0.04, size=(VOCAB_SIZE, HIDDEN_SIZE)).astype(np.float16)
    expected_logits = moonshine_projection(hidden, weights)
    expected_token = int(np.argmax(expected_logits[0]))
    partial_count = moonshine_lm_head_partial_count(VOCAB_SIZE)
    allocations: list[DeviceBuffer] = []
    correctness: dict[str, Any]
    timing: dict[str, Any] | None = None
    try:
        device_hidden = _upload(hidden, runtime, allocations)
        device_weights = _upload(weights, runtime, allocations)
        baseline_logits = _allocate((1, VOCAB_SIZE), np.float16, runtime, allocations)
        candidate_logits = _allocate((1, VOCAB_SIZE), np.float16, runtime, allocations)
        baseline_token = _allocate((1,), np.int64, runtime, allocations)
        candidate_token = _allocate((1,), np.int64, runtime, allocations)
        partial_values = _allocate((partial_count,), np.float16, runtime, allocations)
        partial_indices = _allocate((partial_count,), np.int64, runtime, allocations)

        def launch_baseline() -> None:
            moonshine_f16_lm_head_projection_wave8(
                device_hidden.ptr,
                device_weights.ptr,
                baseline_logits.ptr,
                1,
                HIDDEN_SIZE,
                VOCAB_SIZE,
                library=projection_library,
                runtime=runtime,
            )
            moonshine_argmax_fp16(
                baseline_logits.ptr,
                baseline_token.ptr,
                VOCAB_SIZE,
                library=glue_library,
                runtime=runtime,
            )

        def launch_candidate() -> None:
            moonshine_f16_lm_head_projection_wave8_top1(
                device_hidden.ptr,
                device_weights.ptr,
                candidate_logits.ptr,
                partial_values.ptr,
                partial_indices.ptr,
                candidate_token.ptr,
                1,
                HIDDEN_SIZE,
                VOCAB_SIZE,
                library=projection_library,
                runtime=runtime,
            )

        launch_baseline()
        launch_candidate()
        runtime.device_synchronize()
        baseline_logits_host = _download(
            baseline_logits, (1, VOCAB_SIZE), np.float16, runtime
        )
        candidate_logits_host = _download(
            candidate_logits, (1, VOCAB_SIZE), np.float16, runtime
        )
        baseline_token_host = _download(baseline_token, (1,), np.int64, runtime)
        candidate_token_host = _download(candidate_token, (1,), np.int64, runtime)
        maximum_absolute_error = float(
            np.max(
                np.abs(
                    candidate_logits_host.astype(np.float32)
                    - expected_logits.astype(np.float32)
                )
            )
        )
        correctness = {
            "full_fp16_logits_byte_exact_vs_fallback": bool(
                np.array_equal(candidate_logits_host, baseline_logits_host)
            ),
            "candidate_allclose_cpu_oracle": bool(
                np.allclose(candidate_logits_host, expected_logits, rtol=2e-3, atol=2e-3)
            ),
            "maximum_absolute_error_vs_cpu_oracle": maximum_absolute_error,
            "fallback_token": int(baseline_token_host[0]),
            "candidate_token": int(candidate_token_host[0]),
            "cpu_oracle_token": expected_token,
            "token_exact": bool(
                int(baseline_token_host[0])
                == int(candidate_token_host[0])
                == expected_token
            ),
            "finite": bool(np.isfinite(candidate_logits_host).all()),
        }
        correctness["passed"] = bool(
            correctness["full_fp16_logits_byte_exact_vs_fallback"]
            and correctness["candidate_allclose_cpu_oracle"]
            and correctness["token_exact"]
            and correctness["finite"]
        )
        if not correctness["passed"]:
            raise RuntimeError(f"Moonshine LM-head correctness failed: {correctness}")

        if not args.trace_smoke:
            routes = {"wave8_argmax": launch_baseline, "wave8_top1": launch_candidate}
            for index in range(args.warmups):
                order = tuple(routes) if index % 2 == 0 else tuple(reversed(routes))
                for name in order:
                    routes[name]()
            runtime.device_synchronize()
            samples = {
                name: {"event_us": [], "wall_us": []}
                for name in routes
            }
            orders: list[list[str]] = []
            for repetition in range(args.repetitions):
                order = tuple(routes) if repetition % 2 == 0 else tuple(reversed(routes))
                orders.append(list(order))
                for name in order:
                    event_us, wall_us = _measure(runtime, routes[name], args.burst)
                    samples[name]["event_us"].append(event_us)
                    samples[name]["wall_us"].append(wall_us)
            summaries = {
                name: {
                    scope: summarize_samples(values)
                    for scope, values in route_samples.items()
                }
                for name, route_samples in samples.items()
            }
            baseline_event = summaries["wave8_argmax"]["event_us"]["median_us"]
            candidate_event = summaries["wave8_top1"]["event_us"]["median_us"]
            baseline_wall = summaries["wave8_argmax"]["wall_us"]["median_us"]
            candidate_wall = summaries["wave8_top1"]["wall_us"]["median_us"]
            event_improvement = improvement_percent(baseline_event, candidate_event)
            wall_improvement = improvement_percent(baseline_wall, candidate_wall)
            timing = {
                "routes": summaries,
                "counterbalanced_orders": orders,
                "event_improvement_percent": event_improvement,
                "wall_improvement_percent": wall_improvement,
                "promotion_threshold_percent": args.minimum_improvement_percent,
                "performance_gate_passed": bool(
                    event_improvement >= args.minimum_improvement_percent
                    and wall_improvement >= args.minimum_improvement_percent
                ),
            }
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    after = memory_stats()
    lifecycle = {
        "baseline_current_allocated_bytes": baseline_stats["current_allocated_bytes"],
        "final_current_allocated_bytes": after["current_allocated_bytes"],
        "baseline_active_allocations": baseline_stats["active_allocations"],
        "final_active_allocations": after["active_allocations"],
        "returned_to_baseline": bool(
            after["current_allocated_bytes"] == baseline_stats["current_allocated_bytes"]
            and after["active_allocations"] == baseline_stats["active_allocations"]
        ),
    }
    if not lifecycle["returned_to_baseline"]:
        raise RuntimeError(f"Moonshine LM-head lifecycle failed: {lifecycle}")
    if args.trace_smoke:
        print(json.dumps({"correctness": correctness, "lifecycle": lifecycle}, sort_keys=True))
        return 0

    assert timing is not None
    repo_root = Path(__file__).resolve().parents[1]
    profiler = _load_profiler_summary(args.profiler_summary_json)
    provenance = collect_artifact_provenance(
        repo_root=repo_root,
        configured_backend="hip_gfx1151",
        resolved_backend="hip_gfx1151",
        target_arch="gfx1151",
        model_path=None,
        model_revision=MODEL_REVISION,
        quant="fp16_synthetic_production_shape",
        kv_dtype="not_applicable",
        command=sys.argv,
        build_profile="decode",
        timing_protocol="same_process_counterbalanced_hip_event_and_synchronized_wall",
        warmups=args.warmups,
        repetitions=args.repetitions,
        profiler=profiler,
        hipcc_version=compiler_version,
    )
    accepted = bool(
        correctness["passed"]
        and lifecycle["returned_to_baseline"]
        and timing["performance_gate_passed"]
        and not provenance["dirty"]
    )
    status = "accepted_kernel_microbenchmark" if accepted else (
        "diagnostic_dirty" if provenance["dirty"] else "rejected_performance"
    )
    report = {
        "schema_version": 1,
        "kind": "hipengine_moonshine_lm_head_ab",
        "status": status,
        "performance_claim": accepted,
        "scope": "synthetic_production_shape_kernel_microbenchmark",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights": "deterministic_synthetic_fp16",
        },
        "workload": {
            "rows": 1,
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "weight_bytes": VOCAB_SIZE * HIDDEN_SIZE * np.dtype(np.float16).itemsize,
            "partial_count": partial_count,
            "candidate_scratch_bytes": (
                partial_count * np.dtype(np.float16).itemsize
                + partial_count * np.dtype(np.int64).itemsize
            ),
            "seed": args.seed,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "burst": args.burst,
        },
        "correctness": correctness,
        "lifecycle": lifecycle,
        "timing": timing,
        "profiler": profiler,
        "decision": {
            "retain_candidate": accepted,
            "runtime_default_change": False,
            "reason": (
                "leaf gate passed; model-derived fixture bundle is required for runtime default"
                if accepted
                else "candidate did not clear clean correctness/lifecycle/event/wall gates"
            ),
        },
        "source_files": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in (
                repo_root
                / "hipengine/kernels/hip_gfx1100/linear/moonshine_projection.hip",
                repo_root
                / "hipengine/kernels/hip_gfx1100/linear/moonshine_projection.py",
                repo_root / "scripts/moonshine_lm_head_bench.py",
            )
        },
        "provenance": provenance,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(
        json.dumps(
            {
                "status": status,
                "correctness": correctness["passed"],
                "event_improvement_percent": timing["event_improvement_percent"],
                "wall_improvement_percent": timing["wall_improvement_percent"],
                "output": None if args.output is None else str(args.output),
            },
            sort_keys=True,
        )
    )
    if args.require_accept and not accepted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
