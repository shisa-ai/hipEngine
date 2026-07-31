#!/usr/bin/env python3
"""Benchmark Moonshine fixed-cache self-attention candidates by past length."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.moonshine import moonshine_attention
from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
    build_moonshine_attention,
    moonshine_self_attention_branch_fp16,
    moonshine_self_attention_fp16,
    moonshine_self_attention_parallel_fp16,
)

HEADS = 8
HEAD_DIM = 52
CAPACITY = 194
POSITIONS = (0, 1, 8, 32, 64, 128, 193)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--burst", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0x5E1F)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _upload(
    array: np.ndarray,
    runtime: HipRuntime,
    allocations: list[DeviceBuffer],
) -> DeviceBuffer:
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _allocate(
    shape: tuple[int, ...],
    runtime: HipRuntime,
    allocations: list[DeviceBuffer],
) -> DeviceBuffer:
    size = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    device = malloc(size, runtime=runtime)
    allocations.append(device)
    return device


def _download(
    device: DeviceBuffer,
    shape: tuple[int, ...],
    runtime: HipRuntime,
) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _event_us(
    runtime: HipRuntime,
    launch: Callable[[], None],
    burst: int,
) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return float(runtime.event_elapsed_time_ms(start, stop)) * 1000.0 / burst
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _summary(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    p95_index = max(0, min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1))
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "p95_us": ordered[p95_index],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _correctness(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    relative_l2 = float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(expected.astype(np.float32).ravel()), 1.0e-12)
    )
    return {
        "finite": bool(np.isfinite(actual).all()),
        "all_close": bool(np.allclose(actual, expected, rtol=5.0e-3, atol=5.0e-3)),
        "max_abs": float(np.max(difference)),
        "relative_l2": relative_l2,
    }


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repetitions <= 0 or args.burst <= 0:
        raise ValueError("warmups must be nonnegative; repetitions and burst must be positive")
    compiler_version = args.compiler_version_file.read_text()
    library = build_moonshine_attention(
        compiler_version=compiler_version,
        load=True,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    rng = np.random.default_rng(args.seed)
    report_cases: dict[str, Any] = {}
    all_passed = True

    for position in POSITIONS:
        allocations: list[DeviceBuffer] = []
        try:
            visible_length = position + 1
            query = rng.normal(0.0, 0.08, size=(1, HEADS, 1, HEAD_DIM)).astype(
                np.float16
            )
            key = rng.normal(
                0.0, 0.08, size=(1, HEADS, CAPACITY, HEAD_DIM)
            ).astype(np.float16)
            value = rng.normal(
                0.0, 0.10, size=(1, HEADS, CAPACITY, HEAD_DIM)
            ).astype(np.float16)
            expected = moonshine_attention(
                query,
                key[:, :, :visible_length],
                value[:, :, :visible_length],
            )
            device_query = _upload(query, runtime, allocations)
            device_key = _upload(key, runtime, allocations)
            device_value = _upload(value, runtime, allocations)
            device_position = _upload(
                np.asarray([position], dtype=np.int64), runtime, allocations
            )
            device_output = _allocate(expected.shape, runtime, allocations)
            common = {"library": library, "runtime": runtime}
            launches: dict[str, Callable[[], None]] = {
                "fallback_wave1": lambda: moonshine_self_attention_fp16(
                    device_query.ptr,
                    device_key.ptr,
                    device_value.ptr,
                    device_position.ptr,
                    device_output.ptr,
                    HEADS,
                    HEAD_DIM,
                    CAPACITY,
                    **common,
                ),
                "branch_wave1": lambda: moonshine_self_attention_branch_fp16(
                    device_query.ptr,
                    device_key.ptr,
                    device_value.ptr,
                    device_position.ptr,
                    device_output.ptr,
                    HEADS,
                    HEAD_DIM,
                    CAPACITY,
                    **common,
                ),
            }
            for threads in (64, 128, 256):
                launches[f"parallel_tokens_threads{threads}"] = (
                    lambda threads=threads: moonshine_self_attention_parallel_fp16(
                        device_query.ptr,
                        device_key.ptr,
                        device_value.ptr,
                        device_position.ptr,
                        device_output.ptr,
                        HEADS,
                        HEAD_DIM,
                        CAPACITY,
                        threads=threads,
                        **common,
                    )
                )
            for launch in launches.values():
                for _ in range(args.warmups):
                    launch()
            runtime.device_synchronize()
            samples = {name: [] for name in launches}
            names = tuple(launches)
            for repetition in range(args.repetitions):
                order = names[repetition % len(names) :] + names[: repetition % len(names)]
                for name in order:
                    samples[name].append(_event_us(runtime, launches[name], args.burst))
            variants: dict[str, Any] = {}
            for name, launch in launches.items():
                launch()
                runtime.device_synchronize()
                actual = _download(device_output, expected.shape, runtime)
                correctness = _correctness(actual, expected)
                all_passed = all_passed and correctness["finite"] and correctness["all_close"]
                variants[name] = {
                    **_summary(samples[name]),
                    "correctness": correctness,
                }
            baseline = variants["fallback_wave1"]["median_us"]
            for value_row in variants.values():
                value_row["speedup_vs_fallback"] = baseline / value_row["median_us"]
            report_cases[str(position)] = {
                "position": position,
                "visible_length": visible_length,
                "best_variant": min(
                    variants,
                    key=lambda name: variants[name]["median_us"],
                ),
                "variants": variants,
            }
        finally:
            for allocation in reversed(allocations):
                free(allocation, runtime=runtime)

    report = {
        "schema": 1,
        "kind": "moonshine_self_attention_candidate_benchmark",
        "status": "diagnostic",
        "performance_claim": False,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "burst": args.burst,
        "all_correctness_passed": all_passed,
        "cases": report_cases,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(
        json.dumps(
            {
                "all_correctness_passed": all_passed,
                "cases": {
                    position: {
                        "best_variant": case["best_variant"],
                        "median_us": {
                            variant: round(row["median_us"], 3)
                            for variant, row in case["variants"].items()
                        },
                    }
                    for position, case in report_cases.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
