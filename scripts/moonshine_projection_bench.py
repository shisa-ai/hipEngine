#!/usr/bin/env python3
"""Benchmark production-shape Moonshine FP16 projection launch geometry."""

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
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.cpu_reference.moonshine import moonshine_projection
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    build_dense_gemv,
    dense_gemv_out_fp16,
    dense_gemv_out_fp16_wmma,
)
from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
    build_moonshine_projection,
    moonshine_f16_projection,
    moonshine_f16_projection_bias,
    moonshine_f16_projection_pair_head_major,
    moonshine_f16_projection_triple,
)

THREADS = (32, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--burst", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0x92B)
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
    device = malloc(int(np.prod(shape)) * np.dtype(np.float16).itemsize, runtime=runtime)
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


def _check(
    actual: tuple[np.ndarray, ...],
    expected: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    differences = [
        np.abs(value.astype(np.float32) - reference.astype(np.float32))
        for value, reference in zip(actual, expected, strict=True)
    ]
    max_abs = max(float(np.max(value)) for value in differences)
    relative_l2 = max(
        float(
            np.linalg.norm(value.ravel())
            / max(np.linalg.norm(reference.astype(np.float32).ravel()), 1.0e-12)
        )
        for value, reference in zip(differences, expected, strict=True)
    )
    return {
        "finite": all(bool(np.isfinite(value).all()) for value in actual),
        "max_abs": max_abs,
        "max_relative_l2": relative_l2,
        "all_close": all(
            bool(np.allclose(value, reference, rtol=5.0e-3, atol=5.0e-3))
            for value, reference in zip(actual, expected, strict=True)
        ),
    }


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repetitions <= 0 or args.burst <= 0:
        raise ValueError("warmups must be nonnegative; repetitions and burst must be positive")
    compiler_version = args.compiler_version_file.read_text()
    library = build_moonshine_projection(
        compiler_version=compiler_version,
        load=True,
        require_cached=args.require_cached_build,
    )
    dense_library = build_dense_gemv(
        compiler_version=compiler_version,
        load=True,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    rocblas = Rocblas.load()
    rng = np.random.default_rng(args.seed)
    allocations: list[DeviceBuffer] = []
    try:
        x416 = rng.normal(0.0, 0.05, size=(1, 416)).astype(np.float16)
        x1664 = rng.normal(0.0, 0.05, size=(1, 1664)).astype(np.float16)
        x40 = rng.normal(0.0, 0.05, size=(40, 416)).astype(np.float16)
        w416 = tuple(
            rng.normal(0.0, 0.04, size=(416, 416)).astype(np.float16)
            for _ in range(3)
        )
        w_fc1 = rng.normal(0.0, 0.04, size=(3328, 416)).astype(np.float16)
        b_fc1 = rng.normal(0.0, 0.03, size=(3328,)).astype(np.float16)
        w_fc2 = rng.normal(0.0, 0.04, size=(416, 1664)).astype(np.float16)
        b_fc2 = rng.normal(0.0, 0.03, size=(416,)).astype(np.float16)

        dx416 = _upload(x416, runtime, allocations)
        dx1664 = _upload(x1664, runtime, allocations)
        dx40 = _upload(x40, runtime, allocations)
        dw416 = tuple(_upload(value, runtime, allocations) for value in w416)
        dw_fc1 = _upload(w_fc1, runtime, allocations)
        db_fc1 = _upload(b_fc1, runtime, allocations)
        dw_fc2 = _upload(w_fc2, runtime, allocations)
        db_fc2 = _upload(b_fc2, runtime, allocations)

        single = _allocate((1, 416), runtime, allocations)
        triple = tuple(_allocate((1, 416), runtime, allocations) for _ in range(3))
        fc1 = _allocate((1, 3328), runtime, allocations)
        fc2 = _allocate((1, 416), runtime, allocations)
        head_major = tuple(_allocate((8, 40, 52), runtime, allocations) for _ in range(2))

        launches: dict[str, Callable[[int], None]] = {
            "single_416x416": lambda threads: moonshine_f16_projection(
                dx416.ptr,
                dw416[0].ptr,
                single.ptr,
                1,
                416,
                416,
                threads=threads,
                library=library,
                runtime=runtime,
            ),
            "triple_416x416": lambda threads: moonshine_f16_projection_triple(
                dx416.ptr,
                *(value.ptr for value in dw416),
                *(value.ptr for value in triple),
                1,
                416,
                416,
                416,
                416,
                threads=threads,
                library=library,
                runtime=runtime,
            ),
            "bias_fc1_416x3328": lambda threads: moonshine_f16_projection_bias(
                dx416.ptr,
                dw_fc1.ptr,
                db_fc1.ptr,
                fc1.ptr,
                1,
                416,
                3328,
                threads=threads,
                library=library,
                runtime=runtime,
            ),
            "bias_fc2_1664x416": lambda threads: moonshine_f16_projection_bias(
                dx1664.ptr,
                dw_fc2.ptr,
                db_fc2.ptr,
                fc2.ptr,
                1,
                1664,
                416,
                threads=threads,
                library=library,
                runtime=runtime,
            ),
            "cross_pair_40x416x416": lambda threads: moonshine_f16_projection_pair_head_major(
                dx40.ptr,
                dw416[0].ptr,
                dw416[1].ptr,
                head_major[0].ptr,
                head_major[1].ptr,
                40,
                416,
                416,
                416,
                52,
                threads=threads,
                library=library,
                runtime=runtime,
            ),
        }
        references = {
            "single_416x416": (moonshine_projection(x416, w416[0]),),
            "triple_416x416": tuple(moonshine_projection(x416, value) for value in w416),
            "bias_fc1_416x3328": (moonshine_projection(x416, w_fc1, b_fc1),),
            "bias_fc2_1664x416": (moonshine_projection(x1664, w_fc2, b_fc2),),
            "cross_pair_40x416x416": tuple(
                moonshine_projection(x40, value).reshape(40, 8, 52).transpose(1, 0, 2)
                for value in w416[:2]
            ),
        }
        outputs = {
            "single_416x416": lambda: (_download(single, (1, 416), runtime),),
            "triple_416x416": lambda: tuple(
                _download(value, (1, 416), runtime) for value in triple
            ),
            "bias_fc1_416x3328": lambda: (_download(fc1, (1, 3328), runtime),),
            "bias_fc2_1664x416": lambda: (_download(fc2, (1, 416), runtime),),
            "cross_pair_40x416x416": lambda: tuple(
                _download(value, (8, 40, 52), runtime) for value in head_major
            ),
        }

        cases: dict[str, Any] = {}
        all_passed = True
        for case_index, (name, launch) in enumerate(launches.items()):
            for threads in THREADS:
                for _ in range(args.warmups):
                    launch(threads)
            runtime.device_synchronize()
            samples = {threads: [] for threads in THREADS}
            for repetition in range(args.repetitions):
                order = THREADS[repetition % len(THREADS) :] + THREADS[: repetition % len(THREADS)]
                for threads in order:
                    samples[threads].append(
                        _event_us(runtime, lambda threads=threads: launch(threads), args.burst)
                    )
            variants: dict[str, Any] = {}
            for threads in THREADS:
                launch(threads)
                runtime.device_synchronize()
                correctness = _check(outputs[name](), references[name])
                all_passed = all_passed and correctness["finite"] and correctness["all_close"]
                variants[str(threads)] = {
                    **_summary(samples[threads]),
                    "correctness": correctness,
                }
            baseline = variants["256"]["median_us"]
            for value in variants.values():
                value["speedup_vs_threads256"] = baseline / value["median_us"]
            cases[name] = {
                "case_index": case_index,
                "variants": variants,
                "best_threads": min(
                    THREADS,
                    key=lambda threads: variants[str(threads)]["median_us"],
                ),
            }

        def launch_dense_triple(threads: int | None) -> None:
            for weight, output in zip(dw416, triple, strict=True):
                function = (
                    dense_gemv_out_fp16_wmma
                    if threads is None
                    else dense_gemv_out_fp16
                )
                keywords = {} if threads is None else {"threads": threads}
                function(
                    dx416.ptr,
                    weight.ptr,
                    output.ptr,
                    1,
                    416,
                    416,
                    library=dense_library,
                    runtime=runtime,
                    **keywords,
                )

        alternatives: dict[str, tuple[Callable[[], None], str]] = {}
        for threads in (64, 128, 256):
            alternatives[f"single_dense_threads{threads}"] = (
                lambda threads=threads: dense_gemv_out_fp16(
                    dx416.ptr,
                    dw416[0].ptr,
                    single.ptr,
                    1,
                    416,
                    416,
                    threads=threads,
                    library=dense_library,
                    runtime=runtime,
                ),
                "single_416x416",
            )
            alternatives[f"triple_three_dense_threads{threads}"] = (
                lambda threads=threads: launch_dense_triple(threads),
                "triple_416x416",
            )
        alternatives["single_dense_wmma"] = (
            lambda: dense_gemv_out_fp16_wmma(
                dx416.ptr,
                dw416[0].ptr,
                single.ptr,
                1,
                416,
                416,
                library=dense_library,
                runtime=runtime,
            ),
            "single_416x416",
        )
        alternatives["triple_three_dense_wmma"] = (
            lambda: launch_dense_triple(None),
            "triple_416x416",
        )

        def launch_rocblas_triple() -> None:
            for weight, output in zip(dw416, triple, strict=True):
                rocblas.gemm_ex_rowmajor_nt_fp16_compute_f32(
                    dx416.ptr,
                    weight.ptr,
                    output.ptr,
                    rows=1,
                    in_features=416,
                    out_features=416,
                )

        alternatives["single_rocblas_gemm_ex_fp16_out"] = (
            lambda: rocblas.gemm_ex_rowmajor_nt_fp16_compute_f32(
                dx416.ptr,
                dw416[0].ptr,
                single.ptr,
                rows=1,
                in_features=416,
                out_features=416,
            ),
            "single_416x416",
        )
        alternatives["triple_three_rocblas_gemm_ex_fp16_out"] = (
            launch_rocblas_triple,
            "triple_416x416",
        )
        for launch, _ in alternatives.values():
            for _ in range(args.warmups):
                launch()
        runtime.device_synchronize()
        alternative_samples = {name: [] for name in alternatives}
        names = tuple(alternatives)
        for repetition in range(args.repetitions):
            order = names[repetition % len(names) :] + names[: repetition % len(names)]
            for name in order:
                alternative_samples[name].append(
                    _event_us(runtime, alternatives[name][0], args.burst)
                )
        alternative_results: dict[str, Any] = {}
        for name, (launch, reference_name) in alternatives.items():
            launch()
            runtime.device_synchronize()
            correctness = _check(outputs[reference_name](), references[reference_name])
            all_passed = all_passed and correctness["finite"] and correctness["all_close"]
            alternative_results[name] = {
                **_summary(alternative_samples[name]),
                "correctness": correctness,
                "reference_case": reference_name,
            }
    finally:
        rocblas.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    report = {
        "schema": 1,
        "kind": "moonshine_projection_thread_geometry_benchmark",
        "status": "diagnostic",
        "performance_claim": False,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "burst": args.burst,
        "threads": list(THREADS),
        "all_correctness_passed": all_passed,
        "cases": cases,
        "existing_dense_alternatives": alternative_results,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(
        json.dumps(
            {
                "all_correctness_passed": all_passed,
                "best_threads": {
                    name: value["best_threads"] for name, value in cases.items()
                },
                "median_us": {
                    name: {
                        threads: round(row["median_us"], 3)
                        for threads, row in value["variants"].items()
                    }
                    for name, value in cases.items()
                },
                "existing_dense_alternatives_us": {
                    name: round(value["median_us"], 3)
                    for name, value in alternative_results.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
