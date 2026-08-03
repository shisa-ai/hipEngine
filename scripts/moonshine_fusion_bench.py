#!/usr/bin/env python3
"""Benchmark exact Moonshine residual/LayerNorm and decoder-MLP fusions."""

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
from hipengine.kernels.cpu_reference.moonshine import (
    moonshine_decoder_mlp,
    moonshine_layernorm,
    moonshine_mlp_fc1_gated_silu,
    moonshine_residual,
)
from hipengine.kernels.hip_gfx1100.fused.moonshine_glue import (
    build_moonshine_glue,
    moonshine_residual_fp16,
)
from hipengine.kernels.hip_gfx1100.fused.moonshine_mlp import (
    build_moonshine_mlp,
    moonshine_gated_silu_fp16,
)
from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
    build_moonshine_projection,
    moonshine_f16_projection_bias,
    moonshine_f16_projection_bias_gated_silu,
    moonshine_f16_projection_bias_residual,
)
from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
    build_moonshine_layernorm,
    moonshine_layernorm_fp16,
    moonshine_residual_layernorm_fp16,
)

HIDDEN = 416
INTERMEDIATE = 1664


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--burst", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0xF053D)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _upload(array: np.ndarray, runtime: HipRuntime, allocations: list[DeviceBuffer]):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _allocate(shape: tuple[int, ...], runtime: HipRuntime, allocations):
    size = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    device = malloc(size, runtime=runtime)
    allocations.append(device)
    return device


def _download(device: DeviceBuffer, shape: tuple[int, ...], runtime: HipRuntime):
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _event_us(runtime: HipRuntime, launch: Callable[[], None], burst: int) -> float:
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


def _correctness(actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]):
    differences = [
        np.abs(value.astype(np.float32) - reference.astype(np.float32))
        for value, reference in zip(actual, expected, strict=True)
    ]
    return {
        "finite": all(bool(np.isfinite(value).all()) for value in actual),
        "exact": all(
            bool(np.array_equal(value, reference))
            for value, reference in zip(actual, expected, strict=True)
        ),
        "all_close": all(
            bool(np.allclose(value, reference, rtol=3.0e-3, atol=3.0e-3))
            for value, reference in zip(actual, expected, strict=True)
        ),
        "max_abs": max(float(np.max(value)) for value in differences),
    }


def _measure_case(
    *,
    runtime: HipRuntime,
    launches: dict[str, Callable[[], None]],
    outputs: Callable[[], tuple[np.ndarray, ...]],
    expected: tuple[np.ndarray, ...],
    warmups: int,
    repetitions: int,
    burst: int,
) -> tuple[dict[str, Any], bool]:
    for launch in launches.values():
        for _ in range(warmups):
            launch()
    runtime.device_synchronize()
    samples = {name: [] for name in launches}
    names = tuple(launches)
    for repetition in range(repetitions):
        order = names[repetition % len(names) :] + names[: repetition % len(names)]
        for name in order:
            samples[name].append(_event_us(runtime, launches[name], burst))
    variants: dict[str, Any] = {}
    passed = True
    for name, launch in launches.items():
        launch()
        runtime.device_synchronize()
        correctness = _correctness(outputs(), expected)
        passed = passed and correctness["finite"] and correctness["all_close"]
        variants[name] = {**_summary(samples[name]), "correctness": correctness}
    baseline = variants["unfused"]["median_us"]
    for row in variants.values():
        row["speedup_vs_unfused"] = baseline / row["median_us"]
    return {
        "best_variant": min(variants, key=lambda name: variants[name]["median_us"]),
        "variants": variants,
    }, passed


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repetitions <= 0 or args.burst <= 0:
        raise ValueError("warmups must be nonnegative; repetitions and burst must be positive")
    compiler_version = args.compiler_version_file.read_text()
    build_args = {
        "compiler_version": compiler_version,
        "load": True,
        "require_cached": args.require_cached_build,
    }
    projection = build_moonshine_projection(**build_args)
    glue = build_moonshine_glue(**build_args)
    mlp = build_moonshine_mlp(**build_args)
    layernorm = build_moonshine_layernorm(**build_args)
    runtime = get_hip_runtime()
    rng = np.random.default_rng(args.seed)
    allocations: list[DeviceBuffer] = []
    try:
        normalized = rng.normal(0.0, 0.06, size=(1, HIDDEN)).astype(np.float16)
        residual = rng.normal(0.0, 0.08, size=(1, HIDDEN)).astype(np.float16)
        update = rng.normal(0.0, 0.08, size=(1, HIDDEN)).astype(np.float16)
        norm_weight = rng.normal(1.0, 0.08, size=(HIDDEN,)).astype(np.float16)
        fc1_weight = rng.normal(
            0.0, 0.025, size=(2 * INTERMEDIATE, HIDDEN)
        ).astype(np.float16)
        fc1_bias = rng.normal(0.0, 0.02, size=(2 * INTERMEDIATE,)).astype(np.float16)
        fc2_weight = rng.normal(0.0, 0.025, size=(HIDDEN, INTERMEDIATE)).astype(
            np.float16
        )
        fc2_bias = rng.normal(0.0, 0.02, size=(HIDDEN,)).astype(np.float16)

        device_normalized = _upload(normalized, runtime, allocations)
        device_residual = _upload(residual, runtime, allocations)
        device_update = _upload(update, runtime, allocations)
        device_norm_weight = _upload(norm_weight, runtime, allocations)
        device_fc1_weight = _upload(fc1_weight, runtime, allocations)
        device_fc1_bias = _upload(fc1_bias, runtime, allocations)
        device_fc2_weight = _upload(fc2_weight, runtime, allocations)
        device_fc2_bias = _upload(fc2_bias, runtime, allocations)
        fc1_output = _allocate((1, 2 * INTERMEDIATE), runtime, allocations)
        intermediate_output = _allocate((1, INTERMEDIATE), runtime, allocations)
        projection_output = _allocate((1, HIDDEN), runtime, allocations)
        residual_output = _allocate((1, HIDDEN), runtime, allocations)
        norm_output = _allocate((1, HIDDEN), runtime, allocations)

        common = {"runtime": runtime}

        expected_residual = moonshine_residual(residual, update)
        expected_norm = moonshine_layernorm(expected_residual, norm_weight)

        def residual_norm_unfused() -> None:
            moonshine_residual_fp16(
                device_residual.ptr,
                device_update.ptr,
                residual_output.ptr,
                HIDDEN,
                library=glue,
                **common,
            )
            moonshine_layernorm_fp16(
                residual_output.ptr,
                device_norm_weight.ptr,
                norm_output.ptr,
                1,
                HIDDEN,
                library=layernorm,
                **common,
            )

        def residual_norm_fused() -> None:
            moonshine_residual_layernorm_fp16(
                device_residual.ptr,
                device_update.ptr,
                device_norm_weight.ptr,
                residual_output.ptr,
                norm_output.ptr,
                1,
                HIDDEN,
                library=layernorm,
                **common,
            )

        residual_norm_case, residual_norm_passed = _measure_case(
            runtime=runtime,
            launches={"unfused": residual_norm_unfused, "fused": residual_norm_fused},
            outputs=lambda: (
                _download(residual_output, (1, HIDDEN), runtime),
                _download(norm_output, (1, HIDDEN), runtime),
            ),
            expected=(expected_residual, expected_norm),
            warmups=args.warmups,
            repetitions=args.repetitions,
            burst=args.burst,
        )

        expected_mlp = moonshine_decoder_mlp(
            normalized, fc1_weight, fc1_bias, fc2_weight, fc2_bias
        )
        expected_hidden = moonshine_residual(residual, expected_mlp)
        expected_next_norm = moonshine_layernorm(expected_hidden, norm_weight)

        def fc1_unfused() -> None:
            moonshine_f16_projection_bias(
                device_normalized.ptr,
                device_fc1_weight.ptr,
                device_fc1_bias.ptr,
                fc1_output.ptr,
                1,
                HIDDEN,
                2 * INTERMEDIATE,
                threads=32,
                library=projection,
                **common,
            )
            moonshine_gated_silu_fp16(
                fc1_output.ptr,
                intermediate_output.ptr,
                1,
                INTERMEDIATE,
                library=mlp,
                **common,
            )

        def fc1_fused() -> None:
            moonshine_f16_projection_bias_gated_silu(
                device_normalized.ptr,
                device_fc1_weight.ptr,
                device_fc1_bias.ptr,
                intermediate_output.ptr,
                1,
                HIDDEN,
                INTERMEDIATE,
                library=projection,
                **common,
            )

        expected_intermediate = moonshine_mlp_fc1_gated_silu(
            normalized, fc1_weight, fc1_bias
        )
        fc1_case, fc1_passed = _measure_case(
            runtime=runtime,
            launches={"unfused": fc1_unfused, "fused": fc1_fused},
            outputs=lambda: (_download(intermediate_output, (1, INTERMEDIATE), runtime),),
            expected=(expected_intermediate,),
            warmups=args.warmups,
            repetitions=args.repetitions,
            burst=args.burst,
        )

        fc1_fused()
        runtime.device_synchronize()

        def fc2_unfused() -> None:
            moonshine_f16_projection_bias(
                intermediate_output.ptr,
                device_fc2_weight.ptr,
                device_fc2_bias.ptr,
                projection_output.ptr,
                1,
                INTERMEDIATE,
                HIDDEN,
                threads=64,
                library=projection,
                **common,
            )
            moonshine_residual_fp16(
                device_residual.ptr,
                projection_output.ptr,
                residual_output.ptr,
                HIDDEN,
                library=glue,
                **common,
            )

        def fc2_fused() -> None:
            moonshine_f16_projection_bias_residual(
                intermediate_output.ptr,
                device_fc2_weight.ptr,
                device_fc2_bias.ptr,
                device_residual.ptr,
                residual_output.ptr,
                1,
                INTERMEDIATE,
                HIDDEN,
                threads=64,
                library=projection,
                **common,
            )

        fc2_case, fc2_passed = _measure_case(
            runtime=runtime,
            launches={"unfused": fc2_unfused, "fused": fc2_fused},
            outputs=lambda: (_download(residual_output, (1, HIDDEN), runtime),),
            expected=(expected_hidden,),
            warmups=args.warmups,
            repetitions=args.repetitions,
            burst=args.burst,
        )

        def whole_unfused() -> None:
            fc1_unfused()
            fc2_unfused()
            moonshine_layernorm_fp16(
                residual_output.ptr,
                device_norm_weight.ptr,
                norm_output.ptr,
                1,
                HIDDEN,
                library=layernorm,
                **common,
            )

        def whole_fc2_residual() -> None:
            fc1_fused()
            fc2_fused()
            moonshine_layernorm_fp16(
                residual_output.ptr,
                device_norm_weight.ptr,
                norm_output.ptr,
                1,
                HIDDEN,
                library=layernorm,
                **common,
            )

        def whole_residual_norm() -> None:
            fc1_fused()
            moonshine_f16_projection_bias(
                intermediate_output.ptr,
                device_fc2_weight.ptr,
                device_fc2_bias.ptr,
                projection_output.ptr,
                1,
                INTERMEDIATE,
                HIDDEN,
                threads=64,
                library=projection,
                **common,
            )
            moonshine_residual_layernorm_fp16(
                device_residual.ptr,
                projection_output.ptr,
                device_norm_weight.ptr,
                residual_output.ptr,
                norm_output.ptr,
                1,
                HIDDEN,
                library=layernorm,
                **common,
            )

        whole_case, whole_passed = _measure_case(
            runtime=runtime,
            launches={
                "unfused": whole_unfused,
                "fc1_fused_fc2_residual": whole_fc2_residual,
                "fc1_fused_residual_norm": whole_residual_norm,
            },
            outputs=lambda: (
                _download(residual_output, (1, HIDDEN), runtime),
                _download(norm_output, (1, HIDDEN), runtime),
            ),
            expected=(expected_hidden, expected_next_norm),
            warmups=args.warmups,
            repetitions=args.repetitions,
            burst=args.burst,
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    cases = {
        "residual_layernorm": residual_norm_case,
        "fc1_gated_silu": fc1_case,
        "fc2_residual": fc2_case,
        "whole_mlp_next_norm": whole_case,
    }
    all_passed = all(
        (residual_norm_passed, fc1_passed, fc2_passed, whole_passed)
    )
    report = {
        "schema": 1,
        "kind": "moonshine_bounded_fusion_benchmark",
        "status": "diagnostic",
        "performance_claim": False,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "burst": args.burst,
        "all_correctness_passed": all_passed,
        "cases": cases,
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
                    name: {
                        "best_variant": case["best_variant"],
                        "median_us": {
                            variant: round(row["median_us"], 3)
                            for variant, row in case["variants"].items()
                        },
                    }
                    for name, case in cases.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
