#!/usr/bin/env python3
"""Torch-free gfx11 Moonshine projection correctness/profile child."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.moonshine import moonshine_projection
from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
    build_moonshine_projection,
    moonshine_f16_projection,
    moonshine_f16_projection_bias,
    moonshine_f16_projection_pair,
    moonshine_f16_projection_pair_head_major,
    moonshine_f16_projection_triple,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--prebuild-only", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _output(shape: tuple[int, ...], runtime, allocations):
    device = malloc(int(np.prod(shape)) * 2, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape: tuple[int, ...], runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def main() -> int:
    args = parse_args()
    compiler_version = args.compiler_version_file.read_text()
    library = build_moonshine_projection(
        compiler_version=compiler_version,
        load=not args.prebuild_only,
        require_cached=args.require_cached_build,
    )
    if args.prebuild_only:
        print(library.output_path)
        return 0

    rng = np.random.default_rng(0x92B)
    rows, hidden = 4, 416
    inputs = rng.normal(0.0, 0.05, size=(rows, hidden)).astype(np.float16)
    weights = tuple(
        rng.normal(0.0, 0.04, size=(hidden, hidden)).astype(np.float16)
        for _ in range(3)
    )
    bias = rng.normal(0.0, 0.03, size=(hidden,)).astype(np.float16)
    expected = tuple(moonshine_projection(inputs, weight) for weight in weights)
    expected_bias = moonshine_projection(inputs, weights[0], bias)
    runtime = get_hip_runtime()
    allocations = []
    try:
        device_input = _upload(inputs, runtime, allocations)
        device_weights = tuple(_upload(weight, runtime, allocations) for weight in weights)
        device_bias = _upload(bias, runtime, allocations)
        single = _output((rows, hidden), runtime, allocations)
        biased = _output((rows, hidden), runtime, allocations)
        pair = tuple(_output((rows, hidden), runtime, allocations) for _ in range(2))
        head_major = tuple(_output((8, rows, 52), runtime, allocations) for _ in range(2))
        triple = tuple(_output((rows, hidden), runtime, allocations) for _ in range(3))
        moonshine_f16_projection(
            device_input.ptr,
            device_weights[0].ptr,
            single.ptr,
            rows,
            hidden,
            hidden,
            library=library,
            runtime=runtime,
        )
        moonshine_f16_projection_bias(
            device_input.ptr,
            device_weights[0].ptr,
            device_bias.ptr,
            biased.ptr,
            rows,
            hidden,
            hidden,
            library=library,
            runtime=runtime,
        )
        moonshine_f16_projection_pair(
            device_input.ptr,
            device_weights[0].ptr,
            device_weights[1].ptr,
            pair[0].ptr,
            pair[1].ptr,
            rows,
            hidden,
            hidden,
            hidden,
            library=library,
            runtime=runtime,
        )
        moonshine_f16_projection_pair_head_major(
            device_input.ptr,
            device_weights[0].ptr,
            device_weights[1].ptr,
            head_major[0].ptr,
            head_major[1].ptr,
            rows,
            hidden,
            hidden,
            hidden,
            52,
            library=library,
            runtime=runtime,
        )
        moonshine_f16_projection_triple(
            device_input.ptr,
            *(weight.ptr for weight in device_weights),
            *(output.ptr for output in triple),
            rows,
            hidden,
            hidden,
            hidden,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_single = _download(single, (rows, hidden), runtime)
        actual_bias = _download(biased, (rows, hidden), runtime)
        actual_pair = tuple(_download(output, (rows, hidden), runtime) for output in pair)
        actual_head_major = tuple(
            _download(output, (8, rows, 52), runtime) for output in head_major
        )
        actual_triple = tuple(
            _download(output, (rows, hidden), runtime) for output in triple
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    expected_head_major = tuple(
        value.reshape(rows, 8, 52).transpose(1, 0, 2) for value in expected[:2]
    )
    comparisons = (
        actual_single,
        actual_bias,
        *actual_pair,
        *actual_head_major,
        *actual_triple,
    )
    references = (
        expected[0],
        expected_bias,
        expected[0],
        expected[1],
        *expected_head_major,
        *expected,
    )
    max_abs = max(
        float(np.max(np.abs(actual.astype(np.float32) - reference.astype(np.float32))))
        for actual, reference in zip(comparisons, references, strict=True)
    )
    finite = all(bool(np.isfinite(value).all()) for value in comparisons)
    all_close = all(
        bool(np.allclose(actual, reference, rtol=2.0e-3, atol=2.0e-3))
        for actual, reference in zip(comparisons, references, strict=True)
    )
    report = {
        "all_passed": bool(finite and all_close),
        "finite": finite,
        "max_abs": max_abs,
        "rows": rows,
        "hidden_size": hidden,
        "expected_kernel_names": [
            "moonshine_f16_projection_kernel",
            "moonshine_f16_projection_bias_kernel",
            "moonshine_f16_projection_pair_kernel",
            "moonshine_f16_projection_pair_head_major_kernel",
            "moonshine_f16_projection_triple_kernel",
        ],
    }
    text = json.dumps(report, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
