#!/usr/bin/env python3
"""Torch-free gfx11 Moonshine LayerNorm correctness/profile child."""

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
from hipengine.kernels.cpu_reference.moonshine import moonshine_layernorm
from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
    build_moonshine_layernorm,
    moonshine_layernorm_fp16,
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


def main() -> int:
    args = parse_args()
    compiler_version = args.compiler_version_file.read_text()
    library = build_moonshine_layernorm(
        compiler_version=compiler_version,
        load=not args.prebuild_only,
        require_cached=args.require_cached_build,
    )
    if args.prebuild_only:
        print(library.output_path)
        return 0

    rng = np.random.default_rng(0x1A92)
    rows, hidden = 7, 416
    inputs = rng.normal(0.0, 0.6, size=(rows, hidden)).astype(np.float16)
    weights = rng.normal(1.0, 0.08, size=(hidden,)).astype(np.float16)
    expected = moonshine_layernorm(inputs, weights)
    runtime = get_hip_runtime()
    allocations = []
    try:
        device_input = _upload(inputs, runtime, allocations)
        device_weight = _upload(weights, runtime, allocations)
        device_output = malloc(expected.nbytes, runtime=runtime)
        allocations.append(device_output)
        moonshine_layernorm_fp16(
            device_input.ptr,
            device_weight.ptr,
            device_output.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = np.empty_like(expected)
        copy_device_to_host(host_array_ptr(actual), device_output, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    finite = bool(np.isfinite(actual).all())
    all_close = bool(np.allclose(actual, expected, rtol=3.0e-3, atol=3.0e-3))
    report = {
        "all_passed": bool(finite and all_close),
        "finite": finite,
        "hidden_size": hidden,
        "max_abs": float(np.max(difference)),
        "relative_l2": float(
            np.linalg.norm(difference.ravel())
            / max(np.linalg.norm(expected.astype(np.float32).ravel()), 1.0e-12)
        ),
        "rows": rows,
        "expected_kernel_names": ["moonshine_layernorm_fp16_kernel"],
    }
    text = json.dumps(report, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
