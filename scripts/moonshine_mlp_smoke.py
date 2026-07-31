#!/usr/bin/env python3
"""Torch-free gfx11 Moonshine gated-MLP correctness/profile child."""

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
from hipengine.kernels.cpu_reference.moonshine import (
    moonshine_decoder_mlp,
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


def _empty(shape: tuple[int, ...], runtime, allocations):
    device = malloc(int(np.prod(shape)) * 2, runtime=runtime)
    allocations.append(device)
    return device


def main() -> int:
    args = parse_args()
    compiler_version = args.compiler_version_file.read_text()
    build_arguments = {
        "compiler_version": compiler_version,
        "load": not args.prebuild_only,
        "require_cached": args.require_cached_build,
    }
    projection = build_moonshine_projection(**build_arguments)
    mlp = build_moonshine_mlp(**build_arguments)
    glue = build_moonshine_glue(**build_arguments)
    if args.prebuild_only:
        print(projection.output_path)
        print(mlp.output_path)
        print(glue.output_path)
        return 0

    rng = np.random.default_rng(0xDEC0DE)
    rows, hidden, intermediate = 1, 416, 1664
    normalized = rng.normal(0.0, 0.06, size=(rows, hidden)).astype(np.float16)
    residual = rng.normal(0.0, 0.08, size=(rows, hidden)).astype(np.float16)
    fc1_weight = rng.normal(
        0.0, 0.025, size=(2 * intermediate, hidden)
    ).astype(np.float16)
    fc1_bias = rng.normal(0.0, 0.02, size=(2 * intermediate,)).astype(np.float16)
    fc2_weight = rng.normal(0.0, 0.025, size=(hidden, intermediate)).astype(np.float16)
    fc2_bias = rng.normal(0.0, 0.02, size=(hidden,)).astype(np.float16)
    expected_mlp = moonshine_decoder_mlp(
        normalized, fc1_weight, fc1_bias, fc2_weight, fc2_bias
    )
    expected = moonshine_residual(residual, expected_mlp)

    runtime = get_hip_runtime()
    allocations = []
    try:
        device_normalized = _upload(normalized, runtime, allocations)
        device_residual = _upload(residual, runtime, allocations)
        device_fc1_weight = _upload(fc1_weight, runtime, allocations)
        device_fc1_bias = _upload(fc1_bias, runtime, allocations)
        device_fc2_weight = _upload(fc2_weight, runtime, allocations)
        device_fc2_bias = _upload(fc2_bias, runtime, allocations)
        fc1_output = _empty((rows, 2 * intermediate), runtime, allocations)
        intermediate_output = _empty((rows, intermediate), runtime, allocations)
        fc2_output = _empty((rows, hidden), runtime, allocations)
        final_output = _empty((rows, hidden), runtime, allocations)
        moonshine_f16_projection_bias(
            device_normalized.ptr,
            device_fc1_weight.ptr,
            device_fc1_bias.ptr,
            fc1_output.ptr,
            rows,
            hidden,
            2 * intermediate,
            library=projection,
            runtime=runtime,
        )
        moonshine_gated_silu_fp16(
            fc1_output.ptr,
            intermediate_output.ptr,
            rows,
            intermediate,
            library=mlp,
            runtime=runtime,
        )
        moonshine_f16_projection_bias(
            intermediate_output.ptr,
            device_fc2_weight.ptr,
            device_fc2_bias.ptr,
            fc2_output.ptr,
            rows,
            intermediate,
            hidden,
            library=projection,
            runtime=runtime,
        )
        moonshine_residual_fp16(
            device_residual.ptr,
            fc2_output.ptr,
            final_output.ptr,
            rows * hidden,
            library=glue,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = np.empty_like(expected)
        copy_device_to_host(host_array_ptr(actual), final_output, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    finite = bool(np.isfinite(actual).all())
    all_close = bool(np.allclose(actual, expected, rtol=5.0e-3, atol=5.0e-3))
    report = {
        "all_passed": bool(finite and all_close),
        "finite": finite,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "max_abs": float(np.max(difference)),
        "relative_l2": float(
            np.linalg.norm(difference.ravel())
            / max(np.linalg.norm(expected.astype(np.float32).ravel()), 1.0e-12)
        ),
        "expected_kernel_names": [
            "moonshine_f16_projection_bias_kernel",
            "moonshine_gated_silu_fp16_kernel",
            "moonshine_residual_fp16_kernel",
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
