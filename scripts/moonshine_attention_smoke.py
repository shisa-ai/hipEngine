#!/usr/bin/env python3
"""Torch-free gfx11 Moonshine self/cross attention correctness/profile child."""

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
from hipengine.kernels.cpu_reference.moonshine import moonshine_attention
from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
    build_moonshine_attention,
    moonshine_cross_attention_fp16,
    moonshine_cross_attention_grouped_fp16,
    moonshine_cross_attention_parallel_fp16,
    moonshine_self_attention_branch_fp16,
    moonshine_self_attention_fp16,
    moonshine_self_attention_parallel_fp16,
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


def _download(device, shape: tuple[int, ...], runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    relative_l2 = float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(expected.astype(np.float32).ravel()), 1.0e-12)
    )
    return float(np.max(difference)), relative_l2


def main() -> int:
    args = parse_args()
    library = build_moonshine_attention(
        compiler_version=args.compiler_version_file.read_text(),
        load=not args.prebuild_only,
        require_cached=args.require_cached_build,
    )
    if args.prebuild_only:
        print(library.output_path)
        return 0

    rng = np.random.default_rng(0xA77E)
    heads, head_dim, capacity, position = 8, 52, 194, 193
    encoder_length = 1248
    query_self = rng.normal(0.0, 0.08, size=(1, heads, 1, head_dim)).astype(np.float16)
    key_self = rng.normal(
        0.0, 0.08, size=(1, heads, capacity, head_dim)
    ).astype(np.float16)
    value_self = rng.normal(
        0.0, 0.10, size=(1, heads, capacity, head_dim)
    ).astype(np.float16)
    query_cross = rng.normal(0.0, 0.08, size=(1, heads, 1, head_dim)).astype(np.float16)
    key_cross = rng.normal(
        0.0, 0.08, size=(1, heads, encoder_length, head_dim)
    ).astype(np.float16)
    value_cross = rng.normal(
        0.0, 0.10, size=(1, heads, encoder_length, head_dim)
    ).astype(np.float16)
    mask = np.ones((1, encoder_length), dtype=np.int32)
    mask[:, -173:] = 0
    expected_self = moonshine_attention(query_self, key_self, value_self)
    expected_cross = moonshine_attention(
        query_cross, key_cross, value_cross, mask=mask
    )

    runtime = get_hip_runtime()
    allocations = []
    try:
        device_query_self = _upload(query_self, runtime, allocations)
        device_key_self = _upload(key_self, runtime, allocations)
        device_value_self = _upload(value_self, runtime, allocations)
        device_position = _upload(
            np.asarray([position], dtype=np.int64), runtime, allocations
        )
        output_self = _empty(expected_self.shape, runtime, allocations)
        device_query_cross = _upload(query_cross, runtime, allocations)
        device_key_cross = _upload(key_cross, runtime, allocations)
        device_value_cross = _upload(value_cross, runtime, allocations)
        device_mask = _upload(mask, runtime, allocations)
        output_cross = _empty(expected_cross.shape, runtime, allocations)
        moonshine_self_attention_fp16(
            device_query_self.ptr,
            device_key_self.ptr,
            device_value_self.ptr,
            device_position.ptr,
            output_self.ptr,
            heads,
            head_dim,
            capacity,
            library=library,
            runtime=runtime,
        )
        moonshine_self_attention_branch_fp16(
            device_query_self.ptr,
            device_key_self.ptr,
            device_value_self.ptr,
            device_position.ptr,
            output_self.ptr,
            heads,
            head_dim,
            capacity,
            library=library,
            runtime=runtime,
        )
        for threads in (64, 128, 256):
            moonshine_self_attention_parallel_fp16(
                device_query_self.ptr,
                device_key_self.ptr,
                device_value_self.ptr,
                device_position.ptr,
                output_self.ptr,
                heads,
                head_dim,
                capacity,
                threads=threads,
                library=library,
                runtime=runtime,
            )
        moonshine_cross_attention_fp16(
            device_query_cross.ptr,
            device_key_cross.ptr,
            device_value_cross.ptr,
            device_mask.ptr,
            output_cross.ptr,
            heads,
            head_dim,
            encoder_length,
            library=library,
            runtime=runtime,
        )
        moonshine_cross_attention_grouped_fp16(
            device_query_cross.ptr,
            device_key_cross.ptr,
            device_value_cross.ptr,
            device_mask.ptr,
            output_cross.ptr,
            heads,
            head_dim,
            encoder_length,
            library=library,
            runtime=runtime,
        )
        for threads in (64, 128, 256):
            moonshine_cross_attention_parallel_fp16(
                device_query_cross.ptr,
                device_key_cross.ptr,
                device_value_cross.ptr,
                device_mask.ptr,
                output_cross.ptr,
                heads,
                head_dim,
                encoder_length,
                threads=threads,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        actual_self = _download(output_self, expected_self.shape, runtime)
        actual_cross = _download(output_cross, expected_cross.shape, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    self_max_abs, self_relative_l2 = _error(actual_self, expected_self)
    cross_max_abs, cross_relative_l2 = _error(actual_cross, expected_cross)
    finite = bool(np.isfinite(actual_self).all() and np.isfinite(actual_cross).all())
    all_close = bool(
        np.allclose(actual_self, expected_self, rtol=5.0e-3, atol=5.0e-3)
        and np.allclose(actual_cross, expected_cross, rtol=5.0e-3, atol=5.0e-3)
    )
    report = {
        "all_passed": bool(finite and all_close),
        "cross_encoder_length": encoder_length,
        "cross_max_abs": cross_max_abs,
        "cross_relative_l2": cross_relative_l2,
        "expected_kernel_names": [
            "moonshine_self_attention_fp16_kernel",
            "moonshine_self_attention_branch_fp16_kernel",
            "moonshine_self_attention_parallel_fp16_kernel<2>",
            "moonshine_self_attention_parallel_fp16_kernel<4>",
            "moonshine_self_attention_parallel_fp16_kernel<8>",
            "moonshine_cross_attention_fp16_kernel",
            "moonshine_cross_attention_grouped_fp16_kernel",
            "moonshine_cross_attention_parallel_fp16_kernel<2>",
            "moonshine_cross_attention_parallel_fp16_kernel<4>",
            "moonshine_cross_attention_parallel_fp16_kernel<8>",
        ],
        "finite": finite,
        "head_dim": head_dim,
        "heads": heads,
        "self_max_abs": self_max_abs,
        "self_past_length": position,
        "self_relative_l2": self_relative_l2,
    }
    text = json.dumps(report, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
