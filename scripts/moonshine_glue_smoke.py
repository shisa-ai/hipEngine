#!/usr/bin/env python3
"""Torch-free gfx11 Moonshine embedding/residual/RoPE/cache profile child."""

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
    moonshine_apply_partial_rope,
    moonshine_residual,
    moonshine_rope_tables,
)
from hipengine.kernels.hip_gfx1100.fused.moonshine_glue import (
    build_moonshine_glue,
    moonshine_embedding_lookup_fp16,
    moonshine_partial_rope_cache_append_fp16,
    moonshine_partial_rope_fp16,
    moonshine_residual_fp16,
    moonshine_self_cache_append_fp16,
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


def main() -> int:
    args = parse_args()
    compiler_version = args.compiler_version_file.read_text()
    library = build_moonshine_glue(
        compiler_version=compiler_version,
        load=not args.prebuild_only,
        require_cached=args.require_cached_build,
    )
    if args.prebuild_only:
        print(library.output_path)
        return 0

    rng = np.random.default_rng(0x61E)
    hidden, vocab, token = 416, 37, 23
    heads, head_dim, rotary_dim, capacity, position = 8, 52, 32, 194, 193
    embedding = rng.normal(0.0, 0.1, size=(vocab, hidden)).astype(np.float16)
    residual = rng.normal(0.0, 0.2, size=(hidden,)).astype(np.float16)
    delta = rng.normal(0.0, 0.2, size=(hidden,)).astype(np.float16)
    query = rng.normal(0.0, 0.1, size=(heads, head_dim)).astype(np.float16)
    key = rng.normal(0.0, 0.1, size=(heads, head_dim)).astype(np.float16)
    value = rng.normal(0.0, 0.1, size=(heads, head_dim)).astype(np.float16)
    cos, sin = moonshine_rope_tables(capacity, rotary_dim=rotary_dim)
    expected_query, expected_key = moonshine_apply_partial_rope(
        query[None, :, None, :],
        key[None, :, None, :],
        cos,
        sin,
        position_ids=np.asarray([position], dtype=np.int64),
        rotary_dim=rotary_dim,
    )
    expected_residual = moonshine_residual(residual, delta)

    runtime = get_hip_runtime()
    allocations = []
    try:
        device_embedding = _upload(embedding, runtime, allocations)
        device_token = _upload(np.asarray([token], dtype=np.int64), runtime, allocations)
        embedding_output = _empty((hidden,), runtime, allocations)
        device_residual = _upload(residual, runtime, allocations)
        device_delta = _upload(delta, runtime, allocations)
        residual_output = _empty((hidden,), runtime, allocations)
        device_query = _upload(query, runtime, allocations)
        device_key = _upload(key, runtime, allocations)
        device_value = _upload(value, runtime, allocations)
        device_cos = _upload(cos, runtime, allocations)
        device_sin = _upload(sin, runtime, allocations)
        device_position = _upload(np.asarray([position], dtype=np.int64), runtime, allocations)
        separate_query = _empty((heads, head_dim), runtime, allocations)
        separate_key = _empty((heads, head_dim), runtime, allocations)
        fused_query = _empty((heads, head_dim), runtime, allocations)
        fused_key = _empty((heads, head_dim), runtime, allocations)
        cache_shape = (heads, capacity, head_dim)
        zero_cache = np.zeros(cache_shape, dtype=np.float16)
        separate_k_cache = _upload(zero_cache, runtime, allocations)
        separate_v_cache = _upload(zero_cache, runtime, allocations)
        fused_k_cache = _upload(zero_cache, runtime, allocations)
        fused_v_cache = _upload(zero_cache, runtime, allocations)

        moonshine_embedding_lookup_fp16(
            device_embedding.ptr, device_token.ptr, embedding_output.ptr, hidden, vocab,
            library=library, runtime=runtime,
        )
        moonshine_residual_fp16(
            device_residual.ptr, device_delta.ptr, residual_output.ptr, hidden,
            library=library, runtime=runtime,
        )
        moonshine_partial_rope_fp16(
            device_query.ptr, device_key.ptr, device_cos.ptr, device_sin.ptr,
            device_position.ptr, separate_query.ptr, separate_key.ptr,
            heads, head_dim, rotary_dim, capacity, library=library, runtime=runtime,
        )
        moonshine_self_cache_append_fp16(
            separate_key.ptr, device_value.ptr, device_position.ptr,
            separate_k_cache.ptr, separate_v_cache.ptr, heads, head_dim, capacity,
            library=library, runtime=runtime,
        )
        moonshine_partial_rope_cache_append_fp16(
            device_query.ptr, device_key.ptr, device_value.ptr, device_cos.ptr,
            device_sin.ptr, device_position.ptr, fused_query.ptr, fused_key.ptr,
            fused_k_cache.ptr, fused_v_cache.ptr, heads, head_dim, rotary_dim,
            capacity, capacity, library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        actual_embedding = _download(embedding_output, (hidden,), runtime)
        actual_residual = _download(residual_output, (hidden,), runtime)
        actual_separate_query = _download(separate_query, (heads, head_dim), runtime)
        actual_separate_key = _download(separate_key, (heads, head_dim), runtime)
        actual_fused_query = _download(fused_query, (heads, head_dim), runtime)
        actual_fused_key = _download(fused_key, (heads, head_dim), runtime)
        actual_separate_k = _download(separate_k_cache, cache_shape, runtime)
        actual_separate_v = _download(separate_v_cache, cache_shape, runtime)
        actual_fused_k = _download(fused_k_cache, cache_shape, runtime)
        actual_fused_v = _download(fused_v_cache, cache_shape, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    composite_exact = all(
        np.array_equal(left, right)
        for left, right in (
            (actual_separate_query, actual_fused_query),
            (actual_separate_key, actual_fused_key),
            (actual_separate_k, actual_fused_k),
            (actual_separate_v, actual_fused_v),
        )
    )
    expected_q = expected_query.reshape(heads, head_dim)
    expected_k = expected_key.reshape(heads, head_dim)
    max_abs = max(
        float(np.max(np.abs(actual_fused_query.astype(np.float32) - expected_q.astype(np.float32)))),
        float(np.max(np.abs(actual_fused_key.astype(np.float32) - expected_k.astype(np.float32)))),
    )
    all_passed = bool(
        np.array_equal(actual_embedding, embedding[token])
        and np.array_equal(actual_residual, expected_residual)
        and composite_exact
        and np.allclose(actual_fused_query, expected_q, rtol=1.0e-3, atol=1.0e-3)
        and np.allclose(actual_fused_key, expected_k, rtol=1.0e-3, atol=1.0e-3)
        and all(
            np.isfinite(value).all()
            for value in (actual_embedding, actual_residual, actual_fused_query, actual_fused_key)
        )
    )
    report = {
        "all_passed": all_passed,
        "composite_exact_to_unfused": composite_exact,
        "max_abs_rope_vs_cpu": max_abs,
        "position": position,
        "expected_kernel_names": [
            "moonshine_embedding_lookup_fp16_kernel",
            "moonshine_residual_fp16_kernel",
            "moonshine_partial_rope_fp16_kernel",
            "moonshine_self_cache_append_fp16_kernel",
            "moonshine_partial_rope_cache_append_fp16_kernel",
        ],
    }
    text = json.dumps(report, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
