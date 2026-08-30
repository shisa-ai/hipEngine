#!/usr/bin/env python3
"""Microbench: P1 device-driven grouped Q8_0 down vs the legacy D2H+WMMA loop.

The legacy Q8 path (before the P1 owner) copied ``group_expert_start`` to host
and launched one ``gguf_q8_0_wmma_prefill_bf16_bf16_out`` per non-empty expert
over 512 experts. The P1 owner replaces that with a single device-driven
launch (``q8_0_selected_grouped_prefill_bf16_bf16_kernel``) that reads
``expert_start`` on device.

This compares the two submission paths on the real layer-2 frozen-MoE-map shape
(ffn=640 in -> hidden=2560 out, 512 experts) across a few active-row profiles.

Benchmark only; no retained perf claim — wall-clock microbench on local
hardware, run n-repeats and report medians + the legacy/new ratio.

Usage:
  hipcc --version > /tmp/hipengine-hipcc-version.txt
  python3 scripts/gguf_q8_0_grouped_down_microbench.py --repeats 20 \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    build_gguf_q8_0_prefill,
    gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out,
    gguf_q8_0_wmma_prefill_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_selected_gemv_bf16_bf16_out,
)
from tests.test_gguf_k_gemv import make_q8_0_weight


def _f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _build_compiler_version(args) -> str | None:
    if args.compiler_version_file:
        with open(args.compiler_version_file) as handle:
            return handle.read().strip()
    return None


def _median(times: list[float]) -> float:
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compiler-version-file", default=None)
    args = parser.parse_args()

    num_experts, in_features, out_features = 512, 640, 2560
    compiler_version = _build_compiler_version(args)
    build_gguf_q8_0_prefill(compiler_version=compiler_version)
    build_gguf_k_gemv(compiler_version=compiler_version)
    runtime = get_hip_runtime()
    library = build_gguf_q8_0_prefill(compiler_version=compiler_version)
    gemv_library = build_gguf_k_gemv(compiler_version=compiler_version)

    rng = np.random.default_rng(7)
    base = make_q8_0_weight(out_features, in_features)
    raw = np.stack(
        [np.roll(base, shift=e, axis=0) for e in range(num_experts)], axis=0
    )
    raw = np.ascontiguousarray(raw)
    expert_weight_bytes = out_features * (in_features // 32) * 34

    profiles = [
        ("sparse", 16, [3, 0, 5, 0, 2, 0, 0, 4, 1, 0, 2, 0, 6, 0, 0, 1] + [0] * 496),
        ("dense", 64, None),
        ("deep", 256, None),
    ]

    bufs = []
    try:
        w_dev = malloc(raw.nbytes, runtime=runtime)
        bufs.append(w_dev)
        copy_host_to_device(w_dev, host_array_ptr(raw), raw.nbytes, runtime=runtime)
        copy_host_to_device(
            w_dev, host_array_ptr(raw), raw.nbytes, runtime=runtime
        )  # warm the copy path

        print(
            f"{'profile':8} {'rows':>5} {'legacy_ms':>10} {'grouped_ms':>11} "
            f"{'strict_ms':>10} {'g/legacy':>8} {'g/strict':>8}"
        )
        for name, compact_rows, counts in profiles:
            x_bits = _f32_to_bf16_bits(
                rng.normal(0.0, 0.2, size=(compact_rows, in_features)).astype(np.float32)
            )
            x_bits = np.ascontiguousarray(x_bits)
            if counts is None:
                n_active = min(num_experts, compact_rows)
                counts = [0] * num_experts
                for e in range(n_active):
                    counts[e] = (compact_rows // n_active) + (
                        1 if e < compact_rows % n_active else 0
                    )
            expert_start = np.zeros(num_experts + 1, dtype=np.int64)
            expert_start[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))

            x_dev = malloc(x_bits.nbytes, runtime=runtime)
            start_dev = malloc(expert_start.nbytes, runtime=runtime)
            out_dev = malloc(compact_rows * out_features * 2, runtime=runtime)
            selected_dev = malloc(compact_rows * 8, runtime=runtime)
            bufs.extend((x_dev, start_dev, out_dev, selected_dev))
            # Build per-row expert id for the strict selected gemv.
            sorted_experts = np.empty(compact_rows, dtype=np.int64)
            for e in range(num_experts):
                s = int(expert_start[e])
                er = int(expert_start[e + 1]) - s
                sorted_experts[s : s + er] = e
            copy_host_to_device(
                selected_dev, host_array_ptr(sorted_experts),
                sorted_experts.nbytes, runtime=runtime,
            )
            copy_host_to_device(x_dev, host_array_ptr(x_bits), x_bits.nbytes, runtime=runtime)
            copy_host_to_device(
                start_dev, host_array_ptr(expert_start), expert_start.nbytes, runtime=runtime
            )

            # Legacy: D2H expert_start + Python loop launching WMMA per expert.
            def run_legacy():
                host_start = np.empty(num_experts + 1, dtype=np.int64)
                copy_device_to_host(
                    host_array_ptr(host_start), start_dev, host_start.nbytes, runtime=runtime
                )
                for e in range(num_experts):
                    s = int(host_start[e])
                    er = int(host_start[e + 1]) - s
                    if er <= 0:
                        continue
                    gguf_q8_0_wmma_prefill_bf16_bf16_out(
                        x_dev.ptr + s * in_features * 2,
                        w_dev.ptr + e * expert_weight_bytes,
                        out_dev.ptr + s * out_features * 2,
                        er, in_features, out_features,
                        stream=0, runtime=runtime, library=library,
                    )

            def run_grouped():
                gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out(
                    x_dev.ptr, start_dev.ptr, w_dev.ptr, out_dev.ptr,
                    compact_rows, num_experts, in_features, out_features,
                    library=library, runtime=runtime,
                )

            def run_strict():
                gguf_q8_0_selected_gemv_bf16_bf16_out(
                    x_dev.ptr, selected_dev.ptr, w_dev.ptr, out_dev.ptr,
                    compact_rows, compact_rows, num_experts, in_features,
                    out_features, library=gemv_library, runtime=runtime,
                )

            # Warmup + measure.
            legacy_times, grouped_times, strict_times = [], [], []
            for _ in range(3):
                runtime.device_synchronize()
                t0 = time.perf_counter()
                run_legacy()
                runtime.device_synchronize()
                legacy_times.append((time.perf_counter() - t0) * 1e3)
            for _ in range(args.repeats):
                runtime.device_synchronize()
                t0 = time.perf_counter()
                run_legacy()
                runtime.device_synchronize()
                legacy_times.append((time.perf_counter() - t0) * 1e3)
            for _ in range(3):
                runtime.device_synchronize()
                t0 = time.perf_counter()
                run_grouped()
                runtime.device_synchronize()
                grouped_times.append((time.perf_counter() - t0) * 1e3)
            for _ in range(args.repeats):
                runtime.device_synchronize()
                t0 = time.perf_counter()
                run_grouped()
                runtime.device_synchronize()
                grouped_times.append((time.perf_counter() - t0) * 1e3)
            for _ in range(3):
                runtime.device_synchronize()
                t0 = time.perf_counter()
                run_strict()
                runtime.device_synchronize()
                strict_times.append((time.perf_counter() - t0) * 1e3)
            for _ in range(args.repeats):
                runtime.device_synchronize()
                t0 = time.perf_counter()
                run_strict()
                runtime.device_synchronize()
                strict_times.append((time.perf_counter() - t0) * 1e3)

            legacy_ms = _median(legacy_times)
            grouped_ms = _median(grouped_times)
            strict_ms = _median(strict_times)
            print(
                f"{name:8} {compact_rows:>5} {legacy_ms:>10.3f} {grouped_ms:>11.3f} "
                f"{strict_ms:>10.3f} {legacy_ms / grouped_ms:>8.2f} "
                f"{strict_ms / grouped_ms:>8.2f}"
            )
    finally:
        for b in bufs:
            free(b, runtime=runtime)


if __name__ == "__main__":
    main()
