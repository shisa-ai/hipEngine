#!/usr/bin/env python3
"""Microbench: layer-2 Q5_K gate/up grouped WMMA body vs strict selected gemv.

The dominant layer-2 prefill term (~397.95 ms) is the Q5_K dual gate/up. The
grouped prefill path can route Q5_K/Q5_K gate/up through the existing selected
Q5_K WMMA body (`gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out`, fp16
WMMA, production-class) while the strict per-expert selected Q5_K gemv is the
current default. This compares the two on the layer-2 gate shape
(hidden=2560 in -> ffn=640 out, 512 experts) across active-row profiles.

Diagnostic only; wall-clock local microbench, medians of n repeats. The WMMA
body changes arithmetic (fp16 WMMA), so any promotion still needs the full P1
production gate; this only tells us whether it is worth pursuing.

Usage:
  hipcc --version > /tmp/hipengine-hipcc-version.txt
  python3 scripts/gguf_q5_k_gate_up_microbench.py --repeats 15 \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q5_k_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_selected_prefill import (
    build_gguf_k_selected_prefill,
    gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out,
)
from tests._gguf_synthetic_weights import make_q5_k_weight


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
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--compiler-version-file", default=None)
    args = parser.parse_args()

    # Layer-2 gate shape: hidden=2560 in -> ffn=640 out, 512 experts.
    num_experts, in_features, out_features = 512, 2560, 640
    compiler_version = _build_compiler_version(args)
    build_gguf_k_selected_prefill(compiler_version=compiler_version)
    build_gguf_k_gemv(compiler_version=compiler_version)
    runtime = get_hip_runtime()
    wmma_lib = build_gguf_k_selected_prefill(compiler_version=compiler_version)
    gemv_lib = build_gguf_k_gemv(compiler_version=compiler_version)

    rng = np.random.default_rng(11)
    base = make_q5_k_weight(out_features, in_features)
    raw = np.ascontiguousarray(np.stack(
        [np.roll(base, shift=e, axis=0) for e in range(num_experts)], axis=0
    ))

    profiles = [
        ("sparse", 16, [3, 0, 5, 0, 2, 0, 0, 4, 1, 0, 2, 0, 6, 0, 0, 1] + [0] * 496),
        ("dense", 64, None),
        ("deep", 256, None),
        ("wide", 1024, None),
    ]

    bufs = []
    try:
        w_dev = malloc(raw.nbytes, runtime=runtime)
        bufs.append(w_dev)
        copy_host_to_device(w_dev, host_array_ptr(raw), raw.nbytes, runtime=runtime)

        print(
            f"{'profile':8} {'rows':>5} {'strict_ms':>10} {'wmma_ms':>9} "
            f"{'strict/wmma':>11}"
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
            # WMMA grid: pad each expert's row count up to a 16-row tile.
            padded = [((c + 15) // 16) * 16 for c in counts]
            wmma_start = np.zeros(num_experts + 1, dtype=np.int64)
            wmma_start[1:] = np.cumsum(np.asarray(padded, dtype=np.int64))
            wmma_total = int(wmma_start[-1])
            tile_expert = np.concatenate([
                np.full(v // 16, e, dtype=np.int64)
                for e, v in enumerate(padded)
            ])

            x_dev = malloc(x_bits.nbytes, runtime=runtime)
            start_dev = malloc(expert_start.nbytes, runtime=runtime)
            wmma_start_dev = malloc(wmma_start.nbytes, runtime=runtime)
            tile_dev = malloc(tile_expert.nbytes, runtime=runtime)
            out_dev = malloc(compact_rows * out_features * 2, runtime=runtime)
            selected_dev = malloc(compact_rows * 8, runtime=runtime)
            bufs.extend((x_dev, start_dev, wmma_start_dev, tile_dev, out_dev, selected_dev))

            sorted_experts = np.empty(compact_rows, dtype=np.int64)
            for e in range(num_experts):
                s = int(expert_start[e])
                er = int(expert_start[e + 1]) - s
                sorted_experts[s : s + er] = e
            copy_host_to_device(selected_dev, host_array_ptr(sorted_experts),
                                sorted_experts.nbytes, runtime=runtime)
            copy_host_to_device(x_dev, host_array_ptr(x_bits), x_bits.nbytes, runtime=runtime)
            copy_host_to_device(start_dev, host_array_ptr(expert_start),
                                expert_start.nbytes, runtime=runtime)
            copy_host_to_device(wmma_start_dev, host_array_ptr(wmma_start),
                                wmma_start.nbytes, runtime=runtime)
            copy_host_to_device(tile_dev, host_array_ptr(tile_expert),
                                tile_expert.nbytes, runtime=runtime)

            def run_strict():
                gguf_q5_k_selected_gemv_bf16_bf16_out(
                    x_dev.ptr, selected_dev.ptr, w_dev.ptr, out_dev.ptr,
                    compact_rows, compact_rows, num_experts, in_features,
                    out_features, library=gemv_lib, runtime=runtime,
                )

            def run_wmma():
                gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out(
                    x_dev.ptr, start_dev.ptr, wmma_start_dev.ptr, tile_dev.ptr,
                    w_dev.ptr, out_dev.ptr, compact_rows, in_features,
                    out_features, num_experts, wmma_total,
                    library=wmma_lib, runtime=runtime,
                )

            strict_times, wmma_times = [], []
            for _ in range(3):
                runtime.device_synchronize()
                t0 = time.perf_counter(); run_strict(); runtime.device_synchronize()
                strict_times.append((time.perf_counter() - t0) * 1e3)
            for _ in range(args.repeats):
                runtime.device_synchronize()
                t0 = time.perf_counter(); run_strict(); runtime.device_synchronize()
                strict_times.append((time.perf_counter() - t0) * 1e3)
            for _ in range(3):
                runtime.device_synchronize()
                t0 = time.perf_counter(); run_wmma(); runtime.device_synchronize()
                wmma_times.append((time.perf_counter() - t0) * 1e3)
            for _ in range(args.repeats):
                runtime.device_synchronize()
                t0 = time.perf_counter(); run_wmma(); runtime.device_synchronize()
                wmma_times.append((time.perf_counter() - t0) * 1e3)

            strict_ms = _median(strict_times)
            wmma_ms = _median(wmma_times)
            print(
                f"{name:8} {compact_rows:>5} {strict_ms:>10.3f} {wmma_ms:>9.3f} "
                f"{strict_ms / wmma_ms:>11.2f}"
            )
    finally:
        for b in bufs:
            free(b, runtime=runtime)


if __name__ == "__main__":
    main()
