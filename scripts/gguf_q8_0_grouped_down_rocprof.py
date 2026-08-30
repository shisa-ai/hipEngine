#!/usr/bin/env python3
"""rocprofv3 smoke for the P1 device-driven grouped Q8_0 down owner.

Runs ``gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out`` on a
layer-2-shaped grouped fixture (512 experts, ffn=640 -> hidden=2560) and, in
``--profile`` mode, wraps only a cached-only child in ``rocprofv3 --kernel-trace``
so no profiler-injected child spawns ``hipcc``/clang.

Usage (cache-only build):
  hipcc --version > /tmp/hipengine-hipcc-version.txt
  python3 scripts/gguf_q8_0_grouped_down_rocprof.py --n 128 --prebuild \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt
  rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-q8g-down -- \
      python3 scripts/gguf_q8_0_grouped_down_rocprof.py --n 128 \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached

The direct profiled child never mutates the shared build cache, so the
prebuilt artifact stays byte-identical. Raw profiler dumps stay out of Git.
"""

from __future__ import annotations

import argparse
import os

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
)
from tests.test_gguf_k_gemv import make_q8_0_weight

Q8_0_BLOCK_BYTES = 34


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128, help="compact grouped rows")
    parser.add_argument("--prebuild", action="store_true")
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--compiler-version-file", default=None)
    args = parser.parse_args()

    compiler_version = _build_compiler_version(args)
    # Warm/prebuild the JIT outside the profiler, or require the prebuilt
    # artifact in the cached-only child.
    build_gguf_q8_0_prefill(
        compiler_version=compiler_version,
        require_cached=args.require_cached,
    )
    if args.prebuild:
        return

    # Layer-2 frozen-MoE-map shape: ffn=640 in -> hidden=2560 out, 512 experts.
    num_experts, in_features, out_features = 512, 640, 2560
    compact_rows = args.n
    rng = np.random.default_rng(7)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(compact_rows, in_features)).astype(np.float32)
    )
    base = make_q8_0_weight(out_features, in_features)
    raw = np.stack(
        [np.roll(base, shift=e, axis=0) for e in range(num_experts)], axis=0
    )
    # Spread compact rows across the first experts; the rest are empty.
    n_active = min(num_experts, compact_rows)
    counts = [0] * num_experts
    for e in range(n_active):
        counts[e] = (compact_rows // n_active) + (1 if e < compact_rows % n_active else 0)
    expert_start = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))

    runtime = get_hip_runtime()
    library = build_gguf_q8_0_prefill(
        compiler_version=compiler_version,
        require_cached=args.require_cached,
    )
    host_out = np.zeros((compact_rows, out_features), dtype=np.uint16)
    bufs = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        start_dev = malloc(expert_start.nbytes, runtime=runtime)
        w_dev = malloc(raw.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((x_dev, start_dev, w_dev, out_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bits), x_bits.nbytes, runtime=runtime)
        copy_host_to_device(
            start_dev, host_array_ptr(expert_start), expert_start.nbytes, runtime=runtime
        )
        copy_host_to_device(w_dev, host_array_ptr(raw), raw.nbytes, runtime=runtime)
        gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out(
            x_dev.ptr,
            start_dev.ptr,
            w_dev.ptr,
            out_dev.ptr,
            compact_rows,
            num_experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, host_out.nbytes, runtime=runtime)
    finally:
        for buf in bufs:
            free(buf, runtime=runtime)
    total = float(np.asarray(host_out, dtype=np.int32).astype(np.int64).sum())
    print(
        f"grouped_q8_0_down compact_rows={compact_rows} experts={num_experts} "
        f"in={in_features} out={out_features} checksum={total:.3f}"
    )


if __name__ == "__main__":
    main()
