#!/usr/bin/env python3
"""Per-shape CUDA sm_120a Moonshine gated-MLP boundary profiler.

Batch-timed CUDA-event medians for the standalone gated-SiLU and the fused
fc1/fc2 boundaries. Rows are reported per-shape: M=1 (the production decoder
row count) is listed first and explicitly, and auxiliary rows are listed
separately. No aggregate across unlike row counts is reported. Requires
exclusive GPU0; a pre/post compute-process gate is printed.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cuda_sm120a.fused.moonshine_mlp import (
    build_moonshine_mlp,
    moonshine_gated_silu_fp16,
)
from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
    build_moonshine_projection,
    moonshine_f16_projection_bias_gated_silu,
    moonshine_f16_projection_bias_residual,
)

_rng = np.random.default_rng(0x0C1D)
_HIDDEN = 416
_INTERMEDIATE = 1664
# Production decoder row count first; auxiliary rows listed separately.
_ROW_BUCKETS = (1, 7, 40)
_ITERS = 12
_PER_BATCH = 500


def _gate(label: str) -> None:
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    self_pid = str(os.getpid())
    foreign = [row for row in out.splitlines() if row and not row.startswith(self_pid)]
    print(f"[gate {label}] foreign compute apps on GPU0: {foreign or '(none)'}")
    if foreign:
        raise SystemExit(f"GPU0 not exclusive at {label}; aborting")


def _upload(array: np.ndarray, runtime, allocations) -> DeviceBuffer:
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape: tuple[int, ...], runtime, allocations) -> DeviceBuffer:
    nbytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    device = malloc(nbytes, runtime=runtime)
    allocations.append(device)
    return device


def _bench(fn, runtime) -> float:
    for _ in range(50):
        fn()
    runtime.device_synchronize()
    samples = []
    for _ in range(_ITERS):
        start = runtime.event_create()
        end = runtime.event_create()
        runtime.event_record(start)
        for _ in range(_PER_BATCH):
            fn()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        total_us = runtime.event_elapsed_time_ms(start, end) * 1e3
        samples.append(total_us / _PER_BATCH)
        runtime.event_destroy(start)
        runtime.event_destroy(end)
    return float(np.median(samples))


def main() -> None:
    _gate("pre")
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    mlp_lib = build_moonshine_mlp(load=True)
    proj_lib = build_moonshine_projection(load=True)
    allocations: list[DeviceBuffer] = []
    try:
        fc1_out = {}
        for rows in _ROW_BUCKETS:
            fc1 = _upload(
                _rng.normal(0.0, 0.05, size=(rows, 2 * _INTERMEDIATE)).astype(
                    np.float16
                ),
                runtime,
                allocations,
            )
            fc1_out[rows] = fc1

        silu_out = {
            rows: _alloc((rows, _INTERMEDIATE), runtime, allocations)
            for rows in _ROW_BUCKETS
        }

        print("=== standalone gated-SiLU (1664-input): us/launch (median, M=1 first) ===")
        silu_times = {}
        for rows in _ROW_BUCKETS:
            silu_times[rows] = _bench(
                lambda rows=rows: moonshine_gated_silu_fp16(
                    fc1_out[rows].ptr,
                    silu_out[rows].ptr,
                    rows,
                    _INTERMEDIATE,
                    library=mlp_lib,
                    runtime=runtime,
                ),
                runtime,
            )
            label = "M=1 (decoder)" if rows == 1 else f"M={rows} (aux)"
            print(f"  rows {rows:>4}  {silu_times[rows]:8.3f} us   [{label}]")
        print(f"  (M=1 only: {silu_times[1]:.3f} us; no cross-row aggregate)")

        # Fused fc1 (bias + gated-SiLU, fixed 32 threads) 416 -> 1664.
        fc1_weight = _upload(
            _rng.normal(0.0, 0.04, size=(2 * _INTERMEDIATE, _HIDDEN)).astype(
                np.float16
            ),
            runtime,
            allocations,
        )
        fc1_bias = _upload(
            _rng.normal(0.0, 0.03, size=(2 * _INTERMEDIATE,)).astype(np.float16),
            runtime,
            allocations,
        )
        x = {
            rows: _upload(
                _rng.normal(0.0, 0.05, size=(rows, _HIDDEN)).astype(np.float16),
                runtime,
                allocations,
            )
            for rows in _ROW_BUCKETS
        }
        fused_fc1_out = {
            rows: _alloc((rows, _INTERMEDIATE), runtime, allocations)
            for rows in _ROW_BUCKETS
        }

        print("\n=== fused fc1 (bias_gated_silu, t32) 416->1664: us/launch ===")
        fc1_times = {}
        for rows in _ROW_BUCKETS:
            fc1_times[rows] = _bench(
                lambda rows=rows: moonshine_f16_projection_bias_gated_silu(
                    x[rows].ptr,
                    fc1_weight.ptr,
                    fc1_bias.ptr,
                    fused_fc1_out[rows].ptr,
                    rows,
                    _HIDDEN,
                    _INTERMEDIATE,
                    library=proj_lib,
                    runtime=runtime,
                ),
                runtime,
            )
            label = "M=1 (decoder)" if rows == 1 else f"M={rows} (aux)"
            print(f"  rows {rows:>4}  {fc1_times[rows]:8.3f} us   [{label}]")

        # Fused fc2 (bias_residual, auto-select 256 at M=1) 1664 -> 416.
        fc2_weight = _upload(
            _rng.normal(0.0, 0.04, size=(_HIDDEN, _INTERMEDIATE)).astype(np.float16),
            runtime,
            allocations,
        )
        fc2_bias = _upload(
            _rng.normal(0.0, 0.03, size=(_HIDDEN,)).astype(np.float16),
            runtime,
            allocations,
        )
        residual = {
            rows: _upload(
                _rng.normal(0.0, 0.08, size=(rows, _HIDDEN)).astype(np.float16),
                runtime,
                allocations,
            )
            for rows in _ROW_BUCKETS
        }
        fused_fc2_out = {
            rows: _alloc((rows, _HIDDEN), runtime, allocations)
            for rows in _ROW_BUCKETS
        }

        print("\n=== fused fc2 (bias_residual, auto-select) 1664->416: us/launch ===")
        fc2_times = {}
        for rows in _ROW_BUCKETS:
            fc2_times[rows] = _bench(
                lambda rows=rows: moonshine_f16_projection_bias_residual(
                    silu_out[rows].ptr,
                    fc2_weight.ptr,
                    fc2_bias.ptr,
                    residual[rows].ptr,
                    fused_fc2_out[rows].ptr,
                    rows,
                    _INTERMEDIATE,
                    _HIDDEN,
                    library=proj_lib,
                    runtime=runtime,
                ),
                runtime,
            )
            label = "M=1 (decoder)" if rows == 1 else f"M={rows} (aux)"
            print(f"  rows {rows:>4}  {fc2_times[rows]:8.3f} us   [{label}]")

        fused_fc1_fc2 = fc1_times[1] + fc2_times[1]
        print(
            f"\n  fused fc1+fc2 at M=1: {fc1_times[1]:.3f} + {fc2_times[1]:.3f} "
            f"= {fused_fc1_fc2:.3f} us (per-boundary sum, leaf only)"
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        _gate("post")


if __name__ == "__main__":
    main()
