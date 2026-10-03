"""Time the Q4_K x DS4-Q8_1 MMQ32 leaf at a fixed geometry, narrow K pass vs wide.

Allocation, packing and host-device copies happen once, outside the timed loop, so
the number reported is the kernel's own launch. Both routes are given the same
buffers and the same weights, and the wide route was checked bitwise against the
narrow one by tests/test_gpu_gguf_q4_k_q8_1_selected_prefill.py.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    build_gguf_q4_k_q8_1_selected_prefill,
    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out as narrow,
    gguf_q4_k_selected_dual_q8_1_ds4_mmq32x128_prefill_compact32_bf16_bf16_out as wide,
)
from hipengine.quant.gguf_q4_k import pack_q8_1_mmq_ds4_from_bf16
from tests.test_gpu_gguf_q4_k_q8_1_selected_prefill import _mmq32_metadata
from tests.test_gpu_gguf_q4_k_selected_wmma_prefill import _build_compact_fixture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=lambda s: [int(v) for v in s.split(",")],
                        default=[32, 32, 32])
    parser.add_argument("--in-features", type=int, default=2816)
    parser.add_argument("--out-features-a", type=int, default=1408)
    parser.add_argument("--out-features-b", type=int, default=1408)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--burst", type=int, default=5)
    args = parser.parse_args()

    runtime = get_hip_runtime()
    build_gguf_q4_k_q8_1_selected_prefill(load=True)
    fixture = _build_compact_fixture(
        counts=args.counts,
        in_features=args.in_features,
        out_features_a=args.out_features_a,
        out_features_b=args.out_features_b,
        dtype="bf16",
        seed=23,
    )
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=np.uint16,
    )
    compact_to_source = np.arange(fixture.compact_rows, dtype=np.int64)
    q8_ds4 = pack_q8_1_mmq_ds4_from_bf16(fixture.x_host)
    expert_start_mmq32, tile_expert, mmq_total_rows = _mmq32_metadata(fixture)

    bufs = []
    try:
        source_x_dev = malloc(fixture.x_host.nbytes, runtime=runtime)
        q8_dev = malloc(q8_ds4.nbytes, runtime=runtime)
        c2s_dev = malloc(compact_to_source.nbytes, runtime=runtime)
        start_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_mmq_dev = malloc(expert_start_mmq32.nbytes, runtime=runtime)
        tile_dev = malloc(tile_expert.nbytes, runtime=runtime)
        qwa_dev = malloc(fixture.qweight_a.nbytes, runtime=runtime)
        qwb_dev = malloc(fixture.qweight_b.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((source_x_dev, q8_dev, c2s_dev, start_dev, start_mmq_dev,
                     tile_dev, qwa_dev, qwb_dev, out_dev))
        for dev, arr in (
            (source_x_dev, fixture.x_host),
            (q8_dev, q8_ds4),
            (c2s_dev, compact_to_source),
            (start_dev, fixture.expert_start_compact),
            (start_mmq_dev, expert_start_mmq32),
            (tile_dev, tile_expert),
            (qwa_dev, fixture.qweight_a),
            (qwb_dev, fixture.qweight_b),
        ):
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(arr)),
                                runtime=runtime)

        common = dict(
            x_q8_ptr=q8_dev.ptr,
            compact_to_source_ptr=c2s_dev.ptr,
            expert_start_compact_ptr=start_dev.ptr,
            expert_start_mmq32_ptr=start_mmq_dev.ptr,
            mmq_tile_expert_ptr=tile_dev.ptr,
            qweight_a_ptr=qwa_dev.ptr,
            qweight_b_ptr=qwb_dev.ptr,
            out_ptr=out_dev.ptr,
            compact_rows=fixture.compact_rows,
            in_features=args.in_features,
            out_features_a=fixture.out_features_a,
            out_features_b=fixture.out_features_b,
            num_experts=len(args.counts),
            mmq_total_rows=mmq_total_rows,
        )

        def burst(fn) -> float:
            for _ in range(args.burst):
                fn(**common)
            runtime.device_synchronize()
            t0 = time.perf_counter()
            for _ in range(args.burst):
                fn(**common)
            runtime.device_synchronize()
            return (time.perf_counter() - t0) / args.burst

        results = {}
        for name, fn in (("mmq32 (K=32)", narrow), ("mmq32x128 (K=128)", wide)):
            for _ in range(args.warmups):
                burst(fn)
            samples = [burst(fn) for _ in range(args.samples)]
            results[name] = (statistics.median(samples), min(samples))
            print(f"{name:20s} median {results[name][0] * 1e6:9.1f} us   "
                  f"min {results[name][1] * 1e6:9.1f} us")

        # macs = rows x (out_a + out_b) x in_features
        macs = (fixture.compact_rows
                * (fixture.out_features_a + fixture.out_features_b)
                * args.in_features)
        print()
        for name, (med, _) in results.items():
            print(f"{name:20s} {macs / med / 1e12:6.2f} TMAC/s")

        a, b = results["mmq32 (K=32)"][0], results["mmq32x128 (K=128)"][0]
        print(f"\nwide/narrow: {b / a:.3f}x  ({'faster' if b < a else 'slower'})")

        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in bufs:
            free(buf, runtime=runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
