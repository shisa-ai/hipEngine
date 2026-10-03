#!/usr/bin/env python3
"""Time the YuE2 NAR attention kernel at production shapes, and check its bits.

The full replay takes a minute per measurement and mixes in the decoder, which is
too slow to iterate a kernel against. This drives ``nar_attention_f32`` directly
on synthetic tensors at the production geometry (one song is a single chunk, so
the shapes are the real ones), reports the median call time and the implied
bandwidth, and optionally re-checks the recorded parent bits so a speed change
cannot quietly break correctness.

Usage:
    python3 scripts/yue2_attention_bench.py                       # production shape
    python3 scripts/yue2_attention_bench.py --rows 1299 --ar-rows 1396 --repeat 20
    python3 scripts/yue2_attention_bench.py --fixture-check
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GOLDEN = REPO / "tests/fixtures/yue2/operators/nar_attention_parent.npz"


def _has_hip() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1299)
    parser.add_argument("--ar-rows", type=int, default=1396)
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--kernel", choices=("scalar", "wmma"), default="scalar",
                        help="which attention kernel to time")
    parser.add_argument("--fixture-check", action="store_true",
                        help="also assert bit parity against the recorded parent output")
    args = parser.parse_args()

    if not _has_hip():
        raise SystemExit("ROCm/HIP runtime is not available")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    runtime = get_hip_runtime()
    rng = np.random.default_rng(args.seed)

    def bf16(values):
        wide = np.asarray(values, dtype=np.float32)
        bits = wide.view(np.uint32).astype(np.uint64)
        rounded = ((bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1)))
                   >> np.uint64(16))
        return (rounded.astype(np.uint32) & np.uint32(0xFFFF)).astype(np.uint16)

    def upload(array):
        host = np.ascontiguousarray(array)
        buffer = malloc(max(host.nbytes, 8))
        copy_host_array_to_device(buffer, host)
        return buffer

    rows, ar_rows = args.rows, args.ar_rows
    heads, kv_heads, dim = args.num_q_heads, args.num_kv_heads, args.head_dim
    scale = float(1.0 / np.sqrt(dim))
    q = (rng.standard_normal((rows, heads, dim)) * 0.5).astype(np.float32)
    nar_k = bf16(rng.standard_normal((rows, kv_heads, dim)))
    nar_v = bf16(rng.standard_normal((rows, kv_heads, dim)))
    ar_k = bf16(rng.standard_normal((ar_rows, kv_heads, dim)))
    ar_v = bf16(rng.standard_normal((ar_rows, kv_heads, dim)))
    out = np.zeros((rows, heads, dim), dtype=np.float32)

    buffers = [upload(a) for a in (q, nar_k, nar_v, ar_k, ar_v, out)]
    q_buf, nk_buf, nv_buf, ak_buf, av_buf, out_buf = buffers

    attention = nar.nar_attention_wmma if args.kernel == "wmma" else nar.nar_attention_f32

    def call():
        attention(
            q_buf.ptr, nk_buf.ptr, nv_buf.ptr, ak_buf.ptr, av_buf.ptr, out_buf.ptr,
            rows, ar_rows, heads, kv_heads, dim, scale, runtime=runtime,
        )

    keys = ar_rows + rows
    for _ in range(args.warmup):
        call()
    runtime.device_synchronize()

    # The launch is asynchronous, so time a batch and divide: per-call wall time
    # including launch overhead, which is what the replay actually pays.
    times = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        for _ in range(5):
            call()
        runtime.device_synchronize()
        times.append((time.perf_counter() - started) / 5.0)

    median = statistics.median(times)
    # Global traffic as the kernel is written: every (row, head) block reads its
    # own copy of the K and V rows for its kv head.
    traffic = rows * heads * keys * dim * 2 * 2
    print(
        f"kernel={args.kernel} rows={rows} ar_rows={ar_rows} keys={keys} "
        f"heads={heads}/{kv_heads} dim={dim}\n"
        f"  median {median * 1e3:.2f} ms/call (min {min(times) * 1e3:.2f}, "
        f"max {max(times) * 1e3:.2f}) over {args.repeat} calls\n"
        f"  global K/V traffic {traffic / 1e9:.2f} GB/call -> "
        f"{traffic / median / 1e9:.1f} GB/s if it all missed cache"
    )

    if args.fixture_check:
        golden = np.load(GOLDEN)
        g_rows = int(golden["nar_rows"])
        g_ar = int(golden["ar_rows"])
        g_heads = int(golden["num_q_heads"])
        g_kv = int(golden["num_kv_heads"])
        g_dim = int(golden["head_dim"])
        g_out = np.zeros((g_rows, g_heads, g_dim), dtype=np.float32)
        g_buffers = [upload(a) for a in (
            golden["q"], golden["nar_k"], golden["nar_v"], golden["ar_k"], golden["ar_v"], g_out,
        )]
        nar.nar_attention_f32(
            g_buffers[0].ptr, g_buffers[1].ptr, g_buffers[2].ptr, g_buffers[3].ptr,
            g_buffers[4].ptr, g_buffers[5].ptr, g_rows, g_ar, g_heads, g_kv, g_dim,
            float(golden["scale"]), runtime=runtime,
        )
        copy_device_to_host(host_array_ptr(g_out), g_buffers[5])
        expected = golden["out"]
        identical = bool(np.array_equal(g_out.view(np.uint32), expected.view(np.uint32)))
        print(f"  parent-bits parity: {'bit-identical' if identical else 'DIFFERS'} "
              f"(max abs diff {np.abs(g_out - expected).max()})")
        for buffer in g_buffers:
            free(buffer, runtime=runtime)
        if not identical:
            return 1

    for buffer in buffers:
        free(buffer, runtime=runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
