#!/usr/bin/env python3
"""Thread-geometry sweep for the fixed256 context-batch attention leaf.

Measures the production-geometry probe entrypoint
(fixed256 body, runtime block width) at threads 128/256/512/1024 across the
c1-relevant contexts on the actual c1 shape (16Q/2KV/D256, rows=1, bs=256).
threads=256 is the retained c1-exact default; 512/1024 split the value reduction
(value_groups=2/4) and change the score/denominator warp reduction tree, so
their output is NOT byte-exact vs 256 — the delta is reported as a KL/top-1
sanity signal only. Timing uses median of many reps to beat run noise.
"""

from __future__ import annotations

import ctypes
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

ctypes.CDLL("libamdhip64.so")

from hipengine.core.device import Device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    build_qwen35_paged_attn_decode,
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_threads_spans,
)
from hipengine.kvcache import KVLiveSpans

ROWS, NQH, NKV, HD = 1, 16, 2, 256
BS = 256
SCALE = HD ** -0.5
THREADS = (128, 256, 512, 1024)
CONTEXTS = (128, 256, 513, 640, 1024)


def _to_bf16_bits(x):
    b = x.view(np.uint32)
    lsb = (b >> 16) & 1
    return ((b + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _device_from_handle(ptr, shape, dtype):
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def bench_one(rt, lib, fn, reps, warm=50):
    for _ in range(warm):
        fn()
    rt.device_synchronize()
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        rt.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(walls)


def main() -> int:
    rt = get_hip_runtime()
    lib = build_qwen35_paged_attn_decode(load=True, require_cached=True)
    rng = np.random.default_rng(0x1234)
    print(f"device: gfx1151 (Radeon 8060S), shape 16Q/2KV/D256 rows=1 bs=256\n")

    results = {}
    for context in CONTEXTS:
        blen = (context + BS - 1) // BS
        q = (rng.standard_normal((ROWS, NQH, HD)) * 0.3).astype(np.float32)
        k = _to_bf16_bits(
            (rng.standard_normal((ROWS * blen * BS, NKV, HD)) * 0.3).astype(np.float32)
        )
        v = _to_bf16_bits(
            (rng.standard_normal((ROWS * blen * BS, NKV, HD)) * 0.3).astype(np.float32)
        )
        live = np.array([context], np.int64)
        bt = np.arange(blen, dtype=np.int32)
        bufs = []

        def up(a):
            a = np.ascontiguousarray(a)
            b = malloc(max(a.nbytes, 4), runtime=rt)
            bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=rt)
            return b

        qb, kb, vb, lb, bb = up(q), up(k), up(v), up(live), up(bt)
        out = malloc(ROWS * NQH * HD * 4, runtime=rt)
        bufs.append(out)
        spans = KVLiveSpans.paged_uniform(
            block_table=_device_from_handle(bb.ptr, (blen,), "int32"),
            live_counts=_device_from_handle(lb.ptr, (1,), "int64"),
            max_live_count=context,
            storage_dtype="bf16",
        )

        def make(threads):
            return lambda: qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_threads_spans(
                qb.ptr, kb.ptr, vb.ptr, out.ptr, spans, ROWS, context, BS, NQH, NKV,
                HD, SCALE, threads=threads, library=lib, runtime=rt,
            )

        reps = max(100, min(2000, int(1500 * 0.05 / max(0.005, context * 0.0002))))
        row = {}
        # threads=256 is the reference: capture its output for exactness deltas.
        zero = np.zeros(ROWS * NQH * HD, np.float32)
        zbytes = ROWS * NQH * HD * 4
        copy_host_to_device(out, host_array_ptr(zero), zbytes, runtime=rt)
        make(256)()
        rt.device_synchronize()
        ref = np.empty(ROWS * NQH * HD, np.float32)
        copy_device_to_host(host_array_ptr(ref), out, zbytes, runtime=rt)
        for threads in THREADS:
            f = make(threads)
            # correctness sanity vs the 256-thread output
            copy_host_to_device(out, host_array_ptr(zero), zbytes, runtime=rt)
            f()
            rt.device_synchronize()
            got = np.empty(ROWS * NQH * HD, np.float32)
            copy_device_to_host(host_array_ptr(got), out, zbytes, runtime=rt)
            if threads == 256:
                maxdiff = 0.0
            else:
                maxdiff = float(np.max(np.abs(got - ref)))
            ms = bench_one(rt, lib, f, reps)
            row[threads] = (ms, maxdiff)
            print(f"  ctx={context:5d} threads={threads:4d}: {ms:8.3f} ms/call  maxdiff_vs_256={maxdiff:.3e}")
        results[context] = row
        for b in bufs:
            free(b, runtime=rt)
        print()

    print("== summary (us/call) ==")
    best = {}
    for context, row in results.items():
        line = [f"ctx={context:5d}"]
        for threads in THREADS:
            ms, _ = row[threads]
            line.append(f"t{threads}={ms*1000:.0f}us")
        b = min(row, key=lambda t: row[t][0])
        best[context] = b
        line.append(f"best=t{b}")
        print("  " + "  ".join(line))
    print("\nbest thread per context:", {c: b for c, b in best.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
