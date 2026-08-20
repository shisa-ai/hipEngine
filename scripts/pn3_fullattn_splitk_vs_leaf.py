#!/usr/bin/env python3
"""Split-K (online-softmax, fused gate) vs fixed256-1024 leaf micro-benchmark.

The 35B c1 short-batch leaf is fixed256 at 1024 threads (promoted default) for
contexts < 1024; split-K with fused gate (qwen35_paged_full_attn_decode_split_k_gate_bf16_spans)
handles contexts >= 1024. This measures whether split-K (num_splits 2/3/4)
beats the promoted leaf inside the c1 window (513-1023), and sanity-checks the
split-K output vs the reference fixed256+gate chain (KL/top-1 on the logits
pre-gate is not directly comparable, so compare the final BF16 gated output).
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
    qwen35_full_attn_gate_mul_bf16,
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_threads_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
)
from hipengine.kvcache import KVLiveSpans

ROWS, NQH, NKV, HD = 1, 16, 2, 256
BS = 256
SCALE = HD ** -0.5
CONTEXTS = (513, 640, 768, 1023)
SPLITS = (2, 3, 4)


def _to_bf16_bits(x):
    b = x.view(np.uint32)
    lsb = (b >> 16) & 1
    return ((b + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _dev(ptr, shape, dtype):
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _bf16_to_f32(b):
    return (b.astype(np.int32) << 16).view(np.float32)


def bench_one(rt, fn, reps, warm=50):
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
    rng = np.random.default_rng(0x5EED)
    print("device gfx1151, 16Q/2KV/D256 rows=1 bs=256; split-K vs fixed256-1024+gate\n")

    for context in CONTEXTS:
        blen = (context + BS - 1) // BS
        q = (rng.standard_normal((ROWS, NQH, HD)) * 0.3).astype(np.float32)
        k = _to_bf16_bits((rng.standard_normal((ROWS * blen * BS, NKV, HD)) * 0.3).astype(np.float32))
        v = _to_bf16_bits((rng.standard_normal((ROWS * blen * BS, NKV, HD)) * 0.3).astype(np.float32))
        gate = _to_bf16_bits((rng.standard_normal((ROWS, NQH, HD)) * 2.0 - 2.0).astype(np.float32))
        live = np.array([context], np.int64)
        bt = np.arange(blen, dtype=np.int32)
        bufs = []

        def up(a, dtype=None):
            a = np.ascontiguousarray(a)
            b = malloc(max(a.nbytes, 4), runtime=rt)
            bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=rt)
            return b

        qb, kb, vb, gb, lb, bb = up(q), up(k), up(v), up(gate), up(live), up(bt)
        f32 = up(np.zeros((ROWS, NQH, HD), np.float32))
        gated_b16 = up(np.zeros((ROWS, NQH, HD), np.uint16))
        spans = KVLiveSpans.paged_uniform(
            block_table=_dev(bb.ptr, (blen,), "int32"),
            live_counts=_dev(lb.ptr, (1,), "int64"),
            max_live_count=context,
            storage_dtype="bf16",
        )
        total = ROWS * NQH * HD

        def chain():
            qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_threads_spans(
                qb.ptr, kb.ptr, vb.ptr, f32.ptr, spans, ROWS, context, BS, NQH, NKV,
                HD, SCALE, threads=1024, library=lib, runtime=rt,
            )
            qwen35_full_attn_gate_mul_bf16(f32.ptr, gb.ptr, gated_b16.ptr, total,
                                           library=lib, runtime=rt)

        # reference output (chain)
        chain()
        rt.device_synchronize()
        ref = np.empty(total, np.uint16)
        copy_device_to_host(host_array_ptr(ref), gated_b16, total * 2, runtime=rt)

        reps = max(100, min(1500, int(1000 * 0.06 / max(0.005, context * 0.0002))))
        ms_chain = bench_one(rt, chain, reps)
        line = [f"ctx={context:5d} fixed1024+gate={ms_chain*1000:6.0f}us"]
        best = ms_chain
        bestk = "fixed1024"
        for nsplit in SPLITS:
            ns = min(nsplit, (context + BS - 1) // BS)
            part_out = up(np.zeros((ROWS, NQH, HD * ns), np.float32))
            part_m = up(np.zeros((ROWS, NQH, ns), np.float32))
            part_l = up(np.zeros((ROWS, NQH, ns), np.float32))

            def splitk(ns=ns, part_out=part_out, part_m=part_m, part_l=part_l):
                qwen35_paged_full_attn_decode_split_k_gate_bf16_spans(
                    qb.ptr, kb.ptr, vb.ptr, gb.ptr, gated_b16.ptr,
                    part_out.ptr, part_m.ptr, part_l.ptr, spans,
                    BS, ns, BS, NQH, NKV, HD, HD, 1, SCALE,
                    library=lib, runtime=rt,
                )

            copy_host_to_device(gated_b16, host_array_ptr(np.zeros(total, np.uint16)), total * 2, runtime=rt)
            splitk()
            rt.device_synchronize()
            got = np.empty(total, np.uint16)
            copy_device_to_host(host_array_ptr(got), gated_b16, total * 2, runtime=rt)
            ref_f = _bf16_to_f32(ref); got_f = _bf16_to_f32(got)
            ms = bench_one(rt, splitk, reps)
            line.append(f"splitK{ns}={ms*1000:6.0f}us")
            if ms < best:
                best = ms
                bestk = f"splitK{ns}"
        print("  " + "  ".join(line) + f"  -> best={bestk}")
        for b in bufs:
            free(b, runtime=rt)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
