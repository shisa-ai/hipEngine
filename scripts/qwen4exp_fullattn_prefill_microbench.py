#!/usr/bin/env python3
"""Production-geometry microbench for the full-attention decode-context leaf.

`qwen35_paged_full_attn_decode_context_tensor_batch_kernel` is 801.4 ms of the
code-p4096 prefill (56.4% of the qsa_attention family), and 81.3% of that sits in
a single 1024-row, long-context launch at 54.30 ms. This harness reproduces that
launch standalone so variants can be measured in seconds instead of running a
16 s prefill, and so the limiter can be separated one factor at a time.

Production geometry: rows=1024, 24 query heads, 2 KV heads (12:1 GQA), head_dim
256, block_size 256.

Two context modes:

  uniform  every row has context_len == --context
  causal   row r has context_len == chunk_start + r + 1, clamped to
           --max-context. This is the real chunked-prefill shape.

Variants are compared against the shipped leaf for byte-exactness, because a
bit-exact change needs no numerical gate and a non-exact one does.
"""

from __future__ import annotations

import argparse
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
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_experiment_spans,
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans,
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
)
from hipengine.kvcache import KVLiveSpans

NQH, NKV, HD, BS = 24, 2, 256, 256
SCALE = HD ** -0.5


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    b = x.view(np.uint32)
    lsb = (b >> 16) & 1
    return ((b + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _device_from_handle(ptr, shape, dtype):
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def bench(rt, fn, reps: int, warm: int = 3) -> float:
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


def replay(args, rt, lib) -> int:
    """Run the exact captured production launch and report its time."""

    z = np.load(args.replay)
    rows = int(z["rows"][0])
    max_ctx = int(z["max_context_len"][0])
    block_size = int(z["block_size"][0])
    nqh, nkv, hd = int(z["nqh"][0]), int(z["nkv"][0]), int(z["hd"][0])
    q, k, v = z["query"], z["key"], z["value"]
    bt, live = z["base_offsets"], z["live_counts"]
    blen = bt.size // rows
    print(f"replay {args.replay}: rows={rows} max_ctx={max_ctx} bs={block_size} "
          f"nqh={nqh} nkv={nkv} hd={hd} blen={blen} "
          f"live[min/mean/max]={live.min()}/{live.mean():.1f}/{live.max()}")

    bufs = []

    def up(a):
        a = np.ascontiguousarray(a)
        b = malloc(max(a.nbytes, 4), runtime=rt)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=rt)
        return b

    qb, kb, vb, lb, bb = up(q), up(k), up(v), up(live), up(bt)
    out = malloc(rows * nqh * hd * 4, runtime=rt)
    bufs.append(out)
    spans = KVLiveSpans.paged_uniform(
        block_table=_device_from_handle(bb.ptr, (rows * blen,), "int32"),
        live_counts=_device_from_handle(lb.ptr, (rows,), "int64"),
        max_live_count=max_ctx,
        storage_dtype="bf16",
    )

    def run():
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans(
            qb.ptr, kb.ptr, vb.ptr, out.ptr, spans, rows, max_ctx, block_size,
            nqh, nkv, hd, SCALE, library=lib, runtime=rt,
        )

    run()
    rt.device_synchronize()
    nbytes = rows * nqh * hd * 4
    ref = np.empty(rows * nqh * hd, np.float32)
    copy_device_to_host(host_array_ptr(ref), out, nbytes, runtime=rt)
    base_ms = bench(rt, run, args.repetitions, args.warmup)
    pairs = int(live.sum()) * nqh
    print(f"shipped fixed256 leaf: {base_ms:.3f} ms  finite={np.isfinite(ref).all()}")
    print(f"  (row,head,token) triples = {pairs:,}")
    print(f"  {base_ms * 1e6 / pairs:.4f} ns/triple, "
          f"FLOP {pairs * hd * 4 / (base_ms / 1e3) / 1e12:.3f} TFLOP/s, "
          f"K/V {pairs * 2 * hd * 2 / (base_ms / 1e3) / 1e9:.0f} GB/s")

    got = np.empty(rows * nqh * hd, np.float32)

    auto_capacity = (blen * block_size)

    def make_variant(capacity):
        def run_variant():
            qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans(
                qb.ptr, kb.ptr, vb.ptr, out.ptr, spans, rows, max_ctx, block_size,
                nqh, nkv, hd, SCALE, capacity, library=lib, runtime=rt,
            )
        return run_variant

    for spec in [x.strip() for x in args.capacities.split(",") if x.strip()]:
        cap = auto_capacity if spec == "auto" else int(spec)
        run_variant = make_variant(cap)
        run_variant()
        rt.device_synchronize()
        copy_device_to_host(host_array_ptr(got), out, nbytes, runtime=rt)
        exact = bool(np.array_equal(got.view(np.uint32), ref.view(np.uint32)))
        maxdiff = float(np.max(np.abs(got - ref)))
        ms = bench(rt, run_variant, args.repetitions, args.warmup)
        delta = (base_ms / ms - 1.0) * 100.0
        route = "int64 fallback" if cap == 0 else f"int32 (capacity={cap})"
        print(f"  promoted [{route}]: {ms:8.3f} ms  {base_ms / ms:5.3f}x  "
              f"{delta:+6.2f}%  bit_exact={exact}  maxdiff={maxdiff:.3e}")

    def make_qk(v):
        def run_qk():
            qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_experiment_spans(
                qb.ptr, kb.ptr, vb.ptr, out.ptr, spans, rows, max_ctx, block_size,
                nqh, nkv, hd, SCALE, v, library=lib, runtime=rt,
            )
        return run_qk

    for v in [int(x) for x in args.qk_variants.split(",") if x.strip()]:
        run_qk = make_qk(v)
        out_ref = np.zeros(rows * nqh * hd, np.float32)
        copy_host_to_device(out, host_array_ptr(out_ref), nbytes, runtime=rt)
        run_qk()
        rt.device_synchronize()
        copy_device_to_host(host_array_ptr(got), out, nbytes, runtime=rt)
        exact = bool(np.array_equal(got.view(np.uint32), ref.view(np.uint32)))
        maxdiff = float(np.max(np.abs(got - ref)))
        ms = bench(rt, run_qk, args.repetitions, args.warmup)
        delta = (base_ms / ms - 1.0) * 100.0
        print(f"  qk variant {v}: {ms:8.3f} ms  {base_ms / ms:5.3f}x  "
              f"{delta:+6.2f}%  bit_exact={exact}  maxdiff={maxdiff:.3e}")

    for b in bufs:
        free(b, runtime=rt)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--context", type=int, default=2048,
                   help="uniform context, or the clamp for causal mode")
    p.add_argument("--max-context", type=int, default=None,
                   help="LDS sizing context; defaults to --context")
    p.add_argument("--chunk-start", type=int, default=1024,
                   help="causal mode: row 0 context length minus 1")
    p.add_argument("--mode", choices=("uniform", "causal"), default="uniform")
    p.add_argument("--replay", default=None,
                   help="npz captured by qwen4exp_fullattn_launch_probe.py; "
                        "replays the exact production launch")
    p.add_argument("--capacities", default="0,auto",
                   help="KV capacities to try on the promoted route; 'auto' uses "
                        "the captured cache size, 0 forces the int64 fallback")
    p.add_argument("--qk-variants", default="",
                   help="comma-separated QK unroll experiment ids (0=2-token "
                        "int32, 1=4-token int32, 2=4-token int64)")
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    rows = args.rows
    max_ctx = args.max_context if args.max_context is not None else args.context
    blen = (max_ctx + BS - 1) // BS

    rt = get_hip_runtime()
    lib = build_qwen35_paged_attn_decode(load=True)

    if args.replay:
        return replay(args, rt, lib)

    rng = np.random.default_rng(0x5EED)

    if args.mode == "uniform":
        live = np.full(rows, args.context, dtype=np.int64)
    else:
        live = np.minimum(args.chunk_start + np.arange(rows, dtype=np.int64) + 1, max_ctx)
    print(f"mode={args.mode} rows={rows} max_ctx={max_ctx} blen={blen} "
          f"live[min/mean/max]={live.min()}/{live.mean():.0f}/{live.max()}")

    q = (rng.standard_normal((rows, NQH, HD)) * 0.3).astype(np.float32)
    # One sequence shared by every row: the chunked-prefill KV layout.
    k = _to_bf16_bits((rng.standard_normal((blen * BS, NKV, HD)) * 0.3).astype(np.float32))
    v = _to_bf16_bits((rng.standard_normal((blen * BS, NKV, HD)) * 0.3).astype(np.float32))
    bt = np.tile(np.arange(blen, dtype=np.int32), rows)

    bufs = []

    def up(a):
        a = np.ascontiguousarray(a)
        b = malloc(max(a.nbytes, 4), runtime=rt)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=rt)
        return b

    qb, kb, vb, lb, bb = up(q), up(k), up(v), up(live), up(bt)
    out = malloc(rows * NQH * HD * 4, runtime=rt)
    bufs.append(out)

    spans = KVLiveSpans.paged_uniform(
        block_table=_device_from_handle(bb.ptr, (rows * blen,), "int32"),
        live_counts=_device_from_handle(lb.ptr, (rows,), "int64"),
        max_live_count=max_ctx,
        storage_dtype="bf16",
    )

    def run():
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans(
            qb.ptr, kb.ptr, vb.ptr, out.ptr, spans, rows, max_ctx, BS,
            NQH, NKV, HD, SCALE, library=lib, runtime=rt,
        )

    nbytes = rows * NQH * HD * 4
    run()
    rt.device_synchronize()
    ref = np.empty(rows * NQH * HD, np.float32)
    copy_device_to_host(host_array_ptr(ref), out, nbytes, runtime=rt)
    finite = np.isfinite(ref).all()

    ms = bench(rt, run, args.repetitions, args.warmup)
    # Pairs actually visited: this is what the ns/pair figure is against.
    if args.mode == "uniform":
        pairs = int(rows) * int(args.context)
    else:
        pairs = int(live.sum())
    pairs_all = pairs * NQH
    print(f"shipped leaf: {ms:.3f} ms  finite={finite}")
    print(f"  (row,context) pairs = {pairs:,}; (row,head,token) triples = {pairs_all:,}")
    print(f"  {ms / pairs * 1e6:.2f} ns/pair, {ms / pairs_all * 1e9:.2f} ns/triple")
    print(f"  FLOP = {pairs_all * HD * 4:,} -> {pairs_all * HD * 4 / (ms / 1e3) / 1e12:.3f} TFLOP/s")
    kv_bytes = pairs_all * 1024
    print(f"  K/V load bytes = {kv_bytes / 1e9:.2f} GB -> {kv_bytes / (ms / 1e3) / 1e9:.0f} GB/s")

    for b in bufs:
        free(b, runtime=rt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
