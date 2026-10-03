"""A/B the strict and tiled gemma4 prefill attention kernels on identical input.

The tiled kernel exists in the tree, is exported as
hipengine_gemma4_attention_prefill_tiled_{bf16,f32}, and has no Python wrapper,
so it has never been benched against the strict kernel it would replace. This
script drives both through ctypes on the same device buffers, times them, and
compares the outputs elementwise.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/gemma4_attention_prefill_tiled_ab.py \
      --tokens 512 --iters 5 --warmup 2
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def bf16(a: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--keys", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    ap.add_argument(
        "--mask",
        choices=("keep", "zero"),
        default="keep",
        help="'zero' masks every key. NOTE: this does not cleanly ablate the K/V "
             "loads, because every logit becomes -inf and the softmax then "
             "evaluates expf(-inf - -inf), which is NaN. Use it only to check "
             "what the arithmetic does under an all-masked row.",
    )
    args = ap.parse_args()

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as ga

    tokens = args.tokens
    key_count = args.keys if args.keys is not None else tokens
    n_head, n_kv, head_dim = args.num_heads, args.num_kv_heads, args.head_dim
    window = args.window

    library = ga.build_gemma4_attention()

    rng = np.random.default_rng(20260930)
    q = rng.standard_normal((tokens, n_head, head_dim)).astype(np.float32)
    k = rng.standard_normal((key_count, n_kv, head_dim)).astype(np.float32)
    v = rng.standard_normal((key_count, n_kv, head_dim)).astype(np.float32)
    mask = np.ones((tokens, key_count), dtype=np.uint8)
    for t in range(tokens):
        lo = 0 if window <= 0 else max(0, t - window + 1)
        mask[t, :lo] = 0
        mask[t, t + 1:] = 0

    if args.mask == "zero":
        mask[:] = 0

    if args.dtype == "bf16":
        qd, kd, vd = bf16(q), bf16(k), bf16(v)
        out_dtype, sym_suffix = np.uint16, "bf16"
        strict = ga.gemma4_attention_prefill_bf16
    else:
        qd = np.ascontiguousarray(q, dtype=np.float32)
        kd = np.ascontiguousarray(k, dtype=np.float32)
        vd = np.ascontiguousarray(v, dtype=np.float32)
        out_dtype, sym_suffix = np.float32, "f32"
        strict = ga.gemma4_attention_prefill_f32

    out_strict = np.zeros((tokens, n_head, head_dim), dtype=out_dtype)
    out_tiled = np.zeros((tokens, n_head, head_dim), dtype=out_dtype)

    # The tiled launcher has no Python wrapper; bind the exported symbol directly.
    tiled = getattr(library, f"hipengine_gemma4_attention_prefill_tiled_{sym_suffix}")
    tiled.restype = ctypes.c_int
    tiled.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64,
    ]

    runtime = get_hip_runtime()
    stream = 0  # the default stream; the launcher takes a plain int

    def sync() -> None:
        runtime.device_synchronize()

    bufs = [malloc(a.nbytes) for a in (qd, kd, vd, mask, out_strict)]
    tiled_buf = malloc(out_tiled.nbytes)
    try:
        for buf, arr in zip(bufs, (qd, kd, vd, mask, out_strict)):
            copy_host_array_to_device(buf, arr)

        def run_strict() -> None:
            strict(
                bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, bufs[4].ptr,
                tokens=tokens, num_heads=n_head, num_kv_heads=n_kv,
                head_dim=head_dim, scale=1.0, keys=key_count, window=window,
                row_offset=0,
            )

        def run_tiled() -> None:
            err = tiled(
                ctypes.c_void_p(bufs[0].ptr), ctypes.c_void_p(bufs[1].ptr),
                ctypes.c_void_p(bufs[2].ptr), ctypes.c_void_p(bufs[3].ptr),
                ctypes.c_void_p(tiled_buf.ptr),
                ctypes.c_int64(tokens), ctypes.c_int64(n_head),
                ctypes.c_int64(n_kv), ctypes.c_int64(head_dim),
                ctypes.c_float(1.0), ctypes.c_void_p(stream),
                ctypes.c_int64(key_count), ctypes.c_int64(window),
                ctypes.c_int64(0),
            )
            if err != 0:
                raise RuntimeError(f"tiled launcher returned {err}")

        def timed(fn) -> float:
            for _ in range(args.warmup):
                fn()
            sync()
            times = []
            for _ in range(args.iters):
                started = time.perf_counter()
                fn()
                sync()
                times.append(time.perf_counter() - started)
            return statistics.median(times) * 1e3

        # Correctness first: run each once and compare.
        run_strict()
        run_tiled()
        sync()
        got_strict = np.empty_like(out_strict)
        got_tiled = np.empty_like(out_tiled)
        copy_device_to_host(got_strict.ctypes.data, bufs[4])
        copy_device_to_host(got_tiled.ctypes.data, tiled_buf)

        same = np.array_equal(got_strict, got_tiled)
        if args.dtype == "bf16":
            a = got_strict.view(np.uint16).astype(np.float32)
            b = got_tiled.view(np.uint16).astype(np.float32)
        else:
            a = got_strict.astype(np.float32)
            b = got_tiled.astype(np.float32)
        diff = np.abs(a - b)
        rel = diff / np.maximum(np.abs(a), 1e-30)

        ms_strict = timed(run_strict)
        ms_tiled = timed(run_tiled)

        print(f"tokens={tokens} keys={key_count} head_dim={head_dim} "
              f"heads={n_head}/{n_kv} window={window} dtype={args.dtype} "
              f"iters={args.iters}")
        print(f"{'kernel':<10} {'ms/launch':>10} {'GFLOP/s':>9}")
        # Causal pairs, windowed: the kept range is [lo, min(t+1, keys)), which is
        # what the mask above encodes. Using tokens*keys would report the flops for
        # a different geometry and inflate the rate.
        pairs = 0
        for t in range(tokens):
            lo = 0 if window <= 0 else max(0, t - window + 1)
            pairs += max(0, min(t + 1, key_count) - lo)
        flops = 4.0 * pairs * n_head * head_dim
        for name, ms in (("strict", ms_strict), ("tiled", ms_tiled)):
            print(f"{name:<10} {ms:10.2f} {flops / (ms * 1e6):9.1f}")
        print(f"ratio tiled/strict: {ms_tiled / ms_strict:.3f}  "
              f"({'faster' if ms_tiled < ms_strict else 'slower'})")
        print()
        nan_strict = int(np.isnan(a).sum())
        nan_tiled = int(np.isnan(b).sum())
        print(f"NaN in strict output: {nan_strict} / {a.size}")
        print(f"NaN in tiled output:  {nan_tiled} / {b.size}")
        print(f"bit-identical: {same}")
        print(f"max abs diff:  {diff.max():.6g}")
        print(f"max rel diff:  {rel.max():.6g}")
        print(f"mismatched elements: {(diff > 0).sum()} / {diff.size}")
    finally:
        for buf in bufs:
            free(buf)
        free(tiled_buf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
