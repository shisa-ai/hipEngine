#!/usr/bin/env python3
"""Isolated A/B of the two production candidates for long global decode.

Same buffers, same arguments, alternating arms, median of N. Compares the
incumbent staged singleton chain against the flash-decoding route for the
global packing (head_dim 512, GQA ratio 8) at long key counts. Host-side
allocation is hoisted out of the timed region; this is device latency only.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from hipengine.core.memory import (
        copy_host_to_device, free, host_array_ptr, malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        Gemma4AttentionScratch, build_gemma4_attention,
        gemma4_attention_prefill_bf16,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_staged import (
        build_gemma4_attention_staged, gemma4_attention_staged_bf16,
    )

    attn = build_gemma4_attention(load=True)
    staged = build_gemma4_attention_staged(load=True)
    rng = np.random.default_rng(20261008)
    records = []

    for keys in args.keys:
        q = rng.standard_normal((1, args.num_heads, args.head_dim)).astype(np.float32) * 0.7
        k = rng.standard_normal((keys, args.num_kv_heads, args.head_dim)).astype(np.float32) * 0.7
        v = rng.standard_normal((keys, args.num_kv_heads, args.head_dim)).astype(np.float32) * 0.7
        to_bf16 = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
        arrays = [to_bf16(q), to_bf16(k), to_bf16(v), np.ones((1, keys), dtype=np.uint8)]
        buffers = []
        scratch_flash = Gemma4AttentionScratch()
        scratch_staged = Gemma4AttentionScratch()
        try:
            for array in arrays:
                buffer = malloc(array.nbytes)
                buffers.append(buffer)
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
            out_flash = malloc(1 * args.num_heads * args.head_dim * 2)
            out_staged = malloc(1 * args.num_heads * args.head_dim * 2)
            buffers += [out_flash, out_staged]
            q_ptr, k_ptr, v_ptr, m_ptr = (b.ptr for b in buffers[:4])

            def run_flash():
                gemma4_attention_prefill_bf16(
                    q_ptr, k_ptr, v_ptr, m_ptr, out_flash.ptr,
                    tokens=1, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim, scale=1.0, keys=keys,
                    library=attn, scratch=scratch_flash,
                )

            def run_staged():
                gemma4_attention_staged_bf16(
                    q_ptr, k_ptr, v_ptr, m_ptr, out_staged.ptr,
                    tokens=1, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim, scale=1.0, keys=keys,
                    library=staged, scratch=scratch_staged,
                )

            for _ in range(3):
                run_flash()
                run_staged()
            from hipengine.core.hip import get_hip_runtime
            get_hip_runtime().stream_synchronize(0)

            times = {"flash": [], "staged": []}
            for i in range(args.repeats):
                order = ("flash", "staged") if i % 2 == 0 else ("staged", "flash")
                for arm in order:
                    fn = run_flash if arm == "flash" else run_staged
                    t0 = time.perf_counter()
                    fn()
                    get_hip_runtime().stream_synchronize(0)
                    times[arm].append((time.perf_counter() - t0) * 1e3)
            f = statistics.median(times["flash"])
            s = statistics.median(times["staged"])
            records.append({"keys": keys, "flash_ms": f, "staged_ms": s,
                            "speedup": s / f, "flash_spread_pct":
                            (max(times["flash"]) - min(times["flash"])) / f * 100,
                            "staged_spread_pct":
                            (max(times["staged"]) - min(times["staged"])) / s * 100})
            print(f"keys={keys:7d} flash={f:8.3f}ms staged={s:8.3f}ms "
                  f"speedup={s/f:5.3f}x  spread flash={records[-1]['flash_spread_pct']:.1f}% "
                  f"staged={records[-1]['staged_spread_pct']:.1f}%", flush=True)
        finally:
            for buffer in buffers:
                free(buffer)
            scratch_flash.close()
            scratch_staged.close()

    if args.out:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, indent=2))
        print(f"artifact={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
