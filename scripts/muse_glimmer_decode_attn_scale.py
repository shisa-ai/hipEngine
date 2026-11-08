#!/usr/bin/env python3
"""Isolated Muse Glimmer decode/verify attention cost vs context (diagnostic).

Times ``muse_glimmer_decode_attention`` (split + combine) for one NoPE
full-attention layer over synthetic BF16 K/V at each context, and reports the
ratio to the K/V DRAM-read floor. ``--tokens N`` times an N-row verify block
(each row runs at its own causal position, as the exact verify path does):
the floor is still one K/V read, so the ratio shows how much of the row
count the kernel fails to share. Per-model cost = this x the 13 NoPE layers
(SWA layers are window-bounded).

``--kernel shared`` times the shared-row WMMA kernel instead, at
``--split-keys`` keys per absolute split and ``--blocks`` blocks per KV head
(comma lists sweep both).

  .venv/bin/python scripts/muse_glimmer_decode_attn_scale.py --tokens 1,5
  .venv/bin/python scripts/muse_glimmer_decode_attn_scale.py --kernel shared \
      --split-keys 64,256 --blocks 64,128 --tokens 1,5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NQ, NKV, D = 32, 2, 128


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contexts", default="8192,32768,131072,262144")
    parser.add_argument("--tokens", default="1", help="comma list of verify row counts")
    parser.add_argument("--splits", type=int, default=64, help="decode split count (runtime default 64)")
    parser.add_argument("--window", type=int, default=0, help="0 = NoPE full layer; 2048 = SWA layer")
    parser.add_argument("--kernel", choices=("split", "shared"), default="split")
    parser.add_argument("--split-keys", default="256", help="shared: keys per absolute split")
    parser.add_argument("--blocks", default="128", help="shared: blocks per KV head")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dram-gbps", type=float, default=220.0, help="read ceiling for the floor")
    args = parser.parse_args(argv)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
        build_muse_glimmer_decode_attention,
        muse_glimmer_decode_attention,
        muse_glimmer_decode_attention_shared,
    )

    build_muse_glimmer_decode_attention()
    rt = get_hip_runtime()
    q_dim, kv_dim = NQ * D, NKV * D
    rng = np.random.default_rng(0)

    def bf16(n: int) -> np.ndarray:
        return (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

    if args.kernel == "shared":
        configs = [(int(s), int(b)) for s in args.split_keys.split(",") for b in args.blocks.split(",")]
    else:
        configs = [(0, args.splits)]
    print(f"{'ctx':>8} {'rows':>4} {'S':>5} {'G':>5} {'attn ms':>9} {'K/V MB':>8} {'floor ms':>9} {'ratio':>7} {'ms/row':>8}")
    for ctx in (int(c) for c in args.contexts.split(",")):
        kc, vc = bf16(ctx * kv_dim), bf16(ctx * kv_dim)
        bufs = []

        def up(a: np.ndarray):
            b = malloc(a.nbytes, runtime=rt)
            copy_host_array_to_device(b, np.ascontiguousarray(a), runtime=rt)
            bufs.append(b)
            return b

        dk, dv = up(kc), up(vc)
        try:
            for (split_keys, blocks), tokens in (
                (c, int(t)) for c in configs for t in args.tokens.split(",")
            ):
                dq = up(bf16(tokens * q_dim))
                out = up(np.zeros(tokens * q_dim, dtype=np.uint16))
                parts = up(np.zeros(blocks * tokens * NQ * (D + 2), dtype=np.float32))

                def launch() -> None:
                    if split_keys:
                        muse_glimmer_decode_attention_shared(
                            dq.ptr, dk.ptr, dv.ptr, out.ptr, parts.ptr,
                            n_q_heads=NQ, n_kv_heads=NKV, key_begin=0, key_end=ctx,
                            split_keys=split_keys, n_blocks=blocks, tokens=tokens,
                            window=args.window, scale=D ** -0.5, runtime=rt,
                        )
                        return
                    muse_glimmer_decode_attention(
                        dq.ptr, dk.ptr, dv.ptr, out.ptr, parts.ptr,
                        n_q_heads=NQ, n_kv_heads=NKV, key_begin=0, key_end=ctx,
                        n_splits=args.splits, tokens=tokens, window=args.window,
                        scale=D ** -0.5, runtime=rt,
                    )

                for _ in range(3):
                    launch()
                rt.device_synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    launch()
                rt.device_synchronize()
                ms = (time.perf_counter() - t0) / args.iters * 1e3
                span = min(ctx, args.window) if args.window else ctx
                kv_mb = span * kv_dim * 2 * 2 / 1e6
                floor = kv_mb / 1e3 / args.dram_gbps * 1e3
                print(f"{ctx:>8} {tokens:>4} {split_keys:>5} {blocks:>5} {ms:9.3f} {kv_mb:8.1f} {floor:9.3f} {ms / floor:6.2f}x {ms / tokens:8.3f}")
        finally:
            for b in bufs:
                free(b, runtime=rt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
