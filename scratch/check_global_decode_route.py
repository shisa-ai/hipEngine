#!/usr/bin/env python3
"""Route check: does the production wrapper take flash for long global decode?

Exercises the same entry the engine calls for a one-token block at the global
geometry (head_dim 512, GQA ratio 8) and reports the launcher's own selection.
Before the routing change this reported 4 (owned-scratch class kernel) for any
key count past the resident bound; the target is 3 (flash-decoding).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SELECTION = {
    0: "block",
    1: "class",
    2: "split",
    3: "flash",
    4: "class-global-logits",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=int, nargs="+", default=[1024, 32768])
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--dtype", choices=["f32", "bf16"], default="bf16")
    args = parser.parse_args()

    from hipengine.core.memory import (
        copy_host_to_device, free, host_array_ptr, malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        build_gemma4_attention, decode_selection, gemma4_attention_prefill_bf16,
        gemma4_attention_prefill_f32,
    )

    library = build_gemma4_attention(load=True)
    launcher = (
        gemma4_attention_prefill_f32 if args.dtype == "f32"
        else gemma4_attention_prefill_bf16
    )
    rng = np.random.default_rng(20261008)
    failed = False
    for keys in args.keys:
        q = np.ascontiguousarray(
            rng.standard_normal((1, args.num_heads, args.head_dim)) * 0.7, dtype=np.float32)
        k = np.ascontiguousarray(
            rng.standard_normal((keys, args.num_kv_heads, args.head_dim)) * 0.7, dtype=np.float32)
        v = np.ascontiguousarray(
            rng.standard_normal((keys, args.num_kv_heads, args.head_dim)) * 0.7, dtype=np.float32)
        mask = np.ones((1, keys), dtype=np.uint8)
        out = np.zeros((1, args.num_heads, args.head_dim), dtype=np.float32)
        if args.dtype == "bf16":
            q = (q.view(np.uint32) >> 16).astype(np.uint16)
            k = (k.view(np.uint32) >> 16).astype(np.uint16)
            v = (v.view(np.uint32) >> 16).astype(np.uint16)
            out = out.view(np.uint32).astype(np.uint16)
        arrays = [q, k, v, mask, out]
        buffers = []
        try:
            for array in arrays:
                buffer = malloc(array.nbytes)
                buffers.append(buffer)
                if array is not arrays[-1]:
                    copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
            launcher(
                *(b.ptr for b in buffers),
                tokens=1,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                scale=1.0,
                keys=keys,
                library=library,
            )
            selection = decode_selection(library)
            name = SELECTION.get(selection, f"unknown({selection})")
            expected = "flash" if keys >= 1024 else "class"
            ok = name == expected
            failed = failed or not ok
            print(f"keys={keys:7d} dtype={args.dtype} selection={selection} ({name}) "
                  f"expected={expected} {'OK' if ok else 'MISMATCH'}")
        finally:
            for buffer in buffers:
                free(buffer)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
