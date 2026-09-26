#!/usr/bin/env python3
"""Record the parent ``yue2_nar_attention_kernel`` output bits for parity tests.

The NAR attention kernel's per-tile reduction was rewritten to stop recomputing
every key's exponential in every lane. That rewrite is bit-exact by construction
(each weight is still ``expf(score - tile_max)`` for the same score and the same
tile maximum, and the tile sums and accumulations still walk the tile in the same
order), but "by construction" is not evidence. This script records the parent
kernel's exact output bits on a multi-tile GQA shape so
``tests/test_unit_yue2_nar_kernels.py`` can assert bit equality against them.

Run it against the parent revision (before the kernel change) and commit the
result as ``tests/fixtures/yue2/operators/nar_attention_parent.npz``, then
re-freeze the fixture index:

    python3 scripts/yue2_nar_attention_golden.py
    python3 scripts/yue2_oracle.py freeze

The fixture stores its own inputs, so the parity test does not depend on a
NumPy RNG stream staying stable across versions.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Multi-tile with a partial tail tile (keys = 198 over tiles of 128), grouped
# query attention, and the model's real head width.
SHAPE = {
    "ar_rows": 192,
    "nar_rows": 6,
    "num_q_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 128,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "tests/fixtures/yue2/operators/nar_attention_parent.npz",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    rng = np.random.default_rng(args.seed)
    ar_rows = SHAPE["ar_rows"]
    nar_rows = SHAPE["nar_rows"]
    num_q_heads = SHAPE["num_q_heads"]
    num_kv_heads = SHAPE["num_kv_heads"]
    head_dim = SHAPE["head_dim"]
    scale = np.float32(1.0 / float(np.sqrt(head_dim)))

    def bf16(values):
        wide = np.asarray(values, dtype=np.float32)
        bits = wide.view(np.uint32).astype(np.uint64)
        rounded = ((bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16))
        return (rounded.astype(np.uint32) & np.uint32(0xFFFF)).astype(np.uint16)

    q = (rng.standard_normal((nar_rows, num_q_heads, head_dim)) * 0.5).astype(np.float32)
    nar_k = bf16(rng.standard_normal((nar_rows, num_kv_heads, head_dim)))
    nar_v = bf16(rng.standard_normal((nar_rows, num_kv_heads, head_dim)))
    ar_k = bf16(rng.standard_normal((ar_rows, num_kv_heads, head_dim)))
    ar_v = bf16(rng.standard_normal((ar_rows, num_kv_heads, head_dim)))

    def upload(array):
        host = np.ascontiguousarray(array)
        buffer = malloc(max(host.nbytes, 8))
        copy_host_array_to_device(buffer, host)
        return buffer

    q_buf = upload(q)
    nk_buf = upload(nar_k)
    nv_buf = upload(nar_v)
    ak_buf = upload(ar_k)
    av_buf = upload(ar_v)
    out = np.zeros((nar_rows, num_q_heads, head_dim), dtype=np.float32)
    out_buf = upload(out)

    nar.nar_attention_f32(
        q_buf.ptr, nk_buf.ptr, nv_buf.ptr, ak_buf.ptr, av_buf.ptr, out_buf.ptr,
        nar_rows, ar_rows, num_q_heads, num_kv_heads, head_dim, float(scale),
    )
    copy_device_to_host(host_array_ptr(out), out_buf)

    keys = ar_rows + nar_rows
    tiles = (keys + head_dim - 1) // head_dim
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        q=q, nar_k=nar_k, nar_v=nar_v, ar_k=ar_k, ar_v=ar_v, out=out,
        scale=np.float32(scale),
        ar_rows=np.int32(ar_rows),
        nar_rows=np.int32(nar_rows),
        num_q_heads=np.int32(num_q_heads),
        num_kv_heads=np.int32(num_kv_heads),
        head_dim=np.int32(head_dim),
    )
    print(
        f"[golden] {args.out} keys={keys} tiles={tiles} "
        f"out absmax={float(np.abs(out).max()):.6f} sha256-pending"
    )
    for buffer in (q_buf, nk_buf, nv_buf, ak_buf, av_buf, out_buf):
        free(buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
