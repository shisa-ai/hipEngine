#!/usr/bin/env python3
"""Per-launch cost of the Gemma 4 decode attention kernel at real geometry.

The campaign's decode profile puts ``gemma4_attention_decode_kernel`` at ~53%
of the token (24 ms of 45.7 ms at 1024 prompt / 128 output), so it is measured
on its own rather than inferred from end-to-end rows. This driver launches the
exported decode symbol directly on the artifact's two attention geometries:

* sliding layers (25 of 30): ``num_heads=16``, ``num_kv_heads=2``,
  ``head_dim=256``
* full layers (5 of 30): ``num_heads=16``, ``num_kv_heads=8``,
  ``head_dim=512``

and reports per-launch time plus the implied KV traffic. ``--mask zero`` runs
the same shapes with an all-zero keep-mask, which removes every K/V load while
leaving the tile loop and reductions intact; the gap between the two modes is
the memory share of the cost. Both numbers together decide whether a candidate
must attack traffic or latency.

Usage::

    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \\
        .venv/bin/python scripts/gemma4_attention_decode_bench.py --mask keep

The kernel's ``keys`` argument is the live context; the runner passes its full
capacity, so a row here is a lower bound on the production cost at that context.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    build_gemma4_attention,
    decode_selection,
    decode_slices,
    gemma4_attention_prefill_bf16,
)

GEOMETRIES = {
    "sliding": dict(num_heads=16, num_kv_heads=2, head_dim=256),
    "full": dict(num_heads=16, num_kv_heads=8, head_dim=512),
}


def run_case(library, runtime, *, keys: int, geometry: str, mask_mode: str, iters: int,
             tokens: int = 1) -> dict:
    geo = GEOMETRIES[geometry]
    num_heads = geo["num_heads"]
    num_kv_heads = geo["num_kv_heads"]
    head_dim = geo["head_dim"]

    rng = np.random.default_rng(0)
    q = rng.normal(0, 0.3, size=(tokens, num_heads, head_dim)).astype(np.float16)
    k = rng.normal(0, 0.3, size=(keys, num_kv_heads, head_dim)).astype(np.float16)
    v = rng.normal(0, 0.3, size=(keys, num_kv_heads, head_dim)).astype(np.float16)
    mask = np.ones((tokens, keys), dtype=np.uint8)
    if mask_mode == "zero":
        mask[:] = 0
    out = np.zeros((tokens, num_heads, head_dim), dtype=np.float16)

    buffers = []
    ptrs = []
    for array in (q, k, v, mask, out):
        buf = malloc(array.nbytes)
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes)
        buffers.append(buf)
        ptrs.append(buf.ptr)
    q_p, k_p, v_p, m_p, o_p = ptrs

    def once() -> None:
        gemma4_attention_prefill_bf16(
            q_p, k_p, v_p, m_p, o_p,
            tokens=tokens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=1.0,
            keys=keys,
            library=library,
            runtime=runtime,
        )

    try:
        for _ in range(5):
            once()
        runtime.device_synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            once()
        runtime.device_synchronize()
    finally:
        for buf in buffers:
            free(buf)

    # Which decode path the launcher took, read back rather than inferred. The
    # three are numerically close by design, so a timing or a parity test cannot
    # distinguish them; without this the row does not say what it measured. It is
    # also the only way to see whether ``slices`` engaged, which is the question
    # the block-count lever turns on.
    selection = decode_selection(library)
    slices = decode_slices(keys, head_dim)

    per_launch_us = (time.perf_counter() - started) / iters * 1e6
    unique_kv = tokens * keys * num_kv_heads * head_dim * 2 * 2  # K+V bf16, read once
    issued_kv = tokens * keys * num_heads * head_dim * 2 * 2 * 3  # per query head, 3 passes
    return {
        "geometry": geometry,
        "tokens": tokens,
        "keys": keys,
        "mask": mask_mode,
        "selection": selection,
        "slices": slices,
        "blocks": tokens * num_heads,
        "per_launch_us": per_launch_us,
        "unique_kv_mb": unique_kv / 1e6,
        "unique_kv_gbps": unique_kv / (per_launch_us * 1e-6) / 1e9,
        "issued_kv_mb": issued_kv / 1e6,
        "issued_kv_gbps": issued_kv / (per_launch_us * 1e-6) / 1e9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--geometry", choices=sorted(GEOMETRIES), nargs="+",
                        default=["sliding", "full"])
    parser.add_argument("--mask", choices=("keep", "zero"), default="keep")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--tokens",
        type=int,
        default=1,
        help="query rows (blocks = tokens * num_heads); scales the grid to separate "
             "latency-bound from throughput-bound behaviour",
    )
    args = parser.parse_args()

    runtime = get_hip_runtime()
    library = build_gemma4_attention(load=True)
    rows = []
    for geometry in args.geometry:
        for keys in args.keys:
            row = run_case(
                library, runtime,
                keys=keys, geometry=geometry, mask_mode=args.mask, iters=args.iters,
                tokens=args.tokens,
            )
            rows.append(row)
            print(
                f"{row['geometry']:8s} tokens={row['tokens']:2d} keys={row['keys']:5d} mask={row['mask']:4s} "
                f"sel={row['selection']} slices={row['slices']} blocks={row['blocks']:3d} "
                f"{row['per_launch_us']:9.1f} us/launch  "
                f"unique-KV {row['unique_kv_mb']:6.2f} MB -> {row['unique_kv_gbps']:6.1f} GB/s  "
                f"issued {row['issued_kv_mb']:7.2f} MB -> {row['issued_kv_gbps']:6.1f} GB/s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
