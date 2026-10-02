"""What rate does the Gemma 4 prefill attention kernel actually move KV at?

The attention walk's cost is modelled as ``tokens * keys * 4096`` bytes for a
head_dim-512 full layer: two KV heads, 512 dimensions, two bytes, K and V. That
model is what produces every "N% of the DRAM roofline" figure recorded for the
prefill walk, so this probe times the shipped kernel on one chunk and one full
layer and reports the rate against it.

The mask is all-ones so the byte count is unambiguous -- the real prefill's causal
mask averages half the keys per chunk, so the rate here is the peak.

Two rates are reported, because each query-head tile re-walks the same band:

  * ``unique``  -- the model above, the DRAM demand if L2 held nothing.
  * ``logical`` -- that times the number of query-head tiles, the demand if L2
    held nothing *and* every tile's read missed.

Measured 2026-10-02 on the Radeon 8060S (gfx1151, Strix Halo), the shipped
prefill shape, three runs per point, median, against the 238.5 GB/s the roofline
kernel measured on this device::

    keys    tokens      ms   unique GB/s   % of roofline
    16384      256   17.56         978.4             410%
    65536      256   71.21         965.0             405%
   262144      256  292.64         939.3             394%

and at 65,536 keys::

    tokens      ms   unique GB/s   % of roofline
        16    5.48         783.8             329%
        32   15.11         568.3             238%
        64   18.56         925.5             388%
       128   38.84         884.7             371%
       256   71.87         956.1             401%
       512  131.98        1041.3             437%

Time is linear in both tokens and keys, so the kernel is genuinely doing that much
work; it is moving the counted bytes at three to four times the roofline. At
``tokens=16`` there is exactly one 16-row block per query-head tile, so it is not
cross-block L2 sharing. The ratio between the 16-row and 1-row configurations at
65,536 keys is 4.09x for 16x the rows, which puts the real per-row-per-key cost
nearer 1,024 bytes than 4,096 -- one 512-dimension BF16 row.

**The model above is what is wrong, and hardware counters say so.**
``rocprofv3 --pmc FETCH_SIZE`` reports the actual DRAM traffic, and it was
calibrated first against the decode shape, which reads each band exactly once:
``tokens=1, keys=65536`` is 268,435,456 bytes by the model and 262,313.375 KB by
the counter, 0.06% apart. The same counter on the prefill shape, 65,536 keys,
one full layer, all-ones mask::

    tokens    grid      ms   DRAM fetch MB   GB/s   % of roofline
        16    8192    6.51           271.8   41.8             17.5%
        64     512   19.59           527.5   26.9             11.3%
       256     512   75.10           708.6    9.4              4.0%

The model says 68.7 GB at the largest point; the counter says 708.6 MB. **It
over-counts by 97x, and the prefill walk is at 4% of the DRAM roofline, not the
105-118% that model produced.** The fetch count grows sub-linearly with rows, so
L2 absorbs most of the 16-row blocks' sharing -- the opposite of what the
"reads each band once per query-head tile" reasoning predicted.

The kernel is therefore neither DRAM-bound nor tensor-bound. See
``worklog/entries/20261002T214531.740943Z-lhl-gemma4-gfx1151-prefill-roofline-counters-ad03cb.md``.
Run it under the counter with::

    env -u HIP_VISIBLE_DEVICES PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
        rocprofv3 --pmc FETCH_SIZE -d <dir> -- .venv/bin/python \
        scripts/gemma4_prefill_kernel_bw.py --reps 1 --keys 65536 --tokens 16 64 256

Usage::

    env -u HIP_VISIBLE_DEVICES PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
        .venv/bin/python scripts/gemma4_prefill_kernel_bw.py --reps 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_array_to_device, free, malloc
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    Gemma4AttentionScratch,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
    gemma4_attention_prefill_wmma_full_bf16 as wmma,
)

HEADS, KV_HEADS, HEAD_DIM = 16, 2, 512
ROOFLINE_GBPS = 238.5  # measured by the roofline kernel on this device
# Query-head tiles that re-walk one band: 16 query heads in tiles of 2.
HEAD_TILES = HEADS // 2


def bf16(a):
    bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return (((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000) >> 16).astype(np.uint16)


def unique_bytes(keys: int, tokens: int) -> int:
    return tokens * keys * KV_HEADS * HEAD_DIM * 2 * 2


def measure(tokens: int, keys: int, reps: int) -> dict:
    rng = np.random.default_rng(20261002)
    query = bf16(rng.standard_normal((tokens, HEADS, HEAD_DIM)).astype(np.float32))
    key = bf16(rng.standard_normal((keys, KV_HEADS, HEAD_DIM)).astype(np.float32))
    value = bf16(rng.standard_normal((keys, KV_HEADS, HEAD_DIM)).astype(np.float32))
    out = np.zeros((tokens, HEADS, HEAD_DIM), dtype=np.uint16)
    mask = np.ones((tokens, keys), dtype=np.uint8)

    buffers = []
    for array in (query, key, value, mask, out):
        b = malloc(array.nbytes)
        buffers.append(b)
        copy_host_array_to_device(b, array)
    qb, kb, vb, mb, ob = (b.ptr for b in buffers)

    common = dict(
        tokens=tokens,
        keys=keys,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        scale=1.0,
        window=keys,
        row_offset=keys - tokens,
    )
    scratch = Gemma4AttentionScratch()
    try:
        wmma(qb, kb, vb, mb, ob, scratch=scratch, **common)
        get_hip_runtime().device_synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            wmma(qb, kb, vb, mb, ob, scratch=scratch, **common)
            get_hip_runtime().device_synchronize()
            times.append(time.perf_counter() - t0)
        ms = statistics.median(times) * 1000
        uniq = unique_bytes(keys, tokens)
        gbps = uniq / (ms / 1000) / 1e9
        return {
            "keys": keys,
            "tokens": tokens,
            "ms": ms,
            "unique_bytes": uniq,
            "unique_gbps": gbps,
            "unique_pct_of_roofline": 100 * gbps / ROOFLINE_GBPS,
            "logical_gbps": gbps * HEAD_TILES,
            "logical_pct_of_roofline": 100 * gbps * HEAD_TILES / ROOFLINE_GBPS,
        }
    finally:
        scratch.close()
        for b in buffers:
            free(b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tokens", type=int, nargs="+", default=[256])
    parser.add_argument("--keys", type=int, nargs="+", default=[16384, 65536, 262144])
    args = parser.parse_args()

    results = []
    for keys in args.keys:
        for tokens in args.tokens:
            row = measure(tokens, keys, args.reps)
            results.append(row)
            print(
                f"keys={keys:>7} tokens={tokens:>4}  {row['ms']:8.2f} ms  "
                f"unique {row['unique_gbps']:6.1f} GB/s "
                f"({row['unique_pct_of_roofline']:5.1f}% of roofline)  "
                f"logical {row['logical_gbps']:7.1f} GB/s "
                f"({row['logical_pct_of_roofline']:6.1f}%)",
                flush=True,
            )
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
