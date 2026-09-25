"""Does the Gemma 4 decode attention kernel scale with block count?

``gemma4_attention_decode_launch`` sets ``grid = tokens * num_heads``, so a
single-request decode step launches **16 blocks of 512 threads on a 96-CU GPU**.
This probe measures whether that is the limiter by varying ``num_heads``: each
head reads its own KV rows, so per-launch work and traffic scale with the head
count, and a per-launch time that stays flat while heads grow means the launched
blocks are nowhere near saturating the device.

The reading to take from it is ``us/head``. If it falls as blocks grow, the
kernel is parallelism-starved and a candidate that adds blocks (split-K over the
key range, which keeps total traffic constant instead of growing it the way this
probe does) can move it; if ``us/head`` is flat, the launched blocks already
saturate the machine and adding blocks cannot help.

Measured 2026-09-25 on the RX 7900 XTX (gfx1100), sliding geometry
(``head_dim=256``, ``keys=1024``, keep-mask, 30 iterations)::

    heads  kv_heads  us/launch  blocks   us/head
       16         2      280.6      16     17.54
       32         4      187.8      32      5.87
       64         8      163.2      64      2.55
      128        16      201.7     128      1.58

``us/head`` improves 6.9x from 16 to 64 blocks while per-launch time falls only
1.7x, and 128 blocks is worse in total time than 64 - the machine saturates
somewhere around 64 blocks. At 16 blocks the kernel is moving 16 MB per launch in
280 us, about 57 GB/s of a ~960 GB/s peak.

Usage::

    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \
        .venv/bin/python scripts/gemma4_attention_scale_probe.py
"""

from __future__ import annotations

import argparse
import time

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_prefill_bf16,
)

DEFAULT_HEADS = (16, 32, 64, 128)
DEFAULT_KEYS = 1024
DEFAULT_HEAD_DIM = 256
DEFAULT_ITERS = 30


def bench(
    num_heads: int,
    num_kv_heads: int,
    *,
    keys: int,
    head_dim: int,
    iters: int,
    runtime: object,
) -> float:
    """Return the mean per-launch time in microseconds."""

    # Keep the DeviceBuffer owners alive: the arena frees on garbage collection,
    # so holding only .ptr would be a use-after-free.
    owners = [
        malloc(1 * num_heads * head_dim * 2),
        malloc(keys * num_kv_heads * head_dim * 2),
        malloc(keys * num_kv_heads * head_dim * 2),
        malloc(keys),
        malloc(1 * num_heads * head_dim * 2),
    ]
    query, key, value, mask, out = (buffer.ptr for buffer in owners)
    # The mask pointer is a device address, so it must be filled on the device.
    runtime.memset_async(mask, 1, keys, 0)
    args = {
        "tokens": 1,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "scale": 1.0,
        "keys": keys,
        "stream": 0,
    }
    for _ in range(3):
        gemma4_attention_prefill_bf16(query, key, value, mask, out, **args)
    runtime.device_synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        gemma4_attention_prefill_bf16(query, key, value, mask, out, **args)
    runtime.device_synchronize()
    elapsed = time.perf_counter() - started
    assert owners
    return elapsed / iters * 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heads", type=int, nargs="+", default=list(DEFAULT_HEADS))
    parser.add_argument("--keys", type=int, default=DEFAULT_KEYS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--kv-ratio", type=int, default=8, help="query heads per KV head")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    args = parser.parse_args()

    runtime = get_hip_runtime()
    print(f"{'heads':>6} {'kv_heads':>9} {'us/launch':>10} {'blocks':>7} {'us/head':>9}")
    for heads in args.heads:
        kv_heads = max(1, heads // args.kv_ratio)
        micros = bench(
            heads,
            kv_heads,
            keys=args.keys,
            head_dim=args.head_dim,
            iters=args.iters,
            runtime=runtime,
        )
        print(f"{heads:>6} {kv_heads:>9} {micros:>10.1f} {heads:>7} {micros / heads:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
