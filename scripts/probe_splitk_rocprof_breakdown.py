#!/usr/bin/env python3
"""rocprofv3 driver: context-kernel vs reducer share of split-K batch attention.

Runs one probe case (b8 = long-only 128K, b6 = mixed [128K,1K,1K,1K]) in a
loop under rocprofv3 --kernel-trace; sum DurationNs per kernel family from the
CSV afterwards. Decides whether the batch reducer's serial cross-split merge
is a large enough target to parallelize.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipengine.core.device import Device
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    build_qwen35_paged_attn_decode,
)
from scripts.probe_paged_decode_pool_invariance import Arm, BLOCK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["b8", "b6"], required=True)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    runtime = get_hip_runtime()
    device = Device("hip", 0)
    library = build_qwen35_paged_attn_decode(load=True)
    pool_elems = 2048 * BLOCK * 4 * 256
    pk = runtime.malloc(pool_elems * 2)
    pv = runtime.malloc(pool_elems * 2)
    runtime.memset(pk, 0, pool_elems * 2)
    runtime.memset(pv, 0, pool_elems * 2)

    if args.case == "b8":
        contexts, maps, splits = [131072], [list(range(0, 512))], 512
    else:
        contexts = [131072, 1024, 1024, 1024]
        maps = [list(range(0, 512)), list(range(640, 644)),
                list(range(704, 708)), list(range(768, 772))]
        splits = 512
    arm = Arm(runtime, device, library, name=args.case, contexts=contexts,
              page_maps=maps, pool_pages=2048, num_splits=splits,
              pool_ptr_k=pk, pool_ptr_v=pv)
    for _ in range(5):
        arm.call()
    runtime.stream_synchronize(0)
    for _ in range(args.iters):
        arm.call()
    runtime.stream_synchronize(0)
    print(f"case {args.case}: {args.iters} calls completed")
    arm.free()
    runtime.free(pk)
    runtime.free(pv)


if __name__ == "__main__":
    main()