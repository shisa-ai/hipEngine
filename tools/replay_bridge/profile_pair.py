#!/usr/bin/env python3
"""Kernel-level comparison of the wide-row hipEngine candidate against the comparator.

Both engines run the same Q8_0 projection geometry on the same bytes, in one
process, so the kernel durations rocprofv3 records are directly comparable.

Run it under the profiler with everything already built:

    rocprofv3 --kernel-trace --output-format csv -d /tmp/prof-out -- \\
        .venv/bin/python tools/replay_bridge/profile_pair.py \\
            --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0 \\
            --shim /tmp/replay-bridge/libmmb_replay.so

The wrappers are imported and their shared objects are loaded before any timed
launch, so no compiler runs inside the profiled region.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hipengine_adapter import HipEngineAdapter, load_packet  # noqa: E402
from replay_ab import ComparatorAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--shim", required=True, type=Path)
    parser.add_argument("--variant", default="dense_wide256_f32_f32_out")
    parser.add_argument("--iters", type=int, default=40)
    args = parser.parse_args()

    packet = load_packet(args.packet)

    # Preload every shared object so the profiled region contains no compiler.
    from hipengine.kernels.hip_gfx1100.quant import (  # noqa: F401
        gguf_q8_0_dense_wide,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    # The production runtime requests backend="hip_gfx1151"; the alias pass is
    # what makes the gfx1100 key space reachable under that name.
    register_gfx1151_kernels()

    gguf_q8_0_dense_wide.build_gguf_q8_0_dense_wide(load=True)
    ctypes.CDLL(str(args.shim))

    # --- hipEngine candidate -------------------------------------------------
    hip_packet = load_packet(args.packet)
    hip_packet.key["variant"] = args.variant
    with HipEngineAdapter(hip_packet) as adapter:
        adapter.replay(reps=args.iters, warmup=3)

    # --- comparator ----------------------------------------------------------
    # rotate must be at least the shim's minimum: the shim rejects a smaller
    # value. This is the same protocol the A/B used for its complete-operation
    # number, so the conversion kernel appears once every min_rotate calls.
    comparator = ComparatorAdapter(args.shim)
    if not comparator.available:
        raise RuntimeError("comparator shim reports MMB unavailable")
    comparator.run(packet, reps=args.iters, rotate=comparator.min_rotate)

    print(f"ran {args.iters} iterations of each engine on {args.packet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
