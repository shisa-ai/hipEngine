#!/usr/bin/env python3
"""Launch the iu8 sparse exact repair standalone and report the grid it dispatches.

Iteration 6 needs to know whether the 131072-block grid seen in the production
prefill trace comes from the wrapper's ``grid_blocks`` argument or from somewhere
else.  This launches the registered kernel directly with a known ``grid_blocks``
and a near-empty risk queue, so a rocprofv3 kernel trace reports the grid that
actually went out.

Usage:
  rocprofv3 --kernel-trace --output-format csv -d /tmp/repair-probe -o probe -- \
      python3 scripts/qwen4exp_repair_grid_probe.py --grid-blocks 1024
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (  # noqa: E402
    register_gguf_q4_k_selected_prefill_kernels,
)
from hipengine.kernels.registry import resolve  # noqa: E402

register_gguf_q4_k_selected_prefill_kernels()

HIDDEN = 2560
FFN = 640
EXPERTS = 512
COMPACT_ROWS = 64
QK_K = 256
Q4_K_BLOCK_BYTES = 144


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-blocks", type=int, default=1024)
    parser.add_argument("--risk-count", type=int, default=8)
    args = parser.parse_args()

    runtime = get_hip_runtime()
    kernel = resolve(
        backend="hip_gfx1100", layer="moe_linear", quant="gguf_q4_k",
        variant="selected_dual_sparse_exact_repair_bf16",
    )

    def dev(nbytes: int, pattern: int = 0):
        buf = runtime.malloc(nbytes)
        runtime.memset(buf, pattern, nbytes)
        return buf

    row_bytes = (HIDDEN // QK_K) * Q4_K_BLOCK_BYTES
    weight_bytes = FFN * row_bytes
    input_buf = dev(COMPACT_ROWS * HIDDEN * 2, 0x2D)
    expert_start = dev((EXPERTS + 1) * 8)
    weight_a = dev(weight_bytes, 0x3C)
    weight_b = dev(weight_bytes, 0x5A)
    out_buf = dev(COMPACT_ROWS * 2 * FFN * 2)
    risk_count = dev(4)
    risk_indices = dev(max(4, args.risk_count * 4))
    runtime.memset(risk_count, 0, 4)
    # Publish the requested queue depth so the kernel's grid-stride bound is real.
    import numpy as np

    from hipengine.core.hip import MemcpyKind
    from hipengine.core.memory import host_array_ptr

    indices = np.arange(args.risk_count, dtype=np.int32)
    runtime.memcpy(risk_indices, host_array_ptr(indices), indices.nbytes,
                   MemcpyKind.HOST_TO_DEVICE)
    counts = np.asarray([args.risk_count], dtype=np.int32)
    runtime.memcpy(risk_count, host_array_ptr(counts), counts.nbytes,
                   MemcpyKind.HOST_TO_DEVICE)

    kernel(
        input_buf, expert_start, weight_a, weight_b, out_buf,
        risk_count, risk_indices, args.risk_count, COMPACT_ROWS,
        HIDDEN, FFN, FFN, EXPERTS,
        grid_blocks=args.grid_blocks,
        stream=0, runtime=runtime,
    )
    runtime.device_synchronize()
    print(f"launched with grid_blocks={args.grid_blocks} risk_count={args.risk_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
