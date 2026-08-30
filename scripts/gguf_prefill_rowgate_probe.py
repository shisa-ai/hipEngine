"""Run scripts/qwen35_gguf_bench.py with the dual-fused prefill row gates lowered.

Reads HE_DUAL_WMMA_SILU_MIN_ROWS / HE_UNEQUAL_DUAL_WMMA_MIN_ROWS from the
environment, patches the module-level tuning gates in
hipengine.runtime.gguf_linear (both are read at dispatch time), then executes
HE_PROBE_TARGET (default scripts/qwen35_gguf_bench.py) with the remaining
argv. Diagnostic only: never imported by product code and not a supported way to
select a route. Measurement rules learned the hard way on 2026-08-30: always pass
``--public-ar-profile`` (without it the run measures the low-level selector route
and `shipping_ar_route_mismatch` comes back true - worth 1.65x on 45-row prefill),
never combine with ``--gpu-stage-timings`` (per-stage device syncs inflate wall),
and include a negative control at rows below the candidate floor - identical
dispatch there bounds the arm-order bias, which measured <=0.9% at 0.24 s prefill.
"""

from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipengine.runtime.gguf_linear as gguf_linear  # noqa: E402

silu_min = os.environ.get("HE_DUAL_WMMA_SILU_MIN_ROWS")
unequal_min = os.environ.get("HE_UNEQUAL_DUAL_WMMA_MIN_ROWS")
patched = {}
if silu_min:
    patched["_Q4_T16_DUAL_WMMA_SILU_MIN_ROWS"] = (
        gguf_linear._Q4_T16_DUAL_WMMA_SILU_MIN_ROWS,
        int(silu_min),
    )
    gguf_linear._Q4_T16_DUAL_WMMA_SILU_MIN_ROWS = int(silu_min)
if unequal_min:
    patched["_Q4_T16_UNEQUAL_DUAL_WMMA_MIN_ROWS"] = (
        gguf_linear._Q4_T16_UNEQUAL_DUAL_WMMA_MIN_ROWS,
        int(unequal_min),
    )
    gguf_linear._Q4_T16_UNEQUAL_DUAL_WMMA_MIN_ROWS = int(unequal_min)
print(f"[probe] lowered dual prefill row gates: {patched}", file=sys.stderr, flush=True)

target = os.environ.get("HE_PROBE_TARGET", "scripts/qwen35_gguf_bench.py")
runpy.run_path(target, run_name="__main__")
