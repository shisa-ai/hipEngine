"""Run scripts/gguf_mtp_c1c8_server_bench.py with GGUF prefill row gates patched.

Same diagnostic trick as ``gguf_prefill_rowgate_probe.py`` but for the server
harness, so a route floor can be A/B'ed on the AR/MTP topline (which requires a
tracked-clean tree, so runtime patching is the only way to compare two floors
from one binary). Reads HE_DUAL_WMMA_SILU_MIN_ROWS and
HE_UNEQUAL_DUAL_WMMA_MIN_ROWS; the module-level gates are read at dispatch time.
Diagnostic only: never imported by product code.
"""

from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipengine.runtime.gguf_linear as gguf_linear  # noqa: E402

_PATCHES = {
    "HE_DUAL_WMMA_SILU_MIN_ROWS": "_Q4_T16_DUAL_WMMA_SILU_MIN_ROWS",
    "HE_UNEQUAL_DUAL_WMMA_MIN_ROWS": "_Q4_T16_UNEQUAL_DUAL_WMMA_MIN_ROWS",
}

patched = {}
for env, constant in _PATCHES.items():
    value = os.environ.get(env)
    if not value:
        continue
    previous = getattr(gguf_linear, constant)
    setattr(gguf_linear, constant, int(value))
    patched[constant] = (previous, int(value))
print(f"[server-probe] GGUF prefill row gates: {patched}", file=sys.stderr, flush=True)

runpy.run_path("scripts/gguf_mtp_c1c8_server_bench.py", run_name="__main__")
