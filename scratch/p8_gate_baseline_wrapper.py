"""Capture the pre-change router arm for the P8 teacher-forced gate.

Importable entry: ``python scratch/p8_gate_baseline_wrapper.py capture
--prompt 2048 --prefill 1024 --out ...`` -- same CLI as
``scripts.gemma4_teacher_forced_gate.py capture``, with the router pinned to
the token-tile tier the tree used before the sgemm tier landed.
"""

import sys

import hipengine.kernels.hip_gfx1100.gemma4.gemma4_router as _router

_router._ROUTER_SGEMM_MIN_TOKENS = 10**9

from scripts.gemma4_teacher_forced_gate import main as _gate_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_gate_main(sys.argv[1:]))
