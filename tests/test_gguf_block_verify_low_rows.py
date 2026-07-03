"""GPU exactness gate for B1/B2 block verify (--target-block-min-rows 2).

verify_target_block must be bit-exact vs the trusted serial-exact reference at
rows 2 and 3 (below the historical ssm_conv_kernel=4 gate) before block verify is
allowed at small budgets. Skips without HIP or the local 35B MoE GGUF fixture.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

os.environ.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")
os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")

SEED = [1, 2, 13, 14, 198, 264, 374, 11, 323, 279, 304, 369, 429, 1, 13, 198]


def test_block_verify_matches_serial_exact_below_conv_kernel() -> None:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    # Re-assert kernel registrations (idempotent) so this full-verify test is robust
    # to running after other GPU tests in the same process that may have perturbed
    # the module-global kernel registry.
    from hipengine.kernels.hip_gfx1100.moe.router import register_qwen35_router_kernels
    register_qwen35_router_kernels()

    with Qwen35GGUFResidentSession(MODEL, use_wmma_prefill=True, use_gemv_decode=True) as session:
        session.prefill(SEED, return_logits=False, capture_hidden_seed_fp32=True)
        last = SEED[-1]
        gen = []
        for _ in range(20):
            r = session.step(last, return_logits=False, capture_hidden_seed_fp32=True)
            last = int(r.token_id)
            gen.append(last)
        pos = int(session._position)
        pool = [last] + gen[-6:]
        snap = session._linear_state_snapshot()
        failures = []
        try:
            for rows in (2, 3, 4):
                block_inputs = pool[:rows]
                session._restore_linear_state_snapshot(snap, position=pos)
                ref = [int(t) for t in session.verify_target_block_serial_exact(block_inputs).token_ids]
                for mode in ("bulk", "native"):
                    session._restore_linear_state_snapshot(snap, position=pos)
                    got = [
                        int(t)
                        for t in session.verify_target_block(
                            block_inputs, bulk_attention_mode=mode, use_wmma_prefill=False
                        ).token_ids
                    ]
                    if got != ref:
                        failures.append(f"rows={rows} mode={mode}: block {got} != serial-exact {ref}")
        finally:
            session._free_linear_state_snapshot(snap)

    assert not failures, "; ".join(failures)
