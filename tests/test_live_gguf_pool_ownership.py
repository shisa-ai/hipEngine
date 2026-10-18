"""Live ownership gate for KV pool growth, reclaim, and engine close (task #16).

The CPU seam pins the ownership contract for the retained prefix snapshot arenas
and the dynamic verifier scratch. What it cannot show is that a real device cycle
ends with nothing outstanding: the pool's growth provider allocates real pages
through the same tracker, the packed verify workspace and its scratch are real
mallocs, and the session's close has to release every one of them.

This gate drives that cycle on the device -- grow the pool for real, run MTP
requests on both sides of the growth, shrink, reclaim, and close -- and asserts
the process-local allocation tracker is back to its pre-session baseline. The
pool's own close guard is part of the evidence: it refuses to close with live
request allocations or with an unreleased resident charge, so a leaked verifier
scratch charge fails here rather than passing silently.

Skips unless ROCm and the dense GGUF model are both present.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

MODEL = Path(
    os.environ.get("HIPENGINE_INT8_MTP_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
)
PROMPT = (9707, 11, 220, 264, 12)
GENERATED_TOKENS = 6
# Small on purpose: the cycle has to be a real growth of real device planes, not
# a large one. The workspace is created at GROWTH_ROWS rows and then grown.
GROWTH_ROWS = 8
GROWTH_ROWS_MARGIN = 16
# Wider than the session's own 1536-token ceiling: the workspace allocates at the
# geometry it is asked for, so growth has to exceed the live one to be growth.
GROWTH_MAX_SEQUENCE = 4096


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.exists()),
    reason="requires ROCm + the dense 27B GGUF model",
)


def test_grow_reclaim_and_close_leave_no_orphaned_allocations() -> None:
    """A real grow cycle with MTP interleaved reclaims to the pre-session count."""

    import time

    from hipengine.core.dtype import DType
    from hipengine.core.memory import memory_stats
    from hipengine.kvcache import FixedPagedKVPolicy
    from hipengine.runtime import qwen35_gguf_runner as runner_module
    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession
    from hipengine.runtime.qwen35_gguf_nextn import (
        Qwen35GGUFNextNDraftProvider,
        borrow_qwen35_gguf_nextn_fallback_weights,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    baseline = int(memory_stats()["active_allocations"])
    target = None
    provider = None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner_module, "_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS", 0)
        patch.setenv("HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS", "0")
        patch.setenv("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG", "1")
        target = Qwen35GGUFResidentSession(
            MODEL,
            max_sequence_length=1536,
            kv_scale_dtype=DType.FP32,
            kv_policy=FixedPagedKVPolicy(
                block_size=256, storage_dtype=DType.INT8_PER_TOKEN_HEAD
            ),
        )
        try:
            target.select_prefill_quant("gguf_q4_k_m")
            provider = Qwen35GGUFNextNDraftProvider.from_model(
                MODEL,
                max_positions=target.scratch.max_positions,
                max_requests=1,
                runtime=target.runtime,
                borrowed_fallback_weights=borrow_qwen35_gguf_nextn_fallback_weights(
                    target
                ),
            )
            with Qwen35GGUFMTPDecodeSession(
                target,
                provider,
                candidate_budget=3,
                quant="gguf_q4_k_m",
                target_verify_mode="native",
            ) as decoder:
                # The packed verify workspace and its scratch belong to the
                # persistent session root. Ask for a small one first so the
                # cycle starts from a live workspace, then grow it below.
                target.reset()
                target._ensure_packed_verify_workspace(
                    slot_count=1,
                    rows=GROWTH_ROWS,
                    max_sequence_length=GROWTH_MAX_SEQUENCE,
                    runtime=target.runtime,
                )
                scratch = target._packed_verify_scratch
                assert scratch is not None, "the packed prefill left no verifier scratch"
                assert target._packed_verify_state is not None
                rows_before = int(scratch.rows)
                assert rows_before > 0

                before = decoder.generate(
                    PROMPT, max_new_tokens=GENERATED_TOKENS, use_bulk_prefill=True,
                    prefill_draft=True,
                )
                assert len(before.token_ids) == GENERATED_TOKENS

                # The verifier scratch grows dynamically into the persistent
                # session root: ask for a wider geometry than the live one and
                # the owner frees the planes it had and allocates a new set.
                # That is real device memory moving through the tracked malloc
                # path, so an owner that keeps the old planes or never frees the
                # new ones shows up in the final count.
                target._ensure_packed_verify_workspace(
                    slot_count=1,
                    rows=rows_before + GROWTH_ROWS_MARGIN,
                    max_sequence_length=GROWTH_MAX_SEQUENCE,
                    runtime=target.runtime,
                )
                assert int(target._packed_verify_scratch.rows) > rows_before, (
                    "the requested growth did not widen the packed workspace"
                )
                # MTP on the other side of the growth, so the cycle interleaves
                # rather than growing before or after all speculative work.
                after = decoder.generate(
                    PROMPT, max_new_tokens=GENERATED_TOKENS, use_bulk_prefill=True,
                    prefill_draft=True,
                )
                assert len(after.token_ids) == GENERATED_TOKENS

            # The final reclaim is guarded on purpose: a decode graph still
            # binding the workspace replays raw plane pointers, so freeing those
            # planes under it is refused rather than allowed to dangle. Engine
            # close tears the graphs down and then reclaims, which is why the
            # ownership assertions below are read after close.
            with pytest.raises(RuntimeError, match="live graph still binds it"):
                target.release_idle_packed_workspace()
            assert target._packed_verify_scratch is not None, (
                "the refused reclaim must leave the workspace intact"
            )
        finally:
            if provider is not None:
                provider.close()
            target.close()

    # After engine close: the workspace, its scratch, the retained prefix
    # snapshot arenas, and every device allocation they owned are gone. The
    # pool's own close guard is part of the evidence, since it refuses to close
    # with an unreleased resident charge or a live request allocation.
    assert getattr(target, "_packed_verify_state", None) is None
    assert getattr(target, "_packed_verify_scratch", None) is None
    assert getattr(target, "_gguf_prefix_snapshot_arena_pool", None) is None, (
        "the retained prefix snapshot arena pool outlived engine close"
    )
    outstanding = int(memory_stats()["active_allocations"])
    assert outstanding == baseline, (
        f"{outstanding} tracked device allocations are still live after the final "
        f"reclaim and engine close (baseline {baseline})"
    )
