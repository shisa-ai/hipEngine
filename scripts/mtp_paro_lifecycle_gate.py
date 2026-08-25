#!/usr/bin/env python3
"""Gate PARO MTP borrowed target pointers and private-head memory ownership."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative.mtp_native import NativeMtpChainProposer, NativeMtpW8A16Head


def _delta(after: dict[str, int], before: dict[str, int], key: str) -> int:
    return int(after[key]) - int(before[key])


def run(*, model: Path, backend: str) -> dict[str, Any]:
    started = time.perf_counter()
    reset_memory_stats()
    initial = memory_stats()
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    session = Qwen35ParoResidentSession(runner, max_sequence_length=16, max_batch_size=2)
    borrowed = None
    legacy = None
    stale_pointer_rejected = False
    try:
        session_loaded = memory_stats()
        head = NativeMtpW8A16Head(
            weight_int8_ptr=int(session.lm_head_weight.tensor.ptr),
            scale_f32_ptr=int(session.lm_head_scale.tensor.ptr),
            vocab_size=int(session.vocab_size),
            threads=int(session.lm_head_threads),
            owner=session,
        )
        borrowed = NativeMtpChainProposer(
            model,
            max_positions=16,
            max_mtp_tokens=16,
            runtime=session.runtime,
            compiler_version=session.compiler_version,
            scoring_head=head,
        )
        borrowed_loaded = memory_stats()
        borrowed_private_head_loaded = "lm_head.weight" in borrowed.weights
        borrowed.close()
        borrowed = None
        after_borrowed_close = memory_stats()

        legacy = NativeMtpChainProposer(
            model,
            max_positions=16,
            max_mtp_tokens=16,
            runtime=session.runtime,
            compiler_version=session.compiler_version,
        )
        legacy_loaded = memory_stats()
        legacy_private_head_loaded = "lm_head.weight" in legacy.weights
        legacy.close()
        legacy = None
        after_legacy_close = memory_stats()

        stale_probe = NativeMtpChainProposer(
            model,
            max_positions=16,
            max_mtp_tokens=16,
            runtime=session.runtime,
            compiler_version=session.compiler_version,
            scoring_head=head,
        )
        session.close()
        try:
            head.validate_live()
        except RuntimeError:
            stale_pointer_rejected = True
        finally:
            stale_probe.close()
        residue_before_reuse = memory_stats()
        reuse_probe = NativeMtpChainProposer(
            model,
            max_positions=16,
            max_mtp_tokens=16,
            runtime=session.runtime,
            compiler_version=session.compiler_version,
            scoring_head=head,
        )
        reuse_probe.close()
        after_all_close = memory_stats()
    finally:
        if borrowed is not None:
            borrowed.close()
        if legacy is not None:
            legacy.close()
        if not session.closed:
            session.close()
    final = memory_stats()

    borrowed_delta = _delta(borrowed_loaded, session_loaded, "current_allocated_bytes")
    legacy_delta = _delta(legacy_loaded, session_loaded, "current_allocated_bytes")
    private_head_saving = legacy_delta - borrowed_delta
    teardown_exact = (
        int(final["current_allocated_bytes"]) == int(initial["current_allocated_bytes"])
        and int(final["active_allocations"]) == int(initial["active_allocations"])
    )
    bounded_runtime_residue = (
        int(final["current_allocated_bytes"]) - int(initial["current_allocated_bytes"]) <= 8
        and int(final["active_allocations"]) - int(initial["active_allocations"]) <= 1
        and int(after_all_close["current_allocated_bytes"])
        == int(residue_before_reuse["current_allocated_bytes"])
        and int(after_all_close["active_allocations"])
        == int(residue_before_reuse["active_allocations"])
    )
    checks = {
        "borrowed_private_head_absent": not borrowed_private_head_loaded,
        "legacy_private_head_present": legacy_private_head_loaded,
        "private_head_saving_gt_900mib": private_head_saving > 900 * 1024 * 1024,
        "borrowed_close_returns_to_session_baseline": (
            int(after_borrowed_close["current_allocated_bytes"])
            == int(session_loaded["current_allocated_bytes"])
        ),
        "legacy_close_returns_to_session_baseline": (
            int(after_legacy_close["current_allocated_bytes"])
            == int(session_loaded["current_allocated_bytes"])
        ),
        "closed_owner_rejected_before_launch": stale_pointer_rejected,
        "teardown_exact_or_bounded_non_growing_runtime_residue": (
            teardown_exact or bounded_runtime_residue
        ),
        "after_all_close_matches_final": after_all_close == final,
    }
    status = "passed" if all(checks.values()) and teardown_exact else (
        "passed_with_bounded_runtime_residue" if all(checks.values()) else "failed"
    )
    return {
        "schema": "hipengine.paro_mtp_lifecycle_gate.v1",
        "status": status,
        "performance_claim": False,
        "model": str(model),
        "backend": backend,
        "memory": {
            "initial": initial,
            "session_loaded": session_loaded,
            "borrowed_loaded": borrowed_loaded,
            "after_borrowed_close": after_borrowed_close,
            "legacy_loaded": legacy_loaded,
            "after_legacy_close": after_legacy_close,
            "residue_before_reuse": residue_before_reuse,
            "after_all_close": after_all_close,
            "teardown_exact": teardown_exact,
            "bounded_runtime_residue": bounded_runtime_residue,
            "borrowed_proposer_delta_bytes": borrowed_delta,
            "legacy_proposer_delta_bytes": legacy_delta,
            "private_head_saving_bytes": private_head_saving,
        },
        "checks": checks,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(model=args.model, backend=args.backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "saving_bytes": result["memory"]["private_head_saving_bytes"]}, sort_keys=True))
    return 0 if str(result["status"]).startswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
