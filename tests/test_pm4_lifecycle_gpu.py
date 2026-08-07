"""Guarded safe lifecycle controls; no reset-prone submit/recreate stress."""

from __future__ import annotations

import ctypes
import os

import pytest

from scripts.pm4_lifecycle_repro import ReproConfig, run_reproducer


def _rocm_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("libhsa-runtime64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _rocm_available(), reason="requires ROCm HIP and public HSA runtimes"
)


def _require_gfx1100() -> None:
    if os.environ.get("HIPENGINE_HIP_ARCH", "gfx1100") != "gfx1100":
        pytest.skip("initial native PM4 lifecycle gate is gfx1100-only")


def test_safe_hsa_interop_timestamp_reuse_is_exact() -> None:
    _require_gfx1100()
    result = run_reproducer(
        ReproConfig(cycles=2, allocation_mode="hsa", timestamps=True)
    )

    assert result["status"] == "pass"
    assert result["summary"]["cycles_passed"] == 2
    assert result["summary"]["final_cleanup_passed"] is True
    assert all(cycle["correct"] is True for cycle in result["cycles"])
    assert all(cycle["executable_after"]["timestamp_bytes"] == 16 for cycle in result["cycles"])
    assert len({tuple(cycle["buffer_addresses"]) for cycle in result["cycles"]}) == 1


def test_safe_no_submit_queue_first_quarantine_records_generations() -> None:
    _require_gfx1100()
    result = run_reproducer(
        ReproConfig(
            cycles=2,
            queue_mode="recreate",
            resource_mode="recreate",
            allocation_mode="hsa",
            submit=False,
            quarantine_generations=1,
        )
    )

    assert result["status"] == "pass"
    assert result["summary"]["cycles_passed"] == 2
    assert result["summary"]["final_cleanup_passed"] is True
    queue_ids = [cycle["queue_before_retire"]["queue_id"] for cycle in result["cycles"]]
    assert len(set(queue_ids)) == 2
    assert all(cycle["context_after_queue_retire"]["queue_id"] == 0 for cycle in result["cycles"])
    assert all(cycle["context_after_queue_retire"]["children"] == 4 for cycle in result["cycles"])
