"""Unit tests for scripts/mtp_cycle_accounting.py pass reconstruction.

Synthetic raw fixture: width 5, one prompt, K=3, one 4+1 subgroup partition
and a tail-shrunk final cycle. No GPU required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mtp_cycle_accounting as mca  # noqa: E402


def _rec(candidates, target_rows, target_ms, accepted):
    cycles = len(candidates)
    return {
        "specdec2_mtp2_cycles": cycles,
        "specdec2_mtp2_candidate_counts": list(candidates),
        "specdec2_mtp2_accepted_counts": list(accepted),
        "specdec2_mtp2_target_batch_calls": cycles,
        "specdec2_mtp2_target_physical_rows": list(target_rows),
        "specdec2_mtp2_target_pass_ms": list(target_ms),
        "specdec2_mtp2_target_ms": sum(target_ms),
        "specdec2_mtp2_accept_pass_ms": [10.0] * cycles,
        "specdec2_mtp2_accept_ms": 10.0 * cycles,
        "specdec2_mtp2_provider_update_ms": 0.0,
        "specdec2_mtp2_proposal_batch_calls": cycles,
        "specdec2_mtp2_proposal_ms": 1.0 * cycles,
        "specdec2_mtp2_proposal_physical_rows": [4] * cycles,
        "specdec2_mtp2_selected_commit_ms": 0.0,
        "specdec2_mtp2_prompt_streaming": False,
    }


def _cell(width, records, generated):
    return {
        "prompt_id": "synthetic",
        "width": width,
        "exact": True,
        "mtp_engaged": True,
        "mtp_budget_conformed": True,
        "mtp": {
            "generated_tokens": generated,
            "wall_seconds": 1.0,
            "resident_observability": {
                "routes": {"recent_completed": records}
            },
        },
        "ar": {"wall_seconds": 0.5, "generated_tokens": generated},
    }


def test_partition_and_tail_shrink_reconstruction():
    # ticks 0..1: 4+1 partition -> rows 16 (members A-D) and 4 (member E)
    # tick 2 (tail): E shrinks to 0 candidates -> rows 16 + 1
    a = _rec([3, 3, 3], [16, 16, 16], [100.0, 110.0, 90.0], [2, 1, 0])
    b = _rec([3, 3, 3], [16, 16, 16], [100.0, 110.0, 90.0], [2, 1, 0])
    c = _rec([3, 3, 3], [16, 16, 16], [100.0, 110.0, 90.0], [2, 1, 0])
    d = _rec([3, 3, 3], [16, 16, 16], [100.0, 110.0, 90.0], [2, 1, 0])
    e = _rec([3, 3, 0], [4, 4, 1], [20.0, 21.0, 5.0], [0, 0, 0])
    # A..D: 1+3+3 = 7 each; E: 1+0+3 = 4 -> total 32
    cell = _cell(5, [a, b, c, d, e], 32)
    row = mca._analyze_cell(cell)
    assert row["cycles"] == 3
    assert row["physical_target_passes"] == 6
    assert row["target_pass_shapes"] == {"1": 1, "4": 2, "16": 3}
    assert row["target_rows_total"] == 3 * 16 + 2 * 4 + 1
    assert abs(row["target_ms_total"] - (100.0 + 110.0 + 90.0 + 20.0 + 21.0 + 5.0)) < 1e-6
    assert row["accepted_draft_tokens"] == 12
    assert row["committed_identity_residual_tokens"] == 0


def test_inconsistent_bucket_raises():
    a = _rec([3], [16], [100.0], [1])
    b = _rec([3], [16], [100.0], [1])
    cell = _cell(2, [a, b], 6)  # rows claim 16 but only 2*4 member rows
    with pytest.raises(AssertionError):
        mca._analyze_cell(cell)
