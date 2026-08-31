"""Unit tests for scripts/mtp_cycle_accounting.py pass reconstruction.

Synthetic raw fixture: width 5, one prompt, K=3, one 4+1 subgroup partition
and a tail-shrunk final cycle. No GPU required.
"""

from __future__ import annotations

import csv
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
        "specdec2_mtp2_target_pass_start_ns": [],
        "specdec2_mtp2_target_pass_end_ns": [],
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

    aggregate = mca._aggregate([row])
    assert aggregate["mtp_to_ar_ratio"] == pytest.approx(0.5)
    assert aggregate["observed_output_tokens_per_request_cycle"] == pytest.approx(32 / 15)
    assert aggregate["steady_committed_tokens_per_request_cycle"] == pytest.approx(1.8)
    assert aggregate["cycle_cost_ar_step_equivalents"] == pytest.approx(3.6)
    assert aggregate["observed_cycle_wall_ar_step_equivalents"] == pytest.approx(64 / 15)
    assert aggregate["stage_ms_per_cycle"]["proposal"] == pytest.approx(1.0)
    assert aggregate["stage_ms_per_cycle"]["accept"] == pytest.approx(10.0)


def test_target_window_reconstruction_and_kernel_family_curve(tmp_path: Path):
    a = _rec([3], [8], [5.0], [1])
    b = _rec([3], [8], [5.0], [1])
    for rec in (a, b):
        rec["specdec2_mtp2_target_pass_start_ns"] = [1_000]
        rec["specdec2_mtp2_target_pass_end_ns"] = [11_000]
        rec["specdec2_mtp2_cycle_profile_start_ns"] = [1_000]
        rec["specdec2_mtp2_cycle_profile_end_ns"] = [12_000]
    cell = _cell(2, [a, b], 6)
    windows = mca._target_windows_for_cell(cell)
    assert windows == [{"rows": 8, "start_ns": 1_000, "end_ns": 12_000}]

    trace = tmp_path / "kernel_trace.csv"
    with trace.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=("Kernel_Name", "Start_Timestamp", "End_Timestamp"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "Kernel_Name": "gguf_q4_t16_dense_wmma_prefill_bf16_kernel",
                    "Start_Timestamp": 2_000,
                    "End_Timestamp": 4_000,
                },
                {
                    "Kernel_Name": "q5_k_t16_dense_rowtile_gemv_kernel",
                    "Start_Timestamp": 4_000,
                    "End_Timestamp": 7_000,
                },
                {
                    "Kernel_Name": "q6_k_t16_qmicro_planar_gemv_bf16_kernel",
                    "Start_Timestamp": 7_000,
                    "End_Timestamp": 10_000,
                },
                {
                    "Kernel_Name": "__amd_rocclr_copyBuffer",
                    "Start_Timestamp": 10_000,
                    "End_Timestamp": 10_500,
                },
            )
        )
    curve = mca._kernel_family_row_curve([cell], trace)
    assert curve["8"]["passes"] == 1
    assert curve["8"]["family_ms_median"] == {
        "q4": pytest.approx(0.002),
        "q5": pytest.approx(0.003),
        "q6": pytest.approx(0.003),
    }
    assert curve["8"]["classified_kernel_fraction"] == pytest.approx(8 / 8.5)


def test_stage_and_kernel_family_classification():
    assert mca._model_stage("blk.63.ffn_down.weight") == "target"
    assert mca._model_stage("blk.64.ffn_down.weight") == "proposal"
    assert mca._model_stage("output.weight") == "shared_head"
    assert mca._model_stage("token_embd.weight") is None
    assert mca._kernel_quant_family("q4_k_t16_dense_rowtile_gemv_kernel") == "q4"
    assert mca._kernel_quant_family(
        "qk_t16_selected_direct_gemv_kernel<unsigned short, 5, false>"
    ) == "q5"
    assert mca._kernel_quant_family(
        "gguf_k_prefill_out_rowtile_kernel<unsigned short, unsigned short, 6, 8>"
    ) == "q6"


def test_inconsistent_bucket_raises():
    a = _rec([3], [16], [100.0], [1])
    b = _rec([3], [16], [100.0], [1])
    cell = _cell(2, [a, b], 6)  # rows claim 16 but only 2*4 member rows
    with pytest.raises(AssertionError):
        mca._analyze_cell(cell)
