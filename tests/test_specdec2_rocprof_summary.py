from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.specdec2_rocprof_summary import (
    _interval_union_ns,
    classify_kernel,
    extract_group_telemetry,
    summarize_phase,
)


def _trace_row(name: str, start: int, end: int) -> dict:
    return {
        "name": name,
        "start_ns": start,
        "end_ns": end,
        "duration_ns": end - start,
        "raw": {
            "Grid_Size_X": "1",
            "Grid_Size_Y": "1",
            "Grid_Size_Z": "1",
            "Workgroup_Size_X": "64",
            "VGPR_Count": "16",
            "Scratch_Size": "0",
            "LDS_Block_Size": "0",
        },
    }


def _child() -> dict:
    route = {
        "specdec2_mtp2_used": True,
        "specdec2_mtp2_cycles": 2,
        "specdec2_mtp2_proposal_ms": 20.0,
        "specdec2_mtp2_target_ms": 40.0,
        "specdec2_mtp2_provider_update_ms": 4.0,
        "specdec2_mtp2_accept_ms": 30.0,
        "specdec2_mtp2_accept_enqueue_ms": 1.0,
        "specdec2_mtp2_accept_upload_ms": 2.0,
        "specdec2_mtp2_accept_tail_ms": 3.0,
        "specdec2_mtp2_selected_commit_ms": 8.0,
        "specdec2_mtp2_target_readback_ms": 18.0,
        "specdec2_mtp2_candidate_readback_ms": 0.0,
        "specdec2_mtp2_proposal_physical_rows": [2, 2],
        "specdec2_mtp2_target_physical_rows": [8, 6],
        "specdec2_mtp2_recoverable_failures": 0,
        "specdec2_mtp2_candidate_d2h_after_target": 0,
    }
    return {
        "workload": {"service_capacity": 8},
        "cells": [
            {
                "concurrency": 2,
                "arms": {
                    "specdec2": {
                        "status": "complete",
                        "realized_route": "specdec2",
                        "recent_routes": [route, deepcopy(route)],
                        "stage_ledger": {
                            "totals_seconds": {"cycle_total": 0.1},
                            "call_counts": {"cycle_total": 2},
                        },
                        "generated_tokens": 12,
                        "complete_wall_seconds": 0.2,
                    }
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("name", "family"),
    [
        (
            "gguf_k_raw_mmq32_q8_1_d4s4_f32_kernel<5, unsigned short>",
            "q5_t16",
        ),
        (
            "gguf_q5_k_mmq_i64_j32_k256_q8_1_ds4_kernel<unsigned short>",
            "q5_t16",
        ),
        ("q8_1_d4s4_f32_quantize_bf16_kernel", "q5_activation_quant"),
        ("q8_1_ds4_quantize_bf16_kmajor_kernel", "q5_activation_quant"),
    ],
)
def test_q5_mmq_operation_kernels_have_distinct_profile_families(
    name: str,
    family: str,
) -> None:
    assert classify_kernel(name) == family


def test_interval_union_does_not_double_count_overlapping_streams_or_window_gaps() -> None:
    rows = [
        _trace_row("a", 10, 40),
        _trace_row("b", 20, 50),
        _trace_row("c", 110, 130),
    ]

    assert _interval_union_ns(rows, [(0, 100), (100, 200)]) == 60


def test_phase_summary_uses_final_measured_occurrence_after_same_named_warmup() -> None:
    warmup = _trace_row("warmup_kernel", 10, 40)
    measured = _trace_row("q4_k_t16_kernel", 1010, 1050)

    summary = summarize_phase(
        marker_names=["same_marker"],
        all_markers={"same_marker": [(0, 100), (1000, 1100)]},
        kernels=[warmup, measured],
        hip_api=[],
        copies=[],
        allocations=[],
    )

    assert summary["host_marker_wall_ms"] == pytest.approx(0.0001)
    assert summary["kernel_sum_ms"] == pytest.approx(0.00004)
    assert summary["kernel_interval_union_ms"] == pytest.approx(0.00004)
    assert summary["kernel_families"][0]["name"] == "q4_t16"
    assert summary["kernel_families"][0]["ms_per_phase_call"] == pytest.approx(0.00004)
    assert summary["top_kernels"][0]["name"] == "q4_k_t16_kernel"

    fused = _trace_row("gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel", 1010, 1050)
    fused_summary = summarize_phase(
        marker_names=["same_marker"],
        all_markers={"same_marker": [(1000, 1100)]},
        kernels=[fused],
        hip_api=[],
        copies=[],
        allocations=[],
    )
    assert fused_summary["kernel_families"][0]["name"] == "q4_t16"


def test_group_telemetry_deduplicates_copied_request_counters() -> None:
    group = extract_group_telemetry(_child(), expected_concurrency=2)

    assert group["request_rows"] == 2
    assert group["physical_cycles"] == 2
    assert group["service_capacity"] == 8
    assert group["per_cycle_ms"] == {
        "proposal_ms": 10.0,
        "target_submit_ms": 20.0,
        "provider_update_ms": 2.0,
        "accept_window_ms": 15.0,
        "accept_enqueue_ms": 0.5,
        "accept_upload_ms": 1.0,
        "accept_tail_ms": 1.5,
        "selected_commit_ms": 4.0,
        "target_readback_ms": 9.0,
        "candidate_readback_ms": 0.0,
        "nonoverlap_named_sum_ms": 47.0,
    }

    malformed = _child()
    malformed["cells"][0]["arms"]["specdec2"]["recent_routes"][1][
        "specdec2_mtp2_target_ms"
    ] = 41.0
    with pytest.raises(ValueError, match="counter differs"):
        extract_group_telemetry(malformed, expected_concurrency=2)
