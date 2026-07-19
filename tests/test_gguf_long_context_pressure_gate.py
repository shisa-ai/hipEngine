from __future__ import annotations

from scripts import gguf_long_context_pressure_gate as gate


def test_server_identity_matches_reused_production_workload_driver() -> None:
    assert gate._SERVED_MODEL_NAME == "qwen35-production-load"


def test_pool_plan_covers_mixed_and_forces_pressure_rejection() -> None:
    plan = gate.build_pool_plan(
        decode_tokens=32,
        longer_context_tokens=65_536,
    )

    assert plan.pages_by_context == {
        1_024: 5,
        4_096: 17,
        32_768: 129,
        65_536: 257,
    }
    assert plan.initial_pages == plan.low_water_pages == 5
    assert plan.pressure_high_water_pages == 134
    assert plan.mixed_high_water_pages == 519
    assert plan.chunk_pages == 5

    assert plan.pressure_high_water_pages - plan.initial_pages == plan.pages_by_context[32_768]
    assert plan.pages_by_context[4_096] > plan.initial_pages


def test_workload_plan_covers_each_concurrent_context_and_mixed_rows() -> None:
    workloads = gate.build_workload_specs(
        decode_tokens=32,
        longer_context_tokens=65_536,
    )

    assert tuple(workloads) == (
        "context_1k_c2",
        "context_4k_c2",
        "context_32k_c2",
        "mixed_1k_4k_32k",
        "context_64k_c2",
        "graph_seed_32k_c1",
        "graph_regrow_32k_c1",
    )
    assert [row.prompt_length for row in workloads["context_32k_c2"]] == [32_768, 32_768]
    assert [row.prompt_length for row in workloads["mixed_1k_4k_32k"]] == [1_024, 4_096, 32_768]
    assert [row.prompt_length for row in workloads["context_64k_c2"]] == [65_536, 65_536]
    assert all(row.max_tokens == 32 for rows in workloads.values() for row in rows)


def _passing_packet_inputs():
    plan = gate.build_pool_plan(decode_tokens=32, longer_context_tokens=65_536)
    workloads = {
        name: {"passed": True}
        for name in gate.build_workload_specs(
            decode_tokens=32,
            longer_context_tokens=65_536,
        )
    }
    pressure = {
        "passed": True,
        "long_outcome": "completed",
        "candidate_outcome": "rejected",
        "candidate_error_code": "engine_busy",
        "candidate_error_status_code": 429,
        "candidate_done_sentinel": True,
        "candidate_admission": {
            "resource": "device_kv_pool",
            "requested_units": plan.pages_by_context[4_096],
            "current_units": plan.pressure_high_water_pages,
            "capacity_units": plan.pressure_high_water_pages,
        },
    }
    final_pool = {
        "current_pages": plan.low_water_pages,
        "free_pages": plan.low_water_pages,
        "refcounted_pages": 0,
        "pinned_pages": 0,
        "grow_events": 6,
        "grow_failures": 1,
        "shrink_events": 6,
    }
    graph_delta = {"captures": 2, "replays": 8, "invalidations": 2}
    return plan, workloads, pressure, final_pool, graph_delta


def test_packet_gate_requires_exact_pressure_lifecycle_and_fresh_block_ids() -> None:
    plan, workloads, pressure, final_pool, graph_delta = _passing_packet_inputs()

    result = gate.evaluate_packet(
        plan=plan,
        workloads=workloads,
        pressure=pressure,
        final_pool=final_pool,
        graph_delta=graph_delta,
        pressure_block_ids=(100, 101, 102),
        regrow_block_ids=(200, 201, 202),
    )

    assert result == {"passed": True, "failure_reasons": []}


def test_packet_gate_fails_closed_on_wrong_admission_or_stale_logical_ids() -> None:
    plan, workloads, pressure, final_pool, graph_delta = _passing_packet_inputs()
    pressure["candidate_admission"]["requested_units"] = 16

    result = gate.evaluate_packet(
        plan=plan,
        workloads=workloads,
        pressure=pressure,
        final_pool=final_pool,
        graph_delta=graph_delta,
        pressure_block_ids=(100, 101, 102),
        regrow_block_ids=(102, 200, 201),
    )

    assert result["passed"] is False
    assert result["failure_reasons"] == [
        "pressure_admission_metadata_mismatch",
        "regrow_reused_retired_logical_block_ids",
    ]
