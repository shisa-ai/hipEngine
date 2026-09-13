from __future__ import annotations

import json
from pathlib import Path
import statistics

import pytest


ARTIFACT = Path("benchmarks/results/2026-07-21-w7900-agentic-a1-repeated-baseline.json")


def test_repeated_agentic_a1_baseline_is_complete_and_reaggregates_sse_waves() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["kind"] == "gfx1100_agentic_a1_repeated_baseline"
    assert payload["status"] == "accepted_complete_baseline"
    assert payload["performance_claim"] is True
    assert payload["source_clean_and_pushed"] is True
    assert payload["hardware"]["target"]["rocm_index"] == 0
    assert payload["hardware"]["target"]["gpu"] == "AMD Radeon Pro W7900"
    assert payload["hardware"]["peer"]["rocm_index"] == 1
    assert payload["hardware"]["peer"]["activity_allowed"] is True
    assert payload["timing_rollup_fix"]["legacy_metrics_diagnostic_only"] is True

    configurations = payload["configurations"]
    assert {
        (row["workload"], row["context_tokens"], row["logical_concurrency"])
        for row in configurations
    } == {
        (workload, context, concurrency)
        for workload, context in (
            ("small_repo", 4096),
            ("growing_history", 4096),
            ("medium_repo", 10240),
        )
        for concurrency in (1, 4, 8)
    }

    total_turns = 0
    total_generated_ids = 0
    for configuration in configurations:
        assert configuration["warmup_runs"] == 1
        assert configuration["measured_runs"] == 3
        assert configuration["target_gpu0_exclusive"] is True
        assert configuration["all_correctness_gates_passed"] is True
        assert configuration["all_final_ownership_zero"] is True
        assert configuration["variance_gate_passed"] is True
        assert len(configuration["samples"]) == 3

        active_rates = []
        for sample in configuration["samples"]:
            assert sample["target_gpu0_exclusive"] is True
            assert sample["peer_gpu1_activity_allowed"] is True
            assert all(int(value) == 0 for value in sample["final_ownership"].values())
            records = sample["turn_records"]
            assert len(records) == sample["coverage"]["turns"]
            waves: dict[tuple[str, str, int], list[dict]] = {}
            generated_ids = 0
            for record in records:
                assert record["output"]["generated_token_ids_source"] == "response"
                assert record["output"]["sse_exact_ids_observed"] is True
                assert record["tool"]["arguments_json_valid"] is True
                assert record["tool"]["schema_valid"] is True
                generated_ids += len(record["output"]["generated_token_ids"])
                key = (
                    record["run_id"],
                    record["workload_id"],
                    int(record["turn_index"]),
                )
                waves.setdefault(key, []).append(record)
            active_wall = sum(
                max(row["timing"]["response_done_at_s"] for row in rows)
                - min(row["timing"]["submitted_at_s"] for row in rows)
                for rows in waves.values()
            )
            rollup = sample["corrected_rollup"]
            assert rollup["workload_wall_scope"] == (
                "first_submit_to_last_tool_result_submit_including_inter_turn_control"
            )
            assert rollup["active_sse_wave_count"] == len(waves)
            assert rollup["active_sse_wave_wall_s"] == pytest.approx(active_wall)
            assert rollup["active_sse_exact_generated_tok_s"] == pytest.approx(
                generated_ids / active_wall
            )
            assert generated_ids == sample["coverage"]["generated_tokens"]
            active_rates.append(rollup["active_sse_exact_generated_tok_s"])
            total_turns += len(records)
            total_generated_ids += generated_ids

        metrics = configuration["metrics"]
        assert metrics["active_sse_exact_generated_tok_s"]["samples"] == active_rates
        assert metrics["active_sse_exact_generated_tok_s"]["median"] == pytest.approx(
            statistics.median(active_rates)
        )
        assert metrics["active_sse_exact_generated_tok_s"][
            "stdev_percent_of_median"
        ] <= 5.0

    assert total_turns == 702
    assert total_generated_ids == 17316
    assert payload["correctness_gates"]["live_a1"]["turns"] == total_turns
    assert payload["correctness_gates"]["live_a1"][
        "exact_response_owned_generated_ids"
    ] == total_generated_ids
    assert payload["correctness_gates"]["linked_semantic_gate"]["passed"] is True
    assert payload["correctness_gates"]["linked_native_c8_state_oracle"]["passed"] is True
    assert payload["acceptance"] == {
        "active_sse_performance_baseline_retained": True,
        "all_27_measured_runs_valid": True,
        "all_correctness_gates_passed": True,
        "all_final_ownership_zero": True,
        "all_nine_configurations_complete": True,
        "all_target_gpu0_exclusive": True,
        "all_variance_gates_passed": True,
        "legacy_oracle_inclusive_rate_retained": False,
    }
