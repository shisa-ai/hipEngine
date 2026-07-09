from __future__ import annotations

import json

from hipengine.benchmark.speculative import (
    SpeculativeBenchmarkModels,
    acceptance_histogram,
    acceptance_summary,
    aggregate_speculative_rows,
    build_speculative_artifact,
    first_mismatch,
    normalize_speculative_row,
    record_speculative_graph_shape_stats,
    schema_fixture_row,
)


def test_first_mismatch_detects_value_and_length_drift() -> None:
    assert first_mismatch([1, 2, 3], [1, 2, 3]) is None
    assert first_mismatch([1, 2, 3], [1, 7, 3]) == 1
    assert first_mismatch([1, 2, 3], [1, 2]) == 2


def test_acceptance_summary_matches_parent_metric_shape() -> None:
    lengths = [0, 1, 2, 2, 4]
    assert acceptance_histogram(lengths) == {"0": 1, "1": 1, "2": 2, "4": 1}
    summary = acceptance_summary(lengths, max_positions=4)
    assert summary["steps"] == 5
    assert summary["accepted_output_tokens"] == 9
    assert summary["avg_accept_length"] == 1.8
    assert summary["exact_rate_0_to_n"]["2"] == 0.4
    assert summary["ge_rate_1_to_n"]["2"] == 0.6
    assert summary["multi_token_acceptance_rate"] == 0.6


def test_normalize_speculative_row_records_required_dflash_metrics() -> None:
    raw = schema_fixture_row()
    raw["config"]["profile_route"] = "chain"
    row = normalize_speculative_row(raw)

    assert row["ar"]["same_session_control"] is True
    assert row["config"]["profile_route"] == "chain"
    assert row["ar"]["same_process_control"] is True
    assert row["ar"]["decode_tok_s"] == 2.0
    assert row["spec"]["same_session_control"] is True
    assert row["spec"]["native_bulk_verifier"] is True
    assert row["spec"]["backend"] == "hip_gfx1151"
    assert row["spec"]["target_arch"] == "gfx1151"
    assert row["spec"]["decode_tok_s"] == 8 / 3
    assert row["spec"]["speedup_vs_ar"] == (8 / 3) / 2
    assert row["correctness"]["exact_match_ar"] is True
    assert row["correctness"]["finite_all_logits"] is True
    assert row["correctness"]["passed"] is True
    assert row["acceptance"]["accept_histogram"] == {"1": 1, "2": 2, "3": 1}
    assert row["spec"]["target_verify_rows_per_output_token"] == 2.0
    assert row["spec"]["target_verify_bucket_seconds"]["target_verify_forward"] == 1.50
    assert row["spec"]["phase_split"]["target_verify_fraction"] == 0.75
    assert row["spec"]["draft_context_phase_seconds"] == {
        "full_context_rebuild": 0.20,
        "append_materialize": 0.05,
        "query_only_drafter": 0.10,
    }
    assert row["spec"]["draft_native_phase_seconds"]["decoder_layers"] == 0.25
    assert row["spec"]["drafter_context_mode"] == "append_kv_query_only"
    assert row["spec"]["draft_phase_timing_mode"] == "synchronized"
    assert row["spec"]["proposal_trace_sample"][0]["accepted"] == 2
    assert row["spec"]["proposal_trace_count"] == 4
    assert row["spec"]["draft_kv_bytes"] == 576
    assert row["spec"]["draft_kv_capacity_tokens"] == 6
    assert row["d2h"]["scalar_reads"] == 4
    assert row["d2h"]["vector_reads"] == 1
    assert row["d2h"]["full_logits_readbacks"] == 0
    assert row["graph"]["status"] == "not_captured"
    assert row["memory"]["peak_allocated_bytes"] == 22_000_000_000


def test_aggregate_speculative_rows_preserves_exact_and_speed_gates() -> None:
    rows = [normalize_speculative_row(schema_fixture_row())]
    aggregate = aggregate_speculative_rows(rows)

    assert aggregate["rows"] == 1
    assert aggregate["all_exact_match_ar"] is True
    assert aggregate["all_correctness_passed"] is True
    assert aggregate["all_finite_logits"] is True
    assert aggregate["target_verify_rows_per_output_token"] == 2.0
    assert aggregate["target_verify_bucket_seconds"]["target_verify_forward"] == 1.50
    assert aggregate["d2h"]["scalar_reads"] == 4
    assert aggregate["speed_gate_gt_1p10"] is True


def test_aggregate_uses_explicit_decode_tokens_when_samples_are_truncated() -> None:
    raw = schema_fixture_row()
    raw["decode_tokens"] = 64
    raw["ar"]["generated_ids"] = list(range(64))
    raw["spec"]["generated_ids"] = list(range(64))
    raw["ar"]["decode_seconds"] = 4.0
    raw["spec"]["decode_seconds"] = 8.0
    raw["spec"]["target_verify_rows"] = 128

    row = normalize_speculative_row(raw)
    assert len(row["ar"]["generated_sample"]) == 32

    aggregate = aggregate_speculative_rows([row])
    assert aggregate["decode_tokens"] == 64
    assert aggregate["ar_decode_tok_s"] == 16.0
    assert aggregate["spec_decode_tok_s"] == 8.0
    assert aggregate["target_verify_rows_per_output_token"] == 2.0


def test_record_speculative_graph_shape_stats_groups_runtime_bucket() -> None:
    shape_stats: dict[str, object] = {}
    graph = {
        "mode": "auto",
        "status": "replayed",
        "replayed": True,
        "validation_passed": True,
        "replay_count": 3,
        "bucket_key": {
            "rows": 5,
            "capture_width": 8192,
            "base_slot": 1,
            "chain_attn_mode": "batched",
            "linear_attn_mode": "chain_tloop",
            "batch_mode": "verify_chain",
        },
    }

    record_speculative_graph_shape_stats(
        shape_stats,
        graph,
        verifier_mode="native_bulk_bplus1",
        tree_mode="chain",
        context_tokens=512,
        active_budget=4,
        target_verify_bucket_seconds={"graph_replay": 0.125, "accept_summary": 0.03125},
    )
    record_speculative_graph_shape_stats(
        shape_stats,
        {**graph, "replay_count": 4},
        verifier_mode="native_bulk_bplus1",
        tree_mode="chain",
        context_tokens=516,
        active_budget=4,
        target_verify_bucket_seconds={"graph_replay": 0.125},
    )

    assert len(shape_stats) == 1
    entry = next(iter(shape_stats.values()))
    assert entry["cycles"] == 2
    assert entry["replayed_cycles"] == 2
    assert entry["status_counts"] == {"replayed": 2}
    assert entry["context_tokens_min"] == 512
    assert entry["context_tokens_max"] == 516
    assert entry["context_tokens_sample"] == [512, 516]
    assert entry["active_budget_counts"] == {"4": 2}
    assert entry["validation_passed"] is True
    assert entry["replay_count_max"] == 4
    assert entry["target_verify_bucket_seconds"]["graph_replay"] == 0.25
    assert entry["target_verify_bucket_seconds"]["accept_summary"] == 0.03125


def test_build_speculative_artifact_is_schema2_and_not_claim_for_fixture() -> None:
    artifact = build_speculative_artifact(
        run_tag="unit-dflash-contract",
        summary="unit contract",
        rows=[schema_fixture_row()],
        models=SpeculativeBenchmarkModels(target_path="/models/target", drafter_path="/models/drafter"),
        status="diagnostic",
        timestamp="2026-05-18T00:00:00+00:00",
        hardware={"gpu": "unit", "arch": "gfx1151"},
        software={"hipengine_commit": "abc", "hipengine_dirty": True},
        workload={"shape": "unit"},
        commands={"benchmark_contract": "unit"},
        synthetic_schema_fixture=True,
    )

    assert artifact["schema"] == 2
    assert artifact["speculative_schema"] == 1
    assert artifact["performance_claim"] is False
    assert artifact["synthetic_schema_fixture"] is True
    assert artifact["models"]["target"]["path"] == "/models/target"
    assert artifact["models"]["drafter"]["path"] == "/models/drafter"
    spec = artifact["measurements"]["rows"][0]["spec"]
    assert spec["target_forward_calls"] == 4
    assert spec["target_bulk_forward_calls"] == 4
    assert spec["target_serial_forward_calls"] == 0
    assert spec["target_bulk_rows"] == 20
    assert spec["target_forwards_per_draft_call"] == 1.0
    assert spec["gpu_accept_match_cpu"] is True
    assert artifact["measurements"]["aggregate"]["all_correctness_passed"] is True
    assert artifact["correctness_gate"]["passed"] is True
    assert artifact["baseline"]["type"] == "same_session_ar_control"
    assert artifact["decision_reason"] == artifact["decision"]["reason"]
    assert artifact["decision"]["accepted"] is False
    json.dumps(artifact)
