from __future__ import annotations

import json
from pathlib import Path

import pytest


ARTIFACT = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a2-c1-prefix-rejected.json"
)


def test_agentic_a2_c1_complete_matrix_rejects_sparse_regressive_radix() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["kind"] == "gfx1100_agentic_a2_c1_prefix_paired_rejected"
    assert payload["status"] == "rejected_c1_regression_and_sparse_reuse"
    assert payload["measurement_valid"] is True
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is True
    assert payload["source_clean_and_pushed"] is True
    assert payload["protocol"]["measured_pairs_per_family"] == 3
    assert payload["protocol"]["one_complete_warmup_per_condition"] is True
    assert payload["protocol"]["a1_control_sha256"] == (
        "29133f5fb0fa36f0f83fe34565ad7df93214b8eb7e035ac56b5413713e495f3f"
    )

    acceptance = payload["acceptance"]
    assert acceptance["accepted"] is False
    assert acceptance["all_correctness_gates_passed"] is True
    assert acceptance["all_final_ownership_bounded"] is True
    assert acceptance["all_target_gpu0_exclusive"] is True
    assert acceptance["all_variance_gates_passed"] is False
    assert acceptance["all_a1_no_material_regression_guards_passed"] is False

    expected = {
        "small_repo": {
            "off_rate": 13.060484564850627,
            "radix_rate": 4.727081126430852,
            "tool_ms": 5246.946613013279,
            "hits": 0,
            "turns": 12,
        },
        "growing_history": {
            "off_rate": 12.248230975746699,
            "radix_rate": 4.2155805734175775,
            "tool_ms": 6097.496198955923,
            "hits": 3,
            "turns": 24,
        },
        "medium_repo": {
            "off_rate": 3.8514606174211905,
            "radix_rate": 2.8376728283470665,
            "tool_ms": 6632.126365962904,
            "hits": 3,
            "turns": 18,
        },
    }
    for workload, values in expected.items():
        family = payload["families"][workload]
        off = family["off"]
        radix = family["radix"]
        assert off["active_sse_exact_generated_tok_s"]["count"] == 3
        assert radix["active_sse_exact_generated_tok_s"]["count"] == 3
        assert off["active_sse_exact_generated_tok_s"]["median"] == pytest.approx(
            values["off_rate"]
        )
        assert radix["active_sse_exact_generated_tok_s"]["median"] == pytest.approx(
            values["radix_rate"]
        )
        assert radix["tool_ready_p50_ms"]["median"] == pytest.approx(values["tool_ms"])
        assert radix["prefix"]["hits"] == values["hits"]
        assert radix["prefix"]["lookups"] == values["turns"]
        assert family["radix_vs_a1"]["no_material_regression_gate_passed"] is False

    conditions = payload["condition_evidence"]
    assert len(conditions) == 18
    assert all(row["validation_passed"] is True for row in conditions)
    assert all(row["target_gpu0_exclusive"] is True for row in conditions)
    assert min(row["target_server_samples"] for row in conditions) >= 115
    assert all(
        row["final_ownership"][field] == 0
        for row in conditions
        for field in (
            "active_requests",
            "graph_owners",
            "kv_pinned_pages",
            "kv_refcounted_pages",
            "model_active_requests",
            "pending_requests",
            "session_count",
            "stream_producers",
            "workspace_owners",
        )
    )
    assert all(item["passed"] is True for item in payload["correctness_gate_artifacts"])
