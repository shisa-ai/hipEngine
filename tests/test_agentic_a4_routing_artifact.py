from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPO_ROOT
    / "benchmarks/results/2026-07-22-w7900-agentic-a4-routing-screen-blocked.json"
)
PROTOCOL = (
    REPO_ROOT
    / "benchmarks/results/2026-07-22-w7900-agentic-a4-predeclared-protocol.json"
)
DECISION = (
    REPO_ROOT
    / "benchmarks/results/2026-07-22-w7900-agentic-a4-routing-decision.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a4_routing_screen_stops_fail_closed_before_promotion_matrix() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["kind"] == "gfx1100_agentic_a4_routing_screen_blocked"
    assert payload["status"] == "blocked_no_complete_slo_and_correctness_candidate"
    assert payload["passed"] is False
    assert payload["measurement_valid"] is False
    assert payload["correctness_claim"] is False
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is False
    assert payload["source"]["clean"] is True
    assert payload["protocol"]["sha256"] == _sha256(PROTOCOL)

    selection = payload["selection"]
    assert selection["selected"] is None
    assert selection["selection_error"] == "no SLO-passing tuning candidate"
    assert selection["complete_all_pass_candidates"] == 0
    candidates = selection["candidate_results"]
    assert len(candidates) == 8
    assert all(row["complete"] is True for row in candidates)
    assert all(row["all_repetitions_passed"] is False for row in candidates)
    assert all(len(row["samples"]) == 3 for row in candidates)

    control = next(
        row for row in candidates if row["aggregate"]["candidate_id"] == "pd256_b1_w0_control"
    )
    assert control["samples"][2]["failure_reasons"] == ["slo_ttft_p95_failed"]
    assert control["samples"][2]["exact_rows"] == 12

    mismatches = [
        mismatch
        for row in candidates
        for sample in row["samples"]
        for mismatch in sample["mismatches"]
    ]
    assert len(mismatches) == 9
    assert {row["label"] for row in mismatches} == {"fixed-0011"}
    assert {row["first_mismatch_index"] for row in mismatches} <= {20, 21, 22, 23, 24}

    workload = payload["workload"]
    assert workload["total_requests"] == 288
    assert workload["observed_generated_tokens"] == 8640
    assert workload["exact_generated_tokens"] == 8208
    assert workload["route"]["all_candidate_routes_passed"] is True
    assert workload["route"]["serial_fallback_steps"] == 0
    assert workload["route"]["resident_fallback_requests"] == 0

    disposition = payload["matrix_disposition"]
    assert disposition["stage_1_mixed_arrival_complete"] is True
    assert disposition["stage_1_candidate_samples"] == 24
    assert disposition["stage_1_all_candidates_rejected"] is True
    assert disposition["stage_2_occupancy_c1_c2_c4_c8_skipped"] is True
    assert disposition["stage_3_agentic_guard_skipped"] is True
    assert disposition["stage_3_safety_packet_skipped"] is True
    assert disposition["inferred_timing_used"] is False
    assert disposition["default_change_authorized"] is False

    ownership = payload["final_ownership"]
    assert ownership["final_ownership_passed"] is True
    assert ownership["memory_recovery"]["passed"] is True
    assert ownership["resident_requests"]["pending"] == 0
    assert ownership["resident_requests"]["active"] == 0
    assert ownership["resident_requests"]["admitted_current"] == 0
    assert ownership["resident_requests"]["admitted_total"] == 288
    assert ownership["resident_requests"]["reclaimed_total"] == 288
    assert ownership["model_active_requests"] == 0
    assert ownership["generation_queue_depth"] == 0
    assert ownership["generation_active_requests"] == 0


def test_a4_final_decision_binds_failed_screen_and_preserves_defaults() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))

    assert payload["kind"] == "gfx1100_agentic_a4_routing_decision"
    assert payload["status"] == "final_blocked_no_routing_candidate"
    assert payload["passed"] is False
    assert payload["measurement_valid"] is False
    assert payload["correctness_claim"] is False
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is False
    assert payload["evidence"]["protocol"]["sha256"] == _sha256(PROTOCOL)
    assert payload["evidence"]["routing_screen"]["sha256"] == _sha256(ARTIFACT)

    assert payload["screen_coverage"]["candidate_count"] == 8
    assert payload["screen_coverage"]["candidate_samples"] == 24
    assert payload["screen_coverage"]["requests"] == 288
    assert payload["screen_coverage"]["observed_generated_tokens"] == 8640
    assert payload["screen_coverage"]["fully_exact_row_generated_tokens"] == 8208
    assert len(payload["candidate_disposition"]) == 8
    assert all(row["result"].startswith("rejected_") for row in payload["candidate_disposition"])

    blocker = payload["correctness_blocker"]
    assert blocker["mismatch_count"] == 9
    assert blocker["affected_request"] == "fixed-0011"
    assert blocker["shape"] == "p512/d48"
    assert blocker["expected_token_id"] == 9710
    assert blocker["correct_prefix_before_first_mismatch_tokens"] == {
        "min": 20,
        "max": 24,
    }

    acceptance = payload["acceptance"]
    assert acceptance["stage_1_screen_complete"] is True
    assert acceptance["all_candidates_complete"] is True
    assert acceptance["complete_all_pass_candidates"] == 0
    assert acceptance["candidate_selected"] is False
    assert acceptance["occupancy_c1_c2_c4_c8_matrix_authorized"] is False
    assert acceptance["agentic_strict_tool_guard_authorized"] is False
    assert acceptance["cancellation_backpressure_overload_guard_authorized"] is False
    assert acceptance["default_change_authorized"] is False
    assert acceptance["inferred_timing_used"] is False
    assert acceptance["final_ownership_zero"] is True

    assert payload["runtime_defaults"] == {
        "changed": False,
        "prefill_decode_policy": "protect_decode",
        "prefill_chunk_tokens": 256,
        "fair_prefill_burst_chunks": 1,
        "generation_batch_window_ms": 0.0,
        "prefix_cache": "off",
        "native_gpu_sampler": False,
    }
    assert payload["matrix_disposition"]["published_performance_rows"] == 0
