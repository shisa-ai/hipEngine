from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARTIFACT = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a2-prefix-decision.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_a2_decision_keeps_regressive_radix_default_off() -> None:
    payload = _load(ARTIFACT)

    assert payload["kind"] == "gfx1100_agentic_a2_prefix_decision"
    assert payload["status"] == "rejected_default_off"
    assert payload["passed"] is False
    assert payload["measurement_valid"] is True
    assert payload["correctness_claim"] is True
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is True
    assert payload["protocol"]["metric_scope"] == "active_sse_wave_wall_s"
    assert payload["protocol"]["logical_concurrency_funnel"] == [1, 4, 8]
    assert payload["protocol"]["c1_measured_pairs_per_family"] == 3
    assert payload["protocol"]["one_complete_warmup_per_condition"] is True

    source_payloads = {}
    for name, source in payload["inputs"].items():
        source_path = Path(source["artifact"])
        assert source["sha256"] == _sha256(source_path)
        source_payloads[name] = _load(source_path)

    c1 = source_payloads["c1_paired"]
    assert c1["measurement_valid"] is True
    for family, decision_row in payload["c1_results"]["families"].items():
        source_row = c1["families"][family]
        assert decision_row["off_rate_tok_s_median"] == source_row["off"][
            "active_sse_exact_generated_tok_s"
        ]["median"]
        assert decision_row["radix_rate_tok_s_median"] == source_row["radix"][
            "active_sse_exact_generated_tok_s"
        ]["median"]
        assert decision_row["paired_rate_delta_percent_median"] == source_row[
            "paired_radix_vs_off"
        ]["active_sse_rate_delta_percent"]["median"]
        assert decision_row["paired_tool_ready_delta_percent_median"] == source_row[
            "paired_radix_vs_off"
        ]["tool_ready_p50_delta_percent"]["median"]
        assert decision_row["paired_rate_delta_percent_median"] < 0.0
        assert decision_row["paired_tool_ready_delta_percent_median"] > 0.0
        assert decision_row["a1_no_material_regression_passed"] is False

    c1_results = payload["c1_results"]
    assert c1_results["all_correctness_gates_passed"] is True
    assert c1_results["all_final_ownership_bounded"] is True
    assert c1_results["all_target_gpu0_exclusive"] is True
    assert c1_results["all_variance_gates_passed"] is False
    assert c1_results["all_a1_no_material_regression_guards_passed"] is False
    assert c1_results["any_rate_improvement"] is False
    assert c1_results["any_tool_ready_improvement"] is False

    c48 = payload["c4_c8_disposition"]
    assert c48 == {
        "status": "skipped_failed_c1_prerequisite",
        "protocol_complete_skip": True,
        "c1_prerequisite_passed": False,
        "measurement_authorized": False,
        "gpu_measurements_started": False,
        "medium_c4_guard_evaluated": False,
        "medium_c4_guard_result": (
            "not_evaluated_after_failed_c1_prerequisite"
        ),
        "inferred_timing_used": False,
    }

    lifecycle = payload["lifecycle_pressure"]
    assert lifecycle["passed"] is True
    assert lifecycle["active_and_completed_sources_exact"] is True
    assert lifecycle["exact_survivor_ids_and_state_kv"] is True
    assert lifecycle["bounded_declared_cache_bytes"] is True
    assert lifecycle["zero_non_cache_final_owners"] is True
    assert lifecycle["eviction_and_refcount_drain_exact"] is True
    assert lifecycle["cancellation_disconnect_slow_deadline_fail_closed"] is True
    assert lifecycle["fork_rollback_explicitly_rejected_for_resident_state"] is True
    assert lifecycle["performance_promotion_allowed"] is False

    assert payload["acceptance"] == {
        "predeclared_primary_metric_improved": False,
        "every_exactness_gate_passed": True,
        "every_lifecycle_gate_passed": True,
        "variance_qualified": False,
        "c1_no_material_regression": False,
        "medium_c4_no_material_regression": False,
        "all_promotion_gates_passed": False,
        "promote_radix": False,
        "default_prefix_cache": "off",
        "radix_availability": "explicit_diagnostic_only",
    }
    assert len(payload["rejection_reasons"]) == 5
    assert payload["next_stage"]["stage"] == "A3"
