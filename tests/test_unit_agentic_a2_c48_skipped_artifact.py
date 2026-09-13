from __future__ import annotations

import json
from pathlib import Path


ARTIFACT = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a2-c48-prefix-skipped.json"
)


def test_agentic_a2_c48_is_skipped_only_after_c1_gate_fails() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["kind"] == "gfx1100_agentic_a2_c48_prefix_skipped"
    assert payload["status"] == "skipped_failed_c1_prerequisite"
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is False
    assert payload["measurement_valid"] is False
    assert payload["planned_protocol"]["logical_concurrency"] == [4, 8]
    assert payload["planned_protocol"]["measured_pairs_per_family"] == 3

    prerequisite = payload["prerequisite"]
    assert prerequisite["task"] == 225
    assert prerequisite["required"] == "C1 prefix A/B passes"
    assert prerequisite["passed"] is False
    assert prerequisite["artifact_sha256"] == (
        "8b8e3eb092bdb2061ee1de63a231c4a8fcb5d9a833c7587071d3b038406f5cb2"
    )
    assert prerequisite["all_a1_no_material_regression_guards_passed"] is False
    assert prerequisite["all_variance_gates_passed"] is False
    assert all(
        delta < -20.0
        for delta in prerequisite["paired_active_sse_delta_percent"].values()
    )
    assert all(
        delta > 30.0
        for delta in prerequisite["paired_tool_ready_p50_delta_percent"].values()
    )

    assert payload["acceptance"] == {
        "c1_prerequisite_passed": False,
        "c4_c8_measurement_authorized": False,
        "partial_matrix_used_for_promotion": False,
        "gpu_measurements_started": False,
    }
