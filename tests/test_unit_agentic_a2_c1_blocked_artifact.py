from __future__ import annotations

import json
from pathlib import Path

import pytest


ARTIFACT = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a2-c1-processed-argmax-blocked.json"
)


def test_agentic_a2_c1_attempt_fails_closed_on_processed_argmax_boundary() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["kind"] == "gfx1100_agentic_a2_c1_blocked"
    assert payload["status"] == "blocked_unsupported_sampler_mode"
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is False
    assert payload["source_clean_and_pushed"] is True
    assert payload["control"]["sha256"] == (
        "29133f5fb0fa36f0f83fe34565ad7df93214b8eb7e035ac56b5413713e495f3f"
    )
    assert payload["attempted_protocol"]["full_matrix_stopped"] is True
    assert payload["acceptance"] == {
        "all_workloads_complete": False,
        "performance_row_retained": False,
        "radix_reuse_observed": False,
        "target_gpu0_exclusive_all_rows": False,
        "three_pairs_complete": False,
        "variance_qualified": False,
    }

    blocker = payload["blocker"]
    assert blocker["sampler_mode"] == "processed_argmax"
    assert blocker["fallback_reason"] == "sampling_unsupported"
    assert blocker["turns"] == 4
    for field in (
        "eligible_turns",
        "lookups",
        "hits",
        "reused_tokens",
        "avoided_prefill_tokens",
        "state_clone_bytes",
        "cache_resident_bytes",
    ):
        assert blocker[field] == 0

    off_rows = payload["turn_records"]["off"]
    radix_rows = payload["turn_records"]["radix"]
    assert len(off_rows) == len(radix_rows) == 4
    assert [row["prompt"] for row in off_rows] == [row["prompt"] for row in radix_rows]
    assert [row["output"]["generated_token_ids"] for row in off_rows] == [
        row["output"]["generated_token_ids"] for row in radix_rows
    ]
    assert all(
        row["backend"]["sampler_mode"] == "processed_argmax"
        and row["prefix"]["fallback_reason"] == "sampling_unsupported"
        and row["prefix"]["eligible"] is False
        and row["prefix"]["hit"] is False
        for row in radix_rows
    )

    metrics = payload["diagnostic_only_metrics"]
    assert metrics["radix_over_off_active_sse_rate"] == pytest.approx(
        metrics["radix"]["active_sse_exact_generated_tok_s"]
        / metrics["off"]["active_sse_exact_generated_tok_s"]
    )
    assert metrics["radix_minus_off_tool_ready_ms"] == pytest.approx(
        metrics["radix"]["buffered_tool_ready_p50_ms"]
        - metrics["off"]["buffered_tool_ready_p50_ms"]
    )
