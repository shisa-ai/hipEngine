from __future__ import annotations

import pytest

from scripts.pm4_graph_bench import (
    _rotation,
    _summarize_runs,
    _transport_spec,
    _validate_cross_transport,
    build_parser,
)


def test_pm4_graph_bench_rotates_all_modes_without_duplication() -> None:
    modes = ("hipgraph", "aql", "pm4")

    assert _rotation(modes, 0) == modes
    assert _rotation(modes, 1) == ("aql", "pm4", "hipgraph")
    assert _rotation(modes, 2) == ("pm4", "hipgraph", "aql")
    assert _rotation(modes, 3) == modes


def test_pm4_graph_bench_summary_separates_issue_and_synchronized_wall() -> None:
    runs = [
        {
            "steps": 4,
            "replay_wall_ms": 40.0,
            "host_call_ns": [100, 200, 300, 400],
            "synchronized_step_ns": [9_000_000, 10_000_000, 11_000_000, 10_000_000],
            "prefill_ms": 5.0,
        },
        {
            "steps": 4,
            "replay_wall_ms": 44.0,
            "host_call_ns": [200, 300, 400, 500],
            "synchronized_step_ns": [10_000_000, 11_000_000, 12_000_000, 11_000_000],
            "prefill_ms": 7.0,
        },
        {
            "steps": 4,
            "replay_wall_ms": 36.0,
            "host_call_ns": [50, 100, 150, 200],
            "synchronized_step_ns": [8_000_000, 9_000_000, 10_000_000, 9_000_000],
            "prefill_ms": 6.0,
        },
    ]

    summary = _summarize_runs(runs, capture_ms=8.0)

    assert summary["median_replay_ms_per_token"] == pytest.approx(10.0)
    assert summary["median_replay_tok_s"] == pytest.approx(100.0)
    assert summary["median_host_call_us"] == pytest.approx(0.25)
    assert summary["median_synchronized_step_ms"] == pytest.approx(10.0)
    assert summary["capture_inclusive_ms_per_token"] == pytest.approx(12.0)
    assert summary["median_prefill_ms"] == pytest.approx(6.0)


def test_pm4_graph_bench_stateful_register_comparison_is_explicit_and_default_off() -> None:
    default = build_parser().parse_args([])
    candidate = build_parser().parse_args(["--transports", "pm4", "pm4_stateful"])

    assert default.transports == ["hipgraph", "aql", "pm4"]
    assert candidate.transports == ["pm4", "pm4_stateful"]
    assert _transport_spec("pm4") == ("pm4", False)
    assert _transport_spec("pm4_stateful") == ("pm4", True)
    with pytest.raises(ValueError, match="unknown benchmark transport"):
        _transport_spec("pm4_magic")


def test_pm4_graph_bench_cross_transport_gate_requires_exact_state_logits_and_tokens() -> None:
    runs = {
        mode: [
            {
                "final_token_id": 9707,
                "state_sha256": "state",
                "final_logits_sha256": "logits",
            }
        ]
        for mode in ("hipgraph", "aql", "pm4")
    }

    accepted = _validate_cross_transport(runs, expected_token_id=9707)
    assert accepted["passed"] is True
    assert accepted["state_sha256"] == "state"
    assert accepted["final_logits_sha256"] == "logits"

    runs["pm4"][0]["state_sha256"] = "drift"
    rejected = _validate_cross_transport(runs, expected_token_id=9707)
    assert rejected["passed"] is False
    assert rejected["state_exact"] is False
