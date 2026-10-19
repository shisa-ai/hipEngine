"""CPU contracts for the live prefill-interference report's timing scopes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    "qwen38fn_prefill_interference",
    Path(__file__).parents[1] / "scripts/qwen38fn_prefill_interference.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_interference_counts_the_gap_spanning_the_entire_prefill():
    row = {"token_times": [0.0, 0.1, 2.0, 2.1]}
    assert _module.token_intervals(row, (0.5, 1.5)) == pytest.approx([1900.0])
    assert _module.token_intervals(row) == pytest.approx([100.0, 1900.0, 100.0])


def test_no_intervals_are_not_reported_as_zero_latency():
    assert _module.latency_summary([]) == {
        "intervals": 0, "median_ms": None, "p95_ms": None,
        "p99_ms": None, "maximum_ms": None,
    }


def test_wave_keeps_prompt_ttft_decode_interval_and_group_wall_separate():
    peers = [{"token_times": [1.0, 1.5, 3.0], "generated_ids": [1, 2, 3]}]
    arrival = {
        "started_at": 1.6, "token_times": [2.6, 2.7],
        "generated_ids": [4, 5], "prompt_tokens": 100,
    }
    summary = _module.summarize_wave(peers, arrival, 4.0)
    assert summary["arrival_ttft_ms"] == pytest.approx(1000)
    assert summary["arrival_effective_prompt_tok_s"] == pytest.approx(100)
    assert summary["peer_decode_interval_tok_s"] == pytest.approx(1)
    assert summary["group_end_to_end_tok_s"] == pytest.approx(1.25)
    assert summary["peer_interference_inter_token_latency"]["maximum_ms"] == pytest.approx(1500)
    assert summary["peer_interference_inter_token_latency"]["intervals"] == 1


def test_percentiles_interpolate_without_rounding_away_tail_samples():
    assert _module.percentile([5.0, 1.0, 2.0, 100.0], 0.95) == pytest.approx(85.75)
