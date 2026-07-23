from __future__ import annotations

import pytest

from scripts.laguna_selected_q8_prefill_bench import (
    MODES,
    PROFILE_ROWS,
    _comparison_summary,
    _mode_order,
)


def _samples(*, split_scale: float = 1.0, q8_scale: float = 0.8):
    return {
        "split": {
            rows: [rows * split_scale, rows * split_scale * 1.01, rows * split_scale * 0.99]
            for rows in PROFILE_ROWS
        },
        "q8_dp4a": {
            rows: [rows * q8_scale, rows * q8_scale * 1.01, rows * q8_scale * 0.99]
            for rows in PROFILE_ROWS
        },
    }


def _tokens():
    return {
        mode: {rows: [1000 + rows] * 3 for rows in PROFILE_ROWS}
        for mode in MODES
    }


def test_selected_q8_mode_order_is_counterbalanced() -> None:
    for index, _rows in enumerate(PROFILE_ROWS):
        assert _mode_order(index, 0) == (
            MODES if index % 2 == 0 else tuple(reversed(MODES))
        )
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))


def test_selected_q8_comparison_accepts_inclusive_all_shape_win() -> None:
    summary = _comparison_summary(_samples(), _tokens())

    assert summary["performance_screen"]["pass"] is True
    assert summary["performance_screen"]["all_measured_shapes_strictly_faster"] is True
    assert summary["performance_screen"]["aggregate_speedup"] == pytest.approx(1.0 / 0.8)
    assert summary["performance_screen"]["quality_gate_required_for_promotion"] is True
    assert summary["next_token_diagnostic"]["all_modes_and_repetitions_agree"] is True


def test_selected_q8_comparison_fails_perf_closed_but_keeps_ids_diagnostic() -> None:
    samples = _samples()
    samples["q8_dp4a"][64] = [64.1, 64.2, 64.0]
    tokens = _tokens()
    tokens["q8_dp4a"][55][1] += 1

    summary = _comparison_summary(samples, tokens)

    assert summary["performance_screen"]["pass"] is False
    assert "candidate_not_faster_at_every_shape" in summary["performance_screen"][
        "failed_checks"
    ]
    assert summary["next_token_diagnostic"]["all_modes_and_repetitions_agree"] is False
