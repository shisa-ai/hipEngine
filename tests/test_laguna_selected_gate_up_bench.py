from __future__ import annotations

import pytest

from scripts.laguna_selected_gate_up_bench import (
    MODES,
    PROFILE_ROWS,
    _comparison_summary,
    _mode_order,
)


def _samples(*, split_scale: float = 1.0, fused_scale: float = 0.9):
    return {
        "split": {
            rows: [rows * split_scale, rows * split_scale * 1.01, rows * split_scale * 0.99]
            for rows in PROFILE_ROWS
        },
        "fused_silu": {
            rows: [rows * fused_scale, rows * fused_scale * 1.01, rows * fused_scale * 0.99]
            for rows in PROFILE_ROWS
        },
    }


def _tokens():
    return {
        mode: {rows: [1000 + rows] * 3 for rows in PROFILE_ROWS}
        for mode in MODES
    }


def test_selected_gate_up_mode_order_is_counterbalanced() -> None:
    for index, _rows in enumerate(PROFILE_ROWS):
        assert _mode_order(index, 0) == (
            MODES if index % 2 == 0 else tuple(reversed(MODES))
        )
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))


def test_selected_gate_up_comparison_accepts_exact_all_shape_win() -> None:
    summary = _comparison_summary(_samples(), _tokens())

    assert summary["correctness"]["pass"] is True
    assert summary["promotion"]["pass"] is True
    assert summary["promotion"]["all_measured_shapes_strictly_faster"] is True
    assert summary["promotion"]["aggregate_speedup"] == pytest.approx(1.0 / 0.9)
    assert all(
        shape["fused_silu_vs_split_speedup"] == pytest.approx(1.0 / 0.9)
        for shape in summary["shapes"].values()
    )


def test_selected_gate_up_comparison_fails_closed() -> None:
    tokens = _tokens()
    tokens["fused_silu"][55][1] += 1
    mismatch = _comparison_summary(_samples(), tokens)
    assert mismatch["promotion"]["pass"] is False
    assert "output_ids_not_exact" in mismatch["promotion"]["failed_checks"]

    samples = _samples()
    samples["fused_silu"][64] = [64.1, 64.2, 64.0]
    regression = _comparison_summary(samples, _tokens())
    assert regression["promotion"]["pass"] is False
    assert "candidate_not_faster_at_every_shape" in regression["promotion"]["failed_checks"]
