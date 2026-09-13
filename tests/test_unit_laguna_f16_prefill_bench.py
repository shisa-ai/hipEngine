from __future__ import annotations

import pytest

from scripts.laguna_f16_prefill_bench import (
    PROFILE_ROWS,
    _comparison_summary,
    _mode_order,
)


def _samples(*, gemv_scale: float = 1.0, tiled_scale: float = 0.5):
    return {
        "gemv": {
            rows: [rows * gemv_scale, rows * gemv_scale * 1.02]
            for rows in PROFILE_ROWS
        },
        "tiled": {
            rows: [rows * tiled_scale, rows * tiled_scale * 1.02]
            for rows in PROFILE_ROWS
        },
    }


def _tokens():
    return {
        mode: {rows: [1000 + rows, 1000 + rows] for rows in PROFILE_ROWS}
        for mode in ("gemv", "tiled")
    }


def test_lpf1_mode_order_balances_each_shape_across_repetitions() -> None:
    for index, rows in enumerate(PROFILE_ROWS):
        assert _mode_order(index, 0) == (
            ("gemv", "tiled") if index % 2 == 0 else ("tiled", "gemv")
        )
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))


def test_lpf1_comparison_requires_exact_outputs_and_finds_threshold() -> None:
    samples = _samples()
    # Make tiled slower below row 8. Those rows must remain on GEMV.
    for rows in (2, 3, 4, 5, 7):
        samples["tiled"][rows] = [rows * 1.1, rows * 1.1 * 1.02]

    summary = _comparison_summary(samples, _tokens(), rows=PROFILE_ROWS)

    assert summary["correctness"]["pass"] is True
    assert summary["promotion"]["pass"] is True
    assert summary["promotion"]["measured_min_rows"] == 8
    assert summary["promotion"]["effective_speedup"] > 1.0
    assert summary["shapes"]["7"]["tiled_vs_gemv_speedup"] < 1.0
    assert summary["shapes"]["8"]["tiled_vs_gemv_speedup"] == pytest.approx(2.0)


def test_lpf1_comparison_fails_closed_on_mismatch_or_no_winning_tail() -> None:
    tokens = _tokens()
    tokens["tiled"][16][1] += 1
    mismatch = _comparison_summary(_samples(), tokens, rows=PROFILE_ROWS)
    assert mismatch["correctness"]["pass"] is False
    assert mismatch["promotion"]["pass"] is False
    assert "output_ids_not_exact" in mismatch["promotion"]["failed_checks"]

    no_win = _comparison_summary(
        _samples(gemv_scale=1.0, tiled_scale=1.1),
        _tokens(),
        rows=PROFILE_ROWS,
    )
    assert no_win["promotion"]["pass"] is False
    assert no_win["promotion"]["measured_min_rows"] is None
    assert "no_strictly_faster_measured_tail" in no_win["promotion"]["failed_checks"]
