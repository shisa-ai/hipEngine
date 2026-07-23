from __future__ import annotations

import pytest

from scripts.laguna_dense_shared_prefill_bench import (
    PROFILE_ROWS,
    _comparison_summary,
    _mode_order,
)


def _samples(*, split_scale: float = 1.0, dual_scale: float = 0.7):
    return {
        "split": {
            rows: [rows * split_scale, rows * split_scale * 1.02]
            for rows in PROFILE_ROWS
        },
        "dual": {
            rows: [rows * dual_scale, rows * dual_scale * 1.02]
            for rows in PROFILE_ROWS
        },
    }


def _tokens():
    return {
        mode: {rows: [1000 + rows, 1000 + rows] for rows in PROFILE_ROWS}
        for mode in ("split", "dual")
    }


def test_lpf3_mode_order_balances_each_shape_across_repetitions() -> None:
    for index, _rows in enumerate(PROFILE_ROWS):
        assert _mode_order(index, 0) == (
            ("split", "dual")
            if index % 2 == 0
            else ("dual", "split")
        )
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))


def test_lpf3_comparison_requires_exact_outputs_and_all_shape_wins() -> None:
    summary = _comparison_summary(_samples(), _tokens(), rows=PROFILE_ROWS)

    assert summary["correctness"]["pass"] is True
    assert summary["promotion"]["pass"] is True
    assert summary["promotion"]["slower_or_equal_rows"] == []
    assert summary["promotion"]["effective_speedup"] == pytest.approx(1.0 / 0.7)
    assert summary["shapes"]["55"]["dual_vs_split_speedup"] == pytest.approx(
        1.0 / 0.7
    )


def test_lpf3_comparison_fails_closed_on_mismatch_or_one_slow_shape() -> None:
    tokens = _tokens()
    tokens["dual"][55][1] += 1
    mismatch = _comparison_summary(_samples(), tokens, rows=PROFILE_ROWS)
    assert mismatch["correctness"]["pass"] is False
    assert mismatch["promotion"]["pass"] is False
    assert "output_ids_not_exact" in mismatch["promotion"]["failed_checks"]

    samples = _samples()
    samples["dual"][32] = [32 * 1.1, 32 * 1.1 * 1.02]
    slow_shape = _comparison_summary(samples, _tokens(), rows=PROFILE_ROWS)
    assert slow_shape["promotion"]["pass"] is False
    assert slow_shape["promotion"]["slower_or_equal_rows"] == [32]
    assert "candidate_not_faster_at_every_profile_shape" in slow_shape["promotion"][
        "failed_checks"
    ]
