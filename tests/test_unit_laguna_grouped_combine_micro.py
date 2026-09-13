from __future__ import annotations

import argparse

import pytest

from scripts.laguna_grouped_combine_micro import _parse_rows, _summarize


def test_grouped_combine_micro_rows_are_sorted_and_distinct() -> None:
    assert _parse_rows("128,32,55,32") == (32, 55, 128)
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _parse_rows("32,0")


def test_grouped_combine_micro_summary_requires_exact_all_shape_gain() -> None:
    rows = (32, 55)
    result = _summarize(
        rows,
        {
            32: {"baseline": [4.0, 4.1], "candidate": [3.0, 3.1]},
            55: {"baseline": [6.0, 6.1], "candidate": [5.0, 5.1]},
        },
        {
            32: {"baseline": [3.0, 3.1], "candidate": [2.0, 2.1]},
            55: {"baseline": [5.0, 5.1], "candidate": [4.0, 4.1]},
        },
        {32: True, 55: True},
    )

    assert result["pass"] is True
    assert result["aggregate_gpu_span_speedup"] > 1.0
    assert result["shapes"]["32"]["bit_exact"] is True


def test_grouped_combine_micro_summary_rejects_inexact_or_slow_shape() -> None:
    rows = (32, 55)
    result = _summarize(
        rows,
        {
            32: {"baseline": [4.0], "candidate": [4.1]},
            55: {"baseline": [6.0], "candidate": [5.0]},
        },
        {
            32: {"baseline": [3.0], "candidate": [3.1]},
            55: {"baseline": [5.0], "candidate": [4.0]},
        },
        {32: False, 55: True},
    )

    assert result["pass"] is False
    assert "rows_32_not_bit_exact" in result["failed_checks"]
    assert "rows_32_gpu_span_not_faster" in result["failed_checks"]
