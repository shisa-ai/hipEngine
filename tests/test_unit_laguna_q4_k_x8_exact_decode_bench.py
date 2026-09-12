from __future__ import annotations

import argparse

import numpy as np
import pytest

from scripts.laguna_q4_k_x8_exact_decode_bench import (
    _parse_csv_ints,
    _selected_experts,
)


def test_selected_experts_are_deterministic_distinct_top10_rows() -> None:
    first = _selected_experts(4, layer=1).reshape(4, 10)
    second = _selected_experts(4, layer=1).reshape(4, 10)
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0) & (first < 256))
    assert all(np.unique(row).size == 10 for row in first)


def test_selected_experts_validate_shape() -> None:
    with pytest.raises(ValueError, match="invalid selected-expert shape"):
        _selected_experts(0, layer=1)
    with pytest.raises(ValueError, match="invalid selected-expert shape"):
        _selected_experts(1, layer=1, top_k=11, experts=10)


def test_parse_csv_ints_requires_sorted_unique_positive_values() -> None:
    assert _parse_csv_ints("1,2,4,8") == (1, 2, 4, 8)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_csv_ints("2,1")
