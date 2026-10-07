"""CPU contracts for the natural checked/full-row dense-Q8 owner screen."""

import pytest

from scripts.qwen4exp_dense_dpp_checked_full_rows_natural_owner import (
    CANDIDATE,
    PARENT,
    admitted_shape,
)


def test_natural_screen_uses_separate_parent_and_candidate_variants():
    assert PARENT != CANDIDATE
    assert "checked_full_rows" not in PARENT
    assert "checked_full_rows" in CANDIDATE


@pytest.mark.parametrize(
    "rows,k,n",
    [
        (512, 2560, 10240),
        (1024, 2560, 6144),
        (4096, 6144, 2560),
        (512, 2560, 640),
        (4096, 640, 2560),
        (4096, 10240, 320),
    ],
)
def test_declared_natural_shapes_are_admitted(rows, k, n):
    assert admitted_shape(rows, k, n)


@pytest.mark.parametrize(
    "rows,k,n",
    [
        (256, 2560, 10240),
        (512, 2560, 10240 + 8),
        (512, 10240, 320 + 8),
        (510, 2560, 6144),
        (4096, 2560, 2560),
        (8192, 2560, 10240),
    ],
)
def test_natural_screen_falls_back_outside_declared_scope(rows, k, n):
    assert not admitted_shape(rows, k, n)
