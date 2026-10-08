"""CPU contracts for the natural K320/full-row GR-up owner screen."""

import pytest

from scripts.qwen4exp_gr_k320_full_rows_natural_owner import (
    CANDIDATE,
    PARENT,
    admitted_shape,
)


def test_natural_screen_uses_separate_parent_and_candidate_variants():
    assert PARENT != CANDIDATE
    assert "k320_full_rows" not in PARENT
    assert "k320_full_rows" in CANDIDATE
    assert PARENT.endswith("_dpp_f32_exact")
    assert CANDIDATE.endswith("_dpp_k320_full_rows_f32_exact")


@pytest.mark.parametrize(
    "rows,low_rank,branches,hidden,expected",
    [
        (512, 320, 4, 2560, True),
        (1024, 320, 4, 2560, True),
        (4096, 320, 4, 2560, True),
        (256, 320, 4, 2560, False),
        (1, 320, 4, 2560, False),
        (507, 320, 4, 2560, False),
        (4102, 320, 4, 2560, False),
        (512, 316, 4, 2560, False),
        (512, 320, 8, 2560, False),
        (512, 320, 4, 2048, False),
    ],
)
def test_admission_matches_the_runner_selector_window(
    rows, low_rank, branches, hidden, expected
):
    assert admitted_shape(rows, low_rank, branches, hidden) is expected
