"""CPU contracts for the checked/full-row dense Q8 model-state gate."""

import pytest

from scripts.qwen4exp_dense_dpp_checked_full_rows_state_gate import (
    CANDIDATE,
    DECLARED_SHAPES,
    PARENT,
    admitted_shape,
)


def test_state_gate_keeps_parent_and_candidate_separate():
    assert PARENT != CANDIDATE
    assert "checked_full_rows" not in PARENT
    assert "checked_full_rows" in CANDIDATE


def test_state_gate_declares_only_the_six_natural_dense_shapes():
    assert len(DECLARED_SHAPES) == 6
    assert (2560, 10240) in DECLARED_SHAPES
    assert (10240, 320) in DECLARED_SHAPES
    assert (2560, 2560) not in DECLARED_SHAPES


@pytest.mark.parametrize("rows,k,n", [
    (512, 2560, 10240), (1024, 2560, 6144), (4096, 6144, 2560),
    (512, 2560, 640), (4096, 640, 2560), (4096, 10240, 320),
])
def test_state_gate_admits_declared_shapes(rows, k, n):
    assert admitted_shape(rows, k, n)


@pytest.mark.parametrize("rows,k,n", [
    (256, 2560, 10240), (512, 2560, 2560), (512, 2560, 10248),
    (510, 2560, 6144), (8192, 10240, 320),
])
def test_state_gate_falls_back_outside_scope(rows, k, n):
    assert not admitted_shape(rows, k, n)
