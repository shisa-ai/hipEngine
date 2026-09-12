from __future__ import annotations

import pytest

from scripts.gguf_mtp_lifecycle_gate import (
    EXPECTED_SUCCESS_PHASES,
    FAULT_PHASES,
    InjectedLifecycleFault,
    _expected_prefix_through_token,
)


def test_lifecycle_fault_matrix_covers_every_owned_cycle_boundary() -> None:
    assert FAULT_PHASES == (
        "before_proposal",
        "after_proposal_before_target",
        "after_target_prepare",
        "after_target_commit",
        "after_draft_repair",
        "before_output_publication",
    )
    assert EXPECTED_SUCCESS_PHASES == (*FAULT_PHASES, "after_output_publication")


def test_injected_lifecycle_fault_is_an_explicit_runtime_error() -> None:
    with pytest.raises(InjectedLifecycleFault, match="after_target_prepare"):
        raise InjectedLifecycleFault("injected:after_target_prepare")


def test_terminal_prefix_is_derived_from_the_actual_model_trajectory() -> None:
    expected = (248046, 198, 248045, 248068, 999)

    assert _expected_prefix_through_token(expected, 198) == (248046, 198)
    assert _expected_prefix_through_token(expected, 248068) == (
        248046,
        198,
        248045,
        248068,
    )
    with pytest.raises(ValueError, match="absent"):
        _expected_prefix_through_token(expected, 123)
