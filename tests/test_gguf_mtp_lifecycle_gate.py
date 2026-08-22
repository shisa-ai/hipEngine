from __future__ import annotations

import pytest

from scripts.gguf_mtp_lifecycle_gate import (
    EXPECTED_SUCCESS_PHASES,
    FAULT_PHASES,
    InjectedLifecycleFault,
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
