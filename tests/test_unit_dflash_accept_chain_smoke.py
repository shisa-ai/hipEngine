"""CPU checks for the DFlash accept smoke's budget expectations."""

import numpy as np

from hipengine.speculative import TargetAcceptSummary
from scripts import dflash_accept_chain_smoke as smoke


def test_budgeted_accept_smoke_preserves_the_reserved_bonus(monkeypatch) -> None:
    observed = []

    def accept(runtime, lm_head_lib, accept_lib, target, target_top1, *, remaining_decode, debug_top1):
        summary = TargetAcceptSummary.from_accept_result(
            target, target.accept_from_top1(target_top1, remaining_decode=remaining_decode),
        )
        observed.append((remaining_decode, summary))
        return {
            "accepted_counts": np.asarray(summary.accepted_counts, dtype=np.int32),
            "next_tokens": np.asarray(summary.next_tokens, dtype=np.int32),
        }

    monkeypatch.setattr(smoke, "_run_accept_from_top1", accept)
    smoke._run_budgeted_accept_case(None, None, None, debug_top1=False)
    budgets, summary = observed[0]
    assert budgets == (2, 1)
    assert summary.accepted_counts == (1, 0)
    assert summary.next_tokens == (42, 51)
    assert tuple(count + 1 for count in summary.accepted_counts) == budgets
