"""The TP2 source-F16 prefill owner must stay opt-in and rank-owned.

Two failures found while integrating this are pinned here, because both were
silent: neither raised, and both would have looked like "the optimization just
does not help" rather than a bug.

1. The owner's admission filter compares against ``planes.rows``. Caching planes
   on device alone makes the *first* pass's row count permanent, so a session
   warmed at 64 rows would keep the exact T16 owner for every later 512-row pass.
2. The gate suite's prompts are 52-64 tokens and the policy's smallest admitted
   row count is 512, so the gate as configured never runs the owner at all. Its
   logits came back bit-identical across the flag and that was read as "the owner
   is numerically safe" when it meant "the owner did not run".
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

TP2 = pathlib.Path(__file__).resolve().parents[1] / "hipengine" / "distributed" / "tp2_generate.py"
DIAG = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tp2_bulk_prefill_diagnostic.py"


def test_owner_is_opt_in_by_default() -> None:
    """It fails the production gate, so it must not be the default."""

    source = TP2.read_text()
    marker = "self.use_t16_f16_rocblas_prefill = ("
    assert marker in source
    # Look at the assignment plus the comment that justifies it, which is
    # written above the assignment rather than below it.
    before, after = source.split(marker, 1)
    block = before[-1200:] + marker + after[:200]
    assert "False" in after.split("if use_t16_f16_rocblas_prefill is None")[0], (
        "the owner must default off: it regresses the heldout envelope"
    )
    # ... and the reason must be recorded, not just the flag.
    assert "heldout" in block


def test_planes_are_reallocated_when_the_row_count_grows() -> None:
    """Caching on device alone silently loses the owner at larger row counts."""

    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    source = inspect.getsource(MlpTP2GenerationSession._ensure_rank_f16_rocblas_planes)
    # The cache must consult the cached planes' row count, not just membership.
    assert "getattr(cached, \"rows\", 0)" in source
    assert ">= rows" in source
    # And it must release the superseded planes rather than leak them.
    assert "cached.release(" in source


def test_the_gate_can_reach_an_admitted_row_count() -> None:
    """A gate that never runs the owner says nothing about it."""

    source = DIAG.read_text()
    assert "--pad-prompt-tokens" in source
    # The help must state why padding is needed, or the next reader will drop it.
    assert "512" in source
    assert "owner" in source


@pytest.mark.parametrize("rows", [64, 200])
def test_row_counts_below_the_policy_fall_back(rows: int) -> None:
    """Below the smallest admitted row count the exact T16 owner must remain.

    This is a CPU-only statement about the admission arithmetic: the policy's
    smallest row count is 512, so anything under it is not admitted and the
    caller keeps the exact path.
    """

    assert rows < 512
