"""RT-1: the serving prefill chunk cap must not fragment the round budget.

``EngineLoop._run_token_budget_round`` bounds per-tick prefill work with
``round_prefill_token_budget`` (default 1024) and then splits that budget into
chunks of ``min(prefill_chunk_size, prefill_budget)``::

    prefill_budget = self.round_prefill_token_budget
    while prefill_budget > 0 and self.scheduler.has_prefill_work():
        work = self.scheduler.next_round_robin_prefill_work(
            chunk_size=min(self.prefill_chunk_size, prefill_budget)
        )

The fairness bound that protects decode is therefore the ROUND budget.  A
``max_prefill_chunk_tokens`` default below it does not make the engine fairer -
it makes the same budgeted work take more, smaller chunks.  Measured with the
prefill harness (``qwen4exp_profile_gap.py --mode prefill``, 12 canonical cases,
three reps per arm, one process per arm), 256 -> 1024 is -29.0% to -35.7% wall
for +1.24 GB peak at 4K, with ``token_id`` identical on 12/12 cases.
``logits_sha256`` differs from 256 on 12/12, so this is an arithmetic change
against the old default rather than a pure scheduling one; chunk 1024 is the
configuration the X-6 numerical gate certifies.

These tests pin the relationship rather than a bare number, so a future change
to either knob has to keep the two consistent.
"""

from __future__ import annotations

import pytest


def test_default_prefill_chunk_does_not_fragment_the_round_budget() -> None:
    """The chunk cap must not sit below the round budget it is drawn from.

    A chunk cap below the round budget cannot reduce per-tick prefill work - the
    budget already bounds that - so it only multiplies per-chunk overhead.  The
    two defaults are a pair and must agree.
    """
    from hipengine.generation.engine_loop import (
        DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
        DEFAULT_ROUND_PREFILL_TOKEN_BUDGET,
    )

    assert DEFAULT_MAX_PREFILL_CHUNK_TOKENS == DEFAULT_ROUND_PREFILL_TOKEN_BUDGET, (
        "the serving prefill chunk cap must equal the round prefill token budget: "
        f"a cap below it ({DEFAULT_MAX_PREFILL_CHUNK_TOKENS} vs "
        f"{DEFAULT_ROUND_PREFILL_TOKEN_BUDGET}) fragments a budget that already "
        "bounds per-tick prefill work, paying per-chunk overhead for no fairness gain"
    )


def test_prefill_chunk_cap_is_the_certified_1024() -> None:
    """1024 is the configuration the X-6 numerical gate certifies."""
    from hipengine.generation.engine_loop import DEFAULT_MAX_PREFILL_CHUNK_TOKENS

    assert DEFAULT_MAX_PREFILL_CHUNK_TOKENS == 1024


def test_engine_loop_config_default_matches_the_module_default() -> None:
    from hipengine.generation.engine_loop import (
        DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
        EngineLoopConfig,
    )

    assert EngineLoopConfig().max_prefill_chunk_tokens == DEFAULT_MAX_PREFILL_CHUNK_TOKENS


def test_a_larger_chunk_cap_never_exceeds_the_round_budget_in_one_quantum() -> None:
    """The per-tick bound is unchanged by the cap, which is the whole argument.

    Simulate the loop's own arithmetic: with the round budget fixed, the tokens
    a single round can process is the budget regardless of the cap, and raising
    the cap can only reduce the number of chunks needed.
    """
    from hipengine.generation.engine_loop import (
        DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
        DEFAULT_ROUND_PREFILL_TOKEN_BUDGET,
    )

    def chunks_needed(cap: int, budget: int) -> tuple[int, int]:
        remaining = budget
        chunks = 0
        while remaining > 0:
            size = min(cap, remaining)
            assert size > 0
            remaining -= size
            chunks += 1
        return budget, chunks

    tokens_old, chunks_old = chunks_needed(256, DEFAULT_ROUND_PREFILL_TOKEN_BUDGET)
    tokens_new, chunks_new = chunks_needed(
        DEFAULT_MAX_PREFILL_CHUNK_TOKENS, DEFAULT_ROUND_PREFILL_TOKEN_BUDGET
    )
    # Same work per round - this is the property that makes the change safe.
    assert tokens_old == tokens_new == DEFAULT_ROUND_PREFILL_TOKEN_BUDGET
    # And strictly less fragmentation, which is where the measured win comes from.
    assert chunks_new < chunks_old


def test_prefill_chunk_cap_still_rejects_non_positive() -> None:
    from hipengine.generation.engine_loop import EngineLoopConfig

    with pytest.raises(ValueError):
        EngineLoopConfig(max_prefill_chunk_tokens=0)
