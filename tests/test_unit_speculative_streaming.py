from __future__ import annotations

import pytest

from hipengine.speculative.streaming import (
    SpeculativeCommitEvent,
    stochastic_acceptance_accounting,
    trim_speculative_output,
)


def test_speculative_output_trim_handles_eos_stop_sequence_and_capacity() -> None:
    eos = trim_speculative_output(
        (10, 11, 2, 99), max_tokens=8, min_tokens=0, eos_token_id=2,
        stop_token_ids=(), stop_token_sequences=(), ignore_eos=False,
    )
    assert eos.token_ids == (10, 11, 2)
    assert eos.finish_reason == "eos"

    stop = trim_speculative_output(
        (20, 30, 31, 40), max_tokens=8, min_tokens=0, eos_token_id=None,
        stop_token_ids=(), stop_token_sequences=((30, 31),), ignore_eos=True,
    )
    assert stop.token_ids == (20, 30, 31)
    assert stop.finish_reason == "stop"

    tail = trim_speculative_output(
        (1, 2, 3, 4), max_tokens=2, min_tokens=0, eos_token_id=None,
        stop_token_ids=(), stop_token_sequences=(), ignore_eos=True,
    )
    assert tail.token_ids == (1, 2)
    assert tail.finish_reason == "length"


def test_stochastic_acceptance_accounts_one_rng_draw_per_candidate() -> None:
    accounting = stochastic_acceptance_accounting(
        request_id=7,
        candidate_tokens=(10, 11, 12),
        draft_probabilities=(0.5, 0.8, 0.2),
        target_probabilities=(0.5, 0.4, 0.2),
        uniforms=(0.9, 0.4, 0.1),
        rng_counter_before=100,
        correction_or_bonus_token=99,
    )
    assert accounting.accepted_tokens == (10, 11, 12)
    assert accounting.accepted_count == 3
    assert accounting.rng_counter_before == 100
    assert accounting.rng_counter_after == 103
    assert accounting.correction_or_bonus_token == 99

    rejected = stochastic_acceptance_accounting(
        request_id=8,
        candidate_tokens=(20, 21),
        draft_probabilities=(0.8, 0.8),
        target_probabilities=(0.4, 0.8),
        uniforms=(0.75, 0.1),
        rng_counter_before=5,
    )
    assert rejected.accepted_tokens == ()
    assert rejected.accepted_count == 0
    assert rejected.rng_counter_after == 6


def test_commit_event_never_publishes_provisional_tokens() -> None:
    with pytest.raises(ValueError, match="committed"):
        SpeculativeCommitEvent(
            request_id=1,
            transaction_id=2,
            token_ids=(10,),
            accepted_count=1,
            correction_or_bonus_token=None,
            rng_counter_before=0,
            rng_counter_after=0,
            finish_reason=None,
            committed=False,
        )
