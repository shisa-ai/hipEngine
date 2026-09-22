"""The sampled MTP accept path must use each row's own prefix and the target law.

``sampled_accept_summary`` is the serving-side glue between the verifier's logits
matrix and the accept walk. These tests pin the two things that can silently
break it: row states that do not carry the drafted prefix (history-dependent
penalties would then be wrong) and an emitted token that is not distributed as
the target's own sampler would distribute it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.mtp_sampled_accept import (
    observe_published_tokens,
    published_forced_count,
    row_forced_token_ids,
    row_prefix_states,
    sampled_accept_summary,
)
from hipengine.generation.sampling import RowSamplingState
from hipengine.speculative.interfaces import TargetVerifyBatch


def _params(**overrides):
    values = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": (),
        "suppress_token_ids": (),
        "min_tokens": 0,
        "eos_token_id": None,
        "ignore_eos": False,
        "seed": 7,
        "row_seeds": (),
        "stop_token_ids": (),
        "stop_token_sequences": (),
        "forced_tokens_pending": (),
        "forced_token_reason": None,
        "post_thinking_forced_tokens_pending": (),
        "post_thinking_forced_token_reason": None,
        "force_sequence_completion_token_sequences": (),
        "force_sequence_completion_reason": None,
        "json_object_close_forcing": False,
        "tool_call_constraint": None,
        "thinking_close_token_ids": (),
        "thinking_hard_token_cap": None,
        "thinking_soft_close_window": 0,
        "logprobs": False,
        "top_logprobs": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _chain_batch(tokens: tuple[int, ...], *, request_id: int = 5) -> TargetVerifyBatch:
    rows = len(tokens) + 1
    return TargetVerifyBatch(
        request_ids=(request_id,),
        tokens=(0, *tokens),
        positions=tuple(range(rows)),
        row_to_request=(request_id,) * rows,
        parent_rows=(-1, *range(rows - 1)),
        root_rows=(0,),
        candidate_rows=tuple(range(1, rows)),
        draft_depths=(0, *range(1, rows)),
        active_mask=(True,) * rows,
    )


def _logits_row(values: dict[int, float], vocab: int = 8) -> np.ndarray:
    row = np.full((vocab,), -8.0, dtype=np.float32)
    for token_id, value in values.items():
        row[token_id] = float(value)
    return row


def _point_row(token_id: int, vocab: int = 8) -> np.ndarray:
    """A row whose only finite logit is ``token_id`` (a one-token support)."""

    row = np.full((vocab,), -np.inf, dtype=np.float32)
    row[token_id] = 0.0
    return row


def test_row_prefix_states_carry_the_drafted_prefix() -> None:
    batch = _chain_batch((3, 4))
    state = RowSamplingState(seed=1, prompt_tokens=(10, 11), generated_tokens=(12,))
    states = row_prefix_states(batch, {5: state})
    assert [tuple(item.generated_tokens) for item in states] == [
        (12,),
        (12, 3),
        (12, 3, 4),
    ]
    # The live state must not be advanced by resolving row prefixes.
    assert tuple(state.generated_tokens) == (12,)


def test_row_prefix_states_validate_missing_requests() -> None:
    batch = _chain_batch((3,))
    with pytest.raises(ValueError, match="missing sampler state"):
        row_prefix_states(batch, {})


def test_sampled_accept_summary_takes_the_draft_when_the_target_agrees() -> None:
    batch = _chain_batch((2, 2))
    logits = np.stack((_point_row(2), _point_row(2), _point_row(2)))
    draws = iter([0.0, 0.0, 0.0])
    summary = sampled_accept_summary(
        batch,
        logits,
        {5: RowSamplingState(seed=1, prompt_tokens=(1, 2))},
        params_for=lambda request_id: _params(),
        draws=lambda: next(draws),
        remaining_decode=(4,),
    )
    assert summary.accepted_counts == (2,)
    assert summary.accepted_tokens == ((2, 2),)
    assert summary.next_tokens == (2,)
    assert summary.commit_rows == (2,)
    assert summary.full_accept == (True,)


def test_sampled_accept_summary_resamples_on_rejection() -> None:
    batch = _chain_batch((2,))
    # The target's law at the root is a point mass on token 5, so the drafted
    # token 2 rejects with probability one and the residual emits 5.
    logits = np.stack((_point_row(5), _point_row(5)))
    draws = iter([0.0, 0.0])
    summary = sampled_accept_summary(
        batch,
        logits,
        {5: RowSamplingState(seed=1, prompt_tokens=(1, 2))},
        params_for=lambda request_id: _params(),
        draws=lambda: next(draws),
        remaining_decode=(4,),
    )
    assert summary.accepted_counts == (0,)
    assert summary.accepted_tokens == ((),)
    assert summary.next_tokens == (5,)
    assert summary.commit_rows == (0,)


def test_sampled_accept_summary_rejects_a_foreign_draft_token() -> None:
    """A draft the target filtered out rejects with probability one."""

    batch = _chain_batch((2,))
    logits = np.stack((_point_row(0), _point_row(5)))
    draws = iter([0.999999, 0.0])
    summary = sampled_accept_summary(
        batch,
        logits,
        {5: RowSamplingState(seed=1, prompt_tokens=(1, 2))},
        params_for=lambda request_id: _params(),
        draws=lambda: next(draws),
        remaining_decode=(4,),
    )
    assert summary.accepted_counts == (0,)
    assert summary.next_tokens == (0,)


def test_sampled_accept_summary_uses_each_requests_own_params() -> None:
    batch = TargetVerifyBatch(
        request_ids=(5, 6),
        tokens=(0, 0, 2, 2),
        positions=(0, 0, 1, 1),
        row_to_request=(5, 6, 5, 6),
        parent_rows=(-1, -1, 0, 1),
        root_rows=(0, 1),
        candidate_rows=(2, 3),
        draft_depths=(0, 0, 1, 1),
        active_mask=(True, True, True, True),
    )
    logits = np.stack(
        (
            _point_row(2),
            _point_row(2),
            _point_row(2),
            _point_row(2),
        )
    )
    seen: list[int] = []

    def params_for(request_id: int):
        seen.append(int(request_id))
        return _params()

    draws = iter([0.0] * 4)
    summary = sampled_accept_summary(
        batch,
        logits,
        {
            5: RowSamplingState(seed=1, prompt_tokens=(1,)),
            6: RowSamplingState(seed=2, prompt_tokens=(2,)),
        },
        params_for=params_for,
        draws=lambda: next(draws),
        remaining_decode=(4, 4),
    )
    assert set(seen) == {5, 6}
    assert summary.accepted_counts == (1, 1)


def test_sampled_accept_summary_observes_penalties_per_row() -> None:
    """Each row's law must reflect the drafted tokens already in its history."""

    from hipengine.generation.sampling import processed_distribution

    batch = _chain_batch((3, 3))
    params = _params(frequency_penalty=0.5)
    logits = np.stack((_logits_row({3: 10.0, 4: 6.0}),) * 3)
    state = RowSamplingState(seed=3, prompt_tokens=(1,), generated_tokens=(3, 3, 3))
    states = row_prefix_states(batch, {5: state})
    assert [tuple(item.generated_tokens).count(3) for item in states] == [3, 4, 5]
    probabilities = []
    for row, row_state in enumerate(states):
        token_ids, weights = processed_distribution(logits[row], params, row_state)
        table = {
            int(token): float(weight)
            for token, weight in zip(token_ids, weights, strict=True)
        }
        probabilities.append(table[3])
    # The frequency penalty is applied against each row's own history, so p(3)
    # shrinks as the drafted prefix repeats the token.
    assert probabilities[0] > probabilities[1] > probabilities[2]
    # The live state is never advanced by resolving prefixes.
    assert tuple(state.generated_tokens) == (3, 3, 3)


def test_sampled_accept_summary_validates_logits_shape() -> None:
    batch = _chain_batch((2,))
    with pytest.raises(ValueError, match="two-dimensional"):
        sampled_accept_summary(
            batch,
            np.zeros((2,), dtype=np.float32),
            {5: RowSamplingState(seed=1)},
            params_for=lambda request_id: _params(),
            draws=lambda: 0.0,
        )
    with pytest.raises(ValueError, match="align"):
        sampled_accept_summary(
            batch,
            np.zeros((3, 8), dtype=np.float32),
            {5: RowSamplingState(seed=1)},
            params_for=lambda request_id: _params(),
            draws=lambda: 0.0,
        )


def test_sampled_accept_summary_requires_a_linear_chain() -> None:
    batch = TargetVerifyBatch(
        request_ids=(5,),
        tokens=(0, 2, 3),
        positions=(0, 1, 1),
        row_to_request=(5, 5, 5),
        parent_rows=(-1, 0, 0),
        root_rows=(0,),
        candidate_rows=(1, 2),
        draft_depths=(0, 1, 1),
        active_mask=(True, True, True),
    )
    logits = np.stack(
        (_point_row(2), _point_row(2), _point_row(3))
    )
    with pytest.raises(ValueError, match="single drafted chain"):
        sampled_accept_summary(
            batch,
            logits,
            {5: RowSamplingState(seed=1)},
            params_for=lambda request_id: _params(),
            draws=lambda: 0.0,
            remaining_decode=(4,),
        )


def test_sampled_accept_summary_emits_the_target_law_under_monte_carlo() -> None:
    """The emitted token's histogram must track the target's own distribution."""

    from hipengine.generation.sampling import processed_distribution

    params = _params(temperature=0.8)
    batch = _chain_batch((2,))
    root = _logits_row({0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0})
    candidate = _logits_row({0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0})
    logits = np.stack((root, candidate))
    state = RowSamplingState(seed=5, prompt_tokens=(1, 2, 3))
    token_ids, probabilities = processed_distribution(root, params, state)
    target = {int(token): float(weight) for token, weight in zip(token_ids, probabilities, strict=True)}

    rng = np.random.default_rng(17)
    counts: dict[int, int] = {}
    draws = 20000
    for _ in range(draws):
        values = iter(float(rng.random()) for _ in range(3))

        def draw(values=values) -> float:
            return next(values)

        summary = sampled_accept_summary(
            batch,
            logits,
            {5: state},
            params_for=lambda request_id: params,
            draws=draw,
            remaining_decode=(2,),
        )
        emitted = (
            summary.accepted_tokens[0][0]
            if summary.accepted_counts[0]
            else summary.next_tokens[0]
        )
        counts[int(emitted)] = counts.get(int(emitted), 0) + 1
    support = sorted(set(counts) | set(target))
    observed = np.asarray([counts.get(token, 0) / draws for token in support])
    expected = np.asarray([target.get(token, 0.0) for token in support])
    total_variation = float(0.5 * np.sum(np.abs(observed - expected)))
    sigma = float(np.sqrt(2.0 * float(np.sum(expected * (1.0 - expected))) / draws))
    assert total_variation <= 4.0 * sigma
    # The walk must not have advanced the live state; the scheduler observes the
    # committed tokens itself.
    assert tuple(state.generated_tokens) == ()


def test_row_forced_token_ids_drain_in_row_order() -> None:
    """Each row's override is the queue head after its ancestors consumed theirs."""

    batch = _chain_batch((3, 4))
    state = RowSamplingState(
        seed=1,
        prompt_tokens=(10,),
        forced_tokens_pending=(7, 6),
    )
    assert row_forced_token_ids(batch, {5: state}) == (7, 6, None)
    # Resolving the walk must not consume the live request's queue: only the
    # tokens a cycle actually publishes may be popped.
    assert state.forced_tokens == (7, 6)


def test_sampled_accept_summary_accepts_a_forced_chain() -> None:
    """The override replaces the row's law, so a matching draft is accepted."""

    batch = _chain_batch((5, 6))
    # The target's own law wants 2 then 3; the request forces 5 then 6, so the
    # distribution the accept couples against is not the row's argmax.
    logits = np.stack((_point_row(2), _point_row(3), _point_row(3)))
    state = RowSamplingState(
        seed=1,
        prompt_tokens=(1,),
        forced_tokens_pending=(5, 6),
    )
    summary = sampled_accept_summary(
        batch,
        logits,
        {5: state},
        params_for=lambda request_id: _params(),
        draws=lambda: 0.0,
        remaining_decode=(4,),
    )
    assert summary.accepted_counts == (2,)
    assert summary.accepted_tokens == ((5, 6),)
    # The queue held two tokens, so the third row is an ordinary row: its bonus
    # token is the target's own argmax (3), not a forced token.
    assert summary.next_tokens == (3,)
    assert state.forced_tokens == (5, 6)


def test_sampled_accept_summary_corrects_to_a_forced_token() -> None:
    """A draft the forced token disagrees with is corrected to the forced token."""

    batch = _chain_batch((2,))
    logits = np.stack((_point_row(2), _point_row(2)))
    state = RowSamplingState(
        seed=1,
        prompt_tokens=(1,),
        forced_tokens_pending=(5,),
    )
    summary = sampled_accept_summary(
        batch,
        logits,
        {5: state},
        params_for=lambda request_id: _params(),
        draws=lambda: 0.0,
        remaining_decode=(4,),
    )
    assert summary.accepted_counts == (0,)
    assert summary.accepted_tokens == ((),)
    assert summary.next_tokens == (5,)
    assert summary.commit_rows == (0,)


def test_observe_published_tokens_consumes_the_published_forced_prefix() -> None:
    state = RowSamplingState(
        seed=1,
        prompt_tokens=(1,),
        forced_tokens_pending=(5, 6),
    )
    observe_published_tokens(state, (5, 6), forced_count=2)
    assert tuple(state.generated_tokens) == (5, 6)
    assert state.forced_tokens == ()


def test_observe_published_tokens_leaves_the_unpublished_queue_pending() -> None:
    """A cycle that published one forced token must not consume the next."""

    state = RowSamplingState(
        seed=1,
        prompt_tokens=(1,),
        forced_tokens_pending=(5, 6),
    )
    observe_published_tokens(state, (5,), forced_count=1)
    assert state.forced_tokens == (6,)


def test_observe_published_tokens_refuses_a_forced_mismatch() -> None:
    """The published token and the consumed override must be the same token."""

    state = RowSamplingState(seed=1, prompt_tokens=(1,), forced_tokens_pending=(5,))
    with pytest.raises(RuntimeError, match="pending forced token is 5"):
        observe_published_tokens(state, (9,), forced_count=1)


def test_published_forced_count_counts_only_the_published_prefix() -> None:
    """A stop inside the chain must not consume the overrides behind it."""

    batch = _chain_batch((5, 6))
    forced_ids = (5, 6, None)
    # The full chain published both forced tokens and the bonus row.
    assert published_forced_count(batch, (2,), (5, 6, 3), forced_ids) == 2
    # The finish rule stopped after the first token: only that one was published.
    assert published_forced_count(batch, (2,), (5,), forced_ids) == 1


def test_published_forced_count_ignores_rows_the_accept_walk_never_reached() -> None:
    """A rejected chain publishes the correction row only."""

    batch = _chain_batch((5, 6))
    forced_ids = (5, 6, None)
    # The accept walk stopped at the root: the published token is the root row's.
    assert published_forced_count(batch, (0,), (5,), forced_ids) == 1


def test_cycle_commit_round_trip_consumes_exactly_what_it_published() -> None:
    """The count the commit computes is the count the live state pops."""

    batch = _chain_batch((5, 6))
    state = RowSamplingState(
        seed=1,
        prompt_tokens=(1,),
        forced_tokens_pending=(5, 6),
    )
    forced_ids = row_forced_token_ids(batch, {5: state})
    published = (5, 6, 3)
    observe_published_tokens(
        state,
        published,
        forced_count=published_forced_count(batch, (2,), published, forced_ids),
    )
    assert tuple(state.generated_tokens) == published
    assert state.forced_tokens == ()
