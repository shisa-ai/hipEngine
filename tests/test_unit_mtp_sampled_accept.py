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


def _text_observer(params, texts, *, remaining_decode=8, drafted=False):
    """The adapter's own observer shape, so the wiring is what is under test."""

    from hipengine.generation.qwen35_gguf_mtp2 import _constraint_text_observer

    return _constraint_text_observer(
        params,
        token_text_for_id=lambda token_id: texts[int(token_id)],
        encode_text=lambda text: (7,),
        remaining_decode=remaining_decode,
        drafted=drafted,
    )


def test_a_row_text_is_observed_so_the_next_row_is_masked_by_it():
    """A cycle's later rows must see the DFA state the earlier tokens produced.

    ``json_object_close_forcing`` masks a row from the text of the tokens before
    it. Before the walk observed text, the second row of a cycle was masked
    against a DFA that had not seen the first row's token -- so the route could
    publish a token the request's own constraint rejects. The first half of this
    test is that defect; the second half is the fix.
    """

    from hipengine.generation.sampling import processed_distribution

    batch = _chain_batch((2,))
    params = _params(json_object_close_forcing=True)
    state = RowSamplingState(
        seed=7, generated_tokens=(1,), json_object_close_forcing=True
    )
    # The parent row's decoded text is an object opened but not yet closed, so
    # the child row's outgoing token is the one that closes it.
    texts = {2: '{"a": 1', 3: "}", 5: "{", 6: " "}
    logits = np.full((8,), -6.0, dtype=np.float32)
    kwargs = {"token_text_for_id": lambda token_id: texts[int(token_id)]}

    unobserved = row_prefix_states(batch, {5: state})[1]
    ids_without, _probs = processed_distribution(logits, params, unobserved, **kwargs)
    # The defect, in both directions: the child cannot close the object it is
    # inside, and it may open a second one.
    assert 3 not in ids_without, "the fixture must show the unobserved-row defect"
    assert 5 in ids_without

    observed = row_prefix_states(batch, {5: state}, observe_text=_text_observer(params, texts))[1]
    ids_with, _probs = processed_distribution(logits, params, observed, **kwargs)
    assert 3 in ids_with
    assert 5 not in ids_with
    assert 6 in ids_with

    # The live request's own state is untouched: only the commit advances it.
    assert state.generated_tokens == [1]


def test_a_draft_that_violates_the_constraint_is_marked_invalid_not_fatal():
    """The walk advances rows by drafted tokens, and a draft may be invalid.

    A speculative cycle verifies a draft chain: the row's own token is the one
    the draft proposed, and the constraint may reject it. That is not a defect --
    the row's masked law excludes the token, so the accept walk corrects there
    and never publishes a later row. Raising instead aborted the whole cycle,
    which is what a live required-tool-constraint request did.
    """

    from hipengine.generation.constraints import ToolCallConstraintSpec

    params = _params(
        tool_call_constraint=ToolCallConstraintSpec(tool_names=("read",), mode="required")
    )
    # The root's drafted token is plain text, which a required constraint rejects.
    batch = _chain_batch((4, 5))
    state = RowSamplingState(
        seed=7,
        generated_tokens=(1,),
        tool_call_constraint=ToolCallConstraintSpec(tool_names=("read",), mode="required"),
    )
    texts = {4: "Sure, ", 5: "<", 6: "{"}

    walked = row_prefix_states(
        batch, {5: state}, observe_text=_text_observer(params, texts, drafted=True)
    )
    # The walk completes, and the row past the violating draft is masked empty:
    # it is unreachable, so its law may not admit anything.
    violating = walked[1]
    assert violating._tool_call_constraint.invalid
    assert violating._tool_call_constraint.error_reason == "invalid_tool_call_prefix"
    assert not violating._tool_call_constraint.accepts_text("<tool_call>")

    # The same token on the commit path is a defect: a published token that
    # violates the constraint means the mask and the DFA disagreed.
    with pytest.raises(ValueError, match="violates tool_call_constraint"):
        _text_observer(params, texts)(state, 4, 1)


def test_the_summary_wires_the_text_observer_into_both_walks():
    """The observer must reach the walk, not merely be accepted by the summary.

    ``sampled_accept_summary`` took ``observe_text`` and dropped it, so the
    adapter's walk-time observer never ran: a cycle's later rows were masked
    against a DFA that had not seen the earlier tokens, and the live route
    published a token a required tool constraint rejects. Driving the real
    function -- not ``row_prefix_states`` directly -- is what pins the wiring.
    """

    from hipengine.generation import mtp_sampled_accept as module
    from hipengine.generation.constraints import ToolCallConstraintSpec

    params = _params(
        tool_call_constraint=ToolCallConstraintSpec(tool_names=("read",), mode="required")
    )
    batch = _chain_batch((4,))
    logits = np.stack((_logits_row({4: 4.0, 5: 3.0, 6: 2.0}),) * batch.rows)
    state = RowSamplingState(
        seed=7,
        generated_tokens=(1,),
        tool_call_constraint=ToolCallConstraintSpec(tool_names=("read",), mode="required"),
    )
    # The drafted token is plain text, which a required constraint rejects.
    texts = {4: "Sure, ", 5: "<", 6: "{"}

    seen: dict[str, object] = {}
    real_walk = module._row_prefix_walk

    def spy(batch, states, *, observe_text=None):
        seen["observe_text"] = observe_text
        return real_walk(batch, states, observe_text=observe_text)

    original = module._row_prefix_walk
    module._row_prefix_walk = spy
    try:
        summary = sampled_accept_summary(
            batch,
            logits,
            {5: state},
            params_for=lambda request_id: params,
            draws=lambda: 0.0,
            token_text_for_id=lambda token_id: texts[int(token_id)],
            remaining_decode=(4,),
            observe_text=_text_observer(params, texts, drafted=True),
        )
    finally:
        module._row_prefix_walk = original

    # Both walks went through the observer, and the violating draft was not
    # published: the target law of the row that predicts it excludes it.
    assert seen["observe_text"] is not None
    assert 4 not in summary.accepted_tokens[0]
    assert 4 not in summary.next_tokens


def test_the_walk_observes_the_parent_token_at_the_childs_own_depth():
    """Depth is the count of tokens published before the row's own token."""

    batch = _chain_batch((2, 3))
    state = RowSamplingState(seed=7, generated_tokens=(1,))
    seen: list[tuple[int, int, tuple[int, ...]]] = []

    def observe_text(row_state, token_id, depth):
        seen.append((int(token_id), int(depth), tuple(row_state.generated_tokens)))

    row_prefix_states(batch, {5: state}, observe_text=observe_text)
    # Row 1 is the first candidate (the root's own token is the live state's),
    # and row 2 extends row 1, so it sees both drafted tokens.
    assert seen == [(2, 1, (1, 2)), (3, 2, (1, 2, 3))]


def test_the_commit_observes_only_the_published_tokens_text():
    """A row the accept walk never reached published nothing, so it advances nothing."""

    params = _params(json_object_close_forcing=True)
    state = RowSamplingState(
        seed=7, generated_tokens=(1,), json_object_close_forcing=True
    )
    texts = {2: "{", 3: "}", 4: "x"}
    observe_published_tokens(
        state,
        (2, 3),
        forced_count=0,
        observe_text=_text_observer(params, texts),
    )
    # Both published tokens advanced the live DFA, and the unused row's text
    # never did.
    assert state.generated_tokens == [1, 2, 3]
    assert "x" not in state._json_object_constraint.observed_text
