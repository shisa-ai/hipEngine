from __future__ import annotations

import pytest

from hipengine.generation import ForcedTokenQueue, ThinkingBudgetState, TokenSequenceDFAState, token_sequence_state_for_tokens


def test_token_sequence_dfa_reports_partial_suffix_candidates() -> None:
    state = token_sequence_state_for_tokens(
        [99, 10],
        ((10, 11), (10, 12, 13), (20,)),
    )

    assert state.matched is False
    assert state.to_json_dict() == {
        "partial_suffix": [10],
        "candidate_sequences": [[10, 11], [10, 12, 13]],
    }


def test_token_sequence_dfa_prefers_longest_overlapping_match() -> None:
    state = token_sequence_state_for_tokens(
        [1, 2],
        ((2,), (1, 2)),
    )

    assert state.matched is True
    assert state.matched_sequence == (1, 2)
    assert state.to_json_dict() == {"matched_sequence": [1, 2]}


def test_token_sequence_dfa_can_be_advanced_incrementally() -> None:
    state = TokenSequenceDFAState.from_sequences(((7, 8, 9),))
    state = state.observe(7).observe(8)

    assert state.matched is False
    assert state.to_json_dict() == {
        "partial_suffix": [7, 8],
        "candidate_sequences": [[7, 8, 9]],
    }

    state = state.observe(9)
    assert state.matched is True
    assert state.to_json_dict() == {"matched_sequence": [7, 8, 9]}


def test_token_sequence_dfa_rejects_negative_token_ids() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TokenSequenceDFAState.from_sequences(((-1,),))


def test_forced_token_queue_pops_fifo_and_reports_json_state() -> None:
    queue = ForcedTokenQueue((4, 5), reason="close_think")

    assert queue.pending_tokens == (4, 5)
    assert queue.to_json_dict() == {
        "pending_tokens": [4, 5],
        "reason": "close_think",
    }
    assert queue.peek() == 4
    assert queue.pop() == 4
    assert queue.pending_tokens == (5,)
    queue.extend((6,), reason="grammar")
    assert queue.pending_tokens == (5, 6)
    assert queue.to_json_dict()["reason"] == "grammar"


def test_forced_token_queue_rejects_negative_token_ids() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ForcedTokenQueue((-1,))


def test_thinking_budget_state_reports_soft_and_hard_pressure() -> None:
    state = ThinkingBudgetState(close_sequence=(10, 11), hard_token_cap=5, soft_close_window=2)

    for token_id in (1, 2):
        state.observe(token_id)
    assert state.remaining_think_tokens == 3
    assert state.budget_pressure is None

    state.observe(3)
    assert state.soft_close_active is True
    assert state.budget_pressure == "soft_close"
    assert state.soft_close_progress == 0.5
    assert state.soft_close_bias == 4.0

    state.observe(4)
    assert state.soft_close_progress == 1.0
    assert state.soft_close_bias == 8.0

    state.observe(5)
    assert state.hard_close_due is True
    assert state.budget_pressure == "hard_close"
    assert state.soft_close_bias is None


def test_thinking_budget_state_queues_close_suffix_after_soft_close_start() -> None:
    state = ThinkingBudgetState(close_sequence=(10, 11, 12), hard_token_cap=5, soft_close_window=2)

    state.observe(1).observe(2).observe(3)
    assert state.soft_close_active is True
    state.observe(10)

    assert state.phase == "closing_think"
    assert state.forced_tokens.pending_tokens == (11, 12)
    assert state.forced_tokens.to_json_dict()["reason"] == "thinking_soft_close"


def test_thinking_budget_state_enqueues_hard_close_once() -> None:
    state = ThinkingBudgetState(close_sequence=(10, 11), hard_token_cap=2)
    state.observe(1).observe(2)

    assert state.ensure_hard_close() is True
    assert state.phase == "closing_think"
    assert state.forced_tokens.pending_tokens == (10, 11)
    assert state.forced_tokens.to_json_dict()["reason"] == "thinking_hard_close"
    assert state.ensure_hard_close() is False


def test_thinking_budget_state_transitions_to_answer_after_close_sequence() -> None:
    state = ThinkingBudgetState(close_sequence=(10, 11), hard_token_cap=2)
    state.observe(1).observe(2)
    state.ensure_hard_close()

    state.observe(10)
    assert state.phase == "closing_think"
    assert state.to_json_dict()["close_state"] == {
        "partial_suffix": [10],
        "candidate_sequences": [[10, 11]],
    }

    state.observe(11)
    assert state.phase == "answer"
    assert state.reasoning_tokens == 4
    state.observe(99)
    assert state.answer_tokens == 1


def test_thinking_budget_state_manual_force_close_and_validation() -> None:
    state = ThinkingBudgetState(close_sequence=(10, 11))

    assert state.force_close(reason="controller") is True
    assert state.forced_tokens.pending_tokens == (10, 11)
    assert state.forced_tokens.to_json_dict()["reason"] == "controller"
    assert state.force_close(reason="controller") is False

    with pytest.raises(ValueError, match="close_sequence"):
        ThinkingBudgetState(close_sequence=(-1,))
    with pytest.raises(ValueError, match="hard_token_cap"):
        ThinkingBudgetState(hard_token_cap=-1)
    with pytest.raises(ValueError, match="phase"):
        ThinkingBudgetState(phase="invalid")
