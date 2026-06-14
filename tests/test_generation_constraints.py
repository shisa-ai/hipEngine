from __future__ import annotations

import pytest

from hipengine.generation import TokenSequenceDFAState, token_sequence_state_for_tokens


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
