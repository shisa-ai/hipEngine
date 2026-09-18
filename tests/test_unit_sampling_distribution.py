"""The speculative route must sample the same distribution the AR route samples.

``processed_distribution`` is the single source of truth for a row's sampling
support: ``select_token`` now calls it, and the sampled speculative accept path
calls it too. These tests pin that contract from the AR side — same support,
same weights, same token for the same draw, no state mutation — so a later
change to one route cannot silently desynchronize the other.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.sampling import (
    RowSamplingState,
    processed_distribution,
    sample_support,
    select_token,
)


def _params(**overrides):
    values = {
        "temperature": 0.0,
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
        "seed": 1234,
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


def _row(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


_SAMPLED_PARAMS = {
    "temperature_1": _params(temperature=1.0),
    "temperature_low": _params(temperature=0.35),
    "temperature_high": _params(temperature=1.7),
    "top_k_2": _params(temperature=1.0, top_k=2),
    "top_p_0.6": _params(temperature=1.0, top_p=0.6),
    "min_p_0.2": _params(temperature=1.0, min_p=0.2),
    "top_k_top_p_min_p": _params(temperature=0.8, top_k=3, top_p=0.9, min_p=0.05),
    "repetition_penalty": _params(temperature=1.0, repetition_penalty=1.4),
    "presence_penalty": _params(temperature=1.0, presence_penalty=0.7),
    "frequency_penalty": _params(temperature=1.0, frequency_penalty=0.4),
    "logit_bias": _params(temperature=1.0, logit_bias={3: 6.0, 0: -6.0}),
    "suppressed": _params(temperature=1.0, suppress_token_ids=(0, 1)),
    "logprobs": _params(temperature=1.0, logprobs=True, top_logprobs=3),
}

_ROWS = {
    "peaked": _row((8.0, 4.0, 2.0, 1.0, 0.5, 0.0, -1.0)),
    "flat": _row((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
    "long_tail": _row(tuple(3.0 - 0.5 * index for index in range(12))),
    "negatives": _row((-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)),
    "ties": _row((2.0, 2.0, 2.0, 1.0, 1.0, 1.0)),
    "with_nan": _row((5.0, float("nan"), 1.0, 0.0, -1.0, -2.0, -3.0)),
}


@pytest.mark.parametrize("params_name", sorted(_SAMPLED_PARAMS))
@pytest.mark.parametrize("row_name", sorted(_ROWS))
def test_distribution_matches_select_token_for_the_same_draw(
    params_name: str, row_name: str
) -> None:
    params = _SAMPLED_PARAMS[params_name]
    logits = _ROWS[row_name]
    ar_state = RowSamplingState(seed=99, prompt_tokens=(1, 2, 3))
    result = select_token(logits, params, ar_state)
    distribution_state = RowSamplingState(seed=99, prompt_tokens=(1, 2, 3))
    token_ids, probabilities = processed_distribution(logits, params, distribution_state)
    # The AR path drew exactly one uniform before selecting, so replaying that
    # draw over the shared support must select the identical token.
    draw = RowSamplingState(seed=99, prompt_tokens=(1, 2, 3)).random_unit()
    replayed, replayed_probability = sample_support(
        np.asarray(token_ids, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        draw,
    )
    assert replayed == result.token_id
    assert replayed_probability == pytest.approx(float(np.exp(result.logprob)))


@pytest.mark.parametrize("params_name", sorted(_SAMPLED_PARAMS))
def test_distribution_weights_are_the_reported_logprobs(params_name: str) -> None:
    params = _SAMPLED_PARAMS[params_name]
    logits = _ROWS["peaked"]
    result = select_token(logits, params, RowSamplingState(seed=7))
    token_ids, probabilities = processed_distribution(logits, params)
    weights = {int(token): float(weight) for token, weight in zip(token_ids, probabilities, strict=True)}
    assert result.token_id in weights
    assert weights[result.token_id] == pytest.approx(float(np.exp(result.logprob)))
    assert sum(weights.values()) == pytest.approx(1.0)


def test_greedy_row_reports_the_argmax_as_a_point_mass() -> None:
    logits = _ROWS["peaked"]
    token_ids, probabilities = processed_distribution(logits, _params(temperature=0.0))
    assert token_ids == (0,)
    assert probabilities.tolist() == [1.0]


def test_greedy_row_matches_select_token_argmax_with_processors() -> None:
    params = _params(temperature=0.0, logit_bias={3: 10.0})
    logits = _ROWS["peaked"]
    result = select_token(logits, params, RowSamplingState(seed=3))
    token_ids, _ = processed_distribution(logits, params, RowSamplingState(seed=3))
    assert result.token_id == token_ids[0] == 3


def test_unsorted_tie_breaks_like_the_ar_path() -> None:
    """Lower id wins an exact tie in both routes (``_argmax_lower_id``)."""

    logits = _row((1.0, 5.0, 5.0, 1.0))
    params = _params(temperature=0.0)
    result = select_token(logits, params, RowSamplingState(seed=1))
    token_ids, _ = processed_distribution(logits, params)
    assert result.token_id == token_ids[0] == 1


def test_distribution_does_not_mutate_state_or_consume_draws() -> None:
    params = _params(temperature=1.0, repetition_penalty=1.2)
    logits = _ROWS["peaked"]
    state = RowSamplingState(seed=11, prompt_tokens=(4, 5), generated_tokens=(6,))
    before = (tuple(state.generated_tokens), state.step_index)
    first_ids, first_probs = processed_distribution(logits, params, state)
    second_ids, second_probs = processed_distribution(logits, params, state)
    assert (tuple(state.generated_tokens), state.step_index) == before
    assert first_ids == second_ids
    assert np.array_equal(first_probs, second_probs)
    # The state's first draw is still available to the caller.
    assert state.random_unit() == RowSamplingState(seed=11, prompt_tokens=(4, 5), generated_tokens=(6,)).random_unit()


def test_history_penalties_flow_into_the_distribution() -> None:
    params = _params(temperature=1.0, repetition_penalty=2.0)
    logits = _ROWS["peaked"]
    clean_state = RowSamplingState(seed=5)
    penalized_state = RowSamplingState(seed=5, generated_tokens=(0, 0, 0))
    clean_ids, clean_probs = processed_distribution(logits, params, clean_state)
    penalized_ids, penalized_probs = processed_distribution(logits, params, penalized_state)
    clean = {int(token): float(weight) for token, weight in zip(clean_ids, clean_probs, strict=True)}
    penalized = {
        int(token): float(weight) for token, weight in zip(penalized_ids, penalized_probs, strict=True)
    }
    assert clean[0] > penalized[0]
    assert sum(penalized.values()) == pytest.approx(1.0)


def test_suppression_can_empty_the_support() -> None:
    params = _params(temperature=1.0, suppress_token_ids=(0, 1, 2, 3, 4, 5, 6))
    with pytest.raises(ValueError, match="removed all finite logits"):
        processed_distribution(_ROWS["peaked"], params)


def test_forced_token_state_is_reported_through_the_ar_path() -> None:
    """A pending forced token is the AR decision; the distribution stays defined."""

    params = _params(temperature=1.0, forced_tokens_pending=(3,))
    logits = _ROWS["peaked"]
    result = select_token(logits, params)
    assert result.token_id == 3 and result.forced
    # The distribution helper describes the row's sampling law; a forced token is
    # a caller-level override, so the support is still the sampled support.
    token_ids, _ = processed_distribution(logits, params)
    assert 3 in token_ids


def test_top_k_one_reduces_the_support_to_the_argmax() -> None:
    logits = _ROWS["peaked"]
    token_ids, probabilities = processed_distribution(
        logits, _params(temperature=1.0, top_k=1)
    )
    assert token_ids == (0,)
    assert probabilities.tolist() == [1.0]


def test_distribution_requires_a_positive_temperature_for_support() -> None:
    from hipengine.generation.sampling import processed_support

    with pytest.raises(ValueError, match="positive temperature"):
        processed_support(
            np.asarray([1.0, 2.0], dtype=np.float64),
            _params(temperature=0.0),
            temperature=0.0,
        )
