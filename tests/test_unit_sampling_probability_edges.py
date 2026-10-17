"""Analytic probability gates for the host sampler used by serving models."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.sampling import RowSamplingState, select_token


def params(**kwargs):
    return SimpleNamespace(temperature=0.0, logprobs=True, top_logprobs=4, **kwargs)


def test_forced_tail_logprob_does_not_underflow_or_invent_probability():
    # log p(tail) is finite even though exp(-2000) underflows in float64.
    result = select_token([1000, 999, -1000, -np.inf], params(),
                          RowSamplingState(forced_tokens_pending=(2,)))
    normalizer = np.log1p(np.exp(-1))
    assert result.logprob == pytest.approx(-2000 - normalizer)
    assert [token for token, _ in result.top_logprobs] == [0, 1, 2]
    assert [lp for _, lp in result.top_logprobs] == pytest.approx(
        [-normalizer, -1 - normalizer, -2000 - normalizer])
    assert all(np.isfinite(lp) and lp <= 0 for _, lp in result.top_logprobs)


def test_greedy_top_logprobs_include_finite_underflowed_tail_not_masked_ids():
    result = select_token([0, -1000, -2000, -np.inf], params())
    assert result.logprob == 0
    assert result.top_logprobs == ((0, 0.0), (1, -1000.0), (2, -2000.0))


@pytest.mark.parametrize('vocab', [4, 512])
@pytest.mark.parametrize('top_k', [1, 2])
def test_constrained_sampling_applies_top_k_for_small_and_large_vocab(vocab, top_k):
    values = np.zeros(vocab, dtype=np.float32)
    state = RowSamplingState(seed=17, json_object_close_forcing=True)
    request = SimpleNamespace(temperature=1.0, top_k=top_k, top_p=1.0,
                              logprobs=True, top_logprobs=4)
    # All candidate tokens begin a valid object. Equal logits break ties by ID.
    result = select_token(values, request, state, token_text_for_id=lambda _: '{')
    assert result.candidate_count == top_k
    assert result.token_id in range(top_k)
    assert [token for token, _ in result.top_logprobs] == list(range(top_k))
    assert [lp for _, lp in result.top_logprobs] == pytest.approx([-np.log(top_k)] * top_k)
