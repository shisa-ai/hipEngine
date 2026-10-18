"""Fast host candidate ordering must preserve the original lexicographic law."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation import sampling


def _parent_candidates(values, top_k):
    ids = np.flatnonzero(np.isfinite(values)).astype(np.int64, copy=False)
    order = np.lexsort((ids, -values[ids]))
    result = ids[order]
    return result[:min(top_k, result.size)] if top_k > 0 else result


def test_unique_candidate_order_does_not_use_two_key_sort(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("unique scores do not need two-key sorting")

    monkeypatch.setattr(np, "lexsort", unexpected)
    np.testing.assert_array_equal(
        sampling._top_k_candidate_ids(np.array([1.0, 4.0, 2.0, 3.0]), 0),
        [1, 3, 2, 0],
    )


@pytest.mark.parametrize("size", [0, 1, 2, 257, 248320])
@pytest.mark.parametrize("top_k", [0, 1, 17, 1000000])
@pytest.mark.parametrize("distribution", ["random", "tied", "uniform", "nonfinite"])
def test_fast_candidates_match_frozen_parent(size, top_k, distribution):
    rng = np.random.default_rng(17)
    values = rng.normal(size=size)
    if distribution == "tied":
        values = np.round(values * 8) / 8
    elif distribution == "uniform":
        values[:] = 0.0
        values[::2] = -0.0
    elif distribution == "nonfinite":
        values[::3] = np.nan
        values[1::3] = np.inf
        values[2::7] = -np.inf
    original = values.copy()
    actual = sampling._top_k_candidate_ids(values, top_k)
    np.testing.assert_array_equal(actual, _parent_candidates(values, top_k))
    np.testing.assert_array_equal(values, original)
    assert actual.dtype == np.int64


@pytest.mark.parametrize("top_k,top_p,min_p", [(0, .95, 0), (0, 1, .1), (17, .7, .02)])
def test_candidate_order_preserves_seeded_tokens_and_logprobs(monkeypatch, top_k, top_p, min_p):
    params = SimpleNamespace(temperature=.7, top_k=top_k, top_p=top_p,
                             min_p=min_p, logprobs=True, top_logprobs=5)
    rng = np.random.default_rng(23)
    logits = np.round(rng.normal(size=4096) * 8) / 8
    fast = sampling._top_k_candidate_ids
    results = []
    for candidate_fn in (_parent_candidates, fast):
        monkeypatch.setattr(sampling, "_top_k_candidate_ids", candidate_fn)
        state = sampling.RowSamplingState(seed=31)
        results.append([sampling.select_token(logits, params, state) for _ in range(12)])
    assert results[0] == results[1]
