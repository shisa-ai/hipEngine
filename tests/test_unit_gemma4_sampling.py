"""CPU contracts for the Gemma 4 host sampler and its exact speculative accept."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation import gemma4_sampling as module
from hipengine.generation.gemma4_sampling import (
    Gemma4Distribution,
    Gemma4Sampler,
    residual_distribution,
    speculative_accept,
)
from hipengine.generation.sampling import RowSamplingState, processed_distribution
from hipengine.speculative.sampling import SparseDistribution
from hipengine.speculative.sampling import residual_distribution as reference_residual

_VOCAB = 4096


def _params(**overrides):
    values = dict(
        temperature=0.8, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0,
        presence_penalty=0.0, frequency_penalty=0.0, logit_bias=None,
        suppress_token_ids=(), seed=7,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _row(seed, *, spread=4.0, vocab=_VOCAB):
    return (np.random.default_rng(seed).standard_normal(vocab) * spread).astype(np.float32)


def _dist(mapping):
    ids = sorted(mapping)
    return Gemma4Distribution(
        np.asarray(ids, dtype=np.int64), np.asarray([mapping[i] for i in ids], dtype=np.float64)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"top_k": 40},
        {"top_k": 64, "top_p": 0.95},
        {"top_p": 0.9},
        {"min_p": 0.05},
        {"temperature": 1.3, "top_k": 20, "top_p": 0.8, "min_p": 0.01},
        {"top_k": 40, "repetition_penalty": 1.15},
        {"presence_penalty": 0.4, "frequency_penalty": 0.2},
        {"logit_bias": {3: 4.0, 9: -3.0}, "suppress_token_ids": (5, 6)},
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_law_is_the_shared_host_samplers(overrides, seed):
    """Same support and probabilities as ``processed_distribution`` on the same history."""

    params = _params(**overrides)
    history = [int(t) for t in np.random.default_rng(seed + 50).integers(0, _VOCAB, 300)]
    row = _row(seed)
    sampler = Gemma4Sampler(params, history)
    got = sampler.distribution(row)
    want_ids, want_probs = processed_distribution(
        row, params, RowSamplingState(prompt_tokens=tuple(history), seed=7)
    )
    _assert_same_law(got, want_ids, want_probs, row, params.temperature)
    assert np.all(np.diff(got.token_ids) > 0)
    assert got.probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    if params.top_k:
        # A top-k law goes through the shared helpers on the same candidates.
        order = np.argsort(np.asarray(want_ids))
        np.testing.assert_array_equal(got.token_ids, np.asarray(want_ids)[order])
        np.testing.assert_allclose(
            got.probabilities, np.asarray(want_probs)[order], rtol=1e-12, atol=0
        )


def _assert_same_law(got, want_ids, want_probs, row, temperature):
    """Same law, each probability equal up to single-precision exponent rounding.

    The whole-row softmax exponentiates in single precision, whose relative
    error in ``exp(x)`` grows with ``|x|``: about 1e-7 per nat below the maximum.
    """

    dense = np.zeros(row.size)
    dense[got.token_ids] = got.probabilities
    reference = np.zeros(row.size)
    reference[np.asarray(want_ids)] = want_probs
    assert np.all(dense[reference == 0.0] == 0.0)
    # Single precision underflows some 87 nats below the maximum; what is lost
    # there is far less mass than a double resolves.
    assert dense[reference < 1e-30].sum() < 1e-25
    live = reference >= 1e-30
    assert np.all(dense[live] > 0.0)
    nats = -np.log(reference[live] / reference.max())
    assert np.all(np.abs(np.log(dense[live] / reference[live])) <= 3e-7 * (1.0 + nats))


@pytest.mark.parametrize("overrides", [{}, {"top_p": 0.97}, {"min_p": 1e-9}, {"top_p": 0.999999}])
def test_a_wide_row_matches_the_double_precision_softmax(overrides):
    """Without top-k the softmax covers the whole row, suppressed entries included."""

    rng = np.random.default_rng(9)
    row = (rng.standard_normal(20000) * 12.0).astype(np.float32)
    row[rng.integers(0, row.size, 40)] = -np.inf
    params = _params(temperature=0.7, **overrides)
    got = Gemma4Sampler(params, ()).distribution(row)
    want_ids, want_probs = processed_distribution(row, params, RowSamplingState(seed=7))
    _assert_same_law(got, want_ids, want_probs, row, params.temperature)
    dense = np.zeros(row.size)
    dense[got.token_ids] = got.probabilities
    assert dense[~np.isfinite(row)].sum() == 0.0


def test_a_draft_law_is_the_requests_over_at_most_the_draft_top_k():
    row = _row(8)
    wide = Gemma4Sampler(_params(), ())
    draft = wide.draft_distribution(row)
    assert draft.token_ids.size == module.DRAFT_TOP_K
    want = Gemma4Sampler(_params(top_k=module.DRAFT_TOP_K), ()).distribution(row)
    np.testing.assert_array_equal(draft.token_ids, want.token_ids)
    np.testing.assert_array_equal(draft.probabilities, want.probabilities)
    # A request already narrower than that keeps its own top-k.
    narrow = Gemma4Sampler(_params(top_k=8), ())
    assert narrow.draft_distribution(row).token_ids.size == 8
    greedy = Gemma4Sampler(_params(temperature=0.0), ())
    assert greedy.draft_distribution(row).token_ids.tolist() == [int(np.argmax(row))]


def test_sampling_a_large_law_is_the_inverse_cdf():
    rng = np.random.default_rng(2)
    for size in (5000, 4096 + 512, 262144, 262144 + 77):
        probs = rng.random(size)
        probs /= probs.sum()
        law = Gemma4Distribution(np.arange(size, dtype=np.int64), probs)
        cumulative = np.cumsum(probs)
        for unit in (0.0, 1e-9, 0.25, 0.5, 0.7310585, 0.999999, np.nextafter(1.0, 0.0)):
            token = law.sample(unit)
            # The drawn token's CDF interval contains the draw, up to the
            # rounding of summing in blocks rather than in one pass.
            low = cumulative[token - 1] if token else 0.0
            assert low - 1e-12 <= unit <= cumulative[token] + 1e-12


@pytest.mark.parametrize("overrides", [{"temperature": 0.0}, {"top_k": 1}])
def test_a_greedy_request_is_a_point_mass_on_the_lowest_argmax(overrides):
    row = _row(3)
    row[[11, 700]] = row.max() + 1.0
    sampler = Gemma4Sampler(_params(**overrides), ())
    assert sampler.greedy and not sampler.processes_logits
    got = sampler.distribution(row)
    assert got.token_ids.tolist() == [11] and got.probabilities.tolist() == [1.0]


def test_greedy_with_a_penalty_still_processes_the_row():
    row = np.zeros(16, dtype=np.float32)
    row[4], row[9] = 5.0, 4.9
    sampler = Gemma4Sampler(_params(temperature=0.0, repetition_penalty=2.0), [4])
    assert sampler.greedy and sampler.processes_logits
    assert sampler.distribution(row).token_ids.tolist() == [9]


def test_observe_and_unobserve_restore_the_history_exactly():
    params = _params(top_k=30, repetition_penalty=1.3, frequency_penalty=0.3)
    row = _row(4)
    sampler = Gemma4Sampler(params, [1, 2, 2])
    before = sampler.distribution(row)
    top = int(before.token_ids[int(np.argmax(before.probabilities))])
    for token in (top, 2, 77):
        sampler.observe(token)
    assert sampler.distribution(row).probability(top) < before.probability(top)
    for token in (77, 2, top):
        sampler.unobserve(token)
    after = sampler.distribution(row)
    np.testing.assert_array_equal(after.token_ids, before.token_ids)
    np.testing.assert_array_equal(after.probabilities, before.probabilities)
    with pytest.raises(ValueError, match="was not observed"):
        sampler.unobserve(77)


def test_the_request_and_draft_streams_are_seeded_and_separate():
    first, second = (Gemma4Sampler(_params(seed=11), ()) for _ in range(2))
    assert [first.random_unit() for _ in range(4)] == [second.random_unit() for _ in range(4)]
    assert first.draft_unit() == second.draft_unit()
    other = Gemma4Sampler(_params(seed=12), ())
    assert other.random_unit() != Gemma4Sampler(_params(seed=11), ()).random_unit()
    # Drawing drafts does not advance the request's own stream.
    drafted = Gemma4Sampler(_params(seed=11), ())
    for _ in range(9):
        drafted.draft_unit()
    assert drafted.random_unit() == Gemma4Sampler(_params(seed=11), ()).random_unit()


def test_select_draws_from_the_law_and_records_the_token():
    sampler = Gemma4Sampler(_params(top_k=8, frequency_penalty=1.0, seed=3), ())
    row = _row(5)
    law = sampler.distribution(row)
    token = sampler.select(row)
    assert law.probability(token) > 0.0
    assert sampler.distribution(row).probability(token) < law.probability(token)


def test_a_row_without_a_finite_maximum_is_refused():
    row = _row(6)
    row[10] = np.nan
    with pytest.raises(ValueError, match="no finite maximum"):
        Gemma4Sampler(_params(), ()).distribution(row)


def test_distribution_probability_and_sampling():
    law = _dist({2: 0.25, 5: 0.5, 9: 0.25})
    assert [law.probability(t) for t in (2, 5, 9, 4, 100)] == [0.25, 0.5, 0.25, 0.0, 0.0]
    assert [law.sample(u) for u in (0.0, 0.2499, 0.25, 0.7499, 0.75, 0.9999)] == [2, 2, 5, 5, 9, 9]
    with pytest.raises(ValueError):
        law.sample(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_residual_matches_the_reference_sparse_implementation(seed):
    rng = np.random.default_rng(seed)

    def random_law():
        ids = np.sort(rng.choice(40, size=int(rng.integers(1, 12)), replace=False))
        probs = rng.random(ids.size)
        return _dist({int(i): float(p) for i, p in zip(ids, probs / probs.sum())})

    target, draft = random_law(), random_law()
    got = residual_distribution(target, draft)
    want = reference_residual(
        SparseDistribution(tuple(target.token_ids.tolist()), target.probabilities),
        SparseDistribution(tuple(draft.token_ids.tolist()), draft.probabilities),
    )
    assert got.token_ids.tolist() == list(want.token_ids)
    np.testing.assert_allclose(got.probabilities, want.probabilities, rtol=1e-12)


def test_residual_is_none_when_the_draft_law_is_the_target_law():
    law = _dist({1: 0.5, 2: 0.5})
    assert residual_distribution(law, law) is None


def _draws(values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_greedy_accept_is_argmax_equality_and_a_rejection_emits_the_argmax():
    targets = [Gemma4Distribution.point_mass(t) for t in (10, 11, 12, 13)]
    drafts = [10, 11, 99]
    asked = []

    def target(index):
        asked.append(index)
        return targets[index]

    accepted, token = speculative_accept(
        drafts, [Gemma4Distribution.point_mass(t) for t in drafts], target, _draws([0.0] * 9)
    )
    assert (accepted, token) == (2, 12)
    # The walk never asks for a row past the rejection.
    assert asked == [0, 1, 2]
    accepted, token = speculative_accept(
        drafts[:2], [Gemma4Distribution.point_mass(t) for t in drafts[:2]],
        lambda index: targets[index], _draws([0.999] * 9),
    )
    assert (accepted, token) == (2, 12)


def test_accept_thresholds_and_draw_order():
    target = _dist({1: 0.2, 2: 0.8})
    draft = _dist({1: 0.5, 2: 0.5})
    # p/q for token 1 is 0.4: a draw below accepts, a draw at it rejects, and
    # the rejection's residual is all on token 2.
    assert speculative_accept([1], [draft], lambda i: target, _draws([0.39, 0.1])) == (1, 1)
    assert speculative_accept([1], [draft], lambda i: target, _draws([0.4, 0.99])) == (0, 2)
    # p/q for token 2 is 1.6, capped at one: it accepts for every draw.
    assert speculative_accept([2], [draft], lambda i: target, _draws([0.9999, 0.9])) == (1, 2)


def test_a_draft_outside_its_own_law_is_refused():
    with pytest.raises(ValueError, match="zero probability under its own draft law"):
        speculative_accept([3], [_dist({1: 1.0})], lambda i: _dist({1: 1.0}), _draws([0.0]))
    with pytest.raises(ValueError, match="every draft needs"):
        speculative_accept([3], [], lambda i: _dist({1: 1.0}), _draws([0.0]))


@pytest.mark.parametrize("sampled_draft", [True, False])
def test_the_emitted_token_follows_the_target_law_for_any_draft_law(sampled_draft):
    """The first committed token is distributed as ``p``, sampled draft or point mass."""

    rng = np.random.default_rng(123)
    target = _dist({0: 0.05, 1: 0.45, 2: 0.3, 3: 0.2})
    draft_law = _dist({0: 0.4, 1: 0.1, 2: 0.25, 4: 0.25})
    trials = 60000
    counts = np.zeros(5)
    for _ in range(trials):
        if sampled_draft:
            draft, law = draft_law.sample(rng.random()), draft_law
        else:
            draft, law = 0, Gemma4Distribution.point_mass(0)
        accepted, token = speculative_accept([draft], [law], lambda i: target, rng.random)
        counts[draft if accepted else token] += 1
    expected = np.asarray([0.05, 0.45, 0.3, 0.2, 0.0])
    # Five standard errors of a binomial proportion at this trial count.
    tolerance = 5.0 * np.sqrt(expected * (1.0 - expected) / trials) + 1e-9
    assert np.all(np.abs(counts / trials - expected) <= tolerance)


def test_a_sampled_draft_is_accepted_more_often_than_its_point_mass():
    """Acceptance is ``sum(min(p, q))`` for a sampled draft and ``p(argmax q)`` otherwise."""

    target = _dist({0: 0.4, 1: 0.35, 2: 0.25})
    draft_law = _dist({0: 0.45, 1: 0.35, 2: 0.2})
    rng = np.random.default_rng(5)
    trials = 40000
    sampled = point = 0
    for _ in range(trials):
        draft = draft_law.sample(rng.random())
        sampled += speculative_accept([draft], [draft_law], lambda i: target, rng.random)[0]
        point += speculative_accept(
            [0], [Gemma4Distribution.point_mass(0)], lambda i: target, rng.random
        )[0]
    assert sampled / trials == pytest.approx(0.4 + 0.35 + 0.2, abs=0.01)
    assert point / trials == pytest.approx(0.4, abs=0.01)
