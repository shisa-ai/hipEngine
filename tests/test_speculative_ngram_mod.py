from __future__ import annotations

import pytest

from hipengine.speculative.ngram_mod import (
    NgramModConfig,
    NgramModRequestCache,
    RequestLocalNgramMod,
)


def _replay_history(*, n_match: int = 24, continuation: int = 24) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prefix = tuple(range(100, 100 + n_match))
    suffix = tuple(range(1_000, 1_000 + continuation))
    return (*prefix, *suffix, *prefix), suffix


def test_ngram_mod_requires_a_full_confidence_horizon_before_returning_k3() -> None:
    history, continuation = _replay_history()
    cache = NgramModRequestCache(
        NgramModConfig(n_match=24, min_draft_tokens=24, max_probe_tokens=64)
    )

    cache.sync_committed(history)
    proposal = cache.propose(max_candidates=3)

    assert proposal is not None
    assert proposal.candidate_tokens == continuation[:3]
    assert proposal.probed_tokens == 24
    assert proposal.n_match == 24


def test_ngram_mod_rejects_history_without_an_exact_24_token_replay() -> None:
    history = tuple(range(71))
    cache = NgramModRequestCache(
        NgramModConfig(n_match=24, min_draft_tokens=24, max_probe_tokens=64)
    )

    cache.sync_committed(history)

    assert cache.propose(max_candidates=3) is None


def test_ngram_mod_uses_latest_exact_occurrence_without_hash_collision_aliasing() -> None:
    prefix = tuple(range(24))
    first = tuple(range(100, 124))
    latest = tuple(range(200, 224))
    history = (*prefix, *first, *prefix, *latest, *prefix)
    cache = NgramModRequestCache(
        NgramModConfig(n_match=24, min_draft_tokens=3, max_probe_tokens=3)
    )

    cache.sync_committed(history)

    assert cache.propose(max_candidates=3).candidate_tokens == latest[:3]  # type: ignore[union-attr]


def test_ngram_mod_rebuilds_on_non_append_history_and_validates_tokens() -> None:
    history, continuation = _replay_history()
    cache = NgramModRequestCache(
        NgramModConfig(n_match=24, min_draft_tokens=24, max_probe_tokens=24)
    )
    cache.sync_committed(history)
    assert cache.propose(max_candidates=3) is not None

    replacement = tuple(range(500, 550))
    cache.sync_committed(replacement)

    assert cache.committed_tokens == replacement
    assert cache.propose(max_candidates=3) is None
    with pytest.raises(ValueError, match="non-negative"):
        cache.sync_committed((*replacement, -1))
    assert continuation[:3] != replacement[:3]


def test_request_local_ngram_mod_never_leaks_candidates_between_requests() -> None:
    history, continuation = _replay_history()
    composer = RequestLocalNgramMod(
        NgramModConfig(n_match=24, min_draft_tokens=24, max_probe_tokens=24)
    )

    hit = composer.propose(7, history, max_candidates=3)
    miss = composer.propose(8, history[-24:], max_candidates=3)

    assert hit is not None and hit.candidate_tokens == continuation[:3]
    assert miss is None
    assert composer.request_ids == (7, 8)
    composer.release_request(7)
    assert composer.request_ids == (8,)


def test_ngram_mod_config_fails_closed_on_unbounded_or_incoherent_shapes() -> None:
    with pytest.raises(ValueError, match="n_match"):
        NgramModConfig(n_match=0)
    with pytest.raises(ValueError, match="min_draft_tokens"):
        NgramModConfig(n_match=24, min_draft_tokens=0)
    with pytest.raises(ValueError, match="max_probe_tokens"):
        NgramModConfig(n_match=24, min_draft_tokens=25, max_probe_tokens=24)
