"""Cold DMS cache scores preserve full-prompt selection across extension."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.dms_prefill_cache import DMSColdPrefill, seed_collector
from hipengine.kvcache.dms import build_dms_exact_budget_eviction


def test_predictor_seed_uses_destination_stride_and_preserves_raw_scores(monkeypatch):
    from hipengine.runtime import dms_prefill_cache as cache
    scores = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    decisions = np.ones((2, 3, 2), np.uint8)
    cold = DMSColdPrefill((1, 2, 3), ((np.zeros(1), np.zeros(1)),) * 2,
                          scores, decisions, np.zeros((3, 4), np.uint8), {})
    collector = SimpleNamespace(num_kv_heads=2, token_count=7,
                                _decisions=SimpleNamespace(ptr=100),
                                _logits=SimpleNamespace(ptr=1000), _next=np.zeros(2, np.int32))
    writes = []
    monkeypatch.setattr(cache, "write_prefix", lambda buf, data: writes.append((buf.ptr, data.copy())))
    seed_collector(collector, cold)
    assert [ptr for ptr, _ in writes] == [100, 1000, 114, 1056]
    np.testing.assert_array_equal(writes[3][1], scores[1])
    np.testing.assert_array_equal(collector._next, [3, 3])
    assert cold.reused_tokens((1, 2, 3, 4)) == 3
    assert cold.reused_tokens((1, 2, 9, 4)) == 0
    assert cold.reused_tokens((1, 2)) == 0
    with pytest.raises(ValueError, match="suffix"):
        seed_collector(SimpleNamespace(num_kv_heads=2, token_count=3), cold)


def test_cache_keeps_branches_and_evicts_by_bytes_and_recency():
    from hipengine.runtime.dms_prefix import DMSPrefixCache

    def snapshot(tokens, nbytes=3):
        tokens = tuple(tokens)
        return SimpleNamespace(
            tokens=tokens, nbytes=nbytes, compatible=lambda session: True,
            matches=lambda session, prompt, sampled: tuple(prompt) == tokens,
            cold=SimpleNamespace(boundaries={}, reused_tokens=lambda prompt:
                                 len(tokens) if tuple(prompt[:len(tokens)]) == tokens
                                 and len(prompt) > len(tokens) else 0),
        )

    cache = DMSPrefixCache(max_bytes=9, max_entries=3)
    a, b, c = snapshot((1,)), snapshot((1, 2)), snapshot((9,))
    assert cache.put(a) == cache.put(b) == cache.put(c) == ()
    assert cache.lookup(None, (1, 2, 3), False) is b
    assert cache.lookup(None, (1, 4), False) is a
    d = snapshot((8,))
    assert cache.put(d) == (c,)
    assert cache.nbytes == 9
    oversized = snapshot((7,), 10)
    assert cache.put(oversized) == (oversized,)
    assert cache.nbytes == 9


def test_branch_uses_boundary_recurrence_and_sliced_cold_kv():
    from hipengine.runtime.dms_prefix import DMSPrefixCache, DMSPromptSnapshot
    tokens = tuple(range(300))
    keys = np.arange(600, dtype=np.uint8)
    recurrence = (np.array([17], np.uint8), np.array([23], np.uint8))
    cold = DMSColdPrefill(tokens, ((keys, keys.copy()),),
                          np.zeros((1, 300, 1), np.float32),
                          np.zeros((1, 300, 1), np.uint8),
                          np.zeros((300, 8), np.uint8), {256: recurrence})
    runner = object()
    session = SimpleNamespace(runner=runner, dms_backend_factory=None,
                              _dms_source=SimpleNamespace(config=SimpleNamespace(fingerprint="policy")))
    source = DMSPromptSnapshot(runner, None, "policy", tokens, False,
                               SimpleNamespace(nbytes=1), (np.array([99], np.uint8),),
                               None, cold, (1234, 300))
    cache = DMSPrefixCache(10000)
    cache.put(source)
    prefix = cache.lookup(session, (*tokens[:270], 999, 1000), False)
    assert prefix.tokens == tokens[:256]
    assert not prefix.complete and prefix.compact is None
    assert prefix.checkpoint_key == (1234, 256)
    assert prefix.recurrent_and_logits is recurrence
    np.testing.assert_array_equal(prefix.cold.layers[0][0], keys[:512])
    assert prefix.cold.hidden.shape == (256, 8)
    assert cache.lookup(session, tokens[:256], False) is None  # no boundary logits snapshot
    session._dms_source.config.fingerprint = "other-policy"
    assert cache.lookup(session, tokens, False) is None


def test_completed_boundaries_require_every_recurrent_layer():
    from hipengine.runtime.dms_prefill_cache import completed_boundaries
    a, b = np.array([1], np.uint8), np.array([2], np.uint8)
    session = SimpleNamespace(
        scratch=SimpleNamespace(layer_conv_states=[object(), None, object()],
                                layer_recurrent_states=[object(), None, object()]),
        _dms_cache_boundaries={256: {0: (a,), 2: (b,)}, 512: {0: (a,)}})
    assert completed_boundaries(session) == {256: (a, b)}
    # A boundary that some layer skipped is not a restorable prefix.
    assert 512 not in completed_boundaries(session)


def test_exact_budget_selection_is_recomputed_not_copied_from_shorter_prompt():
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(40, 2, 2)).astype(np.float32)
    prefix = scores[:17].copy()
    reconstructed = np.concatenate((prefix, scores[17:]), axis=0)
    options = dict(window_size=4, target_compression_ratio=2)
    fresh = build_dms_exact_budget_eviction(scores, current_position=39, **options)
    cached = build_dms_exact_budget_eviction(reconstructed, current_position=39, **options)
    np.testing.assert_array_equal(cached, fresh)
    old = build_dms_exact_budget_eviction(prefix, current_position=16, **options)
    assert np.any(old != fresh[:17]), "fixture must require reselection of the cached prefix"
