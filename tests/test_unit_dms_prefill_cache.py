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


@pytest.mark.parametrize("budget,expected", [(6000, 256), (10000, 512), (100, None)])
def test_oversized_snapshot_retains_owned_boundary_within_budget(budget, expected):
    from hipengine.runtime.dms_prefix import DMSPrefixCache, DMSPromptSnapshot
    tokens = tuple(range(900))
    keys = np.arange(1800, dtype=np.uint8)
    cold = DMSColdPrefill(tokens, ((keys, keys.copy()),),
                          np.zeros((1, 900, 1), np.float32),
                          np.zeros((1, 900, 1), np.uint8),
                          np.zeros((900, 8), np.uint8),
                          {256: (np.ones(20, np.uint8),),
                           512: (np.ones(20, np.uint8),)})
    runner = object()
    session = SimpleNamespace(runner=runner, dms_backend_factory=None,
                              _dms_source=SimpleNamespace(config=SimpleNamespace(fingerprint="policy")))
    source = DMSPromptSnapshot(runner, None, "policy", tokens, False,
                               SimpleNamespace(nbytes=1000), (np.zeros(20, np.uint8),),
                               None, cold, (1234, 900))
    cache = DMSPrefixCache(budget)
    assert cache.put(source) == (source,)
    assert cache.nbytes <= budget
    prefix = cache.lookup(session, (*tokens, 9999), False)
    if expected is None:
        assert prefix is None
        return
    assert len(prefix.tokens) == expected
    assert not prefix.complete and prefix.compact is None and prefix.result is None
    assert prefix.checkpoint_key == (1234, expected)
    exact = cache.lookup(session, prefix.tokens, False)
    assert exact is None or (not exact.complete and len(exact.tokens) < expected)
    pairs = [(prefix.cold.layers[0][0], keys),
             (prefix.cold.hidden, cold.hidden),
             (prefix.cold.predictor_logits, cold.predictor_logits),
             (prefix.cold.predictor_decisions, cold.predictor_decisions)]
    for retained, original in pairs:
        assert not np.shares_memory(retained, original)
        assert retained.base is None
        assert not retained.flags.writeable
    if expected == 512:
        branch = cache.lookup(session, (*tokens[:300], -1), False)
        assert len(branch.tokens) == 256


@pytest.mark.parametrize("chunk_outer", [False, True])
def test_cache_boundaries_follow_natural_recurrent_chunks(chunk_outer):
    from hipengine.runtime.dms_prefill_cache import cache_boundary_points
    session = SimpleNamespace(
        _dms_cache_capture_tokens=tuple(range(12001)),
        _dms_cache_byte_budget=1000,
        scratch=SimpleNamespace(layer_conv_states=[SimpleNamespace(nbytes=150)],
                                layer_recurrent_states=[None]),
        _linear_prefill_layer_chunk_size=lambda total: 256 if chunk_outer else 4096,
        _prefill_scratch_rows=lambda total: 4096,
        _prefill_hidden_a=SimpleNamespace(nbytes=(4096 if chunk_outer else 12001) * 16),
        _bulk_prefill_scratch=SimpleNamespace(rows=4096),
        runner=SimpleNamespace(hidden_size=8, weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4))))
    assert {4096, 8192} <= set(cache_boundary_points(session))
    assert len(cache_boundary_points(session)) <= 3
    # A borrowed tail moves the actual capture boundary off the old grid.
    session._dms_cache_capture_tokens = tuple(range(4097))
    assert 4093 in cache_boundary_points(session)
    # Exact prefix reuse has no new chunks, but retains reusable boundaries.
    session._dms_prefill_reuse = SimpleNamespace(
        tokens=session._dms_cache_capture_tokens,
        cold=SimpleNamespace(boundaries={4093: ()}),
    )
    assert {4093, 4097} <= set(cache_boundary_points(session))


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
