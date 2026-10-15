"""Compact prefix copies survive reclaim and restore at different physical bases."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kvcache.dms import DMSCompactBackend, DMSRetrofitConfig
from hipengine.kvcache.dms_prefix import capture_prefix, restore_prefix


def _backend(codec):
    config = DMSRetrofitConfig(
        artifact_fingerprint="fixture:prefix", model_family="qwen35",
        num_layers=1, num_q_heads=4, num_kv_heads=2, head_dim=16,
        window_size=4, target_compression_ratio=2, alpha_scale=100.,
        alpha_offset=5., borrowed_query_channel=15, corrected_mask=True,
        trained_checkpoint=True, evidence_source="unit fixture", source_path="fixture",
    )
    return DMSCompactBackend(retrofit=config, codec=codec, slots_per_layer=256,
                             max_request_rows=4, max_pack_rows=32, device_payloads=False)


def _reserve(backend, request_id, capacity):
    request = SimpleNamespace(request_id=request_id, prompt_tokens=(), max_new_tokens=0)
    return backend.reserve(backend.estimate(request, None, {
        "kind": "admission", "tokens": 0, "max_new_tokens": 0,
        "per_head_slots": capacity, "logical_prompt_tokens": 9,
    }))


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_snapshot_survives_reclaim_and_relocates_without_aliasing(codec):
    source = _backend(codec)
    lease = _reserve(source, 7, 12)
    rng = np.random.default_rng(42)
    values = rng.normal(size=(9, 1, 2, 16)).astype(np.float32)
    eviction = np.zeros((9, 1, 2), dtype=bool)
    eviction[:4, 0, 0] = True  # Head counts differ; logical positions are not slot indices.
    source.streaming_pack(7, values, -values, eviction)
    snapshot = capture_prefix(source, 7)
    assert snapshot.live_counts[0, 0] != snapshot.live_counts[0, 1]
    saved = snapshot.extents[0].k_bits.copy()
    source.reclaim(lease)
    source.close()

    target = _backend(codec)
    _reserve(target, 0, 3)  # Occupy the old physical base.
    _reserve(target, 8, 20)
    state = target.state_for_request(8)
    bases, capacity = state.base_offsets.copy(), state.range_capacity.copy()
    restore_prefix(target, 8, snapshot)
    np.testing.assert_array_equal(state.base_offsets, bases)
    np.testing.assert_array_equal(state.range_capacity, capacity)
    assert state.logical_tokens == 9
    assert np.all(bases > 0)
    for item in snapshot.extents:
        key = item.layer, item.head
        expected_k = ((item.k_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
                      if codec == "bf16" else item.k_bits)
        np.testing.assert_array_equal(state.k_payload[key], expected_k)
        expected_v = ((item.v_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
                      if codec == "bf16" else item.v_bits)
        np.testing.assert_array_equal(state.v_payload[key], expected_v)
        np.testing.assert_array_equal(state.token_positions[key][:item.length], item.positions)
        np.testing.assert_array_equal(state.evict_mask[key][:item.length], item.evict)
        if codec != "bf16":
            np.testing.assert_array_equal(state.k_scales[key], item.k_scales)
            np.testing.assert_array_equal(state.v_scales[key], item.v_scales)
    target.append_decode(8, values[0], -values[0], np.zeros((1, 2), bool), position=9)
    np.testing.assert_array_equal(snapshot.extents[0].k_bits, saved)
    assert not snapshot.extents[0].k_bits.flags.writeable
    target.assert_conserved()
    target.close()


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_device_restore_uses_new_extent_bases_and_live_counts(codec):
    source = _backend(codec)
    _reserve(source, 0, 12)
    values = np.ones((9, 1, 2, 16), np.float32)
    source.streaming_pack(0, values, values, np.zeros((9, 1, 2), bool))
    snapshot = capture_prefix(source, 0)
    target = _backend(codec)
    _reserve(target, 3, 5)
    _reserve(target, 4, 18)
    restored, configured = [], []
    target._device_store = SimpleNamespace(
        restore=restored.append,
        configure_layer=lambda layer, **kw: configured.append((layer, kw)),
        close=lambda: None,
    )
    restore_prefix(target, 4, snapshot)
    state = target.state_for_request(4)
    assert not state.k_payload  # No persistent host shadow for device sessions.
    for item in restored[0].extents:
        assert item.start == state.base_offsets[item.layer, item.head]
        assert item.start > 0
        assert item.length == 9
    np.testing.assert_array_equal(configured[0][1]["capacity"], state.range_capacity[0])
    np.testing.assert_array_equal(configured[0][1]["live"], snapshot.live_counts[0])
    source.close()
    target.close()


def test_incompatible_or_undersized_restore_does_not_mutate_destination():
    source = _backend("bf16")
    _reserve(source, 0, 12)
    values = np.ones((9, 1, 2, 16), np.float32)
    source.streaming_pack(0, values, values, np.zeros((9, 1, 2), bool))
    snapshot = capture_prefix(source, 0)
    target = _backend("bf16")
    _reserve(target, 1, 2)
    with pytest.raises(ValueError, match="capacity"):
        restore_prefix(target, 1, snapshot)
    with pytest.raises(ValueError, match="layout"):
        restore_prefix(target, 1, replace(snapshot, layout="other"))
    state = target.state_for_request(1)
    assert state.logical_tokens == 0
    assert not state.k_payload
    assert not np.any(state.live_counts)
    target.assert_conserved()
    source.close()
    target.close()
