"""Exact-budget prefill packs each completed layer in one trunk execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import dms_prefill_cache
from tests.test_unit_dms_layer_outer_lifetime import (
    _layer_outer_dms_session,
    _record_ownership_calls,
)


@pytest.mark.parametrize("tokens", [17, 127, 8192, 8193, 8194, 10003])
@pytest.mark.parametrize("ratio", [1, 2, 3])
def test_exact_budget_reserves_same_capacity_and_packs_in_one_trunk(monkeypatch, tokens, ratio):
    from hipengine.kvcache.dms import build_dms_exact_budget_eviction, build_dms_live_mask

    session, collector, backend, events, _ = _layer_outer_dms_session(monkeypatch, capacity=16384)
    config = session._dms_source.config
    config.prefill_selection_mode = "exact_budget"
    config.target_compression_ratio = ratio
    config.window_size = 8192
    logits = np.random.default_rng(tokens).normal(size=(tokens, 1, 1)).astype(np.float32)
    decisions = build_dms_exact_budget_eviction(
        logits,
        current_position=tokens - 1,
        window_size=config.window_size,
        target_compression_ratio=ratio,
    )
    finalized = []
    constructions = []
    collector.finalize_layer = (
        lambda layer, **kw: finalized.append(layer)
        or events.append("layer_finalize")
        or decisions[:, layer, :]
    )
    collector.finalize = lambda **kw: events.append("all_finalize") or decisions
    backend.device_streaming_pack_layer = lambda *a, **kw: events.append("pack")
    backend.stage_device_prefill_layer = lambda layer: events.append("stage")
    original_finalize = backend.finalize_device_streaming_pack

    def final_pack(*a, **kw):
        events.append("restore_final")
        assert session._bulk_prefill_scratch is None
        assert session._dms_prefill_oracle_pair is None
        np.testing.assert_array_equal(kw["eviction"], decisions)
        return original_finalize(*a, **kw)

    backend.finalize_device_streaming_pack = final_pack
    session.dms_backend_factory = lambda **kw: constructions.append(kw) or backend
    _record_ownership_calls(monkeypatch, session, events)
    sink = object()

    def bulk(token_ids, **kw):
        events.append("trunk")
        assert kw.get("dms_capture") is collector
        assert kw.get("dms_pack_sink") is session
        assert kw.get("target_hidden_chunk_sink") is sink
        assert kw.get("target_hidden_request_id") == 81
        session.pack_dms_prefill_layer(0, stream=0)
        session._last_target_hidden_ptr = (
            session._prefill_hidden_a.ptr + (tokens - 1) * session.runner.hidden_size * 2
        )
        return SimpleNamespace(logits=np.arange(4, dtype=np.float32))

    session._run_bulk_prefill_and_sample = bulk
    result = session._prefill_layer_outer_dms(
        collector,
        list(range(tokens)),
        bulk_attention_mode="bulk",
        return_logits=True,
        record_gpu_stage_timings=False,
        target_hidden_chunk_sink=sink,
        target_hidden_request_id=81,
    )
    assert events == [
        "trunk",
        "layer_finalize",
        "pack",
        "stage",
        "all_finalize",
        "release_inputs",
        "retain",
        "retain_copy",
        "release_workspace",
        "restore_final",
        "close",
    ]
    assert finalized == [0]
    live = build_dms_live_mask(
        decisions[:, 0, 0][None, :], current_position=tokens - 1, window_size=8192
    ).sum()
    assert constructions[0]["slots_per_layer"] == live + session.dms_max_new_tokens
    assert session._dms_backend is backend and not backend.closed
    np.testing.assert_array_equal(result.logits, np.arange(4, dtype=np.float32))


@pytest.mark.parametrize("failure", ["pack", "global_finalize", "reserve"])
def test_one_pass_failure_closes_unpublished_store_and_pack_state(monkeypatch, failure):
    session, collector, backend, events, _ = _layer_outer_dms_session(monkeypatch)
    session._dms_source.config.target_compression_ratio = 2
    session._dms_source.config.prefill_selection_mode = "exact_budget"
    collector.finalize_layer = lambda layer, **kw: np.zeros((4, 1), dtype=bool)

    def fail(*a, **kw):
        raise RuntimeError("one-pass failure")

    backend.device_streaming_pack_layer = fail if failure == "pack" else lambda *a, **kw: None
    backend.stage_device_prefill_layer = lambda layer: None
    if failure == "global_finalize":
        collector.finalize = fail
    if failure == "reserve":
        backend.reserve = fail

    def bulk(token_ids, **kw):
        assert kw.get("dms_capture") is collector and kw.get("dms_pack_sink") is session
        session.pack_dms_prefill_layer(0, stream=0)

    session._run_bulk_prefill_and_sample = bulk
    with pytest.raises(RuntimeError, match="one-pass failure"):
        session._prefill_layer_outer_dms(
            collector,
            [1, 2, 3, 4],
            bulk_attention_mode="bulk",
            return_logits=False,
            record_gpu_stage_timings=False,
        )
    assert backend.closed and session._dms_backend is None
    assert session._dms_layer_outer_pack is None
    assert session._dms_prefill_oracle_pair is None
    assert collector._closed


def test_one_pass_cold_capture_precedes_score_release(monkeypatch):
    session, collector, backend, events, _ = _layer_outer_dms_session(monkeypatch)
    session._dms_source.config.target_compression_ratio = 2
    session._dms_source.config.prefill_selection_mode = "exact_budget"
    session._dms_cache_capture_tokens = (1, 2, 3, 4)
    collector.finalize_layer = lambda layer, **kw: events.append("layer_finalize") or np.zeros(
        (4, 1), bool
    )
    backend.device_streaming_pack_layer = lambda *a, **kw: events.append("pack")
    backend.stage_device_prefill_layer = lambda layer: events.append("stage")

    def capture(owner, captured, tokens):
        assert captured._logits is not None
        assert len(owner._dms_cache_layers[0]) == 2
        events.append("cold_capture")
        return "cold"

    monkeypatch.setattr(dms_prefill_cache, "capture_cold_prefill", capture)
    monkeypatch.setattr(
        dms_prefill_cache, "read_prefix", lambda buf, nbytes: np.zeros(nbytes, np.uint8)
    )

    def bulk(token_ids, **kw):
        assert kw.get("dms_capture") is collector
        session.pack_dms_prefill_layer(0, stream=0)

    session._run_bulk_prefill_and_sample = bulk
    session._prefill_layer_outer_dms(
        collector,
        [1, 2, 3, 4],
        bulk_attention_mode="bulk",
        return_logits=False,
        record_gpu_stage_timings=False,
    )
    assert events.index("cold_capture") < events.index("release_inputs")
    assert session._dms_cold_prefill == "cold"
    assert session._dms_cache_layers == {}
