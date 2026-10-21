"""Resumable INT8 prefill must publish complete hidden rows for MTP priming."""

from types import SimpleNamespace

from hipengine.runtime.qwen35_gguf_runner import _GGUFResumablePrefillState
from tests.test_unit_gguf_resumable_layer_outer_prefill import (
    _Recorder, _WiringRow, _prompt_rounds, _resumable_owner, _wiring_host,
)


def test_resumable_int8_prefill_primes_only_complete_final_hidden_rows(monkeypatch):
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prompt = tuple(range(32))
    events = []
    sink = SimpleNamespace(
        request_id=17,
        hidden_size=owner.runner.hidden_size,
        total_rows=len(prompt),
        consume=lambda **kwargs: events.append(("consume", kwargs)),
        finish=lambda **kwargs: events.append(("finish", kwargs)),
    )
    state = owner._prefill_batch_native_layer_outer(
        (prompt,), sessions=(owner,), chunks=_prompt_rounds(prompt, rows=8),
        layer_budget=3, target_hidden_chunk_sinks=(sink,),
    )
    assert isinstance(state, _GGUFResumablePrefillState)
    assert events == []
    owner._prefill_batch_native_layer_outer(
        None, sessions=None, chunks=None, resume_state=state, layer_budget=None,
    )
    consumes = [payload for kind, payload in events if kind == "consume"]
    assert [(value["chunk_start"], value["chunk_start"] + value["rows"]) for value in consumes] == [
        (0, 8), (8, 16), (16, 24), (24, 32),
    ]
    assert all(value["request_id"] == 17 for value in consumes)
    assert events[-1][0] == "finish"
    assert events[-1][1]["total_rows"] == 32
    assert len([kind for kind, _ in events if kind == "finish"]) == 1


def test_scheduler_carries_provider_sink_across_resumable_segments(monkeypatch):
    from hipengine.generation import qwen35_gguf as generation

    monkeypatch.setattr(generation, "_gguf_packed_layer_outer_enabled", lambda: True)
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    owner._invalidate_live_packed_decode_graphs = lambda: None
    host = _wiring_host(owner)
    row = _WiringRow(tuple(range(32)))
    row.lease.session = owner
    row.mtp2_candidate_budget = 3
    events = []
    sink = SimpleNamespace(
        request_id=row.request_id, hidden_size=owner.runner.hidden_size,
        total_rows=32, consume=lambda **kwargs: events.append("consume"),
        finish=lambda **kwargs: events.append("sink_finished"),
    )
    host._begin_mtp2_prompt_streaming = lambda rows: events.append("begin") or (sink,)
    host._finish_mtp2_prompt_streaming = lambda rows, sinks, success: events.append(("release", success))
    helper = generation.Qwen35GGUFResidentModelRunner._prefill_resumable_int8_chunk
    for start in range(0, 32, 8):
        row.prefill_tokens_seen = start + 8
        assert helper(host, row, row.prompt_ids[start:start + 8], final_chunk=start == 24)
        if start < 24:
            assert events == ["begin"]
    assert events.count("begin") == 1
    assert events[-2:] == ["sink_finished", ("release", True)]
    assert row.first_token_emitted


def _stream_sink(owner, *, request_id, total_rows, events):
    return SimpleNamespace(
        request_id=request_id,
        hidden_size=owner.runner.hidden_size,
        total_rows=total_rows,
        consume=lambda **kwargs: events.append(("consume", kwargs)),
        finish=lambda **kwargs: events.append(("finish", kwargs)),
    )


def test_layer_outer_prefix_call_feeds_sink_without_finishing(monkeypatch):
    """A boundary-split prefix streams its rows and leaves the sink open for the tail."""

    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prefix = tuple(range(32))
    events = []
    sink = _stream_sink(owner, request_id=5, total_rows=40, events=events)
    owner._prefill_batch_native_layer_outer(
        (prefix,), sessions=(owner,), chunks=_prompt_rounds(prefix, rows=8),
        target_hidden_chunk_sinks=(sink,), finish_target_hidden_sinks=False,
    )
    consumes = [payload for kind, payload in events if kind == "consume"]
    assert [(value["chunk_start"], value["rows"]) for value in consumes] == [
        (0, 8), (8, 8), (16, 8), (24, 8),
    ]
    assert not any(kind == "finish" for kind, _ in events)


def test_layer_outer_finishing_call_requires_the_whole_sink_stream(monkeypatch):
    import pytest

    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prompt = tuple(range(32))
    sink = _stream_sink(owner, request_id=5, total_rows=40, events=[])
    with pytest.raises(ValueError, match="complete prompt"):
        owner._prefill_batch_native_layer_outer(
            (prompt,), sessions=(owner,), chunks=_prompt_rounds(prompt, rows=8),
            target_hidden_chunk_sinks=(sink,),
        )


def test_streamed_multi_round_prompt_takes_the_layer_outer_executor(monkeypatch):
    """Streaming MTP rows must not force the per-layer-oracle chunk-outer route."""

    from hipengine.runtime import qwen35_gguf_runner as gguf_runner

    monkeypatch.setattr(gguf_runner, "_gguf_packed_layer_outer_enabled", lambda: True)
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prefix = tuple(range(32))
    calls = []

    def layer_outer(prompts, **kwargs):
        calls.append(kwargs)
        return [None]

    owner._prefill_batch_native_layer_outer = layer_outer
    events = []
    sink = _stream_sink(owner, request_id=5, total_rows=40, events=events)
    owner._prefill_batch_native_impl(
        [prefix], sessions=[owner], full_prompt_lengths=[40],
        sample_output=False, target_hidden_chunk_sinks=(sink,),
        target_hidden_request_ids=(5,), target_hidden_chunk_starts=(0,),
        finish_target_hidden_sinks=False,
    )
    assert len(calls) == 1
    assert calls[0]["target_hidden_chunk_sinks"] == (sink,)
    assert calls[0]["finish_target_hidden_sinks"] is False
