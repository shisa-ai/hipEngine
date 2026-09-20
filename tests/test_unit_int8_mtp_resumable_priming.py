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
