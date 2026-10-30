"""SSE timing measures token delivery, never terminal-buffer flushing."""
import json

from hipengine.generation.registry import FinishDetails, GenerationStreamChunk
from hipengine.server import api
from hipengine.server.chat_output import _ParsedChatOutput


def clock(monkeypatch, values):
    times = iter(values)
    monkeypatch.setattr(api.time, "perf_counter", lambda: next(times))


def test_single_buffered_chunk_has_no_decode_rate(monkeypatch):
    clock(monkeypatch, [10.0, 10.0001])
    tracker = api._StreamTimingTracker(started_at=0)
    tracker.observe(event="delta", token_event=True, tokens={"streamed_tokens": 256})
    done = tracker.observe(event="done", tokens={"streamed_tokens": 256})
    assert done["ttft_ms"] == 10000
    assert "decode_tokens_per_second" not in done


def test_rate_excludes_first_chunk_and_terminal_flush(monkeypatch):
    clock(monkeypatch, [2.0, 3.0, 9.0, 12.0])
    tracker = api._StreamTimingTracker(started_at=0)
    tracker.observe(event="delta", token_event=True, tokens={"streamed_tokens": 4})
    tracker.observe(event="delta", token_event=True, tokens={"streamed_tokens": 7})
    done = tracker.observe(event="done", tokens={"streamed_tokens": 7})
    usage = tracker.observe(event="usage", usage={"completion_tokens": 7})
    for result in (done, usage):
        assert result["decode_elapsed_ms"] == 1000
        assert result["decode_tokens_per_second"] == 3


def test_missing_first_chunk_count_cannot_define_decode_denominator(monkeypatch):
    clock(monkeypatch, [1.0, 2.0, 3.0])
    tracker = api._StreamTimingTracker(started_at=0)
    tracker.observe(event="delta", token_event=True)
    tracker.observe(event="delta", token_event=True, tokens={"streamed_tokens": 3})
    final = tracker.observe(event="usage", usage={"completion_tokens": 3})
    assert "decode_tokens_per_second" not in final


def test_complete_backend_chunk_split_into_channels_is_still_buffered(monkeypatch):
    clock(monkeypatch, [1.0, 1.0001, 1.0002])
    tracker = api._StreamTimingTracker(started_at=0)
    chunk = GenerationStreamChunk("reasoning and answer", finish_details=FinishDetails(reason="length"))
    for field, count in (("reasoning_content", 80), ("content", 100)):
        api._chat_stream_delta("id", 0, "model", field, field,
            tokens={"streamed_tokens": count}, stream_chunk=chunk,
            include_hipengine=True, stream_started_at=tracker)
    final = tracker.observe(event="done", tokens={"streamed_tokens": 100})
    assert "decode_tokens_per_second" not in final


def test_precomputed_chat_fragments_do_not_claim_model_decode_rate():
    tracker = api._StreamTimingTracker.start()
    accounting = api._StreamTokenAccounting(lambda text: len(text))
    events = list(api._chat_stream_parsed(
        "id", 0, "model", _ParsedChatOutput("<think>why</think>yes", ()), "stop",
        token_accounting=accounting, include_hipengine=True, stream_started_at=tracker))
    payloads = [json.loads(event.removeprefix("data: ").strip()) for event in events]
    done = payloads[-1]["hipengine"]["timing"]
    assert "ttft_ms" in done
    assert "decode_tokens_per_second" not in done
