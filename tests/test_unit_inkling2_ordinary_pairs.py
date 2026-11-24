"""CPU tests for scripts/inkling2_ordinary_pairs.py (no sockets, no GPU)."""

import importlib.util
import io
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "inkling2_ordinary_pairs.py"
spec = importlib.util.spec_from_file_location("inkling2_ordinary_pairs", MODULE_PATH)
pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairs)


def sse(*events):
    lines = []
    for event in events:
        lines.append("data: " + json.dumps(event))
    lines.append("data: [DONE]")
    return lines


def delta(text, field="content", finish=None):
    return {"choices": [{"index": 0, "delta": {field: text}, "finish_reason": finish}]}


def test_iter_sse_events_parses_and_stops_at_done():
    events = list(pairs.iter_sse_events(iter(sse(delta("he"), delta("llo", finish="stop")))))
    assert len(events) == 2
    with pytest.raises((json.JSONDecodeError, ValueError)):
        list(pairs.iter_sse_events(iter(["data: {not json"])))


def test_event_deltas_reads_content_and_reasoning():
    assert pairs.event_deltas(delta("abc")) == ("abc", "")
    assert pairs.event_deltas(delta("think", field="reasoning_content")) == ("", "think")
    assert pairs.event_deltas({"choices": []}) == ("", "")
    assert pairs.event_deltas({"choices": [{"index": 1, "delta": {"content": "x"}}]}) == ("", "")


def test_carries_text_ignores_whitespace_only_deltas():
    assert not pairs.carries_text("")
    assert not pairs.carries_text("  \n\t ")
    assert pairs.carries_text(" a")


def test_summarize_events_collects_text_finish_and_usage():
    events = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "let me "}}]},
        delta("hello", finish="stop"),
        {"choices": [{"index": 0, "delta": {}}], "usage": {"prompt_tokens": 47, "completion_tokens": 2}},
    ]
    out = pairs.summarize_events(events, start=100.0, first_text_at=101.5, end=102.0)
    assert out["content"] == "hello"
    assert out["reasoning_content"] == "let me "
    assert out["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] == 47
    assert out["first_text_seconds"] == pytest.approx(1.5)
    assert out["total_seconds"] == pytest.approx(2.0)


def test_median_handles_odd_and_even():
    assert pairs.median([3, 1, 2]) == 2
    assert pairs.median([1, 2, 3, 4]) == 2.5
    assert pairs.median([]) is None


def test_summarize_records_groups_and_medians():
    records = [
        {"prompt_id": "p", "engine": "a", "error": None, "first_text_seconds": 1.0, "total_seconds": 5.0,
         "usage": {"prompt_tokens": 47}},
        {"prompt_id": "p", "engine": "a", "error": None, "first_text_seconds": 3.0, "total_seconds": 7.0,
         "usage": {"prompt_tokens": 47}},
        {"prompt_id": "p", "engine": "a", "error": "HTTPError: 500", "first_text_seconds": None,
         "total_seconds": None, "usage": None},
        {"prompt_id": "p", "engine": "b", "error": None, "first_text_seconds": 9.0, "total_seconds": 20.0,
         "usage": None},
    ]
    rows = {row["engine"]: row for row in pairs.summarize_records(records)}
    assert rows["a"]["failed"] == 1 and rows["a"]["requests"] == 3
    assert rows["a"]["median_first_text_seconds"] == 2.0
    assert rows["a"]["median_total_seconds"] == 6.0
    assert rows["a"]["prompt_tokens"] == 47
    assert rows["b"]["median_first_text_seconds"] == 9.0
    assert rows["b"]["prompt_tokens"] is None


def test_run_pairs_alternates_order_and_isolates_failures():
    servers = [{"name": "a", "url": "http://a", "api_key": None},
               {"name": "b", "url": "http://b", "api_key": "k"}]
    prompts = [{"id": "p1", "messages": [{"role": "user", "content": "hi"}], "model": "m"}]
    calls = []

    def fake_opener(url, api_key, payload, timeout, now):
        calls.append((url, api_key, payload["model"]))
        if url == "http://b":
            raise RuntimeError("boom")
        return {"first_text_seconds": 1.0, "total_seconds": 2.0, "content": "ok",
                "reasoning_content": "", "finish_reason": "stop", "usage": {"prompt_tokens": 3},
                "hipengine": {"execution_path": "inkling2_gguf_packed_hip"}, "error": None}

    records = pairs.run_pairs(servers, prompts, reps=2, warmup=0, max_tokens=8,
                              temperature=0.0, timeout=5, opener=fake_opener)
    assert [r["engine"] for r in records] == ["a", "b", "b", "a"]  # order alternates per rep
    assert [r["order"] for r in records] == ["first", "second", "first", "second"]
    assert all(r["error"] is None for r in records if r["engine"] == "a")
    assert all("boom" in r["error"] for r in records if r["engine"] == "b")
    assert all(r["usage"]["prompt_tokens"] == 3 for r in records if r["engine"] == "a")
    assert calls[0][2] == "m"


def test_run_pairs_requires_two_servers():
    with pytest.raises(ValueError):
        pairs.run_pairs([{"name": "a", "url": "http://a"}], [], reps=1, warmup=0,
                        max_tokens=1, temperature=0.0, timeout=1)


def test_load_prompts_accepts_prompt_or_messages(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"id": "a", "prompt": "hello"}\n'
        '{"id": "b", "messages": [{"role": "user", "content": "hi"}], "model": "m2"}\n',
        encoding="utf-8",
    )
    prompts = pairs.load_prompts(path)
    assert prompts[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert prompts[1]["model"] == "m2"
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "no id"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        pairs.load_prompts(bad)


def test_stream_once_times_first_text_with_injected_clock():
    body = "".join([
        'data: {"choices": [{"index": 0, "delta": {"content": ""}}]}\n\n',
        'data: {"choices": [{"index": 0, "delta": {"content": "he"}}]}\n\n',
        'data: {"choices": [{"index": 0, "delta": {"content": "llo"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3}}\n\n',
        "data: [DONE]\n\n",
    ])

    class FakeResponse:
        def __init__(self, payload):
            self._lines = io.StringIO(payload).readlines()

        def __enter__(self):
            return iter(self._lines)

        def __exit__(self, *exc):
            return False

    ticks = iter([100.0, 101.5, 102.0])
    monkey_now = lambda: next(ticks)

    import urllib.request
    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = lambda request, timeout=None: FakeResponse(body)
    try:
        result = pairs.stream_once("http://x", None, {"model": "m", "messages": []}, 5.0, now=monkey_now)
    finally:
        urllib.request.urlopen = real_urlopen

    # t=100 start; first text delta arrives at the second clock read (101.5); close at 102.
    assert result["content"] == "hello"
    assert result["first_text_seconds"] == pytest.approx(1.5)
    assert result["total_seconds"] == pytest.approx(2.0)
    assert result["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 3
    assert result["error"] is None
