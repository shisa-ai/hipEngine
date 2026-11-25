"""CPU tests for the Inkling server check's readiness and request parsing (no server, no GPU)."""

import pytest

from scripts import inkling2_serve_check as check


def test_readiness_accepts_the_ready_payload_and_rejects_loading_or_error_states(monkeypatch):
    payloads = iter([{"status": "starting", "ready": False}, {"status": "error", "ready": False}, {"status": "ready", "ready": True}])
    sleeps = []
    monkeypatch.setattr(check, "_request", lambda *a, **k: next(payloads))
    monkeypatch.setattr(check.time, "sleep", sleeps.append)
    monkeypatch.setattr(check.time, "perf_counter", iter([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0]).__next__)
    check.wait_until_ready("http://x", check.Path("unused.log"), seconds=60)
    assert sleeps == [0.5, 0.5]


def test_readiness_gives_up_with_the_log_path(monkeypatch):
    def refuse(*args, **kwargs):
        raise check.urllib.error.URLError("connection refused")

    monkeypatch.setattr(check, "_request", refuse)
    monkeypatch.setattr(check.time, "sleep", lambda seconds: None)
    ticks = iter([0.0, 0.0, 0.0, 61.0, 0.0, 61.0])
    monkeypatch.setattr(check.time, "perf_counter", ticks.__next__)
    with pytest.raises(RuntimeError, match="did not become ready in 60 s; its log is server.log"):
        check.wait_until_ready("http://x", check.Path("server.log"), seconds=60)


def test_completion_times_the_round_trip_and_reads_usage(monkeypatch):
    seen = []

    def reply(base, method, path, body=None, timeout=900.0):
        seen.append((base, method, path, body))
        return {"usage": {"completion_tokens": 5, "prompt_tokens": 37}, "choices": [{"message": {"content": " Paris."}}]}

    monkeypatch.setattr(check, "_request", reply)
    monkeypatch.setattr(check.time, "perf_counter", iter([10.0, 10.25]).__next__)
    row = check.completion("http://x", "inkling", "The capital of France is", 5)
    assert row == {"seconds": 0.25, "completion_tokens": 5, "prompt_tokens": 37, "text": " Paris."}
    base, method, path, body = seen[0]
    assert (base, method, path) == ("http://x", "POST", "/v1/chat/completions")
    assert body == {"model": "inkling", "messages": [{"role": "user", "content": "The capital of France is"}],
                    "max_tokens": 5, "temperature": 0, "stream": False}
