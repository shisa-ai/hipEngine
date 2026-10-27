"""Streaming prompt admission follows adapter capability, not a retired flag."""

from types import SimpleNamespace

from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner


def test_streaming_prompt_ignores_retired_disable(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT", "0")
    sink = object()
    calls = []
    adapter = SimpleNamespace(begin_prompt_streaming=lambda ids, **kw: calls.append(ids) or (sink,))
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._resolved_mtp2_adapter = lambda: adapter
    runner._restorable_prefix_rows = lambda rows: set()
    row = SimpleNamespace(request_id=1, request=object(), mtp2_candidate_budget=2,
                          prefix_reused_tokens=0, mtp2_prompt_fallback_reason=None)
    assert runner._begin_mtp2_prompt_streaming((row,)) == (sink,)
    assert calls == [(1,)]
    assert row.mtp2_candidate_budget == 2
    assert row.mtp2_prompt_fallback_reason is None

    row.prefix_reused_tokens = 4
    assert runner._begin_mtp2_prompt_streaming((row,)) == (None,)
    assert row.mtp2_prompt_fallback_reason == "prefix_reuse_k0"
    assert calls == [(1,)]
