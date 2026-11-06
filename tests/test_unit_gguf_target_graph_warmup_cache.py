"""Host-only target graph warmup cache ownership."""
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_native_spec_cycle as cycle
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def test_eviction_close_failure_keeps_graph_reachable_for_retry(monkeypatch):
    monkeypatch.setattr(cycle, '_NATIVE_TARGET_GRAPH_CACHE_MAX_ENTRIES', 1)
    session = SimpleNamespace()
    events = []
    failing = True
    def close():
        if failing:
            raise RuntimeError('injected close failure')
        events.append('closed')
    first = SimpleNamespace(close=close)
    second = SimpleNamespace(close=lambda: None)
    first_key = (1, True, 256, False)
    second_key = (2, True, 256, False)
    cycle._cache_native_target_graph(session, first_key, first)
    with pytest.raises(RuntimeError, match='injected close failure'):
        cycle._cache_native_target_graph(session, second_key, second)
    assert session._native_spec_target_graphs[first_key] is first
    assert session._native_spec_target_graphs[second_key] is second
    failing = False
    cycle._cache_native_target_graph(session, second_key, second)
    assert events == ['closed']
    assert session._native_spec_target_graphs == {second_key: second}


def test_same_key_replacement_closes_previous_graph_once():
    session = SimpleNamespace()
    closed = []
    first = SimpleNamespace(close=lambda: closed.append('first'))
    second = SimpleNamespace(close=lambda: closed.append('second'))
    key = (2, True, 256, False)
    cycle._cache_native_target_graph(session, key, first)
    cycle._cache_native_target_graph(session, key, first)
    assert closed == []
    cycle._cache_native_target_graph(session, key, second)
    assert closed == ['first']
    assert session._native_spec_target_graphs == {key: second}


def test_warmup_uses_bounded_cache_insertion(monkeypatch):
    monkeypatch.setattr(cycle, '_NATIVE_TARGET_GRAPH_CACHE_MAX_ENTRIES', 1)
    session = object.__new__(Qwen35GGUFResidentSession)
    session._capture_prefill_gdn_for_call = lambda value: False
    closed = []
    graphs = []
    def capture(tokens, **kwargs):
        graph = SimpleNamespace(context_limit=256, closed=False)
        graph.close = lambda: closed.append(len(tokens))
        graphs.append(graph)
        return graph
    session.capture_native_spec_target_graph = capture
    assert session.prepare_native_spec_target_graph((1, 2), request_id=1)
    assert session.prepare_native_spec_target_graph((1, 2, 3), request_id=1, sampled_accept=True)
    assert len(session._native_spec_target_graphs) == 1
    assert closed == [2]
    assert list(session._native_spec_target_graphs.values()) == [graphs[-1]]
