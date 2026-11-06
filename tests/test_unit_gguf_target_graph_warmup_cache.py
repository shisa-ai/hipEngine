"""Host-only target graph warmup cache ownership."""
from types import SimpleNamespace

from hipengine.runtime import gguf_native_spec_cycle as cycle
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


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
