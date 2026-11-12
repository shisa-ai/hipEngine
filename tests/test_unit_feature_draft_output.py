"""Output boundaries and service cancellation for feature-drafter streams."""
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest

from tests.test_unit_qwen35_feature_provider_sessions import provider_case, req
from tests.test_unit_qwen35_feature_provider_streaming import _install_steps


def test_utf8_split_across_cycles_uses_request_decoder(provider_case, monkeypatch):
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer, bytes_to_unicode
    obj, calls, _ = provider_case
    alphabet = bytes_to_unicode()
    obj.target.tokenizer = Qwen35GGUFTokenizer(
        tokens=(alphabet[0xE7], alphabet[0x8C], alphabet[0xAB]), merges=(), token_types=(1, 1, 1),
    )
    _install_steps(monkeypatch, obj, calls, ids=(0, 1, 2))
    chunks = list(obj.stream_detailed(replace(req(), max_tokens=3)))
    assert [c.text for c in chunks] == ["", "", "猫"]
    assert chunks[-1].generated_token_ids == (0, 1, 2)


@pytest.mark.parametrize("boundary", ["zero", "one", "min_tokens", "unmatched_prefix", "ignored_eos"])
def test_output_boundaries(provider_case, monkeypatch, boundary):
    obj, calls, _ = provider_case
    _install_steps(monkeypatch, obj, calls)
    request = replace(req(), max_tokens=3)
    expected = "789"
    if boundary == "zero":
        request = replace(request, max_tokens=0)
        expected = ""
    elif boundary == "one":
        request = replace(request, max_tokens=1)
        expected = "7"
    elif boundary == "min_tokens":
        request = replace(request, min_tokens=3, stop_token_ids=(7, 8), eos_token_id=99)
    elif boundary == "unmatched_prefix":
        request = replace(request, stop_token_sequences=((8, 10),))
    else:
        obj.target.tokenizer.stop_token_ids = (8,)
        obj.target.tokenizer.eos_token_id = 8
    assert "".join(c.text for c in obj.stream_detailed(request)) == expected


def test_cycle_error_releases_session_and_draft(provider_case, monkeypatch):
    obj, calls, _ = provider_case
    def broken(*args, **kwargs):
        yield None
        raise RuntimeError("verify failed")
    monkeypatch.setattr(obj, "_cycle_steps", broken)
    with pytest.raises(RuntimeError, match="verify failed"):
        obj.generate_detailed(req())
    assert not obj._active_children
    assert any(c[0] == "release" for c in calls)
    assert any(c[0] == "draft_close" for c in calls)


def test_service_stream_publishes_before_next_cycle_and_cancels(provider_case, monkeypatch):
    from hipengine.generation import EngineService, SubmitPollTextGenerator
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    obj, calls, _ = provider_case
    first_published = threading.Event()
    next_cycle = threading.Event()
    def steps(*args, **kwargs):
        yield None
        yield dict(output_ids=[7], elapsed_s=1.0, draft_forwards=1,
                   mean_acceptance=0.5, graph_verify_cycles=1, finished=False)
        first_published.set()
        assert next_cycle.wait(timeout=5)
        yield dict(output_ids=[7, 8], elapsed_s=2.0, draft_forwards=2,
                   mean_acceptance=0.5, graph_verify_cycles=2, finished=False)
    monkeypatch.setattr(obj, "_cycle_steps", steps)
    class Inner:
        supports_speculative = True
        supports_speculative_mtp = False
        _speculative_provider = obj
        create_generic_speculative_child = Qwen35GGUFBringupGenerator.create_generic_speculative_child
        def generate_speculative_detailed(self, request):
            raise AssertionError("service must use bounded child")
        def close(self):
            obj.close()
    adapter = SubmitPollTextGenerator(Inner(), capacity=1)
    service = EngineService(adapter)
    stream = service.stream_speculative_detailed(replace(req(), max_tokens=3))
    try:
        chunk = next(stream)
        assert chunk.text == "7" and chunk.finish_details is None
        assert first_published.wait(timeout=5)
        closer = threading.Thread(target=stream.close)
        closer.start()
        next_cycle.set()
        closer.join(timeout=5)
        assert not closer.is_alive()
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
        assert not obj._active_children
        assert any(c[0] == "draft_close" for c in calls)
    finally:
        next_cycle.set()
        stream.close()
        service.close()


def test_service_stops_preserve_final_byte_flush(provider_case, monkeypatch):
    from hipengine.generation import EngineService, SubmitPollTextGenerator
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer, bytes_to_unicode
    obj, calls, _ = provider_case
    alphabet = bytes_to_unicode()
    obj.target.tokenizer = Qwen35GGUFTokenizer(
        tokens=(alphabet[0xE7], "<stop>"), merges=(), token_types=(1, 3), eos_token_id=1,
    )
    _install_steps(monkeypatch, obj, calls, ids=(0, 1))
    class Inner:
        supports_speculative = True
        supports_speculative_mtp = False
        _speculative_provider = obj
        create_generic_speculative_child = Qwen35GGUFBringupGenerator.create_generic_speculative_child
        def generate_speculative_detailed(self, request):
            raise AssertionError("service must use bounded child")
        def close(self):
            obj.close()
    service = EngineService(SubmitPollTextGenerator(Inner(), capacity=1))
    try:
        request = replace(req(), max_tokens=3, ignore_eos=False, eos_token_id=1)
        chunks = list(service.stream_speculative_detailed(request))
        output = service.generate_speculative_detailed(request)[0]
        assert "".join(c.text for c in chunks) == output.text == "\ufffd"
        assert chunks[-1].generated_token_ids == output.generated_token_ids == (0, 1)
        assert chunks[-1].finish_details.reason == "eos"
        assert sum(c.finish_details is not None for c in chunks) == 1
    finally:
        service.close()


def test_deadline_and_shutdown_retire_active_child(provider_case, monkeypatch):
    from hipengine.generation import GenerationDeadlineExceeded
    obj, calls, _ = provider_case
    _install_steps(monkeypatch, obj, calls)
    child = obj.create_child(replace(req(), max_tokens=3))
    child.advance()
    child.request = replace(child.request, deadline_at=0)
    with pytest.raises(GenerationDeadlineExceeded):
        child.advance()
    assert not obj._active_children
    pending = obj.create_child(req())
    obj.close()
    with pytest.raises(RuntimeError, match="closed"):
        pending.advance()
    assert not obj._active_children


def test_allocation_failure_releases_target_and_unregisters_child(provider_case, monkeypatch):
    obj, calls, _ = provider_case
    obj._native = None
    def fail(*args):
        raise MemoryError("draft allocation")
    monkeypatch.setattr(obj, "_make_native", fail)
    with pytest.raises(MemoryError, match="draft allocation"):
        obj.generate_detailed(req())
    assert not obj._active_children
    assert any(c[0] == "release" for c in calls)


def test_failed_session_release_does_not_cache_drafter(provider_case, monkeypatch):
    from contextlib import contextmanager
    obj, calls, _ = provider_case
    @contextmanager
    def broken_scope(**kwargs):
        yield SimpleNamespace(runtime=object()), False
        raise RuntimeError("session reset failed")
    monkeypatch.setattr(obj.target, "_resident_session_scope", broken_scope)
    def steps(*args):
        yield dict(output_ids=[7, 8], elapsed_s=1.0, draft_forwards=1,
                   graph_verify_cycles=1, mean_acceptance=0.5, finished=True)
    monkeypatch.setattr(obj, "_cycle_steps", steps)
    with pytest.raises(RuntimeError, match="session reset failed"):
        obj.generate_detailed(req())
    assert obj._native is None
    assert not obj._active_children
    assert any(c[0] == "draft_close" for c in calls)


def test_llm_closes_underlying_speculative_stream(monkeypatch):
    from hipengine import LLM, SamplingParams
    from hipengine.generation import GenerationStreamChunk
    closed = []
    class Stream:
        def __iter__(self):
            return self
        def __next__(self):
            return GenerationStreamChunk(text="A")
        def close(self):
            closed.append(True)
    stream = Stream()
    llm = object.__new__(LLM)
    monkeypatch.setattr(llm, "_get_text_generator", lambda: SimpleNamespace(
        supports_speculative=True, stream_speculative_detailed=lambda request: stream))
    public = llm.stream_speculative_detailed("hello", SamplingParams(max_tokens=3))
    next(public)
    public.close()
    assert closed == [True]
