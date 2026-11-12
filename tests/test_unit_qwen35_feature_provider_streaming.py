"""Feature-drafter requests publish committed tokens before terminal output."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.test_unit_qwen35_feature_provider_sessions import provider_case, req


def _install_steps(monkeypatch, obj, calls, ids=(7, 8, 9)):
    def steps(*args, **kwargs):
        calls.append(("prefill", args[1]))
        yield None
        for i in range(len(ids)):
            calls.append(("step", args[1]))
            yield dict(output_ids=list(ids[:i + 1]), elapsed_s=1.0, draft_forwards=i + 1,
                       mean_acceptance=0.5, graph_verify_cycles=i + 1,
                       finish_reason="length", eos_token_id=None, finished=i == len(ids) - 1)
    monkeypatch.setattr(f"hipengine.speculative.{obj.provider_name}_cycle.iter_{obj.provider_name}_cycle", steps,
                        raising=False)


def test_stream_yields_before_later_cycles_and_closes_on_abandon(provider_case, monkeypatch):
    obj, calls, _ = provider_case
    _install_steps(monkeypatch, obj, calls)
    stream = obj.stream_detailed(replace(req(), max_tokens=3))
    assert next(stream).text == "7"
    assert len([c for c in calls if c[0] == "step"]) == 1
    stream.close()
    assert any(c[0] == "release" for c in calls)
    assert any(c[0] == "draft_close" for c in calls)


def test_child_admission_is_lazy_and_advance_is_one_work_unit(provider_case, monkeypatch):
    obj, calls, _ = provider_case
    _install_steps(monkeypatch, obj, calls)
    child = obj.create_child(replace(req(), max_tokens=3))
    assert calls == []
    assert child.advance() is None
    assert [c[0] for c in calls].count("prefill") == 1
    assert not child.drain_token_chunks()
    assert child.advance() is None
    assert [token for token, _ in child.drain_token_chunks()] == [7]
    child.close(reason="disconnect")
    assert not obj._active_children


@pytest.mark.parametrize("terminal", ["stop", "eos", "sequence", "length"])
def test_stream_and_blocking_share_stop_safe_text(provider_case, monkeypatch, terminal):
    obj, calls, _ = provider_case
    _install_steps(monkeypatch, obj, calls)
    request = replace(req(), max_tokens=3, ignore_eos=False)
    if terminal == "stop":
        request = replace(request, stop_token_ids=(8,))
    elif terminal == "eos":
        request = replace(request, eos_token_id=8)
    elif terminal == "sequence":
        request = replace(request, stop_token_sequences=((8, 9),))
    chunks = list(obj.stream_detailed(request))
    output = obj.generate_detailed(request)[0]
    assert "".join(c.text for c in chunks) == output.text == ("789" if terminal == "length" else "7")
    assert chunks[-1].generated_token_ids == output.generated_token_ids
    assert sum(c.finish_details is not None for c in chunks) == 1


def test_cancel_between_cycles_closes_request_and_allows_reuse(provider_case, monkeypatch):
    from hipengine.generation import GenerationCancellationToken, GenerationCancelled
    obj, calls, _ = provider_case
    _install_steps(monkeypatch, obj, calls)
    token = GenerationCancellationToken()
    stream = obj.stream_detailed(replace(req(), max_tokens=3, cancellation_token=token))
    next(stream)
    token.cancel()
    with pytest.raises(GenerationCancelled):
        next(stream)
    assert not obj._active_children
    assert any(c[0] == "draft_close" for c in calls)


def test_interleaved_children_finish_without_losing_their_history(provider_case, monkeypatch):
    obj, calls, _ = provider_case
    def make_native(session, max_seq):
        return SimpleNamespace(max_context_len=max_seq, close=lambda: calls.append(("extra_close",)))
    monkeypatch.setattr(obj, "_make_native", make_native)
    def steps(session, native, prompt_ids, request):
        native.value = prompt_ids[0]
        yield None
        for i in range(3):
            yield dict(output_ids=[native.value] * (i + 1), elapsed_s=1.0,
                       draft_forwards=i + 1, graph_verify_cycles=i + 1,
                       mean_acceptance=0.5, finished=i == 2)
    monkeypatch.setattr(obj, "_cycle_steps", steps)
    children = [obj.create_child(replace(req(), prompts=((token,),), max_tokens=3)) for token in (7, 8)]
    try:
        for _ in range(4):
            for child in children:
                child.advance()
                child.drain_token_chunks()
        assert [child.output.generated_token_ids for child in children] == [(7, 7, 7), (8, 8, 8)]
    finally:
        for child in children:
            child.close(reason="completed")
    assert not obj._active_children
    assert len([c for c in calls if c[0] in {"extra_close", "draft_close"}]) == 1


def test_interleaved_children_never_share_mutable_drafter(provider_case, monkeypatch):
    obj, calls, _ = provider_case
    _install_steps(monkeypatch, obj, calls)
    monkeypatch.setattr(obj, "_make_native", lambda session, max_seq: SimpleNamespace(
        max_context_len=max_seq, close=lambda: calls.append(("extra_close",))))
    first = obj.create_child(replace(req(), max_tokens=3))
    second = obj.create_child(replace(req(), max_tokens=3))
    try:
        first.advance()
        second.advance()
        assert first.native is not second.native
    finally:
        first.close(reason="disconnect")
        second.close(reason="disconnect")
    assert not obj._active_children
