"""Independent feature drafters borrow the target owner's session and policy."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import SpeculativeProviderConfig


@pytest.fixture(params=["eagle4", "dspine"])
def provider_case(request, monkeypatch):
    import importlib
    method = request.param
    module = importlib.import_module(f"hipengine.generation.qwen35_gguf_{method}")
    cls = getattr(module, "Qwen35EAGLE4TextProvider" if method == "eagle4" else "Qwen35DSpineTextProvider")
    calls = []
    runner = object()
    session = SimpleNamespace(runtime=object())

    @contextmanager
    def scope(**kwargs):
        calls.append(("scope", kwargs))
        try:
            yield session, len([c for c in calls if c[0] == "scope"]) > 1
        finally:
            calls.append(("release", session))

    target = SimpleNamespace(
        model_path="unused.gguf", backend="hip_gfx1151", target_arch=None,
        _prepared_max_sequence_length=256,
        _prepare_kv_policy=lambda req: calls.append(("kv", req)),
        _get_shared_runner=lambda: runner, _resident_session_scope=scope,
        tokenizer=SimpleNamespace(eos_token_id=99, stop_token_ids=(),
                                  decode=lambda ids, **kw: "".join(map(str, ids))),
        tokenize=lambda text: (1, 2),
    )
    obj = cls(target, SpeculativeProviderConfig(provider=method, draft_model="unused", candidate_budget=3))
    obj._loaded = True
    obj._config = {"block_size": 16}
    obj._weights = {}
    obj._nd = object()
    obj._token_embd = obj._head = object()
    obj._native = SimpleNamespace(max_context_len=256, close=lambda: calls.append(("draft_close",)))

    def cycle(*args, **kwargs):
        assert args[0] is session
        calls.append(("cycle", kwargs))
        yield dict(output_ids=[7, 8], elapsed_s=1.0, draft_forwards=1, finished=True,
                   mean_acceptance=0.5, graph_verify_cycles=1, finish_reason="length", eos_token_id=None)
    monkeypatch.setattr(f"hipengine.speculative.{method}_cycle.iter_{method}_cycle", cycle)

    def forbidden(*args, **kwargs):
        raise AssertionError("provider constructed an unowned target session")
    monkeypatch.setattr("hipengine.runtime.qwen35_gguf_runner.Qwen35GGUFResidentSession", forbidden)
    return obj, calls, runner


def req():
    return GenerationRequest(prompts=((1, 2),), max_tokens=2, temperature=0,
                             top_p=1, ignore_eos=True)


def test_provider_borrows_target_session_on_repeated_requests(provider_case):
    obj, calls, runner = provider_case
    for reused in (False, True):
        output = obj.generate_detailed(req())[0]
        assert output.generated_token_ids == (7, 8)
        assert output.telemetry.diagnostics["target_session_reused"] is reused
        assert output.telemetry.diagnostics["graph_verify_cycles"] == 1
    scopes = [value for name, value in calls if name == "scope"]
    assert all(value["shared_runner"] is runner for value in scopes)
    assert all(value["pool_name"] == f"{obj.provider_name}_target" for value in scopes)
    assert all(value["max_sequence_length"] == 256 for value in scopes)
    assert len([c for c in calls if c[0] == "release"]) == 2
    assert len([c for c in calls if c[0] == "kv"]) == 2


def test_closed_provider_cannot_allocate_again(provider_case):
    obj, calls, _ = provider_case
    obj.close()
    with pytest.raises(RuntimeError, match="closed"):
        obj.generate_detailed(req())
    assert not any(c[0] == "scope" for c in calls)


def test_expired_request_does_not_open_a_session(provider_case):
    from dataclasses import replace
    from hipengine.generation.deadline import GenerationDeadlineExceeded
    obj, calls, _ = provider_case
    with pytest.raises(GenerationDeadlineExceeded):
        obj.generate_detailed(replace(req(), deadline_at=0))
    assert not calls


@pytest.mark.parametrize("failure", [GeneratorExit, RuntimeError, KeyboardInterrupt])
def test_target_session_scope_closes_abandoned_leases(failure):
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    owner = object.__new__(Qwen35GGUFBringupGenerator)
    events = []
    session = SimpleNamespace(close=lambda: events.append("closed"))
    owner._acquire_shared_session = lambda *a, **kw: (session, "key", True)
    owner._configure_session = lambda s: None
    owner._release_shared_session = lambda *a: events.append("pooled")
    with pytest.raises(failure):
        with owner._resident_session_scope(shared_runner=object(), pool_name="test"):
            raise failure()
    assert events == ["closed"]


def test_target_session_scope_passes_explicit_capacity_and_releases():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    owner = object.__new__(Qwen35GGUFBringupGenerator)
    events = []
    session = SimpleNamespace(close=lambda: events.append("closed"))
    def acquire(*args, **kwargs):
        events.append(kwargs["max_sequence_length"])
        return session, "key", True
    owner._acquire_shared_session = acquire
    owner._configure_session = lambda s: None
    owner._release_shared_session = lambda *a: events.append("pooled")
    with owner._resident_session_scope(shared_runner=object(), pool_name="test",
                                       max_sequence_length=256):
        pass
    assert events == [256, "pooled"]


def test_generator_closes_attached_provider_before_shared_weights():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    owner = object.__new__(Qwen35GGUFBringupGenerator)
    owner.backend = "hip_gfx1151"
    events = []
    owner._speculative_provider = SimpleNamespace(close=lambda: events.append("provider"))
    owner._shared_runner = SimpleNamespace(close=lambda: events.append("runner"))
    owner.close()
    owner.close()
    assert events == ["provider", "runner"]
    assert owner._speculative_provider is None
