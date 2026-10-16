"""The Gemma 4 assistant-head MTP serving route.

The assistant-head provider is a model-attached speculative provider, not a
model-plugin evidence row. What makes it reachable through the server is the
same three-piece contract the Qwen route uses:

* the model plugin declares the executable scope (``speculative_mtp_serving_
  implementations``) instead of listing known-good artifacts, so a newly
  resolvable Gemma 4 target that fits the declaration is admitted by capability;
* the generator resolves that declaration from mechanical request and runtime
  identity (``resolve_speculative_mtp_serving_plan``), so the server's
  automatic route, its sampled route, and the startup capability banner all read
  the same answer;
* the generator and the engine loop expose a streaming entry point, so a
  streamed ``speculative_mtp`` request reaches the provider rather than failing
  with a missing-method error.

These tests read the shipped declaration and exercise the shipped server route
helpers, so a declaration no runtime can execute, or a route that never reaches
the provider, does not pass.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine import LLM, SamplingParams
from hipengine.generation import GenerationOutput, GenerationStreamChunk
from hipengine.generation.engine_loop import SubmitPollTextGenerator
from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator
from hipengine.models.gemma4 import Gemma4GGUFModel
from hipengine.server.api import (
    _SPECULATIVE_MTP_AUTO_ROUTE,
    _SPECULATIVE_MTP_BATCH_ROUTE,
    _SPECULATIVE_MTP_DEFAULT_ROUTE,
    _generation_route_for_request,
    _resolve_realized_generation_route,
    _speculative_mtp_route_for_request,
    OpenAIHTTPError,
)

from tests.test_unit_speculative_sampling_modes import (
    _params,
    _serving_engine,
    _server_config,
)


def _declared_modes() -> tuple[str, ...]:
    return tuple(
        mode
        for declaration in Gemma4GGUFModel().speculative_mtp_serving_implementations
        for mode in declaration.sampling_modes
    )


def _gemma_engine():
    """An engine advertising exactly the modes the Gemma plugin declares."""

    return _serving_engine(*_declared_modes())


def _route(*, explicit=True, stream=False, sampling=None, mode="auto", engine=None):
    return _speculative_mtp_route_for_request(
        _server_config(speculative_mtp_serving=mode),
        SimpleNamespace(speculative_mtp=explicit, stream=stream),
        engine=_gemma_engine() if engine is None else engine,
        sampling=_params() if sampling is None else sampling,
    )


# ------------------------------------------------------------- declaration


def test_the_plugin_declares_greedy_and_sampled_on_both_backends() -> None:
    declarations = Gemma4GGUFModel().speculative_mtp_serving_implementations
    assert declarations, "the assistant head must declare its executable scope"
    for declaration in declarations:
        assert declaration.kv_storage == "bf16"
        assert "greedy_fast" in declaration.sampling_modes
        assert "sampled" in declaration.sampling_modes
        assert declaration.automatic_eligible is True
        assert ("hip_gfx1100", "gfx1100") in declaration.backends
        assert ("hip_gfx1151", "gfx1151") in declaration.backends


# ------------------------------------------------------------- server route


def test_normal_request_reaches_mtp() -> None:
    assert _route(explicit=True) == _SPECULATIVE_MTP_BATCH_ROUTE


def test_streaming_request_reaches_the_same_mtp_route() -> None:
    """Streaming is not a separate route; it rides the resolved MTP route."""

    assert _route(explicit=True, stream=True) == _SPECULATIVE_MTP_BATCH_ROUTE
    assert (
        _route(explicit=True, stream=True, sampling=_params(temperature=0.7))
        == _SPECULATIVE_MTP_BATCH_ROUTE
    )


def test_sampled_request_reaches_mtp() -> None:
    assert _route(sampling=_params(temperature=0.7)) == _SPECULATIVE_MTP_BATCH_ROUTE
    assert (
        _route(sampling=_params(temperature=0.7, top_p=0.9, min_p=0.05))
        == _SPECULATIVE_MTP_BATCH_ROUTE
    )


def test_explicit_false_reaches_the_autoregressive_default() -> None:
    assert _route(explicit=False) == _SPECULATIVE_MTP_DEFAULT_ROUTE
    assert (
        _route(explicit=False, sampling=_params(temperature=0.7))
        == _SPECULATIVE_MTP_DEFAULT_ROUTE
    )
    assert _route(explicit=False, stream=True) == _SPECULATIVE_MTP_DEFAULT_ROUTE


def test_explicit_request_is_refused_by_name_when_serving_is_off() -> None:
    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(mode="off", explicit=True)
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "unsupported_parameter"


def test_explicit_request_is_refused_by_name_when_the_engine_cannot_run_mtp() -> None:
    incapable = SimpleNamespace(
        supports_speculative_mtp=False,
        speculative_mtp_sampling_modes=(),
    )
    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(explicit=True, engine=incapable)
    assert excinfo.value.status_code == 501
    assert excinfo.value.code == "unsupported_feature"


# --------------------------------------------------- generator plan resolver


def _generator(*, backend: str = "hip_gfx1151", plugin=None) -> Gemma4GGUFGenerator:
    """A generator carrying only the fields the serving resolver reads."""

    generator = Gemma4GGUFGenerator.__new__(Gemma4GGUFGenerator)
    generator.backend = backend
    generator.model_plugin = Gemma4GGUFModel() if plugin is None else plugin
    return generator


@pytest.mark.parametrize(
    "kwargs,admitted",
    [
        (dict(sampling_mode="greedy_fast", kv_storage="bf16"), True),
        (dict(sampling_mode="sampled", kv_storage="auto"), True),
        (dict(sampling_mode="greedy_fast", kv_storage="int8_per_token_head"), False),
        (dict(sampling_mode="processed_argmax", kv_storage="bf16"), False),
        (dict(sampling_mode="greedy_fast", kv_storage="bf16", realized_group_rows=2), False),
        (dict(sampling_mode="greedy_fast", kv_storage="bf16", candidate_budget=9), False),
    ],
)
def test_generator_resolves_the_scope_from_capability(kwargs, admitted) -> None:
    decision = _generator().resolve_speculative_mtp_serving_plan(
        realized_group_rows=kwargs.pop("realized_group_rows", 1),
        resident_capacity=32,
        candidate_budget=kwargs.pop("candidate_budget", 4),
        sampling_mode=kwargs["sampling_mode"],
        kv_storage=kwargs["kv_storage"],
        memory_fit=True,
        request_mode="automatic",
    )
    assert decision.admitted is admitted
    if admitted:
        assert decision.automatic_eligible is True
        assert decision.selected_candidate_count == 4


def test_generator_resolver_admits_both_gfx_backends() -> None:
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        decision = _generator(backend=backend).resolve_speculative_mtp_serving_plan(
            realized_group_rows=1,
            resident_capacity=32,
            candidate_budget=4,
            sampling_mode="sampled",
            kv_storage="auto",
            memory_fit=True,
        )
        assert decision.admitted, (backend, decision.reason)


# ------------------------------------------------------ streaming reachability


class _FakeProvider:
    def __init__(self) -> None:
        self.streamed: list[object] = []

    def generate_detailed(self, request):
        return [GenerationOutput(text="blocking")]

    def stream_detailed(self, request):
        self.streamed.append(request)
        yield GenerationStreamChunk(text="streamed")

    def capabilities(self):
        return {"provider_name": "gemma4_mtp", "streaming_mode": "buffered_public"}

    def close(self) -> None:
        return None


def test_generator_stream_mtp_delegates_to_the_provider() -> None:
    generator = _generator()
    provider = _FakeProvider()
    generator._speculative_provider = provider
    chunks = list(generator.stream_speculative_mtp_detailed(SimpleNamespace(prompts=("hi",))))
    assert [chunk.text for chunk in chunks] == ["streamed"]
    assert len(provider.streamed) == 1


def test_generator_stream_mtp_requires_a_provider() -> None:
    generator = _generator()
    generator._speculative_provider = None
    with pytest.raises(NotImplementedError, match="provider is not configured"):
        list(generator.stream_speculative_mtp_detailed(SimpleNamespace(prompts=("hi",))))


class _StreamingInner:
    supports_speculative_mtp = True

    def __init__(self) -> None:
        self.requests: list[object] = []

    def generate_speculative_mtp_detailed(self, request):
        raise AssertionError("the blocking route is not exercised here")

    def stream_speculative_mtp_detailed(self, request):
        self.requests.append(request)
        yield GenerationStreamChunk(text="chunk")


def test_engine_loop_stream_mtp_delegates_to_a_streaming_inner() -> None:
    inner = _StreamingInner()
    loop = SubmitPollTextGenerator(inner)
    chunks = list(loop.stream_speculative_mtp_detailed(SimpleNamespace(prompts=("hi",))))
    assert [chunk.text for chunk in chunks] == ["chunk"]
    assert len(inner.requests) == 1


def test_engine_loop_stream_mtp_refuses_a_non_streaming_inner() -> None:
    inner = SimpleNamespace(
        supports_speculative_mtp=True,
        generate_speculative_mtp_detailed=lambda request: [],
    )
    loop = SubmitPollTextGenerator(inner)
    with pytest.raises(NotImplementedError, match="streaming is not supported"):
        list(loop.stream_speculative_mtp_detailed(SimpleNamespace(prompts=("hi",))))


# ------------------------------------------------- end-to-end route through LLM


def _llm_with_generator(monkeypatch) -> tuple[LLM, _FakeProvider]:
    llm = LLM("test")
    provider = _FakeProvider()
    generator = _generator()
    generator._speculative_provider = provider
    monkeypatch.setattr(llm, "_get_text_generator", lambda: generator)
    # ``supports_speculative_mtp`` and ``speculative_capabilities`` read the
    # cached text generator directly rather than the lazy accessor.
    llm._text_generator = generator
    # ``speculative_mtp_sampling_modes`` reads the model plugin metadata; the
    # lightweight generator fixture has no artifact to load.
    monkeypatch.setattr(
        llm, "_load_model_metadata", lambda: (None, Gemma4GGUFModel())
    )
    # A provider default resolves the candidate depth without a resident owner.
    llm.speculative_provider = "gemma4_mtp"
    llm.speculative_candidate_budget = None
    return llm, provider


def test_llm_advertises_the_declared_sampling_modes(monkeypatch) -> None:
    llm, _provider = _llm_with_generator(monkeypatch)
    modes = llm.speculative_mtp_sampling_modes
    assert "greedy_fast" in modes
    assert "sampled" in modes


def test_llm_resolves_an_admitted_sampled_plan(monkeypatch) -> None:
    llm, _provider = _llm_with_generator(monkeypatch)
    decision = llm.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        sampling_mode="sampled",
        kv_storage="auto",
        memory_fit=True,
        request_mode="automatic",
    )
    assert decision is not None
    assert decision.admitted is True
    assert decision.automatic_eligible is True


def test_llm_capability_reports_the_declared_scope(monkeypatch) -> None:
    """A plugin with no evidence rows still reports its executable scope."""

    llm, _provider = _llm_with_generator(monkeypatch)
    capability = llm.speculative_mtp_serving_capability
    assert capability is not None
    assert capability.admitted is True
    assert capability.automatic_eligible is True


def test_server_route_keeps_enabled_mode_on_mtp(monkeypatch) -> None:
    """``enabled`` promotes the automatic request on capability, not evidence."""

    llm, _provider = _llm_with_generator(monkeypatch)
    route, plan = _generation_route_for_request(
        _server_config(speculative_mtp_serving="enabled"),
        SimpleNamespace(speculative_mtp=None, stream=False),
        engine=llm,
        sampling=SamplingParams(max_tokens=8),
        prompts=("hi",),
    )
    assert route == _SPECULATIVE_MTP_BATCH_ROUTE
    assert plan is not None and plan.get("automatic_eligible") is True


def test_server_route_selects_mtp_for_a_sampled_request(monkeypatch) -> None:
    llm, _provider = _llm_with_generator(monkeypatch)
    route, plan = _generation_route_for_request(
        _server_config(speculative_mtp_serving="auto"),
        SimpleNamespace(speculative_mtp=True, stream=False),
        engine=llm,
        sampling=SamplingParams(max_tokens=8, temperature=0.7),
        prompts=("hi",),
    )
    assert route == _SPECULATIVE_MTP_BATCH_ROUTE
    assert plan is not None and plan.get("admitted") is True


def test_server_route_keeps_the_automatic_request_on_mtp(monkeypatch) -> None:
    llm, _provider = _llm_with_generator(monkeypatch)
    route, plan = _generation_route_for_request(
        _server_config(speculative_mtp_serving="auto"),
        SimpleNamespace(speculative_mtp=None, stream=False),
        engine=llm,
        sampling=SamplingParams(max_tokens=8),
        prompts=("hi",),
    )
    assert route == _SPECULATIVE_MTP_AUTO_ROUTE
    assert plan is not None and plan.get("automatic_eligible") is True
    # The automatic route is resolved only after the group is known; the
    # admitted capability plan must realize on the batch route for execution.
    realized, _decision = _resolve_realized_generation_route(
        route,
        group_rows=1,
        sampling=SamplingParams(max_tokens=8),
        precomputed_decision=plan,
    )
    assert realized == _SPECULATIVE_MTP_BATCH_ROUTE


def test_server_route_refuses_an_explicit_width_the_capability_lacks(monkeypatch) -> None:
    """Two rows are above the assistant head's one-request scope: fail loudly."""

    llm, _provider = _llm_with_generator(monkeypatch)
    with pytest.raises(OpenAIHTTPError) as excinfo:
        _generation_route_for_request(
            _server_config(speculative_mtp_serving="auto"),
            SimpleNamespace(speculative_mtp=True, stream=False),
            engine=llm,
            sampling=SamplingParams(max_tokens=8),
            prompts=("a", "b"),
        )
    assert excinfo.value.status_code == 501
    assert excinfo.value.code == "unsupported_feature"


def test_llm_streams_mtp_through_the_generator(monkeypatch) -> None:
    llm, provider = _llm_with_generator(monkeypatch)
    chunks = list(
        llm.stream_speculative_mtp_detailed("hi", SamplingParams(max_tokens=4))
    )
    assert [chunk.text for chunk in chunks] == ["streamed"]
    assert len(provider.streamed) == 1


def test_llm_stream_mtp_requires_exactly_one_prompt(monkeypatch) -> None:
    llm, _provider = _llm_with_generator(monkeypatch)
    with pytest.raises(ValueError, match="exactly one prompt"):
        list(llm.stream_speculative_mtp_detailed(["a", "b"], SamplingParams(max_tokens=4)))
