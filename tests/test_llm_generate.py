from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hipengine import LLM, SamplingParams
from hipengine.generation import (
    GenerationCancellationToken,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
    register_text_generator,
)


def test_generator_engine_loop_defaults_respect_explicit_env() -> None:
    from hipengine.generation import EngineLoopConfig
    from hipengine.llm import _engine_loop_config_with_generator_defaults

    generator = SimpleNamespace(
        engine_loop_config_defaults={
            "prefill_decode_policy": "fair",
            "max_prefill_chunk_tokens": 256,
            "fair_prefill_burst_chunks": 2,
        }
    )
    resolved = _engine_loop_config_with_generator_defaults(
        EngineLoopConfig(),
        generator,
        environ={},
    )
    assert resolved.prefill_decode_policy == "fair"
    assert resolved.max_prefill_chunk_tokens == 256
    assert resolved.fair_prefill_burst_chunks == 2

    explicit = EngineLoopConfig(
        prefill_decode_policy="protect_ttft",
        max_prefill_chunk_tokens=64,
        fair_prefill_burst_chunks=3,
    )
    preserved = _engine_loop_config_with_generator_defaults(
        explicit,
        generator,
        environ={
            "HIPENGINE_PREFILL_DECODE_POLICY": "protect_ttft",
            "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS": "64",
            "HIPENGINE_FAIR_PREFILL_BURST_CHUNKS": "3",
        },
    )
    assert preserved == explicit


def test_gfx1100_gguf_q4_k_m_factory_defaults_to_fair_launch_policy(monkeypatch) -> None:
    """The gfx1100 Q4_K_M plain-AR path launches with the measured fair default."""

    import hipengine.generation.qwen35_gguf as qwen35_gguf
    from hipengine.generation import (
        EngineLoopConfig,
        register_builtin_generators,
        resolve_text_generator,
    )
    from hipengine.llm import _engine_loop_config_with_generator_defaults

    monkeypatch.setattr(
        qwen35_gguf.Qwen35GGUFTokenizer,
        "from_gguf_info",
        classmethod(lambda cls, weight_index: object()),
    )
    register_builtin_generators()
    generator = resolve_text_generator(
        model="qwen3_5_gguf",
        backend="hip_gfx1100",
        quant="gguf_q4_k_m",
    )(
        model_path="/tmp/fake.gguf",
        weight_index=object(),
        model_plugin=object(),
    )
    assert generator.backend == "hip_gfx1100"
    assert generator.engine_loop_config_defaults == {
        "prefill_decode_policy": "fair",
        "max_prefill_chunk_tokens": 256,
        "fair_prefill_burst_chunks": 1,
    }
    # The registry-owned plain-AR route cap is unchanged by the policy default.
    assert generator.server_plain_ar_max_active_requests == 4

    # The explicit env pin must still override the scoped default.
    pinned = _engine_loop_config_with_generator_defaults(
        EngineLoopConfig(),
        generator,
        environ={"HIPENGINE_PREFILL_DECODE_POLICY": "protect_decode"},
    )
    assert pinned.prefill_decode_policy == "protect_decode"

    # Other GGUF quants on gfx1100 must not inherit the scoped loop default.
    other = resolve_text_generator(
        model="qwen3_5_gguf",
        backend="hip_gfx1100",
        quant="gguf_q8_0",
    )(
        model_path="/tmp/fake-q8.gguf",
        weight_index=object(),
        model_plugin=object(),
    )
    assert other.engine_loop_config_defaults == {}


def test_llm_generate_dispatches_through_generation_registry(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    calls = {}

    class FakeGenerator:
        def generate(self, request: GenerationRequest) -> list[str]:
            calls["request"] = request
            return [prompt + "!" for prompt in request.prompts]

    def factory(**kwargs):
        calls["factory_kwargs"] = kwargs
        generator = FakeGenerator()
        calls["generator"] = generator
        return generator

    fake_index = SimpleNamespace(config={"architectures": ["FakeForCausalLM"]}, model_path="/tmp/fake-model")
    fake_plugin = SimpleNamespace(name="fake_model")

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=factory,
        replace=True,
    )

    llm = LLM(
        "/tmp/fake-model",
        backend="fake_backend",
        quant="fake_quant",
        max_active_requests=8,
        prefix_cache="radix",
    )
    out = llm.generate(["a", "b"], SamplingParams(max_tokens=1))

    assert out == ["a!", "b!"]
    assert llm._text_generator._runner.capacity == 8
    assert calls["generator"].speculative_candidate_budget == 4
    assert llm._text_generator._loop.config.prefix_cache == "radix"
    assert calls["factory_kwargs"] == {
        "model_path": "/tmp/fake-model",
        "weight_index": fake_index,
        "model_plugin": fake_plugin,
    }
    assert calls["request"] == GenerationRequest(
        prompts=("a", "b"),
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def test_llm_caps_resident_capacity_to_registered_plain_ar_width(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class CappedFakeGenerator:
        server_plain_ar_max_active_requests = 4

        def generate(self, request: GenerationRequest) -> list[str]:
            return [f"{prompt}!" for prompt in request.prompts]

    fake_index = SimpleNamespace(
        config={"architectures": ["CappedFakeForCausalLM"]},
        model_path="/tmp/fake-model",
    )
    fake_plugin = SimpleNamespace(name="capped_fake_model")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="capped_fake_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: CappedFakeGenerator(),
        replace=True,
    )

    llm = LLM(
        "/tmp/fake-model",
        backend="fake_backend",
        quant="fake_quant",
        max_active_requests=8,
    )
    output = llm.generate(["a", "b"], SamplingParams(max_tokens=1))

    assert output == ["a!", "b!"]
    assert llm.max_active_requests == 8
    assert llm.server_plain_ar_max_active_requests == 4
    assert llm._text_generator._runner.capacity == 4


def test_llm_selects_registered_short_context_plain_ar_width(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class ContextCappedFakeGenerator:
        server_plain_ar_max_active_requests = 4
        server_plain_ar_max_active_requests_by_max_sequence_length = {768: 13}

        def generate(self, request: GenerationRequest) -> list[str]:
            return [f"{prompt}!" for prompt in request.prompts]

    fake_index = SimpleNamespace(
        config={"architectures": ["ContextCappedFakeForCausalLM"]},
        model_path="/tmp/fake-model",
    )
    fake_plugin = SimpleNamespace(name="context_capped_fake_model")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="context_capped_fake_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: ContextCappedFakeGenerator(),
        replace=True,
    )

    short = LLM(
        "/tmp/fake-model",
        backend="fake_backend",
        quant="fake_quant",
        max_active_requests=13,
        max_sequence_length=768,
    )
    long = LLM(
        "/tmp/fake-model",
        backend="fake_backend",
        quant="fake_quant",
        max_active_requests=13,
        max_sequence_length=4096,
    )
    prepared_short = LLM(
        "/tmp/fake-model",
        backend="fake_backend",
        quant="fake_quant",
        max_active_requests=13,
    )

    assert short.generate(["short"], SamplingParams(max_tokens=1)) == ["short!"]
    assert short.server_plain_ar_max_active_requests == 13
    assert short._text_generator._runner.capacity == 13
    assert long.generate(["long"], SamplingParams(max_tokens=1)) == ["long!"]
    assert long.server_plain_ar_max_active_requests == 4
    assert long._text_generator._runner.capacity == 4
    prepared_short.prepare(max_sequence_length=768)
    assert prepared_short.max_sequence_length == 768
    assert prepared_short.server_plain_ar_max_active_requests == 13
    assert prepared_short._text_generator._runner.capacity == 13


def test_llm_generate_detailed_preserves_exact_token_prompt_rows(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    calls = {}

    class FakeGenerator:
        def generate_detailed(self, request: GenerationRequest):
            calls["request"] = request
            return [
                generation.GenerationOutput(
                    text=f"row-{index}",
                    generated_token_ids=(900 + index,),
                )
                for index, _prompt in enumerate(request.prompts)
            ]

    fake_index = SimpleNamespace(
        config={"architectures": ["FakeExactTokensForCausalLM"]},
        model_path="/tmp/fake-model",
    )
    fake_plugin = SimpleNamespace(name="fake_exact_tokens")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_exact_tokens",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")
    outputs = llm.generate_detailed(
        [[10, 11, 12], [20, 21]],
        SamplingParams(max_tokens=1, ignore_eos=True),
    )

    assert [output.generated_token_ids for output in outputs] == [(900,), (901,)]
    assert calls["request"].prompts == ((10, 11, 12), (20, 21))
    assert calls["request"].prompt_input_kind == "token_ids"


def test_llm_attaches_explicit_speculative_provider_without_changing_default_route(
    monkeypatch,
) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models
    import hipengine.speculative.registry as speculative_registry
    from hipengine.generation import GenerationOutput
    from hipengine.speculative.registry import (
        SpeculativeProviderKey,
        register_speculative_provider,
    )

    events: list[object] = []

    class FakeProvider:
        provider_name = "test_dflash"

        def generate_detailed(self, request: GenerationRequest):
            events.append(("spec_generate", request.prompts))
            return [GenerationOutput(text=f"spec:{prompt}") for prompt in request.prompts]

        def stream_detailed(self, request: GenerationRequest):
            events.append(("spec_stream", request.prompts))
            yield GenerationStreamChunk(text="spec-stream")

        def capabilities(self):
            return {"provider": self.provider_name, "candidate_budget": 4}

        def close(self) -> None:
            events.append("provider_close")

    class FakeGenerator:
        def __init__(self) -> None:
            self.provider = None

        def attach_speculative_provider(self, provider) -> None:
            self.provider = provider
            events.append("attached")

        @property
        def supports_speculative(self) -> bool:
            return self.provider is not None

        def speculative_capabilities(self):
            return {} if self.provider is None else self.provider.capabilities()

        def generate_speculative_detailed(self, request: GenerationRequest):
            return self.provider.generate_detailed(request)

        def stream_speculative_detailed(self, request: GenerationRequest):
            return self.provider.stream_detailed(request)

        def generate(self, request: GenerationRequest) -> list[str]:
            events.append(("ar_generate", request.prompts))
            return [f"ar:{prompt}" for prompt in request.prompts]

        def close(self) -> None:
            if self.provider is not None:
                self.provider.close()
            events.append("generator_close")

    fake_index = SimpleNamespace(
        config={"architectures": ["FakeSpecForCausalLM"]},
        model_path="/tmp/fake-model",
    )
    fake_plugin = SimpleNamespace(name="fake_spec_model")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    monkeypatch.setattr(
        speculative_registry,
        "register_builtin_speculative_providers",
        lambda: None,
    )
    register_text_generator(
        model="fake_spec_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )
    register_speculative_provider(
        SpeculativeProviderKey(
            "test_dflash",
            "fake_spec_model",
            "fake_backend",
            "fake_quant",
        ),
        lambda *, target_generator, config: FakeProvider(),
        replace=True,
    )

    llm = LLM(
        "/tmp/fake-model",
        backend="fake_backend",
        quant="fake_quant",
        speculative_provider="test_dflash",
        draft_model="/tmp/fake-drafter",
        speculative_candidate_budget=4,
    )

    assert llm.supports_speculative is True
    assert llm.generate("one", SamplingParams(max_tokens=1)) == ["ar:one"]
    assert [item.text for item in llm.generate_speculative_detailed("two", SamplingParams(max_tokens=1))] == [
        "spec:two"
    ]
    assert [item.text for item in llm.stream_speculative_detailed("three", SamplingParams(max_tokens=1))] == [
        "spec-stream"
    ]
    assert llm.supports_speculative is True
    assert llm.speculative_capabilities == {
        "provider": "test_dflash",
        "candidate_budget": 4,
    }
    llm.close()

    assert events[0] == "attached"
    assert ("ar_generate", ("one",)) in events
    assert events[-2:] == ["provider_close", "generator_close"]


def test_generation_request_rejects_invalid_token_prompt_rows() -> None:
    import pytest

    with pytest.raises(ValueError, match="must not be empty"):
        GenerationRequest(
            prompts=((),),
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
        )
    with pytest.raises(ValueError, match="non-negative"):
        GenerationRequest(
            prompts=((1, -2),),
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
        )


def test_llm_tokenize_delegates_to_generator(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class FakeGenerator:
        def tokenize(self, text: str) -> tuple[int, ...]:
            return tuple(ord(char) for char in text)

        def generate(self, request: GenerationRequest) -> list[str]:
            return ["unused"]

    fake_index = SimpleNamespace(config={"architectures": ["FakeForCausalLM"]}, model_path="/tmp/fake-model")
    fake_plugin = SimpleNamespace(name="fake_tokenizer_model")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_tokenizer_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")

    assert llm.tokenize("Az") == (65, 122)


def test_llm_stream_detailed_preserves_backend_stream_telemetry(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class FakeGenerator:
        def stream_detailed(self, request: GenerationRequest):
            yield GenerationStreamChunk(
                "alpha",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=3,
                    generated_tokens=1,
                    phase="answer",
                    sampler_mode="processed_argmax",
                ),
            )

        def generate(self, request: GenerationRequest) -> list[str]:
            return ["unused"]

    fake_index = SimpleNamespace(config={"architectures": ["FakeStreamForCausalLM"]}, model_path="/tmp/fake-model")
    fake_plugin = SimpleNamespace(name="fake_stream_model")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_stream_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")

    detailed_chunks = list(llm.stream_detailed("hello", SamplingParams(max_tokens=1)))
    assert detailed_chunks[0].text == "alpha"
    assert detailed_chunks[0].telemetry.to_json_dict()["decode_state"]["sampler_mode"] == "processed_argmax"
    assert list(llm.stream("hello", SamplingParams(max_tokens=1))) == ["alpha"]


def test_llm_stream_many_detailed_wraps_generation_request(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    calls = {}

    class FakeGenerator:
        supports_stream_many = True

        def stream_many_detailed(self, request: GenerationRequest):
            calls["request"] = request
            yield GenerationStreamChunk(
                "alpha",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=3,
                    generated_tokens=1,
                    row_index=0,
                    phase="answer",
                    sampler_mode="greedy_fast",
                ),
            )

        def generate(self, request: GenerationRequest) -> list[str]:
            return ["unused" for _prompt in request.prompts]

    fake_index = SimpleNamespace(config={"architectures": ["FakeStreamManyForCausalLM"]}, model_path="/tmp/fake-model")
    fake_plugin = SimpleNamespace(name="fake_stream_many_model")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_stream_many_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")
    assert llm.supports_stream_many is False

    chunks = list(llm.stream_many_detailed(["one", "two"], SamplingParams(max_tokens=2)))

    assert llm.supports_stream_many is True
    assert chunks[0].text == "alpha"
    assert chunks[0].telemetry.decode_state.row_index == 0
    assert calls["request"] == GenerationRequest(
        prompts=("one", "two"),
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def test_llm_detokenize_delegates_to_generator(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class FakeGenerator:
        def detokenize(self, token_ids, *, skip_special: bool = False) -> str:
            suffix = " skip" if skip_special else ""
            return ",".join(str(int(token)) for token in token_ids) + suffix

        def generate(self, request: GenerationRequest) -> list[str]:
            return ["unused"]

    fake_index = SimpleNamespace(config={"architectures": ["FakeForCausalLM"]}, model_path="/tmp/fake-model")
    fake_plugin = SimpleNamespace(name="fake_detokenizer_model")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_detokenizer_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")

    assert llm.detokenize([65, 122], skip_special=True) == "65,122 skip"


def test_llm_generate_plumbs_extended_sampling_params(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    calls = {}

    class FakeGenerator:
        def generate(self, request: GenerationRequest) -> list[str]:
            calls["request"] = request
            return ["ok"]

    fake_index = SimpleNamespace(config={"architectures": ["FakeForCausalLM"]}, model_path="/tmp/fake-model")
    fake_plugin = SimpleNamespace(name="fake_sampling_model")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_sampling_model",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")
    cancellation_token = GenerationCancellationToken()
    assert llm.generate(
        "a",
        SamplingParams(
            max_tokens=2,
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            logit_bias={"12": -1.5},
            suppress_token_ids=(13,),
            min_tokens=2,
            eos_token_id=99,
            stop_token_ids=(99,),
            stop_token_sequences=((100, 101),),
            forced_tokens_pending=(104, 105),
            forced_token_reason="tool_choice_required",
            post_thinking_forced_tokens_pending=(106, 107),
            post_thinking_forced_token_reason="post_think_tool",
            force_sequence_completion_token_sequences=((108, 109),),
            force_sequence_completion_reason="tool_close_repair",
            thinking_close_token_ids=(102, 103),
            thinking_hard_token_cap=8,
            thinking_soft_close_window=2,
            seed=123,
            deadline_at=456.0,
            cancellation_token=cancellation_token,
        ),
    ) == ["ok"]
    assert calls["request"].top_k == 40
    assert calls["request"].min_p == 0.05
    assert calls["request"].repetition_penalty == 1.1
    assert calls["request"].presence_penalty == 0.2
    assert calls["request"].frequency_penalty == 0.3
    assert calls["request"].logit_bias == ((12, -1.5),)
    assert calls["request"].suppress_token_ids == (13,)
    assert calls["request"].min_tokens == 2
    assert calls["request"].eos_token_id == 99
    assert calls["request"].stop_token_ids == (99,)
    assert calls["request"].stop_token_sequences == ((100, 101),)
    assert calls["request"].forced_tokens_pending == (104, 105)
    assert calls["request"].forced_token_reason == "tool_choice_required"
    assert calls["request"].post_thinking_forced_tokens_pending == (106, 107)
    assert calls["request"].post_thinking_forced_token_reason == "post_think_tool"
    assert calls["request"].force_sequence_completion_token_sequences == ((108, 109),)
    assert calls["request"].force_sequence_completion_reason == "tool_close_repair"
    assert calls["request"].thinking_close_token_ids == (102, 103)
    assert calls["request"].thinking_hard_token_cap == 8
    assert calls["request"].thinking_soft_close_window == 2
    assert calls["request"].seed == 123
    assert calls["request"].deadline_at == 456.0
    assert calls["request"].cancellation_token is cancellation_token


def test_llm_reuses_generator_across_generate_calls(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    factory_calls = []
    generate_calls = []

    class FakeGenerator:
        def generate(self, request: GenerationRequest) -> list[str]:
            generate_calls.append(request.prompts)
            return [prompt + "!" for prompt in request.prompts]

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return FakeGenerator()

    fake_index = SimpleNamespace(
        config={"architectures": ["FakeForCausalLM"]},
        model_path="/tmp/fake-model",
    )
    fake_plugin = SimpleNamespace(name="fake_model_cached")

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_model_cached",
        backend="fake_backend",
        quant="fake_quant",
        factory=factory,
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")

    assert llm.generate("a", SamplingParams(max_tokens=1)) == ["a!"]
    assert llm.generate("b", SamplingParams(max_tokens=1)) == ["b!"]
    assert len(factory_calls) == 1
    assert generate_calls == [("a",), ("b",)]


def test_llm_default_backend_auto_resolves_env_override(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class FakeGenerator:
        def generate(self, request: GenerationRequest) -> list[str]:
            return ["ok"]

    monkeypatch.setenv("HIPENGINE_BACKEND", "fake_auto_backend")
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(
        loading,
        "load_weight_index",
        lambda model: SimpleNamespace(
            config={"architectures": ["FakeAuto"]},
            model_path="/tmp/fake-model",
        ),
    )
    monkeypatch.setattr(
        models,
        "resolve_model",
        lambda architecture: SimpleNamespace(name="fake_auto"),
    )
    register_text_generator(
        model="fake_auto",
        backend="fake_auto_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", quant="fake_quant")

    assert llm.generate("hello", SamplingParams(max_tokens=1)) == ["ok"]
    assert llm.backend == "auto"
    assert llm._resolved_backend == "fake_auto_backend"
    assert llm.resolved_backend == "fake_auto_backend"


def test_llm_default_quant_resolves_model_plugin_default(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class FakeGenerator:
        def generate(self, request: GenerationRequest) -> list[str]:
            return ["ok"]

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(
        loading,
        "load_weight_index",
        lambda model: SimpleNamespace(
            config={"architectures": ["FakeAutoQuant"]},
            model_path="/tmp/fake-model",
        ),
    )
    monkeypatch.setattr(
        models,
        "resolve_model",
        lambda architecture: SimpleNamespace(
            name="fake_auto_quant",
            default_quant="fake_default_quant",
        ),
    )
    register_text_generator(
        model="fake_auto_quant",
        backend="fake_backend",
        quant="fake_default_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend")

    assert llm.generate("hello", SamplingParams(max_tokens=1)) == ["ok"]
    assert llm.quant == "auto"
    assert llm.resolved_quant == "fake_default_quant"


def test_llm_explicit_quant_overrides_model_plugin_default(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class FakeGenerator:
        def generate(self, request: GenerationRequest) -> list[str]:
            return ["explicit"]

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(
        loading,
        "load_weight_index",
        lambda model: SimpleNamespace(
            config={"architectures": ["FakeExplicitQuant"]},
            model_path="/tmp/fake-model",
        ),
    )
    monkeypatch.setattr(
        models,
        "resolve_model",
        lambda architecture: SimpleNamespace(
            name="fake_explicit_quant",
            default_quant="wrong_default",
        ),
    )
    register_text_generator(
        model="fake_explicit_quant",
        backend="fake_backend",
        quant="chosen_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="chosen_quant")

    assert llm.generate("hello", SamplingParams(max_tokens=1)) == ["explicit"]
    assert llm.resolved_quant == "chosen_quant"


def test_llm_generate_normalizes_single_prompt(monkeypatch) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    class FakeGenerator:
        def generate(self, request: GenerationRequest) -> list[str]:
            assert request.prompts == ("hello",)
            return ["world"]

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(
        loading,
        "load_weight_index",
        lambda model: SimpleNamespace(
            config={"text_config": {"architectures": ["FakeText"]}},
            model_path="/tmp/fake-model",
        ),
    )
    monkeypatch.setattr(models, "resolve_model", lambda architecture: SimpleNamespace(name="fake_single"))
    register_text_generator(
        model="fake_single",
        backend="fake_backend",
        quant="fake_quant",
        factory=lambda **kwargs: FakeGenerator(),
        replace=True,
    )

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")

    assert llm.generate("hello", SamplingParams(max_tokens=1)) == ["world"]
    assert llm.generate([], SamplingParams(max_tokens=1)) == []


def test_llm_resolves_hf_model_id_before_gguf_detection(monkeypatch, tmp_path) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    calls = {}
    resolved = tmp_path / "snapshots" / "abc123"
    resolved.mkdir(parents=True)
    gguf = resolved / "model.gguf"
    gguf.write_bytes(b"GGUF")
    fake_index = SimpleNamespace(path=gguf, architecture="qwen35moe")
    fake_plugin = SimpleNamespace(name="fake_gguf")

    class FakeGenerator:
        def generate(self, request: GenerationRequest) -> list[str]:
            return ["ok"]

    def factory(**kwargs):
        calls["factory_kwargs"] = kwargs
        return FakeGenerator()

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "resolve_model_path", lambda model: resolved)
    monkeypatch.setattr(loading, "discover_gguf_files", lambda model: (Path(model) / "model.gguf",))
    monkeypatch.setattr(loading, "load_gguf_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: fake_plugin)
    register_text_generator(
        model="fake_gguf",
        backend="fake_backend",
        quant="fake_quant",
        factory=factory,
        replace=True,
    )

    llm = LLM("org/model-gguf", backend="fake_backend", quant="fake_quant")

    assert llm.generate("hello", SamplingParams(max_tokens=1)) == ["ok"]
    assert llm.model == str(gguf)
    assert calls["factory_kwargs"] == {
        "model_path": str(gguf),
        "weight_index": fake_index,
        "model_plugin": fake_plugin,
    }
