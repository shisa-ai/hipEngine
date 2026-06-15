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
        return FakeGenerator()

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

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")
    out = llm.generate(["a", "b"], SamplingParams(max_tokens=1))

    assert out == ["a!", "b!"]
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
