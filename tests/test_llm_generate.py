from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hipengine import LLM, SamplingParams
from hipengine.generation import GenerationRequest, register_text_generator


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
