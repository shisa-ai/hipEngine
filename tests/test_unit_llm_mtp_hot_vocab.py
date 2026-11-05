"""Public forwarding of owner-bound MTP vocabulary choices."""

from types import SimpleNamespace

import pytest

from hipengine import LLM, SamplingParams
from hipengine.llm import ResidentPreparationError


@pytest.mark.parametrize("choice", [None, "auto", "off", "/tmp/selection.json"])
def test_llm_forwards_explicit_hot_vocab_before_generator_construction(monkeypatch, choice):
    import hipengine.generation as generation

    captured = {}

    class Generator:
        def generate(self, request):
            return ["ok" for _ in request.prompts]

    def factory(**kwargs):
        captured.update(kwargs)
        return Generator()

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(generation, "resolve_text_generator", lambda **kw: factory)
    monkeypatch.setattr(LLM, "_load_model_metadata", lambda self: (object(), SimpleNamespace(name="fake")))
    monkeypatch.setattr(LLM, "_resolve_execution_profile", lambda self, **kw: None)
    llm = LLM("/tmp/fake", backend="fake", quant="fake", mtp_hot_vocab=choice)
    try:
        assert llm.generate(["hello"], SamplingParams(max_tokens=1)) == ["ok"]
        assert "mtp_hot_vocab" in captured
        assert captured["mtp_hot_vocab"] == choice
    finally:
        llm.close()


def test_explicit_hot_vocab_is_not_silently_dropped_by_unsupported_factory(monkeypatch):
    import hipengine.generation as generation

    def factory(*, model_path, weight_index, model_plugin):
        raise AssertionError("unsupported factory must not be called")

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(generation, "resolve_text_generator", lambda **kw: factory)
    monkeypatch.setattr(LLM, "_load_model_metadata", lambda self: (object(), SimpleNamespace(name="fake")))
    monkeypatch.setattr(LLM, "_resolve_execution_profile", lambda self, **kw: None)
    llm = LLM("/tmp/fake", backend="fake", quant="fake", mtp_hot_vocab="auto")
    with pytest.raises(ResidentPreparationError, match="mtp_hot_vocab"):
        llm.generate(["hello"], SamplingParams(max_tokens=1))


@pytest.mark.parametrize("factory_name", [
    "make_qwen35_gguf_bringup_generator",
    "make_qwen35_gguf_q4_k_m_generator_gfx1100",
    "make_qwen35_gguf_ud_q3_k_m_generator",
    "make_qwen35_gguf_bringup_generator_gfx1151",
    "make_qwen35_gguf_q4_k_m_generator_gfx1151",
])
@pytest.mark.parametrize("explicit", [False, True])
def test_registered_gguf_factories_preserve_omission_and_explicit_default(monkeypatch, factory_name, explicit):
    import hipengine.generation.qwen35_gguf as module

    captured = {}
    monkeypatch.setattr(module, "Qwen35GGUFBringupGenerator", lambda **kw: captured.update(kw))
    kwargs = dict(model_path="/tmp/fake", weight_index=object(), model_plugin=object())
    if explicit:
        kwargs["mtp_hot_vocab"] = None
    getattr(module, factory_name)(**kwargs)
    assert captured["mtp_hot_vocab"] is None
