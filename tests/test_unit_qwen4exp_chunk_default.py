from types import SimpleNamespace

import pytest

from hipengine.generation import qwen4_exp_gguf as generation
from hipengine.generation import qwen4_exp_profiles as profiles


@pytest.mark.parametrize("explicit,allocated",[(None,1024),(512,512),(1024,1024)])
def test_qualified_factory_chunk_and_capacity(monkeypatch,explicit,allocated):
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(runner=SimpleNamespace(prefill_chunk_size=kwargs["prefill_chunk_size"]))
    monkeypatch.setattr(generation,"Qwen4ExpGGUFTextGenerator",factory)
    result = generation.make_qwen4_exp_ud_q4_k_xl_generator_gfx1151(
        model_path="unused",weight_index=object(),model_plugin=object(),
        max_sequence_length=8192,resident_capacity=2,prefill_chunk_size=explicit)
    assert calls[0]["prefill_chunk_size"] == allocated
    assert calls[0]["max_sequence_length"] == 8192
    assert calls[0]["resident_capacity"] == 2
    profiles._bind_default_chunk(result,production=False)
    assert result.runner.prefill_chunk_size == (512 if explicit is None else explicit)


def test_production_chunk_binding_and_unqualified_factory(monkeypatch):
    seen = []
    def factory(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(runner=SimpleNamespace(prefill_chunk_size=kwargs["prefill_chunk_size"]))
    monkeypatch.setattr(generation,"Qwen4ExpGGUFTextGenerator",factory)
    qualified = generation.make_qwen4_exp_ud_q4_k_xl_generator_gfx1151(
        model_path="unused",weight_index=object(),model_plugin=object())
    profiles._bind_default_chunk(qualified,production=True)
    assert qualified.runner.prefill_chunk_size == 1024
    other = generation.make_qwen4_exp_gguf_generator_gfx1151(
        model_path="unused",weight_index=object(),model_plugin=object())
    profiles._bind_default_chunk(other,production=True)
    assert other.runner.prefill_chunk_size == 512


def test_factories_are_quant_scoped_and_capacity_forwardable():
    from hipengine.generation.registry import resolve_text_generator
    from hipengine.llm import _factory_capacity_kwargs
    factory = resolve_text_generator(model="qwen4_exp_gguf",backend="hip_gfx1151",
                                     quant="gguf_ud_q4_k_xl")
    assert factory is generation.make_qwen4_exp_ud_q4_k_xl_generator_gfx1151
    assert resolve_text_generator(model="qwen4_exp_gguf",backend="hip_gfx1151",
                                  quant="gguf_q4_k_m") is generation.make_qwen4_exp_gguf_generator_gfx1151
    assert _factory_capacity_kwargs(factory,max_sequence_length=8192,resident_capacity=2) == {
        "max_sequence_length":8192,"resident_capacity":2}
