from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.loading.qwen4_exp_gguf import build_qwen4_exp_gguf_tensor_map
from hipengine.loading.qwen4_exp_materialize import plan_qwen4_exp_residency
from tests.test_qwen4_exp_gguf_mapping import _infos


def residency():
    plan = plan_qwen4_exp_residency(build_qwen4_exp_gguf_tensor_map(_infos()))
    return replace(plan,device_weight_bytes=64*1024**3,staging_bytes=16*1024**2)


def test_auto_admission_native_and_memory_clamp():
    from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context
    plan = residency()
    native = resolve_qwen4_exp_context(plan,available_device_bytes=128*1024**3,resident_capacity=2)
    assert native.context_tokens == plan.config.context_length
    assert native.scratch_bytes == 8*1024**3
    limited = resolve_qwen4_exp_context(plan,available_device_bytes=native.required_bytes-1024**3,
                                      resident_capacity=2)
    assert 2051 < limited.context_tokens < native.context_tokens
    assert limited.passed
    with pytest.raises(MemoryError):
        resolve_qwen4_exp_context(plan,available_device_bytes=1,resident_capacity=2)
    with pytest.raises(MemoryError):
        resolve_qwen4_exp_context(plan,available_device_bytes=limited.required_bytes,
                                 requested_context=native.context_tokens,resident_capacity=2)


def test_explicit_context_and_model_limit():
    from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context
    plan = residency()
    selected = resolve_qwen4_exp_context(plan,available_device_bytes=128*1024**3,
                                        requested_context=2051,resident_capacity=1)
    assert selected.context_tokens == 2051
    assert selected.kv_bytes == 2304*plan.config.bf16_kv_bytes_per_token
    for value in (0,-1,2,plan.config.context_length+1):
        with pytest.raises(ValueError):
            resolve_qwen4_exp_context(plan,available_device_bytes=128*1024**3,requested_context=value)
    capped = resolve_qwen4_exp_context(plan,available_device_bytes=128*1024**3,native_context_length=16384)
    assert capped.context_tokens == 16384


def test_factory_capacity_kwargs_are_opt_in():
    from hipengine.llm import _factory_capacity_kwargs
    def old(*,model_path,weight_index,model_plugin): pass
    def new(*,model_path,weight_index,model_plugin,max_sequence_length=None,resident_capacity=None): pass
    assert _factory_capacity_kwargs(old,max_sequence_length=8192,resident_capacity=2)=={}
    assert _factory_capacity_kwargs(new,max_sequence_length=8192,resident_capacity=1)=={
        "max_sequence_length":8192,"resident_capacity":1}
    assert _factory_capacity_kwargs(new,max_sequence_length=None,resident_capacity=None)=={}


def test_generator_prepare_reports_real_capacity():
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    runner = SimpleNamespace(max_sequence_length=262144,close=lambda: None)
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused",weight_index=object(),model_plugin=object(),tokenizer=object(),runner=runner)
    assert generator.prepare() == 262144
    assert generator.prepare(sampling_params=SimpleNamespace(kv_storage="auto")) == 262144
    assert generator.prepare(max_sequence_length=8192) == 8192
    with pytest.raises(ValueError):
        generator.prepare(max_sequence_length=262145)
    assert runner.max_sequence_length == 262144


def test_registered_factory_forwards_capacity(monkeypatch):
    from hipengine.generation import qwen4_exp_gguf as module
    seen = {}
    monkeypatch.setattr(module,"Qwen4ExpGGUFTextGenerator",lambda **kwargs: seen.update(kwargs) or object())
    module.make_qwen4_exp_gguf_generator_gfx1151(
        model_path="unused",weight_index=object(),model_plugin=object(),
        max_sequence_length=16384,resident_capacity=1)
    assert seen["max_sequence_length"] == 16384
    assert seen["resident_capacity"] == 1


def test_llm_forwards_limits_to_declared_factory(monkeypatch):
    from hipengine import LLM
    import hipengine.generation as generation
    seen = {}
    class FakeGenerator:
        def generate(self,request): return []
    def factory(*,model_path,weight_index,model_plugin,max_sequence_length=None,resident_capacity=None):
        seen.update(context=max_sequence_length,capacity=resident_capacity)
        return FakeGenerator()
    generation.register_text_generator(
        model="capacity_test_model",backend="capacity_test_backend",quant="capacity_test_quant",
        factory=factory,replace=True)
    monkeypatch.setattr(generation,"register_builtin_generators",lambda: None)
    llm = LLM("unused",backend="capacity_test_backend",quant="capacity_test_quant",
              max_sequence_length=8192,max_active_requests=1)
    monkeypatch.setattr(llm,"_load_model_metadata",lambda: (object(),SimpleNamespace(name="capacity_test_model")))
    monkeypatch.setattr(llm,"_resolve_backend",lambda: "capacity_test_backend")
    monkeypatch.setattr(llm,"_resolve_quant",lambda model: "capacity_test_quant")
    monkeypatch.setattr(llm,"_resolve_execution_profile",lambda **kwargs: None)
    llm._get_text_generator()
    assert seen == {"context":8192,"capacity":1}
    llm.close()


def test_configured_single_residency_is_advertised():
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused",weight_index=object(),model_plugin=object(),tokenizer=object(),
        runner=SimpleNamespace(max_sequence_length=8192,close=lambda: None),resident_capacity=1)
    assert generator.server_plain_ar_max_active_requests == 1
    assert generator.server_plain_ar_max_active_requests_by_max_sequence_length == {1024:1}
    with pytest.raises(ValueError):
        generator.create_resident_model_runner(capacity=2)


def test_boundary_probe_can_allocate_native_context():
    from scripts.qwen4exp_context_decode_profile import build_parser
    args = build_parser().parse_args([
        "--model-root","/unused","--output","/unused.json","--capacity","262144"])
    assert args.capacity == 262144
    assert args.live_count == [2051,2052,4097]
