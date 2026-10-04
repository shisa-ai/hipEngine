from types import SimpleNamespace

import pytest

from hipengine.generation import qwen4_exp_gguf as generation
from hipengine.generation import qwen4_exp_profiles as profiles


@pytest.mark.parametrize("explicit,allocated",[(None,None),(512,512),(1024,1024),(4096,4096)])
def test_qualified_factory_chunk_and_capacity(monkeypatch,explicit,allocated):
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(runner=SimpleNamespace(prefill_chunk_size=kwargs["prefill_chunk_size"]))
    monkeypatch.setattr(generation,"Qwen4ExpGGUFTextGenerator",factory)
    result = generation.make_qwen4_exp_ud_q4_k_xl_generator_gfx1151(
        model_path="unused",weight_index=object(),model_plugin=object(),
        max_sequence_length=8192,resident_capacity=2,prefill_chunk_size=explicit)
    # An unset chunk is passed through as None so the generator's capacity guard
    # picks it; an explicit chunk is honoured verbatim.
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
    # No admission ran (the constructor is stubbed), so the binder keeps its own
    # production default rather than guessing an admitted chunk.
    assert qualified.runner.prefill_chunk_size == 1024
    other = generation.make_qwen4_exp_gguf_generator_gfx1151(
        model_path="unused",weight_index=object(),model_plugin=object())
    profiles._bind_default_chunk(other,production=True)
    assert other.runner.prefill_chunk_size == 512


def test_production_binds_the_admitted_chunk(monkeypatch):
    """The binder binds what the capacity guard admitted, not a fixed constant."""
    def factory(**kwargs):
        return SimpleNamespace(
            runner=SimpleNamespace(prefill_chunk_size=kwargs["prefill_chunk_size"]))
    monkeypatch.setattr(generation,"Qwen4ExpGGUFTextGenerator",factory)
    for admitted in (4096,2048,1024):
        generator = generation.make_qwen4_exp_ud_q4_k_xl_generator_gfx1151(
            model_path="unused",weight_index=object(),model_plugin=object())
        generator.admitted_prefill_chunk_size = admitted
        profiles._bind_default_chunk(generator,production=True)
        assert generator.runner.prefill_chunk_size == admitted


def _plan(monkeypatch, *, context_tokens, staging_token_capacity):
    return SimpleNamespace(
        context_tokens=context_tokens, staging_token_capacity=staging_token_capacity)


def test_capacity_guard_takes_the_largest_admissible_chunk(monkeypatch):
    """Every rung preserves the requested context, so the largest is taken."""
    seen = []
    def plan_fn(model_map, *, staging_token_capacity):
        seen.append(staging_token_capacity)
        return _plan(monkeypatch,context_tokens=262144,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        return plan
    chunk, plan, admission = generation._select_prefill_chunk(
        object(),ladder=(4096,2048,1024),available_device_bytes=1<<40,
        requested_context=None,native_context_length=262144,resident_capacity=1,
        plan_fn=plan_fn,admit_fn=admit_fn)
    assert chunk == 4096
    # The ladder short-circuits: only the winning rung is evaluated.
    assert seen == [4096]


def test_capacity_guard_falls_back_when_the_larger_chunk_costs_context(monkeypatch):
    """4096 cannot hold 262144 tokens here; 2048 can, so 2048 is taken."""
    seen = []
    def plan_fn(model_map, *, staging_token_capacity):
        seen.append(staging_token_capacity)
        return _plan(monkeypatch,context_tokens=0,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        # Only the 2048 rung admits the full native context.
        return _plan(monkeypatch,context_tokens=262144 if kwargs["prefill_chunk_size"]==2048 else 131072,
                     staging_token_capacity=plan.staging_token_capacity)
    chunk, plan, admission = generation._select_prefill_chunk(
        object(),ladder=(4096,2048,1024),available_device_bytes=1<<40,
        requested_context=None,native_context_length=262144,resident_capacity=1,
        plan_fn=plan_fn,admit_fn=admit_fn)
    assert chunk == 2048
    assert seen == [4096,2048]


def test_capacity_guard_falls_back_on_memory_error(monkeypatch):
    def plan_fn(model_map, *, staging_token_capacity):
        return _plan(monkeypatch,context_tokens=262144,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        if kwargs["prefill_chunk_size"] == 4096:
            raise MemoryError("weights, resident scratch and reserve do not fit")
        return plan
    chunk, _, _ = generation._select_prefill_chunk(
        object(),ladder=(4096,2048,1024),available_device_bytes=1<<40,
        requested_context=None,native_context_length=262144,resident_capacity=1,
        plan_fn=plan_fn,admit_fn=admit_fn)
    assert chunk == 2048


def test_capacity_guard_reraises_when_no_rung_fits(monkeypatch):
    def plan_fn(model_map, *, staging_token_capacity):
        return _plan(monkeypatch,context_tokens=262144,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        raise MemoryError("nothing fits")
    with pytest.raises(MemoryError,match="nothing fits"):
        generation._select_prefill_chunk(
            object(),ladder=(4096,2048,1024),available_device_bytes=1,
            requested_context=None,native_context_length=262144,resident_capacity=1,
            plan_fn=plan_fn,admit_fn=admit_fn)


def test_capacity_guard_takes_the_smallest_when_context_is_unreachable(monkeypatch):
    """A caller asking for more context than any rung admits still gets a runner."""
    def plan_fn(model_map, *, staging_token_capacity):
        return _plan(monkeypatch,context_tokens=262144,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        return _plan(monkeypatch,context_tokens=131072,
                     staging_token_capacity=plan.staging_token_capacity)
    chunk, _, admission = generation._select_prefill_chunk(
        object(),ladder=(4096,2048,1024),available_device_bytes=1<<40,
        requested_context=262144,native_context_length=262144,resident_capacity=1,
        plan_fn=plan_fn,admit_fn=admit_fn)
    assert chunk == 1024
    assert admission.context_tokens == 131072


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
