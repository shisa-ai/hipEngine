"""Q8 DPP family choice is isolated by execution owner and launch session."""
from contextlib import nullcontext
from types import SimpleNamespace
import inspect
import pytest
from hipengine.runtime import gguf_linear as linear
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions


@pytest.mark.parametrize('enabled', [False, True])
def test_owner_scope_overrides_ambient_and_restores(monkeypatch, enabled):
    monkeypatch.setenv(linear._Q8_DPP_REDUCTION_ENV, '0' if enabled else '1')
    owner = SimpleNamespace(runner=SimpleNamespace(q8_dpp_reduction_enabled=not enabled))
    with pytest.raises(RuntimeError, match='restore'):
        with runtime.q8_dpp_reduction_enabled(owner, enabled):
            assert owner.q8_dpp_reduction_enabled is enabled
            assert owner.runner.q8_dpp_reduction_enabled is enabled
            assert _inherit_lane_decisions(owner, SimpleNamespace()).q8_dpp_reduction_enabled is enabled
            with linear.q8_dpp_reduction_session(runtime.resolve_q8_dpp_reduction(owner)):
                assert linear._q8_dpp_reduction_raw() == ('1' if enabled else '0')
            raise RuntimeError('restore')
    assert not hasattr(owner, 'q8_dpp_reduction_enabled')
    assert owner.runner.q8_dpp_reduction_enabled is (not enabled)


def test_nested_session_restores_enabled_default_without_process_fallback(monkeypatch):
    monkeypatch.setenv(linear._Q8_DPP_REDUCTION_ENV, 'custom')
    assert linear._q8_dpp_reduction_raw() == '1'
    with linear.q8_dpp_reduction_session(False):
        assert linear._q8_dpp_reduction_raw() == '0'
        with linear.q8_dpp_reduction_session(True):
            assert linear._q8_dpp_reduction_raw() == '1'
        assert linear._q8_dpp_reduction_raw() == '0'
    assert linear._q8_dpp_reduction_raw() == '1'


@pytest.mark.parametrize('raw', ['0', '1', '', 'custom'])
def test_unbound_default_ignores_retired_dpp_input(monkeypatch, raw):
    monkeypatch.setenv(linear._Q8_DPP_REDUCTION_ENV, raw)
    assert linear._q8_dpp_reduction_raw() == '1'
    assert '_raw_env_value' not in inspect.getsource(linear._q8_dpp_reduction_raw)


@pytest.mark.parametrize('profile', ['strict', 'production', 'batch_invariant'])
def test_profiles_publish_enabled_dpp_without_process_writes(monkeypatch, profile):
    import os
    from hipengine.execution_profiles import resolve_runtime_profile
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    monkeypatch.setenv(linear._Q8_DPP_REDUCTION_ENV, '0')
    owner = SimpleNamespace(runner=SimpleNamespace())
    before = dict(os.environ)
    resolved = resolve_runtime_profile(model='qwen4_exp_gguf', backend='hip_gfx1151',
        quant='gguf_ud_q4_k_xl', profile=profile)
    resolved.construct_generator(lambda: owner)
    assert owner.q8_dpp_reduction_enabled is True
    assert owner.runner.q8_dpp_reduction_enabled is True
    assert _inherit_lane_decisions(owner, SimpleNamespace()).q8_dpp_reduction_enabled is True
    assert dict(os.environ) == before


@pytest.mark.parametrize('enabled', [False, True])
def test_public_prefill_chunk_scopes_the_bound_choice(monkeypatch, enabled):
    monkeypatch.setenv(linear._Q8_DPP_REDUCTION_ENV, '0' if enabled else '1')
    monkeypatch.setattr(runtime, '_qwen4_exp_q8_tile_scope', lambda _: nullcontext())
    seen = []
    def chunk(tokens, **kwargs):
        seen.append(linear._q8_dpp_reduction_raw())
        raise RuntimeError('observed')
    owner = SimpleNamespace(q8_dpp_reduction_enabled=enabled, raw_prefill_policy=None,
        q8_wave_scale_prefill=True, max_sequence_length=4096, prefill_chunk_size=1024,
        position=0, reset=lambda: None, _prefill_chunk=chunk,
        _q8_mmq_prefill_context=nullcontext, _wide_f16_activation_context=nullcontext)
    with pytest.raises(RuntimeError, match='observed'):
        next(runtime.Qwen4ExpGGUFResidentModelRunner._prefill_chunked_steps(owner, [1] * 73))
    assert seen == ['1' if enabled else '0']


def test_late_constructor_carries_owner_choice(monkeypatch):
    from hipengine.generation import qwen4_exp_gguf as generation
    calls = []
    monkeypatch.setattr(generation, 'Qwen4ExpGGUFResidentModelRunner',
        lambda resident, **kwargs: calls.append(kwargs) or SimpleNamespace(**kwargs))
    owner = generation.Qwen4ExpGGUFTextGenerator(model_path='unused.gguf',
        weight_index=SimpleNamespace(), model_plugin=SimpleNamespace(), tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(max_sequence_length=8, prefill_chunk_size=4, close=lambda: None))
    owner.q8_dpp_reduction_enabled = False
    try:
        clone = owner._new_resident_runner(SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        owner.close()
    assert calls[0]['q8_dpp_reduction_enabled'] is clone.q8_dpp_reduction_enabled is False
