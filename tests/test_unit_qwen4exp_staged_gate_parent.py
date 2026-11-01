"""The staged comparison arm must expose its exact parent without caller patches."""
import os
from types import SimpleNamespace

import pytest

from scripts import execution_profile_q8_wmma_prefill_layers_gate as gate
from tests.test_unit_qwen4exp_staged_f32_interface import STAGED, _Weight, _Recorded
from hipengine.runtime import gguf_linear as linear


@pytest.mark.parametrize('route_name', [
    'staged_f32', 'staged_f32_composed_vs_exact', 'staged_f32_over_wmma',
])
@pytest.mark.parametrize('candidate', [False, True])
@pytest.mark.parametrize('rows', [2, 73])
def test_staged_gate_exposes_exact_parent(monkeypatch, route_name, rows, candidate):
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    route = gate.SELECTORS[route_name]
    owner = SimpleNamespace(runner=SimpleNamespace())
    saved = {key: os.environ.get(key) for key in route.env_keys}
    prior_rowtile = linear._resolve_use_q4k_rowtile(None)
    prior_dpp = os.environ.get('HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION')
    cache = linear._DISPATCH_RESOLVE_CACHE
    prior_cache = dict(cache)
    variants = []

    def stop(**kwargs):
        variants.append(kwargs['variant'])
        raise _Recorded

    monkeypatch.setattr(linear, '_record_gguf_linear_dispatch', stop)
    cache.clear()
    try:
        with (
            gate._apply_route_env(route, candidate=candidate, layers='0', generator=owner),
            linear.raw_k_prefill_rowbatch_session(32),
            linear.raw_k_prefill_variant_session('coltile8_wave_scale'),
        ):
            with pytest.raises(_Recorded):
                linear.launch_gguf_linear(
                    _Weight(0), 0, 0, rows, 2560, 10240,
                    activation_dtype=linear.GGUF_ACTIVATION_F32,
                    output_dtype=linear.GGUF_OUTPUT_F32, libraries={},
                )
            if candidate:
                assert variants == [STAGED]
            else:
                assert variants[0] in linear._Q8_DENSE_WIDE_EXACT_PARENTS | {
                    'coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out',
                }
            assert route.process_env_for(candidate=candidate, layers='0')[
                'HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION'] == '0'
        assert linear._resolve_use_q4k_rowtile(None) is prior_rowtile
        assert os.environ.get('HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION') == prior_dpp
    finally:
        gate._restore_env(saved)
        cache.clear()
        cache.update(prior_cache)


@pytest.mark.parametrize('prior', [None, '1', 'diagnostic'])
def test_staged_preemptors_restore_on_capture_failure(monkeypatch, prior):
    key = 'HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION'
    if prior is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, prior)
    route = gate.SELECTORS['staged_f32']
    saved = {name: os.environ.get(name) for name in route.env_keys}
    before = linear._resolve_use_q4k_rowtile(None)
    owner = SimpleNamespace(runner=SimpleNamespace())
    try:
        with pytest.raises(RuntimeError, match='capture failed'):
            with gate._apply_route_env(route, candidate=True, layers='0', generator=owner):
                assert os.environ[key] == '0'
                assert linear._resolve_use_q4k_rowtile(None) is False
                raise RuntimeError('capture failed')
        assert os.environ.get(key) == prior
        assert linear._resolve_use_q4k_rowtile(None) is before
        assert not hasattr(owner, 'q8_wmma_layer_scope')
        assert not hasattr(owner.runner, 'q8_wmma_layer_scope')
    finally:
        gate._restore_env(saved)
