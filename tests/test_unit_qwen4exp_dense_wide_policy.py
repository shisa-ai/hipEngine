"""Wide prefill ownership is explicit while unbound diagnostics keep env fallback."""
import os
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_linear as gl


def test_bound_policy_ignores_ambient_and_restores_after_failure(monkeypatch):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE', '1')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS', '1,3')
    owner = SimpleNamespace(q8_dense_wide_policy=gl.Q8DenseWidePolicy(False, frozenset({7})))
    assert gl.resolve_q8_dense_wide_policy(owner).enabled is False
    assert gl._q8_dense_wide_enabled() is True
    with pytest.raises(RuntimeError):
        with gl.q8_dense_wide_policy_session(gl.resolve_q8_dense_wide_policy(owner)):
            assert gl._q8_dense_wide_enabled() is False
            assert gl._q8_dense_wide_layers() == frozenset({7})
            raise RuntimeError('injected')
    assert gl._q8_dense_wide_enabled() is True
    assert gl._q8_dense_wide_layers() == frozenset({1, 3})


def test_diagnostic_scope_restores_generator_and_runner(monkeypatch):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE', '0')
    runner = SimpleNamespace(q8_dense_wide_policy=gl.Q8DenseWidePolicy(False, None))
    owner = SimpleNamespace(runner=runner)
    policy = gl.Q8DenseWidePolicy(True, frozenset({0, 16}))
    with gl.q8_dense_wide_policy(owner, policy):
        assert owner.q8_dense_wide_policy == runner.q8_dense_wide_policy == policy
        assert gl.resolve_q8_dense_wide_policy(owner) == policy
        with gl.q8_dense_wide_policy(owner, None):
            assert gl.resolve_q8_dense_wide_policy(owner).enabled is False
        assert runner.q8_dense_wide_policy == policy
    assert not hasattr(owner, 'q8_dense_wide_policy')
    assert runner.q8_dense_wide_policy.enabled is False


def test_unbound_layers_empty_means_all_and_zero_is_a_real_layer(monkeypatch):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE', '1')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS', '')
    assert gl.resolve_q8_dense_wide_policy(SimpleNamespace()).layers is None
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS', '0')
    assert gl.resolve_q8_dense_wide_policy(SimpleNamespace()).layers == frozenset({0})


@pytest.mark.parametrize('layer,selected', [(0, True), (7, False), (16, True)])
def test_policy_drives_real_selector_independently_of_env(monkeypatch, layer, selected):
    from hipengine.kernels.registry import KernelKey
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE', '0')
    monkeypatch.setattr(gl, 'is_registered', lambda key: True)
    parent = gl.GGUFLinearDispatch(
        KernelKey('hip_gfx1151', 'linear', 'gguf_q8_0',
                  'coltile8_rowbatch4_wave_scale_f32_f32_out'), 'raw')
    weight = SimpleNamespace(spec=SimpleNamespace(slot_path=f'layers.{layer}.attn_qkv'))
    with gl.q8_dense_wide_policy_session(gl.Q8DenseWidePolicy(True, frozenset({0, 16}))):
        actual = gl._q8_dense_wide_dispatch(parent, rows=273, in_features=2560,
                                            out_features=10240, weight=weight)
    assert (actual.key == KernelKey('hip_gfx1151', 'linear', 'gguf_q8_0',
                                   'dense_wide256_f32_f32_out')) is selected


def test_invalid_owner_policy_fails_loudly():
    with pytest.raises(TypeError, match='Q8DenseWidePolicy'):
        gl.resolve_q8_dense_wide_policy(SimpleNamespace(q8_dense_wide_policy='yes'))


@pytest.mark.parametrize('name', ['exact', 'wmma-16-47', 'wide-16-47'])
def test_dense_arm_binds_wide_owner_without_environment_writes(monkeypatch, name):
    from scripts.qwen4exp_dense_route_ab import ARMS, _apply_arm, DENSE_WIDE_ENV
    monkeypatch.setenv(DENSE_WIDE_ENV, 'ambient')
    prior = gl.Q8DenseWidePolicy(True, frozenset({2}))
    owner = SimpleNamespace(q8_dense_wide_policy=prior)
    arm = next(arm for arm in ARMS if arm.name == name)
    with _apply_arm(arm, owner):
        policy = gl.resolve_q8_dense_wide_policy(owner)
        assert policy.enabled is (name == 'wide-16-47')
        assert os.environ[DENSE_WIDE_ENV] == 'ambient'
    assert owner.q8_dense_wide_policy is prior
    assert policy.to_report() == {
        'enabled': name == 'wide-16-47',
        'layers': list(range(16, 48)) if name == 'wide-16-47' else None,
    }


def test_wide_policy_report_preserves_all_and_layer_zero():
    assert gl.Q8DenseWidePolicy(True, None).to_report() == {'enabled': True, 'layers': None}
    assert gl.Q8DenseWidePolicy(True, {0}).to_report() == {'enabled': True, 'layers': [0]}


@pytest.mark.parametrize('route_name', ['wmma', 'dense_wide', 'staged_f32'])
@pytest.mark.parametrize('candidate', [False, True])
def test_layer_gate_binds_wide_owner_and_restores(monkeypatch, route_name, candidate):
    from scripts import execution_profile_q8_wmma_prefill_layers_gate as gate
    monkeypatch.setenv(gate.DENSE_WIDE_ENV, 'ambient')
    owner = SimpleNamespace(q8_dense_wide_policy=gl.Q8DenseWidePolicy(True, None))
    prior = owner.q8_dense_wide_policy
    route = gate.SELECTORS[route_name]
    saved = {key: os.environ.get(key) for key in route.env_keys}
    try:
        with gate._apply_route_env(route, candidate=candidate, layers='0', generator=owner):
            policy = gl.resolve_q8_dense_wide_policy(owner)
            assert policy.enabled is (route_name == 'dense_wide' and candidate)
            assert os.environ[gate.DENSE_WIDE_ENV] == 'ambient'
    finally:
        gate._restore_env(saved)
    assert owner.q8_dense_wide_policy is prior
