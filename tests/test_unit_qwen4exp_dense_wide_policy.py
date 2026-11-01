"""Wide prefill ownership is explicit while unbound diagnostics keep env fallback."""
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
