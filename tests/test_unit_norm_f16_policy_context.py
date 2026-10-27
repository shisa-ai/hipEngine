"""Norm-side publication must be scoped independently of legacy environment."""
import pytest
from hipengine.runtime import gguf_linear as linear


def test_publication_policy_scope_restores_on_error(monkeypatch):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_NORM_F16_PUBLISH', '1')
    assert not linear.wide_f16_activation_publication_enabled()
    with pytest.raises(RuntimeError):
        with linear.norm_f16_publication_session(True):
            assert linear.wide_f16_activation_publication_enabled()
            with linear.norm_f16_publication_session(False):
                assert not linear.wide_f16_activation_publication_enabled()
            raise RuntimeError('scope failed')
    assert not linear.wide_f16_activation_publication_enabled()


def test_diagnostic_publication_scope_restores_runner_on_error():
    from types import SimpleNamespace
    from scripts.qwen4exp_norm_f16_gate_support import publication_scope

    owner = SimpleNamespace(norm_f16_publication_enabled=True)
    with pytest.raises(RuntimeError):
        with publication_scope(owner, False):
            assert not owner.norm_f16_publication_enabled
            raise RuntimeError('arm failed')
    assert owner.norm_f16_publication_enabled
