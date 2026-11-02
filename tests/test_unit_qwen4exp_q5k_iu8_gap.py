"""Diagnostic exact Q5_K overrides bind owners without process writes."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_profile_gap as gap
from scripts import qwen4exp_candidate_ar as ar

ENV = 'HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT'


@pytest.mark.parametrize('module', [gap, ar])
@pytest.mark.parametrize('raw,expected', [('', False), ('0', False), ('false', False),
                                        ('False', False), ('1', True), ('FALSE', True)])
def test_q5k_override_is_local_and_restoring(monkeypatch, module, raw, expected):
    ambient = '0' if expected else '1'
    monkeypatch.setenv(ENV, ambient)
    owner = SimpleNamespace(q5k_iu8_exact_enabled=False,
                            runner=SimpleNamespace(q5k_iu8_exact_enabled=None))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            assert module.bind_q5k_iu8_override(stack, owner, {ENV: raw}) == {
                'q5k_iu8_exact_override': expected, 'q5k_iu8_exact_enabled': expected}
            gap._apply_post_binder_overrides({ENV: raw})
            assert owner.q5k_iu8_exact_enabled is owner.runner.q5k_iu8_exact_enabled is expected
            assert os.environ[ENV] == ambient
            raise RuntimeError('restore')
    assert owner.q5k_iu8_exact_enabled is False
    assert owner.runner.q5k_iu8_exact_enabled is None


def test_q5k_omission_and_both_entrypoints(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(q5k_iu8_exact_enabled=False)
    with ExitStack() as stack:
        assert gap.bind_q5k_iu8_override(stack, owner, {}) == {
            'q5k_iu8_exact_override': None, 'q5k_iu8_exact_enabled': False}
    assert 'report.update(bind_q5k_iu8_override(' in inspect.getsource(gap.main)
    source = inspect.getsource(ar.main)
    assert source.index('split_q5k_iu8_override(overrides)') < source.index('os.environ.update(overrides)')
    assert "artifact['owner_overrides'].update(bind_q5k_iu8_override(" in source
    assert ar.split_q5k_iu8_override({ENV: '1', 'OTHER': 'x'}) == (
        {'OTHER': 'x'}, {ENV: '1'})
