"""Diagnostic Q4 multipliers bind owners without process writes."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace

import pytest
from scripts import qwen4exp_profile_gap as gap
from scripts import qwen4exp_candidate_ar as ar

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT'


@pytest.mark.parametrize('module', [gap, ar])
@pytest.mark.parametrize('value', ['0.5', '1', '4', '16'])
def test_multiplier_override_is_local_and_restoring(monkeypatch, module, value):
    monkeypatch.setenv(ENV, '8')
    owner = SimpleNamespace(q4_iu8_risk_mult=4.0,
                            runner=SimpleNamespace(q4_iu8_risk_mult=2.0))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            assert module.bind_q4_iu8_risk_override(stack, owner, {ENV: value}) == {
                'q4_iu8_risk_mult_override': value, 'q4_iu8_risk_mult': float(value)}
            gap._apply_post_binder_overrides({ENV: value})
            assert owner.q4_iu8_risk_mult == owner.runner.q4_iu8_risk_mult == float(value)
            assert os.environ[ENV] == '8'
            raise RuntimeError('restore')
    assert owner.q4_iu8_risk_mult == 4.0
    assert owner.runner.q4_iu8_risk_mult == 2.0


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'bad'])
def test_invalid_override_fails_without_losing_owner_state(value):
    owner = SimpleNamespace(q4_iu8_risk_mult=4.0)
    with pytest.raises(ValueError, match='must be a positive float'):
        with ExitStack() as stack:
            gap.bind_q4_iu8_risk_override(stack, owner, {ENV: value})
    assert owner.q4_iu8_risk_mult == 4.0


def test_omission_and_both_entrypoints():
    owner = SimpleNamespace(q4_iu8_risk_mult=1.0)
    with ExitStack() as stack:
        assert gap.bind_q4_iu8_risk_override(stack, owner, {}) == {
            'q4_iu8_risk_mult_override': None, 'q4_iu8_risk_mult': 1.0}
    assert 'report.update(bind_q4_iu8_risk_override(' in inspect.getsource(gap.main)
    source = inspect.getsource(ar.main)
    assert source.index('split_q4_iu8_risk_override(overrides)') < source.index('os.environ.update(overrides)')
    assert "artifact['owner_overrides'].update(bind_q4_iu8_risk_override(" in source
    assert ar.split_q4_iu8_risk_override({ENV: '1', 'OTHER': 'x'}) == (
        {'OTHER': 'x'}, {ENV: '1'})
