"""Profile-gap binds exact Q4 overrides without process writes."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT'


@pytest.mark.parametrize('raw,expected', [('0', False), ('1', True), ('FALSE', True)])
def test_q4_override_owner_local(monkeypatch, raw, expected):
    ambient = '0' if expected else '1'
    monkeypatch.setenv(ENV, ambient)
    runner = SimpleNamespace(q4_iu8_exact_enabled=None)
    owner = SimpleNamespace(runner=runner, q4_iu8_exact_enabled=False)
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            report = gap.bind_q4_iu8_override(stack, owner, {ENV: raw})
            assert report == {'q4_iu8_exact_override': expected,
                              'q4_iu8_exact_enabled': expected}
            assert owner.q4_iu8_exact_enabled is runner.q4_iu8_exact_enabled is expected
            gap._apply_post_binder_overrides({ENV: raw})
            assert os.environ[ENV] == ambient
            raise RuntimeError('restore')
    assert owner.q4_iu8_exact_enabled is False
    assert runner.q4_iu8_exact_enabled is None


def test_q4_omission_keeps_owner(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(q4_iu8_exact_enabled=False)
    with ExitStack() as stack:
        assert gap.bind_q4_iu8_override(stack, owner, {}) == {
            'q4_iu8_exact_override': None, 'q4_iu8_exact_enabled': False}
    assert 'report.update(bind_q4_iu8_override(' in inspect.getsource(gap.main)
