"""Profile-gap binds exact Q5_1 overrides without process writes."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT'


@pytest.mark.parametrize('raw,expected', [('0', False), ('1', True), ('FALSE', True)])
def test_q51_override_owner_local(monkeypatch, raw, expected):
    ambient = '0' if expected else '1'
    monkeypatch.setenv(ENV, ambient)
    runner = SimpleNamespace(q51_iu8_exact_enabled=None)
    owner = SimpleNamespace(runner=runner, q51_iu8_exact_enabled=False)
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            report = gap.bind_q51_iu8_override(stack, owner, {ENV: raw})
            assert report == {'q51_iu8_exact_override': expected,
                              'q51_iu8_exact_enabled': expected}
            assert owner.q51_iu8_exact_enabled is runner.q51_iu8_exact_enabled is expected
            gap._apply_post_binder_overrides({ENV: raw})
            assert os.environ[ENV] == ambient
            raise RuntimeError('restore')
    assert owner.q51_iu8_exact_enabled is False
    assert runner.q51_iu8_exact_enabled is None


def test_q51_omission_keeps_owner(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(q51_iu8_exact_enabled=False)
    with ExitStack() as stack:
        assert gap.bind_q51_iu8_override(stack, owner, {}) == {
            'q51_iu8_exact_override': None, 'q51_iu8_exact_enabled': False}
    assert 'report.update(bind_q51_iu8_override(' in inspect.getsource(gap.main)
