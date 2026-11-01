"""Profile-gap transports selected Q8 down overrides through explicit owners."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('raw,expected', [('0', False), ('1', True), ('FALSE', True)])
def test_selected_down_override_is_owner_local(monkeypatch, raw, expected):
    ambient = '0' if expected else '1'
    monkeypatch.setenv(ENV, ambient)
    runner = SimpleNamespace(q8_selected_down_enabled=None)
    owner = SimpleNamespace(runner=runner, q8_selected_down_enabled=False)
    with ExitStack() as stack:
        report = gap.bind_selected_q8_down_override(stack, owner, {ENV: raw})
        assert report == {'q8_selected_down_override': expected,
                          'q8_selected_down_enabled': expected}
        assert owner.q8_selected_down_enabled is expected
        assert runner.q8_selected_down_enabled is expected
        gap._apply_post_binder_overrides({ENV: raw})
        assert os.environ[ENV] == ambient
    assert owner.q8_selected_down_enabled is False
    assert runner.q8_selected_down_enabled is None


def test_selected_down_omission_reports_effective_owner(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(q8_selected_down_enabled=False)
    with ExitStack() as stack:
        report = gap.bind_selected_q8_down_override(stack, owner, {})
        assert report == {'q8_selected_down_override': None,
                          'q8_selected_down_enabled': False}
    assert 'report.update(bind_selected_q8_down_override(' in inspect.getsource(gap.main)
