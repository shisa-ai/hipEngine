"""Profile-gap GR overrides preserve omitted owner axes and process inputs."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_profile_gap as gap
from hipengine.runtime.qwen4_exp_runner import GRIU8Policy

UP = 'HIPENGINE_QWEN4_EXP_GR_IU8'
DOWN = 'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN'


@pytest.mark.parametrize('overrides,expected', [({}, (False, True)),
    ({UP: '1'}, (True, True)), ({DOWN: '0'}, (False, False)),
    ({UP: 'false', DOWN: 'yes'}, (False, True))])
def test_gap_gr_policy_preserves_omitted_axes(monkeypatch, overrides, expected):
    monkeypatch.setenv(UP, 'ambient-up')
    monkeypatch.setenv(DOWN, 'ambient-down')
    prior = GRIU8Policy(False, True)
    owner = SimpleNamespace(gr_iu8_policy=prior, runner=SimpleNamespace(gr_iu8_policy=prior))
    gap._apply_post_binder_overrides(overrides)
    assert os.environ[UP] == 'ambient-up' and os.environ[DOWN] == 'ambient-down'
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            report = gap.bind_gr_iu8_override(stack, owner, overrides)
            assert report['gr_iu8_policy'] == dict(zip(('up', 'down'), expected))
            assert owner.gr_iu8_policy == GRIU8Policy(*expected)
            assert owner.runner.gr_iu8_policy == owner.gr_iu8_policy
            raise RuntimeError('restore')
    assert owner.gr_iu8_policy is owner.runner.gr_iu8_policy is prior


def test_gap_unbound_no_override_does_not_freeze_process_fallback(monkeypatch):
    monkeypatch.setenv(UP, '1')
    monkeypatch.setenv(DOWN, '0')
    owner = SimpleNamespace(gr_iu8_policy=None)
    with ExitStack() as stack:
        assert gap.bind_gr_iu8_override(stack, owner, {})['gr_iu8_policy'] == {'up': True, 'down': False}
        assert owner.gr_iu8_policy is None
