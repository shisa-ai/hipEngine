"""Candidate AR GR overrides use the paired owner policy at both write stages."""
from contextlib import ExitStack
from types import SimpleNamespace
import inspect
import os
import pytest
from scripts import qwen4exp_candidate_ar as ar
from hipengine.runtime.qwen4_exp_runner import GRIU8Policy

UP = 'HIPENGINE_QWEN4_EXP_GR_IU8'
DOWN = 'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN'


@pytest.mark.parametrize('overrides,expected', [({}, (False, True)),
    ({UP: 'yes'}, (True, True)), ({DOWN: 'false'}, (False, False)),
    ({UP: 'False', DOWN: '1'}, (False, True))])
def test_ar_gr_pair_split_and_bind(monkeypatch, overrides, expected):
    monkeypatch.setenv(UP, 'ambient-up')
    monkeypatch.setenv(DOWN, 'ambient-down')
    environment, pair = ar.split_gr_iu8_overrides({**overrides, 'OTHER': '1'})
    assert environment == {'OTHER': '1'}
    assert pair == overrides
    prior = GRIU8Policy(False, True)
    owner = SimpleNamespace(gr_iu8_policy=prior, runner=SimpleNamespace(gr_iu8_policy=prior))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            report = ar.bind_gr_iu8_override(stack, owner, pair)
            assert report['gr_iu8_policy'] == dict(zip(('up', 'down'), expected))
            assert owner.gr_iu8_policy == owner.runner.gr_iu8_policy == GRIU8Policy(*expected)
            assert os.environ[UP] == 'ambient-up' and os.environ[DOWN] == 'ambient-down'
            raise RuntimeError('restore')
    assert owner.gr_iu8_policy is owner.runner.gr_iu8_policy is prior


def test_ar_gr_pair_main_wiring():
    source = inspect.getsource(ar.main)
    assert source.index('split_gr_iu8_overrides(overrides)') < source.index('os.environ.update(overrides)')
    assert source.index('update(bind_gr_iu8_override(') < source.index('for warmup in range')
    assert ar.split_gr_iu8_overrides({}) == ({}, {})
