"""Candidate AR keeps exact Q5_1 owner inputs out of process writes."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_candidate_ar as ar

ENV = 'HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT'


@pytest.mark.parametrize('raw,expected', [('0', False), ('1', True), ('FALSE', True)])
def test_ar_q51_split_and_bind(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV, 'ambient')
    environment, selected = ar.split_q51_iu8_override({ENV: raw, 'OTHER': '1'})
    assert environment == {'OTHER': '1'}
    assert selected == {ENV: raw}
    owner = SimpleNamespace(q51_iu8_exact_enabled=None,
                            runner=SimpleNamespace(q51_iu8_exact_enabled=False))
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            assert ar.bind_q51_iu8_override(stack, owner, selected) == {
                'q51_iu8_exact_override': expected, 'q51_iu8_exact_enabled': expected}
            assert owner.q51_iu8_exact_enabled is owner.runner.q51_iu8_exact_enabled is expected
            assert os.environ[ENV] == 'ambient'
            raise RuntimeError('restore')
    assert owner.q51_iu8_exact_enabled is None
    assert owner.runner.q51_iu8_exact_enabled is False


def test_ar_q51_main_wiring():
    source = inspect.getsource(ar.main)
    assert source.index('split_q51_iu8_override(overrides)') < source.index('os.environ.update(overrides)')
    assert 'update(bind_q51_iu8_override(' in source
    assert ar.split_q51_iu8_override({}) == ({}, {})
