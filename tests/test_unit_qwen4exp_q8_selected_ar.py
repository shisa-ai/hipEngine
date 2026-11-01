"""Candidate AR separates selected Q8 down owner input from process writes."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_candidate_ar as ar

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('raw,expected', [('0', False), ('1', True), ('FALSE', True)])
def test_ar_selected_down_split_and_bind(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV, 'ambient')
    environment, selected = ar.split_selected_q8_down_override({ENV: raw, 'OTHER': '1'})
    assert environment == {'OTHER': '1'}
    assert selected == {ENV: raw}
    owner = SimpleNamespace(q8_selected_down_enabled=None,
                            runner=SimpleNamespace(q8_selected_down_enabled=False))
    with ExitStack() as stack:
        assert ar.bind_selected_q8_down_override(stack, owner, selected) == {
            'q8_selected_down_override': expected, 'q8_selected_down_enabled': expected}
        assert owner.q8_selected_down_enabled is owner.runner.q8_selected_down_enabled is expected
        assert os.environ[ENV] == 'ambient'
    assert owner.q8_selected_down_enabled is None
    assert owner.runner.q8_selected_down_enabled is False


def test_ar_selected_down_main_wiring():
    source = inspect.getsource(ar.main)
    assert 'split_selected_q8_down_override(overrides)' in source
    assert 'update(bind_selected_q8_down_override(' in source
    assert ar.split_selected_q8_down_override({}) == ({}, {})
