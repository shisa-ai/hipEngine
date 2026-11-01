"""Per-layer selected-down intervention changes call state, not the process."""
import inspect
import os
import pytest
from scripts import qwen4exp_q8_localize as probe

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('prior', [None, False, True])
@pytest.mark.parametrize('disabled_layer', [None, '2', '4'])
def test_layer_override_preserves_other_calls(monkeypatch, prior, disabled_layer):
    monkeypatch.setenv(ENV, 'ambient')
    original = {'q8_selected_down_enabled': prior, 'other': object()}
    result = probe.selected_down_layer_kwargs(
        original, active_layer='2', disabled_layer=disabled_layer)
    assert result['q8_selected_down_enabled'] is (False if disabled_layer == '2' else prior)
    assert result['other'] is original['other']
    assert original['q8_selected_down_enabled'] is prior
    assert os.environ[ENV] == 'ambient'


def test_layer_override_does_not_invent_an_unbound_gate():
    assert probe.selected_down_layer_kwargs({}, active_layer='2', disabled_layer='4') == {}


def test_layer_wrapper_uses_call_override():
    source = inspect.getsource(probe.main)
    assert 'selected_down_layer_kwargs(' in source
    assert 'os.environ[flag]' not in source
    assert 'localize.set_flags({flag:' not in source
