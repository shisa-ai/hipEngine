"""The selected-Q8 probe binds the gate and its candidate leaf locally."""
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_q8_0_wmma_down_logits_probe as probe
from hipengine.runtime.qwen4_exp_runner import Q8_DOWN_WMMA_VARIANT


@pytest.mark.parametrize('enabled', [False, True])
def test_probe_owner_scope_preserves_process_and_restores(monkeypatch, enabled):
    monkeypatch.setenv(probe.FLAG, 'ambient')
    owner = SimpleNamespace(q8_selected_down_enabled=None, q8_down_variant='prior',
        runner=SimpleNamespace(q8_selected_down_enabled=False, q8_down_variant='runner-prior'))
    with pytest.raises(RuntimeError):
        with probe.selected_down_probe_scope(owner, enabled) as report:
            assert report['enabled'] is enabled
            assert report['variant'] == (Q8_DOWN_WMMA_VARIANT if enabled else 'prior')
            assert owner.q8_selected_down_enabled is owner.runner.q8_selected_down_enabled is enabled
            assert os.environ[probe.FLAG] == 'ambient'
            raise RuntimeError('restore')
    assert owner.q8_selected_down_enabled is None
    assert owner.runner.q8_selected_down_enabled is False
    assert owner.q8_down_variant == 'prior'
    assert owner.runner.q8_down_variant == 'runner-prior'


def test_probe_main_binds_scope_and_closes_generator():
    source = inspect.getsource(probe.main)
    assert 'selected_down_probe_scope(generator, enabled == "1")' in source
    assert 'os.environ[FLAG]' not in source
    assert 'generator.close()' in source
