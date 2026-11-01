"""Row4 state gate keeps selected Q8 controls owner-local."""
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_row4_state_gate as gate

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('package', ['q8-wmma-down', 'q5k-iu8-exact'])
@pytest.mark.parametrize('enabled', ['0', '1'])
def test_state_gate_selected_transport(monkeypatch, package, enabled):
    monkeypatch.setenv(ENV, 'ambient')
    env = {ENV: 'custom'}
    owner = SimpleNamespace(q8_selected_down_enabled=True, q8_down_variant='parent')
    flag = '' if package == 'q8-wmma-down' else 'HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT'
    gate.apply_state_gate_mode(owner, package, enabled, flag, environment=env)
    assert env[ENV] == 'custom'
    with pytest.raises(RuntimeError, match='restore'):
        with gate.arm_control(package, enabled, owner, environment=env):
            assert owner.q8_selected_down_enabled is (enabled == '1' if package == 'q8-wmma-down' else False)
            assert env[ENV] == 'custom'
            assert os.environ[ENV] == 'ambient'
            raise RuntimeError('restore')
    assert owner.q8_selected_down_enabled is True
    assert owner.q8_down_variant == 'parent'
