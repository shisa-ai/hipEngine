"""Campaign and state-gate GR controls bind only the requested projection."""
from types import SimpleNamespace
import os
import inspect
import pytest
from hipengine.runtime.qwen4_exp_runner import GRIU8Policy
from scripts import qwen4exp_halo_box_campaign_ab as campaign
from scripts import qwen4exp_row4_state_gate as state


@pytest.mark.parametrize('package', ['gr-iu8', 'gr-iu8-down'])
@pytest.mark.parametrize('mode', ['before', 'after'])
@pytest.mark.parametrize('interface', ['campaign', 'state'])
def test_gr_route_preserves_other_projection_and_restores(monkeypatch, package, mode, interface):
    up_key, down_key = 'HIPENGINE_QWEN4_EXP_GR_IU8', 'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN'
    monkeypatch.setenv(up_key, 'ambient-up')
    monkeypatch.setenv(down_key, 'ambient-down')
    environment = {up_key: 'untouched-up', down_key: 'untouched-down'}
    prior = GRIU8Policy(True, True)
    owner = SimpleNamespace(gr_iu8_policy=prior)
    campaign._apply_mode(mode, route_package=package, environment=environment)
    state.apply_state_gate_mode(owner, package, '1' if mode == 'after' else '0', up_key, environment=environment)
    assert environment == {up_key: 'untouched-up', down_key: 'untouched-down'}
    scope = (campaign.route_scope(package, mode, owner) if interface == 'campaign' else
             state.arm_control(package, '1' if mode == 'after' else '0', owner))
    with pytest.raises(RuntimeError, match='restore'):
        with scope:
            expected = GRIU8Policy(mode == 'after', True) if package == 'gr-iu8' else GRIU8Policy(True, mode == 'after')
            assert owner.gr_iu8_policy == expected
            raise RuntimeError('restore')
    assert owner.gr_iu8_policy is prior
    assert os.environ[up_key] == 'ambient-up' and os.environ[down_key] == 'ambient-down'


def test_gr_route_record_and_cleanup():
    owner = SimpleNamespace(gr_iu8_policy=GRIU8Policy(True, False))
    record = campaign.gr_iu8_control_record('gr-iu8', owner)
    assert record['runner_field'] == 'gr_iu8_policy'
    assert record['axis'] == 'up'
    assert record['published_value'] == {'up': True, 'down': False}
    for module in (campaign, state):
        source = inspect.getsource(module.main)
        assert 'gr_iu8_control_record(' in source
        assert 'gr_iu8_policy' in source
    source = inspect.getsource(state.main)
    assert '"q5k-iu8-exact", "gr-iu8", "gr-iu8-down"' not in source
    assert 'os.environ["HIPENGINE_QWEN4_EXP_GR_IU8"] =' not in inspect.getsource(campaign.main)
    assert 'os.environ["HIPENGINE_QWEN4_EXP_GR_IU8_DOWN"] =' not in inspect.getsource(campaign.main)
