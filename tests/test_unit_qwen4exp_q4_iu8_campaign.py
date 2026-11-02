"""Campaign Q4 IU8 arms follow their owner, not process inputs."""
from types import SimpleNamespace
import inspect
import os
import pytest
from scripts import qwen4exp_halo_box_campaign_ab as campaign
from scripts import qwen4exp_row4_state_gate as state

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT'


@pytest.mark.parametrize('interface', ['campaign', 'state'])
@pytest.mark.parametrize('mode', ['before', 'after'])
def test_q4_arm_owner_and_restoration(monkeypatch, interface, mode):
    monkeypatch.setenv(ENV, 'ambient')
    environment = {ENV: 'unchanged'}
    owner = SimpleNamespace(q4_iu8_exact_enabled=None)
    campaign._apply_mode(mode, route_package='q4-iu8-exact', environment=environment)
    state.apply_state_gate_mode(owner, 'q4-iu8-exact', '1', ENV, environment=environment)
    assert environment == {ENV: 'unchanged'}
    scope = (campaign.route_scope('q4-iu8-exact', mode, owner) if interface == 'campaign'
             else state.arm_control('q4-iu8-exact', '1' if mode == 'after' else '0', owner))
    with pytest.raises(RuntimeError, match='restore'):
        with scope:
            assert owner.q4_iu8_exact_enabled is (mode == 'after')
            assert os.environ[ENV] == 'ambient'
            raise RuntimeError('restore')
    assert owner.q4_iu8_exact_enabled is None


def test_q4_control_record_and_wiring():
    owner = SimpleNamespace(q4_iu8_exact_enabled=False)
    record = campaign.q4_iu8_control_record(owner)
    assert record['runner_field'] == 'q4_iu8_exact_enabled'
    assert record['published_value'] is False
    assert record['before'] is False and record['after'] is True
    for module in (campaign, state):
        source = inspect.getsource(module.main)
        assert 'q4_iu8_control_record(' in source
        assert 'resolve_q4_iu8_exact(' in source
    with pytest.raises(ValueError, match='owning runner'):
        campaign.route_scope('q4-iu8-exact', 'after')
    with pytest.raises(ValueError, match='invalid'):
        campaign.route_scope('q4-iu8-exact', 'invalid', owner)
