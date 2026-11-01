"""Prepack A/B controls bind owners and leave the process environment alone."""
from types import SimpleNamespace
import pytest
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, route_scope
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode, arm_control

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK'


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('interface', ['campaign', 'state_gate'])
def test_prepack_arm_uses_owner_and_restores(interface, enabled):
    env = {ENV: '0' if enabled else '1'}
    original = env.copy()
    owner = SimpleNamespace(q8_mmq_prefill_enabled=None, q8_mmq_prepack_enabled=None,
                            _q8_mmq_buffers=(), _q8_mmq_weight_sidecars=None)
    def configure():
        assert owner.q8_mmq_prefill_enabled is True
        owner._q8_mmq_buffers = (object(),)
        if owner.q8_mmq_prepack_enabled:
            owner._q8_mmq_weight_sidecars = SimpleNamespace(mapping={(1,2,3): (4,5)})
    owner.configure_mmq_prefill_resources = configure
    mode = 'after' if enabled else 'before'
    if interface == 'campaign':
        _apply_mode(mode, route_package='q8-mmq-prepack', environment=env)
        scope = route_scope('q8-mmq-prepack', mode, owner, environment=env)
    else:
        apply_state_gate_mode(owner, 'q8-mmq-prepack', str(int(enabled)), '', environment=env)
        scope = arm_control('q8-mmq-prepack', str(int(enabled)), owner, environment=env)
    with pytest.raises(RuntimeError, match='injected'):
        with scope:
            assert env == original
            assert owner.q8_mmq_prepack_enabled is enabled
            assert owner.q8_mmq_prefill_enabled is True
            assert owner._q8_mmq_buffers
            raise RuntimeError('injected')
    assert env == original
    assert owner.q8_mmq_prefill_enabled is owner.q8_mmq_prepack_enabled is None


def test_control_record_names_shared_resources_and_owner():
    from scripts.qwen4exp_halo_box_campaign_ab import mmq_prepack_control_record
    record = mmq_prepack_control_record(SimpleNamespace(q8_mmq_prepack_enabled=False))
    assert record['runner_field'] == 'q8_mmq_prepack_enabled'
    assert record['published_value'] is False
    assert record['before'] is False and record['after'] is True
    assert record['prerequisite_owner'] == {'q8_mmq_prefill_enabled': True}
    assert record['prerequisite_environment'] == {}


def test_missing_owner_fails_loudly():
    with pytest.raises(ValueError, match='owning runner'):
        route_scope('q8-mmq-prepack', 'after')
