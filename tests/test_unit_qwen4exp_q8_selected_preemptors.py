"""Campaign preemptor closures act on the selected-down owner."""
from types import SimpleNamespace
import os
import pytest
from hipengine.runtime.qwen4_exp_runner import ExpertGridPolicy
from scripts import qwen4exp_halo_box_campaign_ab as campaign

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('kind', ['sorted', 'mapped', 'pf13'])
@pytest.mark.parametrize('prior', [None, False, True])
def test_selected_preemptor_scope(monkeypatch, kind, prior):
    monkeypatch.setenv(ENV, 'ambient')
    env = {ENV: 'custom', 'HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID': 'stale'}
    fields = {field: True for field in campaign.PF13_GATE_FIELDS}
    fields.update({campaign.Q8_DOWN_FORKB_FIELD: True,
                   campaign.Q8_MAPPED_DOWN_MAP_GATE: True,
                   'q8_selected_down_enabled': prior, 'routed_f16_enabled': True,
                   'expert_grid_policy': ExpertGridPolicy(True, True)})
    owner = SimpleNamespace(**fields)
    if kind == 'sorted':
        scope = campaign.q8_down_scope(runner=owner, environment=env)
        record = campaign.q8_down_control_record(runner=owner, environment=env)
    elif kind == 'mapped':
        scope = campaign.q8_mapped_down_scope(runner=owner, environment=env)
        record = campaign.q8_mapped_down_control_record(runner=owner, environment=env)
    else:
        scope = campaign.pf13_scope('before', owner, environment=env)
        record = campaign.pf13_control_record(owner, environment=env)
    with pytest.raises(RuntimeError, match='restore'):
        with scope:
            assert owner.q8_selected_down_enabled is False
            assert owner.routed_f16_enabled is False
            assert env[ENV] == 'custom'
            assert os.environ[ENV] == 'ambient'
            raise RuntimeError('restore')
    assert owner.q8_selected_down_enabled is prior
    assert owner.routed_f16_enabled is True
    assert env == {ENV: 'custom', 'HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID': 'stale'}
    assert record['selected_down_preemptor'] == {
        'runner_field': 'q8_selected_down_enabled', 'bound_value': False,
        'published_value': prior}
    assert ENV not in campaign.Q8_DOWN_PREEMPTING_ENV
    assert ENV not in campaign.Q8_MAPPED_DOWN_PREEMPTING_ENV
    assert ENV not in campaign.PF13_PREEMPTING_ENV
