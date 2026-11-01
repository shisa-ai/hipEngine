"""Campaign selected-down gate and counted leaf are both owner-local."""
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_halo_box_campaign_ab as campaign

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('mode', ['before', 'after'])
@pytest.mark.parametrize('prior', [None, False, True])
def test_campaign_selected_gate_scope(monkeypatch, mode, prior):
    monkeypatch.setenv(ENV, 'ambient')
    env = {ENV: 'custom'}
    owner = SimpleNamespace(q8_selected_down_enabled=prior, q8_down_variant='parent')
    with pytest.raises(RuntimeError, match='restore'):
        with campaign.route_scope('q8-wmma-down', mode, owner, environment=env):
            assert owner.q8_selected_down_enabled is (mode == 'after')
            assert owner.q8_down_variant == campaign.Q8_WMMA_DOWN_VARIANT
            assert env == {ENV: 'custom'}
            assert os.environ[ENV] == 'ambient'
            raise RuntimeError('restore')
    assert owner.q8_selected_down_enabled is prior
    assert owner.q8_down_variant == 'parent'
    campaign._apply_mode(mode, environment=env, route_package='q8-wmma-down')
    assert env == {ENV: 'custom'}
