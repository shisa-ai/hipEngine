"""GR exact-composite scopes close IU8 up on owners, preserving down."""
from types import SimpleNamespace
import os
import pytest
from hipengine.runtime.qwen4_exp_runner import GRIU8Policy
from scripts import qwen4exp_halo_box_campaign_ab as campaign


@pytest.mark.parametrize('package', ['gr-wave-scale', 'gr-dpp-reduction'])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('down', [False, True])
def test_gr_preemptor_scope_preserves_down_and_ambient(monkeypatch, package, enabled, down):
    key = 'HIPENGINE_QWEN4_EXP_GR_IU8'
    monkeypatch.setenv(key, '1')
    environment = {key: 'custom'}
    prior = GRIU8Policy(True, down)
    owner = SimpleNamespace(gr_iu8_policy=prior, gr_wide=True,
                            gr_dpp_reduction=True, gr_wave_scale=True)
    with pytest.raises(RuntimeError, match='restore'):
        with campaign.route_scope(package, 'after' if enabled else 'before', owner, environment=environment):
            assert owner.gr_iu8_policy == GRIU8Policy(False, down)
            assert environment == {key: 'custom'} and os.environ[key] == '1'
            assert owner.gr_wide is False
            assert owner.gr_dpp_reduction is (enabled if package == 'gr-dpp-reduction' else False)
            assert owner.gr_wave_scale is (True if package == 'gr-dpp-reduction' else enabled)
            raise RuntimeError('restore')
    assert owner.gr_iu8_policy is prior
    assert owner.gr_wide is owner.gr_dpp_reduction is owner.gr_wave_scale is True
    record = (campaign.gr_wave_control_record(owner) if package == 'gr-wave-scale'
              else campaign.gr_dpp_control_record(owner))
    assert record['forced_environment'] == {}
    assert record['gr_iu8_preemptor'] == {
        'runner_field': 'gr_iu8_policy', 'up': False, 'down': 'preserve',
        'published_value': {'up': True, 'down': down}}
