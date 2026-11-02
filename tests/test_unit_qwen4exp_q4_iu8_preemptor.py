"""Q4 pair comparisons close exact IU8 through owner state."""
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_halo_box_campaign_ab as campaign

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT'


@pytest.mark.parametrize('enabled', [False, True])
def test_pair_closes_q4_owner_and_restores(monkeypatch, enabled):
    monkeypatch.setenv(ENV, '1')
    environment = {ENV: 'unchanged'}
    owner = SimpleNamespace(q4_iu8_exact_enabled=True, q4_pair_prefill=True,
        q4_bundle_prefill=False, q4_exact_grouped_gate=False,
        q4_exact_grouped_all=False, routed_f16_enabled=True,
        production_moe_prefill_enabled=True)
    prior = vars(owner).copy()
    with pytest.raises(RuntimeError, match='restore'):
        with campaign.q4_pair_scope(owner, enabled, environment=environment):
            assert owner.q4_iu8_exact_enabled is False
            assert owner.q4_pair_prefill is enabled
            assert environment[ENV] == 'unchanged'
            assert os.environ[ENV] == '1'
            raise RuntimeError('restore')
    assert vars(owner) == prior
    assert environment == {ENV: 'unchanged'}
    record = campaign.q4_pair_control_record(owner)
    assert record['forced_runner_fields']['q4_iu8_exact_enabled'] is False
    assert ENV not in record['forced_environment']
