"""Q5_1 diagnostic preemptors close the owner without process writes."""
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_halo_box_campaign_ab as campaign

ENV = 'HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT'


@pytest.mark.parametrize('package', ['pair2', 'pf13'])
@pytest.mark.parametrize('prior', [None, True, False])
def test_preemptor_owner_closes_and_restores(monkeypatch, package, prior):
    monkeypatch.setenv(ENV, '1')
    environment = {ENV: '1'}
    owner = SimpleNamespace(q51_iu8_exact_enabled=prior, q51_exact_grouped_down=True,
                            q4_exact_grouped_gate=True, q4_exact_grouped_all=True,
                            q51_down_m1=True, forkb_grouped_down=True)
    scope = (campaign.q51_pair2_scope(runner=owner, environment=environment)
             if package == 'pair2' else campaign.pf13_scope('after', owner, environment=environment))
    with pytest.raises(RuntimeError, match='restore'):
        with scope:
            assert owner.q51_iu8_exact_enabled is False
            assert environment == {ENV: '1'} and os.environ[ENV] == '1'
            raise RuntimeError('restore')
    assert owner.q51_iu8_exact_enabled is prior
    assert environment == {ENV: '1'}


def test_preemptor_records_owner_not_environment():
    owner = SimpleNamespace(q51_iu8_exact_enabled=True)
    record = campaign.q51_pair2_control_record(owner, environment={ENV: '0'})
    assert record['preempting_owner_fields_closed']['q51_iu8_exact_enabled'] is False
    assert record['preempting_owner_fields_previous']['q51_iu8_exact_enabled'] is True
    assert ENV not in record['preempting_owners_closed']
