"""Plane diagnostic arms bind owners rather than mutate the process."""
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_halo_box_campaign_ab as campaign
from scripts import qwen4exp_q8_mmq_planes_logits_probe as probe

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES'


@pytest.mark.parametrize('mode,planes', [('before', 3), ('after', 2)])
def test_campaign_scope_binds_reports_and_restores(monkeypatch, mode, planes):
    monkeypatch.setenv(ENV, str(5 - planes))
    environment = {ENV: str(5 - planes), 'unrelated': 'keep'}
    prior = dict(environment)
    campaign._apply_mode(mode, environment=environment, route_package='q8-mmq-planes2')
    assert environment == prior
    owner = SimpleNamespace(q8_mmq_f32_planes=3)
    with campaign.route_scope('q8-mmq-planes2', mode, owner):
        assert owner.q8_mmq_f32_planes == planes
        record = campaign.mmq_planes_control_record(owner)
        assert record['effective_f32_planes'] == planes
        assert record['runner_field'] == 'q8_mmq_f32_planes'
    assert owner.q8_mmq_f32_planes == 3


@pytest.mark.parametrize('mode', ['before', 'after'])
def test_campaign_requires_owner(mode):
    with pytest.raises(ValueError, match='owning runner'):
        campaign.route_scope('q8-mmq-planes2', mode)


@pytest.mark.parametrize('arm,planes', [('incumbent', 3), ('candidate', 2)])
def test_probe_arm_scope_restores_after_failure(monkeypatch, arm, planes):
    monkeypatch.setenv(ENV, str(5 - planes))
    owner = SimpleNamespace(q8_mmq_f32_planes=None, runner=SimpleNamespace(q8_mmq_f32_planes=None))
    with pytest.raises(RuntimeError):
        with probe.plane_arm_scope(owner, arm):
            assert owner.q8_mmq_f32_planes == owner.runner.q8_mmq_f32_planes == planes
            raise RuntimeError('injected')
    assert owner.q8_mmq_f32_planes is owner.runner.q8_mmq_f32_planes is None


def test_probe_rejects_unknown_arm():
    with pytest.raises(ValueError, match='arm'):
        probe.plane_arm_scope(SimpleNamespace(), 'typo')
