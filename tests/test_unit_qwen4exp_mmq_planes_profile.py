"""Named profiles bind three F32 MMQ planes independently of ambient input."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES'


@pytest.mark.parametrize('ambient', ['2', '3', 'invalid'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_profile_owners_publish_parent_planes_in_both_orders(monkeypatch, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        plan = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT)
        generator, runner = _bind(plan)
        peers.append((generator, runner))
        assert os.environ[ENV] == ambient
        for owner, current in peers:
            assert owner.q8_mmq_f32_planes == current.q8_mmq_f32_planes == 3
        assert _inherit_lane_decisions(generator, SimpleNamespace()).q8_mmq_f32_planes == 3
