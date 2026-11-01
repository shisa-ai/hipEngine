"""Profile-bound attention policy is independent of stale process input."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE'


@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_attention_profile_owner_isolation(monkeypatch, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        generator, runner = _bind(_resolve(
            ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT))
        peers.append((generator, runner))
        assert os.environ[ENV] == ambient
        for owner, current in peers:
            assert owner.q8_mmq_attn_gate_enabled is False
            assert current.q8_mmq_attn_gate_enabled is False
        clone = _inherit_lane_decisions(generator, SimpleNamespace())
        assert clone.q8_mmq_attn_gate_enabled is False
