"""Profile MMQ prepacking stays local to each generator and runner."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.runtime.qwen4_exp_runner import resolve_q8_mmq_prepack
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK'


@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_profile_owners_stay_independent_in_both_orders(monkeypatch, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for enabled in order:
        plan = _resolve(ExecutionProfile.PRODUCTION if enabled else ExecutionProfile.STRICT)
        generator, runner = _bind(plan)
        peers.append((generator, runner, enabled))
        assert os.environ[ENV] == ambient
        for owner, current, expected in peers:
            assert owner.q8_mmq_prepack_enabled is expected
            assert current.q8_mmq_prepack_enabled is expected
            assert resolve_q8_mmq_prepack(owner) is expected
        clone = _inherit_lane_decisions(generator, SimpleNamespace())
        assert clone.q8_mmq_prepack_enabled is enabled


def test_profile_gap_rejects_shadowed_legacy_override():
    from scripts.qwen4exp_profile_gap import _parse_overrides
    with pytest.raises(ValueError, match='q8_mmq_prepack_enabled'):
        _parse_overrides([ENV + '=0'])
