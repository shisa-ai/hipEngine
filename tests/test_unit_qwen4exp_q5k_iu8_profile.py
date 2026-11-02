"""Profiles publish exact-Q5_K enablement without process writes."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation import qwen4_exp_profiles as profiles
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind

ENV = 'HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT'


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_q5k_profile_isolation(monkeypatch, quant, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        resolved = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        owner, runner = _bind(resolved)
        expected = resolved.binder.__name__ == '_production_binder'
        peers.append((owner, runner, expected))
        assert os.environ[ENV] == ambient
        for peer, current, value in peers:
            assert peer.q5k_iu8_exact_enabled is current.q5k_iu8_exact_enabled is value
        assert _inherit_lane_decisions(owner, SimpleNamespace()).q5k_iu8_exact_enabled is expected
        monkeypatch.setenv(ENV, '0' if ambient == '1' else '1')
        assert owner.q5k_iu8_exact_enabled is expected
        monkeypatch.setenv(ENV, ambient)
