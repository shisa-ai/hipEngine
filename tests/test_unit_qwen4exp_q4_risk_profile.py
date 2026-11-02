"""Profile Q4 risk multipliers preserve the binder's existing 1.0 value."""
import inspect
import os
from types import SimpleNamespace

import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT'


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['0.5', '8', 'invalid'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_multiplier_profile_isolation(monkeypatch, quant, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        resolved = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        owner, runner = _bind(resolved)
        peers.append((owner, runner))
        assert os.environ[ENV] == ambient
        for peer, current in peers:
            assert peer.q4_iu8_risk_mult == current.q4_iu8_risk_mult == 1.0
        assert _inherit_lane_decisions(owner, SimpleNamespace()).q4_iu8_risk_mult == 1.0


def test_multiplier_publication_keeps_ambient_and_does_not_write_selector(monkeypatch):
    monkeypatch.setenv(ENV, '8')
    owner = SimpleNamespace(runner=SimpleNamespace())
    profiles._publish_q4_risk_mult(owner)
    assert owner.q4_iu8_risk_mult == owner.runner.q4_iu8_risk_mult == 1.0
    assert os.environ[ENV] == '8'
    assert '"HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT":' not in inspect.getsource(profiles._bind)
