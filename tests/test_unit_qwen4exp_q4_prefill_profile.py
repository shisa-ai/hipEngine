"""Profiles publish plain-Q4 prefill choices without process mutation."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation import qwen4_exp_profiles as profiles
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_q4_prefill_profile_isolation(monkeypatch, quant, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        resolved = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        owner, runner = _bind(resolved)
        expected = resolved.binder.__name__ == '_production_binder' and quant != 'gguf_ud_q4_k_xl'
        peers.append((owner, runner, expected))
        assert os.environ[ENV] == ambient
        for peer, current, value in peers:
            assert peer.q4_iu8_prefill_enabled is current.q4_iu8_prefill_enabled is value
        assert _inherit_lane_decisions(owner, SimpleNamespace()).q4_iu8_prefill_enabled is expected


def test_q4_prefill_rebinds_same_owner_and_keeps_ambient(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(runner=SimpleNamespace())
    for production, quant, expected in [(True, 'gguf_q4_k_m', True),
                                      (True, 'gguf_ud_q4_k_xl', False),
                                      (False, 'gguf_q4_k_m', False)]:
        profiles._publish_q4_prefill(owner, production=production, quant=quant)
        assert owner.q4_iu8_prefill_enabled is owner.runner.q4_iu8_prefill_enabled is expected
        assert os.environ[ENV] == '1'
