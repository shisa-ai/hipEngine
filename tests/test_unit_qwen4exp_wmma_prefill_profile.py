"""Profiles keep generic WMMA decisions independent of ambient input."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation.qwen4_exp_profiles import QWEN4_EXP_QUANTS
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

ENV = 'HIPENGINE_GGUF_WMMA_PREFILL'


@pytest.mark.parametrize('quant', QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_generic_wmma_profile_owner_isolation(monkeypatch, quant, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        generator, runner = _bind(_resolve(
            ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT,
            quant=quant))
        peers.append((generator, runner))
        assert os.environ[ENV] == ambient
        for owner, current in peers:
            assert owner.gguf_wmma_prefill_enabled is False
            assert current.gguf_wmma_prefill_enabled is False
        clone = _inherit_lane_decisions(generator, SimpleNamespace())
        assert clone.gguf_wmma_prefill_enabled is False
