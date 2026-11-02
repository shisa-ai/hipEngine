"""Profiles publish exact-Q5_1 enablement without process writes."""
import os
import inspect
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation import qwen4_exp_profiles as profiles
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind

ENV = 'HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT'


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_q51_profile_isolation(monkeypatch, quant, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        resolved = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        owner, runner = _bind(resolved)
        expected = resolved.binder.__name__ == '_production_binder'
        peers.append((owner, runner, expected))
        assert os.environ[ENV] == ambient
        for peer, current, value in peers:
            assert peer.q51_iu8_exact_enabled is current.q51_iu8_exact_enabled is value
        assert _inherit_lane_decisions(owner, SimpleNamespace()).q51_iu8_exact_enabled is expected


def test_state_gate_reads_q51_owner_premise():
    from scripts import qwen4exp_row4_state_gate as state
    source = inspect.getsource(state.main)
    assert 'assert resolve_q51_iu8_exact(generator.runner)' in source
    assert 'os.environ.get(Q51_IU8_EXACT_ENV' not in source
