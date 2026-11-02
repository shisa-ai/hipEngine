"""Profile plane publication captures process input without changing selection."""
import inspect
import os
from types import SimpleNamespace

import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES'


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient,expected', [('2', 2), ('3', 3), ('invalid', 3), (None, 3)])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_plane_profile_isolation(monkeypatch, quant, ambient, expected, order):
    if ambient is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        resolved = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        owner, runner = _bind(resolved)
        peers.append((owner, runner, expected))
        assert os.environ.get(ENV) == ambient
        for peer, current, selected in peers:
            assert peer.q4_iu8_planes == current.q4_iu8_planes == selected
        assert _inherit_lane_decisions(owner, SimpleNamespace()).q4_iu8_planes == expected
        ambient = '3' if expected == 2 else '2'
        expected = 3 if expected == 2 else 2
        monkeypatch.setenv(ENV, ambient)


def test_plane_publication_recaptures_input_without_process_writes(monkeypatch):
    owner = SimpleNamespace(q4_iu8_planes=3, runner=SimpleNamespace(q4_iu8_planes=3))
    monkeypatch.setenv(ENV, '2')
    profiles._publish_q4_planes(owner)
    assert owner.q4_iu8_planes == owner.runner.q4_iu8_planes == 2
    assert os.environ[ENV] == '2'
    monkeypatch.setenv(ENV, 'invalid')
    profiles._publish_q4_planes(owner)
    assert owner.q4_iu8_planes == owner.runner.q4_iu8_planes == 3
    assert os.environ[ENV] == 'invalid'
    assert '"HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES":' not in inspect.getsource(profiles._bind)
