"""GR profiles publish independent projection choices without process writes."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.runtime.qwen4_exp_runner import GRIU8Policy
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

KEYS = ('HIPENGINE_QWEN4_EXP_GR_IU8', 'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN')


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_gr_profile_owner_isolation(monkeypatch, quant, ambient, order):
    for key in KEYS:
        monkeypatch.setenv(key, ambient)
    peers = []
    for production in order:
        resolved = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        owner, runner = _bind(resolved)
        enabled = resolved.binder.__name__ == '_production_binder' and quant != 'gguf_ud_q4_k_xl'
        expected = GRIU8Policy(enabled, enabled)
        peers.append((owner, runner, expected))
        assert {os.environ[key] for key in KEYS} == {ambient}
        for peer, current, value in peers:
            assert peer.gr_iu8_policy == current.gr_iu8_policy == value
        assert _inherit_lane_decisions(owner, SimpleNamespace()).gr_iu8_policy == expected


@pytest.mark.parametrize('flag,expected', [('GR_IU8', GRIU8Policy(False, True)),
                                         ('GR_IU8_DOWN', GRIU8Policy(True, False))])
def test_gr_recovery_axes_are_independent(monkeypatch, flag, expected):
    monkeypatch.setattr(profiles, 'PRODUCTION_ARITHMETIC_RECOVERY_FLAGS', (flag,))
    owner, runner = _bind(_resolve(ExecutionProfile.PRODUCTION))
    assert owner.gr_iu8_policy == runner.gr_iu8_policy == expected
    peer = SimpleNamespace(runner=SimpleNamespace())
    profiles._publish_gr_iu8(peer, production=True, quant='gguf_q4_k_m')
    assert peer.gr_iu8_policy == peer.runner.gr_iu8_policy == GRIU8Policy(True, True)
