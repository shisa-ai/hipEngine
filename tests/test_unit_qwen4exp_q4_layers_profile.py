"""Profile Q4 layer scopes are owner-local and preserve the existing suffix."""
import inspect
import os
from types import SimpleNamespace

import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS'
SUFFIX = ','.join(map(str, profiles.PRODUCTION_Q4_IU8_PREFILL_LAYERS))


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['', '999', 'all'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_layer_profile_isolation(monkeypatch, quant, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        resolved = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        owner, runner = _bind(resolved)
        expected = SUFFIX if resolved.binder.__name__ == '_production_binder' else ''
        peers.append((owner, runner, expected))
        assert os.environ[ENV] == ambient
        for peer, current, value in peers:
            assert peer.q4_iu8_layers == current.q4_iu8_layers == value
        assert _inherit_lane_decisions(owner, SimpleNamespace()).q4_iu8_layers == expected


def test_layer_rebind_keeps_ambient_and_does_not_write_selector(monkeypatch):
    monkeypatch.setenv(ENV, '999')
    owner = SimpleNamespace(runner=SimpleNamespace())
    for production, expected in [(True, SUFFIX), (False, ''), (True, SUFFIX)]:
        profiles._publish_q4_layers(owner, production=production)
        assert owner.q4_iu8_layers == owner.runner.q4_iu8_layers == expected
        assert os.environ[ENV] == '999'
    assert '"HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS":' not in inspect.getsource(profiles._strict_binder)
