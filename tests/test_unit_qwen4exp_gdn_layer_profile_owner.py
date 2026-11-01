"""Profile GDN layer scopes are isolated from peer binds and process inputs."""
import os
from types import SimpleNamespace

import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.runtime.qwen4_exp_runner import resolve_gdn_colwarps_layers
from scripts import qwen4exp_profile_gap as gap
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

ENV = 'HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS'


@pytest.mark.parametrize('profile,quant,layers', [
    (ExecutionProfile.PRODUCTION, 'gguf_ud_q4_k_xl', ','.join(map(str, range(48)))),
    (ExecutionProfile.STRICT, 'gguf_ud_q4_k_xl', ''),
    (ExecutionProfile.BATCH_INVARIANT, 'gguf_ud_q4_k_xl', ''),
    (ExecutionProfile.PRODUCTION, 'gguf_q4_k_m', ''),
])
def test_every_bind_publishes_its_scope_without_process_write(monkeypatch, profile, quant, layers):
    monkeypatch.setenv(ENV, '999')
    owner, runner = _bind(_resolve(profile, quant=quant))
    assert owner.gdn_colwarps_layers == runner.gdn_colwarps_layers == layers
    assert resolve_gdn_colwarps_layers(owner) == layers
    assert os.environ[ENV] == '999'
    assert _inherit_lane_decisions(owner, SimpleNamespace()).gdn_colwarps_layers == layers


@pytest.mark.parametrize('order', [(ExecutionProfile.PRODUCTION, ExecutionProfile.STRICT),
                                   (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)])
def test_peer_binding_and_rebinding_preserve_scope(order):
    first, runner = _bind(_resolve(order[0]))
    prior = first.gdn_colwarps_layers
    second, _ = _bind(_resolve(order[1]))
    assert first.gdn_colwarps_layers == runner.gdn_colwarps_layers == prior
    assert second.gdn_colwarps_layers != prior
    resolved = _resolve(order[1])
    resolved.binder(first, resolved)
    assert first.gdn_colwarps_layers == runner.gdn_colwarps_layers == second.gdn_colwarps_layers


def test_gap_rejects_legacy_layer_override():
    with pytest.raises(ValueError, match='--gdn-colwarps-layers'):
        gap._parse_overrides([ENV + '=999'])
