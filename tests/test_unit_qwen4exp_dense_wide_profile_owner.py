"""Profile-bound dense-wide routing is independent of peer process pins."""
import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.runtime.gguf_linear import Q8DenseWidePolicy, resolve_q8_dense_wide_policy
from scripts import qwen4exp_profile_gap as gap
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

ENABLE = 'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE'
LAYERS = 'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS'


@pytest.mark.parametrize('profile,quant,enabled,layers', [
    (ExecutionProfile.PRODUCTION, 'gguf_ud_q4_k_xl', True, frozenset(range(48))),
    (ExecutionProfile.STRICT, 'gguf_ud_q4_k_xl', False, None),
    (ExecutionProfile.BATCH_INVARIANT, 'gguf_ud_q4_k_xl', False, None),
    (ExecutionProfile.PRODUCTION, 'gguf_q4_k_m', False, None),
])
def test_binder_publishes_policy_without_process_writes(monkeypatch, profile, quant, enabled, layers):
    monkeypatch.setenv(ENABLE, 'caller')
    monkeypatch.setenv(LAYERS, '7')
    gen, runner = _bind(_resolve(profile, quant=quant))
    expected = Q8DenseWidePolicy(enabled, layers)
    assert resolve_q8_dense_wide_policy(gen) == expected
    assert runner.q8_dense_wide_policy == expected
    assert os.environ[ENABLE] == 'caller'
    assert os.environ[LAYERS] == '7'
    clone = _inherit_lane_decisions(gen, SimpleNamespace())
    assert clone.q8_dense_wide_policy == expected


@pytest.mark.parametrize('order', [(ExecutionProfile.PRODUCTION, ExecutionProfile.STRICT),
                                    (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)])
def test_binding_peer_does_not_change_existing_owner(order):
    first, _ = _bind(_resolve(order[0]))
    prior = resolve_q8_dense_wide_policy(first)
    second, _ = _bind(_resolve(order[1]))
    assert resolve_q8_dense_wide_policy(first) == prior
    assert resolve_q8_dense_wide_policy(second).enabled is (order[1] == ExecutionProfile.PRODUCTION)


@pytest.mark.parametrize('key,flag', [(ENABLE, '--q8-dense-wide'), (LAYERS, '--q8-dense-wide-layers')])
def test_profile_gap_rejects_ineffective_legacy_override(key, flag):
    with pytest.raises(ValueError, match=flag):
        gap._parse_overrides([key + '=0'])
