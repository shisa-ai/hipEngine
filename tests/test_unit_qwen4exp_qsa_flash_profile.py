"""Profiles publish independent paired flash choices without process writes."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation.qwen4_exp_profiles import QWEN4_EXP_QUANTS, PRODUCTION_QSA_FLASH_PREFILL_LAYERS
from hipengine.runtime.qwen4_exp_runner import QSAFlashPrefillPolicy
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

E = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'
L = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'


@pytest.mark.parametrize('quant', QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_flash_profile_pair_isolation(monkeypatch, quant, ambient, order):
    monkeypatch.setenv(E, ambient)
    monkeypatch.setenv(L, '2')
    peers = []
    for production in order:
        resolved = _resolve(
            ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        generator, runner = _bind(resolved)
        effective_production = resolved.binder.__name__ == '_production_binder'
        expected = QSAFlashPrefillPolicy(
            effective_production and quant != 'gguf_ud_q4_k_xl',
            ','.join(map(str, PRODUCTION_QSA_FLASH_PREFILL_LAYERS)) if effective_production else '')
        peers.append((generator, runner, expected))
        assert os.environ[E] == ambient and os.environ[L] == '2'
        for owner, current, policy in peers:
            assert owner.qsa_flash_prefill_policy == current.qsa_flash_prefill_policy == policy
        assert _inherit_lane_decisions(generator, SimpleNamespace()).qsa_flash_prefill_policy == expected


def test_non_ud_production_publication_keeps_flash_on():
    from hipengine.generation.qwen4_exp_profiles import _publish_flash_prefill
    owner = SimpleNamespace(runner=SimpleNamespace())
    _publish_flash_prefill(owner, production=True, quant='gguf_q4_k_m')
    assert owner.qsa_flash_prefill_policy.enabled is True
    assert owner.runner.qsa_flash_prefill_policy == owner.qsa_flash_prefill_policy


def test_flash_recovery_tuple_controls_only_ud(monkeypatch):
    from hipengine.generation import qwen4_exp_profiles as profiles
    monkeypatch.setattr(profiles, 'PRODUCTION_ARITHMETIC_RECOVERY_FLAGS', ())
    owner, _ = _bind(_resolve(ExecutionProfile.PRODUCTION))
    assert owner.qsa_flash_prefill_policy.enabled is True
