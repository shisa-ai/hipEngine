"""Layer2 flash declarations preserve scope and restore owner state."""
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_layer2_profile_gate as gate
from hipengine.runtime.qwen4_exp_runner import QSAFlashPrefillPolicy


@pytest.mark.parametrize('enabled', [None, False, True])
def test_layer2_flash_bind_restore(monkeypatch, enabled):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL', '0')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS', '2')
    prior = QSAFlashPrefillPolicy(True, '35-47')
    owner = SimpleNamespace(qsa_flash_prefill_policy=prior,
                            runner=SimpleNamespace(qsa_flash_prefill_policy=prior))
    spec = replace(gate.CANDIDATES['production_baseline'],
                   qsa_flash_prefill_enabled=enabled)
    assert gate._candidate_named_profile_intact(spec) is (enabled is None)
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            report = gate.bind_candidate_flash(stack, owner, spec)
            assert report == {'enabled': True if enabled is None else enabled, 'layers': '35-47'}
            assert owner.runner.qsa_flash_prefill_policy == owner.qsa_flash_prefill_policy
            assert os.environ['HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'] == '0'
            assert os.environ['HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'] == '2'
            raise RuntimeError('restore')
    assert owner.qsa_flash_prefill_policy is owner.runner.qsa_flash_prefill_policy is prior


def test_conservative_flash_is_not_a_process_write():
    spec = gate.CANDIDATES['production_conservative']
    assert spec.qsa_flash_prefill_enabled is False
    assert 'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL' not in spec.environment
    assert gate.candidate_owner_changes(spec)['qsa_flash_prefill_enabled'] is False


def test_shared_candidate_binding_and_scope():
    spec = gate.CANDIDATES['production_conservative']
    prior = QSAFlashPrefillPolicy(True, '35-47')
    owner = SimpleNamespace(qsa_flash_prefill_policy=prior)
    with ExitStack() as stack:
        report = gate.bind_candidate_owners(stack, owner, spec)
        assert report['qsa_flash_prefill_policy'] == {'enabled': False, 'layers': '35-47'}
    assert owner.qsa_flash_prefill_policy is prior
    with gate.candidate_flash_scope(owner, None) as report:
        assert report == {'enabled': True, 'layers': '35-47'}
        assert owner.qsa_flash_prefill_policy is prior
    with gate.candidate_flash_scope(owner, spec) as report:
        assert report == {'enabled': False, 'layers': '35-47'}
    assert owner.qsa_flash_prefill_policy is prior
