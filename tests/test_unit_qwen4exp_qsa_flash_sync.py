"""Decode-sync flash policy is owned and restored with its measured generator."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_decode_sync_ab as sync
from hipengine.runtime.qwen4_exp_runner import QSAFlashPrefillPolicy

ENV = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'
LAYERS = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'


def test_sync_flash_scope_and_process_preservation(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    monkeypatch.setenv(LAYERS, '2')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL', '0')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS', '2')
    assert os.environ[ENV] == '0' and os.environ[LAYERS] == '2'
    owner = SimpleNamespace(runner=SimpleNamespace(qsa_flash_prefill_policy=None))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            report = sync.bind_production_flash(stack, owner)
            assert report == {'enabled': True, 'layers': '35-47'}
            assert owner.qsa_flash_prefill_policy == QSAFlashPrefillPolicy(True, '35-47')
            assert owner.runner.qsa_flash_prefill_policy is owner.qsa_flash_prefill_policy
            raise RuntimeError('restore')
    assert not hasattr(owner, 'qsa_flash_prefill_policy')
    assert owner.runner.qsa_flash_prefill_policy is None
