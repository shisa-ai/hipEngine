"""Candidate AR flash syntax uses the shared partial owner-policy binder."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from hipengine.runtime.qwen4_exp_runner import QSAFlashPrefillPolicy
from scripts import qwen4exp_candidate_ar as ar

ENV = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'
LAYERS = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'


@pytest.mark.parametrize('overrides,expected', [
    ({ENV: '0'}, QSAFlashPrefillPolicy(False, '35-47')),
    ({LAYERS: '2'}, QSAFlashPrefillPolicy(True, '2')),
    ({ENV: '1', LAYERS: 'all'}, QSAFlashPrefillPolicy(True, 'all')),
])
def test_candidate_flash_split_bind_restore(monkeypatch, overrides, expected):
    monkeypatch.setenv(ENV, '0')
    monkeypatch.setenv(LAYERS, '7-9')
    environment, requested = ar.split_flash_prefill_overrides({**overrides, 'OTHER': 'x'})
    assert requested == overrides
    assert environment == {'OTHER': 'x'}
    prior = QSAFlashPrefillPolicy(True, '35-47')
    owner = SimpleNamespace(qsa_flash_prefill_policy=prior,
        runner=SimpleNamespace(qsa_flash_prefill_policy=prior))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            record = ar.bind_flash_prefill_override(stack, owner, requested)
            assert record['qsa_flash_prefill_policy'] == {'enabled': expected.enabled, 'layers': expected.layers}
            assert owner.qsa_flash_prefill_policy == owner.runner.qsa_flash_prefill_policy == expected
            assert os.environ[ENV] == '0' and os.environ[LAYERS] == '7-9'
            raise RuntimeError('restore')
    assert owner.qsa_flash_prefill_policy is owner.runner.qsa_flash_prefill_policy is prior
