"""Profile-gap flash overrides preserve the omitted policy axis."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from hipengine.runtime.qwen4_exp_runner import QSAFlashPrefillPolicy
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'
LAYERS = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'


@pytest.mark.parametrize('prior', [None, QSAFlashPrefillPolicy(True, '35-47'),
                                  QSAFlashPrefillPolicy(False, '2')])
@pytest.mark.parametrize('overrides', [{ENV: '0'}, {LAYERS: '2'}, {ENV: 'yes', LAYERS: 'all'}])
def test_flash_override_preserves_other_axis_and_restores(monkeypatch, prior, overrides):
    monkeypatch.setenv(ENV, '1')
    monkeypatch.setenv(LAYERS, '7-9')
    owner = SimpleNamespace(qsa_flash_prefill_policy=prior,
        runner=SimpleNamespace(qsa_flash_prefill_policy=prior))
    expected = QSAFlashPrefillPolicy(
        (overrides[ENV] not in {'', '0', 'false', 'False'}) if ENV in overrides
        else (True if prior is None else prior.enabled),
        overrides.get(LAYERS, prior.layers if prior is not None else '7-9'))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            report = gap.bind_flash_prefill_override(stack, owner, overrides)
            assert report['qsa_flash_prefill_policy'] == {'enabled': expected.enabled, 'layers': expected.layers}
            assert owner.qsa_flash_prefill_policy == owner.runner.qsa_flash_prefill_policy == expected
            bound, effective = gap._apply_post_binder_overrides(overrides)
            assert ENV not in bound and LAYERS not in bound
            assert ENV not in effective and LAYERS not in effective
            assert os.environ[ENV] == '1' and os.environ[LAYERS] == '7-9'
            raise RuntimeError('restore')
    assert owner.qsa_flash_prefill_policy is owner.runner.qsa_flash_prefill_policy is prior
