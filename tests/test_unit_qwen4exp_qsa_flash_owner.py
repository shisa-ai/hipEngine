"""QSA flash enablement and layer scope are one owner decision."""
from types import SimpleNamespace
import os
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

ENV = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'
LAYERS = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('layer,selected', [(2, False), (35, True), (47, True), (48, False)])
def test_owner_flash_policy_ignores_opposite_ambient(monkeypatch, enabled, layer, selected):
    monkeypatch.setenv(ENV, '0' if enabled else '1')
    monkeypatch.setenv(LAYERS, 'all')
    policy = runtime.QSAFlashPrefillPolicy(enabled, '35-47')
    weight = SimpleNamespace(spec=SimpleNamespace(slot_path=f'layers.{layer}.attn_q'))
    assert runtime.qsa_flash_prefill_selected(weight, policy) is (enabled and selected)
    assert not runtime.qsa_flash_prefill_selected(weight, policy, dense_rows=0)
    assert not runtime.qsa_flash_prefill_selected(weight, policy, head_dim=128)
    assert not runtime.qsa_flash_prefill_selected(weight, policy, kv_heads=1)


def test_flash_scope_restores_absence_runner_and_clone(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(runner=SimpleNamespace(qsa_flash_prefill_policy=None))
    policy = runtime.QSAFlashPrefillPolicy(False, '')
    with pytest.raises(RuntimeError, match='restore'):
        with runtime.qsa_flash_prefill(owner, policy):
            assert owner.qsa_flash_prefill_policy is owner.runner.qsa_flash_prefill_policy is policy
            assert _inherit_lane_decisions(owner, SimpleNamespace()).qsa_flash_prefill_policy is policy
            assert os.environ[ENV] == '1'
            raise RuntimeError('restore')
    assert not hasattr(owner, 'qsa_flash_prefill_policy')
    assert owner.runner.qsa_flash_prefill_policy is None


def test_unbound_flash_policy_uses_legacy_parser(monkeypatch):
    weight = SimpleNamespace(spec=SimpleNamespace(slot_path='layers.35.attn_q'))
    monkeypatch.setenv(ENV, 'false')
    assert not runtime.qsa_flash_prefill_selected(weight, None)
    monkeypatch.setenv(ENV, 'yes')
    monkeypatch.setenv(LAYERS, '2,35-47')
    assert runtime.qsa_flash_prefill_selected(weight, None)
