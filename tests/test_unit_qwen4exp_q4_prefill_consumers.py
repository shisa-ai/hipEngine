"""Diagnostic candidate consumers bind and report plain-Q4 prefill owners."""
from dataclasses import replace
from types import SimpleNamespace
import inspect
import os

import pytest
from scripts import qwen4exp_layer2_profile_gate as gate
from scripts import qwen4exp_conservative_cost as cost
from scripts import qwen4exp_prefill_restore_ab as restore
from scripts import qwen4exp_long_context_gate as long_context

KEY = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'


@pytest.mark.parametrize('enabled', [None, False, True])
def test_candidate_scope_restores_and_inventories_owner(monkeypatch, enabled):
    monkeypatch.setenv(KEY, '0')
    spec = replace(gate.CANDIDATES['production_baseline'], q4_iu8_prefill_enabled=enabled)
    owner = SimpleNamespace(q4_iu8_prefill_enabled=True,
                            runner=SimpleNamespace(q4_iu8_prefill_enabled=None))
    with pytest.raises(RuntimeError, match='restore'):
        with gate.candidate_q4_prefill_scope(owner, spec) as selected:
            assert selected is (True if enabled is None else enabled)
            assert owner.q4_iu8_prefill_enabled is selected
            assert owner.runner.q4_iu8_prefill_enabled is enabled
            assert os.environ[KEY] == '0'
            raise RuntimeError('restore')
    assert owner.q4_iu8_prefill_enabled is True
    assert owner.runner.q4_iu8_prefill_enabled is None
    assert gate.candidate_owner_changes(spec).get('q4_iu8_prefill_enabled') is enabled
    with gate.candidate_q4_prefill_scope(owner, None) as selected:
        assert selected is True
        assert owner.runner.q4_iu8_prefill_enabled is None


@pytest.mark.parametrize('module', [cost, restore])
def test_sample_consumers_bind_and_report_choice_inside_scope(module):
    source = inspect.getsource(module.main)
    assert 'candidate_q4_prefill_scope(' in source
    assert "row['q4_iu8_prefill_enabled'] = q4_prefill" in source


def test_long_context_reports_effective_shared_owner():
    source = inspect.getsource(long_context.run)
    assert '"q4_iu8_prefill_enabled": candidate_owners["q4_iu8_prefill_enabled"]' in source
