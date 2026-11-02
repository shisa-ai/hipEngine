"""Candidate multiplier transport preserves profile values on omission."""
from contextlib import ExitStack
from dataclasses import replace
import inspect
import os
from types import SimpleNamespace

import pytest
from scripts import qwen4exp_layer2_profile_gate as gate

KEY = 'HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT'


@pytest.mark.parametrize('value', [None, 0.5, 1.0, 4.0, 16.0])
def test_candidate_multiplier_binding_and_restore(monkeypatch, value):
    monkeypatch.setenv(KEY, '8')
    spec = replace(gate.CANDIDATES['production_baseline'], q4_iu8_risk_mult=value)
    owner = SimpleNamespace(q4_iu8_risk_mult=2.0,
                            runner=SimpleNamespace(q4_iu8_risk_mult=4.0))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            gate.bind_candidate_q4_prefill(stack, owner, spec)
            assert owner.q4_iu8_risk_mult == (2.0 if value is None else value)
            assert owner.runner.q4_iu8_risk_mult == (4.0 if value is None else value)
            assert os.environ[KEY] == '8'
            raise RuntimeError('restore')
    assert owner.q4_iu8_risk_mult == 2.0
    assert owner.runner.q4_iu8_risk_mult == 4.0
    assert gate._candidate_named_profile_intact(spec) is (value is None)
    changes = gate.candidate_owner_changes(spec)
    assert ('q4_iu8_risk_mult' in changes) is (value is not None)
    with gate.candidate_q4_prefill_scope(owner, spec):
        assert owner.q4_iu8_risk_mult == (2.0 if value is None else value)
    assert owner.q4_iu8_risk_mult == 2.0


def test_candidate_paths_report_requested_and_effective_multiplier():
    common = inspect.getsource(gate.bind_candidate_owners)
    primary = inspect.getsource(gate._run)
    assert '"q4_iu8_risk_mult": resolve_q4_iu8_risk_mult(candidate_generator)' in common
    assert 'resolved_q4_risk_mult = resolve_q4_iu8_risk_mult(candidate_generator)' in primary
    assert '"q4_iu8_risk_mult": resolved_q4_risk_mult' in primary
    assert '"q4_iu8_risk_mult_override": candidate_spec.q4_iu8_risk_mult' in primary
