"""Layer-2 attention MMQ candidate binds shape and workspace owners."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_layer2_profile_gate as gate

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE'


def test_candidate_declares_explicit_shape_and_parent():
    spec = gate.CANDIDATES['q8_mmq_attn_gate']
    assert spec.environment == {}
    assert spec.q8_mmq_attn_gate_enabled is True
    assert spec.q8_mmq_prefill_enabled is True
    assert not gate._candidate_named_profile_intact(spec)


def test_candidate_scope_restores_owner_and_preserves_ambient(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    owner = SimpleNamespace(q8_mmq_attn_gate_enabled=False,
        q8_mmq_prefill_enabled=False,
        runner=SimpleNamespace(q8_mmq_attn_gate_enabled=False,
                               q8_mmq_prefill_enabled=False))
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            assert gate._bind_candidate_q8_mmq_prefill(
                stack, owner, gate.CANDIDATES['q8_mmq_attn_gate']) is True
            assert owner.runner.q8_mmq_prefill_enabled is True
            actual = gate.bind_candidate_q8_mmq_attn_gate(
                stack, owner, gate.CANDIDATES['q8_mmq_attn_gate'])
            assert actual is True
            assert owner.q8_mmq_attn_gate_enabled is owner.runner.q8_mmq_attn_gate_enabled is True
            assert os.environ[ENV] == '0'
            raise RuntimeError('injected')
    assert owner.q8_mmq_attn_gate_enabled is owner.runner.q8_mmq_attn_gate_enabled is False
    assert owner.q8_mmq_prefill_enabled is owner.runner.q8_mmq_prefill_enabled is False


def test_provenance_reports_selected_owner():
    record = gate._candidate_owner_provenance(
        gate.CANDIDATES['q8_mmq_attn_gate'], True, True, True)
    assert record['q8_mmq_attn_gate_enabled'] is True
    assert record['q8_mmq_attn_gate_override'] is True
