"""Candidate AR keeps its CLI syntax while binding MMQ owner decisions."""
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_candidate_ar as ar

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE'


@pytest.mark.parametrize('raw,expected', [('1', True), ('0', False), ('false', False)])
def test_split_attention_override_out_of_environment(raw, expected):
    env, attention = ar.split_owner_overrides([ENV + '=' + raw, 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES=2'])
    assert env == {'HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES': '2'}
    assert attention is expected


@pytest.mark.parametrize('enabled', [False, True])
def test_scope_prepares_parent_before_sampling_and_restores(monkeypatch, enabled):
    monkeypatch.setenv(ENV, '0' if enabled else '1')
    calls = []
    runner = SimpleNamespace(q8_mmq_attn_gate_enabled=None, q8_mmq_prefill_enabled=False)
    owner = SimpleNamespace(runner=runner, q8_mmq_attn_gate_enabled=None, q8_mmq_prefill_enabled=False)
    def prepare():
        assert runner.q8_mmq_prefill_enabled is True
        assert runner.q8_mmq_attn_gate_enabled is enabled
        calls.append('prepared')
    runner.configure_mmq_prefill_resources = prepare
    with pytest.raises(RuntimeError):
        with ar.attention_override_scope(owner, enabled) as record:
            assert calls == ['prepared']
            assert record == {'q8_mmq_attn_gate_enabled': enabled, 'q8_mmq_prefill_enabled': True}
            assert os.environ[ENV] == ('0' if enabled else '1')
            raise RuntimeError('injected')
    assert owner.q8_mmq_attn_gate_enabled is runner.q8_mmq_attn_gate_enabled is None
    assert owner.q8_mmq_prefill_enabled is runner.q8_mmq_prefill_enabled is False


def test_omitted_attention_override_does_not_allocate():
    owner = SimpleNamespace(q8_mmq_attn_gate_enabled=None, q8_mmq_prefill_enabled=False)
    with ar.attention_override_scope(owner, None) as record:
        assert record == {'q8_mmq_attn_gate_enabled': None, 'q8_mmq_prefill_enabled': False}
