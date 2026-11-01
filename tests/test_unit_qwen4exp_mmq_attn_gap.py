"""Profile-gap translates legacy attention syntax without process writes."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE'


@pytest.mark.parametrize('enabled', [False, True])
def test_gap_owner_scope_preserves_ambient_and_restores(monkeypatch, enabled):
    monkeypatch.setenv(ENV, '0' if enabled else '1')
    owner = SimpleNamespace(q8_mmq_attn_gate_enabled=False, q8_mmq_prefill_enabled=False,
        runner=SimpleNamespace(q8_mmq_attn_gate_enabled=False, q8_mmq_prefill_enabled=False))
    with ExitStack() as stack:
        record = gap.bind_attention_override(stack, owner, {ENV: '1' if enabled else '0'}, None)
        assert record == {'q8_mmq_attn_gate_override': enabled,
                          'q8_mmq_attn_gate_enabled': enabled}
        assert owner.runner.q8_mmq_attn_gate_enabled is enabled
        assert owner.runner.q8_mmq_prefill_enabled is True
        assert os.environ[ENV] == ('0' if enabled else '1')
    assert owner.runner.q8_mmq_attn_gate_enabled is False
    assert owner.runner.q8_mmq_prefill_enabled is False


def test_attention_key_is_not_applied_to_environment(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    bound, effective = gap._apply_post_binder_overrides({ENV: '1'})
    assert ENV not in bound and ENV not in effective
    assert os.environ[ENV] == '0'


def test_explicit_attention_and_disabled_parent_fail_loudly():
    with ExitStack() as stack, pytest.raises(ValueError, match='q8-mmq-prefill off'):
        gap.bind_attention_override(stack, SimpleNamespace(), {ENV: '1'}, False)
