"""Combined Q4 ablation binds both IU8 choices without process writes."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from unittest.mock import patch
with patch.dict(os.environ):
    from scripts import qwen4exp_r2e_route_ablation as ablation
from hipengine.runtime import qwen4_exp_runner as runtime

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT'


def test_combined_q4_ablation_uses_both_owners(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL', '1')
    assert 'q4_iu8' not in ablation.ROUTES
    bindings, envs, off = ablation.RUNNER_ROUTE_OFF_WITH_ENV['q4_iu8']
    assert dict(bindings) == {'q4_iu8_exact_enabled': False,
                              'q4_iu8_prefill_enabled': False}
    assert envs == ()
    assert off == '0'
    owner = SimpleNamespace(q4_iu8_exact_enabled=True, q4_iu8_prefill_enabled=True)
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            for field, value in bindings:
                stack.enter_context(getattr(runtime, field)(owner, value))
            assert owner.q4_iu8_exact_enabled is False
            assert owner.q4_iu8_prefill_enabled is False
            assert os.environ[ENV] == os.environ['HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'] == '1'
            raise RuntimeError('restore')
    assert owner.q4_iu8_exact_enabled is True
    assert owner.q4_iu8_prefill_enabled is True
    assert os.environ[ENV] == os.environ['HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'] == '1'
