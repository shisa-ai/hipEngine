"""Combined Q4 ablation binds exact IU8 without changing its process input."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_r2e_route_ablation as ablation
from hipengine.runtime import qwen4_exp_runner as runtime

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT'


def test_combined_q4_ablation_uses_owner_and_remaining_prefill_input(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    assert 'q4_iu8' not in ablation.ROUTES
    bindings, envs, off = ablation.RUNNER_ROUTE_OFF_WITH_ENV['q4_iu8']
    assert dict(bindings) == {'q4_iu8_exact_enabled': False}
    assert envs == ('HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL',)
    assert off == '0'
    owner = SimpleNamespace(q4_iu8_exact_enabled=True)
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            for field, value in bindings:
                stack.enter_context(getattr(runtime, field)(owner, value))
            assert owner.q4_iu8_exact_enabled is False
            assert os.environ[ENV] == '1'
            raise RuntimeError('restore')
    assert owner.q4_iu8_exact_enabled is True
    assert os.environ[ENV] == '1'
