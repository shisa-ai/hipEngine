"""Candidate AR generic WMMA requests stay off the process environment."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_candidate_ar as ar

ENV = 'HIPENGINE_GGUF_WMMA_PREFILL'


@pytest.mark.parametrize('raw,enabled', [('0', False), ('1', True), (' YES ', True)])
def test_candidate_wmma_owner_override(monkeypatch, raw, enabled):
    monkeypatch.setenv(ENV, '0' if enabled else '1')
    environment, selected = ar.split_generic_wmma_override({ENV: raw, 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES': '2'})
    assert environment == {'HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES': '2'}
    assert selected is enabled
    owner = SimpleNamespace(runner=SimpleNamespace(gguf_wmma_prefill_enabled=None))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            record = ar.bind_generic_wmma_override(stack, owner, selected)
            assert record == {'gguf_wmma_prefill_enabled': enabled}
            assert owner.runner.gguf_wmma_prefill_enabled is enabled
            assert os.environ[ENV] == ('0' if enabled else '1')
            raise RuntimeError('restore')
    assert not hasattr(owner, 'gguf_wmma_prefill_enabled')
    assert owner.runner.gguf_wmma_prefill_enabled is None
