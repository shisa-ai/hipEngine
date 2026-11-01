"""Generic WMMA owner state composes with the explicit Q8 layer scope."""
from types import SimpleNamespace
import os
import pytest
from hipengine.runtime import gguf_linear as linear
from hipengine.runtime import qwen4_exp_runner as runner
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

ENV = 'HIPENGINE_GGUF_WMMA_PREFILL'


@pytest.mark.parametrize('enabled', [False, True])
def test_generic_wmma_owner_overrides_ambient_and_restores(monkeypatch, enabled):
    monkeypatch.setenv(ENV, '0' if enabled else '1')
    owner = SimpleNamespace(runner=SimpleNamespace(gguf_wmma_prefill_enabled=None))
    with pytest.raises(RuntimeError, match='restore'):
        with runner.gguf_wmma_prefill(owner, enabled):
            assert owner.runner.gguf_wmma_prefill_enabled is enabled
            late = _inherit_lane_decisions(owner, SimpleNamespace())
            assert late.gguf_wmma_prefill_enabled is enabled
            with runner._q8_wmma_scope_sessions(linear.Q8WMMALayerScope.empty(), owner.runner):
                assert linear.gguf_wmma_prefill_enabled() is enabled
                assert linear.gguf_wmma_prefill_enabled(not enabled) is not enabled
            with runner._q8_wmma_scope_sessions(linear.Q8WMMALayerScope.parse('20'), owner.runner):
                assert linear.gguf_wmma_prefill_enabled() is True
            assert os.environ[ENV] == ('0' if enabled else '1')
            raise RuntimeError('restore')
    assert not hasattr(owner, 'gguf_wmma_prefill_enabled')
    assert owner.runner.gguf_wmma_prefill_enabled is None
