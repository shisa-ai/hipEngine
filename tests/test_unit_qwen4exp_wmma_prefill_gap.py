"""Legacy profile-gap generic WMMA syntax is owner-bound."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_GGUF_WMMA_PREFILL'


@pytest.mark.parametrize('raw,enabled', [('0', False), ('1', True), ('false', False)])
def test_generic_override_binds_and_restores(monkeypatch, raw, enabled):
    monkeypatch.setenv(ENV, '0' if enabled else '1')
    owner = SimpleNamespace(runner=SimpleNamespace(gguf_wmma_prefill_enabled=None))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            report = gap.bind_generic_wmma_override(stack, owner, {ENV: raw})
            assert report == {'gguf_wmma_prefill_override': enabled,
                              'gguf_wmma_prefill_enabled': enabled}
            assert owner.runner.gguf_wmma_prefill_enabled is enabled
            bound, effective = gap._apply_post_binder_overrides({ENV: raw})
            assert ENV not in bound and ENV not in effective
            assert os.environ[ENV] == ('0' if enabled else '1')
            raise RuntimeError('restore')
    assert not hasattr(owner, 'gguf_wmma_prefill_enabled')
    assert owner.runner.gguf_wmma_prefill_enabled is None
