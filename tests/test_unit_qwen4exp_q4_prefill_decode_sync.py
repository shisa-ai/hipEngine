"""Decode-sync Q4 prefill and layer binding preserve process inputs."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_decode_sync_ab as diagnostic

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'
LAYERS = 'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS'


def test_plain_prefill_binding_restores_distinct_values(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    monkeypatch.setenv(LAYERS, '999')
    runner = SimpleNamespace(q4_iu8_prefill_enabled=None, q4_iu8_layers='runner')
    owner = SimpleNamespace(runner=runner, q4_iu8_prefill_enabled=False, q4_iu8_layers='owner')
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            assert diagnostic.bind_production_q4_prefill(stack, owner) is True
            assert owner.q4_iu8_prefill_enabled is runner.q4_iu8_prefill_enabled is True
            assert owner.q4_iu8_layers == runner.q4_iu8_layers == '35-47'
            assert os.environ[ENV] == '0' and os.environ[LAYERS] == '999'
            raise RuntimeError('restore')
    assert owner.q4_iu8_prefill_enabled is False
    assert runner.q4_iu8_prefill_enabled is None
    assert owner.q4_iu8_layers == 'owner'
    assert runner.q4_iu8_layers == 'runner'
    source = inspect.getsource(diagnostic.main)
    assert 'bind_production_q4_prefill(owner_scope, generator)' in source
    assert '"q4_iu8_prefill_enabled": q4_prefill_enabled' in source
    assert '"q4_iu8_layers": q4_layers' in source
    assert not hasattr(diagnostic, 'configure_production_routes')
