"""Decode-sync plain Q4 prefill binding preserves process inputs."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_decode_sync_ab as diagnostic

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'


def test_plain_prefill_is_not_written_by_configuration(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS', '35-47')
    diagnostic.configure_production_routes()
    assert os.environ[ENV] == '0'


def test_plain_prefill_binding_restores_distinct_values(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    runner = SimpleNamespace(q4_iu8_prefill_enabled=None)
    owner = SimpleNamespace(runner=runner, q4_iu8_prefill_enabled=False)
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            assert diagnostic.bind_production_q4_prefill(stack, owner) is True
            assert owner.q4_iu8_prefill_enabled is runner.q4_iu8_prefill_enabled is True
            assert os.environ[ENV] == '0'
            raise RuntimeError('restore')
    assert owner.q4_iu8_prefill_enabled is False
    assert runner.q4_iu8_prefill_enabled is None
    source = inspect.getsource(diagnostic.main)
    assert 'bind_production_q4_prefill(owner_scope, generator)' in source
    assert '"q4_iu8_prefill_enabled": q4_prefill_enabled' in source
