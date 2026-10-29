"""Pair diagnostics disable producer mirrors only in their execution context."""
import contextvars
import os
from pathlib import Path

import pytest
from scripts.qwen4exp_pair_harness_controls import (
    pair_staging_without_producer_mirror, producer_mirror_record,
)
from hipengine.runtime import qwen4_exp_runner as runner

KEY = "HIPENGINE_QWEN4_EXP_GR_F16_MIRROR"
ROOT = Path(__file__).resolve().parents[1]
HARNESSES = (
    "scripts/qwen4exp_f16_pair_engine_ab.py",
    "scripts/qwen4exp_f16_pair_native_gate.py",
    "scripts/qwen4exp_shared_f16_cast_owner.py",
)


@pytest.mark.parametrize("stale", ["0", "1", "false"])
def test_scope_preserves_environment_and_restores_decision(monkeypatch, stale):
    monkeypatch.setenv(KEY, stale)
    assert runner._qwen4_exp_gr_f16_mirror_enabled()
    with pair_staging_without_producer_mirror() as prior:
        assert prior is True
        assert not runner._qwen4_exp_gr_f16_mirror_enabled()
        assert os.environ[KEY] == stale
        assert contextvars.Context().run(runner._qwen4_exp_gr_f16_mirror_enabled)
        with pair_staging_without_producer_mirror() as nested:
            assert nested is False
        assert not runner._qwen4_exp_gr_f16_mirror_enabled()
    assert runner._qwen4_exp_gr_f16_mirror_enabled()
    assert os.environ[KEY] == stale


def test_scope_restores_after_failure():
    with pytest.raises(RuntimeError, match="arm failed"):
        with pair_staging_without_producer_mirror():
            raise RuntimeError("arm failed")
    assert runner._qwen4_exp_gr_f16_mirror_enabled()


def test_record_names_execution_local_control():
    record = producer_mirror_record(True)
    assert record["control"] == "gr_f16_mirror"
    assert record["scope"] == "execution-context"
    assert record["value"] is False
    assert record["prior"] is True
    assert "env" not in record


@pytest.mark.parametrize("tokens,expected", [(512, 0), (1024, 0), (4096, 61), (5120, 61)])
def test_native_pair_gate_uses_current_admitted_rows(tokens, expected):
    from scripts.qwen4exp_f16_pair_native_gate import expected_pairs
    assert expected_pairs(tokens) == expected


@pytest.mark.parametrize("path", HARNESSES)
def test_each_pair_harness_uses_scoped_mirror_control(path):
    source = (ROOT / path).read_text()
    assert "mirror_scope.enter_context(pair_staging_without_producer_mirror())" in source
    assert "mirror_scope.close()" in source
    assert "producer_mirror_record(prior_mirror)" in source
    assert "rollback_producer_mirror" not in source
    assert "restore_producer_mirror" not in source
    assert "_qwen4_exp_gr_f16_mirror_enabled()" in source
