"""The pair harnesses must run every arm without the producer F16 mirror."""

import os
from pathlib import Path

import pytest

from scripts.qwen4exp_pair_harness_controls import (
    GR_F16_MIRROR_ENV,
    pair_staging_without_producer_mirror,
    producer_mirror_record,
    restore_producer_mirror,
    rollback_producer_mirror,
)

ROOT = Path(__file__).resolve().parents[1]
HARNESSES = (
    "scripts/qwen4exp_f16_pair_engine_ab.py",
    "scripts/qwen4exp_f16_pair_native_gate.py",
    "scripts/qwen4exp_shared_f16_cast_owner.py",
)


def test_rollback_replaces_the_default_and_restores_an_unset_variable(monkeypatch):
    monkeypatch.delenv(GR_F16_MIRROR_ENV, raising=False)
    prior = rollback_producer_mirror()
    assert prior is None
    assert os.environ[GR_F16_MIRROR_ENV] == "0"
    restore_producer_mirror(prior)
    assert GR_F16_MIRROR_ENV not in os.environ


def test_rollback_restores_a_caller_value(monkeypatch):
    monkeypatch.setenv(GR_F16_MIRROR_ENV, "1")
    with pair_staging_without_producer_mirror() as prior:
        assert prior == "1"
        assert os.environ[GR_F16_MIRROR_ENV] == "0"
    assert os.environ[GR_F16_MIRROR_ENV] == "1"


def test_rollback_restores_after_a_failing_arm(monkeypatch):
    monkeypatch.setenv(GR_F16_MIRROR_ENV, "on")
    with pytest.raises(RuntimeError, match="arm failed"):
        with pair_staging_without_producer_mirror():
            raise RuntimeError("arm failed")
    assert os.environ[GR_F16_MIRROR_ENV] == "on"


def test_rollback_leaves_the_runtime_decision_on_the_parent_route(monkeypatch):
    """The rollback is the GR read's own switch, read where it is consumed."""

    from hipengine.runtime import qwen4_exp_runner as runner

    monkeypatch.setenv(GR_F16_MIRROR_ENV, "1")
    assert runner._qwen4_exp_gr_f16_mirror_enabled()
    with pair_staging_without_producer_mirror():
        assert not runner._qwen4_exp_gr_f16_mirror_enabled()
    assert runner._qwen4_exp_gr_f16_mirror_enabled()


def test_record_names_the_rollback_and_what_it_replaced():
    record = producer_mirror_record(None)
    assert record["env"] == GR_F16_MIRROR_ENV
    assert record["value"] == "0"
    assert record["prior"] is None
    assert "mirror" in record["reason"]
    assert producer_mirror_record("1")["prior"] == "1"


@pytest.mark.parametrize("path", HARNESSES)
def test_each_pair_harness_declares_and_restores_the_rollback(path):
    """A harness that measured both arms under the mirror would fail anyway."""

    source = (ROOT / path).read_text()
    assert "rollback_producer_mirror()" in source
    assert "restore_producer_mirror(prior_mirror)" in source
    assert "producer_mirror_record(prior_mirror)" in source
    assert '"controls"' in source
    # A harness that ran with the mirror serving would fail on engagement
    # counts; the guard names the cause at the arm instead.
    assert "_qwen4_exp_gr_f16_mirror_enabled()" in source
