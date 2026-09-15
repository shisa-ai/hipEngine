from types import SimpleNamespace
import os

import pytest

from scripts.qwen4exp_dense_q8_nongr_gate import (
    FLAG, exclude_gr_reads, validate_exclusion,
)


def test_exclusion_counts_eligible_reads_and_restores_environment(monkeypatch):
    calls = []
    original = lambda *args, **kwargs: calls.append(os.environ.get(FLAG))
    module = SimpleNamespace(run_qwen4_exp_gr_read=original)
    monkeypatch.setenv(FLAG, "1")
    with exclude_gr_reads(module) as counter:
        for rows in (1, 256, 512, 1024):
            module.run_qwen4_exp_gr_read(rows=rows)
            assert os.environ[FLAG] == "1"
        assert counter == {"calls": 2, "rows": {512: 1, 1024: 1}}
    assert calls == ["0"] * 4
    assert module.run_qwen4_exp_gr_read is original


@pytest.mark.parametrize("value", [None, "0", "", "false", "False"])
def test_strict_control_is_unchanged(value, monkeypatch):
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    module = SimpleNamespace(run_qwen4_exp_gr_read=lambda **kwargs: os.environ.get(FLAG))
    with exclude_gr_reads(module) as counter:
        assert module.run_qwen4_exp_gr_read(rows=1024) == value
        assert counter["calls"] == 0
    assert os.environ.get(FLAG) == value


def test_exception_restores_function_and_flag(monkeypatch):
    def original(**kwargs):
        assert os.environ[FLAG] == "0"
        raise RuntimeError("projection failed")

    module = SimpleNamespace(run_qwen4_exp_gr_read=original)
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(RuntimeError, match="projection failed"):
        with exclude_gr_reads(module):
            module.run_qwen4_exp_gr_read(rows=512)
    assert module.run_qwen4_exp_gr_read is original
    assert os.environ[FLAG] == "1"


def packet():
    return {
        "status": "completed", "candidate": "production_dense_q8_restore",
        "candidate_dispatch_calls": 21744,
        "candidate_dispatch_shapes": [
            {"arguments": [512, 2560, 640], "calls": 21744}],
        "protocol": {"complete_fixture": True, "chunk": 1024, "repeats": 3},
    }


def test_full_matrix_requires_exclusion_and_candidate_engagement():
    counts = {"calls": 6912, "rows": {512: 1152, 1024: 5760}}
    validate_exclusion(packet(), counts)
    for changed in (
        {"calls": 0, "rows": {}},
        {"calls": 6911, "rows": {512: 1151, 1024: 5760}},
    ):
        with pytest.raises(ValueError):
            validate_exclusion(packet(), changed)
    escaped = packet()
    escaped["candidate_dispatch_shapes"][0]["arguments"] = [512, 10240, 320]
    with pytest.raises(ValueError, match="escaped"):
        validate_exclusion(escaped, counts)
    missing = packet()
    missing["candidate_dispatch_calls"] = 0
    with pytest.raises(ValueError):
        validate_exclusion(missing, counts)
