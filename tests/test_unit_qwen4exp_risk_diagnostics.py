"""Focused CPU tests for the opt-in iu8 risk+repair instrumentation.

The readback is diagnostic only: it must be inert unless the env gate is set,
must record one observation per repair call, and must aggregate into repair
rates that rank candidate families by how much of the fast kernel survives.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen4_exp_runner as runner


class FakeBuffer:
    def __init__(self, nbytes: int) -> None:
        self.ptr = 0x1000
        self.nbytes = nbytes


class FakeRuntime:
    def __init__(self) -> None:
        self.syncs: list[int] = []

    def stream_synchronize(self, stream: int) -> None:
        self.syncs.append(stream)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(runner.RISK_DIAGNOSTICS_ENV, raising=False)
    runner.reset_qwen4_exp_risk_diagnostics()
    yield
    runner.reset_qwen4_exp_risk_diagnostics()


def _observe(runtime, *, risk: int, compact: int = 4, features: int = 8,
             layer: str = "layers.35.ffn_gate_exps", rows: int = 4,
             capacity: int | None = None, stream: int = 7,
             monkeypatch: pytest.MonkeyPatch | None = None):
    def fake_copy(host_ptr, buffer, nbytes=None, *, runtime=None):
        np.ctypeslib.as_array(
            (np.ctypeslib.ctypes.c_int32 * 1).from_address(host_ptr)
        )[0] = risk

    if monkeypatch is not None:
        monkeypatch.setattr(runner, "copy_device_to_host", fake_copy)
    runner._observe_qwen4_exp_risk_queue(
        runtime=runtime,
        stream=stream,
        risk_count=FakeBuffer(DType.INT32.itemsize),
        route="q4_iu8_exact",
        role="expert_gate_up",
        layer=layer,
        rows=rows,
        compact_rows=compact,
        out_features_total=features,
        experts=8,
        risk_capacity=(compact * features if capacity is None else capacity),
    )


def test_disabled_by_default_records_nothing_and_never_syncs(monkeypatch):
    runtime = FakeRuntime()
    _observe(runtime, risk=3, monkeypatch=monkeypatch)
    assert runner.qwen4_exp_risk_diagnostics() == ()
    assert runtime.syncs == []
    assert runner.qwen4_exp_risk_diagnostics_enabled() is False


def test_enabled_readback_records_repair_rate(monkeypatch):
    monkeypatch.setenv(runner.RISK_DIAGNOSTICS_ENV, "1")
    runtime = FakeRuntime()
    _observe(runtime, risk=6, compact=4, features=8, monkeypatch=monkeypatch)
    records = runner.qwen4_exp_risk_diagnostics()
    assert len(records) == 1
    record = records[0]
    assert record["route"] == "q4_iu8_exact"
    assert record["role"] == "expert_gate_up"
    assert record["layer"] == "layers.35.ffn_gate_exps"
    assert record["risk"] == 6
    assert record["outputs"] == 32
    assert record["repair_rate"] == pytest.approx(6 / 32)
    assert record["over_capacity"] is False
    # The diagnostic synchronizes the caller's stream exactly once.
    assert runtime.syncs == [7]


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_enabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv(runner.RISK_DIAGNOSTICS_ENV, value)
    assert runner.qwen4_exp_risk_diagnostics_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "False"])
def test_disabled_falsy_values(monkeypatch, value):
    monkeypatch.setenv(runner.RISK_DIAGNOSTICS_ENV, value)
    assert runner.qwen4_exp_risk_diagnostics_enabled() is False


def test_over_capacity_is_flagged_not_silently_clamped(monkeypatch):
    monkeypatch.setenv(runner.RISK_DIAGNOSTICS_ENV, "1")
    _observe(FakeRuntime(), risk=9, compact=4, features=8, capacity=8,
             monkeypatch=monkeypatch)
    record = runner.qwen4_exp_risk_diagnostics()[0]
    assert record["over_capacity"] is True
    # The reported rate is the raw measurement, not a clamped one.
    assert record["repair_rate"] == pytest.approx(9 / 32)


def test_summary_aggregates_by_role_and_layer(monkeypatch):
    monkeypatch.setenv(runner.RISK_DIAGNOSTICS_ENV, "1")
    for risk, layer in ((0, "layers.35.ffn_gate_exps"),
                        (4, "layers.35.ffn_gate_exps"),
                        (32, "layers.36.ffn_gate_exps")):
        _observe(FakeRuntime(), risk=risk, layer=layer, monkeypatch=monkeypatch)
    summary = runner.summarize_qwen4_exp_risk_diagnostics()
    assert summary["calls"] == 3
    role = summary["by_role"]["q4_iu8_exact:expert_gate_up"]
    assert role["calls"] == 3
    assert role["outputs"] == 96
    assert role["risk"] == 36
    assert role["aggregate_repair_rate"] == pytest.approx(36 / 96)
    assert role["max_repair_rate"] == pytest.approx(1.0)
    assert role["over_capacity_calls"] == 0
    layer35 = summary["by_layer"]["q4_iu8_exact:expert_gate_up:"
                                   "layers.35.ffn_gate_exps"]
    assert layer35["calls"] == 2
    assert layer35["risk"] == 4
    assert layer35["aggregate_repair_rate"] == pytest.approx(4 / 64)


def test_summary_accepts_explicit_records_and_handles_zero_outputs():
    summary = runner.summarize_qwen4_exp_risk_diagnostics(
        [{"route": "r", "role": "x", "layer": "l", "outputs": 0, "risk": 0}]
    )
    entry = summary["by_role"]["r:x"]
    assert entry["aggregate_repair_rate"] is None
    assert entry["max_repair_rate"] == 0.0


def test_record_limit_bounds_memory(monkeypatch):
    monkeypatch.setenv(runner.RISK_DIAGNOSTICS_ENV, "1")
    monkeypatch.setattr(runner, "_RISK_DIAGNOSTIC_LIMIT", 3)
    for _ in range(5):
        _observe(FakeRuntime(), risk=1, monkeypatch=monkeypatch)
    assert len(runner.qwen4_exp_risk_diagnostics()) == 3
