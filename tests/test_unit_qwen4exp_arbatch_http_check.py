"""Unit contracts for the AR-batch HTTP harness's telemetry scoring.

The live check runs against a real server; these tests pin the pure helpers it
uses to turn two telemetry snapshots into a physical-width delta, so a broken
or unavailable payload cannot be scored as a passing phase.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_ar_batch_http_check.py"


def _load_harness():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_ar_batch_http_check", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _body(decode: object, *, available: bool = True) -> dict:
    return {"object": "hipengine.decode_telemetry", "available": available, "decode": decode}


def test_width_delta_reports_per_width_step_counts() -> None:
    before = _body({"ar_batch": {"step_widths": {"1": 2}}})
    after = _body({"ar_batch": {"step_widths": {"1": 2, "4": 17}}})
    assert harness.width_delta(
        harness.require_telemetry(before), harness.require_telemetry(after)
    ) == {"1": 0, "4": 17}


def test_width_delta_ignores_widths_absent_from_the_later_snapshot() -> None:
    before = harness.require_telemetry(_body({"ar_batch": {"step_widths": {"2": 5}}}))
    after = harness.require_telemetry(_body({"ar_batch": {"step_widths": {}}}))
    assert harness.width_delta(before, after) == {}


def test_batch_state_tolerates_a_missing_ar_batch() -> None:
    assert harness.batch_state(harness.require_telemetry(_body({}))) == {}
    assert harness.batch_state(
        harness.require_telemetry(_body({"ar_batch": None}))
    ) == {}


@pytest.mark.parametrize(
    "body",
    [
        ["not", "an", "object"],
        "not an object",
        {"available": False, "reason": "engine_has_no_decode_telemetry"},
        {"available": "true", "decode": {}},
        {"object": "hipengine.decode_telemetry", "available": True},
        {"available": True, "decode": "not an object"},
        {"available": True, "decode": {"ar_batch": "not an object"}},
    ],
)
def test_require_telemetry_rejects_unavailable_or_malformed_payloads(body) -> None:
    with pytest.raises(RuntimeError):
        harness.require_telemetry(body)


def test_require_telemetry_keeps_the_servers_error_reason_out_of_scoring() -> None:
    body = {
        "object": "hipengine.decode_telemetry",
        "available": False,
        "reason": "decode_telemetry_unavailable",
    }
    with pytest.raises(RuntimeError, match="decode_telemetry_unavailable"):
        harness.require_telemetry(body)
