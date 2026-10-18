"""The advertised decode route and its opt-in batch envelope match execution.

These are unit tests: they drive the shared probe and the capabilities summary
with fake engines, so no GPU, server process, or model load is involved. The
live agreement between ``/v1/hipengine/decode`` and ``/v1/hipengine/capabilities``
is asserted in ``test_integration_server_api.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_ar_batch import (
    AR_BATCH_ENV,
    AR_BATCH_QUALIFIED_MAX_POSITIONS,
    AR_BATCH_QUALIFIED_MAX_SLOTS,
    AR_BATCH_SLOTS_ENV,
)
from hipengine.server import api as api_module
from hipengine.server.api import _decode_capability_summary, _probe_decode_telemetry


class _Engine:
    """An engine whose generator exposes the decode hook."""

    def __init__(self, hook: object) -> None:
        self._text_generator = SimpleNamespace(decode_telemetry=hook)


def _payload(value: object) -> object:
    return lambda: value


def test_probe_reports_an_engine_without_the_hook() -> None:
    assert _probe_decode_telemetry(None) == {
        "available": False,
        "reason": "engine_has_no_decode_telemetry",
    }
    assert _probe_decode_telemetry(SimpleNamespace()) == {
        "available": False,
        "reason": "engine_has_no_decode_telemetry",
    }


def test_probe_reaches_the_generator_hop_and_a_direct_engine_hook() -> None:
    through_generator = _Engine(_payload({"decode_path": "generator_path"}))
    assert _probe_decode_telemetry(through_generator) == {
        "available": True,
        "decode": {"decode_path": "generator_path"},
    }
    direct = SimpleNamespace(decode_telemetry=_payload({"decode_path": "direct_path"}))
    assert _probe_decode_telemetry(direct) == {
        "available": True,
        "decode": {"decode_path": "direct_path"},
    }


@pytest.mark.parametrize("bad", [None, ["decode_path"], "decode_path", 3])
def test_probe_refuses_a_payload_that_is_not_a_mapping(bad: object) -> None:
    assert _probe_decode_telemetry(_Engine(_payload(bad))) == {
        "available": False,
        "reason": "decode_telemetry_malformed",
    }


def test_probe_sanitizes_a_failing_hook() -> None:
    def boom() -> dict[str, object]:
        raise RuntimeError("slot 3 at 0x7fdeadbeef leaked")

    probed = _probe_decode_telemetry(_Engine(boom))
    assert probed == {
        "available": False,
        "reason": "decode_telemetry_unavailable",
    }
    assert "0x7fdeadbeef" not in repr(probed)


def test_summary_reports_the_envelope_before_the_first_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uninitialized runner still advertises the build's qualification bound."""

    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    monkeypatch.delenv(AR_BATCH_SLOTS_ENV, raising=False)
    summary = _decode_capability_summary(
        _Engine(_payload({"decode_path": "uninitialized", "ar_batch": None}))
    )

    assert summary["available"] is True
    assert summary["telemetry_url"] == "/v1/hipengine/decode"
    assert summary["path"] == "uninitialized"
    envelope = summary["ar_batch"]
    assert envelope["enabled"] is False
    assert envelope["env"] == AR_BATCH_ENV
    assert envelope["slots_env"] == AR_BATCH_SLOTS_ENV
    assert envelope["configured_slots"] is None
    assert envelope["invalid_configuration"] is None
    assert envelope["qualified_max_slots"] == AR_BATCH_QUALIFIED_MAX_SLOTS
    assert envelope["qualified_max_positions"] == AR_BATCH_QUALIFIED_MAX_POSITIONS
    assert envelope["above_qualified_positions"] == "production_capacity_bound"
    assert envelope["mtp_rows_join_batch"] is False
    assert envelope["active"] is None


def test_summary_follows_the_switch_and_a_valid_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AR_BATCH_ENV, "1")
    monkeypatch.setenv(AR_BATCH_SLOTS_ENV, str(AR_BATCH_QUALIFIED_MAX_SLOTS))
    envelope = _decode_capability_summary(
        _Engine(_payload({"decode_path": "uninitialized", "ar_batch": None}))
    )["ar_batch"]

    assert envelope["enabled"] is True
    assert envelope["configured_slots"] == AR_BATCH_QUALIFIED_MAX_SLOTS
    assert envelope["invalid_configuration"] is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", f"{AR_BATCH_SLOTS_ENV}=0 is outside 1..{AR_BATCH_QUALIFIED_MAX_SLOTS}"),
        ("9", f"{AR_BATCH_SLOTS_ENV}=9 is outside 1..{AR_BATCH_QUALIFIED_MAX_SLOTS}"),
        ("two", f"{AR_BATCH_SLOTS_ENV} is not an integer"),
    ],
)
def test_summary_reports_an_invalid_width_without_raising(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    """Capabilities must not turn a bad operator setting into a 500.

    The request path still fails closed when it resolves the width; this surface
    only reports what the setting is and why it cannot be used.
    """

    monkeypatch.setenv(AR_BATCH_ENV, "1")
    monkeypatch.setenv(AR_BATCH_SLOTS_ENV, raw)
    envelope = _decode_capability_summary(
        _Engine(_payload({"decode_path": "uninitialized", "ar_batch": None}))
    )["ar_batch"]

    assert envelope["configured_slots"] is None
    assert envelope["invalid_configuration"] == expected


def test_summary_passes_the_live_batch_state_through() -> None:
    live = {"state_slots": 2, "steps": 3, "step_widths": {"2": 3}, "live_slots": 0}
    summary = _decode_capability_summary(
        _Engine(
            _payload({"decode_path": "qwen4exp_full_model_ar_batch", "ar_batch": live})
        )
    )

    assert summary["path"] == "qwen4exp_full_model_ar_batch"
    assert summary["ar_batch"]["active"] == live


def test_summary_does_not_lend_the_envelope_to_another_provider() -> None:
    """A provider whose telemetry has no batch key keeps no invented envelope."""

    summary = _decode_capability_summary(
        _Engine(_payload({"decode_path": "some_other_engine_step"}))
    )

    assert summary["available"] is True
    assert summary["path"] == "some_other_engine_step"
    assert summary["ar_batch"] is None


def test_summary_reports_an_unavailable_probe_without_inventing_a_path() -> None:
    summary = _decode_capability_summary(SimpleNamespace())

    assert summary["available"] is False
    assert summary["reason"] == "engine_has_no_decode_telemetry"
    assert summary["ar_batch"] is None
    assert "path" not in summary


def test_summary_uses_the_shared_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capabilities must not re-implement the probe the endpoint uses."""

    calls: list[object] = []
    live = {"decode_path": "qwen4exp_full_model_ar_batch", "ar_batch": {"steps": 7}}

    def fake_probe(engine: object) -> dict[str, object]:
        calls.append(engine)
        return {"available": True, "decode": live}

    monkeypatch.setattr(api_module, "_probe_decode_telemetry", fake_probe)
    engine = SimpleNamespace()
    summary = api_module._decode_capability_summary(engine)

    assert calls == [engine]
    assert summary["path"] == "qwen4exp_full_model_ar_batch"
    assert summary["ar_batch"]["active"] == {"steps": 7}
