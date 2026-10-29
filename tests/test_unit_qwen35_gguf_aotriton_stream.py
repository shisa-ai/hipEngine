"""Unit coverage for GGUF AOTriton queue isolation."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_paro_runner as paro_module
from hipengine.runtime import qwen35_gguf_runner as runner_module
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)


def test_gpf5a_two_wave_prefill_policy_uses_backend_specific_ceiling() -> None:
    assert runner_module._gguf_q8_t16_two_wave_prefill_applies("hip_gfx1151", 512) is True
    assert runner_module._gguf_q8_t16_two_wave_prefill_applies("hip_gfx1151", 65536) is True
    assert runner_module._gguf_q8_t16_two_wave_prefill_applies("hip_gfx1151", 65537) is False
    assert runner_module._gguf_q8_t16_two_wave_prefill_applies("hip_gfx1100", 4096) is True
    assert runner_module._gguf_q8_t16_two_wave_prefill_applies("hip_gfx1100", 4097) is False


def test_gguf_aotriton_isolated_stream_policy_is_opt_in_after_stability_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM", raising=False)
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1151", 511) is False
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1151", 512) is False
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1100", 4096) is False

    monkeypatch.setenv("HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM", "0")
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1151", 4096) is False

    monkeypatch.setenv("HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM", "1")
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1151", 4096) is True
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1100", 4096) is True


@pytest.mark.parametrize("session_class,default", [
    (paro_module.Qwen35ParoResidentSession, True), (Qwen35GGUFResidentSession, False),
])
@pytest.mark.parametrize("control", [None, "0", "1"])
@pytest.mark.parametrize("rows", [37, 511, 512, 777])
def test_shared_isolation_selector_preserves_family_defaults_and_row_gate(monkeypatch, session_class, default, control, rows):
    name = "HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM"
    if control is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, control)
    session = object.__new__(session_class)
    session.runner = SimpleNamespace(backend="hip_gfx1151")
    bridge = object()
    allocations = []
    session._ensure_prefill_aotriton_bridge = lambda: allocations.append(True) or bridge
    result = session._prefill_aotriton_bridge_for_rows(rows)
    expected = rows >= 512 and (default if control is None else control == "1")
    assert result is (bridge if expected else None)
    assert allocations == ([True] if expected else [])


@pytest.mark.parametrize("session_class", [paro_module.Qwen35ParoResidentSession, Qwen35GGUFResidentSession])
@pytest.mark.parametrize("failure_at", [1, 2])
def test_isolation_bridge_partial_allocation_is_cleaned_and_retryable(session_class, failure_at):
    session = object.__new__(session_class)
    calls = []

    class Runtime:
        def stream_create(self):
            calls.append(("stream_create", 7))
            return 7

        def event_create(self):
            index = sum(kind == "event_create" for kind, _ in calls) + 1
            calls.append(("event_create", index))
            if index == failure_at:
                raise RuntimeError("synthetic event failure")
            return 10 + index

        def event_destroy(self, handle):
            calls.append(("event_destroy", handle))

        def stream_destroy(self, handle):
            calls.append(("stream_destroy", handle))

    session.runtime = Runtime()
    with pytest.raises(RuntimeError, match="synthetic event failure"):
        session._ensure_prefill_aotriton_bridge()
    cleanup = [(kind, handle) for kind, handle in calls if kind.endswith("destroy")]
    assert cleanup == ([("event_destroy", 11)] if failure_at == 2 else []) + [("stream_destroy", 7)]
    assert not getattr(session, "_prefill_aotriton_stream", 0)
    assert not getattr(session, "_prefill_aotriton_input_ready_event", 0)
    assert not getattr(session, "_prefill_aotriton_output_ready_event", 0)
    retry_events = iter((22, 23))
    session.runtime.event_create = lambda: next(retry_events)
    assert session._ensure_prefill_aotriton_bridge().stream == 7
    session._release_prefill_aotriton_bridge()
    count = len(calls)
    session._release_prefill_aotriton_bridge()
    assert len(calls) == count


def test_gguf_full_attention_prefill_accepts_aotriton_stream_bridge() -> None:
    parameters = inspect.signature(
        Qwen35GGUFFullStackRunner._run_full_attention_prefill_layer_aotriton
    ).parameters
    assert "aotriton_bridge" in parameters


def test_gguf_resident_aotriton_bridge_is_reused_and_released() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    calls: list[tuple[str, int]] = []

    class FakeRuntime:
        def stream_create(self) -> int:
            calls.append(("stream_create", 7))
            return 7

        def event_create(self) -> int:
            handle = 11 + sum(name == "event_create" for name, _ in calls)
            calls.append(("event_create", handle))
            return handle

        def event_destroy(self, handle: int) -> None:
            calls.append(("event_destroy", int(handle)))

        def stream_destroy(self, handle: int) -> None:
            calls.append(("stream_destroy", int(handle)))

    session.runtime = FakeRuntime()

    first = session._ensure_prefill_aotriton_bridge()
    second = session._ensure_prefill_aotriton_bridge()

    assert first == second
    assert (first.stream, first.input_ready_event, first.output_ready_event) == (7, 11, 12)
    assert calls == [("stream_create", 7), ("event_create", 11), ("event_create", 12)]

    session._release_prefill_aotriton_bridge()

    assert calls[-3:] == [
        ("event_destroy", 12),
        ("event_destroy", 11),
        ("stream_destroy", 7),
    ]
