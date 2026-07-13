"""Unit coverage for GGUF AOTriton queue isolation."""

from __future__ import annotations

import inspect

import pytest

from hipengine.runtime import qwen35_gguf_runner as runner_module
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)


def test_gguf_aotriton_isolated_stream_policy_is_default_on_only_for_gfx1151(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM", raising=False)
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1151", 511) is False
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1151", 512) is True
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1100", 4096) is False

    monkeypatch.setenv("HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM", "0")
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1151", 4096) is False

    monkeypatch.setenv("HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM", "1")
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1151", 4096) is True
    assert runner_module._gguf_aotriton_isolated_prefill_stream_applies("hip_gfx1100", 4096) is True


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
