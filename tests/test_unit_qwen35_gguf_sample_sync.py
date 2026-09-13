from __future__ import annotations

from types import SimpleNamespace

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


class _Runtime:
    def __init__(self, calls: list[tuple[str, int]]) -> None:
        self.calls = calls

    def device_synchronize(self) -> None:
        self.calls.append(("device_synchronize", 0))

    def stream_synchronize(self, stream: int) -> None:
        self.calls.append(("stream_synchronize", int(stream)))


def _session(calls: list[tuple[str, int]]) -> Qwen35GGUFResidentSession:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = _Runtime(calls)
    session._sample_device_from_hidden = lambda hidden_ptr, *, stream=0: calls.append(
        ("sample_device", int(stream))
    )
    session._read_sample = lambda *, return_logits: (
        calls.append(("blocking_read", int(return_logits)))
        or SimpleNamespace(token_id=9707)
    )
    return session


def test_default_stream_sample_uses_blocking_readback_as_completion_boundary() -> None:
    calls: list[tuple[str, int]] = []

    result = Qwen35GGUFResidentSession._sample_from_hidden(
        _session(calls),
        0x1234,
        return_logits=False,
        stream=0,
    )

    assert result.token_id == 9707
    assert calls == [("sample_device", 0), ("blocking_read", 0)]


def test_non_default_stream_sample_retains_explicit_stream_synchronization() -> None:
    calls: list[tuple[str, int]] = []

    result = Qwen35GGUFResidentSession._sample_from_hidden(
        _session(calls),
        0x1234,
        return_logits=True,
        stream=7,
    )

    assert result.token_id == 9707
    assert calls == [
        ("sample_device", 7),
        ("stream_synchronize", 7),
        ("blocking_read", 1),
    ]
