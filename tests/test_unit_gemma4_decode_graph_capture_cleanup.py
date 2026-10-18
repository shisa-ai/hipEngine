"""A failed decode-graph capture must not poison the stream or leak a graph.

The capture records a full decode step inside ``hipStreamBeginCapture`` /
``hipStreamEndCapture``. A launch that raises mid-recording -- a kernel
rejecting an argument, an unexpected memcpy, the host synchronize the
MMQ-family leaves used to perform -- leaves the stream capturing. The session
must end the capture and destroy whatever partial graph HIP hands back before
it re-raises, or the next capture on the same stream fails with HIP 904.

No device: the runtime is a stub that answers only the capture calls the
failure path touches, and the attention workspace pre-touch is replaced.
"""

from __future__ import annotations

import pytest

from hipengine.core.hip import HipError
from hipengine.runtime import gemma4_decode_graph as graph_module
from hipengine.runtime.gemma4_decode_graph import Gemma4DecodeGraphSession, _Bucket


class _StubRuntime:
    def __init__(self, *, end_raises: bool = False) -> None:
        self.events: list[tuple[str, int]] = []
        self.destroyed: list[int] = []
        self._end_raises = end_raises

    def graph_exec_destroy(self, handle: int) -> None:
        self.events.append(("exec_destroy", handle))

    def stream_create(self, **kwargs: object) -> int:
        return 9

    def stream_destroy(self, stream: int) -> None:  # pragma: no cover - not reached
        self.events.append(("stream_destroy", stream))

    def graph_destroy(self, handle: int) -> None:
        self.destroyed.append(handle)
        self.events.append(("graph_destroy", handle))

    def stream_begin_capture(self, stream: int, mode: int = 2) -> None:
        self.events.append(("begin", stream))

    def stream_end_capture(self, stream: int) -> int:
        self.events.append(("end", stream))
        if self._end_raises:
            # What HIP returns when the failed launch already invalidated the
            # capture. The cleanup must swallow it and still re-raise the
            # original failure rather than masking it with the teardown error.
            raise HipError(904, "operation not permitted when stream is capturing")
        return 0xABC  # a partial graph to destroy


class _RunnerStub:
    """Weak-referenceable stand-in with an empty layer list.

    ``_capture`` iterates ``runner.weights.layers`` to pre-touch attention
    scratch; an empty tuple skips the loop while still exercising the capture
    begin/launch/end sequence.
    """

    def __init__(self) -> None:
        self.position = 5
        self.capacity = 64
        self.weights = _Weights()

    def _launch_block(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("launch failed mid-capture")


class _Weights:
    layers: tuple[object, ...] = ()


@pytest.fixture
def _stub_runtime(monkeypatch: pytest.MonkeyPatch):
    def install(runtime: _StubRuntime) -> _StubRuntime:
        monkeypatch.setattr(graph_module, "get_hip_runtime", lambda: runtime)
        monkeypatch.setattr(
            "hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention.build_gemma4_attention",
            lambda **kwargs: object(),
        )
        return runtime

    return install


def _session(runner: _RunnerStub, runtime: _StubRuntime) -> Gemma4DecodeGraphSession:
    session = Gemma4DecodeGraphSession(runner)
    session._stream = 9
    # Keep the runner alive: the session holds it weakly.
    session._keep_alive = runner
    return session


def test_failed_capture_ends_the_capture_and_destroys_the_partial_graph(
    _stub_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _stub_runtime(_StubRuntime())
    session = _session(_RunnerStub(), runtime)

    with pytest.raises(RuntimeError, match="launch failed mid-capture"):
        session._capture(_Bucket(4, 8), {}, {}, ("key",), token=1)

    # The capture was begun, then ended once to reclaim the partial graph, and
    # that graph was destroyed. The session holds no dangling handles.
    assert runtime.events == [
        ("begin", 9),
        ("end", 9),
        ("graph_destroy", 0xABC),
    ]
    assert runtime.destroyed == [0xABC]
    assert session._graph == 0 and session._exec == 0


def test_cleanup_survives_an_end_capture_that_reports_the_capture_already_invalidated(
    _stub_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _stub_runtime(_StubRuntime(end_raises=True))
    session = _session(_RunnerStub(), runtime)

    # The original failure must reach the caller; the 904 from ending an
    # already-invalidated capture is teardown noise, not the error to report.
    with pytest.raises(RuntimeError, match="launch failed mid-capture"):
        session._capture(_Bucket(4, 8), {}, {}, ("key",), token=1)

    assert runtime.events == [("begin", 9), ("end", 9)]
    assert runtime.destroyed == []
    assert session._graph == 0 and session._exec == 0
