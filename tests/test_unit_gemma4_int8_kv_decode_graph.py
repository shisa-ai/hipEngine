"""The decode-graph session refuses INT8 KV storage up front.

The direct INT8 consumer validates its live counts, row positions and page
table with a synchronous device-to-host read. That read cannot be recorded into
a HIP graph, so capture is genuinely unsupported on the INT8 path -- not a
missing qualification. The session must say so before it stages anything or
launches a kernel, and it must not quietly downgrade the storage the caller
requested. The requested INT8 cache still runs without capture.

No device: the session's runtime is a stub that only answers ``stream_create``.
"""

from __future__ import annotations

import pytest

from hipengine.runtime import gemma4_decode_graph as graph_module
from hipengine.runtime.gemma4_decode_graph import Gemma4DecodeGraphSession


class _StubRuntime:
    def stream_create(self, **kwargs: object) -> int:
        return 1

    def stream_destroy(self, stream: int) -> None:  # pragma: no cover - not reached
        pass


class _Int8RunnerStub:
    """Weak-referenceable stand-in; ``step`` reads only the storage flag first."""

    uses_int8_kv = True


@pytest.fixture
def _stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "get_hip_runtime", lambda: _StubRuntime())


def test_capture_is_refused_for_int8_kv_before_any_staging(
    _stub_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _Int8RunnerStub()
    session = Gemma4DecodeGraphSession(runner)

    staged: list[object] = []
    runner._stage_block_content = lambda *a, **k: staged.append((a, k))
    monkeypatch.setattr(
        graph_module.Gemma4DecodeGraphSession,
        "_capture",
        lambda *a, **k: pytest.fail("capture must not be attempted for INT8 KV"),
    )

    with pytest.raises(RuntimeError, match="decode-graph capture is not supported"):
        session.step(0)
    assert staged == [], "nothing may be staged before the refusal"


def test_capture_refusal_names_the_clearing_alternative(_stub_runtime: None) -> None:
    runner = _Int8RunnerStub()
    session = Gemma4DecodeGraphSession(runner)
    with pytest.raises(RuntimeError) as excinfo:
        session.step(3)
    message = str(excinfo.value)
    # A gate nothing can clear is a bug in the gate: the message names both
    # routes that make the request runnable.
    assert "Run without capture" in message
    assert "bf16" in message
