"""The decode-graph node's public ``LLM`` owner is released on every exit path.

``test_gpu_gemma4_decode_graph_capture`` loads a 17 GB model through the public
``hipengine.LLM`` surface. When that owner is not closed, the model weights and
KV cache stay resident for the rest of the process — measured at ~28.5 GB of
tracked device allocations — which is enough to make a later large-model load in
the same process fail with a HIP out-of-memory error.

This CPU regression pins the failure-safe close on the shared owner helper and
the wiring that makes the GPU node use it, without needing a device.
"""

from __future__ import annotations

import inspect

import pytest


class _FakeLLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _install_fake_llm(monkeypatch):
    import hipengine

    created: list[_FakeLLM] = []

    def factory(*, model: str) -> _FakeLLM:
        llm = _FakeLLM(model)
        created.append(llm)
        return llm

    monkeypatch.setattr(hipengine, "LLM", factory)
    return created


def test_owner_closes_on_normal_exit(monkeypatch) -> None:
    from tests import test_gpu_gemma4_decode_graph_capture as node

    created = _install_fake_llm(monkeypatch)

    with node._owned_llm("model") as llm:
        assert llm is created[-1]
        assert llm.closed == 0

    assert created[-1].closed == 1


def test_owner_closes_when_the_body_raises(monkeypatch) -> None:
    """A failure inside the owned block must still release the model owner."""

    from tests import test_gpu_gemma4_decode_graph_capture as node

    created = _install_fake_llm(monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        with node._owned_llm("model"):
            raise RuntimeError("boom")

    assert created[-1].closed == 1


def test_owner_closes_when_construction_succeeds_but_setup_fails(monkeypatch) -> None:
    """The close runs even when the owned block raises before doing any work."""

    from tests import test_gpu_gemma4_decode_graph_capture as node

    created = _install_fake_llm(monkeypatch)

    with pytest.raises(AssertionError):
        with node._owned_llm("model") as llm:
            assert llm.closed == 1  # no-op assertion that fails immediately

    assert created[-1].closed == 1


def test_gpu_node_requests_the_owner_fixture() -> None:
    """The node must request the failure-safe owner, not build a bare ``LLM``."""

    from tests import test_gpu_gemma4_decode_graph_capture as node

    params = inspect.signature(
        node.test_gemma4_decode_graph_replay_matches_launched_decode
    ).parameters
    assert "gemma4_llm" in params
