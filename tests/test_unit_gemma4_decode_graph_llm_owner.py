"""The decode-graph node's public ``LLM`` owner is released on every exit path.

``test_gpu_gemma4_decode_graph_capture`` loads a 17 GB model through the public
``hipengine.LLM`` surface. When that owner is not closed, the model weights and
KV cache stay resident for the rest of the process — measured at ~28.5 GB of
tracked device allocations — which is enough to make a later large-model load in
the same process fail with a HIP out-of-memory error.

This CPU regression pins the failure-safe close on the shared owner helper and
the wiring that makes the GPU node use it, without needing a device. The helper
lives in ``tests/_gemma4_decode_graph_llm.py`` rather than the GPU-tier test
module, so the unit tier does not import an expensive test (the tier gate in
``scripts/check_test_tiers.py`` forbids that); the wiring check reads the GPU
node's source instead of importing it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._gemma4_decode_graph_llm import owned_llm

_GPU_NODE = Path(__file__).parent / "test_gpu_gemma4_decode_graph_capture.py"


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
    created = _install_fake_llm(monkeypatch)

    with owned_llm("model") as llm:
        assert llm is created[-1]
        assert llm.closed == 0

    assert created[-1].closed == 1


def test_owner_closes_when_the_body_raises(monkeypatch) -> None:
    """A failure inside the owned block must still release the model owner."""

    created = _install_fake_llm(monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        with owned_llm("model"):
            raise RuntimeError("boom")

    assert created[-1].closed == 1


def test_owner_closes_when_construction_succeeds_but_setup_fails(monkeypatch) -> None:
    """The close runs even when the owned block raises before doing any work."""

    created = _install_fake_llm(monkeypatch)

    with pytest.raises(AssertionError):
        with owned_llm("model") as llm:
            assert llm.closed == 1  # no-op assertion that fails immediately

    assert created[-1].closed == 1


def _gpu_node_functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(_GPU_NODE.read_text())
    return {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }


def test_gpu_node_requests_the_owner_fixture() -> None:
    """The node must request the failure-safe owner, not build a bare ``LLM``."""

    functions = _gpu_node_functions()
    target = functions["test_gemma4_decode_graph_replay_matches_launched_decode"]
    params = {arg.arg for arg in target.args.args}
    params |= {arg.arg for arg in target.args.kwonlyargs}
    assert "gemma4_llm" in params


def test_gpu_node_builds_the_owner_through_the_shared_helper() -> None:
    """The requested fixture must open the model through ``owned_llm``."""

    source = _GPU_NODE.read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "tests._gemma4_decode_graph_llm"
        for alias in node.names
    }
    assert "owned_llm" in imported
    fixture = _gpu_node_functions()["gemma4_llm"]
    called = {
        node.func.id
        for node in ast.walk(fixture)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "owned_llm" in called
