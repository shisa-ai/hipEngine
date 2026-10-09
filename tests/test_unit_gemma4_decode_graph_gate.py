"""The generator's decode-graph gate: env, INT8 refusal, and runner keying."""

import os
from types import SimpleNamespace

import pytest

from hipengine.generation import gemma4_gguf


@pytest.fixture()
def generator():
    gen = object.__new__(gemma4_gguf.Gemma4GGUFGenerator)
    gen._decode_graph_session = None
    gen._decode_graph_runner = None
    return gen


def _fake_session(runner):
    holder = SimpleNamespace(closed=False, runner=runner)

    class _Session:
        def __init__(self, r):
            self.runner = r

        def close(self):
            holder.closed = True

    return _Session(runner)


def test_env_off_returns_none(generator, monkeypatch):
    runner = SimpleNamespace(uses_int8_kv=False)
    monkeypatch.setattr(gemma4_gguf, "Gemma4DecodeGraphSession", None, raising=False)
    monkeypatch.setenv("HIPENGINE_GEMMA4_DECODE_GRAPH", "0")
    assert generator._decode_graph_session_for(runner) is None


def test_int8_kv_runner_returns_none(generator, monkeypatch):
    # INT8 KV storage needs a checked readback a capture cannot hold; the
    # requested storage must keep running its launched step, not be
    # downgraded or captured.
    runner = SimpleNamespace(uses_int8_kv=True)
    monkeypatch.delenv("HIPENGINE_GEMMA4_DECODE_GRAPH", raising=False)
    assert generator._decode_graph_session_for(runner) is None


def test_default_on_creates_one_session_keyed_to_runner(generator, monkeypatch):
    monkeypatch.delenv("HIPENGINE_GEMMA4_DECODE_GRAPH", raising=False)
    monkeypatch.setattr(
        "hipengine.runtime.gemma4_decode_graph.Gemma4DecodeGraphSession",
        _fake_session,
    )
    runner = SimpleNamespace(uses_int8_kv=False)
    session = generator._decode_graph_session_for(runner)
    assert session.runner is runner
    # Same runner: the cached session is reused, not rebuilt.
    assert generator._decode_graph_session_for(runner) is session
    # A rebuilt runner (KV-storage change): the stale session is closed and
    # a fresh one captures against the new runner.
    runner2 = SimpleNamespace(uses_int8_kv=False)
    session2 = generator._decode_graph_session_for(runner2)
    assert session2.runner is runner2
    assert generator._decode_graph_session_for(runner) is not session2
