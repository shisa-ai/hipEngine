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


def test_capture_window_is_the_backends_declared_value_else_the_default(monkeypatch):
    from hipengine.kernels import backends
    from hipengine.runtime import gemma4_decode_graph as graph

    declared = {"hip_gfx1100": 320}
    monkeypatch.setattr(
        backends, "backend_package_capability",
        lambda backend, name, default=None: declared.get(backend, default)
        if name == graph.CAPTURE_MAX_KEYS_CAPABILITY else default,
    )
    assert graph._capture_max_keys("hip_gfx1100") == 320
    assert graph._capture_max_keys("hip_gfx1151") == graph._CAPTURE_MAX_KEYS
    # A runner that names no backend keeps the default window.
    assert graph._capture_max_keys(None) == graph._CAPTURE_MAX_KEYS


@pytest.mark.parametrize(("window", "position", "captures"), [(0, 4, False), (64, 4, False),
                                                              (65, 4, True), (2176, 2174, True),
                                                              (2176, 2175, False)])
def test_step_stays_launched_at_and_past_the_window(monkeypatch, window, position, captures):
    from hipengine.runtime import gemma4_decode_graph as graph

    staged, launched = [], []
    runner = SimpleNamespace(
        uses_int8_kv=False, position=position, capacity=4096,
        weights=SimpleNamespace(layers=[], backend="hip_test",
                                config=SimpleNamespace(vocab_size=10)),
        _last_logits_rows=0, _normalized_hidden_rows=0,
        forward_argmax=lambda tokens, **kw: launched.append(tokens) or 3,
        _stage_block_content=lambda tokens, **kw: staged.append(kw) or ({}, {}),
        _collect_block_argmax=lambda *a, **kw: 5,
    )
    monkeypatch.setattr(graph, "_capture_max_keys", lambda backend: window)
    monkeypatch.setattr(graph, "get_hip_runtime",
                        lambda: SimpleNamespace(graph_launch=lambda *a: None))
    session = object.__new__(graph.Gemma4DecodeGraphSession)
    session._runner_ref = lambda: runner
    session._stream, session._exec, session._key = 2, 7, ("reused",)
    session._appends, session._captures, session._launched_fallbacks = (), 0, 0
    session._capture_key = lambda *a: ("reused",)
    session._retarget_appends = lambda pos: None

    token = session.step_argmax(1)

    assert (token, bool(staged), bool(launched)) == ((5, True, False) if captures
                                                     else (3, False, True))
    assert session.launched_fallbacks == (0 if captures else 1)
