"""The capacity probe must be able to certify a non-default prefill chunk.

The chunk lever (worklog entry 20260921T155134.440179Z) trades activation
scratch for routed-MoE weight traffic, so its gating question is a capacity
question: does a 4096-row chunk still fit at the target context? A probe that
can only express ``PrefillConfig()``'s 1024/1024 defaults cannot answer it.

These tests pin the mapping from the probe's flag to the session's config, and
they do it without loading a model: the probe's own session class is replaced
with a recorder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gguf_capacity_probe as probe  # noqa: E402


class _Recorder:
    """Stand-in for Qwen35GGUFResidentSession that records its own config."""

    captured: dict = {}

    def __init__(self, model, **kwargs):
        type(self).captured = {"model": model, **kwargs}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run(monkeypatch, argv: list[str]) -> dict:
    import numpy as np

    from hipengine.core import hip as hip_module
    from hipengine.core import memory as memory_module
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime import qwen35_gguf_runner as runner_module

    monkeypatch.setattr(hip_module, "get_hip_runtime", lambda: object())
    monkeypatch.setattr(
        memory_module,
        "memory_stats",
        lambda: {"current_allocated_bytes": 0, "peak_allocated_bytes": 0},
    )
    monkeypatch.setattr(memory_module, "reset_memory_stats", lambda: None)
    monkeypatch.setattr(
        runner_module,
        "Qwen35GGUFResidentSession",
        _Recorder,
    )

    class _Logits:
        logits = np.zeros((1, 4), dtype=np.float32)
        token_id = 1

    class _Session(_Recorder):
        scratch = None

        def prefill(self, ids, **kwargs):
            return _Logits()

        def step(self, token, **kwargs):
            return _Logits()

    monkeypatch.setattr(runner_module, "Qwen35GGUFResidentSession", _Session)
    monkeypatch.setattr(probe, "resolve_kv_policy", resolve_kv_policy, raising=False)
    monkeypatch.setattr(sys, "argv", ["gguf_capacity_probe.py", *argv])
    assert probe.main() == 0
    return _Session.captured


def test_default_keeps_the_shipped_chunk_sizes(monkeypatch):
    captured = _run(
        monkeypatch,
        ["--max-sequence-length", "4096", "--model", "/nonexistent.gguf"],
    )
    config = captured["prefill_config"]
    assert config.linear_chunk_size == 0
    assert config.moe_chunk_size == 0


def test_chunk_flag_raises_every_chunk_sized_knob(monkeypatch):
    """The runner takes the smallest positive chunk, so one alone is not enough."""

    captured = _run(
        monkeypatch,
        [
            "--max-sequence-length",
            "4096",
            "--model",
            "/nonexistent.gguf",
            "--prefill-chunk-size",
            "4096",
        ],
    )
    config = captured["prefill_config"]
    assert config.linear_chunk_size == 4096
    assert config.moe_chunk_size == 4096
    assert config.full_attn_post_chunk_size == 4096
    assert config.full_attn_rope_chunk_size == 4096
    # The query chunk is not a per-row scratch knob and must stay at its own
    # default; raising it would probe a different allocation.
    assert config.full_attn_query_chunk_size == 0
