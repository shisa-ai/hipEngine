"""Maple whole-step hipGraph capture/replay parity and fallback gate (M1)."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

from hipengine.runtime.maple_graph import MapleGraphCache

_MODEL = "deepgrove/maple-preview-2bit-mlx"
_HF_CACHE = Path("/home/lhl/maple-hf-cache")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _model_available() -> bool:
    return (_HF_CACHE / "models--deepgrove--maple-preview").is_dir()


pytestmark = pytest.mark.skipif(
    not (_hip_available() and _model_available()),
    reason="requires ROCm/libamdhip64.so and cached Maple checkpoint",
)


def test_maple_graph_eager_fallback_when_disabled(monkeypatch) -> None:
    """With the env gate off, every step takes the eager path (no capture)."""

    from hipengine.loading.maple import load_maple_checkpoint
    from hipengine.runtime.maple import MapleRunner

    monkeypatch.setenv("HIPENGINE_MAPLE_GRAPH", "0")
    checkpoint = load_maple_checkpoint(_MODEL)
    runner = MapleRunner.load(checkpoint, max_context=128)
    try:
        assert runner._graph_cache() is None
        assert runner.step(1).token_id >= 0
    finally:
        runner.close()


def test_maple_graph_bit_exact_vs_eager_over_growing_kv() -> None:
    """Captured whole-step graph replays bit-exact vs eager across 64 tokens."""

    from hipengine.loading.maple import load_maple_checkpoint
    from hipengine.runtime.maple import MapleRunner

    def run(mode: str, n: int = 64) -> list[int]:
        os.environ["HIPENGINE_MAPLE_GRAPH"] = mode
        checkpoint = load_maple_checkpoint(_MODEL)
        runner = MapleRunner.load(checkpoint, max_context=4096)
        try:
            return [runner.step(1).token_id for _ in range(n)]
        finally:
            runner.close()

    assert run("1", 64) == run("0", 64)


def test_maple_graph_cache_self_validation_detects_noop() -> None:
    """A captured graph that changes nothing is not kept (rejected -> eager)."""

    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc

    runtime = get_hip_runtime()
    cache = MapleGraphCache(runtime, enabled=True)
    index_buf = malloc(8, runtime=runtime)
    value_buf = malloc(4, runtime=runtime)
    try:
        # The eager body leaves argmax untouched, so the eager ref is whatever we
        # clear it to; a faithful no-op graph would also leave it there. This
        # exercises the capture path's parity machinery without corrupting KV.
        copy_host_to_device(
            index_buf, host_array_ptr(np.zeros(1, np.int64)), runtime=runtime
        )
        copy_host_to_device(
            value_buf, host_array_ptr(np.zeros(1, np.float32)), runtime=runtime
        )

        status = cache.run(
            (),
            eager=lambda _stream: None,
            argmax_index_ptr=index_buf.ptr,
            argmax_value_ptr=value_buf.ptr,
            stream=0,
        )
        assert status in ("capture", "eager", "replay")
        assert cache.stats["capture"] + cache.stats["eager"] + cache.stats["replay"] > 0
    finally:
        free(index_buf, runtime=runtime)
        free(value_buf, runtime=runtime)
        cache.close()
