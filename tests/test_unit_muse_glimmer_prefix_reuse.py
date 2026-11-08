"""Muse Glimmer resident prefix bookkeeping, snapshot metadata, and mode plumbing.

CPU only: the session's token history and resume arithmetic are exercised on
an unallocated instance; the GPU behavior is covered by
tests/test_live_muse_glimmer_prefix_cache.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.engine_loop import _SubmitPollTextRunner
from hipengine.generation.muse_glimmer_gguf import MuseGlimmerGGUFGenerator
from hipengine.runtime.muse_glimmer import MuseGlimmerResidentSession
from hipengine.runtime.muse_glimmer_context import read_meta, write_meta


def _session(max_context: int = 64, chunk: int = 8) -> MuseGlimmerResidentSession:
    s = object.__new__(MuseGlimmerResidentSession)
    s.max_context, s.prefill_chunk, s._closed = max_context, chunk, False
    s._length, s._history_valid = 0, 0
    s._history = np.zeros(max_context, dtype=np.int64)
    s._image_overrides = None
    return s


def _write(s, ids, offset: int) -> None:
    ids = np.asarray(ids, dtype=np.int64)
    s._record_history(ids, offset)
    s._length = offset + ids.size


def test_reuse_is_the_common_prefix_rounded_to_a_chunk() -> None:
    s = _session()
    _write(s, range(100, 130), 0)
    prompt = list(range(100, 125)) + [7, 8, 9]
    assert s.reusable_prefix(prompt) == 24  # common 25 -> chunk boundary 24
    assert s.reusable_prefix(list(range(100, 105))) == 0  # one token always prefilled
    assert s.reusable_prefix([1, 2, 3] + list(range(103, 130))) == 0
    assert s.reusable_prefix(list(range(100, 140)), limit=20) == 16


def test_truncate_and_rewrite_bound_the_reusable_history() -> None:
    s = _session()
    _write(s, range(100, 130), 0)
    s.truncate(10)
    assert s.history.tolist() == list(range(100, 110))
    _write(s, [1, 2], 10)
    assert s.history.tolist() == list(range(100, 110)) + [1, 2]
    s.reset()
    assert s.history.size == 0


def test_image_rows_end_the_reusable_history() -> None:
    s = _session()
    s._image_overrides = {12: np.zeros(4, dtype=np.float32)}
    _write(s, range(100, 120), 0)
    s._image_overrides = None
    _write(s, [5, 6], 20)
    assert s.history.size == 12


def test_resumed_chunks_keep_the_fresh_schedule() -> None:
    s = _session(chunk=8)
    assert list(s._prefill_chunks(20, 0)) == [(0, 8), (8, 16), (16, 20)]
    assert list(s._prefill_chunks(20, 8)) == [(8, 16), (16, 20)]


def test_prefill_refuses_a_start_past_the_cached_prefix() -> None:
    s = _session()
    _write(s, range(100, 116), 0)
    ids = np.arange(100, 120, dtype=np.int64)
    s._begin_prefill(ids, 16)
    assert s.length == 16
    ids[3] = 0
    with pytest.raises(ValueError, match="cached prefix"):
        s._begin_prefill(ids, 16)


def test_snapshot_metadata_round_trip(tmp_path) -> None:
    tokens = np.arange(5, dtype=np.int64)
    write_meta(tmp_path, {"tokens": 5, "target": {"n_layers": 2, "kv_dim": 4}}, tokens)
    meta, loaded = read_meta(tmp_path)
    assert meta["target"] == {"n_layers": 2, "kv_dim": 4}
    assert loaded.tolist() == tokens.tolist()
    (tmp_path / "context.json").write_text('{"format": "other", "tokens": 5}')
    with pytest.raises(ValueError, match="not a"):
        read_meta(tmp_path)


def test_serving_prefix_mode_reaches_the_generator() -> None:
    gen = object.__new__(MuseGlimmerGGUFGenerator)
    gen._prefix_cache_mode = "radix"
    runner = _SubmitPollTextRunner(gen, capacity=1)
    runner.configure_engine_loop(SimpleNamespace(prefix_cache="off"))
    assert gen.prefix_cache_mode == "off"
    assert gen.prefix_reuse_start(_session(), [1, 2, 3]) == 0
