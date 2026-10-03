"""CPU contracts for bounded, position-only Gemma host RoPE reuse."""

from dataclasses import replace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.cpu_reference.gemma4 import Gemma4RopeConfig
from hipengine.runtime import gemma4 as module
from tests.test_unit_gemma4_runner_lifecycle import _config, _make_runner


def _staging_runner(monkeypatch):
    config = _config(num_layers=3)
    global_rope = Gemma4RopeConfig(rope_theta=1_000_000.0, head_dim=16, rope_angles=2)
    config = replace(config, attention=(
        config.attention[0],
        replace(config.attention[1], rope=global_rope, head_dim=16, sliding_window=None),
        config.attention[2],
    ))
    runner, allocator = _make_runner(monkeypatch, config=config)
    original = module.gemma4_rope_cos_sin_tables
    calls = []
    uploads = []
    ids = []

    def compute(rope, positions):
        calls.append((rope, positions.copy()))
        return original(rope, positions)

    def upload(name, values, *, stream):
        uploads.append((name, values, stream))
        return DeviceBuffer(ptr=len(uploads) * 4096, nbytes=values.nbytes)

    monkeypatch.setattr(module, "gemma4_rope_cos_sin_tables", compute)
    monkeypatch.setattr(runner, "_stage_upload", upload)
    monkeypatch.setattr(module, "enqueue_host_to_device", lambda *a, **kw: ids.append(kw["stream"]))
    return runner, calls, uploads, ids, original, allocator


def _assert_tables(runner, uploads, original, start, rows, stream):
    """Compare every staged table byte-for-byte to the independent uncached path."""
    by_name = {name: (values, used_stream) for name, values, used_stream in uploads}
    for rope in {geometry.rope for geometry in runner.weights.config.attention}:
        expected = original(rope, np.arange(start, start + rows, dtype=np.int64))
        for kind, value in zip(("cos", "sin"), expected):
            actual, used_stream = by_name[f"{kind}{rope}"]
            assert actual.dtype == np.float32 and actual.flags.c_contiguous
            assert actual.tobytes() == value.tobytes()
            assert used_stream == stream


@pytest.mark.parametrize("rows", [2, 3, 7, 8])
def test_same_range_reuses_host_tables_but_uploads_each_block(monkeypatch, rows):
    runner, calls, uploads, ids, original, _ = _staging_runner(monkeypatch)
    try:
        runner._position = 5
        runner._stage_block_content([1] * rows, stream=17)
        first = {name: value for name, value, _ in uploads}
        _assert_tables(runner, uploads, original, 5, rows, 17)
        uploads.clear()
        # Reuse is independent of tokens, layer multiplicity, or submission stream.
        runner._stage_block_content([3] * rows, stream=29)
        assert len(calls) == 2, "the same position range was recomputed"
        assert ids == [17, 29], "new token ids must still upload"
        _assert_tables(runner, uploads, original, 5, rows, 29)
        for name, value, _ in uploads:
            if name.startswith(("cos", "sin")):
                assert value is first[name]
    finally:
        runner.close()


def test_decode_does_not_evict_multirow_entry_and_reset_can_reuse_it(monkeypatch):
    runner, calls, uploads, _, original, _ = _staging_runner(monkeypatch)
    try:
        runner._stage_block_content([1, 2, 3], stream=0)
        for position in range(3, 7):
            runner._position = position
            uploads.clear()
            runner._stage_block_content([4], stream=19)
            _assert_tables(runner, uploads, original, position, 1, 19)
        assert len(calls) == 10  # Two configurations, prefill plus four decode blocks.
        runner.reset()
        runner._stage_block_content([5, 6, 7], stream=0)
        assert len(calls) == 10, "decode or reset evicted position-only prefill tables"
        assert len(runner._rope_prefill_tables) == 2
    finally:
        runner.close()


def test_range_replacement_is_bounded_and_changed_positions_or_rows_miss(monkeypatch):
    runner, calls, uploads, _, original, _ = _staging_runner(monkeypatch)
    try:
        for index, (start, rows) in enumerate([(0, 3), (1, 3), (1, 2), (0, 3)]):
            runner._position = start
            uploads.clear()
            runner._stage_block_content([1] * rows, stream=11)
            _assert_tables(runner, uploads, original, start, rows, 11)
            assert len(calls) == 2 * (index + 1)
            assert len(runner._rope_prefill_tables) == 2
        before = len(calls)
        runner._stage_block_content([2] * 3, stream=11)
        assert len(calls) == before
    finally:
        runner.close()


def test_cache_is_runner_local_and_close_releases_host_entries(monkeypatch):
    runner, calls, _, _, _, allocator = _staging_runner(monkeypatch)
    other, _ = _make_runner(monkeypatch, allocator=allocator)
    try:
        runner._stage_block_content([1, 2], stream=0)
        assert len(calls) == 2
        assert other._rope_prefill_tables == {}
        assert other._rope_prefill_tables is not runner._rope_prefill_tables
        runner.close()
        runner.close()
        assert runner._rope_prefill_tables == {}
        other.close()
        # Cleanup is separate from the allocation-growth contract below.
        assert len(allocator.freed) == len(allocator.allocated)
    finally:
        runner.close()
        other.close()


def test_real_staging_reuses_device_allocations_on_hits_and_decode(monkeypatch):
    """Host hits still upload, but reuse the already sized device staging."""
    runner, allocator = _make_runner(monkeypatch, max_block=8)
    uploads = []
    monkeypatch.setattr(module, "enqueue_host_to_device", lambda *a, **kw: uploads.append((a, kw)))
    try:
        before = allocator.calls
        runner._stage_block_content([1] * 8, stream=7)
        # Only the existing cos, sin and mask staging buffers are allocated.
        assert allocator.calls == before + 3
        primed = allocator.calls
        runner._stage_block_content([2] * 8, stream=11)
        assert allocator.calls == primed
        for position in (8, 9):
            runner._position = position
            runner._stage_block_content([3], stream=11)
            assert allocator.calls == primed
        runner.reset()
        runner._stage_block_content([4] * 8, stream=17)
        assert allocator.calls == primed
        # IDs, cos, sin and mask are copied on every block, including cache hits.
        assert len(uploads) == 5 * 4
        assert [kw["stream"] for _, kw in uploads] == [7] * 4 + [11] * 12 + [17] * 4
    finally:
        runner.close()
    assert len(allocator.freed) == len(allocator.allocated)
