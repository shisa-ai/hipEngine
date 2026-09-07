"""Full-attention split-K partials must be demand-driven.

The split-K partial buffers (``full_attn_split_partial/m/l``) used to be
allocated at scratch construction with the native-prefill fallback's batch
row count (up to 16) times the full-capacity split count, regardless of
whether the native path ever runs. AOTriton-handled prefill chunks never
touch them, and decode needs a single query row, so eager 16-row sizing is
pure peak-memory waste (109 MiB at 72K, 387 MiB at 256K for the W7900-class
dense geometry).

Contract under test:
- allocation sizes the split buffers for one query row (the decode floor);
- ``ensure_full_attn_split_query_rows`` grows them on demand (native
  prefill fallback, multi-row batch decode), tracks the grown buffers for
  freeing, and is monotonic/no-op when capacity already suffices;
- chunk views grow through the root scratch so later views observe the
  grown buffers.
"""
from __future__ import annotations

from types import SimpleNamespace

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import _GGUFFullAttentionPrefillScratch

from tests.test_qwen35_gguf_prefill_scratch_liveness import (
    _fake_dense_qwen36_runner,
    _install_fake_device,
)


def _allocate(monkeypatch, *, rows: int = 768, capacity: int | None = None):
    _install_fake_device(monkeypatch)
    # Keep chunk views on the host-metadata path: the device metadata kernel
    # cannot run against fake buffers.
    monkeypatch.setattr(
        gguf_runner, "_gguf_prefill_device_metadata_enabled", lambda **kwargs: False
    )
    return _GGUFFullAttentionPrefillScratch.allocate(
        _fake_dense_qwen36_runner(),
        rows=rows,
        capacity=capacity if capacity is not None else rows,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )


def test_split_buffers_start_at_single_query_decode_floor(monkeypatch) -> None:
    scratch = _allocate(monkeypatch, rows=768, capacity=73_728)
    split_count = scratch.full_attn_split_count
    assert split_count == (73_728 + 255) // 256
    q_width = 6_144
    head_count = 24
    # One query row, not the 16-row native-prefill batch floor.
    assert scratch.full_attn_split_partial.nbytes == q_width * split_count * 4
    assert scratch.full_attn_split_m.nbytes == head_count * split_count * 4
    assert scratch.full_attn_split_l.nbytes == head_count * split_count * 4
    assert scratch.full_attn_split_capacity_rows == 1


def test_split_grow_on_demand_tracks_buffers(monkeypatch) -> None:
    scratch = _allocate(monkeypatch, rows=768, capacity=73_728)
    initial_partial = scratch.full_attn_split_partial
    scratch.ensure_full_attn_split_query_rows(16, runtime=SimpleNamespace())
    split_count = scratch.full_attn_split_count
    assert scratch.full_attn_split_capacity_rows == 16
    assert scratch.full_attn_split_partial is not initial_partial
    assert scratch.full_attn_split_partial.nbytes == 16 * 6_144 * split_count * 4
    assert scratch.full_attn_split_m.nbytes == 16 * 24 * split_count * 4
    grown = (scratch.full_attn_split_partial, scratch.full_attn_split_m, scratch.full_attn_split_l)
    assert all(buffer in scratch.full_attn_split_growth_buffers for buffer in grown)
    # Growth is monotonic: a smaller request is a no-op.
    partial = scratch.full_attn_split_partial
    scratch.ensure_full_attn_split_query_rows(1, runtime=SimpleNamespace())
    assert scratch.full_attn_split_partial is partial
    assert scratch.full_attn_split_capacity_rows == 16


def test_chunk_view_grows_through_root_scratch(monkeypatch) -> None:
    scratch = _allocate(monkeypatch, rows=768, capacity=73_728)
    view = scratch.for_chunk(0, 768, 768, runtime=SimpleNamespace())
    view.ensure_full_attn_split_query_rows(16, runtime=SimpleNamespace())
    # The root scratch (and any later view) must observe the growth.
    assert scratch.full_attn_split_capacity_rows == 16
    assert view.full_attn_split_partial is scratch.full_attn_split_partial
    later = scratch.for_chunk(0, 768, 768, runtime=SimpleNamespace())
    assert later.full_attn_split_partial is scratch.full_attn_split_partial
    # Views must not record their own growth buffers; the root owns them.
    assert later.full_attn_split_growth_buffers == scratch.full_attn_split_growth_buffers


def test_grown_split_buffers_are_fully_owned(monkeypatch) -> None:
    _install_fake_device(monkeypatch)
    fake_malloc = gguf_runner.malloc
    allocations: list[DeviceBuffer] = []

    def tracking_malloc(nbytes: int, *, runtime):
        buffer = fake_malloc(nbytes, runtime=runtime)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", tracking_malloc)
    monkeypatch.setattr(
        gguf_runner, "_gguf_prefill_device_metadata_enabled", lambda **kwargs: False
    )
    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_dense_qwen36_runner(),
        rows=768,
        capacity=73_728,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )
    base_count = len(allocations)
    scratch.ensure_full_attn_split_query_rows(16, runtime=SimpleNamespace())
    grown = [b for b in allocations[base_count:]]
    assert len(grown) == 3
    assert set(grown) <= set(scratch.full_attn_split_growth_buffers)
