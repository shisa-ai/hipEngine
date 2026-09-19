"""Layout contract for the portable NextN prefix checkpoint.

A prefix-cache hit cannot be primed from prompt hidden rows, because a reused
prefix produces none. The fix is a snapshot of the provider's own state at the
prefix boundary that outlives the request it came from, so this test pins the
parts that are easy to get wrong and expensive to debug on a live server: which
buffers are copied, the slot and row strides of the packed KV caches, how many
rows are copied (the prefix, not the whole provider window), and the cursor.

It deliberately does not test numerics. The real round trip -- snapshot, advance
the provider, restore, and check that the next proposal matches an unbroken run
-- needs real weights and belongs with the integration unit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
import hipengine.runtime.qwen35_gguf_nextn as nextn_module
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNExecutor,
    Qwen35GGUFNextNPrefixCheckpoint,
)

PHYSICAL_SLOTS = 2
MAX_POSITIONS = 8
ROW_NBYTES = 16
HIDDEN_NBYTES = 32


class _Runtime:
    """Records device copies instead of performing them."""

    def __init__(self) -> None:
        self.memcpy_calls: list[tuple[int, int, int, int]] = []

    def memcpy(self, dst, src, nbytes, kind) -> None:
        self.memcpy_calls.append((int(dst), int(src), int(nbytes), int(kind)))


def _cursor(value: int) -> np.ndarray:
    """Stand-in for the slot's pinned host cursor cell."""

    cell = np.zeros(1, dtype=np.int64)
    cell[0] = int(value)
    return cell


def _cache(ptr: int) -> DeviceBuffer:
    return DeviceBuffer(ptr, PHYSICAL_SLOTS * MAX_POSITIONS * ROW_NBYTES)


def _slot_scratch(*, position: int, context: int, ptr_base: int = 0x1000):
    """One provider slot view: two layers, the second without full attention."""

    return SimpleNamespace(
        layer_conv_states=(
            DeviceBuffer(ptr_base + 0x000, 64),
            DeviceBuffer(ptr_base + 0x100, 64),
        ),
        layer_recurrent_states=(
            DeviceBuffer(ptr_base + 0x200, 64),
            DeviceBuffer(ptr_base + 0x300, 64),
        ),
        full_key_caches=(
            _cache(ptr_base + 0x400),
            None,
        ),
        full_value_caches=(
            _cache(ptr_base + 0x800),
            None,
        ),
        position_host=_cursor(position),
        context_host=_cursor(context),
        position_buf=DeviceBuffer(ptr_base + 0xC00, 8),
        context_buf=DeviceBuffer(ptr_base + 0xC08, 8),
        max_positions=MAX_POSITIONS,
        hidden_size=HIDDEN_NBYTES // 2,
    )


def _executor(monkeypatch, scratches: dict[int, SimpleNamespace]):
    allocations: list[int] = []
    frees: list[int] = []
    host_copies: list[tuple[int, int]] = []

    def fake_malloc(nbytes: int, *, runtime=None) -> DeviceBuffer:
        del runtime
        ptr = 0x9000 + 0x100 * len(allocations)
        allocations.append(ptr)
        return DeviceBuffer(ptr, int(nbytes))

    def fake_free(buffer, *, runtime=None) -> None:
        del runtime
        frees.append(int(buffer.ptr))

    def fake_copy_host_to_device(buffer, host_ptr, nbytes, *, runtime) -> None:
        del host_ptr, runtime
        host_copies.append((int(buffer.ptr), int(nbytes)))

    monkeypatch.setattr(nextn_module, "malloc", fake_malloc)
    monkeypatch.setattr(nextn_module, "free", fake_free)
    monkeypatch.setattr(nextn_module, "copy_host_to_device", fake_copy_host_to_device)

    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.runtime = _Runtime()
    executor.max_requests = len(scratches)
    executor._request_slots = {rid: slot for slot, rid in enumerate(scratches)}
    executor._batch_sessions = None
    by_slot = {slot: scratch for slot, scratch in enumerate(scratches.values())}
    executor.scratch = SimpleNamespace(
        slot_count=PHYSICAL_SLOTS,
        for_slot=lambda slot, span_role: by_slot[int(slot)],
    )
    return executor, allocations, frees, host_copies


def test_snapshot_copies_the_prefix_rows_at_the_slot_stride(monkeypatch) -> None:
    # The host mirrors are (last consumed position, consumed count), so the
    # cursor for a prefix of 5 is position 4 with context 5.
    scratch = _slot_scratch(position=4, context=5)
    executor, allocations, _frees, _copies = _executor(monkeypatch, {7: scratch})
    boundary = DeviceBuffer(0x2000, HIDDEN_NBYTES)

    checkpoint = executor.snapshot_prefix_state(
        7, prefix_len=5, boundary_hidden=boundary
    )

    assert isinstance(checkpoint, Qwen35GGUFNextNPrefixCheckpoint)
    assert checkpoint.prefix_len == 5
    assert checkpoint.position == 4
    assert checkpoint.context_length == 5
    assert checkpoint.hidden_size == HIDDEN_NBYTES // 2
    # Layer 1 has no full-attention cache, and the blob keeps that shape so a
    # restore can zip it against the destination by layer index.
    assert len(checkpoint.key_buffers) == 2
    assert checkpoint.key_buffers[1] is None
    assert checkpoint.value_buffers[1] is None
    assert len(checkpoint.state_backups) == 4

    slot_nbytes = MAX_POSITIONS * ROW_NBYTES
    kv_copies = [
        call
        for call in executor.runtime.memcpy_calls
        if call[2] == 5 * ROW_NBYTES
    ]
    # One key cache and one value cache, each read from this slot's slice and
    # only as far as the prefix: the provider window is eight positions and the
    # checkpoint is five.
    assert kv_copies == [
        (
            int(checkpoint.key_buffers[0].ptr),
            int(scratch.full_key_caches[0].ptr) + 0 * slot_nbytes,
            5 * ROW_NBYTES,
            int(HipMemcpyKind.DEVICE_TO_DEVICE),
        ),
        (
            int(checkpoint.value_buffers[0].ptr),
            int(scratch.full_value_caches[0].ptr) + 0 * slot_nbytes,
            5 * ROW_NBYTES,
            int(HipMemcpyKind.DEVICE_TO_DEVICE),
        ),
    ]
    # The boundary hidden row is copied too: it is the input the first draft
    # step after a restore consumes, and no hidden row for the prefix survives
    # anywhere else.
    assert (int(checkpoint.boundary_hidden.ptr), int(boundary.ptr), HIDDEN_NBYTES, int(HipMemcpyKind.DEVICE_TO_DEVICE)) in (
        executor.runtime.memcpy_calls
    )
    # Every mutable state is backed up whole, in layer-major order.
    assert [
        call for call in executor.runtime.memcpy_calls if call[2] == 64
    ] == [
        (int(backup.ptr), int(state.ptr), 64, int(HipMemcpyKind.DEVICE_TO_DEVICE))
        for backup, state in zip(
            checkpoint.state_backups,
            (
                scratch.layer_conv_states[0],
                scratch.layer_recurrent_states[0],
                scratch.layer_conv_states[1],
                scratch.layer_recurrent_states[1],
            ),
            strict=True,
        )
    ]
    # Four state backups, one key cache, one value cache, one hidden row. The
    # count matters: a snapshot that allocates per layer instead of per cache
    # would pass a looser assertion and leak on every layer without attention.
    assert len(allocations) == 4 + 1 + 1 + 1


def test_snapshot_requires_the_cursor_at_the_boundary(monkeypatch) -> None:
    scratch = _slot_scratch(position=3, context=4)
    executor, _allocations, frees, _copies = _executor(monkeypatch, {7: scratch})

    with pytest.raises(ValueError, match="cursor"):
        executor.snapshot_prefix_state(7, prefix_len=5)
    # A refused snapshot allocates nothing and frees nothing.
    assert frees == []

    with pytest.raises(ValueError, match="positive prefix"):
        executor.snapshot_prefix_state(7, prefix_len=0)

    with pytest.raises(ValueError, match="exceeds the provider window"):
        boundary_scratch = _slot_scratch(
            position=MAX_POSITIONS, context=MAX_POSITIONS + 1
        )
        executor.scratch.for_slot = lambda slot, span_role: boundary_scratch
        executor.snapshot_prefix_state(7, prefix_len=MAX_POSITIONS + 1)

    with pytest.raises(ValueError, match="active request"):
        executor.snapshot_prefix_state(99, prefix_len=3)


def test_restore_writes_into_a_different_request_slot(monkeypatch) -> None:
    source = _slot_scratch(position=4, context=5, ptr_base=0x1000)
    destination = _slot_scratch(position=0, context=0, ptr_base=0x5000)
    executor, _allocations, _frees, host_copies = _executor(
        monkeypatch, {7: source, 8: destination}
    )

    checkpoint = executor.snapshot_prefix_state(
        7, prefix_len=5, boundary_hidden=DeviceBuffer(0x2000, HIDDEN_NBYTES)
    )
    executor.runtime.memcpy_calls.clear()

    executor.restore_prefix_state(checkpoint, 8)

    slot_nbytes = MAX_POSITIONS * ROW_NBYTES
    destination_slot = 1
    assert [
        call for call in executor.runtime.memcpy_calls if call[2] == 5 * ROW_NBYTES
    ] == [
        (
            int(destination.full_key_caches[0].ptr) + destination_slot * slot_nbytes,
            int(checkpoint.key_buffers[0].ptr),
            5 * ROW_NBYTES,
            int(HipMemcpyKind.DEVICE_TO_DEVICE),
        ),
        (
            int(destination.full_value_caches[0].ptr) + destination_slot * slot_nbytes,
            int(checkpoint.value_buffers[0].ptr),
            5 * ROW_NBYTES,
            int(HipMemcpyKind.DEVICE_TO_DEVICE),
        ),
    ]
    # The destination's own live states receive the backups: the blob is
    # content, so it lands in whatever slot the new request owns.
    assert [
        call for call in executor.runtime.memcpy_calls if call[2] == 64
    ] == [
        (int(state.ptr), int(backup.ptr), 64, int(HipMemcpyKind.DEVICE_TO_DEVICE))
        for state, backup in zip(
            (
                destination.layer_conv_states[0],
                destination.layer_recurrent_states[0],
                destination.layer_conv_states[1],
                destination.layer_recurrent_states[1],
            ),
            checkpoint.state_backups,
            strict=True,
        )
    ]
    # The cursor moves to the boundary, so the next step consumes the token at
    # position 5 with the checkpointed hidden row.
    assert int(destination.position_host[0]) == 4
    assert int(destination.context_host[0]) == 5
    assert host_copies == [
        (int(destination.position_buf.ptr), 8),
        (int(destination.context_buf.ptr), 8),
    ]
    # The source slot is untouched by a restore.
    assert int(source.position_host[0]) == 4


def test_restore_refuses_a_mismatched_destination(monkeypatch) -> None:
    source = _slot_scratch(position=4, context=5)
    executor, _allocations, _frees, _copies = _executor(
        monkeypatch, {7: source, 8: _slot_scratch(position=0, context=0)}
    )
    checkpoint = executor.snapshot_prefix_state(7, prefix_len=5)

    released = executor.snapshot_prefix_state(7, prefix_len=5)
    executor.release_prefix_state(released)
    with pytest.raises(RuntimeError, match="released"):
        executor.restore_prefix_state(released, 8)

    thin = _slot_scratch(position=0, context=0)
    thin.layer_conv_states = thin.layer_conv_states[:1]
    thin.layer_recurrent_states = thin.layer_recurrent_states[:1]
    executor.scratch.for_slot = lambda slot, span_role: thin if int(slot) == 1 else source
    with pytest.raises(ValueError, match="state layout mismatch"):
        executor.restore_prefix_state(checkpoint, 8)

    wide = _slot_scratch(position=0, context=0)
    wide.max_positions = 4
    executor.scratch.for_slot = lambda slot, span_role: wide if int(slot) == 1 else source
    with pytest.raises(ValueError, match="exceeds the destination window"):
        executor.restore_prefix_state(checkpoint, 8)


def test_release_frees_every_buffer_once(monkeypatch) -> None:
    scratch = _slot_scratch(position=4, context=5)
    executor, allocations, frees, _copies = _executor(monkeypatch, {7: scratch})
    checkpoint = executor.snapshot_prefix_state(
        7, prefix_len=5, boundary_hidden=DeviceBuffer(0x2000, HIDDEN_NBYTES)
    )

    executor.release_prefix_state(checkpoint)
    assert sorted(frees) == sorted(allocations)
    assert checkpoint.released is True

    executor.release_prefix_state(checkpoint)
    assert sorted(frees) == sorted(allocations)


def test_prefix_checkpoint_is_a_distinct_type_from_the_request_checkpoint() -> None:
    """The two are not interchangeable, and the type says so.

    The request-local checkpoint is bound to a provider slot and carries no KV,
    which is exactly why it cannot serve a prefix-cache hit. Keeping them
    distinct types is what stops the slot-bound one being stored in a prefix
    table.
    """

    from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNRequestCheckpoint

    assert Qwen35GGUFNextNPrefixCheckpoint is not Qwen35GGUFNextNRequestCheckpoint
    blob_fields = set(Qwen35GGUFNextNPrefixCheckpoint.__dataclass_fields__)
    assert {"key_buffers", "value_buffers", "boundary_hidden", "prefix_len"} <= blob_fields
    assert "request_id" not in blob_fields
    assert "slot" not in blob_fields
