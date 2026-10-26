"""Ownership and layout contract for the portable NextN prefix checkpoint.

A prefix-cache hit cannot be primed from prompt hidden rows, because a reused
prefix produces none. The fix is a snapshot of the provider's own state at the
prefix boundary that outlives the request it came from, so this test pins the
parts that are easy to get wrong and expensive to debug on a live server: which
buffers are copied, which of them may hold device memory, the slot and row
strides of the packed KV caches, how many rows are copied (the prefix, not the
whole provider window), and the cursor.

The prefix KV is host memory. That is not an implementation detail: a device
snapshot of the prefix KV is the largest thing a checkpoint holds, and on a
long-context server it is exactly the device memory the request cannot spare.
The tests below therefore assert that a snapshot takes **no** device allocation
for KV, that the host bytes are owned byte arrays, and that a restore puts the
same bytes back into a different request's slot.

It deliberately does not test numerics. The real round trip -- snapshot, advance
the provider, restore, and check that the next proposal matches an unbroken run
-- needs real weights and belongs with the integration unit.
"""

from __future__ import annotations

import ctypes
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
STATE_NBYTES = 64
DEVICE_BYTES = 1 << 20
# Bump allocations live above the scratch regions the tests hand the executor.
BUMP_BASE = 1 << 19


class _FakeDevice:
    """A byte-accurate fake device: a pointer is an offset into one bytearray.

    Copies really move bytes, so a device-to-host capture followed by a
    host-to-device restore can be compared byte for byte without a GPU.
    """

    def __init__(self) -> None:
        self.memory = bytearray(DEVICE_BYTES)
        self._bump = BUMP_BASE
        self.allocation_sizes: dict[int, int] = {}
        self.frees: list[int] = []
        self.calls: list[tuple[int, int, int, int]] = []

    @property
    def allocations(self) -> tuple[int, ...]:
        return tuple(self.allocation_sizes)

    def malloc(self, nbytes: int) -> int:
        ptr = self._bump
        self._bump += (int(nbytes) + 63) & ~63
        if self._bump > DEVICE_BYTES:
            raise MemoryError("fake device is full")
        self.allocation_sizes[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.frees.append(int(ptr))

    def write(self, ptr: int, payload: bytes) -> None:
        self.memory[int(ptr) : int(ptr) + len(payload)] = payload

    def fill(self, ptr: int, nbytes: int, value: int) -> None:
        self.memory[int(ptr) : int(ptr) + int(nbytes)] = bytes([value]) * int(nbytes)

    def read(self, ptr: int, nbytes: int) -> bytes:
        return bytes(self.memory[int(ptr) : int(ptr) + int(nbytes)])

    def memcpy(self, dst, src, nbytes, kind) -> None:
        count = int(nbytes)
        kind = int(kind)
        self.calls.append((int(dst), int(src), count, kind))
        if kind == int(HipMemcpyKind.DEVICE_TO_DEVICE):
            self.memory[int(dst) : int(dst) + count] = self.memory[
                int(src) : int(src) + count
            ]
        elif kind == int(HipMemcpyKind.DEVICE_TO_HOST):
            ctypes.memmove(int(dst), bytes(self.memory[int(src) : int(src) + count]), count)
        elif kind == int(HipMemcpyKind.HOST_TO_DEVICE):
            self.memory[int(dst) : int(dst) + count] = ctypes.string_at(int(src), count)
        else:
            raise AssertionError(f"unexpected memcpy kind {kind}")

    def device_synchronize(self) -> None:
        return None


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
            DeviceBuffer(ptr_base + 0x000, STATE_NBYTES),
            DeviceBuffer(ptr_base + 0x100, STATE_NBYTES),
        ),
        layer_recurrent_states=(
            DeviceBuffer(ptr_base + 0x200, STATE_NBYTES),
            DeviceBuffer(ptr_base + 0x300, STATE_NBYTES),
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


def _executor(monkeypatch, device: _FakeDevice, scratches: dict[int, SimpleNamespace]):
    """Bind an executor to ``device`` without touching global memory stats."""

    def fake_malloc(nbytes: int, *, runtime=None) -> DeviceBuffer:
        del runtime
        return DeviceBuffer(device.malloc(int(nbytes)), int(nbytes))

    def fake_free(buffer, *, runtime=None) -> None:
        del runtime
        device.free(buffer.ptr)

    def fake_copy_host_array_to_device(buffer, array, nbytes=None, *, runtime) -> None:
        del runtime
        count = array.nbytes if nbytes is None else int(nbytes)
        device.memcpy(buffer.ptr, int(array.ctypes.data), count,
                      HipMemcpyKind.HOST_TO_DEVICE)

    def fake_copy_device_to_host(host_ptr, buffer, nbytes=None, *, runtime) -> None:
        del runtime
        device.memcpy(host_ptr, buffer.ptr, buffer.nbytes if nbytes is None else nbytes,
                      HipMemcpyKind.DEVICE_TO_HOST)

    monkeypatch.setattr(nextn_module, "malloc", fake_malloc)
    monkeypatch.setattr(nextn_module, "free", fake_free)
    monkeypatch.setattr(
        nextn_module, "copy_host_array_to_device", fake_copy_host_array_to_device
    )
    monkeypatch.setattr(nextn_module, "copy_device_to_host", fake_copy_device_to_host)

    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.runtime = device
    executor.max_requests = len(scratches)
    executor._request_slots = {rid: slot for slot, rid in enumerate(scratches)}
    executor._batch_sessions = None
    by_slot = {slot: scratch for slot, scratch in enumerate(scratches.values())}
    executor.scratch = SimpleNamespace(
        slot_count=PHYSICAL_SLOTS,
        for_slot=lambda slot, span_role: by_slot[int(slot)],
    )
    return executor


def test_snapshot_keeps_the_prefix_rows_in_host_memory_at_the_slot_stride(
    monkeypatch,
) -> None:
    # The host mirrors are (last consumed position, consumed count), so the
    # cursor for a prefix of 5 is position 4 with context 5.
    scratch = _slot_scratch(position=4, context=5)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: scratch})
    boundary = DeviceBuffer(0x2000, HIDDEN_NBYTES)
    device.fill(boundary.ptr, HIDDEN_NBYTES, 0x5A)
    device.fill(scratch.full_key_caches[0].ptr, scratch.full_key_caches[0].nbytes, 0x11)
    device.fill(scratch.full_value_caches[0].ptr, scratch.full_value_caches[0].nbytes, 0x22)

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

    # The prefix KV is owned host memory, not a device snapshot.
    key_host = checkpoint.key_buffers[0]
    value_host = checkpoint.value_buffers[0]
    assert isinstance(key_host, np.ndarray)
    assert isinstance(value_host, np.ndarray)
    assert key_host.dtype == np.uint8
    assert key_host.flags.owndata and key_host.flags.c_contiguous
    assert key_host.nbytes == 5 * ROW_NBYTES
    assert value_host.nbytes == 5 * ROW_NBYTES
    assert key_host.tobytes() == bytes([0x11]) * (5 * ROW_NBYTES)
    assert value_host.tobytes() == bytes([0x22]) * (5 * ROW_NBYTES)

    slot_nbytes = MAX_POSITIONS * ROW_NBYTES
    # One key cache and one value cache, each read from this slot's slice and
    # only as far as the prefix: the provider window is eight positions and the
    # checkpoint is five.
    assert [call for call in device.calls if call[2] == 5 * ROW_NBYTES] == [
        (
            int(key_host.ctypes.data),
            int(scratch.full_key_caches[0].ptr) + 0 * slot_nbytes,
            5 * ROW_NBYTES,
            int(HipMemcpyKind.DEVICE_TO_HOST),
        ),
        (
            int(value_host.ctypes.data),
            int(scratch.full_value_caches[0].ptr) + 0 * slot_nbytes,
            5 * ROW_NBYTES,
            int(HipMemcpyKind.DEVICE_TO_HOST),
        ),
    ]
    # The boundary hidden row is copied too: it is the input the first draft
    # step after a restore consumes, and no hidden row for the prefix survives
    # anywhere else. It stays device-resident because the adapter hands its
    # pointer to a kernel launch.
    assert (
        int(checkpoint.boundary_hidden.ptr),
        int(boundary.ptr),
        HIDDEN_NBYTES,
        int(HipMemcpyKind.DEVICE_TO_DEVICE),
    ) in device.calls
    # Every mutable state is backed up whole, in layer-major order.
    assert [call for call in device.calls if call[2] == STATE_NBYTES] == [
        (int(backup.ptr), int(state.ptr), STATE_NBYTES, int(HipMemcpyKind.DEVICE_TO_DEVICE))
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
    # Four state backups and one hidden row, and nothing else: the count is what
    # proves no device KV snapshot was taken. A per-cache device copy would show
    # up here as a 5 * ROW_NBYTES allocation.
    assert sorted(device.allocation_sizes.values()) == [
        HIDDEN_NBYTES,
        STATE_NBYTES,
        STATE_NBYTES,
        STATE_NBYTES,
        STATE_NBYTES,
    ]
    # Byte accounting separates host KV from device footprint, because serving
    # telemetry reports the latter as cache_provider_device_bytes.
    assert checkpoint.nbytes == 4 * STATE_NBYTES + HIDDEN_NBYTES
    assert checkpoint.host_nbytes == 2 * 5 * ROW_NBYTES


def test_snapshot_requires_the_cursor_at_the_boundary(monkeypatch) -> None:
    scratch = _slot_scratch(position=3, context=4)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: scratch})

    with pytest.raises(ValueError, match="cursor"):
        executor.snapshot_prefix_state(7, prefix_len=5)
    # A refused snapshot allocates nothing and frees nothing.
    assert device.frees == []
    assert device.allocations == ()

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

    assert device.allocations == ()


def test_restore_writes_into_a_different_request_slot(monkeypatch) -> None:
    source = _slot_scratch(position=4, context=5, ptr_base=0x1000)
    destination = _slot_scratch(position=0, context=0, ptr_base=0x5000)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: source, 8: destination})

    checkpoint = executor.snapshot_prefix_state(
        7, prefix_len=5, boundary_hidden=DeviceBuffer(0x2000, HIDDEN_NBYTES)
    )
    device.calls.clear()

    executor.restore_prefix_state(checkpoint, 8)

    slot_nbytes = MAX_POSITIONS * ROW_NBYTES
    destination_slot = 1
    # The host bytes go back up into this request's slot, one upload per cache.
    assert [call for call in device.calls if call[2] == 5 * ROW_NBYTES] == [
        (
            int(destination.full_key_caches[0].ptr) + destination_slot * slot_nbytes,
            int(checkpoint.key_buffers[0].ctypes.data),
            5 * ROW_NBYTES,
            int(HipMemcpyKind.HOST_TO_DEVICE),
        ),
        (
            int(destination.full_value_caches[0].ptr) + destination_slot * slot_nbytes,
            int(checkpoint.value_buffers[0].ctypes.data),
            5 * ROW_NBYTES,
            int(HipMemcpyKind.HOST_TO_DEVICE),
        ),
    ]
    # The destination's own live states receive the backups: the blob is
    # content, so it lands in whatever slot the new request owns.
    assert [call for call in device.calls if call[2] == STATE_NBYTES] == [
        (int(state.ptr), int(backup.ptr), STATE_NBYTES, int(HipMemcpyKind.DEVICE_TO_DEVICE))
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
    assert [
        (call[0], call[2], call[3]) for call in device.calls if call[3] == int(HipMemcpyKind.HOST_TO_DEVICE)
    ][-2:] == [
        (int(destination.position_buf.ptr), 8, int(HipMemcpyKind.HOST_TO_DEVICE)),
        (int(destination.context_buf.ptr), 8, int(HipMemcpyKind.HOST_TO_DEVICE)),
    ]
    # The source slot is untouched by a restore.
    assert int(source.position_host[0]) == 4


def test_clone_request_state_uploads_position_and_context_through_the_owning_helper(
    monkeypatch,
) -> None:
    source = _slot_scratch(position=4, context=5, ptr_base=0x1000)
    destination = _slot_scratch(position=0, context=0, ptr_base=0x5000)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: source, 8: destination})
    executor._provider_root_state_metadata = {}

    uploads = []
    fake_upload = nextn_module.copy_host_array_to_device

    def recording_upload(buffer, array, nbytes=None, *, runtime):
        uploads.append((buffer, array, nbytes, runtime))
        return fake_upload(buffer, array, nbytes, runtime=runtime)

    monkeypatch.setattr(
        nextn_module, "copy_host_array_to_device", recording_upload
    )
    device.calls.clear()

    executor.clone_request_state(7, 8)

    # The clone publishes the source cursor into the destination slot while
    # leaving the source untouched.
    assert int(destination.position_host[0]) == 4
    assert int(destination.context_host[0]) == 5
    assert int(source.position_host[0]) == 4
    assert int(source.context_host[0]) == 5

    # Only the two cursor cells cross the owning upload boundary. Each upload
    # carries the destination buffer, the live host owner, and its exact size.
    assert len(uploads) == 2
    assert [buffer for buffer, _array, _nbytes, _runtime in uploads] == [
        destination.position_buf,
        destination.context_buf,
    ]
    assert [array for _buffer, array, _nbytes, _runtime in uploads] == [
        destination.position_host,
        destination.context_host,
    ]
    assert [nbytes for _buffer, _array, nbytes, _runtime in uploads] == [8, 8]
    assert all(runtime is device for _buffer, _array, _nbytes, runtime in uploads)

    assert [
        call
        for call in device.calls
        if call[3] == int(HipMemcpyKind.HOST_TO_DEVICE)
    ] == [
        (
            int(destination.position_buf.ptr),
            int(destination.position_host.ctypes.data),
            8,
            int(HipMemcpyKind.HOST_TO_DEVICE),
        ),
        (
            int(destination.context_buf.ptr),
            int(destination.context_host.ctypes.data),
            8,
            int(HipMemcpyKind.HOST_TO_DEVICE),
        ),
    ]
    assert device.read(destination.position_buf.ptr, 8) == np.asarray(
        [4], dtype=np.int64
    ).tobytes()
    assert device.read(destination.context_buf.ptr, 8) == np.asarray(
        [5], dtype=np.int64
    ).tobytes()


def test_restore_reproduces_the_prefix_kv_bytes_in_another_slot(monkeypatch) -> None:
    """The host round trip is byte-exact, and it moves the prefix only."""

    source = _slot_scratch(position=4, context=5, ptr_base=0x1000)
    destination = _slot_scratch(position=0, context=0, ptr_base=0x5000)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: source, 8: destination})

    slot_nbytes = MAX_POSITIONS * ROW_NBYTES
    prefix_nbytes = 5 * ROW_NBYTES
    key_pattern = bytes((0x40 + index) & 0xFF for index in range(prefix_nbytes))
    value_pattern = bytes((0xC0 - index) & 0xFF for index in range(prefix_nbytes))
    # Slot 0 holds the live prefix; slot 1 of the same cache holds something
    # else, and rows past the prefix hold something else again.
    device.fill(source.full_key_caches[0].ptr, source.full_key_caches[0].nbytes, 0xAA)
    device.fill(source.full_value_caches[0].ptr, source.full_value_caches[0].nbytes, 0xBB)
    device.write(source.full_key_caches[0].ptr, key_pattern)
    device.write(source.full_value_caches[0].ptr, value_pattern)

    checkpoint = executor.snapshot_prefix_state(7, prefix_len=5)

    assert checkpoint.key_buffers[0].tobytes() == key_pattern
    assert checkpoint.value_buffers[0].tobytes() == value_pattern

    # A different request's slot starts dirty; the restore must overwrite the
    # prefix rows and leave everything else alone.
    device.fill(destination.full_key_caches[0].ptr, destination.full_key_caches[0].nbytes, 0x11)
    device.fill(destination.full_value_caches[0].ptr, destination.full_value_caches[0].nbytes, 0x22)
    executor.restore_prefix_state(checkpoint, 8)

    destination_prefix = destination.full_key_caches[0].ptr + 1 * slot_nbytes
    assert device.read(destination_prefix, prefix_nbytes) == key_pattern
    assert device.read(destination_prefix + prefix_nbytes, 3 * ROW_NBYTES) == (
        bytes([0x11]) * (3 * ROW_NBYTES)
    )
    # The destination's slot 0 is another request's KV and is never written.
    assert device.read(destination.full_key_caches[0].ptr, prefix_nbytes) == (
        bytes([0x11]) * prefix_nbytes
    )
    value_prefix = destination.full_value_caches[0].ptr + 1 * slot_nbytes
    assert device.read(value_prefix, prefix_nbytes) == value_pattern
    assert device.read(value_prefix + prefix_nbytes, 3 * ROW_NBYTES) == (
        bytes([0x22]) * (3 * ROW_NBYTES)
    )
    # The checkpoint still owns its bytes: a restore is a copy, not a move.
    assert checkpoint.key_buffers[0].tobytes() == key_pattern


def test_restore_refuses_a_mismatched_destination(monkeypatch) -> None:
    source = _slot_scratch(position=4, context=5)
    device = _FakeDevice()
    executor = _executor(
        monkeypatch, device, {7: source, 8: _slot_scratch(position=0, context=0)}
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


def test_restore_refuses_a_kv_mismatch_before_writing(monkeypatch) -> None:
    """A layout or size mismatch must not half-restore the destination slot."""

    source = _slot_scratch(position=4, context=5)
    destination = _slot_scratch(position=0, context=0)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: source, 8: destination})
    checkpoint = executor.snapshot_prefix_state(7, prefix_len=5)

    device.calls.clear()
    missing_cache = _slot_scratch(position=0, context=0)
    missing_cache.full_key_caches = (None, None)
    executor.scratch.for_slot = (
        lambda slot, span_role: missing_cache if int(slot) == 1 else source
    )
    with pytest.raises(ValueError, match="KV layout mismatch"):
        executor.restore_prefix_state(checkpoint, 8)
    assert device.calls == []

    executor.scratch.for_slot = (
        lambda slot, span_role: destination if int(slot) == 1 else source
    )
    checkpoint.key_buffers = (checkpoint.key_buffers[0][: ROW_NBYTES], None)
    with pytest.raises(ValueError, match="row size mismatch"):
        executor.restore_prefix_state(checkpoint, 8)
    assert device.calls == []


def test_restore_refuses_a_device_buffer_in_place_of_host_bytes(monkeypatch) -> None:
    """A legacy device-buffer payload is refused instead of read as host memory."""

    source = _slot_scratch(position=4, context=5)
    device = _FakeDevice()
    executor = _executor(
        monkeypatch, device, {7: source, 8: _slot_scratch(position=0, context=0)}
    )
    checkpoint = executor.snapshot_prefix_state(7, prefix_len=5)
    checkpoint.key_buffers = (DeviceBuffer(0x3000, 5 * ROW_NBYTES), None)

    device.calls.clear()
    with pytest.raises(ValueError, match="host byte array"):
        executor.restore_prefix_state(checkpoint, 8)
    assert device.calls == []


def test_release_frees_every_device_buffer_once_and_drops_the_host_kv(
    monkeypatch,
) -> None:
    scratch = _slot_scratch(position=4, context=5)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: scratch})
    checkpoint = executor.snapshot_prefix_state(
        7, prefix_len=5, boundary_hidden=DeviceBuffer(0x2000, HIDDEN_NBYTES)
    )
    assert checkpoint.host_nbytes == 2 * 5 * ROW_NBYTES

    executor.release_prefix_state(checkpoint)
    assert sorted(device.frees) == sorted(device.allocations)
    assert checkpoint.released is True
    # Host KV needs no device release, so it is dropped from the checkpoint and
    # stops counting toward either accounting total.
    assert checkpoint.key_buffers == ()
    assert checkpoint.value_buffers == ()
    assert checkpoint.nbytes == 0
    assert checkpoint.host_nbytes == 0

    executor.release_prefix_state(checkpoint)
    assert sorted(device.frees) == sorted(device.allocations)


def test_snapshot_failure_frees_every_device_buffer_it_allocated(monkeypatch) -> None:
    scratch = _slot_scratch(position=4, context=5)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: scratch})
    real_malloc = device.malloc
    calls = {"count": 0}

    def failing_malloc(nbytes: int) -> int:
        calls["count"] += 1
        if calls["count"] == 3:
            raise MemoryError("device allocation refused")
        return real_malloc(nbytes)

    monkeypatch.setattr(device, "malloc", failing_malloc)

    with pytest.raises(MemoryError, match="refused"):
        executor.snapshot_prefix_state(
            7, prefix_len=5, boundary_hidden=DeviceBuffer(0x2000, HIDDEN_NBYTES)
        )

    # The boundary row and the first state backup had landed; both are returned.
    assert sorted(device.allocation_sizes.values()) == [HIDDEN_NBYTES, STATE_NBYTES]
    assert sorted(device.frees) == sorted(device.allocations)


def test_snapshot_failure_during_a_host_copy_frees_every_device_buffer(
    monkeypatch,
) -> None:
    scratch = _slot_scratch(position=4, context=5)
    device = _FakeDevice()
    executor = _executor(monkeypatch, device, {7: scratch})

    def failing_copy_device_to_host(host_ptr, buffer, nbytes=None, *, runtime) -> None:
        del host_ptr, buffer, nbytes, runtime
        raise RuntimeError("device-to-host copy failed")

    monkeypatch.setattr(nextn_module, "copy_device_to_host", failing_copy_device_to_host)

    with pytest.raises(RuntimeError, match="copy failed"):
        executor.snapshot_prefix_state(
            7, prefix_len=5, boundary_hidden=DeviceBuffer(0x2000, HIDDEN_NBYTES)
        )

    # Every device allocation the snapshot made is released; the host bytes it
    # had filled are garbage and simply go away with the call.
    assert sorted(device.allocation_sizes.values()) == [
        HIDDEN_NBYTES,
        STATE_NBYTES,
        STATE_NBYTES,
        STATE_NBYTES,
        STATE_NBYTES,
    ]
    assert sorted(device.frees) == sorted(device.allocations)


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
