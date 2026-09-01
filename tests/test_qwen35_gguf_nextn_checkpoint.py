from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNExecutor


class _Runtime:
    def __init__(self) -> None:
        self.copies = []

    def memcpy(self, dst, src, nbytes, kind):
        self.copies.append((int(dst), int(src), int(nbytes), kind))

    def device_synchronize(self):
        return None


class _Buffer:
    def __init__(self, ptr, nbytes):
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)


def _executor(monkeypatch):
    allocated = []
    freed = []

    def malloc(nbytes, *, runtime):
        buffer = _Buffer(0x9000 + len(allocated) * 0x1000, nbytes)
        allocated.append(buffer)
        return buffer

    def free(buffer, *, runtime):
        freed.append(buffer)

    monkeypatch.setattr("hipengine.runtime.qwen35_gguf_nextn.malloc", malloc)
    monkeypatch.setattr("hipengine.runtime.qwen35_gguf_nextn.free", free)
    slot = SimpleNamespace(
        layer_conv_states=(_Buffer(0x1000, 16), None),
        layer_recurrent_states=(_Buffer(0x2000, 32), None),
        position_host=__import__("numpy").asarray([17], dtype="int64"),
        context_host=__import__("numpy").asarray([18], dtype="int64"),
        position_buf=_Buffer(0x3000, 8),
        context_buf=_Buffer(0x4000, 8),
    )
    scratch = SimpleNamespace(for_slot=lambda index, span_role="decode": slot)
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor._request_slots = {7: 0}
    executor._provider_root_state_metadata = {}
    executor.scratch = scratch
    executor.runtime = _Runtime()
    executor.closed = False
    return executor, slot, allocated, freed


def test_nextn_checkpoint_restores_linear_state_and_cursors(monkeypatch) -> None:
    executor, slot, allocated, freed = _executor(monkeypatch)

    checkpoint = executor.capture_request_checkpoint(7)

    assert len(allocated) == 2
    assert checkpoint.position == 17
    assert checkpoint.context_length == 18
    assert [copy[:3] for copy in executor.runtime.copies[:2]] == [
        (allocated[0].ptr, 0x1000, 16),
        (allocated[1].ptr, 0x2000, 32),
    ]

    slot.position_host[0] = 99
    slot.context_host[0] = 100
    executor.restore_request_checkpoint(checkpoint)

    assert slot.position_host[0] == 17
    assert slot.context_host[0] == 18
    assert [copy[0] for copy in executor.runtime.copies[-4:]] == [
        0x1000,
        0x2000,
        0x3000,
        0x4000,
    ]
    assert [copy[2] for copy in executor.runtime.copies[-4:]] == [16, 32, 8, 8]
    executor.release_request_checkpoint(checkpoint)
    assert freed == list(reversed(allocated))
    with pytest.raises(RuntimeError, match="released"):
        executor.restore_request_checkpoint(checkpoint)


def test_nextn_checkpoint_uses_batch_session_logical_cursor(monkeypatch) -> None:
    executor, _slot, _allocated, _freed = _executor(monkeypatch)
    executor._batch_sessions = (SimpleNamespace(position=18, _position=18),)

    checkpoint = executor.capture_request_checkpoint(7)

    assert checkpoint.position == 18
    assert checkpoint.context_length == 19


def test_nextn_root_snapshot_captures_and_restores_slot_state(monkeypatch) -> None:
    executor, slot, _allocated, _freed = _executor(monkeypatch)
    executor.max_requests = 2
    executor._request_slots = {7: 1}
    executor.scratch.layer_conv_states = (_Buffer(0x10000, 32),)
    executor.scratch.layer_recurrent_states = (_Buffer(0x20000, 64),)
    executor._provider_root_state_snapshots = (
        _Buffer(0x30000, 32),
        _Buffer(0x40000, 64),
    )
    executor._batch_sessions = (
        SimpleNamespace(position=0, _position=0),
        SimpleNamespace(position=18, _position=18),
    )

    executor.capture_request_root_state(7)

    assert executor._provider_root_state_metadata[7] == (1, 17, 18)
    assert [copy[:3] for copy in executor.runtime.copies[-2:]] == [
        (0x30010, 0x10010, 16),
        (0x40020, 0x20020, 32),
    ]

    slot.position_host[0] = 99
    slot.context_host[0] = 100
    executor._batch_sessions[1].position = 99
    executor.restore_request_root_state(7)

    assert [copy[:3] for copy in executor.runtime.copies[-4:]] == [
        (0x10010, 0x30010, 16),
        (0x20020, 0x40020, 32),
        (0x3000, slot.position_host.ctypes.data, 8),
        (0x4000, slot.context_host.ctypes.data, 8),
    ]
    assert slot.position_host[0] == 17
    assert slot.context_host[0] == 18
    assert executor._batch_sessions[1]._position == 18


def test_nextn_root_snapshot_plan_precomputes_slot_adjusted_pointer_tables(
    monkeypatch,
) -> None:
    executor, _slot, allocated, _freed = _executor(monkeypatch)
    executor.max_requests = 2
    executor.backend = "test_backend"
    executor.scratch.layer_conv_states = (
        _Buffer(0x10000, 32),
        _Buffer(0x11000, 32),
    )
    executor.scratch.layer_recurrent_states = (
        _Buffer(0x20000, 64),
        _Buffer(0x21000, 64),
    )
    executor._provider_root_state_snapshots = (
        _Buffer(0x30000, 32),
        _Buffer(0x31000, 32),
        _Buffer(0x40000, 64),
        _Buffer(0x41000, 64),
    )
    monkeypatch.setenv("TEST_ROOT_COPY", "1")
    monkeypatch.setattr(
        "hipengine.runtime.qwen35_gguf_nextn.backend_package_capability",
        lambda *args: {"enabled_env": "TEST_ROOT_COPY", "enabled_default": False},
    )
    monkeypatch.setattr(
        "hipengine.runtime.qwen35_gguf_nextn.is_registered", lambda key: True
    )
    kernel = object()
    monkeypatch.setattr(
        "hipengine.runtime.qwen35_gguf_nextn.resolve", lambda **kwargs: kernel
    )
    uploads = []

    def copy_host(buffer, host_ptr, nbytes, *, runtime):
        del runtime
        if int(nbytes) == 4:
            values = tuple(ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_int32))[:1])
        else:
            count = int(nbytes) // 8
            values = tuple(
                ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_uint64))[:count]
            )
        uploads.append((buffer, values))

    monkeypatch.setattr(
        "hipengine.runtime.qwen35_gguf_nextn.copy_host_to_device", copy_host
    )

    plan = executor._allocate_provider_root_state_copy_plan()

    assert plan is not None
    assert plan.kernel is kernel
    assert plan.layer_count == 2
    assert plan.conv_row_nbytes == 16
    assert plan.recurrent_row_nbytes == 32
    assert [values for _buffer, values in uploads] == [
        (0x10000, 0x11000, 0x10010, 0x11010),
        (0x30000, 0x31000, 0x30010, 0x31010),
        (0x20000, 0x21000, 0x20020, 0x21020),
        (0x40000, 0x41000, 0x40020, 0x41020),
        (0,),
    ]
    assert tuple(allocated[-5:]) == plan.buffers


def test_nextn_root_snapshot_uses_fused_pointer_table_copy_when_available(
    monkeypatch,
) -> None:
    executor, slot, _allocated, _freed = _executor(monkeypatch)
    executor.max_requests = 2
    executor._request_slots = {7: 1}
    executor.scratch.layer_conv_states = (_Buffer(0x10000, 32),)
    executor.scratch.layer_recurrent_states = (_Buffer(0x20000, 64),)
    executor._provider_root_state_snapshots = (
        _Buffer(0x30000, 32),
        _Buffer(0x40000, 64),
    )
    executor._batch_sessions = (
        SimpleNamespace(position=0, _position=0),
        SimpleNamespace(position=18, _position=18),
    )
    executor._batch_session = SimpleNamespace(_dflash_commit_library="library")
    calls = []
    executor._provider_root_state_copy_plan = SimpleNamespace(
        kernel=lambda *args, **kwargs: calls.append((args, kwargs)),
        live_conv_table=_Buffer(0x50000, 32),
        snapshot_conv_table=_Buffer(0x60000, 32),
        live_recurrent_table=_Buffer(0x70000, 32),
        snapshot_recurrent_table=_Buffer(0x80000, 32),
        row_zero_i32=_Buffer(0x90000, 4),
        layer_count=2,
        conv_row_nbytes=16,
        recurrent_row_nbytes=32,
    )

    executor.capture_request_root_state(7)
    assert executor.runtime.copies == []
    args, kwargs = calls.pop(0)
    assert args == (
        0x50010,
        0x60010,
        16,
        0x70010,
        0x80010,
        32,
        0x90000,
        2,
    )
    assert kwargs["library"] == "library"

    executor.restore_request_root_state(7)
    args, _kwargs = calls.pop(0)
    assert args[:6] == (0x60010, 0x50010, 16, 0x80010, 0x70010, 32)
    assert [copy[0] for copy in executor.runtime.copies] == [0x3000, 0x4000]


def test_nextn_root_snapshot_rejects_slot_reuse_and_reset_invalidates(monkeypatch) -> None:
    executor, slot, _allocated, _freed = _executor(monkeypatch)
    executor.max_requests = 2
    executor._request_slots = {7: 1}
    executor._provider_root_state_metadata[7] = (0, 18, 18)
    executor._provider_root_state_snapshots = ()
    executor.scratch.layer_conv_states = ()
    executor.scratch.layer_recurrent_states = ()

    with pytest.raises(RuntimeError, match="unavailable"):
        executor.restore_request_root_state(7)

    executor._provider_root_state_metadata[7] = (1, 18, 18)
    slot.zero_states = lambda runtime: None
    executor._batch_sessions = (
        SimpleNamespace(position=0, _position=0),
        SimpleNamespace(position=18, _position=18),
    )
    executor.reset_request(7)

    assert 7 not in executor._provider_root_state_metadata
    assert executor._batch_sessions[1]._position == 0


def test_nextn_fingerprint_reads_only_owned_kv_slot(monkeypatch) -> None:
    executor, slot, _allocated, _freed = _executor(monkeypatch)
    executor._request_slots = {7: 1}
    executor.scratch.slot_count = 2
    slot.max_positions = 4
    slot.context_host[0] = 2
    slot.full_key_caches = (_Buffer(0x10000, 24),)
    slot.full_value_caches = (_Buffer(0x20000, 40),)
    reads = []

    def copy_to_host(host_ptr, buffer, nbytes, *, runtime):
        reads.append((int(buffer.ptr), int(buffer.nbytes), int(nbytes)))
        ctypes.memset(int(host_ptr), int(buffer.ptr) & 0xFF, int(nbytes))

    monkeypatch.setattr(
        "hipengine.runtime.qwen35_gguf_nextn.copy_device_to_host",
        copy_to_host,
    )

    fingerprint = executor.request_state_fingerprint(7)

    assert reads[-2:] == [
        (0x1000C, 6, 6),
        (0x20014, 10, 10),
    ]
    assert fingerprint["visible_kv_bytes"] == 16


def test_nextn_checkpoint_rejects_wrong_request_owner(monkeypatch) -> None:
    executor, _slot, _allocated, _freed = _executor(monkeypatch)
    checkpoint = executor.capture_request_checkpoint(7)
    executor._request_slots[7] = 1

    with pytest.raises(RuntimeError, match="slot ownership"):
        executor.restore_request_checkpoint(checkpoint)
