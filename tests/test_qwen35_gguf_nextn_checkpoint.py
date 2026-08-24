from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNExecutor


class _Runtime:
    def __init__(self) -> None:
        self.copies = []

    def memcpy(self, dst, src, nbytes, kind):
        self.copies.append((int(dst), int(src), int(nbytes), kind))


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


def test_nextn_checkpoint_rejects_wrong_request_owner(monkeypatch) -> None:
    executor, _slot, _allocated, _freed = _executor(monkeypatch)
    checkpoint = executor.capture_request_checkpoint(7)
    executor._request_slots[7] = 1

    with pytest.raises(RuntimeError, match="slot ownership"):
        executor.restore_request_checkpoint(checkpoint)
