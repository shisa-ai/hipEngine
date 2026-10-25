"""Explicit canonical, provisional and rollback storage contracts (CPU only)."""
from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.qwen35_gguf_state import (
    CanonicalLinearState, CanonicalResidentState, LinearRollbackStorage,
    LinearStateStorage, StateMemory,
)
from tests.test_unit_qwen35_gguf_state_ownership import (
    _FakeDeviceAllocator, _minimal_session,
)


@pytest.mark.parametrize("failed_allocation", [1, 2, 4])
@pytest.mark.parametrize("kind", ["rollback", "initial"])
def test_snapshot_growth_retains_old_owners_on_failure(failed_allocation, kind):
    allocator = _FakeDeviceAllocator()
    copies = []
    runtime = SimpleNamespace(memcpy=lambda *args: copies.append(args))
    memory = StateMemory(runtime, allocator.malloc, allocator.free)
    old_source = CanonicalLinearState((DeviceBuffer(1, 16),), (DeviceBuffer(2, 32),))
    new_source = CanonicalLinearState(
        (DeviceBuffer(3, 64), DeviceBuffer(4, 64)),
        (DeviceBuffer(5, 64), DeviceBuffer(6, 64)),
    )
    store = LinearRollbackStorage() if kind == "rollback" else LinearStateStorage()

    def capture(source):
        if kind == "rollback":
            return store.capture(source, memory)
        return store.ensure(source, rows=1, exact=True, memory=memory)

    capture(old_source)
    old = store.buffers if kind == "rollback" else (*store.conv, *store.recurrent)
    allocator.fail_nth_next(failed_allocation)
    with pytest.raises(RuntimeError, match="injected"):
        capture(new_source)
    retained = store.buffers if kind == "rollback" else (*store.conv, *store.recurrent)
    assert retained == old
    allocator.assert_live(old, context="snapshot growth")
    store.close(memory)
    store.close(memory)
    assert not allocator.live
    assert all(allocator.release_count(buffer) == 1 for buffer in old)


def test_canonical_slot_borrows_scratch_and_has_independent_cursor():
    allocator = _FakeDeviceAllocator()
    memory = StateMemory(None, allocator.malloc, allocator.free)
    buffer = allocator.malloc(64)
    root = CanonicalResidentState(scratch_owner=SimpleNamespace(buffers=(buffer,)), position=8)
    slot = CanonicalResidentState(scratch_owner=root.scratch_owner, owns_scratch=False, position=3)
    slot.close_scratch(memory)
    assert root.position == 8
    assert slot.position == 3
    allocator.assert_live((buffer,), context="slot close")
    root.close_scratch(memory)
    root.close_scratch(memory)
    assert not allocator.live
    assert allocator.release_count(buffer) == 1


def test_slot_cleanup_releases_its_rollback_storage(monkeypatch):
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    backups = tuple(allocator.malloc(32) for _ in range(2))
    session._linear_state_snapshot_backups = backups
    session._close_resident_slot_view_buffers(runtime=session.runtime)
    session._close_resident_slot_view_buffers(runtime=session.runtime)
    assert not allocator.live
    assert all(allocator.release_count(buffer) == 1 for buffer in backups)


@pytest.mark.parametrize("failure_stage", ["dms", "resolve", "build"])
def test_verifier_constructor_failure_releases_producer_capture(monkeypatch, failure_stage):
    from hipengine.runtime import qwen35_gguf_mtp as mtp
    session = _minimal_session()
    session.backend = "hip_gfx1151"
    session.compiler_version = "test"
    session.require_cached_build = True
    session._dms_backend = object() if failure_stage == "dms" else None
    closes = []
    def allocate(*args, **kwargs):
        session._verify_linear_initial_snapshot_users += 1
        def close():
            closes.append(True)
            session._release_verify_initial_state_capture()
        return SimpleNamespace(close=close)
    def fail(*args, **kwargs):
        raise RuntimeError("injected constructor failure")
    monkeypatch.setattr(mtp, "_resolve_gguf_verifier_backend", lambda *args: "test")
    monkeypatch.setattr(mtp._StateJournal, "allocate", allocate)
    monkeypatch.setattr(mtp, "resolve", fail if failure_stage == "resolve" else lambda **kwargs: object())
    monkeypatch.setattr(mtp, "build_dflash_accept", fail if failure_stage == "build" else lambda **kwargs: object())
    if failure_stage == "dms":
        monkeypatch.setattr(mtp, "_DMSStoreJournal", fail)
    with pytest.raises(RuntimeError, match="injected constructor failure"):
        mtp.Qwen35GGUFTransactionalVerifier(session, target_verify_mode="native")
    assert closes == [True]
    assert session._verify_linear_initial_snapshot_users == 0


def test_session_close_refuses_live_capture_before_mutating_any_owner():
    session = _minimal_session()
    session._verify_linear_initial_snapshot_users = 1
    closed = []
    pool = SimpleNamespace(close=lambda: closed.append(True))
    session._gguf_prefix_snapshot_arena_pool = pool
    with pytest.raises(RuntimeError, match="active initial-state"):
        session.close()
    assert closed == []
    assert session._gguf_prefix_snapshot_arena_pool is pool


def test_root_close_checks_slot_captures_before_closing_root_pool():
    root = _minimal_session()
    view = _minimal_session()
    view._verify_linear_initial_snapshot_users = 1
    root._resident_slot_views = [view]
    closed = []
    root._gguf_prefix_snapshot_arena_pool = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="active initial-state"):
        root.close()
    assert not closed
    assert root._resident_slot_views == [view]


def test_session_keeps_prefix_pool_when_close_needs_retry():
    session = _minimal_session()
    attempts = []
    def close():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("pending copies")
    pool = SimpleNamespace(close=close)
    session._gguf_prefix_snapshot_arena_pool = pool
    with pytest.raises(RuntimeError, match="pending copies"):
        session._close_prefix_snapshot_pool()
    assert session._gguf_prefix_snapshot_arena_pool is pool
    session._close_prefix_snapshot_pool()
    assert "_gguf_prefix_snapshot_arena_pool" not in session.__dict__


def test_clone_does_not_publish_boundary_when_packed_flush_fails():
    session = _minimal_session()
    session._position = 17
    session._packed_decode_state_dirty = True
    session.flush_packed_decode_state = lambda: False
    published = []
    session._set_full_attention_position_device = lambda *args, **kwargs: published.append(args)
    with pytest.raises(RuntimeError, match="could not flush"):
        session._commit_prefix_state_clone(256, stream=0)
    assert session.position == 17
    assert not published


def test_active_producer_capture_cannot_free_initial_state():
    allocator = _FakeDeviceAllocator()
    memory = StateMemory(None, allocator.malloc, allocator.free)
    source = CanonicalLinearState((DeviceBuffer(1, 16),), (DeviceBuffer(2, 32),))
    storage = LinearStateStorage()
    storage.ensure(source, rows=1, exact=True, memory=memory)
    storage.users = 1
    with pytest.raises(RuntimeError, match="active initial-state"):
        storage.close(memory)
    allocator.assert_live((*storage.conv, *storage.recurrent), context="capture lease")
    storage.users = 0
    storage.close(memory)
    assert not allocator.live
