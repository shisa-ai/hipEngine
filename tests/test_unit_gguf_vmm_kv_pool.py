"""Stable-address VMM growth for the GGUF global KV pool.

The pool's planes are reserved through :class:`VirtualMemoryBuffer` so shared
prefix admissions can grow past the initial physical allocation without moving a
published page pointer. These tests mock the HIP VMM surface and the pointer
table transfers; no GPU is touched.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer, memory_stats
from hipengine.core.virtual_memory import VirtualMemoryBuffer
from hipengine.runtime import qwen35_gguf_runner as runner
from tests.test_unit_gguf_packed_workspace_stability import (
    _allocator_fake_runner,
    _int8_kv_layout,
)
from tests.test_unit_virtual_memory import FakeVmmLibrary, _runtime

# ---------------------------------------------------------------------------
# Fake HIP runtime with a device-synchronize entry point.
# ---------------------------------------------------------------------------


class FakeSessionVmmLibrary(FakeVmmLibrary):
    def __init__(self, device: int = 0) -> None:
        super().__init__(device=device)
        self.sync_calls = 0
        self.hipDeviceSynchronize = self._make_sync()

    def _make_sync(self):
        def sync():
            self.sync_calls += 1
            return 0

        from tests.test_unit_virtual_memory import FakeFunction

        return FakeFunction(sync)


def _session_runtime(library: FakeSessionVmmLibrary):
    runtime = _runtime(library)
    # `VirtualMemoryBuffer.reserve` reads the current device through the library.
    return runtime


# ---------------------------------------------------------------------------
# Mocked device allocator for pointer tables.
# ---------------------------------------------------------------------------


class _FakeDeviceAllocator:
    def __init__(self) -> None:
        self.next_pointer = 0x100000
        self.uploaded: dict[int, bytes] = {}
        self.live: set[int] = set()

    def malloc(self, nbytes, **kwargs):
        buffer = DeviceBuffer(ptr=self.next_pointer, nbytes=int(nbytes))
        self.next_pointer += int(nbytes) + 256
        self.live.add(buffer.ptr)
        return buffer

    def free(self, buffer, **kwargs):
        self.live.remove(buffer.ptr)

    def upload(self, buffer, host, nbytes, **kwargs):
        self.uploaded[buffer.ptr] = ctypes.string_at(host, nbytes)


def _session(
    monkeypatch,
    *,
    library: FakeSessionVmmLibrary | None = None,
    budget_mib: int | None = 128,
):
    allocator = _FakeDeviceAllocator()
    monkeypatch.setattr(runner, "malloc", allocator.malloc)
    monkeypatch.setattr(runner, "free", allocator.free)
    monkeypatch.setattr(runner, "copy_host_to_device", allocator.upload)
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    session.defer_kv_allocation = True
    session.runner = _allocator_fake_runner()
    session.scratch = SimpleNamespace()
    session.runtime = _session_runtime(library or FakeSessionVmmLibrary())
    session._device_kv_layout = _int8_kv_layout()
    session.kv_pool_memory_budget_mib = budget_mib
    session.model_path = "test.gguf"
    session.kv_storage_dtype = runner.DType.INT8_PER_TOKEN_HEAD
    session.kv_storage_layout = "uniform"
    return session, allocator


def _page_pointers(allocator, pool):
    global_pool = pool.global_pool
    result: dict[str, tuple[int, ...]] = {}
    for role, pointer in global_pool._pointer_table_pointers.items():
        table = np.frombuffer(allocator.uploaded[pointer], dtype=np.uint64)
        result[role] = tuple(int(value) for value in table)
    return result


# ---------------------------------------------------------------------------
# Repeated shared-prefix admissions crossing the initial boundary.
# ---------------------------------------------------------------------------


def test_vmm_growth_preserves_pointer_values_across_shared_prefix_admissions(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        initial_pages = pool.current_pages
        initial_pointers = _page_pointers(allocator, pool)
        # Keep the growth step small so each admission crosses the boundary.
        pool._max_pages = 12
        pool._growth_chunk_pages = 1
        # Every plane's scale metadata is sized for the committed capacity.
        backing = pool.backing
        for layer_id in (1, 3):
            metadata = backing.full_kv_scale_metadata[layer_id]
            assert metadata is not None
            assert int(metadata.k_scale.shape[0]) == initial_pages

        first = pool.allocate(1, initial_pages)
        assert first.pool_page_capacity == initial_pages
        assert len(pool._chunks) == 1

        second = pool.admit_with_shared_prefix(
            2, first.block_ids, suffix_pages=3
        )
        assert second.block_ids[:initial_pages] == first.block_ids
        assert pool.current_pages > initial_pages
        assert len(pool._chunks) == 1
        assert second.chunk_start_block_id == 0

        # Previously published page pointers must not move.
        grown_pointers = _page_pointers(allocator, pool)
        for role, pointers in initial_pointers.items():
            assert grown_pointers[role][: len(pointers)] == pointers

        third = pool.admit_with_shared_prefix(
            3, second.block_ids, suffix_pages=2
        )
        assert third.block_ids[: len(second.block_ids)] == second.block_ids
        assert pool.current_pages > len(second.block_ids)
        assert third.pool_page_capacity == pool.current_pages

        # Scale metadata tracks the committed capacity honestly.
        backing = pool.backing
        for layer_id in (1, 3):
            metadata = backing.full_kv_scale_metadata[layer_id]
            assert metadata is not None
            assert int(metadata.k_scale.shape[0]) == pool.current_pages
            assert int(metadata.v_scale.shape[0]) == pool.current_pages

        for request_id in (3, 2, 1):
            pool.release(request_id)
        pool.close()
    finally:
        pass
    assert allocator.live == set()


def test_vmm_reserves_to_max_pages_but_commits_only_the_initial_floor(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, _allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        expected_max_pages = max_pages_from_budget(session, pool.page_bytes)
        assert pool.max_pages == expected_max_pages
        for buffer in pool.backing.buffers:
            owner = buffer.owner
            page_nbytes = int(buffer.nbytes) // 4
            # The reservation covers the whole budget; only four pages are
            # committed, rounded up to the device granularity.
            assert owner.capacity_bytes >= expected_max_pages * page_nbytes
            expected_committed = -(-(4 * page_nbytes) // owner.granularity) * owner.granularity
            assert owner.committed_bytes == expected_committed
            assert owner.committed_bytes % owner.granularity == 0
        assert library.reserved_sizes
    finally:
        pool.close()


def max_pages_from_budget(session, page_bytes: int) -> int:
    return int(session.kv_pool_memory_budget_mib) * 1024**2 // page_bytes


def test_virtual_reservation_is_not_counted_as_physical_memory(monkeypatch) -> None:
    session, _allocator = _session(monkeypatch)
    before = memory_stats()
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        assert memory_stats() == before
    finally:
        pool.close()


def test_runtime_without_vmm_keeps_chunk_local_confinement(monkeypatch) -> None:
    """An unsupported runtime must not turn on contiguous growth."""

    session, _allocator = _session(monkeypatch)
    # A runtime without a HIP library cannot provide stable addresses.
    session.runtime = SimpleNamespace(memset=lambda *args: None)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        assert pool._contiguous_growth is False
        assert pool.current_pages == 4
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Rollback of failed growth.
# ---------------------------------------------------------------------------


def _committed_snapshot(pool) -> dict[int, int]:
    return {
        id(buffer.owner): buffer.owner.committed_bytes
        for buffer in pool.backing.buffers
    }


@pytest.mark.parametrize("failure", ["hipMemCreate", "hipMemMap", "hipMemSetAccess"])
def test_growth_commit_failure_rolls_back_physical_and_state(
    monkeypatch, failure: str
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        baseline_pages = pool.current_pages
        baseline_committed = _committed_snapshot(pool)
        baseline_uploads = dict(allocator.uploaded)
        library.fail[failure] = 2
        with pytest.raises(Exception):
            pool.grow(1)
        del library.fail[failure]

        assert pool.current_pages == baseline_pages
        assert _committed_snapshot(pool) == baseline_committed
        assert dict(allocator.uploaded) == baseline_uploads
        assert len(pool._chunks) == 1
    finally:
        pool.close()


def test_growth_partial_commit_failure_rolls_back_earlier_planes(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        baseline_pages = pool.current_pages
        baseline_committed = _committed_snapshot(pool)
        baseline_mapped = len(library.mapped)
        baseline_unmapped = len(library.unmapped)
        baseline_released = len(library.released)
        real_create = library.hipMemCreate.func
        calls = {"n": 0}

        def failing_create(out_handle, size, prop, flags):
            calls["n"] += 1
            if calls["n"] == 3:
                return 2
            return real_create(out_handle, size, prop, flags)

        library.hipMemCreate.func = failing_create
        with pytest.raises(Exception):
            pool.grow(1)
        library.hipMemCreate.func = real_create

        # The two planes committed before the failure were rolled back.
        assert pool.current_pages == baseline_pages
        assert _committed_snapshot(pool) == baseline_committed
        assert len(library.mapped) == baseline_mapped + 2
        assert len(library.unmapped) == baseline_unmapped + 2
        assert len(library.released) == baseline_released + 2
    finally:
        pool.close()


def test_growth_table_upload_failure_rolls_back_commit_and_tables(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    real_upload = runner.copy_host_to_device
    try:
        baseline_pages = pool.current_pages
        baseline_committed = _committed_snapshot(pool)
        baseline_tables = dict(pool.global_pool._pointer_table_pointers)

        def failing_upload(*args, **kwargs):
            raise RuntimeError("pointer table upload failed")

        monkeypatch.setattr(runner, "copy_host_to_device", failing_upload)
        with pytest.raises(RuntimeError, match="pointer table upload failed"):
            pool.grow(1)
        monkeypatch.setattr(runner, "copy_host_to_device", real_upload)

        assert pool.current_pages == baseline_pages
        assert _committed_snapshot(pool) == baseline_committed
        assert dict(pool.global_pool._pointer_table_pointers) == baseline_tables
    finally:
        pool.close()


def test_growth_descriptor_upload_failure_rolls_back_commit_and_tables(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    real_upload = runner.copy_host_to_device
    descriptor_pointer = pool.global_pool.storage_view().metadata_descriptor_ptr
    try:
        baseline_pages = pool.current_pages
        baseline_committed = _committed_snapshot(pool)

        def failing_upload(buffer, host, nbytes, **kwargs):
            if int(buffer.ptr) == int(descriptor_pointer):
                raise RuntimeError("descriptor upload failed")
            return real_upload(buffer, host, nbytes, **kwargs)

        monkeypatch.setattr(runner, "copy_host_to_device", failing_upload)
        with pytest.raises(RuntimeError, match="descriptor upload failed"):
            pool.grow(1)
        monkeypatch.setattr(runner, "copy_host_to_device", real_upload)

        assert pool.current_pages == baseline_pages
        assert _committed_snapshot(pool) == baseline_committed
    finally:
        pool.close()


def test_close_releases_every_arena_exactly_once(monkeypatch) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    pool.grow(2)
    created_handles = [handle for handle, _size in library.created]
    reservation_count = len(library.reserved_sizes)
    pool.close()
    pool.close()

    # One release per created handle, one unmap per mapped segment, and one
    # address free per reservation, with no double free.
    assert sorted(library.released) == sorted(created_handles)
    assert len(library.unmapped) == len(library.mapped)
    assert len(library.freed_addresses) == reservation_count
    assert library.reservations == {}
    assert allocator.live == set()


# ---------------------------------------------------------------------------
# Transactional rollback primitive.
# ---------------------------------------------------------------------------


def test_rollback_to_releases_mappings_above_target() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(3 * 4096, runtime=_runtime(library))
    base = buffer.ptr
    buffer.commit_to(4096)
    buffer.commit_to(3 * 4096)
    assert buffer.committed_bytes == 3 * 4096

    assert buffer.rollback_to(4096) == 4096
    assert buffer.ptr == base
    assert buffer.committed_bytes == 4096
    assert library.unmapped == [(base + 4096, 8192)]
    assert library.released == [0x9010]

    # The surviving segment is still usable and close is still exact-once.
    buffer.commit_to(2 * 4096)
    assert buffer.committed_bytes == 2 * 4096
    buffer.close()
    assert len(library.freed_addresses) == 1


def test_rollback_to_rejects_non_granular_and_forward_targets() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(3 * 4096, runtime=_runtime(library))
    buffer.commit_to(4096)
    buffer.commit_to(3 * 4096)  # one segment covering [4096, 12288)
    with pytest.raises(ValueError, match="multiple of the granularity"):
        buffer.rollback_to(1)
    with pytest.raises(ValueError, match="exceeds the committed size"):
        buffer.rollback_to(4 * 4096)
    with pytest.raises(ValueError, match="boundary"):
        buffer.rollback_to(8192)
    assert buffer.committed_bytes == 3 * 4096
    assert library.unmapped == []
