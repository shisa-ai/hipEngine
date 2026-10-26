"""Stable-address VMM growth for the GGUF global KV pool.

The pool's planes are reserved through :class:`VirtualMemoryBuffer` so shared
prefix admissions can grow past the initial physical allocation without moving a
published page pointer. These tests mock the HIP VMM surface and the pointer
table transfers; no GPU is touched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer, memory_stats
from hipengine.core.virtual_memory import VirtualMemoryBuffer
from hipengine.kvcache.pool import KVStorageFault
from hipengine.runtime import qwen35_gguf_kv as gguf_kv
from hipengine.runtime import qwen35_gguf_runner as runner
from hipengine.runtime.memory_admission import MemoryAdmissionRefused
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


class CoarseGranularityVmmLibrary(FakeSessionVmmLibrary):
    """A device whose allocation granularity exceeds one KV plane's payload."""

    GRANULARITY = 2 * 1024**2


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

    def upload(self, buffer, array, nbytes=None, **kwargs):
        count = int(array.nbytes if nbytes is None else nbytes)
        self.uploaded[buffer.ptr] = np.ascontiguousarray(array).tobytes()[:count]


def _session(
    monkeypatch,
    *,
    library: FakeSessionVmmLibrary | None = None,
    budget_mib: int | None = 128,
):
    allocator = _FakeDeviceAllocator()
    monkeypatch.setattr(runner, "malloc", allocator.malloc)
    monkeypatch.setattr(runner, "free", allocator.free)
    monkeypatch.setattr(gguf_kv, "malloc", allocator.malloc)
    monkeypatch.setattr(gguf_kv, "free", allocator.free)
    monkeypatch.setattr(runner, "copy_host_array_to_device", allocator.upload)
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
    before = memory_stats()["current_allocated_bytes"]
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        # Only the committed, granularity-aligned physical bytes are counted;
        # the much larger address reservation is not physical memory.
        committed = sum(
            int(buffer.owner.committed_bytes) for buffer in pool.backing.buffers
        )
        reserved = sum(
            int(buffer.owner.capacity_bytes) for buffer in pool.backing.buffers
        )
        assert committed > 0
        assert reserved > committed
        assert memory_stats()["current_allocated_bytes"] == before + committed
    finally:
        pool.close()
    assert memory_stats()["current_allocated_bytes"] == before


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


def test_supported_symbols_with_unsupported_driver_falls_back_to_chunks(
    monkeypatch,
) -> None:
    """Exported VMM symbols do not prove driver support.

    A runtime that exports every entry point but returns ``hipErrorNotSupported``
    (801) must release whatever it reserved and fall back to chunk-local storage
    instead of failing the session.
    """

    library = FakeSessionVmmLibrary()
    library.fail["hipMemAddressReserve"] = 801
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        assert pool._contiguous_growth is False
        assert pool.current_pages == 4
        # The failed VMM attempt left no address reservation behind.
        assert library.reserved_sizes == []
    finally:
        pool.close()
    assert allocator.live == set()


def test_pool_budget_accounts_physical_commitment_not_logical_pages(
    monkeypatch,
) -> None:
    """The pool budget prices granularity-aligned physical bytes, not pages."""

    session, _allocator = _session(monkeypatch, library=CoarseGranularityVmmLibrary())
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        logical = pool.current_pages * pool.page_bytes
        physical = sum(
            int(buffer.owner.committed_bytes) for buffer in pool.backing.buffers
        )
        # Alignment rounding makes the physical commitment strictly larger than
        # the logical page payload at this granularity.
        assert physical > logical
        assert pool.accounted_bytes == physical
        assert pool.resident_consumers[0].name == "kv_arena"
        assert pool.resident_consumers[0].bytes == physical
        # Prospective pricing agrees with the committed bytes and is padded.
        assert pool._physical_bytes_for_pages is not None
        assert pool._physical_bytes_for_pages(pool.current_pages) == physical
        assert pool._physical_bytes_for_pages(pool.current_pages) > logical
    finally:
        pool.close()


def test_initial_admission_refuses_padded_physical_before_allocating(
    monkeypatch,
) -> None:
    """A coarse granularity refuses before any reservation or malloc.

    The configured budget fits the logical page payload but not the
    per-plane-granularity physical commitment, so the initial admission must
    refuse rather than map padded memory and discover the overcommit later.
    """

    library = CoarseGranularityVmmLibrary()
    session, allocator = _session(monkeypatch, library=library, budget_mib=None)
    cfg = session.runner.weights.config
    layout = session._device_kv_layout
    page_bytes = runner._qwen35_gguf_kv_page_bytes(cfg, layout)
    plane_nbytes = dict(runner._qwen35_gguf_kv_plane_specs(cfg, layout))
    logical = 4 * page_bytes
    padded = runner._qwen35_gguf_vmm_physical_bytes(
        4, plane_page_nbytes=plane_nbytes, granularity=library.GRANULARITY
    )
    assert padded > logical
    session.kv_pool_memory_budget_mib = logical // 1024**2 + 1

    with pytest.raises(MemoryAdmissionRefused):
        session.create_global_device_kv_pool(page_capacity=4, generation=1)

    # Nothing was allocated or reserved.
    assert allocator.live == set()
    assert library.reserved_sizes == []


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
    real_upload = runner.copy_host_array_to_device
    try:
        baseline_pages = pool.current_pages
        baseline_committed = _committed_snapshot(pool)
        baseline_tables = dict(pool.global_pool._pointer_table_pointers)
        baseline_live = set(allocator.live)

        def failing_upload(*args, **kwargs):
            raise RuntimeError("pointer table upload failed")

        monkeypatch.setattr(runner, "copy_host_array_to_device", failing_upload)
        with pytest.raises(RuntimeError, match="pointer table upload failed"):
            pool.grow(1)
        monkeypatch.setattr(runner, "copy_host_array_to_device", real_upload)

        assert pool.current_pages == baseline_pages
        assert _committed_snapshot(pool) == baseline_committed
        assert dict(pool.global_pool._pointer_table_pointers) == baseline_tables
        # The table allocated for the failed upload is freed, not leaked.
        assert allocator.live == baseline_live
    finally:
        pool.close()


def test_growth_descriptor_upload_failure_rolls_back_commit_and_tables(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    real_upload = runner.copy_host_array_to_device
    descriptor_pointer = pool.global_pool.storage_view().metadata_descriptor_ptr
    try:
        baseline_pages = pool.current_pages
        baseline_committed = _committed_snapshot(pool)
        baseline_live = set(allocator.live)

        def failing_upload(buffer, host, nbytes, **kwargs):
            if int(buffer.ptr) == int(descriptor_pointer):
                raise RuntimeError("descriptor upload failed")
            return real_upload(buffer, host, nbytes, **kwargs)

        monkeypatch.setattr(runner, "copy_host_array_to_device", failing_upload)
        with pytest.raises(RuntimeError, match="descriptor upload failed"):
            pool.grow(1)
        monkeypatch.setattr(runner, "copy_host_array_to_device", real_upload)

        assert pool.current_pages == baseline_pages
        assert _committed_snapshot(pool) == baseline_committed
        # The tables uploaded before the descriptor failure are freed too.
        assert allocator.live == baseline_live
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


# ---------------------------------------------------------------------------
# Idle reclamation of grown VMM capacity.
# ---------------------------------------------------------------------------


def _descriptor_values(allocator, pool) -> tuple[int, int]:
    pointer = int(pool.global_pool.storage_view().metadata_descriptor_ptr)
    raw = np.frombuffer(allocator.uploaded[pointer], dtype=np.uint64)
    return (int(raw[0]), int(raw[1]))


def test_vmm_shrink_unmaps_the_grown_commit_and_republishes_tables(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        initial_pages = pool.current_pages
        initial_committed = _committed_snapshot(pool)
        initial_live = set(allocator.live)
        initial_generation = _descriptor_values(allocator, pool)[0]
        pool._max_pages = 12
        pool._growth_chunk_pages = 1
        pool.allocate(1, 8)
        assert pool.current_pages == 8
        assert _committed_snapshot(pool) != initial_committed
        assert _descriptor_values(allocator, pool)[1] == 8
        pool.release(1)

        released = pool.shrink_idle(now_seconds=1.0)

        assert released == 4
        assert pool.current_pages == initial_pages
        # The physical commitment returns to the load-time floor and the arena
        # base never moved.
        assert _committed_snapshot(pool) == initial_committed
        assert pool.backing.pages == initial_pages
        assert pool.storage_view().planes[0].shape == (initial_pages,)
        # The descriptor names the retained capacity at a newer generation.
        generation, pages = _descriptor_values(allocator, pool)
        assert pages == initial_pages
        assert generation > initial_generation
        # One table per plane, republished for the retained pages, and the
        # tables for the grown capacity were freed rather than leaked.
        assert len(allocator.live) == len(initial_live)
        for role, pointer in pool.global_pool._pointer_table_pointers.items():
            table = np.frombuffer(allocator.uploaded[pointer], dtype=np.uint64)
            assert tuple(int(value) for value in table) == tuple(
                pool.global_pool.page_pointer(role, page)
                for page in range(initial_pages)
            )
        # Retained page pointers are unchanged, so no consumer rebinding is
        # needed for the pages that remain.
        assert pool.pointer_for(0) == pool.global_pool.page_pointer(
            next(iter(pool.global_pool._pointer_table_pointers)), 0
        )
    finally:
        pool.close()
    assert allocator.live == set()
    assert len(library.unmapped) == len(library.mapped)


def test_vmm_shrink_rolls_back_the_commit_when_a_plane_cannot_unmap(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        initial_pages = pool.current_pages
        initial_committed = _committed_snapshot(pool)
        pool._max_pages = 12
        pool._growth_chunk_pages = 1
        pool.allocate(1, 8)
        pool.release(1)
        baseline_committed = _committed_snapshot(pool)
        baseline_tables = dict(pool.global_pool._pointer_table_pointers)
        real_unmap = library.hipMemUnmap.func
        calls = {"n": 0}

        def failing_unmap(ptr, size):
            calls["n"] += 1
            if calls["n"] == 2:
                return 2
            return real_unmap(ptr, size)

        library.hipMemUnmap.func = failing_unmap
        try:
            with pytest.raises(Exception):
                pool.shrink_idle(now_seconds=1.0)
        finally:
            library.hipMemUnmap.func = real_unmap

        # The plane that did unmap is committed again, so the arena is exactly
        # where it was and the pool still owns every grown page.
        assert pool.current_pages == 8
        assert _committed_snapshot(pool) == baseline_committed
        assert dict(pool.global_pool._pointer_table_pointers) == baseline_tables
        assert pool.stats.shrink_events == 0

        # The retry succeeds once the driver cooperates.
        assert pool.shrink_idle(now_seconds=1.0) == 4
        assert pool.current_pages == initial_pages
        assert _committed_snapshot(pool) == initial_committed
    finally:
        pool.close()
    assert allocator.live == set()


def test_vmm_shrink_restores_the_commit_when_table_upload_fails(
    monkeypatch,
) -> None:
    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    real_upload = runner.copy_host_array_to_device
    try:
        pool._max_pages = 12
        pool._growth_chunk_pages = 1
        pool.allocate(1, 8)
        pool.release(1)
        baseline_committed = _committed_snapshot(pool)
        baseline_live = set(allocator.live)

        def failing_upload(*args, **kwargs):
            raise RuntimeError("retained table upload failed")

        monkeypatch.setattr(runner, "copy_host_array_to_device", failing_upload)
        with pytest.raises(RuntimeError, match="retained table upload failed"):
            pool.shrink_idle(now_seconds=1.0)
        monkeypatch.setattr(runner, "copy_host_array_to_device", real_upload)

        assert pool.current_pages == 8
        assert _committed_snapshot(pool) == baseline_committed
        # The table allocated for the failed upload is freed, not leaked.
        assert allocator.live == baseline_live
        assert pool.shrink_idle(now_seconds=1.0) == 4
    finally:
        pool.close()
    assert allocator.live == set()


def test_vmm_shrink_to_the_floor_can_grow_and_shrink_again(monkeypatch) -> None:
    """The retained floor stays a legal rollback boundary across cycles."""

    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        initial_pages = pool.current_pages
        initial_committed = _committed_snapshot(pool)
        baseline_live = len(allocator.live)
        pool._max_pages = 12
        pool._growth_chunk_pages = 1
        for cycle in range(3):
            pool.allocate(cycle + 1, 8)
            assert pool.current_pages == 8
            assert _committed_snapshot(pool) != initial_committed
            pool.release(cycle + 1)

            assert pool.shrink_idle(now_seconds=float(cycle + 1)) == 4
            assert pool.current_pages == initial_pages
            assert _committed_snapshot(pool) == initial_committed
            assert pool.storage_view().planes[0].shape == (initial_pages,)
            # No table pair leaks across a cycle.
            assert len(allocator.live) == baseline_live
    finally:
        pool.close()
    assert allocator.live == set()
    assert len(library.unmapped) == len(library.mapped)


def test_vmm_shrink_fails_closed_when_the_arena_cannot_be_restored(
    monkeypatch,
) -> None:
    """A partially unmapped arena must never stay advertised as usable."""

    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        initial_pages = pool.current_pages
        pool._max_pages = 12
        pool._growth_chunk_pages = 1
        pool.allocate(1, 8)
        pool.release(1)
        baseline_tables = dict(pool.global_pool._pointer_table_pointers)
        real_unmap = library.hipMemUnmap.func
        real_create = library.hipMemCreate.func
        state = {"unmap_calls": 0, "restore_window": False}

        def failing_unmap(ptr, size):
            state["unmap_calls"] += 1
            if state["unmap_calls"] == 2:
                # One plane is unmapped before the next one refuses, so the
                # shrink has to restore the first.
                state["restore_window"] = True
                return 2
            return real_unmap(ptr, size)

        def failing_create(out_handle, size, prop, flags):
            if state["restore_window"]:
                return 2
            return real_create(out_handle, size, prop, flags)

        library.hipMemUnmap.func = failing_unmap
        library.hipMemCreate.func = failing_create
        try:
            with pytest.raises(KVStorageFault, match="could not restore"):
                pool.shrink_idle(now_seconds=1.0)
        finally:
            library.hipMemUnmap.func = real_unmap
            library.hipMemCreate.func = real_create

        # Part of the arena is unmapped while the ledger still advertises it, so
        # the pool must refuse further admission rather than serve unbacked pages.
        assert pool.current_pages == 8
        assert dict(pool.global_pool._pointer_table_pointers) == baseline_tables
        assert pool.stats.shrink_events == 0
        with pytest.raises(RuntimeError, match="unusable"):
            pool.allocate(2, 1)
        with pytest.raises(RuntimeError, match="unusable"):
            pool.grow(1)
    finally:
        pool.close()
    assert allocator.live == set()
    # The floor the pool would retry towards is intact; only the arena behind it
    # is untrustworthy, which is why admission refuses.
    assert pool.low_water_pages == initial_pages


def test_vmm_failed_shrink_restore_keeps_intermediate_growth_boundaries(
    monkeypatch,
) -> None:
    """Restoring after a partial rollback must rebuild every segment boundary.

    ``commit_to`` appends one segment per call, so restoring straight to the old
    total collapses the intermediate growth boundaries into a single segment
    while ``committed_capacities`` keeps advertising them as legal shrink
    targets. The next shrink to such a target would then fail.
    """

    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        pool._max_pages = 16
        pool._growth_chunk_pages = 1
        for _ in range(4):
            pool.grow(1)
        assert pool.current_pages == 8
        baseline_committed = _committed_snapshot(pool)
        pool.allocate(1, 8)
        pool.release(1)

        real_unmap = library.hipMemUnmap.func
        calls = {"n": 0}

        def failing_unmap(ptr, size):
            calls["n"] += 1
            # The first plane releases its four segments; the fifth call is the
            # second plane's first segment, so the first plane has already
            # rolled back to the floor when the failure lands.
            if calls["n"] == 5:
                return 2
            return real_unmap(ptr, size)

        library.hipMemUnmap.func = failing_unmap
        try:
            with pytest.raises(Exception):
                pool.shrink_idle(now_seconds=1.0)
        finally:
            library.hipMemUnmap.func = real_unmap

        # Every boundary is back, so the pool is exactly where it was.
        assert pool.current_pages == 8
        assert _committed_snapshot(pool) == baseline_committed
        assert pool.stats.shrink_events == 0

        # Holding page 4 makes 5 the free tail, which is an intermediate growth
        # boundary rather than the floor.
        pool.allocate(1, 5)
        try:
            assert pool.shrink_idle(now_seconds=2.0) == 3
            assert pool.current_pages == 5
            assert pool.storage_view().planes[0].shape == (5,)
            assert _committed_snapshot(pool) != baseline_committed
        finally:
            pool.release(1)
    finally:
        pool.close()
    assert allocator.live == set()
    assert len(library.unmapped) == len(library.mapped)


def test_vmm_adoption_failure_frees_the_prepared_tables_and_fails_closed(
    monkeypatch,
) -> None:
    """A ledger retraction that fails after prepare leaves the pool unusable."""

    library = FakeSessionVmmLibrary()
    session, allocator = _session(monkeypatch, library=library)
    pool = session.create_global_device_kv_pool(page_capacity=4, generation=1)
    try:
        pool._max_pages = 12
        pool._growth_chunk_pages = 1
        pool.allocate(1, 8)
        pool.release(1)
        real_truncate = pool.global_pool.truncate_pages

        def failing_truncate(pages: int, *, pointer_table_pointers=None) -> int:
            raise RuntimeError("ledger truncate failed")

        pool.global_pool.truncate_pages = failing_truncate
        try:
            with pytest.raises(RuntimeError, match="ledger truncate failed"):
                pool.shrink_idle(now_seconds=1.0)
        finally:
            pool.global_pool.truncate_pages = real_truncate

        # Prepare unmapped the arena above the floor while the ledger still
        # advertises all eight pages, so admission must refuse.
        assert pool.current_pages == 8
        assert pool.stats.shrink_events == 0
        with pytest.raises(RuntimeError, match="unusable"):
            pool.allocate(2, 1)
        with pytest.raises(RuntimeError, match="unusable"):
            pool.grow(1)
    finally:
        # Release anything a refusal assertion managed to allocate, so a
        # regression reports the refusal rather than a close error.
        for request_id in list(pool._request_allocations):
            pool.release(request_id)
        pool.close()
    # The prepared pointer tables were freed rather than leaked; everything the
    # pool still owns is released by close.
    assert allocator.live == set()
