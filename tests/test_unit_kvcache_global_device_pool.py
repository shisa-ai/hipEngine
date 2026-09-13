from __future__ import annotations

import pytest

from hipengine.kvcache.device_global import GlobalDeviceKVPool


def _pool(*, pages: int = 6):
    closed: list[bool] = []
    pool = GlobalDeviceKVPool(
        page_bytes=128,
        backend_fingerprint="artifact:gguf-test",
        generation=3,
        backing={"arena": 1},
        plane_page_pointers={
            "layer0.key": tuple(0x1000 + page * 0x100 for page in range(pages)),
            "layer0.value": tuple(0x4000 + page * 0x100 for page in range(pages)),
        },
        pointer_table_pointers={
            "layer0.key": 0x8000,
            "layer0.value": 0x9000,
        },
        metadata_descriptor_pointer=0xA000,
        close_storage=lambda: closed.append(True),
    )
    return pool, closed


def test_global_device_pool_allocates_arbitrary_free_pages_without_chunks() -> None:
    pool, closed = _pool()
    first = pool.allocate(1, 2)
    second = pool.allocate(2, 2)
    pool.release(1)

    fragmented = pool.allocate(3, 3)

    assert first.block_ids == (0, 1)
    assert second.block_ids == (2, 3)
    assert fragmented.block_ids == (0, 1, 4)
    assert fragmented.chunk_start_block_id == 0
    assert fragmented.backing == {"arena": 1}
    assert fragmented.pointers == (0x1000, 0x1100, 0x1400)
    assert tuple(pool.chunks[0].block_ids) == tuple(range(6))
    assert pool.storage_view().layout_key == "global-arbitrary-pages:g3"
    assert pool.generation2_compatible is True
    assert pool.shrink_idle(now_seconds=10_000.0) == 0

    pool.release(2)
    pool.release(3)
    pool.close()
    pool.close()
    assert closed == [True]


def test_global_device_pool_preserves_cache_and_pin_ownership() -> None:
    pool, _closed = _pool()
    source = pool.allocate(10, 2)
    pool.retain_blocks((source.block_ids[0],))
    pool.pin(source.block_ids)
    pool.release(10)

    assert pool.refcount(source.block_ids[0]) == 1
    assert pool.pin_count(source.block_ids[0]) == 1
    assert pool.stats.pinned_pages == 2
    with pytest.raises(RuntimeError, match="retained or pinned"):
        pool.close()

    pool.unpin(source.block_ids)
    shared = pool.admit_with_shared_prefix(
        11,
        (source.block_ids[0],),
        suffix_pages=1,
    )
    assert shared.reused_block_ids == (source.block_ids[0],)
    assert len(shared.allocated_block_ids) == 1
    assert pool.stats.prefix_reuse_events == 1
    assert pool.stats.prefix_reused_pages == 1

    pool.release(11)
    pool.release_blocks((source.block_ids[0],))
    assert pool.stats.free_pages == pool.current_pages
    pool.close()


def test_global_device_pool_rejects_capacity_and_live_close() -> None:
    pool, _closed = _pool(pages=2)
    pool.allocate(1, 2)
    with pytest.raises(MemoryError, match="cannot allocate"):
        pool.allocate(2, 1)
    assert pool.stats.grow_failures == 1
    with pytest.raises(RuntimeError, match="live request"):
        pool.close()
    pool.release(1)
    pool.close()


def test_global_device_pool_rejects_budget_below_initial_capacity() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        GlobalDeviceKVPool(
            page_bytes=128,
            backend_fingerprint="artifact:gguf-test",
            generation=3,
            backing={"arena": 1},
            plane_page_pointers={
                "layer0.key": (0x1000, 0x1100),
                "layer0.value": (0x4000, 0x4100),
            },
            pointer_table_pointers={"layer0.key": 0x8000, "layer0.value": 0x9000},
            metadata_descriptor_pointer=0xA000,
            close_storage=lambda: None,
            max_pages=1,
        )


def test_global_device_pool_grows_on_pressure_with_graph_invalidation() -> None:
    invalidated: list[bool] = []

    def grow_storage(pages: int, start: int):
        assert (pages, start) == (2, 2)
        return (
            {
                "layer0.key": (0x1200, 0x1300),
                "layer0.value": (0x4200, 0x4300),
            },
            {"layer0.key": 0xA000, "layer0.value": 0xB000},
        )

    pool = GlobalDeviceKVPool(
        page_bytes=128,
        backend_fingerprint="artifact:gguf-test",
        generation=3,
        backing={"arena": 1},
        plane_page_pointers={
            "layer0.key": (0x1000, 0x1100),
            "layer0.value": (0x4000, 0x4100),
        },
        pointer_table_pointers={"layer0.key": 0x8000, "layer0.value": 0x9000},
        metadata_descriptor_pointer=0xA000,
        close_storage=lambda: None,
        grow_storage=grow_storage,
        before_grow=lambda: invalidated.append(True),
        max_pages=4,
        growth_chunk_pages=2,
    )

    first = pool.allocate(1, 2)
    second = pool.allocate(2, 1)
    assert first.block_ids == (0, 1)
    assert second.block_ids == (2,)
    assert pool.current_pages == 4
    assert invalidated == [True]
    assert pool.stats.grow_events == 1
    pool.release(1)
    pool.release(2)
    pool.close()


def test_shared_prefix_admission_grows_only_for_private_suffix() -> None:
    grown: list[int] = []

    def grow_storage(pages: int, start: int):
        grown.append(pages)
        return (
            {
                "layer0.key": tuple(0x1200 + index * 0x100 for index in range(pages)),
                "layer0.value": tuple(0x4200 + index * 0x100 for index in range(pages)),
            },
            {"layer0.key": 0xA000, "layer0.value": 0xB000},
        )

    pool = GlobalDeviceKVPool(
        page_bytes=128,
        backend_fingerprint="artifact:gguf-test",
        generation=3,
        backing={"arena": 1},
        plane_page_pointers={
            "layer0.key": (0x1000, 0x1100, 0x1200),
            "layer0.value": (0x4000, 0x4100, 0x4200),
        },
        pointer_table_pointers={"layer0.key": 0x8000, "layer0.value": 0x9000},
        metadata_descriptor_pointer=0xA000,
        close_storage=lambda: None,
        grow_storage=grow_storage,
        max_pages=5,
        growth_chunk_pages=2,
    )
    source = pool.allocate(1, 2)
    pool.retain_blocks(source.block_ids)
    pool.release(1)

    reused = pool.admit_with_shared_prefix(2, source.block_ids, suffix_pages=2)
    assert reused.reused_block_ids == source.block_ids
    assert len(reused.allocated_block_ids) == 2
    assert grown == [2]
    assert pool.stats.prefix_reused_pages == 2

    pool.release(2)
    pool.release_blocks(source.block_ids)
    pool.close()


def test_global_device_pool_pressure_can_reclaim_cached_pages_without_growth() -> None:
    pressure_calls: list[int] = []
    pool, _closed = _pool(pages=3)
    source = pool.allocate(1, 1)
    pool.retain_blocks(source.block_ids)
    pool.release(1)

    def reclaim(required: int) -> None:
        pressure_calls.append(required)
        pool.release_blocks(source.block_ids)

    pool._on_pressure = reclaim
    pool.allocate(2, 3)

    assert pressure_calls == [1]
    assert pool.stats.grow_events == 0
    assert pool.current_pages == 3
    assert pool.stats.free_pages == 0
    pool.release(2)
    pool.close()


def test_global_device_pool_workspace_lease_is_pinned_accounted_and_close_guarded() -> None:
    pool, _closed = _pool(pages=6)

    pages = pool.lease_workspace("packed-ar", 2)
    assert len(pages) == 2
    assert len(set(pages)) == 2
    assert pool.workspace_pages("packed-ar") == pages
    stats = pool.stats
    assert stats.free_pages == 4
    assert stats.pinned_pages == 2
    assert stats.refcounted_pages == 2

    # Request leases cannot claim workspace pages.
    allocation = pool.allocate(1, 4)
    assert set(allocation.block_ids).isdisjoint(pages)
    pool.release(1)

    with pytest.raises(ValueError, match="already exists"):
        pool.lease_workspace("packed-ar", 1)
    with pytest.raises(RuntimeError, match="retained or pinned"):
        pool.close()

    released = pool.release_workspace("packed-ar")
    assert released == pages
    assert pool.workspace_pages("packed-ar") is None
    assert pool.stats.free_pages == 6
    assert pool.stats.pinned_pages == 0
    pool.close()


def test_global_device_pool_workspace_lease_exhaustion_and_missing_release() -> None:
    pool, _closed = _pool(pages=3)
    pool.lease_workspace("packed-ar", 2)
    with pytest.raises(MemoryError):
        pool.lease_workspace("second-workspace", 2)
    assert pool.stats.grow_failures == 1
    with pytest.raises(KeyError, match="second-workspace"):
        pool.release_workspace("second-workspace")
    pool.release_workspace("packed-ar")
    pool.close()
