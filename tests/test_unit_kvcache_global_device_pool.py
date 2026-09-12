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
