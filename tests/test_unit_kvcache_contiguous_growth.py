"""Stable-address providers can extend a shared prefix across growth events."""
from tests.test_unit_kvcache_global_device_pool import _pool


def test_stable_address_growth_extends_shared_prefix():
    pool, closed = _pool(pages=2)
    pool._contiguous_growth = True
    pool._max_pages = 8
    pool._growth_chunk_pages = 1
    root = {"arena": "stable"}

    def grow(count, start):
        return (
            {role: tuple(base + (start + i) * 256 for i in range(count))
             for role, base in (("layer0.key", 0x1000), ("layer0.value", 0x4000))},
            {"layer0.key": 0x8000, "layer0.value": 0x9000},
            root,
        )

    pool._grow_storage = grow
    first = pool.allocate(1, 2)
    second = pool.admit_with_shared_prefix(2, first.block_ids, suffix_pages=3)
    assert second.block_ids[:2] == first.block_ids
    assert len(second.block_ids) == 5
    assert second.chunk_start_block_id == 0
    assert second.pool_page_capacity == 5
    assert pool.backing is root
    assert len(pool.chunks) == 1
    third = pool.admit_with_shared_prefix(3, second.block_ids, suffix_pages=2)
    assert third.block_ids[:5] == second.block_ids
    assert third.pool_page_capacity == 7
    for rid in (3, 2, 1):
        pool.release(rid)
    pool.global_pool.assert_conserved()
    pool.close()
    assert closed == [True]
