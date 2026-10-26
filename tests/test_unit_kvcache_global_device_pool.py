from __future__ import annotations

import threading

import pytest

from hipengine.kvcache.device_global import GlobalDeviceKVPool
from hipengine.kvcache.pool import DeviceKVContiguityError, KVStorageFault
from hipengine.runtime.memory_admission import (
    ADMISSION_REFUSAL_CODE,
    MemoryAdmissionRefused,
)


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


@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("max_pages", [4, 8])
def test_private_allocation_grows_a_whole_chunk_or_refuses(
    monkeypatch, fragmented: bool, max_pages: int
) -> None:
    pool, _ = _pool(pages=2)
    pool._max_pages = max_pages
    pool._growth_chunk_pages = 1
    grown = []

    def grow(pages, start):
        grown.append(pages)
        return (
            {role: tuple(0xD000 + (start + i) * 256 for i in range(pages))
             for role in ("layer0.key", "layer0.value")},
            {"layer0.key": 0xE000, "layer0.value": 0xF000},
            {"arena": start},
        )

    pool._grow_storage = grow
    if fragmented:
        pool.grow(2)
    pool.allocate(1, 1)
    grown.clear()
    allocate = pool._allocate_within_one_chunk
    attempts = 0

    def bounded_allocate(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        assert attempts <= 3, "private allocation retried without placement progress"
        return allocate(*args, **kwargs)

    monkeypatch.setattr(pool, "_allocate_within_one_chunk", bounded_allocate)
    if max_pages == 4:
        with pytest.raises(MemoryError):
            pool.allocate(2, 3)
        assert grown == []
        assert set(pool.allocations) == {1}
    else:
        allocation = pool.allocate(2, 3)
        assert grown == [3]
        assert len(allocation.block_ids) == 3
        assert allocation.chunk_start_block_id == (4 if fragmented else 2)
        pool.release(2)
    pool.release(1)
    pool.close()


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
            {"arena": 2},
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
            {"arena": 2},
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

    # The prefix sits in the first storage chunk, which has one page left. A
    # two-page suffix cannot join it there, and growth cannot help: it appends a
    # separate chunk, and every KV kernel addresses one chunk from its own base,
    # so a prefix and suffix split across chunks would index past the first one.
    # The admission declines instead and the caller falls back to a private
    # allocation, which is confined to one chunk as a whole and may use the
    # appended one.
    with pytest.raises(MemoryError):
        pool.admit_with_shared_prefix(2, source.block_ids, suffix_pages=2)
    assert grown == []
    assert pool.stats.prefix_reused_pages == 0

    private = pool.allocate(3, 2)
    assert grown == [2]
    assert private.chunk_start_block_id == 3

    # A suffix that does fit beside its prefix still reuses it.
    reused = pool.admit_with_shared_prefix(4, source.block_ids, suffix_pages=1)
    assert reused.reused_block_ids == source.block_ids
    assert len(reused.allocated_block_ids) == 1
    assert reused.chunk_start_block_id == 0
    assert pool.stats.prefix_reused_pages == 2

    pool.release(3)
    pool.release(4)
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


def test_global_device_pool_shared_admission_continues_the_prefix_run() -> None:
    """A shared suffix continues the prefix instead of taking the lowest free pages."""

    pool, _closed = _pool(pages=8)
    pool.allocate(1, 4)
    source = pool.allocate(2, 2)
    pool.retain_blocks(source.block_ids)
    pool.release(1)

    shared = pool.admit_with_shared_prefix(
        3,
        source.block_ids,
        suffix_pages=2,
        require_contiguous=True,
    )

    assert source.block_ids == (4, 5)
    assert shared.block_ids == (4, 5, 6, 7)
    assert shared.reused_block_ids == (4, 5)
    assert shared.allocated_block_ids == (6, 7)
    pool.release(3)
    pool.release(2)
    pool.release_blocks(source.block_ids)
    pool.close()


def test_global_device_pool_shared_admission_rejects_a_gapped_run_when_required() -> None:
    """A contiguous requirement declines the reuse instead of returning a gapped lease."""

    pool, _closed = _pool(pages=8)
    pool.allocate(1, 2)
    source = pool.allocate(2, 2)
    pool.retain_blocks(source.block_ids)
    pool.release(2)
    pool.allocate(3, 2)

    with pytest.raises(DeviceKVContiguityError, match="contiguous shared-plus-private run"):
        pool.admit_with_shared_prefix(
            4,
            source.block_ids,
            suffix_pages=2,
            require_contiguous=True,
        )

    # A declined admission leaves the pool conserved and the same pages reusable.
    assert pool.stats.free_pages == 2
    assert pool.refcount(source.block_ids[0]) == 1
    shared = pool.admit_with_shared_prefix(4, source.block_ids, suffix_pages=2)
    assert shared.block_ids == (2, 3, 6, 7)
    assert pool.stats.prefix_reuse_events == 1
    pool.release(4)
    pool.release(3)
    pool.release(1)
    pool.release_blocks(source.block_ids)
    pool.close()


def test_private_workspace_shares_budget_with_arena_growth() -> None:
    pool, _ = _pool(pages=2)
    pool._max_pages = 5
    grown = []

    def grow(pages, start):
        grown.append(pages)
        return ({role: tuple(0xD000 + (start + i) * 256 for i in range(pages))
                 for role in ("layer0.key", "layer0.value")},
                {"layer0.key": 0xE000, "layer0.value": 0xF000})

    pool._grow_storage = grow
    token = pool.reserve_private_workspace(2 * pool.page_bytes)
    assert pool.private_workspace_bytes == 256
    assert pool.accounted_bytes == 512
    with pytest.raises(MemoryError, match="budget"):
        pool.grow(2)
    assert grown == []
    pool._growth_chunk_pages = 8
    pool.allocate(1, 2)
    pool.allocate(2, 1)  # automatic growth clamps to the remaining shared budget
    assert grown == [1]
    assert pool.accounted_bytes == pool.budget_bytes
    with pytest.raises(MemoryError, match="budget"):
        pool.reserve_private_workspace(1)
    pool.release(1)
    pool.release(2)
    with pytest.raises(RuntimeError, match="private workspace"):
        pool.close()
    pool.release_private_workspace(token)
    assert pool.private_workspace_bytes == 0
    pool.grow(2)
    pool.close()


def test_private_workspace_counts_allocated_arena_not_only_live_pages() -> None:
    pool, _ = _pool(pages=2)
    pool._max_pages = 2
    assert pool.stats.free_pages == 2
    with pytest.raises(MemoryError, match="budget"):
        pool.reserve_private_workspace(128)
    assert pool.private_workspace_bytes == 0
    pool.close()


# --- one budget over every named resident consumer (task #15) -----------------

#: The consumers the admission budget has to price. The pool's own arena covers
#: the target KV and its scale planes; the rest are charged by name.
_NAMED_CONSUMERS = (
    "target_kv_scales",
    "draft_kv",
    "verifier_scratch",
    "captured_graphs",
    "retained_prefix",
)


def _budgeted_pool(pages: int = 2, max_pages: int = 4):
    pool, closed = _pool(pages=pages)
    pool._max_pages = max_pages
    return pool, closed


def test_resident_consumers_report_arena_first_then_charges_in_order() -> None:
    pool, _ = _budgeted_pool()
    first = pool.reserve_resident_bytes("verifier_scratch", 64)
    second = pool.reserve_resident_bytes("captured_graphs", 32)
    assert [item.name for item in pool.resident_consumers] == [
        "kv_arena",
        "verifier_scratch",
        "captured_graphs",
    ]
    assert pool.resident_consumers[0].bytes == pool.current_pages * pool.page_bytes
    assert pool.accounted_bytes == 256 + 96
    pool.release_private_workspace(first)
    pool.release_private_workspace(second)
    assert [item.name for item in pool.resident_consumers] == ["kv_arena"]
    assert pool.accounted_bytes == 256
    pool.close()


def test_private_workspace_charge_can_be_named() -> None:
    """The KV-payload charge is one named consumer among the others."""

    pool, _ = _budgeted_pool()
    token = pool.reserve_private_workspace(64, "packed_verify_kv")
    assert [item.name for item in pool.resident_consumers] == [
        "kv_arena",
        "packed_verify_kv",
    ]
    assert pool.private_workspace_bytes == 64
    pool.release_private_workspace(token)
    assert pool.private_workspace_bytes == 0
    pool.close()


def test_exact_boundary_charge_is_admitted_and_one_byte_more_is_refused() -> None:
    """A consumer that fits exactly at the boundary succeeds."""

    pool, _ = _budgeted_pool()
    arena = pool.current_pages * pool.page_bytes
    remaining = pool.budget_bytes - arena
    assert remaining == 256
    token = pool.reserve_resident_bytes("verifier_scratch", remaining)
    assert pool.accounted_bytes == pool.budget_bytes
    with pytest.raises(MemoryAdmissionRefused) as caught:
        pool.reserve_resident_bytes("captured_graphs", 1)
    assert caught.value.refused_consumer == "captured_graphs"
    # The refused attempt charged nothing, so the caller allocated nothing.
    assert pool.accounted_bytes == pool.budget_bytes
    pool.release_private_workspace(token)
    pool.close()


@pytest.mark.parametrize("name", _NAMED_CONSUMERS)
def test_each_named_consumer_is_the_refusal_reason_at_its_own_edge(name: str) -> None:
    """Push each consumer to its budget edge and assert the refusal names it.

    An unrelated consumer takes half the headroom first, so the refusal can only
    be attributed to the consumer under test rather than to the arena.
    """

    pool, _ = _budgeted_pool()
    arena = pool.current_pages * pool.page_bytes
    remaining = pool.budget_bytes - arena
    prior = remaining // 2
    filler = pool.reserve_resident_bytes("unrelated_consumer", prior)

    # The admitted side of this consumer's edge.
    edge = pool.reserve_resident_bytes(name, remaining - prior)
    assert pool.accounted_bytes == pool.budget_bytes
    pool.release_private_workspace(edge)

    before = pool.accounted_bytes
    with pytest.raises(MemoryAdmissionRefused) as caught:
        pool.reserve_resident_bytes(name, remaining - prior + 1)
    refusal = caught.value
    assert refusal.refused_consumer == name, (name, refusal.refused_consumer)
    assert name in str(refusal)
    assert refusal.code == ADMISSION_REFUSAL_CODE
    assert isinstance(refusal, MemoryError)
    assert pool.accounted_bytes == before
    pool.release_private_workspace(filler)
    pool.close()


def test_unrelated_charge_value_behaves_the_same_on_both_sides() -> None:
    """A byte count unrelated to any threshold is priced like any other."""

    pool, _ = _budgeted_pool()
    arena = pool.current_pages * pool.page_bytes
    remaining = pool.budget_bytes - arena
    odd = 77
    filler = pool.reserve_resident_bytes("captured_graphs", remaining - odd)
    edge = pool.reserve_resident_bytes("verifier_scratch", odd)
    assert pool.accounted_bytes == pool.budget_bytes
    pool.release_private_workspace(edge)
    with pytest.raises(MemoryAdmissionRefused) as caught:
        pool.reserve_resident_bytes("verifier_scratch", odd + 1)
    assert caught.value.refused_consumer == "verifier_scratch"
    pool.release_private_workspace(filler)
    pool.close()


def test_resident_charge_requires_a_name_and_positive_bytes() -> None:
    pool, _ = _budgeted_pool()
    with pytest.raises(ValueError):
        pool.reserve_resident_bytes("   ", 8)
    with pytest.raises(ValueError):
        pool.reserve_resident_bytes("verifier_scratch", 0)
    assert pool.private_workspace_bytes == 0
    pool.close()


def _growth_provider(grown: list[int]):
    def grow(pages, start):
        grown.append(pages)
        return (
            {
                role: tuple(0xD000 + (start + i) * 256 for i in range(pages))
                for role in ("layer0.key", "layer0.value")
            },
            {"layer0.key": 0xE000, "layer0.value": 0xF000},
            {"arena": start},
        )

    return grow


def test_growth_refuses_padded_physical_before_calling_the_provider() -> None:
    """A granularity-padded growth is refused before any provider call.

    ``budget_bytes`` is the logical page budget, but the prospective growth is
    priced with the padded physical callback, so the boundary is respected
    before a single byte is mapped.
    """

    pool, _ = _pool(pages=2)
    pool.page_bytes = 1024
    pool._max_pages = 4  # budget_bytes = 4096
    pool._growth_chunk_pages = 1
    # Physical commitment is twice the logical payload.
    pool._physical_bytes_for_pages = lambda pages: pages * 2048
    grown: list[int] = []
    pool._grow_storage = _growth_provider(grown)

    # Current padded commitment is 4096; one more page adds 2048 and exceeds
    # the 4096-byte budget.
    with pytest.raises(MemoryAdmissionRefused):
        pool.grow(1)
    assert grown == []
    pool.close()


def test_growth_without_padding_keeps_the_logical_boundary() -> None:
    """A plain hipMalloc provider keeps its logical page budget."""

    pool, _ = _pool(pages=2)
    pool.page_bytes = 1024
    pool._max_pages = 4  # budget_bytes = 4096
    pool._growth_chunk_pages = 1
    grown: list[int] = []
    pool._grow_storage = _growth_provider(grown)

    # Logical 2 * 1024 + 1 * 1024 = 3072 <= 4096, so growth proceeds.
    pool.grow(1)
    assert grown == [1]
    pool.close()


def test_ensure_free_pages_caps_at_the_padded_physical_budget() -> None:
    """The contiguous-growth target is capped by padded physical pricing."""

    pool, _ = _pool(pages=2)
    pool.page_bytes = 1024
    pool._max_pages = 8  # budget_bytes = 8192
    pool._growth_chunk_pages = 1
    pool._contiguous_growth = True
    # Three logical pages fit (3072 <= 8192); four padded pages do not
    # (4 * 2048 = 8192 exactly fits, 5 * 2048 = 10240 does not).
    pool._physical_bytes_for_pages = lambda pages: pages * 2048
    grown: list[int] = []
    pool._grow_storage = _growth_provider(grown)

    pool._ensure_free_pages(8)
    # Uncapped the target would be 2 + 6 = 8 pages; padded physical pricing caps
    # it at four (4 * 2048 = 8192 == budget, 5 * 2048 > budget).
    assert grown == [2]
    assert pool.global_pool.page_capacity == 4
    pool.close()


# ---------------------------------------------------------------------------
# Idle reclamation of grown capacity.
#
# The pool retracts only a whole growth unit (one appended chunk, or one
# committed arena segment) whose pages are all free in the ledger. Release is
# two-phase: the provider publishes the retained pointer tables and returns a
# handle without freeing anything, the pool retracts its ledger, and only then
# does the provider free what the pool no longer owns. These tests drive that
# policy with a host-side provider so the guards are checkable without a GPU.
# ---------------------------------------------------------------------------


class _FakeStorageProvider:
    """Host-side stand-in for the runtime storage owner behind a global pool.

    ``published`` is the capacity the provider advertises (plane pointers and
    pointer tables); ``held`` is the physical capacity it still owns. Prepare
    moves neither, so a test can show that the ledger was retracted before
    anything was freed, and a failed free leaves ``held`` above ``published``,
    which is exactly the leak the pool reports rather than hides.
    """

    def __init__(self, pages: int) -> None:
        self.published = int(pages)
        self.held = int(pages)
        self.grown: list[int] = []
        self.prepared: list[int] = []
        self.shrunk: list[int] = []
        self.failure: BaseException | None = None
        self.release_failure: BaseException | None = None
        self.table_serial = 0

    def grow(self, pages: int, start: int):
        assert start == self.published
        self.published += int(pages)
        self.held += int(pages)
        self.grown.append(int(pages))
        self.table_serial += 1
        return (
            {
                role: tuple(
                    0xD000 + (start + index) * 0x100 for index in range(int(pages))
                )
                for role in ("layer0.key", "layer0.value")
            },
            {
                "layer0.key": 0xE000 + self.table_serial,
                "layer0.value": 0xF000 + self.table_serial,
            },
            {"arena": self.published},
        )

    def shrink(self, target_pages: int):
        """Prepare: publish retained tables for ``target_pages``, free nothing."""

        if self.failure is not None:
            raise self.failure
        assert target_pages < self.published
        self.prepared.append(int(target_pages))
        self.table_serial += 1
        return (
            {
                "layer0.key": 0xE000 + self.table_serial,
                "layer0.value": 0xF000 + self.table_serial,
            },
            {"arena": int(target_pages)},
            int(target_pages),
        )

    def release(self, handle) -> None:
        """Release: adopt the retained publication, then free the rest."""

        self.published = int(handle)
        if self.release_failure is not None:
            # Mirrors a device free failing after publication: the pool and the
            # published capacity agree, and only the physical commitment leaks.
            raise self.release_failure
        self.held = int(handle)
        self.shrunk.append(int(handle))


def _shrinkable_pool(
    *,
    pages: int = 2,
    growth: int = 2,
    max_pages: int | None = 8,
    contiguous_growth: bool = False,
    shrink: bool = True,
    abort=None,
):
    provider = _FakeStorageProvider(pages)
    invalidated: list[int] = []
    pool = GlobalDeviceKVPool(
        page_bytes=128,
        backend_fingerprint="artifact:gguf-test",
        generation=3,
        backing={"arena": pages},
        plane_page_pointers={
            "layer0.key": tuple(0x1000 + page * 0x100 for page in range(pages)),
            "layer0.value": tuple(0x4000 + page * 0x100 for page in range(pages)),
        },
        pointer_table_pointers={"layer0.key": 0x8000, "layer0.value": 0x9000},
        metadata_descriptor_pointer=0xA000,
        close_storage=lambda: None,
        grow_storage=provider.grow,
        shrink_storage=provider.shrink if shrink else None,
        release_storage=provider.release if shrink else None,
        abort_shrink=abort,
        before_shrink=lambda: invalidated.append(pool.current_pages),
        max_pages=max_pages,
        growth_chunk_pages=growth,
        contiguous_growth=contiguous_growth,
    )
    return pool, provider, invalidated


def test_shrink_idle_releases_a_free_grown_tail_chunk() -> None:
    pool, provider, invalidated = _shrinkable_pool()
    allocation = pool.allocate(1, 3)
    assert provider.grown == [3]
    assert pool.current_pages == 5
    # The allocation landed in the appended chunk, which is the unit the shrink
    # may retract once it is free.
    assert allocation.chunk_start_block_id == 2
    pool.release(1)
    generation = pool.global_pool.generation

    released = pool.shrink_idle(now_seconds=1.0)

    assert released == 3
    assert provider.prepared == [2]
    assert provider.shrunk == [2]
    assert provider.published == 2
    assert provider.held == 2
    assert pool.current_pages == 2
    assert tuple(pool.chunks[0].block_ids) == (0, 1)
    assert pool._chunks == [(0, 2, {"arena": 2})]
    assert pool._release_points == [2]
    assert pool.stats.shrink_events == 1
    assert pool.stats.retired_pages == 3
    assert pool.stats.retired_bytes == 3 * pool.page_bytes
    # Graphs are invalidated before the provider publishes the retained tables.
    assert invalidated == [5]
    # The published view covers only the retained pages, at the new table.
    view = pool.storage_view()
    assert view.planes[0].shape == (2,)
    assert view.planes[0].ptr == 0xE002
    assert view.generation == generation + 1
    pool.global_pool.assert_conserved()
    pool.close()


def test_shrink_idle_keeps_capacity_a_live_lease_still_uses() -> None:
    pool, provider, _invalidated = _shrinkable_pool()
    pool.allocate(1, 2)
    live = pool.allocate(2, 2)
    assert live.chunk_start_block_id == 2
    assert pool.current_pages == 4

    assert pool.shrink_idle(now_seconds=1.0) == 0
    assert provider.prepared == []

    pool.release(2)
    assert pool.shrink_idle(now_seconds=1.0) == 2
    assert provider.shrunk == [2]
    assert pool.current_pages == 2
    pool.release(1)
    pool.close()


def test_shrink_idle_keeps_a_pinned_or_cache_owned_tail() -> None:
    pool, provider, _invalidated = _shrinkable_pool()
    pool.allocate(1, 2)
    tail = pool.allocate(2, 2)
    pool.pin(tail.block_ids)
    pool.retain_blocks(tail.block_ids)
    pool.release(2)

    # A graph pin and a cache-owned prefix both keep the pages, so the chunk
    # that holds them cannot be retracted.
    assert pool.shrink_idle(now_seconds=1.0) == 0
    pool.unpin(tail.block_ids)
    assert pool.shrink_idle(now_seconds=1.0) == 0
    assert provider.prepared == []

    pool.release_blocks(tail.block_ids)
    assert pool.shrink_idle(now_seconds=1.0) == 2
    assert provider.shrunk == [2]
    pool.release(1)
    pool.close()


def test_shrink_idle_waits_for_an_in_flight_graph_epoch() -> None:
    pool, provider, _invalidated = _shrinkable_pool()
    pool.allocate(1, 2)
    tail = pool.allocate(2, 2)
    pool.global_pool.mark_in_flight("request:2", tail.block_ids, epoch=7)
    pool.release(2)

    # A replaying graph still reads those pages, so the ledger keeps them and the
    # chunk stays allocated.
    assert pool.shrink_idle(now_seconds=1.0) == 0
    assert provider.prepared == []

    pool.global_pool.retire_epoch(7)
    assert pool.shrink_idle(now_seconds=1.0) == 2
    assert provider.shrunk == [2]
    pool.release(1)
    pool.close()


def test_shrink_idle_honours_the_idle_grace_period() -> None:
    pool, provider, _invalidated = _shrinkable_pool()
    pool.idle_grace_seconds = 30.0
    pool.allocate(1, 3, now_seconds=100.0)
    pool.release(1, now_seconds=110.0)

    assert pool.shrink_idle(now_seconds=139.0) == 0
    assert provider.prepared == []
    assert pool.shrink_idle(now_seconds=140.0) == 3
    assert provider.shrunk == [2]
    pool.close()


def test_shrink_idle_without_a_release_provider_keeps_grown_capacity() -> None:
    pool, provider, _invalidated = _shrinkable_pool(shrink=False)
    pool.allocate(1, 3)
    pool.release(1)

    # No provider means no way to retract physical backing, so the grown chunk
    # stays allocated rather than the ledger forgetting it.
    assert pool.shrink_idle(now_seconds=1.0) == 0
    assert pool.current_pages == 5
    assert provider.shrunk == []
    pool.close()


def test_shrink_idle_prepare_failure_leaves_the_pool_unchanged() -> None:
    pool, provider, _invalidated = _shrinkable_pool()
    pool.allocate(1, 3)
    pool.release(1)
    generation = pool.global_pool.generation
    release_points = list(pool._release_points)
    chunks = list(pool._chunks)
    provider.failure = RuntimeError("unmap failed")

    with pytest.raises(RuntimeError, match="unmap failed"):
        pool.shrink_idle(now_seconds=1.0)

    assert pool.current_pages == 5
    assert provider.published == 5
    assert provider.held == 5
    assert pool.global_pool.generation == generation
    assert pool._release_points == release_points
    assert pool._chunks == chunks
    assert pool.stats.shrink_events == 0

    provider.failure = None
    assert pool.shrink_idle(now_seconds=1.0) == 3
    pool.close()


def test_shrink_idle_release_failure_keeps_the_retracted_ledger() -> None:
    """A free failure after retraction leaks storage; it cannot corrupt the pool."""

    pool, provider, _invalidated = _shrinkable_pool()
    pool.allocate(1, 3)
    pool.release(1)
    provider.release_failure = RuntimeError("hipFree failed")

    with pytest.raises(RuntimeError, match="hipFree failed"):
        pool.shrink_idle(now_seconds=1.0)

    # The ledger already dropped the pages and publishes the retained tables, so
    # nothing points at the storage the provider failed to free. The provider's
    # physical commitment stays above its published capacity: a reported leak.
    assert provider.prepared == [2]
    assert provider.published == 2
    assert provider.held == 5
    assert pool.current_pages == 2
    assert pool.stats.shrink_events == 1
    assert pool.storage_view().planes[0].shape == (2,)
    assert pool._chunks == [(0, 2, {"arena": 2})]
    pool.global_pool.assert_conserved()
    # The pool still serves the retained capacity, and a later growth works.
    assert pool.allocate(2, 2).block_ids == (0, 1)
    pool.release(2)
    pool.grow(2)
    assert pool.current_pages == 4
    pool.close()


def test_shrink_idle_fails_closed_when_the_provider_cannot_restore() -> None:
    pool, provider, _invalidated = _shrinkable_pool()
    pool.allocate(1, 3)
    pool.release(1)

    def unrestorable(target: int):
        raise KVStorageFault("arena restore failed")

    pool._shrink_storage = unrestorable
    with pytest.raises(KVStorageFault, match="arena restore failed"):
        pool.shrink_idle(now_seconds=1.0)

    # The ledger still advertises pages the provider could not keep backed, so
    # admission must fail closed instead of handing them out.
    assert pool.current_pages == 5
    assert provider.prepared == []
    with pytest.raises(RuntimeError, match="unusable"):
        pool.allocate(2, 1)
    with pytest.raises(RuntimeError, match="unusable"):
        pool.grow(1)
    with pytest.raises(RuntimeError, match="unusable"):
        pool.lease_workspace("packed", 1)
    # Draining and closing still work.
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.allocate(3, 1)


def test_shrink_idle_holds_the_ledger_lock_against_direct_ledger_access() -> None:
    """A direct ledger caller cannot claim a page inside the retracted range.

    ``GlobalDeviceKVPool.global_pool`` is a public ledger, so the pool cannot
    assume every caller goes through its own lock. The retraction therefore holds
    the ledger lock from the free-tail test until the truncation, which is what
    this test pins down.
    """

    pool, provider, _invalidated = _shrinkable_pool()
    pool.allocate(1, 3)
    pool.release(1)
    attempting = threading.Event()
    intruder_done = threading.Event()
    intruder: list[object] = []

    def direct_allocate() -> None:
        attempting.set()
        try:
            intruder.append(
                pool.global_pool.allocate(
                    "intruder", private_pages=2, growth_credit_pages=0
                )
            )
        finally:
            intruder_done.set()

    worker = threading.Thread(target=direct_allocate)

    def prepare(target: int):
        # Start the intruder once the shrink holds the ledger lock, so its
        # allocation attempt cannot land before the free-tail test runs.
        worker.start()
        assert attempting.wait(5.0)
        # The intruder is inside the ledger's allocate; it must be blocked on the
        # lock the shrink holds, not served a page that is about to be dropped.
        assert intruder_done.wait(0.2) is False
        return provider.shrink(target)

    pool._shrink_storage = prepare
    try:
        assert pool.shrink_idle(now_seconds=1.0) == 3
    finally:
        if worker.ident is not None:
            worker.join(5.0)
    assert not worker.is_alive()
    assert pool.current_pages == 2
    # The direct allocation was served after the retraction, from the retained
    # capacity only, and the pool is consistent.
    assert len(intruder) == 1
    assert intruder[0].private_page_ids == (0, 1)
    pool.global_pool.release("intruder")
    pool.global_pool.assert_conserved()
    pool.close()


def test_shrink_idle_retracts_a_merged_arena_and_grows_again() -> None:
    pool, provider, _invalidated = _shrinkable_pool(contiguous_growth=True)
    allocation = pool.allocate(1, 3)
    assert pool.current_pages == 4
    assert allocation.chunk_start_block_id == 0
    pool.release(1)

    assert pool.shrink_idle(now_seconds=1.0) == 2
    assert pool.backing == {"arena": 2}
    assert pool._chunks == [(0, 2, {"arena": 2})]
    assert pool._release_points == [2]

    # A later request grows again from the retained floor instead of reusing a
    # stale release point.
    again = pool.allocate(2, 3)
    assert pool.current_pages == 4
    assert pool._release_points == [2, 4]
    assert again.chunk_start_block_id == 0
    pool.release(2)
    assert pool.shrink_idle(now_seconds=2.0) == 2
    assert provider.shrunk == [2, 2]
    pool.close()


def test_shrink_idle_stops_at_the_load_time_floor() -> None:
    pool, provider, _invalidated = _shrinkable_pool()
    pool.allocate(1, 3)
    pool.release(1)
    assert pool.shrink_idle(now_seconds=1.0) == 3
    assert pool.shrink_idle(now_seconds=2.0) == 0
    assert pool.current_pages == 2
    assert pool.low_water_pages == 2
    assert provider.shrunk == [2]
    pool.close()


def test_shrink_idle_adoption_failure_fails_closed_and_aborts_the_prepare() -> None:
    """A retraction the pool cannot adopt must not leave the pool serving pages."""

    aborted: list[object] = []
    pool, provider, _invalidated = _shrinkable_pool(
        abort=lambda handle: aborted.append(handle)
    )
    pool.allocate(1, 3)
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

    # Prepare already published the retained tables and, for a VMM provider,
    # unmapped the arena above the target. The ledger still advertises all five
    # pages, so admission must refuse rather than serve unbacked pages.
    assert len(aborted) == 1
    assert pool.current_pages == 5
    assert pool.stats.shrink_events == 0
    assert pool.stats.retired_pages == 0
    with pytest.raises(RuntimeError, match="unusable"):
        pool.allocate(2, 1)
    with pytest.raises(RuntimeError, match="unusable"):
        pool.grow(1)
    with pytest.raises(RuntimeError, match="unusable"):
        pool.lease_workspace("packed", 1)
    # The provider never reached its release phase, so nothing was freed.
    assert provider.prepared == [2]
    assert provider.shrunk == []
    assert provider.published == 5
    # Release anything a refusal assertion managed to allocate, so a regression
    # reports the refusal rather than a close error.
    for request_id in list(pool._request_allocations):
        pool.release(request_id)
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.allocate(3, 1)
