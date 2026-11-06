"""The provider prefix-checkpoint store: keying, validation, and release.

The store answers the design's open question -- where a portable checkpoint
lives and what ends its life. These tests pin the three properties that make it
safe to hand a row device state: a reused block id cannot match, every dropped
payload is released exactly once, and capacity is a bound rather than a
suggestion.
"""

from __future__ import annotations

import pytest

from hipengine.speculative.prefix_checkpoint import PrefixCheckpointStore


class _Recorder:
    """A release callback that records payloads and rejects double releases."""

    def __init__(self) -> None:
        self.released: list[object] = []
        self.double: list[object] = []

    def __call__(self, payload: object) -> None:
        if payload in self.released:
            self.double.append(payload)
        self.released.append(payload)


def _store(capacity: int = 4) -> tuple[PrefixCheckpointStore, _Recorder]:
    recorder = _Recorder()
    return PrefixCheckpointStore(capacity=capacity, release=recorder), recorder


def test_device_trim_counts_shared_owners_and_unpaired_payloads():
    from types import SimpleNamespace
    store, recorder = _store()
    first = SimpleNamespace(nbytes=1, device_allocations=((4096, 100),))
    second = SimpleNamespace(nbytes=1, device_allocations=((4096, 100), (8192, 40)))
    store.put((1,), (10,), first)
    store.put((2,), (11,), second)
    assert store.owning_device_nbytes == 140
    assert store.trim_device_bytes(100) == 2
    assert recorder.released == [first, second]
    assert store.owning_device_nbytes == 0
    with pytest.raises(ValueError):
        store.trim_device_bytes(-1)


def test_host_byte_trim_evicts_lru_without_changing_device_semantics():
    from types import SimpleNamespace
    store, recorder = _store()
    first = SimpleNamespace(nbytes=64, host_nbytes=1000)
    second = SimpleNamespace(nbytes=32, host_nbytes=500)
    store.put((1,), (10,), first)
    store.put((2,), (11,), second)
    assert store.trim_host_bytes(600) == 1
    assert recorder.released == [first]
    assert store.host_nbytes == 500
    assert store.nbytes == 32
    assert store.get((2,), (11,)) is second
    assert store.trim_host_bytes(0) == 1
    assert len(store) == 0
    with pytest.raises(ValueError):
        store.trim_host_bytes(-1)


def test_host_and_device_byte_totals_remain_distinct():
    from types import SimpleNamespace
    store, _ = _store()
    store.put((1,), (10,), SimpleNamespace(nbytes=64, host_nbytes=1000))
    assert store.nbytes == 64
    assert store.host_nbytes == 1000
    store.drop((1,))
    assert store.host_nbytes == 0


def test_allocation_inventory_does_not_touch_lru_or_hit_counters() -> None:
    from types import SimpleNamespace
    store, _ = _store()
    store.put((1,), (10,), SimpleNamespace(device_allocations=((4096, 8192),)))
    store.put((2,), (11,), SimpleNamespace(device_allocations=((12288, 64),)))
    keys, stats = store.token_keys(), store.stats()
    assert store.device_allocations_for((1,)) == ((4096, 8192),)
    assert store.device_allocations_for((3,)) == ()
    assert store.token_keys() == keys
    assert store.stats() == stats
    store.drop((1,))
    assert store.device_allocations_for((1,)) == ()


def test_clear_drains_later_payloads_after_release_failure() -> None:
    released = []
    def release(payload):
        released.append(payload)
        if payload == "first":
            raise RuntimeError("release failed")
    store = PrefixCheckpointStore(capacity=2, release=release)
    store.put((1,), (10,), "first")
    store.put((2,), (11,), "second")
    with pytest.raises(RuntimeError, match="release failed"):
        store.clear()
    assert released == ["first", "second"]
    assert len(store) == 0
    assert store.clear() == 0


def test_a_stored_prefix_round_trips_by_tokens_and_block_ids() -> None:
    store, recorder = _store()
    blob = object()

    store.put((1, 2, 3), (10, 11), blob)

    assert store.get((1, 2, 3), (10, 11)) is blob
    assert len(store) == 1
    assert recorder.released == []
    stats = store.stats()
    assert (stats.stored, stats.hits, stats.misses) == (1, 1, 0)


def test_a_reused_block_id_is_a_miss_rather_than_a_wrong_checkpoint() -> None:
    """The whole reason the key is the tokens and the ids are a check.

    A block id is not stable content: the allocator hands a freed id to
    different tokens. If the ids were the key, the second prefix below would
    find the first prefix's checkpoint and prime a row with state that describes
    tokens it never ran.
    """

    store, recorder = _store()
    first = object()
    store.put((1, 2, 3), (10, 11), first)

    # Same ids, different tokens: no entry, and nothing is released by a miss.
    assert store.get((4, 5, 6), (10, 11)) is None

    # Same tokens, different ids: the blocks no longer hold this prefix.
    assert store.get((1, 2, 3), (12, 13)) is None
    assert recorder.released == [first]
    assert len(store) == 0
    stats = store.stats()
    assert stats.block_id_mismatches == 1
    assert stats.misses == 2


def test_a_mismatching_entry_is_dropped_because_its_id_is_gone() -> None:
    store, recorder = _store()
    blob = object()
    store.put((1, 2), (7,), blob)

    store.get((1, 2), (8,))
    store.get((1, 2), (7,))

    # The entry could never be validated again, so the second lookup misses too
    # and the payload was released exactly once.
    assert recorder.released == [blob]
    assert store.stats().misses == 2


def test_replacing_a_prefix_releases_the_previous_payload_once() -> None:
    store, recorder = _store()
    first, second = object(), object()

    store.put((1, 2), (7,), first)
    store.put((1, 2), (8,), second)

    assert recorder.released == [first]
    assert store.get((1, 2), (8,)) is second
    assert len(store) == 1
    assert store.stats().replaced == 1


def test_capacity_evicts_the_least_recently_used_entry_and_releases_it() -> None:
    store, recorder = _store(capacity=2)
    first, second, third = object(), object(), object()

    store.put((1,), (1,), first)
    store.put((2,), (2,), second)
    # Reading the first entry makes the second the least recently used one.
    assert store.get((1,), (1,)) is first
    store.put((3,), (3,), third)

    assert recorder.released == [second]
    assert store.get((2,), (2,)) is None
    assert store.get((1,), (1,)) is first
    assert store.get((3,), (3,)) is third
    assert store.stats().evicted == 1
    assert store.token_keys() == ((3,), (1,))


def test_a_zero_capacity_store_releases_every_payload_immediately() -> None:
    store, recorder = _store(capacity=0)
    blob = object()

    store.put((1, 2), (7,), blob)

    assert recorder.released == [blob]
    assert len(store) == 0
    assert store.get((1, 2), (7,)) is None


def test_drop_and_clear_release_each_payload_exactly_once() -> None:
    store, recorder = _store()
    first, second, third = object(), object(), object()
    store.put((1,), (1,), first)
    store.put((2,), (2,), second)
    store.put((3,), (3,), third)

    assert store.drop((1,)) is True
    assert store.drop((1,)) is False
    assert store.clear() == 2
    assert store.clear() == 0

    assert sorted(recorder.released, key=id) == sorted(
        [first, second, third], key=id
    )
    assert recorder.double == []


def test_drop_missing_keeps_only_the_named_prefixes() -> None:
    store, recorder = _store()
    keep, drop = object(), object()
    store.put((1,), (1,), keep)
    store.put((2,), (2,), drop)

    assert store.drop_missing([(1,)]) == 1

    assert recorder.released == [drop]
    assert store.get((1,), (1,)) is keep
    assert len(store) == 1


def test_an_empty_prefix_or_block_list_is_refused() -> None:
    store, _recorder = _store()

    with pytest.raises(ValueError):
        store.put((), (1,), object())
    with pytest.raises(ValueError):
        store.put((1,), (), object())


def test_a_negative_capacity_or_missing_release_is_refused() -> None:
    with pytest.raises(ValueError):
        PrefixCheckpointStore(capacity=-1, release=lambda payload: None)
    with pytest.raises(ValueError):
        PrefixCheckpointStore(capacity=1, release=None)  # type: ignore[arg-type]


def test_stats_account_for_every_decision() -> None:
    store, _recorder = _store(capacity=1)
    store.put((1,), (1,), object())
    store.put((2,), (2,), object())  # evicts (1,)
    store.get((2,), (2,))  # hit
    store.get((2,), (9,))  # block-id mismatch
    store.get((3,), (3,))  # miss

    stats = store.stats()
    assert stats.entries == 0
    assert stats.capacity == 1
    assert stats.stored == 2
    assert stats.hits == 1
    assert stats.misses == 2
    assert stats.block_id_mismatches == 1
    assert stats.evicted == 1
    assert stats.released == 2
