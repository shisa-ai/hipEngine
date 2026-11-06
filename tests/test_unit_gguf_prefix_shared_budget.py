"""Retained prefix budget prices target/provider owner unions, not view sums."""
from types import SimpleNamespace

from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.speculative.prefix_checkpoint import PrefixCheckpointStore


def test_owner_trim_bounds_provider_host_storage_without_dropping_target():
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._prefix_state_snapshots = {(1,): SimpleNamespace(
        snapshot=SimpleNamespace(nbytes=10), retained=True)}
    store = PrefixCheckpointStore(capacity=4, release=lambda _: None)
    store.put((1,), (10,), SimpleNamespace(nbytes=0, host_nbytes=100))
    runner._mtp2_adapter = SimpleNamespace(_prefix_checkpoint_store_instance=store)
    runner._prefix_snapshot_limit = 4
    runner._prefix_retained_limit = 4
    runner._prefix_retained_state_bytes_limit = 50
    runner._trim_prefix_snapshots()
    assert store.host_nbytes == 0
    assert (1,) in runner._prefix_state_snapshots


def test_prefix_observability_separates_host_and_device_checkpoint_bytes():
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._prefix_cache = None
    runner._kv_pool = None
    runner._prefix_state_snapshots = {}
    store = PrefixCheckpointStore(capacity=4, release=lambda _: None)
    store.put((1,), (10,), SimpleNamespace(nbytes=64, host_nbytes=1000))
    runner._mtp2_adapter = SimpleNamespace(_prefix_checkpoint_store_instance=store)
    result = runner._prefix_cache_observability()
    assert result["provider_checkpoint_device_bytes"] == 64
    assert result["provider_checkpoint_host_bytes"] == 1000
    assert result["retained_checkpoint_owner_device_bytes"] == 0


def test_trim_prices_idle_pool_against_remaining_byte_budget():
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    calls = []
    pool = SimpleNamespace(trim_idle=lambda limit: calls.append(limit))
    snapshot = SimpleNamespace(nbytes=30, arena_pool=pool)
    runner._prefix_state_snapshots = {(1,): SimpleNamespace(snapshot=snapshot, retained=True)}
    runner._mtp2_adapter = None
    runner._prefix_snapshot_limit = 4
    runner._prefix_retained_limit = 4
    runner._prefix_retained_state_bytes_limit = 50
    runner._available = []
    runner._rows = {}
    runner._trim_prefix_snapshots()
    assert calls == [20]


def test_trim_evicts_provider_bytes_with_target_boundary():
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._prefix_state_snapshots = {
        (1,): SimpleNamespace(snapshot=SimpleNamespace(nbytes=10), retained=True),
        (2,): SimpleNamespace(snapshot=SimpleNamespace(nbytes=10), retained=True),
    }
    released = []
    store = PrefixCheckpointStore(capacity=4, release=lambda p: released.append(p))
    payload = SimpleNamespace(device_allocations=((4096, 100),))
    store.put((1,), (10,), payload)
    runner._mtp2_adapter = SimpleNamespace(_prefix_checkpoint_store_instance=store)
    runner._prefix_snapshot_limit = 4
    runner._prefix_retained_limit = 4
    runner._prefix_retained_state_bytes_limit = 50
    evicted = []
    def evict(key, *, reason):
        evicted.append(key)
        runner._prefix_state_snapshots.pop(key)
        store.drop(key)
    runner._evict_prefix_snapshot = evict
    runner._trim_prefix_snapshots()
    assert evicted == [(1,)]
    assert released == [payload]
    assert tuple(runner._prefix_state_snapshots) == ((2,),)


def test_retained_budget_counts_shared_owners_once_and_arena_padding():
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    shared = SimpleNamespace(device_allocations=((4096, 8192),), nbytes=192)
    legacy = SimpleNamespace(nbytes=100)
    runner._prefix_state_snapshots = {
        (1,): SimpleNamespace(snapshot=shared, retained=True),
        (2,): SimpleNamespace(snapshot=shared, retained=True),
        (3,): SimpleNamespace(snapshot=legacy, retained=True),
        (4,): SimpleNamespace(snapshot=legacy, retained=True),
    }
    store = PrefixCheckpointStore(capacity=4, release=lambda _: None)
    store.put((1,), (10,), SimpleNamespace(device_allocations=((4096, 8192), (16384, 64))))
    runner._mtp2_adapter = SimpleNamespace(_prefix_checkpoint_store_instance=store)
    assert runner._retained_prefix_device_bytes(((1,), (2,), (3,), (4,))) == 8192 + 64 + 100
    assert runner._retained_prefix_device_bytes(((2,),)) == 8192
