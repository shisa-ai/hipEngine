"""Exercise runtime pointer-table publication across repeated KV pool growth."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_kv as gguf_kv
from hipengine.runtime import qwen35_gguf_runner as runner
from tests.test_unit_gguf_packed_workspace_stability import (
    _allocator_fake_runner,
    _int8_kv_layout,
)


def test_runtime_growth_preserves_every_previous_device_page_pointer(monkeypatch):
    next_pointer = 0x100000
    uploaded = {}
    live = set()

    def malloc(nbytes, **kwargs):
        nonlocal next_pointer
        buffer = DeviceBuffer(ptr=next_pointer, nbytes=int(nbytes))
        next_pointer += int(nbytes) + 256
        live.add(buffer.ptr)
        return buffer

    def free(buffer, **kwargs):
        live.remove(buffer.ptr)

    def upload(buffer, array, nbytes=None, **kwargs):
        count = int(array.nbytes if nbytes is None else nbytes)
        uploaded[buffer.ptr] = np.ascontiguousarray(array).tobytes()[:count]

    monkeypatch.setattr(runner, 'malloc', malloc)
    monkeypatch.setattr(runner, 'free', free)
    monkeypatch.setattr(gguf_kv, 'malloc', malloc)
    monkeypatch.setattr(gguf_kv, 'free', free)
    monkeypatch.setattr(runner, 'copy_host_array_to_device', upload)
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    session.defer_kv_allocation = True
    session.runner = _allocator_fake_runner()
    session.scratch = SimpleNamespace()
    session.runtime = SimpleNamespace(memset=lambda *args: None)
    session._device_kv_layout = _int8_kv_layout()
    session.kv_pool_memory_budget_mib = 128
    session.model_path = 'test.gguf'
    session.kv_storage_dtype = runner.DType.INT8_PER_TOKEN_HEAD
    session.kv_storage_layout = 'uniform'
    pool = session.create_global_device_kv_pool(page_capacity=2, generation=1)
    try:
        for growth in (1, 3, 2):
            pool.grow(growth)
            global_pool = pool.global_pool
            for role, pointer in global_pool._pointer_table_pointers.items():
                table = np.frombuffer(uploaded[pointer], dtype=np.uint64)
                expected = tuple(global_pool.page_pointer(role, page)
                                 for page in range(pool.current_pages))
                assert tuple(table) == expected
    finally:
        pool.close()
    assert not live


def test_auto_budget_shrinks_the_initial_floor_before_allocation(monkeypatch):
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    session.defer_kv_allocation = True
    session.runner = _allocator_fake_runner()
    session.scratch = SimpleNamespace()
    session._device_kv_layout = _int8_kv_layout()
    page_bytes = runner._qwen35_gguf_kv_page_bytes(
        session.runner.weights.config, session._device_kv_layout,
    )
    session.runtime = SimpleNamespace(mem_get_info=lambda: (1 * 1024**3 + 20 * page_bytes, 24 * 1024**3))
    session.kv_pool_memory_budget_mib = None

    def allocate(*args, **kwargs):
        assert kwargs['pages'] == 10
        raise RuntimeError('reached budgeted allocator')

    monkeypatch.setattr(runner, '_allocate_qwen35_gguf_kv_chunk', allocate)
    with pytest.raises(RuntimeError, match='reached budgeted allocator'):
        session.create_global_device_kv_pool(page_capacity=128, generation=1)


def test_copy_on_write_preserves_appended_chunk_identity(monkeypatch):
    from hipengine.kvcache.device_global import GlobalDeviceKVPool
    from hipengine.kvcache.pool import DeviceKVPoolAllocation

    allocation = DeviceKVPoolAllocation(
        request_id=7, block_ids=(4, 5), pointers={},
        chunk_start_block_id=4, backing=object(),
        reused_block_ids=(4,), allocated_block_ids=(5,),
        pool_page_capacity=8,
    )
    pool = object.__new__(GlobalDeviceKVPool)
    import threading
    pool._lock = threading.RLock()
    pool._request_allocations = {}
    pool._cow_fork_events = 0
    pool._cow_forked_pages = 0
    monkeypatch.setattr(pool, 'admit_with_shared_prefix', lambda *a, **k: allocation)
    fork = pool.fork_copy_on_write(7, (4,), suffix_pages=1, first_divergent_token=256)
    assert fork.chunk_start_block_id == 4
    assert fork.pool_page_capacity == 8
    assert fork.backing is allocation.backing
    assert fork.first_divergent_token == 256


def _chunk_provider_pool(monkeypatch):
    """Build a chunk-confined pool over a fake device allocator.

    Returns the pool, the live-pointer set, and the recorded uploads so a shrink
    can be checked against the storage the provider actually holds.
    """

    next_pointer = 0x100000
    uploaded: dict[int, bytes] = {}
    live: set[int] = set()

    def malloc(nbytes, **kwargs):
        nonlocal next_pointer
        buffer = DeviceBuffer(ptr=next_pointer, nbytes=int(nbytes))
        next_pointer += int(nbytes) + 256
        live.add(buffer.ptr)
        return buffer

    def free(buffer, **kwargs):
        live.remove(buffer.ptr)

    def upload(buffer, array, nbytes=None, **kwargs):
        count = int(array.nbytes if nbytes is None else nbytes)
        uploaded[buffer.ptr] = np.ascontiguousarray(array).tobytes()[:count]

    monkeypatch.setattr(runner, 'malloc', malloc)
    monkeypatch.setattr(runner, 'free', free)
    monkeypatch.setattr(gguf_kv, 'malloc', malloc)
    monkeypatch.setattr(gguf_kv, 'free', free)
    monkeypatch.setattr(runner, 'copy_host_array_to_device', upload)
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    session.defer_kv_allocation = True
    session.runner = _allocator_fake_runner()
    session.scratch = SimpleNamespace()
    session.runtime = SimpleNamespace(memset=lambda *args: None)
    session._device_kv_layout = _int8_kv_layout()
    session.kv_pool_memory_budget_mib = 128
    session.model_path = 'test.gguf'
    session.kv_storage_dtype = runner.DType.INT8_PER_TOKEN_HEAD
    session.kv_storage_layout = 'uniform'
    pool = session.create_global_device_kv_pool(page_capacity=2, generation=1)
    return pool, live, uploaded


def test_runtime_shrink_releases_the_appended_chunk_and_republishes_tables(
    monkeypatch,
):
    """A chunk-confined provider frees only the chunks above the retained capacity."""

    pool, live, uploaded = _chunk_provider_pool(monkeypatch)
    try:
        pool.grow(3)
        assert pool.current_pages == 5
        assert len(pool._chunks) == 2
        appended = pool._chunks[-1][2]
        appended_buffers = {int(buffer.ptr) for buffer in appended.buffers}
        assert appended_buffers <= live

        released = pool.shrink_idle(now_seconds=1.0)

        assert released == 3
        assert pool.current_pages == 2
        assert len(pool._chunks) == 1
        assert pool.stats.shrink_events == 1
        # The appended chunk's plane buffers are freed, and only the republished
        # tables remain beyond the load-time chunk.
        assert appended_buffers.isdisjoint(live)
        descriptor = np.frombuffer(
            uploaded[int(pool.global_pool.storage_view().metadata_descriptor_ptr)],
            dtype=np.uint64,
        )
        assert int(descriptor[1]) == 2
        for role, pointer in pool.global_pool._pointer_table_pointers.items():
            table = np.frombuffer(uploaded[pointer], dtype=np.uint64)
            expected = tuple(
                pool.global_pool.page_pointer(role, page)
                for page in range(pool.current_pages)
            )
            assert tuple(int(value) for value in table) == expected

        # The load-time chunk is the floor and is never retracted.
        assert pool.shrink_idle(now_seconds=2.0) == 0
        assert pool.current_pages == 2
    finally:
        pool.close()
    assert not live


def test_runtime_shrink_keeps_a_chunk_a_live_allocation_still_uses(monkeypatch):
    pool, live, _uploaded = _chunk_provider_pool(monkeypatch)
    try:
        pool.grow(3)
        allocation = pool.allocate(1, 3)
        assert allocation.chunk_start_block_id == 2

        assert pool.shrink_idle(now_seconds=1.0) == 0
        assert pool.current_pages == 5

        pool.release(1)
        assert pool.shrink_idle(now_seconds=1.0) == 3
        assert pool.current_pages == 2
    finally:
        pool.close()
    assert not live


def test_runtime_shrink_reports_a_device_free_failure_without_double_freeing(
    monkeypatch,
):
    """A free failure after retraction leaks; it must not corrupt the pool."""

    pool, live, _uploaded = _chunk_provider_pool(monkeypatch)
    fake_free = gguf_kv.free
    appended_buffers: set[int] = set()
    try:
        pool.grow(3)
        appended = pool._chunks[-1][2]
        appended_buffers = {int(buffer.ptr) for buffer in appended.buffers}

        def failing_free(buffer, **kwargs):
            if int(buffer.ptr) in appended_buffers:
                raise RuntimeError("hipFree failed")
            return fake_free(buffer, **kwargs)

        monkeypatch.setattr(runner, 'free', failing_free)
        monkeypatch.setattr(gguf_kv, 'free', failing_free)
        with pytest.raises(RuntimeError, match="device free"):
            pool.shrink_idle(now_seconds=1.0)

        # The pool retracted the chunk before freeing it, so it no longer tracks
        # the storage and close cannot free it a second time.
        assert pool.current_pages == 2
        assert len(pool._chunks) == 1
        assert pool.stats.shrink_events == 1
        assert appended_buffers <= live
    finally:
        monkeypatch.setattr(runner, 'free', fake_free)
        monkeypatch.setattr(gguf_kv, 'free', fake_free)
        pool.close()
    # The failed frees leaked exactly the released chunk's plane buffers.
    assert live == appended_buffers
