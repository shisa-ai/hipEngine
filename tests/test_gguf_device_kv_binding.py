from __future__ import annotations

import ctypes
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import DType
from hipengine.kvcache import DeviceChunkedKVPool, DeviceKVPoolAllocation
from hipengine.runtime import qwen35_gguf_runner as gguf_runner


@dataclass(frozen=True)
class _FakeScratch:
    block_table_tensor: object
    block_table: DeviceBuffer
    full_key_caches: tuple[DeviceBuffer | None, ...]
    full_value_caches: tuple[DeviceBuffer | None, ...]


def test_gguf_device_kv_binding_uses_full_backing_and_logical_block_table(monkeypatch) -> None:
    page_nbytes = 1024
    key_cache = DeviceBuffer(ptr=0x100000, nbytes=4 * page_nbytes)
    value_cache = DeviceBuffer(ptr=0x200000, nbytes=4 * page_nbytes)
    backing = gguf_runner.Qwen35GGUFBF16KVChunkBacking(
        start_block_id=8,
        pages=4,
        page_nbytes_per_tensor=page_nbytes,
        full_key_caches=(None, key_cache),
        full_value_caches=(None, value_cache),
        buffers=(key_cache, value_cache),
    )
    allocation = DeviceKVPoolAllocation(
        request_id=20,
        block_ids=(8, 9, 11),
        pointers=(0x100000, 0x100400, 0x100C00),
        chunk_start_block_id=8,
        backing=backing,
        reused_block_ids=(8, 9),
        allocated_block_ids=(11,),
        first_divergent_token=512,
    )
    copied_tables: list[np.ndarray] = []

    def capture_table(buffer, host_ptr, nbytes, *, runtime) -> None:
        del runtime
        assert buffer.ptr == 0x300000
        values = (ctypes.c_int32 * (int(nbytes) // 4)).from_address(int(host_ptr))
        copied_tables.append(np.ctypeslib.as_array(values).copy())

    monkeypatch.setattr(gguf_runner, "copy_host_to_device", capture_table)
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.defer_kv_allocation = True
    session.kv_storage_dtype = DType.BF16
    session.scratch = _FakeScratch(
        block_table_tensor=SimpleNamespace(numel=4, shape=(4,)),
        block_table=DeviceBuffer(ptr=0x300000, nbytes=16),
        full_key_caches=(None, None),
        full_value_caches=(None, None),
    )
    session.runtime = object()
    session._device_kv_pool = None
    session._device_kv_allocation = None
    pool = object.__new__(DeviceChunkedKVPool)

    session.bind_device_kv_allocation(pool, allocation)

    assert session.scratch.full_key_caches == (None, key_cache)
    assert session.scratch.full_value_caches == (None, value_cache)
    assert len(copied_tables) == 1
    assert np.array_equal(copied_tables[0], np.asarray([0, 1, 3, 0], dtype=np.int32))
    assert session.device_kv_allocation is allocation
    assert session.device_kv_capacity_tokens == 3 * 256
