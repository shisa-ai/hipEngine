from __future__ import annotations

import ctypes
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import DType
from hipengine.kvcache import DeviceChunkedKVPool, DeviceKVPoolAllocation
from hipengine.runtime import qwen35_gguf_runner as gguf_runner


class _FakeRuntime:
    def __init__(self) -> None:
        self.memcpy_async_calls: list[tuple[int, int, int, object, int]] = []

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: object, stream: int) -> None:
        self.memcpy_async_calls.append((int(dst), int(src), int(nbytes), kind, int(stream)))


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


def test_gguf_prefix_state_clone_copies_exact_current_hybrid_boundary(monkeypatch) -> None:
    runtime = _FakeRuntime()
    shared_runner = object()
    shared_backing = object()

    def make_session(*, state_base: int, allocation) -> gguf_runner.Qwen35GGUFResidentSession:
        session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
        session.runtime = runtime
        session.runner = shared_runner
        session.scratch = SimpleNamespace(
            max_positions=1024,
            layer_conv_states=(
                DeviceBuffer(state_base + 0x000, 64),
                None,
                DeviceBuffer(state_base + 0x100, 64),
            ),
            layer_recurrent_states=(
                DeviceBuffer(state_base + 0x200, 128),
                None,
                DeviceBuffer(state_base + 0x400, 128),
            ),
            position_host=np.zeros((1,), dtype=np.int64),
            context_host=np.ones((1,), dtype=np.int64),
            position_buf=DeviceBuffer(state_base + 0x800, 8),
            context_buf=DeviceBuffer(state_base + 0x900, 8),
        )
        session.kv_storage_dtype = DType.BF16
        session.kv_storage_layout = "uniform"
        session._device_kv_allocation = allocation
        session._device_kv_pool = object()
        session._device_kv_graph_handles = {}
        session._decode_graphs = []
        session._packed_decode_state_dirty = False
        session._runtime_state_library = object()
        session._position = 0
        session._hidden_seed_fp32_populated = True
        session._last_pre_output_norm_hidden = np.ones((1, 1), dtype=np.float32)
        session._last_layer_output_hidden = {0: np.ones((1, 1), dtype=np.float32)}
        return session

    source_allocation = SimpleNamespace(
        block_ids=(8, 9, 10),
        backing=shared_backing,
        reused_block_ids=(),
    )
    destination_allocation = SimpleNamespace(
        block_ids=(8, 9, 11),
        backing=shared_backing,
        reused_block_ids=(8, 9),
    )
    source = make_session(state_base=0x1000, allocation=source_allocation)
    destination = make_session(state_base=0x5000, allocation=destination_allocation)
    source._position = 512
    source.scratch.position_host[0] = 512
    source.scratch.context_host[0] = 513
    position_calls: list[tuple[int, int, int, int]] = []
    monkeypatch.setattr(
        gguf_runner,
        "set_decode_position_i64",
        lambda position_ptr, context_ptr, position, *, stream, **kwargs: position_calls.append(
            (int(position_ptr), int(context_ptr), int(position), int(stream))
        ),
    )

    copied_bytes = destination.clone_prefix_state_from(source, stream=7)

    assert copied_bytes == 64 + 128 + 64 + 128
    assert runtime.memcpy_async_calls == [
        (0x5000, 0x1000, 64, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x5200, 0x1200, 128, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x5100, 0x1100, 64, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x5400, 0x1400, 128, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
    ]
    assert position_calls == [(0x5800, 0x5900, 512, 7)]
    assert destination.position == 512
    assert destination.scratch.position_host.tolist() == [512]
    assert destination.scratch.context_host.tolist() == [513]
    assert not destination._hidden_seed_fp32_populated
    assert destination._last_pre_output_norm_hidden is None
    assert destination._last_layer_output_hidden == {}

    with pytest.raises(ValueError, match="current source boundary"):
        destination.clone_prefix_state_from(source, position=256)
