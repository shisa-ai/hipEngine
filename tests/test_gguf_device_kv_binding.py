from __future__ import annotations

import ctypes
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import DType, Tensor
from hipengine.kvcache import DeviceChunkedKVPool, DeviceKVPoolAllocation, KVScaleMetadata
from hipengine.runtime import qwen35_gguf_runner as gguf_runner


class _FakeRuntime:
    def __init__(self) -> None:
        self.memcpy_async_calls: list[tuple[int, int, int, object, int]] = []
        self.malloc_calls: list[int] = []
        self.free_calls: list[int] = []
        self.device_synchronize_calls = 0
        self.stream_synchronize_calls: list[int] = []
        self._next_ptr = 0x9000

    def malloc(self, nbytes: int) -> int:
        ptr = self._next_ptr
        self._next_ptr += 0x1000
        self.malloc_calls.append(int(nbytes))
        return ptr

    def free(self, ptr: int) -> None:
        self.free_calls.append(int(ptr))

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: object, stream: int) -> None:
        self.memcpy_async_calls.append((int(dst), int(src), int(nbytes), kind, int(stream)))

    def device_synchronize(self) -> None:
        self.device_synchronize_calls += 1

    def stream_synchronize(self, stream: int) -> None:
        self.stream_synchronize_calls.append(int(stream))


@dataclass(frozen=True)
class _FakeScratch:
    block_table_tensor: object
    block_table: DeviceBuffer
    full_key_caches: tuple[DeviceBuffer | None, ...]
    full_value_caches: tuple[DeviceBuffer | None, ...]
    full_bf16_mirror_key_caches: tuple[DeviceBuffer | None, ...]
    full_bf16_mirror_value_caches: tuple[DeviceBuffer | None, ...]
    full_k_scale_caches: tuple[DeviceBuffer | None, ...]
    full_v_scale_caches: tuple[DeviceBuffer | None, ...]
    full_kv_scale_metadata: tuple[KVScaleMetadata | None, ...]


def test_gguf_device_kv_binding_uses_full_backing_and_logical_block_table(monkeypatch) -> None:
    page_nbytes = 1024
    key_cache = DeviceBuffer(ptr=0x100000, nbytes=4 * page_nbytes)
    value_cache = DeviceBuffer(ptr=0x200000, nbytes=4 * page_nbytes)
    layout = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.BF16,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(None, DType.BF16),
    )
    backing = gguf_runner.Qwen35GGUFKVChunkBacking(
        layout=layout,
        start_block_id=8,
        pages=4,
        full_key_caches=(None, key_cache),
        full_value_caches=(None, value_cache),
        full_bf16_mirror_key_caches=(None, None),
        full_bf16_mirror_value_caches=(None, None),
        full_k_scale_caches=(None, None),
        full_v_scale_caches=(None, None),
        full_kv_scale_metadata=(None, None),
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
    session._device_kv_layout = layout
    session.scratch = _FakeScratch(
        block_table_tensor=SimpleNamespace(numel=4, shape=(4,)),
        block_table=DeviceBuffer(ptr=0x300000, nbytes=16),
        full_key_caches=(None, None),
        full_value_caches=(None, None),
        full_bf16_mirror_key_caches=(None, None),
        full_bf16_mirror_value_caches=(None, None),
        full_k_scale_caches=(None, None),
        full_v_scale_caches=(None, None),
        full_kv_scale_metadata=(None, None),
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


def test_gguf_device_kv_chunk_allocates_policy_shaped_payload_and_scales() -> None:
    runtime = _FakeRuntime()
    runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(
                layer_types=(gguf_runner.LINEAR_ATTENTION, gguf_runner.FULL_ATTENTION, gguf_runner.FULL_ATTENTION),
                head_count_kv=2,
                key_length=64,
            )
        )
    )
    layout = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(None, DType.BF16, DType.INT8_PER_TOKEN_HEAD),
    )

    backing = gguf_runner._allocate_qwen35_gguf_kv_chunk(
        runner,
        runtime=runtime,
        start_block_id=12,
        pages=2,
        layout=layout,
    )

    assert runtime.malloc_calls == [131072, 131072, 65536, 65536, 2048, 2048]
    assert backing.layout == layout
    assert backing.total_nbytes == sum(runtime.malloc_calls)
    assert backing.page_pointer(1) == backing.full_key_caches[1].ptr + 65536
    assert backing.full_kv_scale_metadata[1] is None
    int8_metadata = backing.full_kv_scale_metadata[2]
    assert int8_metadata is not None
    assert int8_metadata.k_scale.shape == (2, 256, 2)
    assert int8_metadata.v_scale.shape == (2, 256, 2)
    assert backing.full_bf16_mirror_key_caches == (None, None, None)
    assert backing.full_bf16_mirror_value_caches == (None, None, None)

    gguf_runner._free_qwen35_gguf_kv_chunk(backing, runtime=runtime)
    assert runtime.free_calls == [buffer.ptr for buffer in reversed(backing.buffers)]


def test_gguf_device_kv_binding_binds_and_unbinds_int8_scale_backing(monkeypatch) -> None:
    device = Device("hip", 0)
    key_cache = DeviceBuffer(ptr=0x110000, nbytes=4096)
    value_cache = DeviceBuffer(ptr=0x120000, nbytes=4096)
    k_scale = DeviceBuffer(ptr=0x130000, nbytes=256)
    v_scale = DeviceBuffer(ptr=0x140000, nbytes=256)
    metadata = KVScaleMetadata(
        k_scale=Tensor.from_handle(k_scale.ptr, (2, 256, 2), DType.FP16, device),
        v_scale=Tensor.from_handle(v_scale.ptr, (2, 256, 2), DType.FP16, device),
        scale_dtype=DType.FP16,
        granularity="per_token_head",
    )
    layout = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(None, DType.INT8_PER_TOKEN_HEAD),
    )
    backing = gguf_runner.Qwen35GGUFKVChunkBacking(
        layout=layout,
        start_block_id=4,
        pages=2,
        full_key_caches=(None, key_cache),
        full_value_caches=(None, value_cache),
        full_bf16_mirror_key_caches=(None, None),
        full_bf16_mirror_value_caches=(None, None),
        full_k_scale_caches=(None, k_scale),
        full_v_scale_caches=(None, v_scale),
        full_kv_scale_metadata=(None, metadata),
        buffers=(key_cache, value_cache, k_scale, v_scale),
    )
    allocation = DeviceKVPoolAllocation(
        request_id=7,
        block_ids=(4, 5),
        pointers=(key_cache.ptr, key_cache.ptr + 2048),
        chunk_start_block_id=4,
        backing=backing,
    )
    copied_tables: list[np.ndarray] = []
    monkeypatch.setattr(
        gguf_runner,
        "copy_host_to_device",
        lambda _buffer, host_ptr, nbytes, **_kwargs: copied_tables.append(
            np.ctypeslib.as_array(
                (ctypes.c_int32 * (int(nbytes) // 4)).from_address(int(host_ptr))
            ).copy()
        ),
    )
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.defer_kv_allocation = True
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    session._device_kv_layout = layout
    session.scratch = _FakeScratch(
        block_table_tensor=SimpleNamespace(numel=2, shape=(2,)),
        block_table=DeviceBuffer(ptr=0x150000, nbytes=8),
        full_key_caches=(None, None),
        full_value_caches=(None, None),
        full_bf16_mirror_key_caches=(None, None),
        full_bf16_mirror_value_caches=(None, None),
        full_k_scale_caches=(None, None),
        full_v_scale_caches=(None, None),
        full_kv_scale_metadata=(None, None),
    )
    session.runtime = object()
    session._device_kv_pool = None
    session._device_kv_allocation = None
    session._device_kv_graph_handles = {}
    pool = object.__new__(DeviceChunkedKVPool)

    session.bind_device_kv_allocation(pool, allocation)

    assert session.scratch.full_key_caches == (None, key_cache)
    assert session.scratch.full_value_caches == (None, value_cache)
    assert session.scratch.full_k_scale_caches == (None, k_scale)
    assert session.scratch.full_v_scale_caches == (None, v_scale)
    assert session.scratch.full_kv_scale_metadata == (None, metadata)
    assert copied_tables[0].tolist() == [0, 1]

    assert session.unbind_device_kv_allocation() is allocation
    assert session.scratch.full_key_caches == (None, None)
    assert session.scratch.full_value_caches == (None, None)
    assert session.scratch.full_k_scale_caches == (None, None)
    assert session.scratch.full_v_scale_caches == (None, None)
    assert session.scratch.full_kv_scale_metadata == (None, None)
    assert copied_tables[1].tolist() == [0, 0]


def test_gguf_device_kv_binding_rejects_policy_mismatch_before_table_copy(monkeypatch) -> None:
    expected = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.INT8_PER_TOKEN_HEAD,),
    )
    incompatible = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP32,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.INT8_PER_TOKEN_HEAD,),
    )
    backing = SimpleNamespace(layout=incompatible)
    allocation = DeviceKVPoolAllocation(
        request_id=8,
        block_ids=(0,),
        pointers=(0x1000,),
        chunk_start_block_id=0,
        backing=backing,
    )
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.defer_kv_allocation = True
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    session._device_kv_layout = expected
    session.scratch = SimpleNamespace()
    session._device_kv_allocation = None
    monkeypatch.setattr(
        gguf_runner,
        "copy_host_to_device",
        lambda *args, **kwargs: pytest.fail("policy mismatch touched the device block table"),
    )

    with pytest.raises(TypeError, match="layout does not match"):
        session.bind_device_kv_allocation(object.__new__(DeviceChunkedKVPool), allocation)


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


def test_gguf_prefix_state_snapshot_outlives_source_session_and_restores_boundary(
    monkeypatch,
) -> None:
    runtime = _FakeRuntime()
    shared_runner = object()
    shared_backing = object()

    def make_session(*, state_base: int, allocation) -> gguf_runner.Qwen35GGUFResidentSession:
        session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
        session.runtime = runtime
        session.runner = shared_runner
        session.scratch = SimpleNamespace(
            max_positions=1024,
            layer_conv_states=(DeviceBuffer(state_base, 64), None),
            layer_recurrent_states=(DeviceBuffer(state_base + 0x200, 128), None),
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

    source = make_session(
        state_base=0x1000,
        allocation=SimpleNamespace(
            block_ids=(8, 9),
            backing=shared_backing,
            reused_block_ids=(),
        ),
    )
    destination = make_session(
        state_base=0x5000,
        allocation=SimpleNamespace(
            block_ids=(8, 10),
            backing=shared_backing,
            reused_block_ids=(8,),
        ),
    )
    source._position = 256
    source.scratch.position_host[0] = 256
    source.scratch.context_host[0] = 257
    position_calls: list[tuple[int, int, int, int]] = []
    monkeypatch.setattr(
        gguf_runner,
        "set_decode_position_i64",
        lambda position_ptr, context_ptr, position, *, stream, **kwargs: position_calls.append(
            (int(position_ptr), int(context_ptr), int(position), int(stream))
        ),
    )

    snapshot = source.capture_prefix_state_snapshot()
    assert snapshot.position == 256
    assert snapshot.block_ids == (8,)
    assert snapshot.nbytes == 192
    assert runtime.malloc_calls == [64, 128]
    assert runtime.device_synchronize_calls == 1

    source._position = 0
    copied_bytes = destination.clone_prefix_state_from_snapshot(snapshot, stream=7)
    assert copied_bytes == 192
    assert destination.position == 256
    assert position_calls == [(0x5800, 0x5900, 256, 7)]
    assert runtime.stream_synchronize_calls == [7]
    assert runtime.memcpy_async_calls[-2:] == [
        (0x5000, 0x9000, 64, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x5200, 0xA000, 128, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
    ]

    snapshot.close()
    assert snapshot.closed is True
    assert runtime.free_calls == [0xA000, 0x9000]
    with pytest.raises(RuntimeError, match="closed"):
        destination.clone_prefix_state_from_snapshot(snapshot)
