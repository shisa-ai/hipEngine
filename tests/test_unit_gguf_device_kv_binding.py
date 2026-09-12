from __future__ import annotations

import ctypes
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import DType, Tensor
from hipengine.kvcache import (
    DeviceChunkedKVPool,
    DeviceKVPoolAllocation,
    KVScaleMetadata,
)
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
    assert session.device_kv_layout_audit() == {
        "pool_contract": "legacy_single_backing",
        "pool_storage_layout": None,
        "storage_dtype": "int8_per_token_head",
        "storage_layout": "uniform",
        "scale_dtype": "fp16",
        "scale_granularity": "per_token_head",
        "kv_attention_source": "int8_direct",
        "request_pages": 2,
        "request_capacity_tokens": 512,
        "request_block_ids": [4, 5],
        "one_backing_chunk": True,
        "contiguous_in_backing": True,
        "persistent_int8_payload_bytes": 8192,
        "persistent_bf16_payload_bytes": 0,
        "persistent_scale_bytes": 512,
        "persistent_bf16_mirror_bytes": 0,
        "persistent_total_bytes": 8704,
    }

    assert session.unbind_device_kv_allocation() is allocation
    assert session.scratch.full_key_caches == (None, None)
    assert session.scratch.full_value_caches == (None, None)
    assert session.scratch.full_k_scale_caches == (None, None)
    assert session.scratch.full_v_scale_caches == (None, None)
    assert session.scratch.full_kv_scale_metadata == (None, None)
    assert copied_tables[1].tolist() == [0, 0]


def test_shifted_direct_int8_prefill_repeats_physical_page_table_per_row() -> None:
    allocation = DeviceKVPoolAllocation(
        request_id=2,
        block_ids=(11, 12, 13),
        pointers=(0x1000, 0x2000, 0x3000),
        reused_block_ids=(),
        allocated_block_ids=(11, 12, 13),
        first_divergent_token=None,
        chunk_start_block_id=8,
        backing=object(),
    )

    table = gguf_runner._gguf_retained_prefill_block_table_host(
        allocation,
        rows=3,
        blocks_per_row=4,
    )

    assert table.dtype == np.int32
    assert table.shape == (3, 4)
    assert table.tolist() == [[3, 4, 5, 0]] * 3


def test_shifted_direct_int8_prefill_oracle_covers_backing_physical_pages() -> None:
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.scratch = SimpleNamespace(max_positions=768, block_size=256)
    session._device_kv_allocation = DeviceKVPoolAllocation(
        request_id=3,
        block_ids=(7, 8, 9),
        pointers=(0x810000, 0x820000, 0x830000),
        chunk_start_block_id=4,
        backing=SimpleNamespace(pages=6),
    )

    assert session._int8_prefill_oracle_capacity_positions() == 1536


def test_shifted_direct_int8_prefill_keeps_paged_payload_at_backing_base() -> None:
    backing = SimpleNamespace(pages=4)
    allocation = DeviceKVPoolAllocation(
        request_id=9,
        block_ids=(6, 7),
        pointers=(0x220000, 0x230000),
        chunk_start_block_id=4,
        backing=backing,
    )
    session = SimpleNamespace(_device_kv_allocation=allocation)
    retained_key = DeviceBuffer(0x210000, 4 * 256 * 128)
    retained_value = DeviceBuffer(0x310000, 4 * 256 * 128)
    scale_metadata = object()
    retained_spans = SimpleNamespace(
        block_table=DeviceBuffer(0x410000, 8),
        scale_metadata=scale_metadata,
    )
    scratch = SimpleNamespace(
        key_cache=DeviceBuffer(0x510000, 2 * 256 * 256),
        value_cache=DeviceBuffer(0x610000, 2 * 256 * 256),
        retained_key_cache=retained_key,
        retained_value_cache=retained_value,
        retained_append_spans=retained_spans,
    )

    bound = gguf_runner._gguf_slot_local_prefill_cache_views(
        session,
        scratch,
        row_nbytes=256,
        direct_int8=True,
    )

    assert bound is scratch
    assert bound.retained_key_cache is retained_key
    assert bound.retained_value_cache is retained_value
    assert bound.retained_append_spans is retained_spans
    assert bound.retained_append_spans.scale_metadata is scale_metadata


def test_resident_slot_view_shares_owner_but_resets_slot_local_bookkeeping() -> None:
    slot_scratch = object()
    target_owner = SimpleNamespace(
        slot_count=4,
        for_slot=lambda slot: slot_scratch if slot == 2 else object(),
    )
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.max_batch_size = 4
    session.scratch = object()
    session._target_scratch_owner = target_owner
    session._target_layout = gguf_runner.Qwen35GGUFResidentTargetLayout(
        max_batch_size=4,
        hidden_size=8,
        vocab_size=16,
        max_sequence_length=256,
    )
    session._decode_graphs = [object()]
    session._decode_graph_submission_contexts = {"old": object()}
    session._device_kv_graph_handles = {1: object()}
    session._int8_prefill_oracle_buffers = {1: (object(), object())}
    session._linear_state_snapshot_backups = (object(),)
    session._packed_verify_state = object()
    session._packed_verify_scratch = object()
    session._native_sampler_workspace = object()

    view = session.resident_slot_view(2)

    assert view is not session
    assert view.max_batch_size == 1
    assert view.scratch is slot_scratch
    assert view._target_scratch_owner is target_owner
    assert view._reset_current_slot_only is True
    assert view._target_layout.max_batch_size == 1
    assert view._buffers == ()
    assert view._position == 0
    assert view._decode_graphs == []
    assert view._device_kv_graph_handles == {}
    # Views share the batch owner's packed workspace bookkeeping instead of
    # owning a private allocation (serving-load stability contract).
    assert view._packed_verify_state is session._packed_verify_state
    assert view._packed_verify_scratch is session._packed_verify_scratch
    assert view._packed_decode_state_dirty is session._packed_decode_state_dirty
    assert view._resident_batch_owner is session
    assert view._resident_slot_index == 2


def test_resident_slot_view_releases_view_local_verify_buffers() -> None:
    slot_scratch = object()
    target_owner = SimpleNamespace(
        slot_count=2,
        for_slot=lambda slot: slot_scratch,
    )
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.max_batch_size = 2
    session.runner = object()
    session.scratch = object()
    session._target_scratch_owner = target_owner
    session._target_layout = gguf_runner.Qwen35GGUFResidentTargetLayout(
        max_batch_size=2,
        hidden_size=8,
        vocab_size=16,
        max_sequence_length=256,
    )
    session._decode_graphs = []
    session._decode_graph_submission_contexts = {}
    session._device_kv_graph_handles = {}
    session._int8_prefill_oracle_buffers = {}
    session._linear_state_snapshot_backups = ()
    session._packed_verify_state = object()
    session._packed_verify_scratch = object()
    session._native_sampler_workspace = None
    session._verify_hidden_seed_buf = DeviceBuffer(0x1000, 64)
    session._verify_hidden_f32_a = DeviceBuffer(0x2000, 64)
    session._verify_hidden_f32_b = DeviceBuffer(0x3000, 64)
    session._verify_token_ids_i64 = DeviceBuffer(0x4000, 16)
    session._verify_token_counter_i64 = DeviceBuffer(0x5000, 8)
    session._verify_block_rows_capacity = 2
    session._verify_hidden_seed_rows_populated = 2
    session._verify_linear_recurrent_state_rows = (DeviceBuffer(0x6000, 32),)
    session._verify_linear_conv_state_rows = (DeviceBuffer(0x7000, 32),)
    session._verify_linear_state_rows_capacity = 2
    session._verify_linear_recurrent_initial_snapshots = (DeviceBuffer(0x8000, 32),)
    session._verify_linear_conv_initial_snapshots = (DeviceBuffer(0x9000, 32),)

    view = session.resident_slot_view(1)

    assert view._verify_hidden_seed_buf is None
    assert view._verify_hidden_f32_a is None
    assert view._verify_hidden_f32_b is None
    assert view._verify_token_ids_i64 is None
    assert view._verify_token_counter_i64 is None
    assert view._verify_linear_recurrent_state_rows == ()
    assert view._verify_linear_conv_state_rows == ()
    assert view._verify_linear_recurrent_initial_snapshots == ()
    assert view._verify_linear_conv_initial_snapshots == ()

    local_buffers = tuple(
        DeviceBuffer(0xA000 + index * 0x1000, size)
        for index, size in enumerate((64, 64, 64, 16, 8, 32, 32, 32, 32))
    )
    (
        view._verify_hidden_seed_buf,
        view._verify_hidden_f32_a,
        view._verify_hidden_f32_b,
        view._verify_token_ids_i64,
        view._verify_token_counter_i64,
        recurrent,
        conv,
        recurrent_initial,
        conv_initial,
    ) = local_buffers
    view._verify_linear_recurrent_state_rows = (recurrent,)
    view._verify_linear_conv_state_rows = (conv,)
    view._verify_linear_recurrent_initial_snapshots = (recurrent_initial,)
    view._verify_linear_conv_initial_snapshots = (conv_initial,)
    runtime = _FakeRuntime()

    view._close_resident_slot_view_buffers(runtime=runtime)

    assert sorted(runtime.free_calls) == sorted(buffer.ptr for buffer in local_buffers)
    assert view._target_scratch_owner is target_owner
    assert view.scratch is slot_scratch
    assert view.runner is session.runner


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


def test_gguf_current_state_clone_copies_private_kv_at_arbitrary_boundary(
    monkeypatch,
) -> None:
    """B3 RED: clone independent KV plus recurrent state at non-page C1."""

    assert hasattr(gguf_runner.Qwen35GGUFResidentSession, "clone_current_state_from")
    runtime = _FakeRuntime()
    layout = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.BF16,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.BF16, None),
    )
    shared_runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(
                layer_types=(gguf_runner.FULL_ATTENTION, gguf_runner.LINEAR_ATTENTION),
                head_count_kv=2,
                key_length=2,
            )
        )
    )

    def make_session(*, base: int, block_ids, chunk_start):
        session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
        session.runtime = runtime
        session.runner = shared_runner
        session.scratch = SimpleNamespace(
            max_positions=1024,
            layer_conv_states=(None, DeviceBuffer(base + 0x4000, 64)),
            layer_recurrent_states=(None, DeviceBuffer(base + 0x5000, 128)),
            full_key_caches=(DeviceBuffer(base + 0x0000, 8192), None),
            full_value_caches=(DeviceBuffer(base + 0x2000, 8192), None),
            position_host=np.zeros((1,), dtype=np.int64),
            context_host=np.ones((1,), dtype=np.int64),
            position_buf=DeviceBuffer(base + 0x6000, 8),
            context_buf=DeviceBuffer(base + 0x7000, 8),
        )
        session.kv_storage_dtype = DType.BF16
        session.kv_storage_layout = "uniform"
        session._device_kv_layout = layout
        session._device_kv_allocation = SimpleNamespace(
            block_ids=tuple(block_ids),
            chunk_start_block_id=int(chunk_start),
            backing=object(),
        )
        session._device_kv_pool = object()
        session._device_kv_graph_handles = {}
        session._decode_graphs = []
        session._packed_decode_state_dirty = False
        session._packed_decode_sessions = ()
        session._packed_decode_session_ids = ()
        session._packed_decode_positions = ()
        session._packed_verify_session_ids = ()
        session._packed_verify_max_written_positions = ()
        session._runtime_state_library = object()
        session._position = 0
        session._hidden_seed_fp32_populated = True
        session._last_pre_output_norm_hidden = np.ones((1, 1), dtype=np.float32)
        session._last_layer_output_hidden = {0: np.ones((1, 1), dtype=np.float32)}
        return session

    source = make_session(base=0x10000, block_ids=(8, 10), chunk_start=8)
    destination = make_session(base=0x30000, block_ids=(20, 21), chunk_start=20)
    source._position = 300
    source.scratch.position_host[0] = 300
    source.scratch.context_host[0] = 301
    position_calls = []
    monkeypatch.setattr(
        gguf_runner,
        "set_decode_position_i64",
        lambda position_ptr, context_ptr, position, *, stream, **kwargs: position_calls.append(
            (int(position_ptr), int(context_ptr), int(position), int(stream))
        ),
    )

    copied = destination.clone_current_state_from(source, stream=7)

    assert copied == (300 * 8 * 2) + 64 + 128
    assert runtime.memcpy_async_calls == [
        (0x30000, 0x10000, 2048, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x32000, 0x12000, 2048, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x30800, 0x11000, 352, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x32800, 0x13000, 352, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x34000, 0x14000, 64, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x35000, 0x15000, 128, gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
    ]
    assert destination.position == 300
    assert position_calls == [(0x36000, 0x37000, 300, 7)]
    assert source._device_kv_allocation.backing is not destination._device_kv_allocation.backing


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
