from __future__ import annotations

import ctypes
from dataclasses import dataclass
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen35_gguf_runner as gguf_runner
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.qwen35_gguf_runner import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    _GGUFPackedTargetState,
    _GGUFPackedVerifySlotBlock,
    _GGUFFullAttentionPrefillScratch,
    _HipEventStageRecorder,
    _build_gguf_packed_verify_layout,
)


def test_prefill_device_metadata_uses_backend_ceiling_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_PREFILL_DEVICE_METADATA", raising=False)
    assert not gguf_runner._gguf_prefill_device_metadata_enabled()
    assert gguf_runner._gguf_prefill_device_metadata_enabled(
        backend="hip_gfx1151", prompt_tokens=4096
    )
    assert not gguf_runner._gguf_prefill_device_metadata_enabled(
        backend="hip_gfx1151", prompt_tokens=4097
    )
    assert gguf_runner._gguf_prefill_device_metadata_enabled(
        backend="hip_gfx1100", prompt_tokens=4096
    )
    assert not gguf_runner._gguf_prefill_device_metadata_enabled(
        backend="hip_gfx1100", prompt_tokens=4097
    )

    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_DEVICE_METADATA", "1")
    assert gguf_runner._gguf_prefill_device_metadata_enabled(
        backend="hip_gfx1151", prompt_tokens=131072
    )
    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_DEVICE_METADATA", "0")
    assert not gguf_runner._gguf_prefill_device_metadata_enabled(
        backend="hip_gfx1151", prompt_tokens=512
    )


def test_gguf_resident_reset_invalidates_packed_state_metadata(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeScratch:
        def zero_states(self, runtime, *, stream: int = 0, set_position: bool = False):
            calls.append(("zero_states", runtime, int(stream), bool(set_position)))

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_set_full_attention_position_device",
        lambda self, position, *, stream=0: calls.append(("set_position", int(position), int(stream))),
    )
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.scratch = FakeScratch()
    session.runtime = SimpleNamespace(name="runtime")
    session._position = 17
    session._hidden_seed_fp32_populated = True
    session._last_pre_output_norm_hidden = np.ones((1,), dtype=np.float32)
    session._last_layer_output_hidden = {3: np.ones((1,), dtype=np.float32)}
    session._verify_hidden_seed_rows_populated = 2
    session._packed_verify_session_ids = (11, 22)
    session._packed_verify_max_written_positions = (4, 4)
    session._packed_decode_sessions = (object(),)
    session._packed_decode_last_layout = object()
    session._packed_decode_state_dirty = True
    session._packed_decode_session_ids = (33,)
    session._packed_decode_positions = (5,)

    session.reset(stream=7)

    assert calls == [
        ("zero_states", session.runtime, 7, False),
        ("set_position", 0, 7),
    ]
    assert session.position == 0
    assert session._hidden_seed_fp32_populated is False
    assert session._last_pre_output_norm_hidden is None
    assert session._last_layer_output_hidden == {}
    assert session._verify_hidden_seed_rows_populated == 0
    assert session._packed_verify_session_ids == ()
    assert session._packed_verify_max_written_positions == ()
    assert session._packed_decode_sessions == ()
    assert session._packed_decode_last_layout is None
    assert session._packed_decode_state_dirty is False
    assert session._packed_decode_session_ids == ()
    assert session._packed_decode_positions == ()


def test_gguf_packed_verify_layout_maps_rows_and_slot_state() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11, 12, 13), start_position=4),
            _GGUFPackedVerifySlotBlock(input_token_ids=(21, 22, 23), start_position=8),
        ),
        block_size=4,
    )

    np.testing.assert_array_equal(layout.input_token_ids, np.asarray([11, 12, 13, 21, 22, 23], dtype=np.int64))
    np.testing.assert_array_equal(layout.row_slot_indices, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(layout.row_offsets_in_slot, np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(layout.row_positions, np.asarray([4, 5, 6, 8, 9, 10], dtype=np.int64))
    np.testing.assert_array_equal(layout.live_counts, np.asarray([5, 6, 7, 9, 10, 11], dtype=np.int64))
    np.testing.assert_array_equal(layout.cu_seqlens, np.asarray([0, 3, 6], dtype=np.int32))
    np.testing.assert_array_equal(layout.state_indices, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        layout.block_table,
        np.asarray(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [3, 4, 5],
                [3, 4, 5],
                [3, 4, 5],
            ],
            dtype=np.int32,
        ),
    )
    assert layout.rows == 6
    assert layout.slot_count == 2
    assert layout.blocks_per_slot == 3
    assert layout.max_live_count == 11
    assert layout.total_physical_positions == 24


def test_gguf_packed_verify_layout_supports_variable_rows() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11, 12, 13), start_position=4),
            _GGUFPackedVerifySlotBlock(input_token_ids=(21,), start_position=2),
        ),
        block_size=4,
    )

    np.testing.assert_array_equal(layout.input_token_ids, np.asarray([11, 12, 13, 21], dtype=np.int64))
    np.testing.assert_array_equal(layout.row_slot_indices, np.asarray([0, 0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(layout.row_positions, np.asarray([4, 5, 6, 2], dtype=np.int64))
    np.testing.assert_array_equal(layout.live_counts, np.asarray([5, 6, 7, 3], dtype=np.int64))
    np.testing.assert_array_equal(layout.cu_seqlens, np.asarray([0, 3, 4], dtype=np.int32))
    assert layout.blocks_per_slot == 2


def test_gguf_packed_prefill_uses_slot_local_full_attention_at_c1_threshold() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(1, 2, 3, 4), start_position=0),
            _GGUFPackedVerifySlotBlock(input_token_ids=(5, 6, 7), start_position=0),
        )
    )

    assert gguf_runner._packed_prefill_requires_slot_local_full_attention(
        layout,
        aotriton_threshold=4,
    )
    assert not gguf_runner._packed_prefill_requires_slot_local_full_attention(
        layout,
        aotriton_threshold=5,
    )
    assert not gguf_runner._packed_prefill_requires_slot_local_full_attention(
        layout,
        aotriton_threshold=0,
    )


def test_gguf_packed_verify_layout_honors_slot_capacity() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11, 12), start_position=4),
            _GGUFPackedVerifySlotBlock(input_token_ids=(21, 22), start_position=4),
        ),
        block_size=4,
        slot_capacity=16,
    )

    assert layout.max_live_count == 6
    assert layout.blocks_per_slot == 4
    np.testing.assert_array_equal(layout.block_table[0], np.asarray([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(layout.block_table[2], np.asarray([4, 5, 6, 7], dtype=np.int32))
    assert layout.total_physical_positions == 32


def test_gguf_packed_verify_layout_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _build_gguf_packed_verify_layout(())
    with pytest.raises(ValueError, match="non-empty"):
        _GGUFPackedVerifySlotBlock(input_token_ids=(), start_position=0)
    with pytest.raises(ValueError, match="non-negative"):
        _GGUFPackedVerifySlotBlock(input_token_ids=(1,), start_position=-1)
    with pytest.raises(ValueError, match="block_size"):
        _build_gguf_packed_verify_layout(
            (_GGUFPackedVerifySlotBlock(input_token_ids=(1,), start_position=0),),
            block_size=0,
        )
    with pytest.raises(ValueError, match="slot_capacity"):
        _build_gguf_packed_verify_layout(
            (_GGUFPackedVerifySlotBlock(input_token_ids=(1,), start_position=4),),
            slot_capacity=4,
        )


def test_gguf_prefill_scratch_uploads_packed_verify_layout(monkeypatch) -> None:
    next_ptr = 0x100000
    copies: dict[int, bytes] = {}

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal next_ptr
        ptr = next_ptr
        next_ptr += max(8, int(nbytes) + 8)
        return DeviceBuffer(ptr=ptr, nbytes=int(nbytes))

    def fake_copy_host_to_device(buffer, host_ptr, nbytes=None, *, runtime):
        count = int(buffer.nbytes if nbytes is None else nbytes)
        copies[int(buffer.ptr)] = ctypes.string_at(int(host_ptr), count)

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "copy_host_to_device", fake_copy_host_to_device)

    cfg = SimpleNamespace(
        expert_used_count=2,
        is_moe=True,
        expert_count=4,
        expert_shared_feed_forward_length=8,
        ssm_inner_size=6,
        ssm_conv_kernel=4,
        ssm_time_step_rank=2,
        ssm_state_size=3,
        head_count_kv=2,
        key_length=4,
        rope_dimension_count=4,
        rope_freq_base=10000.0,
        head_count=4,
    )
    runner = SimpleNamespace(
        hidden_size=8,
        q_width=16,
        kv_width=8,
        ffn_size=12,
        linear_qkv_width=10,
        ssm_value_dim=2,
        backend="hip_gfx1151",
        weights=SimpleNamespace(config=cfg),
    )
    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        runner,
        rows=6,
        capacity=1024,
        segments=2,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11, 12, 13), start_position=4),
            _GGUFPackedVerifySlotBlock(input_token_ids=(21, 22, 23), start_position=8),
        ),
        slot_capacity=1024,
    )
    copies.clear()

    view = scratch.for_packed_verify_layout(layout, runtime=SimpleNamespace())

    block_upload = np.frombuffer(copies[scratch.block_table.ptr], dtype=np.int32).reshape(layout.block_table.shape)
    pos_upload = np.frombuffer(copies[scratch.positions.ptr], dtype=np.int64)
    live_upload = np.frombuffer(copies[scratch.context_counts.ptr], dtype=np.int64)
    cu_upload = np.frombuffer(copies[scratch.gdn_cu_seqlens.ptr], dtype=np.int32)
    state_upload = np.frombuffer(copies[scratch.gdn_state_indices.ptr], dtype=np.int64)
    np.testing.assert_array_equal(block_upload, layout.block_table)
    np.testing.assert_array_equal(pos_upload, layout.row_positions)
    np.testing.assert_array_equal(live_upload, layout.live_counts)
    np.testing.assert_array_equal(cu_upload, layout.cu_seqlens)
    np.testing.assert_array_equal(state_upload, layout.state_indices)
    assert view.rows == layout.rows
    assert view.block_table_tensor.shape == layout.block_table.shape
    assert view.positions_tensor.shape == layout.row_positions.shape
    assert view.context_counts_tensor.shape == layout.live_counts.shape


def test_gguf_packed_target_state_allocates_per_slot_state(monkeypatch) -> None:
    next_ptr = 0x200000
    allocations: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal next_ptr
        buffer = DeviceBuffer(ptr=next_ptr, nbytes=int(nbytes))
        next_ptr += max(8, int(nbytes) + 8)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    cfg = SimpleNamespace(
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION),
        ssm_conv_kernel=4,
        ssm_time_step_rank=2,
        ssm_state_size=3,
        head_count_kv=2,
        key_length=4,
    )
    runner = SimpleNamespace(
        linear_qkv_width=10,
        ssm_value_dim=2,
        weights=SimpleNamespace(config=cfg),
    )

    state = _GGUFPackedTargetState.allocate(
        runner,
        slot_count=2,
        max_sequence_length=300,
        runtime=SimpleNamespace(),
    )

    conv_nbytes = 2 * 10 * 4 * 4
    recurrent_nbytes = 2 * 2 * 3 * 2 * 4
    kv_nbytes = 2 * 512 * 2 * 4 * 2
    assert state.slot_count == 2
    assert state.blocks_per_slot == 2
    assert state.total_positions == 1024
    assert [buffer.nbytes for buffer in allocations] == [
        conv_nbytes,
        recurrent_nbytes,
        kv_nbytes,
        kv_nbytes,
        conv_nbytes,
        recurrent_nbytes,
    ]
    assert state.linear_state_pair(0) == (state.layer_conv_states[0], state.layer_recurrent_states[0])
    assert state.full_cache(1) == (state.full_key_caches[1], state.full_value_caches[1])
    with pytest.raises(ValueError, match="no packed full-attention"):
        state.full_cache(0)
    with pytest.raises(ValueError, match="no packed linear-attention"):
        state.linear_state_pair(1)


def test_gguf_packed_target_state_rejects_invalid_inputs(monkeypatch) -> None:
    monkeypatch.setattr(gguf_runner, "malloc", lambda nbytes, *, runtime: DeviceBuffer(ptr=1, nbytes=int(nbytes)))
    runner = SimpleNamespace(
        linear_qkv_width=10,
        ssm_value_dim=2,
        weights=SimpleNamespace(
            config=SimpleNamespace(
                layer_types=(LINEAR_ATTENTION,),
                ssm_conv_kernel=4,
                ssm_time_step_rank=2,
                ssm_state_size=3,
                head_count_kv=2,
                key_length=4,
            )
        ),
    )

    with pytest.raises(ValueError, match="slot_count"):
        _GGUFPackedTargetState.allocate(runner, slot_count=0, max_sequence_length=1, runtime=SimpleNamespace())
    with pytest.raises(ValueError, match="max_sequence_length"):
        _GGUFPackedTargetState.allocate(runner, slot_count=1, max_sequence_length=0, runtime=SimpleNamespace())
    with pytest.raises(ValueError, match="block_size"):
        _GGUFPackedTargetState.allocate(
            runner,
            slot_count=1,
            max_sequence_length=1,
            block_size=0,
            runtime=SimpleNamespace(),
        )


def test_gguf_packed_ar_exact_linear_attention_slices_slot_state() -> None:
    cfg = SimpleNamespace(
        hidden_size=8,
        ssm_group_count=1,
        ssm_conv_kernel=4,
        ssm_time_step_rank=2,
        ssm_state_size=2,
        ssm_inner_size=6,
    )
    runner = object.__new__(gguf_runner.Qwen35GGUFFullStackRunner)
    runner.weights = SimpleNamespace(config=cfg)
    conv_row_nbytes = 10 * 4 * 4
    recurrent_row_nbytes = 2 * 2 * 3 * 4

    @dataclass(frozen=True)
    class DecodeScratch:
        layer_conv_states: tuple[DeviceBuffer | None, ...]
        layer_recurrent_states: tuple[DeviceBuffer | None, ...]

    decode_scratch = DecodeScratch(
        layer_conv_states=(DeviceBuffer(0x2000, 2 * conv_row_nbytes),),
        layer_recurrent_states=(DeviceBuffer(0x4000, 2 * recurrent_row_nbytes),),
    )
    scratch = SimpleNamespace(attn_out=DeviceBuffer(0x6000, 2 * 8 * 2))
    attention_calls: list[tuple[int, ...]] = []
    ffn_calls: list[tuple[int, int, int, int]] = []

    def fake_attention(
        self,
        layer_id,
        hidden_ptr,
        attn_out_ptr,
        row_scratch,
        *,
        hidden_f32_ptr=None,
        stream=0,
    ):
        attention_calls.append(
            (
                int(layer_id),
                int(hidden_ptr),
                int(attn_out_ptr),
                int(row_scratch.layer_conv_states[layer_id].ptr),
                int(row_scratch.layer_recurrent_states[layer_id].ptr),
                int(hidden_f32_ptr),
                int(stream),
            )
        )

    def fake_ffn(self, layer_id, hidden_ptr, attn_out_ptr, out_ptr, scratch_arg, *, rows, **kwargs):
        assert scratch_arg is scratch
        ffn_calls.append((int(layer_id), int(hidden_ptr), int(attn_out_ptr), int(rows)))

    runner._run_linear_attention_attn_only = MethodType(fake_attention, runner)
    runner._run_post_attention_ffn_rows = MethodType(fake_ffn, runner)

    runner._run_linear_attention_decode_slot_rows_exact(
        0,
        0x8000,
        0x9000,
        scratch,
        rows=2,
        state_indices=(1, 0),
        decode_scratch=decode_scratch,
        hidden_f32_ptr=0xA000,
        stream=7,
    )

    assert attention_calls == [
        (
            0,
            0x8000,
            0x6000,
            0x2000 + conv_row_nbytes,
            0x4000 + recurrent_row_nbytes,
            0xA000,
            7,
        ),
        (
            0,
            0x8000 + 8 * 2,
            0x6000 + 8 * 2,
            0x2000,
            0x4000,
            0xA000 + 8 * 4,
            7,
        ),
    ]
    assert ffn_calls == [(0, 0x8000, 0x6000, 2)]


def test_gguf_deferred_packed_decode_flush_copies_full_live_kv(monkeypatch) -> None:
    cfg = SimpleNamespace(
        layer_types=(FULL_ATTENTION,),
        head_count_kv=2,
        key_length=4,
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.runner = SimpleNamespace(weights=SimpleNamespace(config=cfg))
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11,), start_position=5),
            _GGUFPackedVerifySlotBlock(input_token_ids=(21,), start_position=7),
        ),
        block_size=4,
    )
    packed_key = DeviceBuffer(0x10000, layout.total_physical_positions * 16)
    packed_value = DeviceBuffer(0x20000, layout.total_physical_positions * 16)
    packed_state = SimpleNamespace(
        blocks_per_slot=layout.blocks_per_slot,
        block_size=layout.block_size,
        full_cache=lambda layer_id: (packed_key, packed_value),
    )

    def slot(slot_id: int):
        key = DeviceBuffer(0x30000 + slot_id * 0x1000, 16 * 32)
        value = DeviceBuffer(0x40000 + slot_id * 0x1000, 16 * 32)
        scratch = SimpleNamespace(
            full_cache=lambda layer_id: (key, value),
            position_host=np.zeros((1,), dtype=np.int64),
            context_host=np.zeros((1,), dtype=np.int64),
            position_buf=DeviceBuffer(0x50000 + slot_id * 0x100, 8),
            context_buf=DeviceBuffer(0x60000 + slot_id * 0x100, 8),
        )
        return SimpleNamespace(
            scratch=scratch,
            _position=0,
            _runtime_state_library=object(),
        )

    sessions = (slot(0), slot(1))

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            self.copies.append((int(dst), int(src), int(nbytes)))

    runtime = FakeRuntime()
    positions: list[int] = []
    monkeypatch.setattr(
        gguf_runner,
        "set_decode_position_i64",
        lambda position_ptr, context_ptr, position, **kwargs: positions.append(int(position)),
    )

    owner._scatter_packed_decode_state(
        sessions,
        layout,
        packed_state,
        runtime=runtime,
        stream=0,
        copy_full_kv=True,
    )

    row_nbytes = 16
    slot_physical_rows = layout.blocks_per_slot * layout.block_size
    assert runtime.copies == [
        (0x30000, 0x10000, 6 * row_nbytes),
        (0x40000, 0x20000, 6 * row_nbytes),
        (
            0x31000,
            0x10000 + slot_physical_rows * row_nbytes,
            8 * row_nbytes,
        ),
        (
            0x41000,
            0x20000 + slot_physical_rows * row_nbytes,
            8 * row_nbytes,
        ),
    ]
    assert positions == [6, 8]
    assert [session._position for session in sessions] == [6, 8]

    runtime.copies.clear()
    positions.clear()
    owner._scatter_packed_decode_state(
        sessions,
        layout,
        packed_state,
        runtime=runtime,
        stream=0,
        copy_kv=False,
    )

    assert runtime.copies == []
    assert positions == [6, 8]


def test_hip_event_stage_recorder_accumulates_aliases_and_closes() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.next_event = 100
            self.records: list[tuple[int, int]] = []
            self.synced: list[int] = []
            self.destroyed: list[int] = []

        def event_create(self) -> int:
            event = self.next_event
            self.next_event += 1
            return event

        def event_record(self, event: int, stream: int = 0) -> None:
            self.records.append((int(event), int(stream)))

        def event_synchronize(self, event: int) -> None:
            self.synced.append(int(event))

        def event_elapsed_time_ms(self, start: int, stop: int) -> float:
            return float(int(stop) - int(start))

        def event_destroy(self, event: int) -> None:
            self.destroyed.append(int(event))

    runtime = FakeRuntime()
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=7)
    recorder.start()
    recorder.mark("first", "total")
    recorder.mark("second")
    timings: dict[str, float] = {}

    recorder.resolve_into(timings)

    assert timings == {"first": 1.0, "total": 1.0, "second": 1.0}
    assert runtime.records == [(100, 7), (101, 7), (102, 7)]
    assert runtime.synced == [101, 102]
    assert sorted(runtime.destroyed) == [100, 101, 102]

    class NegativeElapsedRuntime(FakeRuntime):
        def event_elapsed_time_ms(self, start: int, stop: int) -> float:
            return -0.25

    negative_runtime = NegativeElapsedRuntime()
    negative_recorder = _HipEventStageRecorder(negative_runtime, enabled=True, stream=3)
    negative_recorder.start()
    negative_recorder.mark("bad_interval")
    negative_timings: dict[str, float] = {}

    negative_recorder.resolve_into(negative_timings)

    assert negative_timings == {
        "bad_interval_negative_event_elapsed": 0.25,
        "packed_verify_gpu_negative_event_elapsed": 0.25,
        "bad_interval": 0.0,
    }
