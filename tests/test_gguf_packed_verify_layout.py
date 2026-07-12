from __future__ import annotations

import ctypes
from types import SimpleNamespace

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
