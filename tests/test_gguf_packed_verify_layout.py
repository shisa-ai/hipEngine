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
    Qwen35GGUFPackedPrefillResult,
    _GGUFPackedARAttentionWorkspace,
    _GGUFPackedTargetState,
    _GGUFPackedVerifySlotBlock,
    _GGUFFullAttentionPrefillScratch,
    _HipEventStageRecorder,
    _build_gguf_packed_verify_layout,
    _packed_ar_prefill_linear_state_plan,
    _packed_ar_slot_capacity,
    _packed_decode_metadata_device_eligible,
    _plan_packed_ar_prefill_chunks,
    _scatter_packed_layer_output_hidden,
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


def test_gguf_resident_release_idle_packed_workspace_requires_safe_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        gguf_runner,
        "free",
        lambda buffer, *, runtime: freed.append((int(buffer.ptr), int(buffer.nbytes))),
    )
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.runtime = SimpleNamespace(name="runtime")
    session._packed_ar_attention_workspace = SimpleNamespace(
        buffers=(DeviceBuffer(ptr=0x1000, nbytes=10),)
    )
    session._packed_verify_scratch = SimpleNamespace(
        buffers=(DeviceBuffer(ptr=0x2000, nbytes=20), DeviceBuffer(ptr=0x3000, nbytes=30))
    )
    session._packed_verify_state = SimpleNamespace(
        buffers=(DeviceBuffer(ptr=0x4000, nbytes=40),)
    )
    session._packed_verify_session_ids = (11, 22)
    session._packed_verify_max_written_positions = (4, 4)
    session._packed_decode_sessions = (object(),)
    session._packed_decode_last_layout = object()
    session._packed_decode_state_dirty = True
    session._packed_decode_session_ids = (33,)
    session._packed_decode_positions = (5,)
    session._decode_graphs = []
    session._device_kv_graph_handles = {}

    with pytest.raises(RuntimeError, match="unflushed packed state"):
        session.release_idle_packed_workspace()

    session._packed_decode_state_dirty = False
    live_graph = SimpleNamespace(closed=False)
    session._decode_graphs = [live_graph]
    with pytest.raises(RuntimeError, match="live graph"):
        session.release_idle_packed_workspace()

    live_graph.closed = True
    assert session.release_idle_packed_workspace() == 100
    assert freed == [(0x1000, 10), (0x3000, 30), (0x2000, 20), (0x4000, 40)]
    assert session._packed_ar_attention_workspace is None
    assert session._packed_verify_scratch is None
    assert session._packed_verify_state is None
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


def test_gguf_packed_verify_layout_preserves_inactive_physical_lanes() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11,), start_position=5),
            _GGUFPackedVerifySlotBlock(input_token_ids=(0,), start_position=-1, active=False),
            _GGUFPackedVerifySlotBlock(input_token_ids=(33,), start_position=7),
            _GGUFPackedVerifySlotBlock(input_token_ids=(0,), start_position=-1, active=False),
        ),
        block_size=4,
        slot_capacity=12,
    )

    np.testing.assert_array_equal(layout.active_mask, np.asarray([True, False, True, False]))
    np.testing.assert_array_equal(layout.input_token_ids, np.asarray([11, 0, 33, 0], dtype=np.int64))
    np.testing.assert_array_equal(layout.row_positions, np.asarray([5, -1, 7, -1], dtype=np.int64))
    np.testing.assert_array_equal(layout.live_counts, np.asarray([6, 0, 8, 0], dtype=np.int64))
    np.testing.assert_array_equal(layout.cu_seqlens, np.arange(5, dtype=np.int32))
    np.testing.assert_array_equal(layout.state_indices, np.arange(4, dtype=np.int64))
    np.testing.assert_array_equal(layout.block_table[1], np.full((3,), -1, dtype=np.int32))
    np.testing.assert_array_equal(layout.block_table[3], np.full((3,), -1, dtype=np.int32))
    assert layout.rows == 4
    assert layout.slot_count == 4
    assert layout.max_live_count == 8


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


def test_gguf_packed_ar_slot_capacity_rounds_without_per_token_reallocation() -> None:
    assert _packed_ar_slot_capacity(1) == 1024
    assert _packed_ar_slot_capacity(1024) == 1024
    assert _packed_ar_slot_capacity(1025) == 1280
    assert _packed_ar_slot_capacity(1279) == 1280
    assert _packed_ar_slot_capacity(1280) == 1280
    assert _packed_ar_slot_capacity(1281) == 1536


@pytest.mark.parametrize("max_live_count", (0, -1))
def test_gguf_packed_ar_slot_capacity_rejects_invalid_context(max_live_count: int) -> None:
    with pytest.raises(ValueError, match="max_live_count"):
        _packed_ar_slot_capacity(max_live_count)


def test_gguf_packed_decode_device_metadata_requires_singleton_c4_layout() -> None:
    singleton = _build_gguf_packed_verify_layout(
        tuple(
            _GGUFPackedVerifySlotBlock(input_token_ids=(slot + 1,), start_position=position)
            for slot, position in enumerate((513, 517, 521, 525))
        ),
        slot_capacity=1024,
    )
    multi_token = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(1, 2), start_position=4),
            _GGUFPackedVerifySlotBlock(input_token_ids=(3, 4), start_position=8),
        ),
        slot_capacity=1024,
    )

    assert _packed_decode_metadata_device_eligible(singleton)
    assert not _packed_decode_metadata_device_eligible(multi_token)


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


def test_gguf_packed_layer_output_hidden_scatter_selects_slot_rows() -> None:
    sessions = tuple(SimpleNamespace(_last_layer_output_hidden={}) for _ in range(3))
    hidden = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)

    _scatter_packed_layer_output_hidden(
        sessions,
        layer_id=7,
        hidden_rows=hidden,
        row_indices=(1, 3, 5),
    )

    for session, row_index in zip(sessions, (1, 3, 5), strict=True):
        np.testing.assert_array_equal(
            session._last_layer_output_hidden[7],
            hidden[row_index : row_index + 1],
        )
        assert session._last_layer_output_hidden[7].shape == (1, 4)


def test_gguf_packed_ar_prefill_chunks_all_row_c4_without_slot_serialization() -> None:
    prompts = tuple(
        tuple(slot * 1000 + row for row in range(512))
        for slot in range(4)
    )

    chunks = _plan_packed_ar_prefill_chunks(prompts, row_capacity=768)

    assert [chunk.rows for chunk in chunks] == [768, 768, 512]
    assert [chunk.slot_indices for chunk in chunks] == [
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
    ]
    assert [tuple(len(tokens) for tokens in chunk.prompt_token_ids) for chunk in chunks] == [
        (192, 192, 192, 192),
        (192, 192, 192, 192),
        (128, 128, 128, 128),
    ]
    assert [chunk.start_offsets for chunk in chunks] == [
        (0, 0, 0, 0),
        (192, 192, 192, 192),
        (384, 384, 384, 384),
    ]
    for slot_index, prompt in enumerate(prompts):
        reconstructed = tuple(
            token
            for chunk in chunks
            for chunk_slot, tokens in zip(
                chunk.slot_indices,
                chunk.prompt_token_ids,
                strict=True,
            )
            if chunk_slot == slot_index
            for token in tokens
        )
        assert reconstructed == prompt


def test_gguf_packed_ar_prefill_chunk_plan_preserves_fitting_ragged_slab() -> None:
    prompts = (
        (1,) * 512,
        (2,) * 64,
        (3,) * 64,
        (4,) * 64,
    )

    chunks = _plan_packed_ar_prefill_chunks(prompts, row_capacity=768)

    assert len(chunks) == 1
    assert chunks[0].slot_indices == (0, 1, 2, 3)
    assert chunks[0].start_offsets == (0, 0, 0, 0)
    assert chunks[0].prompt_token_ids == prompts
    assert chunks[0].rows == 704


def test_gguf_packed_ar_prefill_chunk_plan_refuses_dropping_active_slots() -> None:
    with pytest.raises(ValueError, match="active slots"):
        _plan_packed_ar_prefill_chunks(((1,), (2,), (3,)), row_capacity=2)


def test_gguf_packed_ar_prefill_executes_each_round_with_all_active_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    routes: list[tuple[bool | None, tuple[int, ...]]] = []
    sessions = tuple(SimpleNamespace(position=0) for _ in range(4))
    session_index = {id(session): index for index, session in enumerate(sessions)}

    def fake_single_slab(self, prompt_token_ids, *, sessions, **kwargs):
        prompt_tuple = tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids)
        slot_indices = tuple(session_index[id(session)] for session in sessions)
        calls.append((slot_indices, prompt_tuple))
        routes.append(
            (
                kwargs.get("_slot_local_full_attention"),
                tuple(kwargs.get("_force_aotriton_slot_indices", ())),
            )
        )
        for session, prompt in zip(sessions, prompt_tuple, strict=True):
            session.position += len(prompt)
        return [SimpleNamespace(token_id=int(prompt[-1])) for prompt in prompt_tuple]

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_prefill_batch_native_single_slab",
        fake_single_slab,
        raising=False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "PrefillConfig",
        lambda: SimpleNamespace(attn_aotriton_min_tokens=4),
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._bulk_prefill_scratch = SimpleNamespace(rows=8)
    prompts = tuple(tuple(slot * 100 + row for row in range(6)) for slot in range(4))

    results = owner.prefill_batch_native(prompts, sessions=sessions)

    assert [slots for slots, _ in calls] == [(0, 1, 2, 3)] * 3
    assert [[len(prompt) for prompt in chunk] for _, chunk in calls] == [[2, 2, 2, 2]] * 3
    assert routes == [(True, (0, 1, 2, 3))] * 3
    assert [result.token_id for result in results] == [prompt[-1] for prompt in prompts]
    assert [session.position for session in sessions] == [6, 6, 6, 6]
    assert owner.last_packed_prefill_plan["chunk_rows"] == [8, 8, 8]
    assert owner.last_packed_prefill_plan["slot_indices"] == [[0, 1, 2, 3]] * 3
    assert owner.last_packed_prefill_plan["all_active_slots_represented"] is True


def test_gguf_packed_ar_prefill_preserves_full_prompt_attention_route_across_scheduler_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes: list[tuple[bool | None, tuple[int, ...]]] = []
    session = SimpleNamespace(position=0)

    def fake_single_slab(self, prompt_token_ids, *, sessions, **kwargs):
        routes.append(
            (
                kwargs.get("_slot_local_full_attention"),
                tuple(kwargs.get("_force_aotriton_slot_indices", ())),
            )
        )
        prompt = tuple(int(token) for token in prompt_token_ids[0])
        sessions[0].position += len(prompt)
        return [SimpleNamespace(token_id=prompt[-1])]

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_prefill_batch_native_single_slab",
        fake_single_slab,
        raising=False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "PrefillConfig",
        lambda: SimpleNamespace(attn_aotriton_min_tokens=4),
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._bulk_prefill_scratch = SimpleNamespace(rows=8)

    for chunk in ((10, 11), (12, 13), (14, 15)):
        owner.prefill_batch_native(
            (chunk,),
            sessions=(session,),
            full_prompt_lengths=(6,),
        )

    assert routes == [(True, (0,))] * 3
    assert session.position == 6
    assert owner.last_packed_prefill_plan["full_prompt_lengths"] == [6]
    assert owner.last_packed_prefill_plan["aotriton_eligible_slots"] == [0]
    assert owner.last_packed_prefill_plan["aotriton_eligibility_preserved_across_chunks"] is True


def test_gguf_packed_ar_prefill_concatenates_hidden_seed_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = tuple(SimpleNamespace(position=position) for position in (3, 5))

    def fake_single_slab(self, prompt_token_ids, *, sessions, return_hidden_seeds, **kwargs):
        assert return_hidden_seeds
        results = []
        for session, prompt in zip(sessions, prompt_token_ids, strict=True):
            start = int(session.position)
            tokens = [int(token) for token in prompt]
            hidden = np.repeat(np.asarray(tokens, dtype=np.float32)[:, None], 3, axis=1)
            session.position += len(tokens)
            results.append(
                Qwen35GGUFPackedPrefillResult(
                    input_token_ids=tokens,
                    token_id=tokens[-1] + 1,
                    hidden_seeds=hidden,
                    start_position=start,
                )
            )
        return results

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_prefill_batch_native_single_slab",
        fake_single_slab,
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._bulk_prefill_scratch = SimpleNamespace(rows=4)
    prompts = ((10, 11, 12, 13), (20, 21, 22, 23))

    results = owner.prefill_batch_native(
        prompts,
        sessions=sessions,
        return_hidden_seeds=True,
    )

    assert [result.input_token_ids for result in results] == [list(prompt) for prompt in prompts]
    assert [result.start_position for result in results] == [3, 5]
    assert [result.token_id for result in results] == [14, 24]
    for result, prompt in zip(results, prompts, strict=True):
        np.testing.assert_array_equal(
            result.hidden_seeds,
            np.repeat(np.asarray(prompt, dtype=np.float32)[:, None], 3, axis=1),
        )


def test_gguf_packed_ar_prefill_keeps_only_final_segment_state() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(1,) * 512, start_position=0),
            _GGUFPackedVerifySlotBlock(input_token_ids=(2,) * 64, start_position=0),
            _GGUFPackedVerifySlotBlock(input_token_ids=(3,) * 64, start_position=0),
            _GGUFPackedVerifySlotBlock(input_token_ids=(4,) * 64, start_position=0),
        )
    )

    plan = _packed_ar_prefill_linear_state_plan(layout)

    assert layout.rows == 704
    assert plan.route == "segmented_in_place_final_state"
    assert plan.state_slots == 4
    assert plan.transient_state_rows == 0
    assert not plan.capture_token_state_rows
    assert not plan.commit_captured_state_rows


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
        segments=4,
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
    assert view.metadata_prepare_path == "host_upload"

    device_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_device_prepare(*args, **kwargs):
        device_calls.append((args, kwargs))

    singleton_layout = _build_gguf_packed_verify_layout(
        tuple(
            _GGUFPackedVerifySlotBlock(input_token_ids=(slot + 1,), start_position=position)
            for slot, position in enumerate((513, 517, 521, 525))
        ),
        slot_capacity=1024,
    )
    copies.clear()

    device_view = scratch.for_packed_verify_layout(
        singleton_layout,
        runtime=SimpleNamespace(),
        stream=7,
        metadata_prepare_fn=fake_device_prepare,
    )

    assert copies == {}
    assert len(device_calls) == 1
    args, kwargs = device_calls[0]
    assert args[:8] == (
        scratch.block_table.ptr,
        scratch.positions.ptr,
        scratch.context_counts.ptr,
        scratch.cu_q.ptr,
        scratch.cu_k.ptr,
        scratch.atomic.ptr,
        scratch.gdn_cu_seqlens.ptr,
        scratch.gdn_state_indices.ptr,
    )
    assert args[8] == (513, 517, 521, 525)
    assert args[9] == 4
    assert kwargs["stream"] == 7
    assert device_view.metadata_prepare_path == "device_prepare_persistent"


def test_gguf_packed_ar_attention_workspace_sizes_rows_and_long_context_splits(
    monkeypatch,
) -> None:
    next_ptr = 0x180000
    allocations: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal next_ptr
        buffer = DeviceBuffer(ptr=next_ptr, nbytes=int(nbytes))
        next_ptr += max(8, int(nbytes) + 8)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    runner = SimpleNamespace(
        q_width=4096,
        weights=SimpleNamespace(config=SimpleNamespace(head_count=16)),
    )

    workspace = _GGUFPackedARAttentionWorkspace.allocate(
        runner,
        rows=2,
        max_context_len=1025,
        runtime=SimpleNamespace(),
    )

    assert workspace.rows == 2
    assert workspace.chunk_size == 256
    assert workspace.num_splits == 5
    assert [buffer.nbytes for buffer in allocations] == [
        2 * 4096 * 5 * 4,
        2 * 16 * 5 * 4,
        2 * 16 * 5 * 4,
    ]
    assert workspace.buffers == tuple(allocations)


@pytest.mark.parametrize(
    ("rows", "max_context_len", "match"),
    ((0, 1, "rows"), (1, 0, "max_context_len")),
)
def test_gguf_packed_ar_attention_workspace_rejects_invalid_shape(
    monkeypatch, rows: int, max_context_len: int, match: str
) -> None:
    monkeypatch.setattr(
        gguf_runner,
        "malloc",
        lambda nbytes, *, runtime: DeviceBuffer(ptr=1, nbytes=int(nbytes)),
    )
    runner = SimpleNamespace(
        q_width=4096,
        weights=SimpleNamespace(config=SimpleNamespace(head_count=16)),
    )

    with pytest.raises(ValueError, match=match):
        _GGUFPackedARAttentionWorkspace.allocate(
            runner,
            rows=rows,
            max_context_len=max_context_len,
            runtime=SimpleNamespace(),
        )


def test_gguf_packed_workspace_resize_scatters_deferred_decode_state_first(
    monkeypatch,
) -> None:
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.runner = SimpleNamespace()
    owner._packed_verify_state = SimpleNamespace(slot_count=1, max_sequence_length=1024)
    owner._packed_verify_scratch = SimpleNamespace(
        rows=1,
        max_positions=1024,
        gdn_segment_capacity=1,
    )
    owner._packed_decode_state_dirty = True
    events: list[tuple[object, ...]] = []

    def fake_flush(self, *, stream=0):
        events.append(("flush", int(stream)))
        self._packed_decode_state_dirty = False
        return True

    def fake_free(self, *, runtime):
        events.append(("free", runtime))
        self._packed_verify_state = None
        self._packed_verify_scratch = None

    owner.flush_packed_decode_state = MethodType(fake_flush, owner)
    owner._free_packed_verify_workspace = MethodType(fake_free, owner)
    new_state = SimpleNamespace(name="state")
    new_scratch = SimpleNamespace(name="scratch")
    monkeypatch.setattr(
        gguf_runner._GGUFPackedTargetState,
        "allocate",
        lambda *args, **kwargs: new_state,
    )
    monkeypatch.setattr(
        gguf_runner._GGUFFullAttentionPrefillScratch,
        "allocate",
        lambda *args, **kwargs: new_scratch,
    )
    runtime = SimpleNamespace(name="runtime")

    state, scratch = owner._ensure_packed_verify_workspace(
        slot_count=1,
        rows=1,
        max_sequence_length=1280,
        runtime=runtime,
        stream=7,
    )

    assert (state, scratch) == (new_state, new_scratch)
    assert events == [("flush", 7), ("free", runtime)]


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


def test_gguf_packed_ar_exact_linear_attention_dispatches_indexed_batch_plan() -> None:
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
    batch_calls: list[tuple[int, ...]] = []
    ffn_calls: list[tuple[int, int, int, int]] = []
    batch_plan = SimpleNamespace(
        available=True,
        conv_indexed=object(),
        gdn_segments=object(),
    )

    def reject_scalar(*args, **kwargs):
        raise AssertionError("indexed batch plan must not replay scalar attention rows")

    def fake_indexed_attention(
        self,
        layer_id,
        hidden_ptr,
        attn_out_ptr,
        scratch_arg,
        *,
        rows,
        decode_scratch,
        batch_plan,
        gdn_cu_seqlens_ptr,
        state_indices_ptr,
        hidden_f32_ptr=None,
        stream=0,
    ):
        assert scratch_arg is scratch
        batch_calls.append(
            (
                int(layer_id),
                int(hidden_ptr),
                int(attn_out_ptr),
                int(rows),
                int(decode_scratch.layer_conv_states[layer_id].ptr),
                int(decode_scratch.layer_recurrent_states[layer_id].ptr),
                int(gdn_cu_seqlens_ptr),
                int(state_indices_ptr),
                int(hidden_f32_ptr),
                int(stream),
            )
        )

    def fake_ffn(self, layer_id, hidden_ptr, attn_out_ptr, out_ptr, scratch_arg, *, rows, **kwargs):
        assert scratch_arg is scratch
        ffn_calls.append((int(layer_id), int(hidden_ptr), int(attn_out_ptr), int(rows)))

    runner._run_linear_attention_attn_only = MethodType(reject_scalar, runner)
    runner._run_linear_attention_attn_rows_indexed_exact = MethodType(
        fake_indexed_attention,
        runner,
    )
    runner._run_post_attention_ffn_rows = MethodType(fake_ffn, runner)

    path = runner._run_linear_attention_decode_slot_rows_exact(
        0,
        0x8000,
        0x9000,
        scratch,
        rows=2,
        state_indices=(1, 0),
        decode_scratch=decode_scratch,
        batch_plan=batch_plan,
        gdn_cu_seqlens_ptr=0xA000,
        state_indices_ptr=0xB000,
        hidden_f32_ptr=0xC000,
        stream=7,
    )

    assert path == "indexed_batch"
    assert batch_calls == [
        (0, 0x8000, 0x6000, 2, 0x2000, 0x4000, 0xA000, 0xB000, 0xC000, 7)
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


def test_gguf_contiguous_device_kv_cache_view_rebases_shifted_allocation() -> None:
    cache = DeviceBuffer(0x10000, 8 * 256 * 16)
    unbound = SimpleNamespace(_device_kv_allocation=None)
    identity = SimpleNamespace(
        _device_kv_allocation=SimpleNamespace(
            block_ids=(8, 9, 10),
            chunk_start_block_id=8,
        )
    )
    shifted = SimpleNamespace(
        _device_kv_allocation=SimpleNamespace(
            block_ids=(11, 12, 13),
            chunk_start_block_id=8,
        )
    )
    noncontiguous = SimpleNamespace(
        _device_kv_allocation=SimpleNamespace(
            block_ids=(8, 10),
            chunk_start_block_id=8,
        )
    )

    assert gguf_runner._gguf_device_kv_contiguous_base_row(unbound) == 0
    assert gguf_runner._gguf_device_kv_contiguous_base_row(identity) == 0
    assert gguf_runner._gguf_device_kv_contiguous_base_row(shifted) == 3 * 256
    assert gguf_runner._gguf_device_kv_contiguous_base_row(noncontiguous) is None
    assert gguf_runner._gguf_device_kv_contiguous_cache_view(
        unbound,
        cache,
        row_nbytes=16,
    ) is cache
    assert gguf_runner._gguf_device_kv_contiguous_cache_view(
        identity,
        cache,
        row_nbytes=16,
    ) is cache
    shifted_view = gguf_runner._gguf_device_kv_contiguous_cache_view(
        shifted,
        cache,
        row_nbytes=16,
    )
    assert shifted_view == DeviceBuffer(
        0x10000 + 3 * 256 * 16,
        5 * 256 * 16,
    )
    assert gguf_runner._gguf_device_kv_contiguous_cache_view(
        noncontiguous,
        cache,
        row_nbytes=16,
    ) is None


def test_gguf_deferred_packed_state_scatter_follows_noncontiguous_device_pages(
    monkeypatch,
) -> None:
    cfg = SimpleNamespace(
        layer_types=(FULL_ATTENTION,),
        head_count_kv=2,
        key_length=4,
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.runner = SimpleNamespace(weights=SimpleNamespace(config=cfg))
    layout = _build_gguf_packed_verify_layout(
        (_GGUFPackedVerifySlotBlock(input_token_ids=(11, 12), start_position=255),),
        block_size=256,
    )
    row_nbytes = 16
    packed_key = DeviceBuffer(0x10000, layout.total_physical_positions * row_nbytes)
    packed_value = DeviceBuffer(0x20000, layout.total_physical_positions * row_nbytes)
    packed_state = SimpleNamespace(
        blocks_per_slot=layout.blocks_per_slot,
        block_size=layout.block_size,
        full_cache=lambda layer_id: (packed_key, packed_value),
    )
    key = DeviceBuffer(0x30000, 3 * 256 * row_nbytes)
    value = DeviceBuffer(0x40000, 3 * 256 * row_nbytes)
    scratch = SimpleNamespace(
        full_cache=lambda layer_id: (key, value),
        position_host=np.zeros((1,), dtype=np.int64),
        context_host=np.zeros((1,), dtype=np.int64),
        position_buf=DeviceBuffer(0x50000, 8),
        context_buf=DeviceBuffer(0x60000, 8),
    )
    session = SimpleNamespace(
        scratch=scratch,
        _position=255,
        _runtime_state_library=object(),
        _device_kv_allocation=SimpleNamespace(
            block_ids=(8, 10),
            chunk_start_block_id=8,
        ),
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            self.copies.append((int(dst), int(src), int(nbytes)))

    runtime = FakeRuntime()
    monkeypatch.setattr(gguf_runner, "set_decode_position_i64", lambda *args, **kwargs: None)

    owner._scatter_packed_decode_state(
        (session,),
        layout,
        packed_state,
        runtime=runtime,
        stream=0,
    )

    assert runtime.copies == [
        (0x30000 + 255 * row_nbytes, 0x10000 + 255 * row_nbytes, row_nbytes),
        (0x40000 + 255 * row_nbytes, 0x20000 + 255 * row_nbytes, row_nbytes),
        (0x30000 + 512 * row_nbytes, 0x10000 + 256 * row_nbytes, row_nbytes),
        (0x40000 + 512 * row_nbytes, 0x20000 + 256 * row_nbytes, row_nbytes),
    ]
    assert session._position == 257


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
