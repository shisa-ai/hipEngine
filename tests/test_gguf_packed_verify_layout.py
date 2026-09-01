from __future__ import annotations

import ctypes
import inspect
from dataclasses import dataclass
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen35_gguf_runner as gguf_runner
from hipengine.core.dtype import DType
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
    _HipWallClockStageRecorder,
    _build_gguf_packed_verify_layout,
    _gguf_device_kv_copy_segments,
    _packed_ar_prefill_linear_state_plan,
    _packed_ar_slot_capacity,
    _packed_decode_metadata_device_eligible,
    _plan_packed_ar_prefill_chunks,
    _scatter_packed_layer_output_hidden,
)


def test_private_resident_slot_kv_copy_segments_include_slot_base() -> None:
    owner = SimpleNamespace(
        _target_scratch_owner=SimpleNamespace(max_positions=1024)
    )
    slot = SimpleNamespace(
        _device_kv_allocation=None,
        _resident_batch_owner=owner,
        _resident_slot_index=2,
    )

    assert _gguf_device_kv_copy_segments(
        slot,
        start_position=37,
        rows=5,
    ) == ((37, 2 * 1024 + 37, 5),)
    assert _gguf_device_kv_copy_segments(
        SimpleNamespace(_device_kv_allocation=None),
        start_position=37,
        rows=5,
    ) == ((37, 37, 5),)


def test_long_packed_prefill_requires_slot_local_full_attention() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(
                input_token_ids=(17,),
                start_position=1023,
            ),
        ),
        slot_capacity=1024,
    )

    gguf_runner._validate_packed_ar_prefill_context(
        layout,
        slot_local_full_prefill=True,
    )
    with pytest.raises(NotImplementedError, match="packed paged AR prefill"):
        gguf_runner._validate_packed_ar_prefill_context(
            layout,
            slot_local_full_prefill=False,
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
    session._target_scratch_owner = object()
    session._reset_current_slot_only = True
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
    flush_calls: list[int] = []

    def fake_flush(self, *, stream: int = 0):
        flush_calls.append(int(stream))
        return True

    session.flush_packed_decode_state = MethodType(fake_flush, session)

    session.reset(stream=7)

    assert calls == [
        ("zero_states", session.runtime, 7, False),
        ("set_position", 0, 7),
    ]
    # Deferred packed decode state is scattered back before the shared
    # bookkeeping is cleared (other resident rows may still need it).
    assert flush_calls == [0]
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


def test_gguf_direct_resident_linear_state_maps_owner_slots(monkeypatch) -> None:
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.backend = "hip_gfx1151"
    slab = SimpleNamespace(slot_count=17)
    owner._target_scratch_owner = slab
    peer = SimpleNamespace(
        _resident_batch_owner=owner,
        _target_scratch_owner=slab,
        _resident_slot_index=5,
    )
    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default: name == "GGUF_DIRECT_RESIDENT_LINEAR_STATE",
    )

    assert owner._direct_resident_linear_state((owner, peer, None)) == (
        (0, 5, 0),
        slab,
    )
    assert owner._direct_resident_linear_state((SimpleNamespace(),)) is None


def test_gguf_packed_verify_initial_state_uses_fused_pair_copy_when_enabled() -> None:
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.runner = SimpleNamespace(
        weights=SimpleNamespace(config=SimpleNamespace(layer_types=(LINEAR_ATTENTION,)))
    )
    owner._packed_verify_session_ids = ()
    owner._packed_verify_max_written_positions = ()
    fused_calls = []
    owner._fused_packed_verify_initial_state_transfer_enabled = MethodType(
        lambda self: True, owner
    )
    owner._fused_linear_state_pair_copy = MethodType(
        lambda self, copies, **kwargs: fused_calls.append((tuple(copies), kwargs)) or True,
        owner,
    )
    sessions = tuple(
        object.__new__(gguf_runner.Qwen35GGUFResidentSession) for _ in range(2)
    )
    for session, position, conv_ptr, recurrent_ptr in zip(
        sessions,
        (5, 7),
        (0x1000, 0x1100),
        (0x2000, 0x2100),
        strict=True,
    ):
        session.runner = owner.runner
        session._position = position
        session.scratch = SimpleNamespace(
            layer_conv_states=(DeviceBuffer(conv_ptr, 64),),
            layer_recurrent_states=(DeviceBuffer(recurrent_ptr, 128),),
        )
    jobs = [{"session": session} for session in sessions]
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11,), start_position=5),
            _GGUFPackedVerifySlotBlock(input_token_ids=(12,), start_position=7),
        ),
        block_size=4,
        slot_capacity=16,
    )
    packed_state = SimpleNamespace(
        linear_state_pair=lambda layer_id: (
            DeviceBuffer(0x3000, 128),
            DeviceBuffer(0x4000, 256),
        )
    )
    runtime = SimpleNamespace(memcpy_async=lambda *args: pytest.fail("unfused copy"))

    owner._sync_packed_verify_initial_state(
        jobs,
        layout,
        packed_state,
        runtime=runtime,
        stream=7,
    )

    assert fused_calls == [
        (
            (
                (0x1000, 0x3000, 0x2000, 0x4000, 64, 128),
                (0x1100, 0x3040, 0x2100, 0x4080, 64, 128),
            ),
            {"runtime": runtime, "stream": 7},
        )
    ]


def test_gguf_fused_linear_state_pair_copy_batches_and_caches_tables(monkeypatch) -> None:
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._buffers = ()
    owner.runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(layer_types=(LINEAR_ATTENTION,))
        )
    )
    owner._dflash_commit_library = None
    owner.compiler_version = "test"
    owner.require_cached_build = True
    owner._verify_linear_state_src_conv_table_buf = None
    owner._verify_linear_state_src_recurrent_table_buf = None
    owner._verify_linear_state_dst_conv_table_buf = None
    owner._verify_linear_state_dst_recurrent_table_buf = None
    owner._verify_linear_state_commit_row_i32_buf = None
    owner._verify_linear_state_src_conv_host = None
    owner._verify_linear_state_src_recurrent_host = None
    owner._verify_linear_state_src_conv_cached = None
    owner._verify_linear_state_src_recurrent_cached = None
    owner._verify_linear_state_dst_conv_host = None
    owner._verify_linear_state_dst_recurrent_host = None
    owner._verify_linear_state_layer_count = 0
    owner._verify_linear_state_conv_row_nbytes = 0
    owner._verify_linear_state_recurrent_row_nbytes = 0
    owner._fused_linear_state_transfer_enabled = MethodType(lambda self: True, owner)
    owner._chunked_linear_state_commit_enabled = MethodType(lambda self: False, owner)

    allocations: list[DeviceBuffer] = []
    uploads: list[tuple[int, int]] = []
    launches: list[tuple[int, int, int]] = []

    def fake_malloc(nbytes, *, runtime):
        del runtime
        buffer = DeviceBuffer(0x10000 + len(allocations) * 0x1000, int(nbytes))
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(
        gguf_runner,
        "copy_host_to_device",
        lambda buffer, host_ptr, nbytes, *, runtime: uploads.append((int(buffer.ptr), int(nbytes))),
    )
    monkeypatch.setattr(gguf_runner, "build_dflash_commit", lambda **kwargs: object())
    monkeypatch.setattr(
        gguf_runner,
        "linear_state_pair_commit_i32",
        lambda *args, **kwargs: launches.append((int(args[2]), int(args[5]), int(args[7]))),
    )
    copies = [
        (0x10, 0x20, 0x30, 0x40, 64, 128),
        (0x50, 0x60, 0x70, 0x80, 64, 128),
    ]

    assert owner._fused_linear_state_pair_copy(copies, runtime=object(), stream=7)
    assert launches == [(64, 128, 2)]
    assert len(allocations) == 5
    assert len(uploads) == 5

    assert owner._fused_linear_state_pair_copy(copies, runtime=object(), stream=7)
    assert launches == [(64, 128, 2), (64, 128, 2)]
    assert len(allocations) == 5
    assert len(uploads) == 7

def test_gfx1100_fused_linear_state_transfer_policy_is_default_off(
    monkeypatch,
) -> None:
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.backend = "hip_gfx1100"

    monkeypatch.delenv("HIPENGINE_GGUF_FUSED_PACKED_STATE_TRANSFER", raising=False)
    assert owner._fused_linear_state_transfer_enabled() is False
    monkeypatch.setenv("HIPENGINE_GGUF_FUSED_PACKED_STATE_TRANSFER", "1")
    assert owner._fused_linear_state_transfer_enabled() is True
    assert owner._fused_packed_verify_initial_state_transfer_enabled() is True
    monkeypatch.setenv("HIPENGINE_GGUF_FUSED_PACKED_STATE_TRANSFER", "0")
    assert owner._fused_linear_state_transfer_enabled() is False
    assert owner._fused_packed_verify_initial_state_transfer_enabled() is False


def test_gguf_single_slot_state_import_uses_strict_unfused_copy() -> None:
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.backend = "hip_gfx1151"
    owner.runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(layer_types=(LINEAR_ATTENTION,))
        )
    )
    owner._fused_linear_state_transfer_enabled = MethodType(lambda self: True, owner)

    assert not owner._fused_linear_state_pair_copy(
        [(0x10, 0x20, 0x30, 0x40, 64, 128)],
        runtime=object(),
        stream=0,
    )


def test_gguf_resident_discards_terminal_packed_decode_state_without_scatter() -> None:
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session._packed_decode_sessions = (object(),)
    session._packed_decode_last_layout = object()
    session._packed_decode_state_dirty = True
    session._packed_decode_session_ids = (33,)
    session._packed_decode_positions = (5,)

    assert session.discard_packed_decode_state() is True
    assert session._packed_decode_sessions == ()
    assert session._packed_decode_last_layout is None
    assert session._packed_decode_state_dirty is False
    assert session._packed_decode_session_ids == ()
    assert session._packed_decode_positions == ()
    assert session.discard_packed_decode_state() is False


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


def test_gguf_packed_ar_prefill_forwards_unsampled_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_flags: list[bool] = []
    sessions = tuple(SimpleNamespace(position=0) for _ in range(2))

    def fake_single_slab(self, prompt_token_ids, *, sessions, sample_output, **kwargs):
        del self, kwargs
        sample_flags.append(bool(sample_output))
        for session, prompt in zip(sessions, prompt_token_ids, strict=True):
            session.position += len(prompt)
        return [None for _ in prompt_token_ids]

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_prefill_batch_native_single_slab",
        fake_single_slab,
        raising=False,
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._bulk_prefill_scratch = SimpleNamespace(rows=4)

    results = owner.prefill_batch_native(
        ((10, 11, 12, 13), (20, 21, 22, 23)),
        sessions=sessions,
        sample_output=False,
    )

    assert results == [None, None]
    assert sample_flags == [False, False]
    assert owner.last_packed_prefill_plan["sample_output"] is False
    assert owner.last_packed_prefill_plan["output_norm_rows"] == 0
    assert owner.last_packed_prefill_plan["lm_head_sample_rows"] == 0


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
        ssm_group_count=2,
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
        conv_nbytes,
        recurrent_nbytes,
        kv_nbytes,
        kv_nbytes,
    ]
    assert state.linear_state_pair(0) == (state.layer_conv_states[0], state.layer_recurrent_states[0])
    assert state.full_cache(1) == (state.full_key_caches[1], state.full_value_caches[1])
    with pytest.raises(ValueError, match="no packed full-attention"):
        state.full_cache(0)
    with pytest.raises(ValueError, match="no packed linear-attention"):
        state.linear_state_pair(1)


def test_gguf_packed_target_state_allocates_mirrored_int8_payload_and_scales(monkeypatch) -> None:
    next_ptr = 0x400000
    allocations: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal next_ptr
        buffer = DeviceBuffer(ptr=next_ptr, nbytes=int(nbytes))
        next_ptr += int(nbytes) + 8
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    cfg = SimpleNamespace(
        layer_types=(FULL_ATTENTION,),
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
    kv_layout = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.INT8_PER_TOKEN_HEAD,),
        bf16_mirror_layer_indices=(0,),
    )

    state = _GGUFPackedTargetState.allocate(
        runner,
        slot_count=2,
        max_sequence_length=300,
        runtime=SimpleNamespace(),
        kv_layout=kv_layout,
    )

    assert state.total_positions == 1024
    assert [buffer.nbytes for buffer in allocations] == [
        8192,
        8192,
        16384,
        16384,
        4096,
        4096,
    ]
    assert state.full_bf16_mirror_cache(0) == (
        state.full_bf16_mirror_key_caches[0],
        state.full_bf16_mirror_value_caches[0],
    )
    metadata = state.full_scale_metadata(0)
    assert metadata is not None
    assert metadata.k_scale.shape == (4, 256, 2)
    assert metadata.v_scale.shape == (4, 256, 2)


def test_gguf_packed_int8_copy_moves_payload_mirror_and_scale_planes() -> None:
    cfg = SimpleNamespace(layer_types=(FULL_ATTENTION,), head_count_kv=2, key_length=4)
    layout = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.INT8_PER_TOKEN_HEAD,),
        bf16_mirror_layer_indices=(0,),
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.runner = SimpleNamespace(weights=SimpleNamespace(config=cfg))
    owner._device_kv_layout = layout

    def state(base: int):
        return SimpleNamespace(
            kv_layout=layout,
            full_key_caches=(DeviceBuffer(base + 0x0000, 8192),),
            full_value_caches=(DeviceBuffer(base + 0x1000, 8192),),
            full_bf16_mirror_key_caches=(DeviceBuffer(base + 0x2000, 16384),),
            full_bf16_mirror_value_caches=(DeviceBuffer(base + 0x3000, 16384),),
            full_k_scale_caches=(DeviceBuffer(base + 0x4000, 4096),),
            full_v_scale_caches=(DeviceBuffer(base + 0x5000, 4096),),
        )

    source = state(0x100000)
    destination = state(0x200000)

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            self.copies.append((int(dst), int(src), int(nbytes)))

    runtime = FakeRuntime()
    owner._copy_packed_kv_rows(
        source,
        destination,
        0,
        source_start=3,
        destination_start=7,
        rows=2,
        runtime=runtime,
        stream=9,
    )

    assert runtime.copies == [
        (destination.full_key_caches[0].ptr + 7 * 8, source.full_key_caches[0].ptr + 3 * 8, 2 * 8),
        (destination.full_value_caches[0].ptr + 7 * 8, source.full_value_caches[0].ptr + 3 * 8, 2 * 8),
        (
            destination.full_bf16_mirror_key_caches[0].ptr + 7 * 16,
            source.full_bf16_mirror_key_caches[0].ptr + 3 * 16,
            2 * 16,
        ),
        (
            destination.full_bf16_mirror_value_caches[0].ptr + 7 * 16,
            source.full_bf16_mirror_value_caches[0].ptr + 3 * 16,
            2 * 16,
        ),
        (destination.full_k_scale_caches[0].ptr + 7 * 4, source.full_k_scale_caches[0].ptr + 3 * 4, 2 * 4),
        (destination.full_v_scale_caches[0].ptr + 7 * 4, source.full_v_scale_caches[0].ptr + 3 * 4, 2 * 4),
    ]


@dataclass(frozen=True)
class _DirectInt8PrefillSpans:
    kind: str = "unknown"
    storage_dtype: DType = DType.BF16
    scale_metadata: object | None = None


@dataclass(frozen=True)
class _DirectInt8PrefillScratch:
    append_spans: _DirectInt8PrefillSpans
    prefill_spans: _DirectInt8PrefillSpans
    key_cache: object | None = None
    value_cache: object | None = None
    retained_key_cache: object | None = None
    retained_value_cache: object | None = None
    retained_append_spans: object | None = None
    retained_decode_spans: object | None = None
    int8_kv_value_bf16: bool = False
    retained_decode_kernel: object | None = None


def test_gguf_packed_single_row_prefill_uses_transient_oracle_without_persistent_mirror() -> None:
    metadata = object()
    retained_key = DeviceBuffer(0x1000, 1024)
    retained_value = DeviceBuffer(0x2000, 1024)
    oracle_key = DeviceBuffer(0x3000, 2048)
    oracle_value = DeviceBuffer(0x4000, 2048)
    layout = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP32,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.INT8_PER_TOKEN_HEAD,),
    )
    state = SimpleNamespace(
        kv_layout=layout,
        full_cache=lambda layer_id: (retained_key, retained_value),
        full_scale_metadata=lambda layer_id: metadata,
        full_bf16_mirror_cache=lambda layer_id: None,
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._int8_prefill_oracle_cache_for_layer = lambda layer_id: (oracle_key, oracle_value)
    scratch = _DirectInt8PrefillScratch(
        append_spans=_DirectInt8PrefillSpans(kind="append_position"),
        prefill_spans=_DirectInt8PrefillSpans(kind="decode_context"),
    )

    with pytest.raises(NotImplementedError, match="without a bounded BF16 mirror"):
        owner._packed_full_attention_scratch_for_layer(scratch, state, 0)

    direct = owner._packed_full_attention_scratch_for_layer(
        scratch,
        state,
        0,
        allow_direct_int8_prefill=True,
    )

    assert (direct.key_cache, direct.value_cache) == (oracle_key, oracle_value)
    assert (direct.retained_key_cache, direct.retained_value_cache) == (
        retained_key,
        retained_value,
    )
    assert direct.retained_append_spans.kind == "append_position"
    assert direct.retained_decode_spans.kind == "decode_context"
    assert direct.retained_append_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
    assert direct.retained_append_spans.scale_metadata is metadata
    assert direct.retained_decode_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
    assert direct.retained_decode_spans.scale_metadata is metadata

    owner._int8_prefill_oracle_cache_for_layer = lambda layer_id: pytest.fail(
        "direct decode must not allocate the transient BF16 prefill oracle"
    )
    decode = owner._packed_full_attention_scratch_for_layer(
        _DirectInt8PrefillScratch(
            append_spans=_DirectInt8PrefillSpans(kind="append_position"),
            prefill_spans=_DirectInt8PrefillSpans(kind="decode_context"),
            retained_decode_kernel=lambda *args, **kwargs: None,
        ),
        state,
        0,
    )
    assert (decode.key_cache, decode.value_cache) == (retained_key, retained_value)
    assert callable(decode.retained_decode_kernel)


def test_gguf_direct_int8_batch_uses_decode_counts_and_skips_bf16_oracle_write() -> None:
    source = inspect.getsource(
        gguf_runner.Qwen35GGUFFullStackRunner._run_full_attention_decode_batch_layer_rows
    )
    direct_call = source.split("retained_decode_kernel(", 1)[1].split(")\n", 1)[0]

    assert "retained_decode_spans" in direct_call
    assert "retained_spans," not in direct_call
    assert "if not direct_retained_batch:" in source
    assert 'attention_route = "kv_live_spans_int8_batch"' in source


def test_gguf_packed_ar_admits_mirrored_int8_and_fails_closed_without_mirror() -> None:
    mirrored = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.INT8_PER_TOKEN_HEAD,),
        bf16_mirror_layer_indices=(0,),
    )
    direct = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.INT8_PER_TOKEN_HEAD,),
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    peer = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._device_kv_layout = mirrored
    peer._device_kv_layout = mirrored

    assert owner._packed_ar_kv_layout_for_sessions((owner, peer)) == mirrored

    owner._device_kv_layout = direct
    peer._device_kv_layout = direct
    assert owner._resident_ar_kv_layout_for_sessions((owner, peer)) == direct
    assert owner._packed_ar_kv_layout_for_sessions(
        (owner,),
        allow_direct_int8_prefill=True,
    ) == direct
    with pytest.raises(NotImplementedError, match="single-row prefill"):
        owner._packed_ar_kv_layout_for_sessions(
            (owner, peer),
            allow_direct_int8_prefill=True,
        )
    with pytest.raises(NotImplementedError, match="without a bounded BF16 mirror"):
        owner._packed_ar_kv_layout_for_sessions((owner, peer))

    owner.packed_decode_max_rows = 4
    peer.packed_decode_max_rows = 4
    kernel = lambda *args, **kwargs: None
    owner._retained_decode_kernel = kernel
    peer._retained_decode_kernel = kernel
    assert owner._packed_ar_direct_decode_kernel_for_sessions(
        (owner, peer),
        physical_rows=4,
    ) is kernel
    peer._retained_decode_kernel = lambda *args, **kwargs: None
    assert owner._packed_ar_direct_decode_kernel_for_sessions(
        (owner, peer),
        physical_rows=4,
    ) is None
    peer._retained_decode_kernel = kernel
    assert owner._packed_ar_kv_layout_for_sessions(
        (owner, peer),
        allow_direct_int8_decode=True,
        physical_rows=4,
    ) == direct
    peer.packed_decode_max_rows = 1
    with pytest.raises(NotImplementedError, match="physical width 4"):
        owner._packed_ar_kv_layout_for_sessions(
            (owner, peer),
            allow_direct_int8_decode=True,
            physical_rows=4,
        )


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
    packed_state = _GGUFPackedTargetState(
        slot_count=2,
        max_sequence_length=layout.blocks_per_slot * layout.block_size,
        block_size=layout.block_size,
        blocks_per_slot=layout.blocks_per_slot,
        total_positions=layout.total_physical_positions,
        kv_layout=gguf_runner.Qwen35GGUFKVChunkLayout(
            storage_dtype=DType.BF16,
            storage_layout="uniform",
            scale_dtype=DType.FP16,
            scale_granularity="per_token_head",
            int8_kv_value_bf16=False,
            layer_storage_dtypes=(DType.BF16,),
        ),
        layer_conv_states=(None,),
        layer_recurrent_states=(None,),
        full_key_caches=(packed_key,),
        full_value_caches=(packed_value,),
        full_bf16_mirror_key_caches=(None,),
        full_bf16_mirror_value_caches=(None,),
        full_k_scale_caches=(None,),
        full_v_scale_caches=(None,),
        full_kv_scale_metadata=(None,),
        buffers=(),
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
    position_streams: list[int] = []

    def record_position(position_ptr, context_ptr, position, **kwargs) -> None:
        del position_ptr, context_ptr
        positions.append(int(position))
        position_streams.append(int(kwargs.get("stream", 0)))

    monkeypatch.setattr(
        gguf_runner,
        "set_decode_position_i64",
        record_position,
    )

    owner._scatter_packed_decode_state(
        sessions,
        layout,
        packed_state,
        runtime=runtime,
        stream=7,
        copy_full_kv=True,
    )

    row_nbytes = 16
    slot_physical_rows = layout.blocks_per_slot * layout.block_size
    assert runtime.copies == [
        (0x30000, 0x10000, 4 * row_nbytes),
        (0x40000, 0x20000, 4 * row_nbytes),
        (0x30000 + 4 * row_nbytes, 0x10000 + 4 * row_nbytes, 2 * row_nbytes),
        (0x40000 + 4 * row_nbytes, 0x20000 + 4 * row_nbytes, 2 * row_nbytes),
        (
            0x31000,
            0x10000 + slot_physical_rows * row_nbytes,
            4 * row_nbytes,
        ),
        (
            0x41000,
            0x20000 + slot_physical_rows * row_nbytes,
            4 * row_nbytes,
        ),
        (
            0x31000 + 4 * row_nbytes,
            0x10000 + (slot_physical_rows + 4) * row_nbytes,
            4 * row_nbytes,
        ),
        (
            0x41000 + 4 * row_nbytes,
            0x20000 + (slot_physical_rows + 4) * row_nbytes,
            4 * row_nbytes,
        ),
    ]
    assert positions == [6, 8]
    assert position_streams == [7, 7]
    assert [session._position for session in sessions] == [6, 8]

    runtime.copies.clear()
    positions.clear()
    position_streams.clear()
    owner._scatter_packed_decode_state(
        sessions,
        layout,
        packed_state,
        runtime=runtime,
        stream=7,
        copy_kv=False,
    )

    assert runtime.copies == []
    assert positions == [6, 8]
    assert position_streams == [7, 7]


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
    packed_state = _GGUFPackedTargetState(
        slot_count=1,
        max_sequence_length=layout.blocks_per_slot * layout.block_size,
        block_size=layout.block_size,
        blocks_per_slot=layout.blocks_per_slot,
        total_positions=layout.total_physical_positions,
        kv_layout=gguf_runner.Qwen35GGUFKVChunkLayout(
            storage_dtype=DType.BF16,
            storage_layout="uniform",
            scale_dtype=DType.FP16,
            scale_granularity="per_token_head",
            int8_kv_value_bf16=False,
            layer_storage_dtypes=(DType.BF16,),
        ),
        layer_conv_states=(None,),
        layer_recurrent_states=(None,),
        full_key_caches=(packed_key,),
        full_value_caches=(packed_value,),
        full_bf16_mirror_key_caches=(None,),
        full_bf16_mirror_value_caches=(None,),
        full_k_scale_caches=(None,),
        full_v_scale_caches=(None,),
        full_kv_scale_metadata=(None,),
        buffers=(),
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


def test_wall_clock_stage_recorder_converts_ticks_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace()
    buffer = DeviceBuffer(ptr=0x1000, nbytes=32)
    launches: list[tuple[int, int, int]] = []
    freed: list[DeviceBuffer] = []
    ticks = np.asarray([100, 112, 150], dtype=np.uint64)

    monkeypatch.setattr(gguf_runner, "malloc", lambda nbytes, **_: buffer)
    monkeypatch.setattr(gguf_runner, "free", lambda item, **_: freed.append(item))
    monkeypatch.setattr(
        gguf_runner,
        "wall_clock_mark_u64",
        lambda ptr, index, **kwargs: launches.append((int(ptr), int(index), int(kwargs["stream"]))),
    )
    monkeypatch.setattr(gguf_runner, "wall_clock_rate_khz", lambda **_: 2)

    def fake_copy(host_ptr: int, _buffer: DeviceBuffer, *, nbytes: int, **_kwargs) -> None:
        ctypes.memmove(host_ptr, ticks.ctypes.data, nbytes)

    monkeypatch.setattr(gguf_runner, "copy_device_to_host", fake_copy)

    recorder = _HipWallClockStageRecorder(runtime, enabled=True, stream=9, capacity=4)
    recorder.start()
    recorder.mark("first", "total")
    recorder.mark("second")
    timings: dict[str, float] = {}
    recorder.resolve_into(timings)
    recorder.close()

    assert launches == [(0x1000, 0, 9), (0x1000, 1, 9), (0x1000, 2, 9)]
    assert timings == {"first": 6.0, "total": 6.0, "second": 19.0}
    assert freed == [buffer]


def _segment_test_layout() -> gguf_runner.Qwen35GGUFKVChunkLayout:
    return gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.BF16,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.BF16,),
    )


def _segment_test_state(page_ids: tuple[int, ...] = ()) -> _GGUFPackedTargetState:
    return _GGUFPackedTargetState(
        slot_count=2,
        max_sequence_length=512,
        block_size=256,
        blocks_per_slot=2,
        total_positions=1024,
        kv_layout=_segment_test_layout(),
        layer_conv_states=(None,),
        layer_recurrent_states=(None,),
        full_key_caches=(DeviceBuffer(ptr=0x100000, nbytes=1 << 20),),
        full_value_caches=(DeviceBuffer(ptr=0x180000, nbytes=1 << 20),),
        full_bf16_mirror_key_caches=(None,),
        full_bf16_mirror_value_caches=(None,),
        full_k_scale_caches=(None,),
        full_v_scale_caches=(None,),
        full_kv_scale_metadata=(None,),
        buffers=(),
        page_ids=page_ids,
    )


def test_gguf_packed_target_state_copy_segments_identity_matches_dense_mapping() -> None:
    state = _segment_test_state()

    assert state.page_ids == (0, 1, 2, 3)
    segments = state.copy_segments(1, start_position=100, rows=400)

    # Legacy dense mapping: physical_base = slot * blocks_per_slot * block_size.
    physical_base = 1 * 2 * 256
    assert segments == (
        (100, physical_base + 100, 156),
        (256, physical_base + 256, 244),
    )


def test_gguf_packed_target_state_copy_segments_arbitrary_pages() -> None:
    state = _segment_test_state(page_ids=(10, 11, 20, 21))

    assert state.copy_segments(0, start_position=0, rows=300) == (
        (0, 10 * 256, 256),
        (256, 11 * 256, 44),
    )
    assert state.copy_segments(1, start_position=100, rows=400) == (
        (100, 20 * 256 + 100, 156),
        (256, 21 * 256, 244),
    )

    with pytest.raises(ValueError, match="outside slot_count"):
        state.copy_segments(2, start_position=0, rows=1)
    with pytest.raises(ValueError, match="non-negative"):
        state.copy_segments(0, start_position=-1, rows=1)
    with pytest.raises(ValueError, match="exceeds the slot page reservation"):
        state.copy_segments(0, start_position=300, rows=300)
    with pytest.raises(ValueError, match="page_ids length"):
        _segment_test_state(page_ids=(1, 2, 3))


def test_gguf_session_packed_kv_segments_walk_both_page_tables() -> None:
    class _RecordingRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            self.copies.append((int(dst), int(src), int(nbytes), int(stream)))

    layout = _segment_test_layout()
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(
                head_count_kv=2,
                key_length=4,
                layer_types=(FULL_ATTENTION,),
            )
        )
    )
    session._device_kv_layout = layout
    session._device_kv_allocation = SimpleNamespace(
        block_ids=(7, 3),
        chunk_start_block_id=0,
    )
    session.scratch = SimpleNamespace(
        full_key_caches=(DeviceBuffer(ptr=0x200000, nbytes=1 << 20),),
        full_value_caches=(DeviceBuffer(ptr=0x300000, nbytes=1 << 20),),
        full_bf16_mirror_key_caches=(None,),
        full_bf16_mirror_value_caches=(None,),
        full_k_scale_caches=(None,),
        full_v_scale_caches=(None,),
    )
    packed_state = _segment_test_state(page_ids=(10, 11, 20, 21))
    row_nbytes = 2 * 4 * DType.BF16.itemsize
    runtime = _RecordingRuntime()

    session._copy_session_packed_kv_segments(
        session,
        packed_state,
        1,
        0,
        start_position=100,
        rows=400,
        packed_to_session=False,
        runtime=runtime,
        stream=3,
    )

    assert runtime.copies == [
        (0x100000 + (20 * 256 + 100) * row_nbytes, 0x200000 + (7 * 256 + 100) * row_nbytes, 156 * row_nbytes, 3),
        (0x180000 + (20 * 256 + 100) * row_nbytes, 0x300000 + (7 * 256 + 100) * row_nbytes, 156 * row_nbytes, 3),
        (0x100000 + (21 * 256) * row_nbytes, 0x200000 + (3 * 256) * row_nbytes, 244 * row_nbytes, 3),
        (0x180000 + (21 * 256) * row_nbytes, 0x300000 + (3 * 256) * row_nbytes, 244 * row_nbytes, 3),
    ]

    runtime.copies.clear()
    session._copy_session_packed_kv_segments(
        session,
        packed_state,
        0,
        0,
        start_position=0,
        rows=300,
        packed_to_session=True,
        runtime=runtime,
        stream=5,
    )

    assert runtime.copies == [
        (0x200000 + (7 * 256) * row_nbytes, 0x100000 + (10 * 256) * row_nbytes, 256 * row_nbytes, 5),
        (0x300000 + (7 * 256) * row_nbytes, 0x180000 + (10 * 256) * row_nbytes, 256 * row_nbytes, 5),
        (0x200000 + (3 * 256) * row_nbytes, 0x100000 + (11 * 256) * row_nbytes, 44 * row_nbytes, 5),
        (0x300000 + (3 * 256) * row_nbytes, 0x180000 + (11 * 256) * row_nbytes, 44 * row_nbytes, 5),
    ]


def _rebind_test_state(
    *,
    slot_count: int = 3,
    blocks_per_slot: int = 3,
    page_ids: tuple[int, ...] = (),
) -> _GGUFPackedTargetState:
    return _GGUFPackedTargetState(
        slot_count=slot_count,
        max_sequence_length=blocks_per_slot * 4,
        block_size=4,
        blocks_per_slot=blocks_per_slot,
        total_positions=slot_count * blocks_per_slot * 4,
        kv_layout=_segment_test_layout(),
        layer_conv_states=(None,),
        layer_recurrent_states=(None,),
        full_key_caches=(DeviceBuffer(ptr=0x100000, nbytes=1 << 20),),
        full_value_caches=(DeviceBuffer(ptr=0x180000, nbytes=1 << 20),),
        full_bf16_mirror_key_caches=(None,),
        full_bf16_mirror_value_caches=(None,),
        full_k_scale_caches=(None,),
        full_v_scale_caches=(None,),
        full_kv_scale_metadata=(None,),
        buffers=(),
        page_ids=page_ids,
    )


def test_rebind_packed_verify_layout_pages_identity_keeps_layout() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11, 12, 13), start_position=4),
            _GGUFPackedVerifySlotBlock(input_token_ids=(21,), start_position=8),
            _GGUFPackedVerifySlotBlock(input_token_ids=(31, 32), start_position=1),
        ),
        block_size=4,
        slot_capacity=12,
    )

    rebound = gguf_runner._rebind_packed_verify_layout_pages(layout, _rebind_test_state())

    assert rebound is layout


def test_rebind_packed_verify_layout_pages_maps_arena_pages() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(11, 12, 13), start_position=4),
            _GGUFPackedVerifySlotBlock(input_token_ids=(0,), start_position=-1, active=False),
            _GGUFPackedVerifySlotBlock(input_token_ids=(31, 32), start_position=1),
        ),
        block_size=4,
        slot_capacity=12,
    )
    # Slot-major arena pages: slot0 -> (40,41,42), slot1 -> (50,51,52), slot2 -> (60,61,62).
    state = _rebind_test_state(page_ids=(40, 41, 42, 50, 51, 52, 60, 61, 62))

    rebound = gguf_runner._rebind_packed_verify_layout_pages(layout, state)

    assert rebound is not layout
    np.testing.assert_array_equal(rebound.block_table[0], np.asarray([40, 41, 42], dtype=np.int32))
    np.testing.assert_array_equal(rebound.block_table[2], np.asarray([40, 41, 42], dtype=np.int32))
    np.testing.assert_array_equal(rebound.block_table[3], np.full((3,), -1, dtype=np.int32))
    np.testing.assert_array_equal(rebound.block_table[4], np.asarray([60, 61, 62], dtype=np.int32))
    np.testing.assert_array_equal(rebound.block_table[5], np.asarray([60, 61, 62], dtype=np.int32))
    np.testing.assert_array_equal(rebound.row_positions, layout.row_positions)
    np.testing.assert_array_equal(rebound.cu_seqlens, layout.cu_seqlens)
    np.testing.assert_array_equal(rebound.active_mask, layout.active_mask)
    assert rebound.blocks_per_slot == layout.blocks_per_slot
    # The block table must agree with the page-aware copy segment mapping.
    segments = state.copy_segments(0, start_position=0, rows=12)
    assert tuple(physical // 4 for _, physical, _ in segments) == (40, 41, 42)


def test_rebind_packed_verify_layout_pages_uses_state_slot_stride() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(7,), start_position=5),
            _GGUFPackedVerifySlotBlock(input_token_ids=(9,), start_position=6),
        ),
        block_size=4,
        slot_capacity=8,
    )
    # Grown workspace: 4 pages per slot while the layout only spans 2.
    state = _rebind_test_state(
        slot_count=2,
        blocks_per_slot=4,
        page_ids=(10, 11, 12, 13, 20, 21, 22, 23),
    )

    rebound = gguf_runner._rebind_packed_verify_layout_pages(layout, state)

    np.testing.assert_array_equal(rebound.block_table[0], np.asarray([10, 11], dtype=np.int32))
    np.testing.assert_array_equal(rebound.block_table[1], np.asarray([20, 21], dtype=np.int32))

    # Even an identity page list must follow the state's wider slot stride.
    identity_wide = _rebind_test_state(slot_count=2, blocks_per_slot=4)
    rebound_wide = gguf_runner._rebind_packed_verify_layout_pages(layout, identity_wide)
    assert rebound_wide is not layout
    np.testing.assert_array_equal(rebound_wide.block_table[0], np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(rebound_wide.block_table[1], np.asarray([4, 5], dtype=np.int32))


def test_rebind_packed_verify_layout_pages_rejects_narrow_state() -> None:
    layout = _build_gguf_packed_verify_layout(
        (_GGUFPackedVerifySlotBlock(input_token_ids=(11, 12, 13), start_position=4),),
        block_size=4,
        slot_capacity=12,
    )
    narrow_pages = _rebind_test_state(slot_count=1, blocks_per_slot=2)
    with pytest.raises(ValueError, match="blocks_per_slot"):
        gguf_runner._rebind_packed_verify_layout_pages(layout, narrow_pages)

    two_slot_layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(7,), start_position=5),
            _GGUFPackedVerifySlotBlock(input_token_ids=(9,), start_position=6),
        ),
        block_size=4,
        slot_capacity=8,
    )
    narrow_slots = _rebind_test_state(slot_count=1, blocks_per_slot=2)
    with pytest.raises(ValueError, match="slot_count"):
        gguf_runner._rebind_packed_verify_layout_pages(two_slot_layout, narrow_slots)


def test_packed_decode_metadata_gate_rejects_nonidentity_rebound_layout() -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock(input_token_ids=(7,), start_position=5),
            _GGUFPackedVerifySlotBlock(input_token_ids=(9,), start_position=6),
        ),
        block_size=4,
        slot_capacity=8,
    )

    identity = gguf_runner._rebind_packed_verify_layout_pages(
        layout, _rebind_test_state(slot_count=2, blocks_per_slot=2)
    )
    assert _packed_decode_metadata_device_eligible(identity)

    arena = gguf_runner._rebind_packed_verify_layout_pages(
        layout,
        _rebind_test_state(slot_count=2, blocks_per_slot=2, page_ids=(4, 5, 8, 9)),
    )
    assert not _packed_decode_metadata_device_eligible(arena)


def _lease_test_runner() -> SimpleNamespace:
    return SimpleNamespace(
        linear_qkv_width=10,
        ssm_value_dim=2,
        weights=SimpleNamespace(
            config=SimpleNamespace(
                layer_types=(FULL_ATTENTION,),
                ssm_conv_kernel=4,
                ssm_time_step_rank=2,
                ssm_state_size=3,
                head_count_kv=2,
                key_length=4,
            )
        ),
    )


def _lease_test_layout() -> gguf_runner.Qwen35GGUFKVChunkLayout:
    return gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.BF16,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.BF16,),
    )


def _lease_test_backing(layout: gguf_runner.Qwen35GGUFKVChunkLayout) -> SimpleNamespace:
    # One full-attention layer, BF16 payload: 256 * head_count_kv * key_length * 2
    # bytes per plane page; the fake arena holds 8 pages per plane.
    page_nbytes = 256 * 2 * 4 * DType.BF16.itemsize
    key_plane = DeviceBuffer(ptr=0xA00000, nbytes=8 * page_nbytes)
    value_plane = DeviceBuffer(ptr=0xB00000, nbytes=8 * page_nbytes)
    return SimpleNamespace(
        layout=layout,
        full_key_caches=(key_plane,),
        full_value_caches=(value_plane,),
        full_bf16_mirror_key_caches=(None,),
        full_bf16_mirror_value_caches=(None,),
        full_k_scale_caches=(None,),
        full_v_scale_caches=(None,),
        full_kv_scale_metadata=(None,),
        buffers=(key_plane, value_plane),
    )


def test_gguf_packed_target_state_allocate_uses_pool_workspace_lease(monkeypatch) -> None:
    allocated: list[int] = []

    def fake_malloc(nbytes, *, runtime):
        allocated.append(int(nbytes))
        return DeviceBuffer(ptr=0x500000 + len(allocated) * 0x10000, nbytes=int(nbytes))

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    layout = _lease_test_layout()
    backing = _lease_test_backing(layout)
    lease_pages = (5, 6, 7, 8)
    pool = SimpleNamespace(
        backing=backing,
        workspace_pages=lambda key: (
            lease_pages
            if key == gguf_runner._GGUF_PACKED_WORKSPACE_LEASE_KEY
            else None
        ),
    )

    state = _GGUFPackedTargetState.allocate(
        _lease_test_runner(),
        slot_count=2,
        max_sequence_length=512,
        runtime=SimpleNamespace(),
        kv_layout=layout,
        kv_pool=pool,
    )

    assert state.kv_backing_kind == "pool_lease"
    assert state.page_ids == lease_pages
    assert state.full_key_caches == backing.full_key_caches
    assert state.full_value_caches == backing.full_value_caches
    # The arena planes are borrowed, never owned: no fresh allocation and the
    # plane buffers are not part of the state's owned buffer set.
    assert allocated == []
    assert backing.full_key_caches[0] not in state.buffers
    assert backing.full_value_caches[0] not in state.buffers


def test_gguf_packed_target_state_allocate_private_without_lease(monkeypatch) -> None:
    allocated: list[int] = []

    def fake_malloc(nbytes, *, runtime):
        allocated.append(int(nbytes))
        return DeviceBuffer(ptr=0x500000 + len(allocated) * 0x10000, nbytes=int(nbytes))

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    layout = _lease_test_layout()

    state = _GGUFPackedTargetState.allocate(
        _lease_test_runner(),
        slot_count=2,
        max_sequence_length=512,
        runtime=SimpleNamespace(),
        kv_layout=layout,
    )
    assert state.kv_backing_kind == "private"
    assert state.page_ids == (0, 1, 2, 3)
    assert len(allocated) == 2  # private key + value planes
    assert all(buffer in state.buffers for buffer in state.full_key_caches)

    allocated.clear()
    leaseless_pool = SimpleNamespace(workspace_pages=lambda key: None)
    state2 = _GGUFPackedTargetState.allocate(
        _lease_test_runner(),
        slot_count=2,
        max_sequence_length=512,
        runtime=SimpleNamespace(),
        kv_layout=layout,
        kv_pool=leaseless_pool,
    )
    assert state2.kv_backing_kind == "private"
    assert len(allocated) == 2


def test_gguf_packed_target_state_allocate_rejects_bad_lease(monkeypatch) -> None:
    monkeypatch.setattr(
        gguf_runner,
        "malloc",
        lambda nbytes, *, runtime: DeviceBuffer(ptr=0x500000, nbytes=int(nbytes)),
    )
    layout = _lease_test_layout()
    backing = _lease_test_backing(layout)

    small_pool = SimpleNamespace(
        backing=backing,
        workspace_pages=lambda key: (1, 2),
    )
    with pytest.raises(RuntimeError, match="pages"):
        _GGUFPackedTargetState.allocate(
            _lease_test_runner(),
            slot_count=2,
            max_sequence_length=512,
            runtime=SimpleNamespace(),
            kv_layout=layout,
            kv_pool=small_pool,
        )

    mismatched_layout = gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(DType.INT8_PER_TOKEN_HEAD,),
    )
    mismatched_pool = SimpleNamespace(
        backing=_lease_test_backing(mismatched_layout),
        workspace_pages=lambda key: (5, 6, 7, 8),
    )
    with pytest.raises(RuntimeError, match="layout"):
        _GGUFPackedTargetState.allocate(
            _lease_test_runner(),
            slot_count=2,
            max_sequence_length=512,
            runtime=SimpleNamespace(),
            kv_layout=layout,
            kv_pool=mismatched_pool,
        )


def test_gguf_session_workspace_pool_binding_delegates_to_owner() -> None:
    from hipengine.kvcache.device_global import GlobalDeviceKVPool

    pool = GlobalDeviceKVPool(
        page_bytes=4096,
        backend_fingerprint="test",
        generation=1,
        backing=None,
        plane_page_pointers={"payload": (0x1000, 0x2000)},
        pointer_table_pointers={"payload": 0x3000},
        metadata_descriptor_pointer=0x4000,
        close_storage=lambda: None,
    )
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._workspace_kv_pool = None
    view = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    view._workspace_kv_pool = None
    view._resident_batch_owner = owner

    view.bind_workspace_kv_pool(pool)
    assert owner._workspace_kv_pool is pool
    assert view._workspace_kv_pool is None
    view.bind_workspace_kv_pool(None)
    assert owner._workspace_kv_pool is None

    with pytest.raises(TypeError, match="GlobalDeviceKVPool"):
        owner.bind_workspace_kv_pool(object())
