from __future__ import annotations

import numpy as np
import pytest

from hipengine.runtime.qwen35_gguf_runner import (
    _GGUFPackedVerifySlotBlock,
    _build_gguf_packed_verify_layout,
)


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
