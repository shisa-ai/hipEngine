from __future__ import annotations

import numpy as np
import pytest

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpHostQSAIndexState


def test_qwen4_exp_host_index_state_selects_blocks_tail_and_rolls_back() -> None:
    state = Qwen4ExpHostQSAIndexState.allocate(
        capacity=8,
        index_dim=2,
        compression_ratio=2,
        block_budget=2,
    )
    keys = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    for position, key in enumerate(keys[:4]):
        state.append(key, position=position)
    snapshot = state.snapshot()
    state.append(keys[4], position=4)
    selection = state.select(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_position=4,
        key_norm_weight=np.ones(2, dtype=np.float32),
        rotary_dim=2,
        theta=10_000_000.0,
        eps=1e-6,
    )
    np.testing.assert_array_equal(selection.selected_block_starts[0], [0, 2])
    np.testing.assert_array_equal(selection.selected_positions[0], [0, 1, 2, 3, 4])

    state.restore(snapshot)
    assert state.count == 4
    np.testing.assert_array_equal(state.raw_keys[:4], keys[:4])
    state.append(np.array([9.0, 9.0], dtype=np.float32), position=4)
    assert state.count == 5


def test_qwen4_exp_host_index_state_rejects_noncontiguous_or_overflow_append() -> None:
    state = Qwen4ExpHostQSAIndexState.allocate(
        capacity=1,
        index_dim=2,
        compression_ratio=2,
        block_budget=1,
    )
    with pytest.raises(ValueError, match="position"):
        state.append([1.0, 2.0], position=1)
    state.append([1.0, 2.0], position=0)
    with pytest.raises(ValueError, match="capacity"):
        state.append([3.0, 4.0], position=1)
