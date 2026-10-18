"""CPU contracts for the indexed Qwen4Exp F32 conv decode primitive.

Covers the host-side pieces of the indexed decode ABI:

* per-row state-slot admission that bounds an independent-request batch to a
  declared Conv state slab and forbids two rows sharing one slot, and
* the CPU oracle the GPU test compares against, so a wrong ``cu_seqlens`` or
  ``state_indices`` construction is caught without a device.

The oracle is the committed segmented convolution reference
(:func:`hipengine.kernels.cpu_reference.ops.linear_attn_conv_prefill_segments`)
driven with one-token segments: row ``r`` reads and writes exactly the state slot
``state_indices[r]``, which is the shape the GPU primitive uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.ops import linear_attn_conv_prefill_segments
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS,
    QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE,
    QWEN4_EXP_CONV_INDEXED_MAX_ROWS,
    QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS,
    qwen4_exp_conv_indexed_state_indices,
    qwen4_exp_linear_attn_conv_decode_indexed_f32,
)


def _segments(rows: int) -> np.ndarray:
    """One-token segments over a packed row block."""

    return np.arange(rows + 1, dtype=np.int64)


def _fixture(rng, rows, state_slots, channels, kernel_size):
    hidden = rng.normal(0.0, 0.4, (rows, channels)).astype(np.float32)
    weight = rng.normal(0.0, 0.3, (channels, kernel_size)).astype(np.float32)
    state = rng.normal(0.0, 0.25, (state_slots, channels, kernel_size)).astype(
        np.float32
    )
    return hidden, weight, state


# ---------------------------------------------------------------------------
# Slot admission.
# ---------------------------------------------------------------------------


def test_state_indices_accepts_permuted_distinct_slots() -> None:
    indices = qwen4_exp_conv_indexed_state_indices(
        [3, 1, 5, 0], rows=4, state_slots=6
    )
    assert indices.dtype == np.int64
    assert indices.tolist() == [3, 1, 5, 0]
    assert indices.flags["C_CONTIGUOUS"]


def test_state_indices_accepts_a_numpy_int32_vector() -> None:
    indices = qwen4_exp_conv_indexed_state_indices(
        np.array([2, 0], dtype=np.int32), rows=2, state_slots=3
    )
    assert indices.tolist() == [2, 0]


@pytest.mark.parametrize(
    "indices,rows,state_slots",
    [
        ([2, 2], 2, 3),
        (np.array([2, 2], dtype=np.int32), 2, 3),
        ([0, 1, 0], 3, 4),
        ([3, 1, 1, 0], 4, 6),
        ([0] * 4, 4, 4),
    ],
)
def test_state_indices_rejects_duplicate_slots(indices, rows, state_slots) -> None:
    """Two independent rows must not name one Conv state slot.

    The decode body shifts and appends inside one slot, so two rows sharing it
    would interleave their shifts: the committed state would be neither row's
    result, and which bytes survive would depend on block completion order. That
    is an ownership violation, so the host rejects it before any launch.
    """

    with pytest.raises(ValueError, match="duplicate"):
        qwen4_exp_conv_indexed_state_indices(
            indices, rows=rows, state_slots=state_slots
        )


def test_state_indices_duplicate_error_names_the_slot_and_both_rows() -> None:
    with pytest.raises(ValueError) as error:
        qwen4_exp_conv_indexed_state_indices([4, 1, 4], rows=3, state_slots=5)
    message = str(error.value)
    assert "duplicate" in message
    assert "slot 4" in message
    assert "rows 0 and 2" in message


@pytest.mark.parametrize(
    "indices,rows,state_slots,match",
    [
        ([0, 1], 3, 4, "one index per row"),
        ([0, 1, 2, 3, 4], 4, 5, "one index per row"),
        ([0, 1], 2, 0, "state_slots must be positive"),
        ([0, 1], 2, -1, "state_slots must be positive"),
        ([0, 2], 2, 2, "outside the declared state slab"),
        ([0, -1], 2, 4, "non-negative"),
        ([0, 4], 2, 4, "outside the declared state slab"),
        ([0], 0, 4, "rows must be in 1"),
        ([], 0, 4, "rows must be in 1"),
        (
            [0] * (QWEN4_EXP_CONV_INDEXED_MAX_ROWS + 1),
            QWEN4_EXP_CONV_INDEXED_MAX_ROWS + 1,
            QWEN4_EXP_CONV_INDEXED_MAX_ROWS + 1,
            "rows must be in 1",
        ),
        ([0.0, 1.0], 2, 4, "integers"),
        (["0", "1"], 2, 4, "integers"),
        (0, 2, 4, "sequence"),
    ],
)
def test_state_indices_reject_unbounded_or_malformed_batches(
    indices, rows, state_slots, match
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        qwen4_exp_conv_indexed_state_indices(
            indices, rows=rows, state_slots=state_slots
        )


def test_state_indices_rejects_a_slab_wider_than_the_device_check() -> None:
    """The device duplicate scan is a bounded bitmap, so the slab is bounded."""

    limit = QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS
    assert limit >= QWEN4_EXP_CONV_INDEXED_MAX_ROWS
    with pytest.raises(ValueError, match="state_slots must be in"):
        qwen4_exp_conv_indexed_state_indices([0, 1], rows=2, state_slots=limit + 1)
    accepted = qwen4_exp_conv_indexed_state_indices(
        [0, limit - 1], rows=2, state_slots=limit
    )
    assert accepted.tolist() == [0, limit - 1]


def test_state_indices_boundary_is_inclusive() -> None:
    limit = QWEN4_EXP_CONV_INDEXED_MAX_ROWS
    indices = qwen4_exp_conv_indexed_state_indices(
        list(range(limit)), rows=limit, state_slots=limit
    )
    assert indices.size == limit


# ---------------------------------------------------------------------------
# Wrapper admission (no device access).
# ---------------------------------------------------------------------------


def _admit(**overrides):
    """Call the wrapper far enough to validate, without touching a device.

    Admission runs before the library/runtime defaults are resolved, so
    placeholder objects prove the rejection happened on the host.
    """

    kwargs = {
        "rows": 2,
        "state_slots": 2,
        "channels": 64,
        "kernel_size": 4,
        **overrides,
    }
    return qwen4_exp_linear_attn_conv_decode_indexed_f32(
        *([16] * 5), runtime=object(), library=object(), **kwargs
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rows": 0}, "rows must be in 1"),
        ({"rows": QWEN4_EXP_CONV_INDEXED_MAX_ROWS + 1}, "rows must be in 1"),
        ({"state_slots": 0}, "state_slots must be positive"),
        ({"state_slots": QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS + 1},
         "state_slots must be in"),
        ({"state_slots": 1}, "state slab too small"),
        ({"channels": 0}, "channels must be positive"),
        ({"channels": -4}, "channels must be positive"),
        ({"kernel_size": 0}, "kernel_size must be positive"),
        ({"channels": QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS + 1},
         "channels must be in 1"),
        ({"kernel_size": QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE + 1},
         "kernel_size must be in 1"),
    ],
)
def test_admission_rejects_unbounded_shapes(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _admit(**overrides)


def test_admission_rejects_a_misaligned_slot_vector() -> None:
    """The slot vector is read as int64 on the device."""

    pointers = [16] * 5
    pointers[4] = 20
    with pytest.raises(ValueError, match="8-byte aligned"):
        qwen4_exp_linear_attn_conv_decode_indexed_f32(
            *pointers, rows=2, state_slots=2, channels=64, kernel_size=4,
            runtime=object(), library=object(),
        )


def test_admission_passes_a_valid_call_through_to_symbol_lookup() -> None:
    """Guard against admission that rejects everything."""

    with pytest.raises(
        AttributeError, match="hipengine_qwen4_exp_linear_attn_conv_decode_indexed_f32"
    ):
        _admit(rows=4, state_slots=6, channels=1024, kernel_size=4)


def test_bounds_match_the_c_launcher() -> None:
    """The Python and C admission bounds must agree.

    The host check runs first, so a drift widening only the Python side would
    make the C guards dead code.
    """

    source = (
        Path(__file__).resolve().parents[1]
        / "hipengine/kernels/hip_gfx1100/linear_attn/conv.hip"
    ).read_text()
    for name, value in (
        ("kQwen4ExpConvIndexedMaxRows", QWEN4_EXP_CONV_INDEXED_MAX_ROWS),
        ("kQwen4ExpConvIndexedMaxStateSlots", QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS),
        ("kQwen4ExpConvIndexedMaxChannels", QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS),
        (
            "kQwen4ExpConvIndexedMaxKernelSize",
            QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE,
        ),
    ):
        assert f"constexpr int64_t {name} = {value};" in source


# ---------------------------------------------------------------------------
# Oracle behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size", [2, 4])
def test_oracle_one_token_segments_match_per_slot_sequential_steps(
    kernel_size,
) -> None:
    """One-token segments equal stepping each row alone, slot by slot."""

    rng = np.random.default_rng(2024 + kernel_size)
    rows, state_slots, channels = 4, 6, 12
    hidden, weight, state = _fixture(rng, rows, state_slots, channels, kernel_size)
    slots = np.array([4, 1, 5, 0], dtype=np.int64)

    out, final = linear_attn_conv_prefill_segments(
        hidden, state, weight, _segments(rows), slots
    )

    replay_state = state.copy()
    replay_out = np.empty_like(out)
    for row, slot in enumerate(slots):
        row_out, replay_state = linear_attn_conv_prefill_segments(
            hidden[row : row + 1],
            replay_state,
            weight,
            np.array([0, 1], dtype=np.int64),
            np.array([slot], dtype=np.int64),
        )
        replay_out[row] = row_out[0]

    np.testing.assert_array_equal(out, replay_out)
    np.testing.assert_array_equal(final, replay_state)
    # Slots no row selected are untouched, the canary property the GPU test
    # asserts on device.
    for slot in range(state_slots):
        if slot in slots:
            continue
        np.testing.assert_array_equal(final[slot], state[slot])


def test_oracle_repeated_steps_follow_changing_slots() -> None:
    """Three steps with a changing composition keep per-slot state separate."""

    rng = np.random.default_rng(31337)
    channels, kernel_size, state_slots = 8, 4, 4
    steps = [(1, 0), (3, 1), (0, 2)]
    weight = rng.normal(0.0, 0.3, (channels, kernel_size)).astype(np.float32)
    initial = rng.normal(0.0, 0.25, (state_slots, channels, kernel_size)).astype(
        np.float32
    )
    step_hidden = [
        rng.normal(0.0, 0.4, (len(slots), channels)).astype(np.float32)
        for slots in steps
    ]

    batched_state = initial.copy()
    reference_state = initial.copy()
    for hidden, slots in zip(step_hidden, steps):
        slots_array = np.asarray(slots, dtype=np.int64)
        batched_out, batched_state = linear_attn_conv_prefill_segments(
            hidden, batched_state, weight, _segments(len(slots)), slots_array
        )
        reference_out = np.empty_like(batched_out)
        for row, slot in enumerate(slots):
            single_out, reference_state = linear_attn_conv_prefill_segments(
                hidden[row : row + 1],
                reference_state,
                weight,
                np.array([0, 1], dtype=np.int64),
                np.array([slot], dtype=np.int64),
            )
            reference_out[row] = single_out[0]
        np.testing.assert_array_equal(batched_out, reference_out)
        np.testing.assert_array_equal(batched_state, reference_state)


def test_oracle_is_order_independent_across_rows() -> None:
    """Permuting rows with their slots permutes outputs, not the state."""

    rng = np.random.default_rng(90210)
    rows, state_slots, channels, kernel_size = 3, 3, 6, 3
    hidden, weight, state = _fixture(rng, rows, state_slots, channels, kernel_size)
    slots = np.array([2, 0, 1], dtype=np.int64)
    order = np.array([2, 0, 1], dtype=np.int64)

    base_out, base_state = linear_attn_conv_prefill_segments(
        hidden, state, weight, _segments(rows), slots
    )
    permuted_out, permuted_state = linear_attn_conv_prefill_segments(
        hidden[order], state, weight, _segments(rows), slots[order]
    )

    np.testing.assert_array_equal(permuted_state, base_state)
    np.testing.assert_array_equal(permuted_out, base_out[order])
