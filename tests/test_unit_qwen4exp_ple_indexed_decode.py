"""CPU contracts for the indexed Qwen4Exp PLE dilated-conv decode primitive.

The indexed ABI turns the single-request PLE Conv into one token per row for
independent requests: row ``r`` reads and writes exactly PLE history slot
``state_indices[r]`` of a ``[state_slots][(kernel_size - 1) * dilation][channels]``
FP32 slab, and slots no row selects keep their bytes.

This module covers the host-side pieces without a device:

* per-row slot admission that bounds an independent-request batch to the declared
  history slab and forbids two rows sharing one slot, and
* the oracle the GPU test compares against, so a wrong history layout or a wrong
  slot map is caught on the CPU.

The oracle is the committed PLE reference
(:func:`hipengine.kernels.cpu_reference.qwen4_exp.dilated_depthwise_conv`) driven
with one-token position vectors, not a restatement of the kernel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.qwen4_exp import (
    PLEConvState,
    dilated_depthwise_conv,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
    QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS,
    QWEN4_EXP_PLE_INDEXED_MAX_DILATION,
    QWEN4_EXP_PLE_INDEXED_MAX_HISTORY_ROWS,
    QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE,
    QWEN4_EXP_PLE_INDEXED_MAX_ROWS,
    QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS,
    qwen4_exp_ple_dilated_depthwise_conv_indexed_f32,
    qwen4_exp_ple_indexed_state_indices,
)


def _fixture(rng, rows, state_slots, channels, kernel_size, dilation):
    token = rng.normal(0.0, 0.4, (rows, channels)).astype(np.float32)
    weight = rng.normal(0.0, 0.3, (channels, kernel_size)).astype(np.float32)
    history_rows = (kernel_size - 1) * dilation
    slab = rng.normal(0.0, 0.25, (state_slots, history_rows, channels)).astype(
        np.float32
    )
    return token, weight, slab


def _oracle_step(slot_history, token_row, weight, *, dilation, position):
    """One independent decode step for one request through the CPU reference."""

    out, state = dilated_depthwise_conv(
        token_row[None, :],
        weight,
        dilation=dilation,
        positions=[position],
        state=PLEConvState(slot_history.copy(), position),
    )
    return out[0], state.history


def _oracle_replay(token, weight, slab, slots, *, dilation, positions):
    """Serial per-row replay over a copy of the slab, in row order."""

    out = np.empty((len(slots), weight.shape[0]), np.float32)
    state = slab.copy()
    for row, slot in enumerate(slots):
        out[row], state[slot] = _oracle_step(
            state[slot], token[row], weight,
            dilation=dilation, position=positions[slot],
        )
    return out, state


# ---------------------------------------------------------------------------
# Slot admission.
# ---------------------------------------------------------------------------


def test_state_indices_accepts_permuted_distinct_slots() -> None:
    indices = qwen4_exp_ple_indexed_state_indices(
        [3, 1, 5, 0], rows=4, state_slots=6
    )
    assert indices.dtype == np.int64
    assert indices.tolist() == [3, 1, 5, 0]
    assert indices.flags["C_CONTIGUOUS"]


def test_state_indices_accepts_a_numpy_int32_vector() -> None:
    indices = qwen4_exp_ple_indexed_state_indices(
        np.array([2, 0], dtype=np.int32), rows=2, state_slots=3
    )
    assert indices.dtype == np.int64
    assert indices.tolist() == [2, 0]


@pytest.mark.parametrize("slots,rows,state_slots", [
    ((1, 1), 2, 2),
    ((0, 2, 0), 3, 4),
    ((3, 3, 1, 0), 4, 4),
])
def test_state_indices_rejects_duplicate_slots(slots, rows, state_slots) -> None:
    """Two rows naming one slot would interleave two requests' shifts."""

    with pytest.raises(ValueError, match="duplicate slot"):
        qwen4_exp_ple_indexed_state_indices(slots, rows=rows, state_slots=state_slots)


def test_state_indices_duplicate_error_names_the_slot_and_both_rows() -> None:
    with pytest.raises(ValueError) as error:
        qwen4_exp_ple_indexed_state_indices([4, 2, 4], rows=3, state_slots=5)
    message = str(error.value)
    assert "slot 4" in message and "rows 0 and 2" in message


@pytest.mark.parametrize("slots,rows,state_slots,match", [
    ((0, 5), 2, 5, r"state_indices\[1\]=5 is outside"),
    ((-1, 0), 2, 4, "non-negative"),
    ((0,), 2, 4, "one index per row"),
    ((0, 1, 2), 3, 2, "outside the declared state slab"),
    ((0,), QWEN4_EXP_PLE_INDEXED_MAX_ROWS + 1, 65, "rows must be in"),
    ((0,), 1, QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS + 1, "state_slots must be in"),
    ((0,), 1, 0, "state_slots must be positive"),
])
def test_state_indices_reject_unbounded_or_malformed_batches(
    slots, rows, state_slots, match
) -> None:
    with pytest.raises(ValueError, match=match):
        qwen4_exp_ple_indexed_state_indices(slots, rows=rows, state_slots=state_slots)


def test_state_indices_boundary_is_inclusive() -> None:
    """The row bound is a closed interval; one past it is rejected."""

    slots = tuple(range(QWEN4_EXP_PLE_INDEXED_MAX_ROWS))
    indices = qwen4_exp_ple_indexed_state_indices(
        slots, rows=len(slots), state_slots=QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS
    )
    assert indices.size == QWEN4_EXP_PLE_INDEXED_MAX_ROWS


# ---------------------------------------------------------------------------
# Launcher admission.
# ---------------------------------------------------------------------------


def _admit(**overrides):
    """Call the wrapper far enough to validate, without touching a device."""

    kwargs = {
        "rows": 2,
        "state_slots": 2,
        "channels": 64,
        "kernel_size": 4,
        "dilation": 3,
        **overrides,
    }
    return qwen4_exp_ple_dilated_depthwise_conv_indexed_f32(
        *([16] * 5), runtime=object(), library=object(), **kwargs
    )


@pytest.mark.parametrize("overrides,match", [
    ({"rows": 0}, "rows must be in"),
    ({"rows": QWEN4_EXP_PLE_INDEXED_MAX_ROWS + 1}, "rows must be in"),
    ({"state_slots": 0}, "state_slots must be positive"),
    ({"state_slots": QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS + 1},
     "state_slots must be in"),
    ({"state_slots": 1}, "state slab too small"),
    ({"channels": 0}, "channels must be positive"),
    ({"channels": QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS + 1}, "channels must be in"),
    ({"kernel_size": 0}, "kernel_size must be positive"),
    ({"kernel_size": QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE + 1},
     "kernel_size must be in"),
    ({"dilation": 0}, "dilation must be positive"),
    ({"dilation": QWEN4_EXP_PLE_INDEXED_MAX_DILATION + 1}, "dilation must be in"),
    # (kernel_size - 1) * dilation is the history depth this ABI derives.
    ({"kernel_size": 1024, "dilation": 1024}, "history rows must be in"),
])
def test_admission_rejects_unbounded_shapes(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _admit(**overrides)


def test_admission_rejects_a_misaligned_slot_vector() -> None:
    """The slot vector is read as int64 on the device."""

    pointers = [16] * 5
    pointers[4] = 20
    with pytest.raises(ValueError, match="8-byte aligned"):
        qwen4_exp_ple_dilated_depthwise_conv_indexed_f32(
            *pointers, rows=2, state_slots=2, channels=64, kernel_size=4,
            dilation=3, runtime=object(), library=object(),
        )


def test_admission_passes_a_valid_call_through_to_symbol_lookup() -> None:
    """Guard against admission that rejects everything."""

    with pytest.raises(
        AttributeError,
        match="hipengine_qwen4_exp_ple_dilated_depthwise_conv_indexed_f32",
    ):
        _admit(rows=4, state_slots=6, channels=1024, kernel_size=4, dilation=3)


def test_admission_keeps_the_history_rows_bound_inclusive() -> None:
    """``(kernel_size - 1) * dilation`` is bounded, not the operands alone."""

    # 1023 * 64 = 65472 is the largest admitted depth below the 65536 bound.
    with pytest.raises(AttributeError, match="indexed_f32"):
        _admit(rows=1, state_slots=1, channels=8, kernel_size=1024, dilation=64)
    with pytest.raises(ValueError, match="history rows must be in"):
        _admit(rows=1, state_slots=1, channels=8, kernel_size=1024, dilation=65)


def test_bounds_match_the_c_launcher() -> None:
    """The Python and C admission bounds must agree.

    The host check runs first, so a drift widening only the Python side would
    make the C guards dead code.
    """

    source = (
        Path(__file__).resolve().parents[1]
        / "hipengine/kernels/hip_gfx1100/fused/qwen4_exp_ple.hip"
    ).read_text()
    for name, value in (
        ("kQwen4ExpPleIndexedMaxRows", QWEN4_EXP_PLE_INDEXED_MAX_ROWS),
        ("kQwen4ExpPleIndexedMaxStateSlots", QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS),
        ("kQwen4ExpPleIndexedMaxChannels", QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS),
        ("kQwen4ExpPleIndexedMaxKernelSize", QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE),
        ("kQwen4ExpPleIndexedMaxDilation", QWEN4_EXP_PLE_INDEXED_MAX_DILATION),
        (
            "kQwen4ExpPleIndexedMaxHistoryRows",
            QWEN4_EXP_PLE_INDEXED_MAX_HISTORY_ROWS,
        ),
    ):
        assert f"constexpr int64_t {name} = {value};" in source


# ---------------------------------------------------------------------------
# Oracle behaviour: history layout and the slot map.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size,dilation", [(4, 3), (2, 1), (4, 1), (3, 2)])
def test_oracle_one_token_steps_match_a_serial_per_slot_replay(
    kernel_size, dilation
) -> None:
    """A permuted slot map must land on the same per-request results."""

    slots = (3, 1, 5, 0)
    channels = 48
    rng = np.random.default_rng(17 + kernel_size * dilation)
    token, weight, slab = _fixture(
        rng, len(slots), 6, channels, kernel_size, dilation
    )
    positions = {slot: 5 + 3 * slot for slot in slots}

    out, state = _oracle_replay(
        token, weight, slab, slots, dilation=dilation, positions=positions
    )

    for row, slot in enumerate(slots):
        expected_out, expected_state = _oracle_step(
            slab[slot], token[row], weight,
            dilation=dilation, position=positions[slot],
        )
        np.testing.assert_array_equal(out[row].view(np.uint32),
                                      expected_out.view(np.uint32))
        np.testing.assert_array_equal(state[slot].view(np.uint32),
                                      expected_state.view(np.uint32))
    for slot in (2, 4):
        np.testing.assert_array_equal(state[slot].view(np.uint32),
                                      slab[slot].view(np.uint32))


@pytest.mark.parametrize("kernel_size,dilation", [(4, 3), (4, 1), (2, 2)])
def test_oracle_history_is_the_owner_shift_and_append(kernel_size, dilation) -> None:
    """Pin the layout the kernel indexes: tap ``t`` reads history row ``t*D``.

    For one token the reference appends the token and keeps the last
    ``(kernel_size - 1) * dilation`` rows, so the updated slot is the old slot
    shifted up by one row with the token appended. The Conv therefore reads
    ``history[t * dilation]`` for ``t < kernel_size - 1`` and the token itself for
    the last tap.
    """

    channels = 32
    history_rows = (kernel_size - 1) * dilation
    rng = np.random.default_rng(91 + history_rows)
    slot_history = rng.normal(0.0, 0.3, (history_rows, channels)).astype(np.float32)
    token_row = rng.normal(0.0, 0.3, channels).astype(np.float32)
    weight = rng.normal(0.0, 0.3, (channels, kernel_size)).astype(np.float32)

    out, state = _oracle_step(
        slot_history, token_row, weight, dilation=dilation, position=11
    )

    expected = np.concatenate(
        (slot_history[1:], token_row[None, :]), axis=0
    )
    np.testing.assert_array_equal(state.view(np.uint32), expected.view(np.uint32))

    expected_out = np.zeros(channels, np.float32)
    for tap in range(kernel_size):
        source = (
            token_row if tap == kernel_size - 1 else slot_history[tap * dilation]
        )
        expected_out = (expected_out + source * weight[:, tap]).astype(np.float32)
    np.testing.assert_array_equal(out.view(np.uint32), expected_out.view(np.uint32))


def test_oracle_repeated_steps_advance_only_the_selected_slots() -> None:
    """Three steps with a rotating slot map; unselected slots stay put."""

    channels, kernel_size, dilation = 40, 4, 2
    history_rows = (kernel_size - 1) * dilation
    rng = np.random.default_rng(613)
    weight = rng.normal(0.0, 0.3, (channels, kernel_size)).astype(np.float32)
    slab = rng.normal(0.0, 0.25, (5, history_rows, channels)).astype(np.float32)
    untouched = slab[[1, 3]].copy()

    state = slab.copy()
    for step, slots in enumerate(((0, 4), (2, 0), (4, 2))):
        token = rng.normal(0.0, 0.4, (2, channels)).astype(np.float32)
        positions = {slot: 20 + step for slot in slots}
        _, state = _oracle_replay(
            token, weight, state, slots,
            dilation=dilation, positions=positions,
        )
        np.testing.assert_array_equal(state[[1, 3]].view(np.uint32),
                                      untouched.view(np.uint32))
    # Slot 0 was advanced by steps 0 and 1 and slot 2 by steps 1 and 2.
    for slot in (0, 2, 4):
        assert not np.array_equal(state[slot], slab[slot])


def test_oracle_is_order_independent_across_rows() -> None:
    """Permuting rows permutes results; it does not change them."""

    channels, kernel_size, dilation = 24, 4, 3
    slots = (0, 2, 3)
    rng = np.random.default_rng(2_048)
    token, weight, slab = _fixture(
        rng, len(slots), 4, channels, kernel_size, dilation
    )
    positions = {slot: 7 for slot in slots}

    forward = _oracle_replay(
        token, weight, slab, slots, dilation=dilation, positions=positions
    )
    order = np.array([2, 0, 1])
    reverse = _oracle_replay(
        token[order], weight, slab, [slots[i] for i in order],
        dilation=dilation, positions=positions,
    )
    np.testing.assert_array_equal(forward[0][order].view(np.uint32),
                                  reverse[0].view(np.uint32))
    np.testing.assert_array_equal(forward[1].view(np.uint32),
                                  reverse[1].view(np.uint32))


def test_oracle_kernel_size_one_has_no_history() -> None:
    """``(kernel_size - 1) * dilation == 0`` is a stateless pass-through."""

    channels = 16
    rng = np.random.default_rng(7)
    token_row = rng.normal(0.0, 0.4, channels).astype(np.float32)
    weight = rng.normal(0.0, 0.3, (channels, 1)).astype(np.float32)

    out, state = _oracle_step(
        np.zeros((0, channels), np.float32), token_row, weight,
        dilation=1, position=3,
    )
    assert state.shape == (0, channels)
    np.testing.assert_array_equal(out.view(np.uint32),
                                  (token_row * weight[:, 0]).view(np.uint32))
