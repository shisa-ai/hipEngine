"""CPU contracts for the indexed Qwen4Exp GDN decode primitive.

Covers the two host-side pieces of the indexed decode ABI:

* the per-row state-slot index admission that bounds an independent-request
  batch to a declared state slab, and
* the CPU oracle the GPU test compares against, so a wrong ``cu_seqlens`` or
  ``state_indices`` construction is caught without a device.

The oracle is the committed segmented recurrence
(:func:`hipengine.kernels.cpu_reference.ops.gdn_prefill_recurrent_segments`)
driven with one-token segments. That is the same shape the GPU primitive uses:
row ``r`` reads and writes exactly the state slot ``state_indices[r]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.ops import gdn_prefill_recurrent_segments
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    QWEN4_EXP_GDN_INDEXED_MAX_ROWS,
    QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS,
    qwen4_exp_gdn_decode_rowsplit_indexed_f32,
    qwen4_exp_gdn_indexed_state_indices,
)


def _segments(rows: int) -> np.ndarray:
    """One-token segments over a packed row block."""

    return np.arange(rows + 1, dtype=np.int64)


def _prepared(rng, rows, v_heads, dim):
    return (
        rng.normal(0.0, 0.5, (rows, v_heads, dim)).astype(np.float32),
        rng.normal(0.0, 0.5, (rows, v_heads, dim)).astype(np.float32),
        rng.normal(0.0, 0.5, (rows, v_heads, dim)).astype(np.float32),
        rng.uniform(0.1, 0.9, (rows, v_heads)).astype(np.float32),
        rng.uniform(0.5, 0.99, (rows, v_heads)).astype(np.float32),
    )


def test_state_indices_accepts_permuted_distinct_slots() -> None:
    indices = qwen4_exp_gdn_indexed_state_indices(
        [3, 1, 5, 0], rows=4, state_slots=6
    )
    assert indices.dtype == np.int64
    assert indices.tolist() == [3, 1, 5, 0]
    assert indices.flags["C_CONTIGUOUS"]


def test_state_indices_accepts_numpy_and_permuted_values() -> None:
    indices = qwen4_exp_gdn_indexed_state_indices(
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
    """Two independent rows must not name one recurrent slot.

    A duplicate would have both rows read the same initial state and both write
    it, so the result depends on block completion order: the surviving state is
    one row's update plus the other's, or just one of them. That is a violation
    of exact ownership, not numerical drift, so it is rejected on the host with a
    diagnosable error instead of being left to the device.
    """

    with pytest.raises(ValueError, match="duplicate"):
        qwen4_exp_gdn_indexed_state_indices(
            indices, rows=rows, state_slots=state_slots
        )


def test_state_indices_duplicate_error_names_the_slot_and_both_rows() -> None:
    with pytest.raises(ValueError) as error:
        qwen4_exp_gdn_indexed_state_indices([4, 1, 4], rows=3, state_slots=5)
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
        ([0] * (QWEN4_EXP_GDN_INDEXED_MAX_ROWS + 1),
         QWEN4_EXP_GDN_INDEXED_MAX_ROWS + 1,
         QWEN4_EXP_GDN_INDEXED_MAX_ROWS + 1,
         "rows must be in 1"),
        ([0.0, 1.0], 2, 4, "integers"),
        (["0", "1"], 2, 4, "integers"),
        (0, 2, 4, "sequence"),
    ],
)
def test_state_indices_reject_unbounded_or_malformed_batches(
    indices, rows, state_slots, match
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        qwen4_exp_gdn_indexed_state_indices(
            indices, rows=rows, state_slots=state_slots
        )


def test_state_indices_rejects_a_slab_wider_than_the_device_check() -> None:
    """The device duplicate scan is a bounded bitmap, so the slab is bounded."""

    limit = QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS
    assert limit >= QWEN4_EXP_GDN_INDEXED_MAX_ROWS
    with pytest.raises(ValueError, match="state_slots must be in"):
        qwen4_exp_gdn_indexed_state_indices([0, 1], rows=2, state_slots=limit + 1)
    accepted = qwen4_exp_gdn_indexed_state_indices(
        [0, limit - 1], rows=2, state_slots=limit
    )
    assert accepted.tolist() == [0, limit - 1]


def test_state_indices_boundary_is_inclusive() -> None:
    limit = QWEN4_EXP_GDN_INDEXED_MAX_ROWS
    indices = qwen4_exp_gdn_indexed_state_indices(
        list(range(limit)), rows=limit, state_slots=limit
    )
    assert indices.size == limit


@pytest.mark.parametrize("v_heads,dim", [(2, 4), (3, 8)])
def test_oracle_one_token_segments_match_per_slot_sequential_replay(
    v_heads: int, dim: int
) -> None:
    """The oracle's permuted one-token segments equal per-slot replay."""

    rng = np.random.default_rng(4242 + v_heads * dim)
    rows = 4
    query, key, value, beta, decay = _prepared(rng, rows, v_heads, dim)
    slots = np.array([3, 0, 2, 1], dtype=np.int64)
    initial = rng.normal(0.0, 0.05, (4, v_heads, dim, dim)).astype(np.float32)

    batched_out, batched_state = gdn_prefill_recurrent_segments(
        query, key, value, beta, decay, initial, _segments(rows), slots
    )

    replay_out = np.empty_like(batched_out)
    replay_state = initial.copy()
    for row, slot in enumerate(slots.tolist()):
        single_out, single_state = gdn_prefill_recurrent_segments(
            query[row : row + 1],
            key[row : row + 1],
            value[row : row + 1],
            beta[row : row + 1],
            decay[row : row + 1],
            replay_state[int(slot) : int(slot) + 1],
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
        replay_out[row] = single_out[0]
        replay_state[int(slot)] = single_state[0]

    np.testing.assert_array_equal(batched_out, replay_out)
    np.testing.assert_array_equal(batched_state, replay_state)
    # Slots no row selected are untouched, which is the canary property the
    # GPU test asserts on device.
    np.testing.assert_array_equal(batched_state[0], replay_state[0])


def test_oracle_is_order_independent_across_rows() -> None:
    """Permuting rows with their slots permutes the outputs, not the state."""

    rng = np.random.default_rng(90210)
    rows, v_heads, dim = 3, 2, 4
    query, key, value, beta, decay = _prepared(rng, rows, v_heads, dim)
    initial = rng.normal(0.0, 0.05, (3, v_heads, dim, dim)).astype(np.float32)
    slots = np.array([2, 0, 1], dtype=np.int64)
    order = np.array([2, 0, 1], dtype=np.int64)

    base_out, base_state = gdn_prefill_recurrent_segments(
        query, key, value, beta, decay, initial, _segments(rows), slots
    )
    permuted_out, permuted_state = gdn_prefill_recurrent_segments(
        query[order],
        key[order],
        value[order],
        beta[order],
        decay[order],
        initial,
        _segments(rows),
        slots[order],
    )

    np.testing.assert_array_equal(permuted_state, base_state)
    np.testing.assert_array_equal(permuted_out, base_out[order])


def test_row_bound_matches_the_c_launcher() -> None:
    """The Python and C row bounds must agree.

    ``QWEN4_EXP_GDN_INDEXED_MAX_ROWS`` is the host admission bound and
    ``kQwen4ExpGdnIndexedMaxRows`` is the device-side guard. The host check runs
    first, so a drift that widened only the Python side would silently make the
    C bound dead code. This guard was added after the primitive landed; it pins
    the pairing rather than describing new behaviour.
    """

    source = (
        Path(__file__).resolve().parents[1]
        / "hipengine/kernels/hip_gfx1100/linear_attn/qwen4_exp_gdn.hip"
    ).read_text()
    assert (
        f"constexpr int64_t kQwen4ExpGdnIndexedMaxRows = {QWEN4_EXP_GDN_INDEXED_MAX_ROWS};"
        in source
    )
    assert (
        "constexpr int64_t kQwen4ExpGdnIndexedMaxStateSlots = "
        f"{QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS};"
        in source
    )


# ---------------------------------------------------------------------------
# Stride and alignment admission.
# ---------------------------------------------------------------------------

_HEADS = {
    "num_k_heads": 16,
    "num_v_heads": 48,
    "head_k_dim": 128,
    "head_v_dim": 128,
}
_CONV_WIDTH = 2 * 16 * 128 + 48 * 128
_CORE_WIDTH = 48 * 128


def _admit(**overrides):
    """Call the wrapper far enough to validate, without any device access.

    Validation runs before the library/runtime defaults are resolved, so passing
    placeholders proves the rejection happens on the host.
    """

    kwargs = {
        "rows": 2,
        "state_slots": 2,
        **_HEADS,
        **overrides,
    }
    return qwen4_exp_gdn_decode_rowsplit_indexed_f32(
        *([16] * 10), runtime=object(), library=object(), **kwargs
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        # A stride shorter than one packed row would make adjacent rows overlap.
        ({"conv_row_stride": _CONV_WIDTH - 4}, "conv_row_stride is shorter"),
        ({"conv_row_stride": _CONV_WIDTH - 1}, "conv_row_stride is shorter"),
        ({"scalar_row_stride": 47}, "scalar_row_stride is shorter"),
        ({"gate_row_stride": _CORE_WIDTH - 4}, "gate_row_stride is shorter"),
        ({"output_row_stride": _CORE_WIDTH - 1}, "output_row_stride is shorter"),
        # Non-positive strides are rejected before the width comparison.
        ({"conv_row_stride": 0}, "conv_row_stride must be positive"),
        ({"scalar_row_stride": 0}, "scalar_row_stride must be positive"),
        ({"gate_row_stride": 0}, "gate_row_stride must be positive"),
        ({"output_row_stride": -1}, "output_row_stride must be positive"),
        # The Conv Q/K loads are float4, so every row base must stay 16-byte
        # aligned; that needs the stride to be a whole number of float4s.
        ({"conv_row_stride": _CONV_WIDTH + 1}, "multiple of 4 floats"),
        ({"conv_row_stride": _CONV_WIDTH + 2}, "multiple of 4 floats"),
        ({"conv_row_stride": _CONV_WIDTH + 3}, "multiple of 4 floats"),
    ],
)
def test_stride_admission(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _admit(**overrides)


@pytest.mark.parametrize("position,match", [(0, "16-byte aligned"), (9, "8-byte aligned")])
def test_pointer_alignment_admission(position, match) -> None:
    """The Conv Q/K loads are float4 and the slot vector is int64."""

    pointers = [16] * 10
    pointers[position] = 20  # 4-byte aligned only
    with pytest.raises(ValueError, match=match):
        qwen4_exp_gdn_decode_rowsplit_indexed_f32(
            *pointers, rows=2, state_slots=2, **_HEADS,
            runtime=object(), library=object(),
        )


def test_padded_strides_are_admitted() -> None:
    """Padding is legal as long as it stays aligned and at least as wide."""

    with pytest.raises(
        AttributeError, match="hipengine_qwen4_exp_gdn_decode_rowsplit_indexed_f32"
    ):
        # Admission passed, so execution reached the symbol lookup on the
        # placeholder library object. That is the assertion: no stride error.
        _admit(
            conv_row_stride=_CONV_WIDTH + 4,
            scalar_row_stride=56,
            gate_row_stride=_CORE_WIDTH + 4,
            output_row_stride=_CORE_WIDTH + 8,
        )
