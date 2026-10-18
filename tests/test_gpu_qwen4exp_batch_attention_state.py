"""GPU ownership proof for the per-slot Qwen4Exp batch attention state.

The claim under test is that ``Qwen4ExpBatchAttentionState`` gives N independent
requests the same K/V they would each get from their own c1 state, through the
existing batch paged-write and batch dense-attention wrappers, without any
cross-request aliasing.

Protocol per case:

* ``prepare(slot_ids, positions)`` admits the rows and uploads per-row tables.
* Tokens ``0 .. position - 1`` are bulk-filled into the request's own physical
  blocks (setup; a real prefill writes them through the prompt path).
* The token at ``position`` is appended through
  ``qwen35_write_paged_kv_f32_batch_spans`` (or the single-request wrapper for
  the c1 reference).
* The decode row is produced by
  ``qwen35_paged_full_attn_decode_context_bf16_batch_spans`` (or the
  single-request wrapper for the c1 reference).

The c1 reference uses the production ``Qwen4ExpDenseAttentionState`` with the
same ``max_positions``, so the comparison also proves that batching did not
shrink any request's capacity.

Arithmetic note: the batch decode kernel selects a 1024-thread geometry when
``rows <= 2 and max_context_len < 1024`` while the c1 kernel always uses 256
threads, so only that regime folds the softmax denominator in a different warp
order. ``_assert_matches`` therefore asserts bit-exactness wherever the two
launches share geometry and a measured bound (printed) in the one regime where
the kernels do not promise parity.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.qwen4_exp import qsa_sparse_gqa_attention
from hipengine.kernels.hip_gfx1100.attention import (
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_write_paged_kv_f32_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_f32_batch_spans,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpDenseAttentionState

SLOTS = 4
MAX_POSITIONS = 1280
BLOCK_SIZE = 256
KV_HEADS = 2
Q_HEADS = 4
HEAD_DIM = 64
SCALE = float(1.0 / np.sqrt(HEAD_DIM))
# CPU-oracle bound: the reference accumulates the softmax and the value sum in a
# different order than the warp-strided kernel.
ORACLE_RTOL = 5e-5
ORACLE_ATOL = 5e-5
# Bound for the one regime where the batch and c1 launches use different
# thread counts. Measured deltas are printed by ``_assert_matches``.
MIXED_GEOMETRY_ATOL = 1e-5


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()

pytestmark = pytest.mark.skipif(
    not HIP_AVAILABLE, reason="HIP runtime is not available"
)


@pytest.fixture(scope="module", autouse=True)
def _build_for_detected_target(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment

    with hip_target_arch_environment(hip_test_target_arch):
        yield


class _Buffers:
    """Owns every raw device allocation made by one test case."""

    def __init__(self) -> None:
        self._buffers = []

    def from_host(self, array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes)
        self._buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
        return buffer

    def empty(self, nbytes: int):
        buffer = malloc(int(nbytes))
        self._buffers.append(buffer)
        return buffer

    def close(self) -> None:
        for buffer in reversed(self._buffers):
            free(buffer)
        self._buffers.clear()


def _from_device(buffer, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes)
    return out


def _assert_matches(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rows: int,
    max_context_len: int,
    label: str,
) -> None:
    delta = float(np.abs(actual - expected).max())
    exact_geometry = rows > 2 or max_context_len >= 1024
    print(
        f"{label}: rows={rows} max_context_len={max_context_len} "
        f"max|c1-batch|={delta:.3e} geometry_matched={exact_geometry}"
    )
    if exact_geometry:
        assert np.array_equal(actual, expected), delta
    else:
        assert delta < MIXED_GEOMETRY_ATOL, delta


def _paged_host_cache(
    total_blocks: int,
    placements: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Place per-request token runs into their own physical blocks."""

    cache = np.zeros((total_blocks * BLOCK_SIZE, KV_HEADS, HEAD_DIM), dtype=np.uint16)
    for physical_ids, tokens_bits in placements:
        for logical, physical in enumerate(physical_ids):
            start = logical * BLOCK_SIZE
            stop = min(start + BLOCK_SIZE, tokens_bits.shape[0])
            if stop <= start:
                break
            base = int(physical) * BLOCK_SIZE
            cache[base : base + (stop - start)] = tokens_bits[start:stop]
    return cache


def _stack_rows(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack per-row arrays with a common padded token axis.

    Rows keep their own context lengths, so the token axis is zero-padded to the
    longest row. Padding is never read: the bulk fill stops at ``position`` and
    the append reads exactly ``position``.
    """

    longest = max(array.shape[0] for array in arrays)
    if all(array.shape[0] == longest for array in arrays):
        return np.stack(arrays)
    padded = [
        np.concatenate(
            [array, np.zeros((longest - array.shape[0],) + array.shape[1:], array.dtype)],
            axis=0,
        )
        for array in arrays
    ]
    return np.stack(padded)


def _inputs(
    rows: int, positions: list[int], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_ctx = max(positions) + 1
    query = rng.normal(0.0, 0.5, size=(rows, Q_HEADS, HEAD_DIM)).astype(np.float32)
    key = rng.normal(0.0, 0.5, size=(rows, max_ctx, KV_HEADS, HEAD_DIM)).astype(np.float32)
    value = rng.normal(0.0, 0.5, size=(rows, max_ctx, KV_HEADS, HEAD_DIM)).astype(np.float32)
    return query, key, value


def _fill_state_cache(
    state,
    slots: list[int],
    positions: list[int],
    key: np.ndarray,
    value: np.ndarray,
    total_blocks: int,
    blocks_of,
) -> None:
    """Bulk-fill tokens ``0 .. position - 1`` for each active row."""

    key_host = _paged_host_cache(
        total_blocks,
        [
            (blocks_of(slot), float_array_to_bf16_bits(key[row][: positions[row]]))
            for row, slot in enumerate(slots)
        ],
    )
    value_host = _paged_host_cache(
        total_blocks,
        [
            (blocks_of(slot), float_array_to_bf16_bits(value[row][: positions[row]]))
            for row, slot in enumerate(slots)
        ],
    )
    copy_host_to_device(state.key_cache, host_array_ptr(key_host), key_host.nbytes)
    copy_host_to_device(state.value_cache, host_array_ptr(value_host), value_host.nbytes)


def _run_batch_case(
    state: Qwen4ExpBatchAttentionState,
    buffers: _Buffers,
    slots: list[int],
    positions: list[int],
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
) -> tuple[np.ndarray, int]:
    rows = len(slots)
    active = state.prepare(slots, positions)
    _fill_state_cache(
        state,
        slots,
        positions,
        key,
        value,
        state.total_capacity_tokens // state.block_size,
        state.blocks_for_slot,
    )
    # The appended token goes through the batch wrapper under test.
    last_key = buffers.from_host(key[np.arange(rows), positions])
    last_value = buffers.from_host(value[np.arange(rows), positions])
    qwen35_write_paged_kv_f32_batch_spans(
        last_key.ptr,
        last_value.ptr,
        state.key_cache.ptr,
        state.value_cache.ptr,
        active.append_spans,
        rows,
        state.block_size,
        KV_HEADS,
        HEAD_DIM,
    )
    query_buffer = buffers.from_host(query)
    out = buffers.empty(rows * Q_HEADS * HEAD_DIM * 4)
    qwen35_paged_full_attn_decode_context_bf16_batch_spans(
        query_buffer.ptr,
        state.key_cache.ptr,
        state.value_cache.ptr,
        out.ptr,
        active.decode_spans,
        rows,
        active.max_context_len,
        state.block_size,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        SCALE,
    )
    return _from_device(out, (rows, Q_HEADS, HEAD_DIM), np.float32), active.max_context_len


def _run_c1_case(
    buffers: _Buffers,
    position: int,
    query_row: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    """The production single-request path for one row."""

    state = Qwen4ExpDenseAttentionState.allocate(
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )
    try:
        _fill_state_cache(
            state,
            [0],
            [position],
            key[None],
            value[None],
            len(state.block_host),
            lambda _slot: state.block_host,
        )
        state.set_position(position)
        last_key = buffers.from_host(key[position][None])
        last_value = buffers.from_host(value[position][None])
        qwen35_write_paged_kv_f32_spans(
            last_key.ptr,
            last_value.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            state.append_spans,
            BLOCK_SIZE,
            KV_HEADS,
            HEAD_DIM,
        )
        query_buffer = buffers.from_host(query_row[None])
        out = buffers.empty(Q_HEADS * HEAD_DIM * 4)
        qwen35_paged_full_attn_decode_context_bf16_spans(
            query_buffer.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            out.ptr,
            state.decode_spans,
            position + 1,
            BLOCK_SIZE,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            SCALE,
        )
        return _from_device(out, (Q_HEADS, HEAD_DIM), np.float32)
    finally:
        state.close()


def _c1_rows(
    buffers: _Buffers,
    positions: list[int],
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            _run_c1_case(buffers, positions[row], query[row], key[row], value[row])
            for row in range(len(positions))
        ]
    )


def _oracle_row(
    position: int, query_row: np.ndarray, key: np.ndarray, value: np.ndarray
) -> np.ndarray:
    context = position + 1
    return qsa_sparse_gqa_attention(
        query_row[None],
        bf16_to_float32(float_array_to_bf16_bits(key[:context])),
        bf16_to_float32(float_array_to_bf16_bits(value[:context])),
        query_positions=[position],
        key_positions=np.arange(context),
        selected_positions=[np.arange(context)],
        scale=SCALE,
    )[0]


def _state() -> Qwen4ExpBatchAttentionState:
    return Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )


def test_batch_n4_block_boundary_and_permuted_slots_match_c1() -> None:
    slots = [2, 0, 3, 1]
    positions = [3, 255, 256, 700]
    query, key, value = _inputs(len(slots), positions, seed=4104)
    buffers = _Buffers()
    state = _state()
    try:
        actual, max_context_len = _run_batch_case(
            state, buffers, slots, positions, query, key, value
        )
        expected = _c1_rows(buffers, positions, query, key, value)
    finally:
        state.close()
        buffers.close()

    assert actual.dtype == np.float32
    _assert_matches(
        actual,
        expected,
        rows=len(slots),
        max_context_len=max_context_len,
        label="n4 boundary",
    )
    oracle = np.stack(
        [
            _oracle_row(positions[row], query[row], key[row], value[row])
            for row in range(len(positions))
        ]
    )
    print("n4 boundary: max|oracle-batch|", np.abs(actual - oracle).max())
    np.testing.assert_allclose(actual, oracle, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)


def test_batch_n4_long_context_matches_c1() -> None:
    slots = [3, 0, 2, 1]
    positions = [1023, 700, 255, 3]
    query, key, value = _inputs(len(slots), positions, seed=4105)
    buffers = _Buffers()
    state = _state()
    try:
        actual, max_context_len = _run_batch_case(
            state, buffers, slots, positions, query, key, value
        )
        expected = _c1_rows(buffers, positions, query, key, value)
    finally:
        state.close()
        buffers.close()

    _assert_matches(
        actual,
        expected,
        rows=len(slots),
        max_context_len=max_context_len,
        label="n4 long",
    )
    oracle = _oracle_row(positions[0], query[0], key[0], value[0])
    print("n4 long row0: max|oracle-batch|", np.abs(actual[0] - oracle).max())
    np.testing.assert_allclose(actual[0], oracle, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)


def test_batch_n2_distinct_contexts_match_c1() -> None:
    slots = [1, 3]
    positions = [255, 1023]
    query, key, value = _inputs(len(slots), positions, seed=4106)
    buffers = _Buffers()
    state = _state()
    try:
        active = state.prepare(slots, positions)
        assert active.rows == 2
        assert active.max_context_len == 1024
        actual, max_context_len = _run_batch_case(
            state, buffers, slots, positions, query, key, value
        )
        expected = _c1_rows(buffers, positions, query, key, value)
    finally:
        state.close()
        buffers.close()

    _assert_matches(
        actual,
        expected,
        rows=len(slots),
        max_context_len=max_context_len,
        label="n2",
    )


def test_batch_n1_block_boundary_matches_c1() -> None:
    slots = [2]
    positions = [256]
    query, key, value = _inputs(len(slots), positions, seed=4107)
    buffers = _Buffers()
    state = _state()
    try:
        actual, max_context_len = _run_batch_case(
            state, buffers, slots, positions, query, key, value
        )
        expected = _c1_rows(buffers, positions, query, key, value)
    finally:
        state.close()
        buffers.close()

    # rows=1 with max_context_len=257 takes the batch kernel's 1024-thread
    # geometry, so this is the mixed-geometry regime: bounded, not bit-exact.
    _assert_matches(
        actual,
        expected,
        rows=len(slots),
        max_context_len=max_context_len,
        label="n1 boundary",
    )
    oracle = _oracle_row(positions[0], query[0], key[0], value[0])
    print("n1 boundary: max|oracle-batch|", np.abs(actual[0] - oracle).max())
    np.testing.assert_allclose(actual[0], oracle, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)


def test_untouched_slots_and_unwritten_blocks_stay_zero() -> None:
    slots = [0, 2]
    positions = [3, 700]
    query, key, value = _inputs(len(slots), positions, seed=4108)
    buffers = _Buffers()
    state = _state()
    try:
        _run_batch_case(state, buffers, slots, positions, query, key, value)
        key_view = _from_device(
            state.key_cache,
            (state.total_capacity_tokens, KV_HEADS, HEAD_DIM),
            np.uint16,
        )
        value_view = _from_device(
            state.value_cache,
            (state.total_capacity_tokens, KV_HEADS, HEAD_DIM),
            np.uint16,
        )
        assert state.total_capacity_tokens == SLOTS * MAX_POSITIONS
        for view in (key_view, value_view):
            # Slots 1 and 3 were never admitted, so no block of theirs may hold
            # a byte from slots 0 or 2.
            for idle_slot in (1, 3):
                blocks = state.blocks_for_slot(idle_slot)
                idle = view[
                    blocks[0] * BLOCK_SIZE : (blocks[-1] + 1) * BLOCK_SIZE
                ]
                assert not idle.any(), f"slot {idle_slot} was written by another request"
            # Within a written slot, everything above the appended position is
            # still the allocate-time zero.
            for row, slot in enumerate(slots):
                blocks = state.blocks_for_slot(slot)
                written = positions[row] + 1
                start = blocks[0] * BLOCK_SIZE
                above = view[start + written : (blocks[-1] + 1) * BLOCK_SIZE]
                assert not above.any(), f"slot {slot} wrote past its position"
                assert view[start : start + written].any()
    finally:
        state.close()
        buffers.close()


def test_decode_is_deterministic_across_reruns_and_reprepares() -> None:
    slots = [3, 1]
    positions = [4, 300]
    query, key, value = _inputs(len(slots), positions, seed=4109)
    buffers = _Buffers()
    state = _state()
    try:
        first, _ = _run_batch_case(
            state, buffers, slots, positions, query, key, value
        )
        # Same active metadata, same cache: the row tables must be re-uploaded
        # identically and the decode must reproduce the bytes.
        second, _ = _run_batch_case(
            state, buffers, slots, positions, query, key, value
        )
        assert np.array_equal(first, second)
        # A permuted prepare that covers the same slots must keep each row's
        # output attached to its slot, not to its row index.
        permuted, _ = _run_batch_case(
            state,
            buffers,
            [1, 3],
            [300, 4],
            query[::-1],
            key[::-1],
            value[::-1],
        )
        assert np.array_equal(permuted, first[::-1])
    finally:
        state.close()
        buffers.close()

    # A fresh state instance must produce the same bytes as the reused one.
    fresh_buffers = _Buffers()
    fresh = _state()
    try:
        third, _ = _run_batch_case(
            fresh, fresh_buffers, slots, positions, query, key, value
        )
    finally:
        fresh.close()
        fresh_buffers.close()
    assert np.array_equal(third, first)


def test_slot_state_is_isolated_across_prepare_rewrites() -> None:
    buffers = _Buffers()
    state = _state()
    first_slots = [0, 2]
    first_positions = [5, 256]
    query_a, key_a, value_a = _inputs(len(first_slots), first_positions, seed=4110)
    second_positions = [301]
    query_b, key_b, value_b = _inputs(1, second_positions, seed=4111)
    try:
        _run_batch_case(
            state, buffers, first_slots, first_positions, query_a, key_a, value_a
        )
        # Rewriting slot 1 must not disturb slots 0 or 2.
        _run_batch_case(
            state, buffers, [1], second_positions, query_b, key_b, value_b
        )
        combined, max_context_len = _run_batch_case(
            state,
            buffers,
            [2, 1, 0],
            [first_positions[1], second_positions[0], first_positions[0]],
            np.stack([query_a[1], query_b[0], query_a[0]]),
            _stack_rows([key_a[1], key_b[0], key_a[0]]),
            _stack_rows([value_a[1], value_b[0], value_a[0]]),
        )
        expected_a = _c1_rows(buffers, first_positions, query_a, key_a, value_a)
        expected_b = _c1_rows(buffers, second_positions, query_b, key_b, value_b)
    finally:
        state.close()
        buffers.close()

    assert max_context_len == 302
    for row, expected in (
        (0, expected_a[1]),
        (1, expected_b[0]),
        (2, expected_a[0]),
    ):
        _assert_matches(
            combined[row],
            expected,
            rows=3,
            max_context_len=max_context_len,
            label=f"isolation row {row}",
        )


def test_active_width_shrink_and_regrow_reuses_the_same_slots() -> None:
    positions = [9, 40, 300, 1023]
    query, key, value = _inputs(SLOTS, positions, seed=4112)
    buffers = _Buffers()
    state = _state()
    try:
        # Grow 1 -> 2 -> 4 rows on the same state, one slot at a time, and check
        # every width against the c1 path for exactly that active set.
        for width in (1, 2, 4):
            slots = list(range(width))
            active = state.prepare(slots, positions[:width])
            assert active.rows == width
            assert state.state_slots == SLOTS
            actual, max_context_len = _run_batch_case(
                state,
                buffers,
                slots,
                positions[:width],
                query[:width],
                key[:width],
                value[:width],
            )
            expected = _c1_rows(
                buffers,
                positions[:width],
                query[:width],
                key[:width],
                value[:width],
            )
            _assert_matches(
                actual,
                expected,
                rows=width,
                max_context_len=max_context_len,
                label=f"width {width}",
            )
        # Shrink back to one row and confirm the surviving slot is unchanged.
        actual, max_context_len = _run_batch_case(
            state, buffers, [3], [positions[3]], query[3:], key[3:], value[3:]
        )
        expected = _c1_rows(
            buffers, [positions[3]], query[3:], key[3:], value[3:]
        )
    finally:
        state.close()
        buffers.close()
    _assert_matches(
        actual,
        expected,
        rows=1,
        max_context_len=max_context_len,
        label="shrink to slot 3",
    )
