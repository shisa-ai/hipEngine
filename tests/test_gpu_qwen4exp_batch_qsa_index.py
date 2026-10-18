"""N-request sparse QSA index state against independent c1 owners and a CPU oracle.

Three contracts, asserted separately per ``docs/EXECUTION-PROFILES.md``:

* **Ownership (exact).** Row-to-slot mapping, slot-disjoint key storage, untouched
  slots, row permutations, slot reuse, and the bounded rollback transaction are
  byte- or index-level assertions. A failure is a bug, not drift.
* **Arithmetic.** Append, pool, and score are the same committed primitives the
  c1 owner calls, with the same per-slot arguments, and selection is the same
  kernel body with a row dimension. The declared expectation is therefore
  *exact* equality with N independent ``Qwen4ExpQSAIndexDeviceState`` instances:
  every case asserts bit-identical counts and positions. The CPU oracle
  (``qsa_prepare_index_keys`` / ``qsa_index_scores`` / ``qsa_select_positions``)
  is a second, independent check of the pooled keys and of the selected
  position set.
* **Determinism.** Equal scores must resolve to the lowest block indices, and a
  repeat from the same state must reproduce the same selection.

Fixtures are compact: four slots, 16 positions each, ratio 2, budget 2, two
index heads of four dimensions. Distinct per-slot key streams make a row reading
the wrong slot impossible to miss.
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
from hipengine.kernels.cpu_reference.qwen4_exp import (
    qsa_index_scores,
    qsa_prepare_index_keys,
    qsa_select_positions,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpDenseAttentionState,
    Qwen4ExpQSAIndexDeviceState,
)

SLOTS = 4
MAX_POSITIONS = 16
BLOCK_SIZE = 8
RATIO = 2
BUDGET = 2
INDEX_HEADS = 2
INDEX_DIM = 4
THETA = 100.0
COMPLETE_BLOCKS = MAX_POSITIONS // RATIO
OUTPUT_STRIDE = BUDGET * RATIO + RATIO - 1
# Sparse selection is only defined past the dense equivalence bound.
DENSE_LIMIT = OUTPUT_STRIDE


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


@pytest.fixture(scope="module", autouse=True)
def _build_for_detected_target(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment

    with hip_target_arch_environment(hip_test_target_arch):
        yield


def _upload(array: np.ndarray, runtime, allocations) -> object:
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    assert host.nbytes <= device.nbytes, (host.nbytes, device.nbytes)
    copy_device_to_host(
        host_array_ptr(host), device, nbytes=host.nbytes, runtime=runtime
    )
    return host


def _key_stream(seed: int, count: int) -> np.ndarray:
    """A per-slot key stream whose scale identifies the slot."""

    rng = np.random.default_rng(seed)
    scale = 0.2 + 0.35 * (seed % 7)
    return (rng.normal(0.0, scale, size=(count, INDEX_DIM))).astype(np.float32)


def _queries(seed: int, rows: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.4, size=(rows, INDEX_HEADS, INDEX_DIM)).astype(
        np.float32
    )


class Reference:
    """One independent c1 index owner per slot: the strict comparison basis."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.allocations: list = []
        self.attention = Qwen4ExpDenseAttentionState.allocate(
            max_positions=MAX_POSITIONS,
            block_size=BLOCK_SIZE,
            kv_heads=1,
            head_dim=4,
            runtime=runtime,
        )
        self.states = [
            Qwen4ExpQSAIndexDeviceState.allocate(
                attention_state=self.attention,
                index_heads=INDEX_HEADS,
                index_dim=INDEX_DIM,
                compression_ratio=RATIO,
                block_budget=BUDGET,
                runtime=runtime,
            )
            for _ in range(SLOTS)
        ]

    def push(self, slot: int, keys: np.ndarray) -> None:
        device = _upload(keys, self.runtime, self.allocations)
        stride = INDEX_DIM * 4
        # Positions continue from the owner's own count, so a slot can be filled
        # in several calls exactly as the batch owner's cursor does.
        start = int(self.states[slot].count)
        for index in range(keys.shape[0]):
            self.states[slot].append(
                device.ptr + index * stride, position=start + index
            )

    def select(self, slot: int, query: np.ndarray, norm_ptr: int) -> np.ndarray:
        state = self.states[slot]
        device = _upload(query[None], self.runtime, self.allocations)
        count = state.select_positions_device(
            device.ptr,
            query_position=state.count - 1,
            key_norm_weight_ptr=norm_ptr,
            rotary_dim=INDEX_DIM,
            theta=THETA,
        )
        return _download(state.selected_positions, (count,), np.int64, self.runtime)

    def pooled(self, slot: int, blocks: int) -> np.ndarray:
        state = self.states[slot]
        flat = _download(
            state.pooled_keys,
            (COMPLETE_BLOCKS, INDEX_DIM),
            np.float32,
            self.runtime,
        )
        return flat[:blocks]

    def close(self) -> None:
        for state in self.states:
            state.close()
        self.attention.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


class Harness:
    """The batch index owner driven through the batch attention owner."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.allocations: list = []
        self.attention = Qwen4ExpBatchAttentionState.allocate(
            state_slots=SLOTS,
            max_positions=MAX_POSITIONS,
            block_size=BLOCK_SIZE,
            kv_heads=1,
            head_dim=4,
            runtime=runtime,
        )
        self.index = Qwen4ExpBatchQsaIndexState.allocate(
            attention_state=self.attention,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=runtime,
        )

    def push(self, slot: int, keys: np.ndarray) -> None:
        device = _upload(keys, self.runtime, self.allocations)
        stride = INDEX_DIM * 4
        for index in range(keys.shape[0]):
            cursor = int(self.index.cursor_host[slot])
            self.attention.prepare([slot], [cursor])
            self.index.prepare([slot])
            self.index.append_active(device.ptr + index * stride)

    def select(
        self, slots, queries: np.ndarray, norm_ptr: int
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = [int(self.index.cursor_host[slot]) - 1 for slot in slots]
        self.attention.prepare(slots, positions)
        self.index.prepare(slots)
        device = _upload(queries, self.runtime, self.allocations)
        return self.index.select_active(
            device.ptr,
            key_norm_weight_ptr=norm_ptr,
            rotary_dim=INDEX_DIM,
            theta=THETA,
        )

    def raw_slab(self) -> np.ndarray:
        return _download(
            self.index.raw_keys,
            (SLOTS * self.index.slot_capacity_tokens, INDEX_DIM),
            np.float32,
            self.runtime,
        )

    def pooled_slab(self) -> np.ndarray:
        return _download(
            self.index.pooled_keys,
            (SLOTS * COMPLETE_BLOCKS, INDEX_DIM),
            np.float32,
            self.runtime,
        )

    def close(self) -> None:
        self.index.close()
        self.attention.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


@pytest.fixture
def rig():
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    allocations: list = []
    norm = np.asarray([1.0, 0.9, 1.1, 0.95], dtype=np.float32)
    norm_ptr = _upload(norm, runtime, allocations).ptr
    harness = Harness(runtime)
    reference = Reference(runtime)
    try:
        yield SimpleRig(runtime, harness, reference, norm, norm_ptr, allocations)
    finally:
        reference.close()
        harness.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


class SimpleRig:
    def __init__(self, runtime, harness, reference, norm, norm_ptr, allocations):
        self.runtime = runtime
        self.harness = harness
        self.reference = reference
        self.norm = norm
        self.norm_ptr = norm_ptr
        self.allocations = allocations

    def oracle(self, keys: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pooled keys and selected position set from the CPU reference."""

        pooled = qsa_prepare_index_keys(
            keys,
            np.arange(keys.shape[0], dtype=np.int64),
            self.norm,
            compression_ratio=RATIO,
            rotary_dim=INDEX_DIM,
            theta=THETA,
        )
        scores = qsa_index_scores(query[None], pooled.keys)
        selection = qsa_select_positions(
            scores,
            pooled.block_starts,
            query_positions=[keys.shape[0] - 1],
            available_positions=np.arange(keys.shape[0], dtype=np.int64),
            compression_ratio=RATIO,
            block_budget=BUDGET,
        )
        return pooled.keys, selection.selected_positions[0]


@pytest.mark.parametrize(
    ("seed", "slots", "lengths"),
    [
        (6101, [1], [6]),
        (6102, [2, 0], [6, 8]),
        (6103, [3, 1, 0, 2], [6, 7, 8, 9]),
        # Compression boundaries: (6, 8) end on a block edge (no tail) while
        # (7, 9) leave a one-token tail.
        (6104, [1, 2], [8, 7]),
        (6105, [0, 3], [9, 6]),
        # Sparse budget overflow: eight complete blocks against a budget of two.
        (6106, [2, 1], [16, 13]),
    ],
)
def test_batch_index_matches_c1_and_cpu_oracle(rig, seed, slots, lengths) -> None:
    histories = {
        slot: _key_stream(seed * 10 + slot, length)
        for slot, length in zip(slots, lengths)
    }
    queries = _queries(seed + 1, len(slots))
    for slot, keys in histories.items():
        rig.harness.push(slot, keys)
        rig.reference.push(slot, keys)
        assert int(rig.harness.index.cursor_host[slot]) == keys.shape[0]

    counts, positions = rig.harness.select(slots, queries, rig.norm_ptr)
    assert counts.shape == (len(slots),)
    for row, slot in enumerate(slots):
        length = lengths[row]
        blocks = length // RATIO
        assert blocks > BUDGET, (slot, length, blocks)
        expected = rig.reference.select(slot, queries[row], rig.norm_ptr)
        assert int(counts[row]) == expected.size
        # Exact: same primitives, same per-slot arguments, same selection kernel.
        np.testing.assert_array_equal(positions[row, : counts[row]], expected)

        oracle_pooled, oracle_positions = rig.oracle(histories[slot], queries[row])
        assert np.array_equal(np.sort(expected), oracle_positions), (
            slot,
            length,
        )
        np.testing.assert_allclose(
            rig.harness.pooled_slab()[slot * COMPLETE_BLOCKS :][:blocks],
            oracle_pooled,
            rtol=1e-5,
            atol=1e-6,
        )
        # The tail rule is the c1 rule: a partial block contributes its tokens.
        tail = length - blocks * RATIO
        assert expected.size == BUDGET * RATIO + tail
        np.testing.assert_array_equal(
            expected[BUDGET * RATIO :], np.arange(blocks * RATIO, length)
        )


def test_batch_index_ties_resolve_to_the_lowest_blocks(rig) -> None:
    # Zero keys pool, normalize, and rotate to zero, so every block scores
    # exactly 0.0 and the strict tie contract must pick the lowest indices.
    slots = [2, 0]
    lengths = [8, 9]
    zero_keys = {slot: np.zeros((length, INDEX_DIM), np.float32) for slot, length in zip(slots, lengths)}
    for slot, keys in zero_keys.items():
        rig.harness.push(slot, keys)
        rig.reference.push(slot, keys)
    queries = _queries(6201, len(slots))
    counts, positions = rig.harness.select(slots, queries, rig.norm_ptr)
    for row, slot in enumerate(slots):
        blocks = lengths[row] // RATIO
        tail = lengths[row] - blocks * RATIO
        expected = np.concatenate(
            (
                np.arange(BUDGET * RATIO, dtype=np.int64),
                np.arange(blocks * RATIO, lengths[row], dtype=np.int64),
            )
        )
        assert int(counts[row]) == BUDGET * RATIO + tail
        np.testing.assert_array_equal(positions[row, : counts[row]], expected)
        # The independent c1 owner agrees, and so does the CPU oracle.
        np.testing.assert_array_equal(
            rig.reference.select(slot, queries[row], rig.norm_ptr), expected
        )
        _, oracle_positions = rig.oracle(zero_keys[slot], queries[row])
        np.testing.assert_array_equal(np.sort(expected), oracle_positions)


def test_batch_index_leaves_untouched_slots_byte_identical(rig) -> None:
    histories = {slot: _key_stream(6300 + slot, 8) for slot in range(SLOTS)}
    for slot in (0, 2):
        rig.harness.push(slot, histories[slot])
    raw_before = rig.harness.raw_slab()
    pooled_before = rig.harness.pooled_slab()
    cursors_before = rig.harness.index.cursor_host.copy()

    queries = _queries(6301, 2)
    rig.harness.select([0, 2], queries, rig.norm_ptr)

    raw_after = rig.harness.raw_slab()
    pooled_after = rig.harness.pooled_slab()
    for slot in (1, 3):
        window = slice(slot * rig.harness.index.slot_capacity_tokens, (slot + 1) * rig.harness.index.slot_capacity_tokens)
        np.testing.assert_array_equal(
            raw_after[window].view(np.uint32), raw_before[window].view(np.uint32)
        )
        pooled_window = slice(slot * COMPLETE_BLOCKS, (slot + 1) * COMPLETE_BLOCKS)
        np.testing.assert_array_equal(
            pooled_after[pooled_window].view(np.uint32),
            pooled_before[pooled_window].view(np.uint32),
        )
    np.testing.assert_array_equal(
        rig.harness.index.cursor_host[[1, 3]], cursors_before[[1, 3]]
    )
    # The selected slots did move.
    assert rig.harness.index.cursor_host[0] == cursors_before[0]
    assert rig.harness.index.pooled_count_host[0] == 4


def test_batch_index_row_permutation_follows_the_slot(rig) -> None:
    histories = {slot: _key_stream(6400 + slot, length) for slot, length in ((0, 7), (2, 9))}
    for slot, keys in histories.items():
        rig.harness.push(slot, keys)
        rig.reference.push(slot, keys)
    queries = _queries(6401, 2)

    counts_a, positions_a = rig.harness.select([2, 0], queries, rig.norm_ptr)
    counts_b, positions_b = rig.harness.select([0, 2], queries, rig.norm_ptr)
    # The same rows in the opposite order swap results, and each result is the
    # independent owner's result for that slot.
    for row, slot in enumerate([2, 0]):
        expected = rig.reference.select(slot, queries[row], rig.norm_ptr)
        np.testing.assert_array_equal(positions_a[row, : counts_a[row]], expected)
    for row, slot in enumerate([0, 2]):
        expected = rig.reference.select(slot, queries[row], rig.norm_ptr)
        np.testing.assert_array_equal(positions_b[row, : counts_b[row]], expected)
    assert rig.harness.index.active.slot_ids == (0, 2)


def test_batch_index_slot_reuse_has_no_stale_keys(rig) -> None:
    stale = _key_stream(6500, 9)
    rig.harness.push(1, stale)
    rig.reference.push(1, stale)
    queries = _queries(6501, 1)
    before, _ = rig.harness.select([1], queries, rig.norm_ptr)
    assert before.size == 1

    rig.harness.index.reset_slot(1)
    assert int(rig.harness.index.cursor_host[1]) == 0
    assert int(rig.harness.index.pooled_count_host[1]) == 0

    fresh = _key_stream(6599, 8)
    rig.harness.push(1, fresh)
    counts, positions = rig.harness.select([1], queries, rig.norm_ptr)
    # A fresh independent owner holding only the new history agrees exactly.
    fresh_reference = Reference(rig.runtime)
    try:
        fresh_reference.push(1, fresh)
        expected = fresh_reference.select(1, queries[0], rig.norm_ptr)
        np.testing.assert_array_equal(positions[0, : counts[0]], expected)
        _, oracle_positions = rig.oracle(fresh, queries[0])
        np.testing.assert_array_equal(np.sort(expected), oracle_positions)
    finally:
        fresh_reference.close()


def test_batch_index_rollback_is_exact(rig) -> None:
    real = _key_stream(6600, 9)
    # Slot 0 and slot 1 get the same history, so slot 1 is the never-speculated
    # twin that slot 0 must match after a rollback.
    rig.harness.push(0, real[:7])
    rig.harness.push(1, real[:7])
    # Both rows must receive the same query for the twin comparison to mean
    # anything: the query buffer is ``[rows, heads, dim]``, so a single-row
    # buffer read by two rows would compare unrelated selections.
    queries = np.repeat(_queries(6601, 1), 2, axis=0)
    token = rig.harness.index.begin_rollback()
    assert token.cursors == (7, 7, 0, 0)
    # Nothing has been selected yet, so no block has been pooled.
    assert token.pooled_counts == (0, 0, 0, 0)

    # Speculate the key that completes block 3 on slot 0, then select with it.
    speculation = _key_stream(6699, 1)
    rig.harness.push(0, speculation)
    assert int(rig.harness.index.cursor_host[0]) == 8
    speculative_counts, _ = rig.harness.select([0], queries[:1], rig.norm_ptr)
    assert int(speculative_counts[0]) == BUDGET * RATIO
    speculative_block = rig.harness.pooled_slab()[3].copy()

    rig.harness.index.rollback(token)
    assert rig.harness.index.cursor_host.tolist() == [7, 7, 0, 0]
    assert rig.harness.index.pooled_count_host.tolist() == [0, 0, 0, 0]
    assert rig.harness.index.active is None
    assert rig.harness.index.pending_rollback is None

    # The real continuation on slot 0 must match the untouched twin on slot 1.
    rig.harness.push(0, real[7:])
    rig.harness.push(1, real[7:])
    counts, positions = rig.harness.select([0, 1], queries, rig.norm_ptr)
    np.testing.assert_array_equal(positions[0, : counts[0]], positions[1, : counts[1]])

    # The speculation really reached the pooled slab, and the rewind plus the
    # real continuation replaced it: block 3 now holds the real key's pooled
    # value, so the rollback is exact rather than merely unused.
    real_block = rig.harness.pooled_slab()[3]
    assert not np.array_equal(speculative_block, real_block)

    rig.reference.push(0, real)
    expected = rig.reference.select(0, queries[0], rig.norm_ptr)
    np.testing.assert_array_equal(positions[0, : counts[0]], expected)
    np.testing.assert_allclose(
        rig.harness.pooled_slab()[:4], rig.reference.pooled(0, 4), rtol=1e-6, atol=1e-7
    )


def test_batch_index_rollback_rejects_a_consumed_token_after_overwrite(rig) -> None:
    keys = _key_stream(6700, 8)
    rig.harness.push(0, keys)
    queries = _queries(6701, 1)
    token = rig.harness.index.begin_rollback()

    # Rewind, overwrite the rewound position with a different key, then try to
    # re-advance with the consumed token.
    speculation = _key_stream(6799, 1)
    rig.harness.push(0, speculation)
    rig.harness.index.rollback(token)
    assert int(rig.harness.index.cursor_host[0]) == 8
    rig.harness.push(0, keys[:1])
    assert int(rig.harness.index.cursor_host[0]) == 9

    raw_before = rig.harness.raw_slab().copy()
    cursors_before = rig.harness.index.cursor_host.copy()
    with pytest.raises(RuntimeError, match="already used"):
        rig.harness.index.rollback(token)
    np.testing.assert_array_equal(rig.harness.index.cursor_host, cursors_before)
    np.testing.assert_array_equal(
        rig.harness.raw_slab().view(np.uint32), raw_before.view(np.uint32)
    )

    # The refused rollback left a state that still selects like the reference.
    counts, positions = rig.harness.select([0], queries, rig.norm_ptr)
    rig.reference.push(0, keys)
    rig.reference.push(0, keys[:1])
    expected = rig.reference.select(0, queries[0], rig.norm_ptr)
    np.testing.assert_array_equal(positions[0, : counts[0]], expected)


def test_batch_index_rollback_rejects_a_reset_and_reused_slot(rig) -> None:
    old = _key_stream(6800, 8)
    rig.harness.push(1, old)
    rig.reference.push(1, old)
    queries = _queries(6801, 1)
    token = rig.harness.index.begin_rollback()
    assert token.cursors == (0, 8, 0, 0)

    rig.harness.index.reset_slot(1)
    new = _key_stream(6899, 7)
    rig.harness.push(1, new)
    assert int(rig.harness.index.cursor_host[1]) == 7
    cursors_before = rig.harness.index.cursor_host.copy()
    pooled_before = rig.harness.index.pooled_count_host.copy()
    raw_before = rig.harness.raw_slab().copy()

    # Restoring the pre-reset counters would re-expose the previous request's
    # keys to the new one, so the token must be refused with no state change.
    with pytest.raises(ValueError, match=r"slot\(s\) \[1\] were reset"):
        rig.harness.index.rollback(token)
    np.testing.assert_array_equal(rig.harness.index.cursor_host, cursors_before)
    np.testing.assert_array_equal(rig.harness.index.pooled_count_host, pooled_before)
    np.testing.assert_array_equal(
        rig.harness.raw_slab().view(np.uint32), raw_before.view(np.uint32)
    )

    # The reused slot selects exactly like a fresh owner holding only the new
    # history, so the refused rollback changed nothing.
    rig.harness.index.discard_rollback()
    counts, positions = rig.harness.select([1], queries, rig.norm_ptr)
    fresh = Reference(rig.runtime)
    try:
        fresh.push(1, new)
        expected = fresh.select(1, queries[0], rig.norm_ptr)
        np.testing.assert_array_equal(positions[0, : counts[0]], expected)
    finally:
        fresh.close()
    _, oracle_positions = rig.oracle(new, queries[0])
    np.testing.assert_array_equal(np.sort(expected), oracle_positions)


def test_batch_index_rollback_rejects_another_owners_token(rig) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    keys = _key_stream(6900, 8)
    rig.harness.push(0, keys)
    token = rig.harness.index.begin_rollback()

    other = Harness(runtime)
    try:
        assert other.index.identity != rig.harness.index.identity
        other.push(2, _key_stream(6901, 8))
        cursors_before = other.index.cursor_host.copy()
        with pytest.raises(ValueError, match="different index state"):
            other.index.rollback(token)
        np.testing.assert_array_equal(other.index.cursor_host, cursors_before)
    finally:
        other.close()
    # Still valid on the owner that issued it.
    rig.harness.index.rollback(token)
    assert rig.harness.index.cursor_host.tolist() == [8, 0, 0, 0]
