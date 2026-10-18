"""GPU: N-row dense QSA mixer vs independent c1 mixer and the CPU oracle.

Each row is an independent request: its own slot, page table, and position.
Tokens ``0 .. position - 1`` are bulk-filled into that row's own physical blocks
from CPU-computed K/V, then the mixer under test appends token ``position``
itself and decodes over ``position + 1`` tokens. The c1 reference does the same
thing through ``run_qwen4_exp_dense_qsa_token_mixer`` on the production
single-request state, so both sides own the same control flow and differ only in
the row count of each dispatch.

Exact control is asserted: the prepared positions and page tables are unchanged,
each row's appended K/V lands in its own slot, slots outside the active set stay
bit-zero, and a repeat run reproduces the output bytes. Arithmetic is measured
rather than assumed bit-identical, because a rows>1 GGUF projection and a
single-row one are different GEMM variants.
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
    qsa_interleaved_rope,
    qsa_sparse_gqa_attention,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.qwen4_exp_batch_qsa import (
    run_qwen4_exp_batch_dense_qsa_token_mixer,
)
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpDenseAttentionState,
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAScratch,
    run_qwen4_exp_dense_qsa_token_mixer,
)
from tests.test_gpu_qwen4_exp_runner_gr import _dense_f32_weight

SLOTS = 4
MAX_POSITIONS = 1024
BLOCK_SIZE = 256
HIDDEN = 8
QUERY_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 4
ROTARY_DIM = 4
THETA = 100.0
Q_WIDTH = QUERY_HEADS * HEAD_DIM
KV_WIDTH = KV_HEADS * HEAD_DIM
# Rows in one launch are a different GEMM variant than a single row, so the
# c1-vs-batch comparison is bounded and printed, not asserted bit-identical.
C1_ATOL = 2e-5
ORACLE_ATOL = 2e-4
ORACLE_RTOL = 2e-4


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


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, host.nbytes, runtime=runtime)
    return host


def _rmsnorm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-6) * weight


class Weights:
    """Small valid dense-F32 GGUF weights plus the CPU copies."""

    def __init__(self, rng: np.random.Generator, runtime) -> None:
        self.arrays = {
            "attn_q": rng.normal(0.0, 0.1, size=(Q_WIDTH * 2, HIDDEN)).astype(np.float32),
            "attn_k": rng.normal(0.0, 0.1, size=(KV_WIDTH, HIDDEN)).astype(np.float32),
            "attn_v": rng.normal(0.0, 0.1, size=(KV_WIDTH, HIDDEN)).astype(np.float32),
            "attn_output": rng.normal(0.0, 0.1, size=(HIDDEN, Q_WIDTH)).astype(np.float32),
        }
        self.q_norm = rng.normal(1.0, 0.05, size=HEAD_DIM).astype(np.float32)
        self.k_norm = rng.normal(1.0, 0.05, size=HEAD_DIM).astype(np.float32)
        self._allocations: list = []
        self._runtime = runtime
        self.device = Qwen4ExpQSAMixerDeviceWeights(
            projections={
                name: _dense_f32_weight(name, array, runtime, self._allocations)
                for name, array in self.arrays.items()
            },
            q_norm_weight_ptr=_upload(self.q_norm, runtime, self._allocations).ptr,
            k_norm_weight_ptr=_upload(self.k_norm, runtime, self._allocations).ptr,
        )

    def cpu_key_value(self, mixed: np.ndarray, position: int) -> tuple[np.ndarray, np.ndarray]:
        key = _rmsnorm((self.arrays["attn_k"] @ mixed).reshape(KV_HEADS, HEAD_DIM), self.k_norm)
        key = qsa_interleaved_rope(
            key[None], positions=[position], rotary_dim=ROTARY_DIM, theta=THETA
        )[0]
        value = (self.arrays["attn_v"] @ mixed).reshape(KV_HEADS, HEAD_DIM)
        return key.astype(np.float32), value.astype(np.float32)

    def cpu_expected(self, mixed: np.ndarray, position: int, keys, values) -> np.ndarray:
        qfull = (self.arrays["attn_q"] @ mixed).reshape(QUERY_HEADS, 2, HEAD_DIM)
        query = _rmsnorm(qfull[:, 0], self.q_norm)
        query = qsa_interleaved_rope(
            query[None], positions=[position], rotary_dim=ROTARY_DIM, theta=THETA
        )[0]
        # ``keys``/``values`` already cover tokens 0..position, including the
        # current token that the mixer appends itself.
        context = qsa_sparse_gqa_attention(
            query[None],
            np.asarray(keys, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            query_positions=[position],
            key_positions=np.arange(position + 1),
            selected_positions=(np.arange(position + 1),),
        )[0]
        gated = context * (1.0 / (1.0 + np.exp(-qfull[:, 1])))
        return (self.arrays["attn_output"] @ gated.reshape(-1)).astype(np.float32)

    def close(self) -> None:
        for allocation in reversed(self._allocations):
            free(allocation, runtime=self._runtime)


def _paged_host_cache(total_blocks: int, placements) -> np.ndarray:
    cache = np.zeros((total_blocks * BLOCK_SIZE, KV_HEADS, HEAD_DIM), dtype=np.uint16)
    for physical_ids, tokens in placements:
        for logical, physical in enumerate(physical_ids):
            start = logical * BLOCK_SIZE
            stop = min(start + BLOCK_SIZE, tokens.shape[0])
            if stop <= start:
                break
            base = int(physical) * BLOCK_SIZE
            cache[base : base + (stop - start)] = tokens[start:stop]
    return cache


def _fill_prefix(state, *, blocks_of, total_blocks, row_slots, positions, keys, values):
    """Bulk-fill tokens ``0 .. position - 1`` for each active row."""

    key_host = _paged_host_cache(
        total_blocks,
        [
            (blocks_of(slot), float_array_to_bf16_bits(keys[row][: positions[row]]))
            for row, slot in enumerate(row_slots)
        ],
    )
    value_host = _paged_host_cache(
        total_blocks,
        [
            (blocks_of(slot), float_array_to_bf16_bits(values[row][: positions[row]]))
            for row, slot in enumerate(row_slots)
        ],
    )
    copy_host_to_device(state.key_cache, host_array_ptr(key_host), key_host.nbytes)
    copy_host_to_device(state.value_cache, host_array_ptr(value_host), value_host.nbytes)


def _cpu_kv_table(weights: Weights, mixed_rows: np.ndarray, positions: list[int]):
    keys, values = [], []
    for row, position in enumerate(positions):
        row_keys, row_values = [], []
        for token in range(position + 1):
            key, value = weights.cpu_key_value(mixed_rows[row, token], token)
            row_keys.append(key)
            row_values.append(value)
        keys.append(np.stack(row_keys))
        values.append(np.stack(row_values))
    return keys, values


def _run_batch(
    *,
    weights: Weights,
    state: Qwen4ExpBatchAttentionState,
    scratch: Qwen4ExpQSAScratch,
    slots: list[int],
    positions: list[int],
    mixed_rows: np.ndarray,
    keys,
    values,
    runtime,
) -> tuple[np.ndarray, list]:
    state.prepare(slots, positions)
    _fill_prefix(
        state,
        blocks_of=state.blocks_for_slot,
        total_blocks=state.total_capacity_tokens // state.block_size,
        row_slots=slots,
        positions=positions,
        keys=keys,
        values=values,
    )
    allocations: list = []
    try:
        # One hidden row per request: row r's own token at its own position.
        mixed = _upload(
            mixed_rows[np.arange(len(slots)), positions], runtime, allocations
        )
        output = run_qwen4_exp_batch_dense_qsa_token_mixer(
            mixed.ptr,
            weights.device,
            attention_state=state,
            scratch=scratch,
            rows=len(slots),
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            theta=THETA,
            dense_equivalent_limit=MAX_POSITIONS,
            runtime=runtime,
        )
        runtime.device_synchronize()
        return _download(output, (len(slots), HIDDEN), np.float32, runtime), allocations
    except BaseException:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        raise


def _run_c1(
    *,
    weights: Weights,
    position: int,
    mixed: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    runtime,
) -> np.ndarray:
    state = Qwen4ExpDenseAttentionState.allocate(
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    scratch = Qwen4ExpQSAScratch.allocate(
        rows=1,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    allocations: list = []
    try:
        _fill_prefix(
            state,
            blocks_of=lambda _slot: state.block_host,
            total_blocks=len(state.block_host),
            row_slots=[0],
            positions=[position],
            keys=[keys],
            values=[values],
        )
        d_mixed = _upload(mixed[None], runtime, allocations)
        output = run_qwen4_exp_dense_qsa_token_mixer(
            d_mixed.ptr,
            weights.device,
            attention_state=state,
            scratch=scratch,
            position=position,
            rows=1,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            theta=THETA,
            runtime=runtime,
        )
        runtime.device_synchronize()
        return _download(output, (HIDDEN,), np.float32, runtime)
    finally:
        scratch.close()
        state.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def _case(seed: int, positions: list[int]):
    rng = np.random.default_rng(seed)
    max_context = max(positions) + 1
    mixed_rows = rng.normal(0.0, 0.2, size=(len(positions), max_context, HIDDEN)).astype(
        np.float32
    )
    return rng, mixed_rows


@pytest.mark.parametrize(
    ("seed", "slots", "positions"),
    [
        (4401, [1], [256]),
        (4402, [3, 1], [255, 256]),
        (4403, [2, 0, 3, 1], [3, 255, 256, 700]),
    ],
    ids=["c1", "c2", "c4"],
)
def test_batch_mixer_matches_independent_c1_mixer(
    seed: int, slots: list[int], positions: list[int]
) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng, mixed_rows = _case(seed, positions)
    weights = Weights(rng, runtime)
    state = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    scratch = Qwen4ExpQSAScratch.allocate(
        rows=len(slots),
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    allocations: list = []
    try:
        keys, values = _cpu_kv_table(weights, mixed_rows, positions)
        actual, allocations = _run_batch(
            weights=weights,
            state=state,
            scratch=scratch,
            slots=slots,
            positions=positions,
            mixed_rows=mixed_rows,
            keys=keys,
            values=values,
            runtime=runtime,
        )
        # Exact control: prepare published these rows and the mixer kept them.
        assert state.active is not None
        assert state.active_rows == len(slots)
        assert state.active.slot_ids == tuple(slots)
        assert state.position_host[: len(slots)].tolist() == positions
        assert state.block_table_host[: len(slots)].tolist() == [
            state.block_host[slot].tolist() for slot in slots
        ]

        # Each row's appended K/V landed in its own slot at its own position.
        key_view = _download(
            state.key_cache, (state.total_capacity_tokens, KV_HEADS, HEAD_DIM), np.uint16, runtime
        )
        for row, slot in enumerate(slots):
            physical = int(state.blocks_for_slot(slot)[positions[row] // BLOCK_SIZE])
            token = physical * BLOCK_SIZE + positions[row] % BLOCK_SIZE
            expected_bits = float_array_to_bf16_bits(keys[row][positions[row]])
            assert np.array_equal(key_view[token], expected_bits), (
                f"row {row} appended K to the wrong place"
            )

        expected = np.stack(
            [
                _run_c1(
                    weights=weights,
                    position=positions[row],
                    mixed=mixed_rows[row, positions[row]],
                    keys=keys[row],
                    values=values[row],
                    runtime=runtime,
                )
                for row in range(len(slots))
            ]
        )
        c1_delta = float(np.abs(actual - expected).max())
        print(
            f"rows={len(slots)} positions={positions} max|c1-batch|={c1_delta:.3e}"
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=C1_ATOL)

        oracle = np.stack(
            [
                weights.cpu_expected(
                    mixed_rows[row, positions[row]], positions[row], keys[row], values[row]
                )
                for row in range(len(slots))
            ]
        )
        print(
            f"rows={len(slots)} max|oracle-batch|={np.abs(actual - oracle).max():.3e}"
        )
        np.testing.assert_allclose(actual, oracle, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)

        # Deterministic repeat: same prepared rows, same cache, same bytes.
        repeat, repeat_allocations = _run_batch(
            weights=weights,
            state=state,
            scratch=scratch,
            slots=slots,
            positions=positions,
            mixed_rows=mixed_rows,
            keys=keys,
            values=values,
            runtime=runtime,
        )
        allocations.extend(repeat_allocations)
        assert np.array_equal(repeat, actual)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        scratch.close()
        state.close()
        weights.close()


def test_batch_mixer_writes_only_its_own_slots() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    slots = [0, 2]
    positions = [3, 300]
    rng, mixed_rows = _case(4404, positions)
    weights = Weights(rng, runtime)
    state = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    scratch = Qwen4ExpQSAScratch.allocate(
        rows=len(slots),
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    allocations: list = []
    try:
        keys, values = _cpu_kv_table(weights, mixed_rows, positions)
        _run_batch(
            weights=weights,
            state=state,
            scratch=scratch,
            slots=slots,
            positions=positions,
            mixed_rows=mixed_rows,
            keys=keys,
            values=values,
            runtime=runtime,
        )
        for cache in (state.key_cache, state.value_cache):
            view = _download(
                cache,
                (state.total_capacity_tokens, KV_HEADS, HEAD_DIM),
                np.uint16,
                runtime,
            )
            for idle in (1, 3):
                blocks = state.blocks_for_slot(idle)
                idle_bytes = view[blocks[0] * BLOCK_SIZE : (blocks[-1] + 1) * BLOCK_SIZE]
                assert not idle_bytes.any(), f"slot {idle} received another row's KV"
            for row, slot in enumerate(slots):
                blocks = state.blocks_for_slot(slot)
                written = positions[row] + 1
                start = blocks[0] * BLOCK_SIZE
                above = view[start + written : (blocks[-1] + 1) * BLOCK_SIZE]
                assert not above.any(), f"slot {slot} wrote past its position"
                assert view[start : start + written].any()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        scratch.close()
        state.close()
        weights.close()


def test_batch_mixer_rejects_admission_failures_without_writing() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng, mixed_rows = _case(4405, [4, 5])
    weights = Weights(rng, runtime)
    state = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    scratch = Qwen4ExpQSAScratch.allocate(
        rows=2,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    allocations: list = []

    def _call(**overrides):
        kwargs = dict(
            attention_state=state,
            scratch=scratch,
            rows=2,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            theta=THETA,
            dense_equivalent_limit=MAX_POSITIONS,
            runtime=runtime,
        )
        kwargs.update(overrides)
        return run_qwen4_exp_batch_dense_qsa_token_mixer(
            mixed.ptr, weights.device, **kwargs
        )

    try:
        mixed = _upload(mixed_rows[:, [4, 5]], runtime, allocations)
        state.prepare([0, 1], [4, 5])
        with pytest.raises(ValueError, match="does not match"):
            _call(rows=3)
        # Contexts are 5 and 6; a dense-equivalent bound of 5 rejects row 1 before
        # any dispatch, so the KV cache is still untouched by either call.
        with pytest.raises(ValueError, match="row 1 context 6 exceeds"):
            _call(dense_equivalent_limit=5)
        runtime.device_synchronize()
        # Nothing was written by the rejected calls.
        view = _download(
            state.key_cache,
            (state.total_capacity_tokens, KV_HEADS, HEAD_DIM),
            np.uint16,
            runtime,
        )
        assert not view.any()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        scratch.close()
        state.close()
        weights.close()
