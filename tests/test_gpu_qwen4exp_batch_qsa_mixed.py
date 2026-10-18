"""GPU: mixed dense/sparse N-row QSA mixer vs c1 owners and the CPU oracle.

Each row is an independent request with its own slot, page table, position, and
index history. A run starts every row from a distinct prefix, then advances the
whole batch for several continuation steps, so at least one row crosses the
dense-equivalence bound *during* the run: the keys it captured while it was
still dense are what its first sparse step selects from.

Contracts, asserted separately per ``docs/EXECUTION-PROFILES.md``:

* **Control (exact).** Row-to-slot mapping, the appended K/V landing in each
  row's own slot, untouched slots staying bit-zero, the dense/sparse split, and
  the selection bytes. Dense rows must carry every visible position; sparse rows
  must match an independent c1 index owner that was fed the *same* key and query
  bytes, and the CPU selection oracle, exactly.
* **Arithmetic.** Measured, not assumed: dense rows in a mixed batch run the
  online-softmax rows kernel instead of the dense decode kernel, and the index
  projections are N-row GEMM variants of the c1 single-row ones. Both are bounded
  and printed against the c1 mixer and the CPU layer oracle.
* **Determinism and recovery.** A repeat from the same state reproduces the
  output bytes, a row permutation reproduces each slot's own result, and a
  launch failure after the index append is recovered by ``rollback`` plus a
  re-run of the same step.
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
    qsa_interleaved_rope,
    qsa_prepare_index_keys,
    qsa_select_positions,
    qsa_sparse_gqa_attention,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaSelectionScratch,
    run_qwen4_exp_batch_qsa_mixed_token_mixer,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpDenseAttentionState,
    Qwen4ExpQSAIndexDeviceState,
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAScratch,
    run_qwen4_exp_dense_qsa_token_mixer,
)
from tests.test_gpu_qwen4exp_batch_qsa_mixer import _fill_prefix
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
INDEX_HEADS = 2
INDEX_DIM = 4
INDEX_ROTARY_DIM = 4
RATIO = 2
BUDGET = 2
# ``blocks = (cursor + 1) // ratio`` and dense while ``blocks <= budget``, so a
# row is dense exactly while its context is within ``OUTPUT_STRIDE``.
OUTPUT_STRIDE = BUDGET * RATIO + RATIO - 1
DENSE_LIMIT = OUTPUT_STRIDE
UNTOUCHED_SLOT = 3
# The rows are a different GEMM variant than one row, and a mixed batch attends a
# dense row with the rows kernel instead of the dense decode kernel, so both
# comparisons are bounded and printed rather than asserted bit-identical.
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


def _rmsnorm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-6) * weight


class Weights:
    """Small valid dense-F32 GGUF weights plus their CPU copies."""

    def __init__(self, rng: np.random.Generator, runtime) -> None:
        self.arrays = {
            "attn_q": rng.normal(0.0, 0.1, size=(Q_WIDTH * 2, HIDDEN)).astype(np.float32),
            "attn_k": rng.normal(0.0, 0.1, size=(KV_WIDTH, HIDDEN)).astype(np.float32),
            "attn_v": rng.normal(0.0, 0.1, size=(KV_WIDTH, HIDDEN)).astype(np.float32),
            "attn_output": rng.normal(0.0, 0.1, size=(HIDDEN, Q_WIDTH)).astype(np.float32),
            "index_q": rng.normal(
                0.0, 0.1, size=(INDEX_HEADS * INDEX_DIM, HIDDEN)
            ).astype(np.float32),
            "index_k": rng.normal(0.0, 0.1, size=(INDEX_DIM, HIDDEN)).astype(np.float32),
        }
        self.q_norm = rng.normal(1.0, 0.05, size=HEAD_DIM).astype(np.float32)
        self.k_norm = rng.normal(1.0, 0.05, size=HEAD_DIM).astype(np.float32)
        self.index_q_norm = rng.normal(1.0, 0.05, size=INDEX_DIM).astype(np.float32)
        self.index_k_norm = rng.normal(1.0, 0.05, size=INDEX_DIM).astype(np.float32)
        self._allocations: list = []
        self._runtime = runtime
        self.device = Qwen4ExpQSAMixerDeviceWeights(
            projections={
                name: _dense_f32_weight(name, array, runtime, self._allocations)
                for name, array in self.arrays.items()
            },
            q_norm_weight_ptr=_upload(self.q_norm, runtime, self._allocations).ptr,
            k_norm_weight_ptr=_upload(self.k_norm, runtime, self._allocations).ptr,
            index_q_norm_weight_ptr=_upload(
                self.index_q_norm, runtime, self._allocations
            ).ptr,
            index_k_norm_weight_ptr=_upload(
                self.index_k_norm, runtime, self._allocations
            ).ptr,
        )

    def cpu_key_value(self, mixed: np.ndarray, position: int) -> tuple[np.ndarray, np.ndarray]:
        key = _rmsnorm((self.arrays["attn_k"] @ mixed).reshape(KV_HEADS, HEAD_DIM), self.k_norm)
        key = qsa_interleaved_rope(
            key[None], positions=[position], rotary_dim=ROTARY_DIM, theta=THETA
        )[0]
        value = (self.arrays["attn_v"] @ mixed).reshape(KV_HEADS, HEAD_DIM)
        return key.astype(np.float32), value.astype(np.float32)

    def cpu_index_key(self, mixed: np.ndarray) -> np.ndarray:
        return (self.arrays["index_k"] @ mixed).astype(np.float32)

    def cpu_layer(
        self,
        mixed: np.ndarray,
        position: int,
        keys: np.ndarray,
        values: np.ndarray,
        selected: np.ndarray,
    ) -> np.ndarray:
        qfull = (self.arrays["attn_q"] @ mixed).reshape(QUERY_HEADS, 2, HEAD_DIM)
        query = _rmsnorm(qfull[:, 0], self.q_norm)
        query = qsa_interleaved_rope(
            query[None], positions=[position], rotary_dim=ROTARY_DIM, theta=THETA
        )[0]
        context = qsa_sparse_gqa_attention(
            query[None],
            np.asarray(keys, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            query_positions=[position],
            key_positions=np.arange(position + 1),
            selected_positions=(np.asarray(selected, dtype=np.int64),),
        )[0]
        gated = context * (1.0 / (1.0 + np.exp(-qfull[:, 1])))
        return (self.arrays["attn_output"] @ gated.reshape(-1)).astype(np.float32)

    def close(self) -> None:
        for allocation in reversed(self._allocations):
            free(allocation, runtime=self._runtime)


def _cpu_kv_table(weights: Weights, mixed_rows: np.ndarray):
    """K/V for every token of every row's history, not just the first step."""

    keys, values = [], []
    for row in range(mixed_rows.shape[0]):
        row_keys, row_values = [], []
        for token in range(mixed_rows.shape[1]):
            key, value = weights.cpu_key_value(mixed_rows[row, token], token)
            row_keys.append(key)
            row_values.append(value)
        keys.append(np.stack(row_keys))
        values.append(np.stack(row_values))
    return keys, values


def _cpu_index_table(weights: Weights, mixed_rows: np.ndarray):
    return [
        np.stack(
            [
                weights.cpu_index_key(mixed_rows[row, token])
                for token in range(mixed_rows.shape[1])
            ]
        )
        for row in range(mixed_rows.shape[0])
    ]


class BatchRig:
    """The mixed mixer over one batch owner triple."""

    def __init__(self, runtime, weights: Weights) -> None:
        self.runtime = runtime
        self.weights = weights
        self.allocations: list = []
        self.attention = Qwen4ExpBatchAttentionState.allocate(
            state_slots=SLOTS,
            max_positions=MAX_POSITIONS,
            block_size=BLOCK_SIZE,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
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
        self.selection = Qwen4ExpBatchQsaSelectionScratch.allocate(
            rows=SLOTS,
            stride=OUTPUT_STRIDE,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=runtime,
        )
        self.scratch = Qwen4ExpQSAScratch.allocate(
            rows=SLOTS,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=runtime,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
        )
        self.positions: list[int] = []

    def seed(
        self,
        slots: list[int],
        positions: list[int],
        keys,
        values,
        index_keys,
    ) -> None:
        """Install tokens ``0 .. position - 1`` for every row, one token at a time."""

        _fill_prefix(
            self.attention,
            blocks_of=self.attention.blocks_for_slot,
            total_blocks=self.attention.total_capacity_tokens // BLOCK_SIZE,
            row_slots=slots,
            positions=positions,
            keys=keys,
            values=values,
        )
        stride = INDEX_DIM * 4
        for row, slot in enumerate(slots):
            device = _upload(index_keys[row], self.runtime, self.allocations)
            for token in range(positions[row]):
                self.attention.prepare([slot], [token])
                self.index.prepare([slot])
                self.index.append_active(device.ptr + token * stride)
        self.positions = list(positions)

    def step(self, slots: list[int], mixed_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance every row by one token and return ``(output, index_keys)``."""

        self.attention.prepare(slots, self.positions)
        self.index.prepare(slots)
        mixed = _upload(
            mixed_rows[np.arange(len(slots)), self.positions], self.runtime, self.allocations
        )
        output = run_qwen4_exp_batch_qsa_mixed_token_mixer(
            mixed.ptr,
            self.weights.device,
            attention_state=self.attention,
            index_state=self.index,
            selection=self.selection,
            scratch=self.scratch,
            rows=len(slots),
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            theta=THETA,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            index_rotary_dim=INDEX_ROTARY_DIM,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        rows = len(slots)
        result = _download(output, (rows, HIDDEN), np.float32, self.runtime)
        index_keys = _download(
            self.scratch.index_k_projected, (rows, INDEX_DIM), np.float32, self.runtime
        )
        self.positions = [position + 1 for position in self.positions]
        return result, index_keys

    def selection_rows(self, rows: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.selection.counts_host[:rows].copy(),
            self.selection.positions_host[:rows].copy(),
        )

    def index_query_rows(self, rows: int) -> np.ndarray:
        return _download(
            self.scratch.index_query,
            (rows, INDEX_HEADS, INDEX_DIM),
            np.float32,
            self.runtime,
        )

    def close(self) -> None:
        self.scratch.close()
        self.selection.close()
        self.index.close()
        self.attention.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


class RowReference:
    """One row's independent c1 owners: mixer output and selection oracle."""

    def __init__(self, runtime, weights: Weights) -> None:
        self.runtime = runtime
        self.weights = weights
        self.allocations: list = []
        self.attention = Qwen4ExpDenseAttentionState.allocate(
            max_positions=MAX_POSITIONS,
            block_size=BLOCK_SIZE,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=runtime,
        )
        self.scratch = Qwen4ExpQSAScratch.allocate(
            rows=1,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=runtime,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
        )
        self.mixer_index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=self.attention,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=runtime,
        )
        # Fed the batch owner's own key and query bytes, so a selection mismatch
        # is a selection bug and never a projection rounding difference.
        self.selection_index = Qwen4ExpQSAIndexDeviceState.allocate(
            attention_state=self.attention,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=runtime,
        )

    def seed(self, position: int, keys, values, index_keys) -> None:
        _fill_prefix(
            self.attention,
            blocks_of=lambda _slot: self.attention.block_host,
            total_blocks=len(self.attention.block_host),
            row_slots=[0],
            positions=[position],
            keys=[keys],
            values=[values],
        )
        stride = INDEX_DIM * 4
        device = _upload(index_keys, self.runtime, self.allocations)
        for token in range(position):
            self.mixer_index.append(device.ptr + token * stride, position=token)
            self.selection_index.append(device.ptr + token * stride, position=token)

    def push_batch_key(self, index_key: np.ndarray) -> None:
        stride = INDEX_DIM * 4
        device = _upload(index_key[None], self.runtime, self.allocations)
        self.selection_index.append(
            device.ptr, position=int(self.selection_index.count)
        )

    def select(self, query: np.ndarray) -> np.ndarray:
        """The c1 owner's selection in kernel order, matching the batch layout."""

        device = _upload(query[None], self.runtime, self.allocations)
        count = self.selection_index.select_positions_device(
            device.ptr,
            query_position=int(self.selection_index.count) - 1,
            key_norm_weight_ptr=self.weights.device.index_k_norm_weight_ptr,
            rotary_dim=INDEX_ROTARY_DIM,
            theta=THETA,
        )
        return _download(
            self.selection_index.selected_positions, (count,), np.int64, self.runtime
        )

    def mixer_step(self, mixed: np.ndarray, position: int) -> np.ndarray:
        device = _upload(mixed[None], self.runtime, self.allocations)
        output = run_qwen4_exp_dense_qsa_token_mixer(
            device.ptr,
            self.weights.device,
            attention_state=self.attention,
            scratch=self.scratch,
            position=position,
            rows=1,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            theta=THETA,
            index_state=self.mixer_index,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            index_rotary_dim=INDEX_ROTARY_DIM,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        return _download(output, (HIDDEN,), np.float32, self.runtime)

    def mixer_selection(self, position: int) -> np.ndarray | None:
        if self.mixer_index.count <= DENSE_LIMIT:
            return None
        count = BUDGET * RATIO
        if position % RATIO != RATIO - 1:
            count += position % RATIO + 1
        return _download(
            self.mixer_index.selected_positions, (count,), np.int64, self.runtime
        )

    def close(self) -> None:
        self.selection_index.close()
        self.mixer_index.close()
        self.scratch.close()
        self.attention.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


# Row 0 starts dense and crosses the bound on its third step, row 1 is sparse
# from the first step, row 2 stays dense for the whole run.
CASES = {
    "mixed": ([0, 1, 2], [3, 8, 2]),
    "permuted": ([2, 0, 1], [2, 3, 8]),
}
STEPS = 3


def _case(seed: int, slots: list[int], positions: list[int]):
    """A per-slot history plus the row view for this row order.

    The history is drawn per slot, so a permuted run replays the *same* request
    in a different row and a mismatch can only be a row-addressing bug.
    """

    rng = np.random.default_rng(seed)
    max_context = max(positions) + STEPS + 1
    histories = {
        slot: rng.normal(0.0, 0.2, size=(max_context, HIDDEN)).astype(np.float32)
        for slot in range(SLOTS)
    }
    return rng, np.stack([histories[slot] for slot in slots])


def _run_steps(
    runtime,
    weights: Weights,
    slots: list[int],
    positions: list[int],
    mixed_rows: np.ndarray,
    *,
    print_label: str,
) -> dict:
    """Advance every row ``STEPS`` times, verifying each step as it runs."""

    keys, values = _cpu_kv_table(weights, mixed_rows)
    index_keys = _cpu_index_table(weights, mixed_rows)
    rig = BatchRig(runtime, weights)
    references = [RowReference(runtime, weights) for _ in slots]
    record = {"outputs": [], "selections": [], "modes": [], "c1": [], "oracle": []}
    try:
        rig.seed(slots, positions, keys, values, index_keys)
        for reference, row in zip(references, range(len(slots))):
            reference.seed(positions[row], keys[row], values[row], index_keys[row])
        for step in range(STEPS):
            outputs, step_index_keys = rig.step(slots, mixed_rows)
            counts, selected = rig.selection_rows(len(slots))
            queries = rig.index_query_rows(len(slots))
            step_outputs, step_selections, step_c1, step_oracle = [], [], [], []
            step_modes = []
            for row in range(len(slots)):
                position = positions[row] + step
                context = position + 1
                reference = references[row]
                # The c1 mixer appends its own key; the selection oracle is fed
                # the batch's key bytes so the comparison is exact. Its key is
                # pushed on dense steps too, so its cursor tracks the batch's.
                c1_output = reference.mixer_step(mixed_rows[row, position], position)
                reference.push_batch_key(step_index_keys[row])
                batch_selection = selected[row, : counts[row]]
                if context <= DENSE_LIMIT:
                    assert counts[row] == context, (
                        f"row {row} step {step} dense selection count {counts[row]} "
                        f"is not its context {context}"
                    )
                    assert np.array_equal(batch_selection, np.arange(context)), (
                        f"row {row} step {step} dense selection is not every visible "
                        "position"
                    )
                    assert reference.mixer_selection(position) is None
                else:
                    c1_selection = reference.select(queries[row])
                    assert np.array_equal(batch_selection, c1_selection), (
                        f"row {row} step {step} selection differs from the c1 owner: "
                        f"{batch_selection.tolist()} vs {c1_selection.tolist()}"
                    )
                    c1_mixer_selection = reference.mixer_selection(position)
                    assert c1_mixer_selection is not None
                    assert np.array_equal(batch_selection, c1_mixer_selection), (
                        f"row {row} step {step} c1 mixer selected "
                        f"{c1_mixer_selection.tolist()} but the batch selected "
                        f"{batch_selection.tolist()}"
                    )
                    oracle_selection = _cpu_selection(
                        weights,
                        np.concatenate(
                            [index_keys[row][:position], step_index_keys[row][None, :]]
                        ),
                        queries[row],
                        position,
                    )
                    # The CPU oracle emits the selected set in position order; the
                    # kernels emit it in block order, so compare as sets.
                    assert np.array_equal(np.sort(batch_selection), oracle_selection), (
                        f"row {row} step {step} selection differs from the CPU oracle: "
                        f"{np.sort(batch_selection).tolist()} vs {oracle_selection.tolist()}"
                    )
                token_key, token_value = weights.cpu_key_value(
                    mixed_rows[row, position], position
                )
                row_keys = np.concatenate([keys[row][:position], token_key[None]])
                row_values = np.concatenate([values[row][:position], token_value[None]])
                oracle = weights.cpu_layer(
                    mixed_rows[row, position],
                    position,
                    row_keys,
                    row_values,
                    batch_selection,
                )
                np.testing.assert_allclose(
                    outputs[row], c1_output, rtol=0.0, atol=C1_ATOL,
                    err_msg=f"row {row} step {step} c1 mismatch",
                )
                np.testing.assert_allclose(
                    outputs[row], oracle, rtol=ORACLE_RTOL, atol=ORACLE_ATOL,
                    err_msg=f"row {row} step {step} oracle mismatch",
                )
                step_modes.append("dense" if context <= DENSE_LIMIT else "sparse")
                step_outputs.append(outputs[row])
                step_selections.append(batch_selection.copy())
                step_c1.append(c1_output)
                step_oracle.append(oracle)
                print(
                    f"{print_label} row={row} slot={slots[row]} position={position} "
                    f"context={context} selected={counts[row]} "
                    f"max|c1-batch|={np.abs(outputs[row] - c1_output).max():.3e} "
                    f"max|oracle-batch|={np.abs(outputs[row] - oracle).max():.3e}"
                )
            record["outputs"].append(np.stack(step_outputs))
            record["modes"].append(step_modes)
            record["selections"].append(step_selections)
            record["c1"].append(np.stack(step_c1))
            record["oracle"].append(np.stack(step_oracle))
            # Exact control: prepare published these rows and the mixer kept them.
            assert rig.attention.active.slot_ids == tuple(slots)
            assert rig.attention.position_host[: len(slots)].tolist() == [
                positions[row] + step for row in range(len(slots))
            ]
            assert rig.index.active.slot_ids == tuple(slots)
            # Cursors live on slots, not rows, so read them through the row map.
            assert rig.index.cursor_host[slots].tolist() == [
                positions[row] + step + 1 for row in range(len(slots))
            ]
        record["rig"] = rig
        record["references"] = references
        return record
    except BaseException:
        for reference in references:
            reference.close()
        rig.close()
        raise


def _cpu_selection(
    weights: Weights, raw_keys: np.ndarray, query: np.ndarray, position: int
) -> np.ndarray:
    pooled = qsa_prepare_index_keys(
        raw_keys,
        np.arange(raw_keys.shape[0], dtype=np.int64),
        weights.index_k_norm,
        compression_ratio=RATIO,
        rotary_dim=INDEX_ROTARY_DIM,
        theta=THETA,
    )
    scores = qsa_index_scores(query[None], pooled.keys)
    selection = qsa_select_positions(
        scores,
        pooled.block_starts,
        query_positions=[position],
        available_positions=np.arange(raw_keys.shape[0], dtype=np.int64),
        compression_ratio=RATIO,
        block_budget=BUDGET,
    )
    return selection.selected_positions[0]


def _close_record(record: dict) -> None:
    for reference in record["references"]:
        reference.close()
    record["rig"].close()


def test_mixed_rows_across_the_dense_threshold() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    slots, positions = CASES["mixed"]
    rng, mixed_rows = _case(5511, slots, positions)
    weights = Weights(rng, runtime)
    record = None
    try:
        record = _run_steps(
            runtime, weights, slots, positions, mixed_rows, print_label="mixed"
        )
        # The run really crossed the bound: row 0 is dense at step 0 and sparse
        # at its last step, so its sparse selection came from keys captured while
        # it was still dense. Row 1 is sparse throughout and row 2 stays dense.
        rig = record["rig"]
        # Indexed ``[step][row]``: row 0 crosses the bound on the last step,
        # row 1 is sparse throughout, row 2 stays dense.
        assert record["modes"] == [
            ["dense", "sparse", "dense"],
            ["dense", "sparse", "dense"],
            ["sparse", "sparse", "dense"],
        ]
        # Row 0's first step is dense: every visible position.
        assert record["selections"][0][0].tolist() == list(range(positions[0] + 1))
        # Its last step is sparse and ends on a block edge, so it selects exactly
        # the budget's blocks and is not the dense set.
        assert len(record["selections"][2][0]) == BUDGET * RATIO
        assert not np.array_equal(
            record["selections"][2][0], np.arange(positions[0] + STEPS)
        )
        # Untouched slot stays bit-zero in both the KV cache and the index slab.
        key_view = _download(
            rig.attention.key_cache,
            (rig.attention.total_capacity_tokens, KV_HEADS, HEAD_DIM),
            np.uint16,
            runtime,
        )
        value_view = _download(
            rig.attention.value_cache,
            (rig.attention.total_capacity_tokens, KV_HEADS, HEAD_DIM),
            np.uint16,
            runtime,
        )
        blocks = rig.attention.blocks_for_slot(UNTOUCHED_SLOT)
        for physical in blocks:
            start = int(physical) * BLOCK_SIZE
            assert not key_view[start : start + BLOCK_SIZE].any()
            assert not value_view[start : start + BLOCK_SIZE].any()
        raw = _download(
            rig.index.raw_keys,
            (SLOTS * rig.index.slot_capacity_tokens, INDEX_DIM),
            np.float32,
            runtime,
        )
        slot_start = UNTOUCHED_SLOT * rig.index.slot_capacity_tokens
        assert not raw[slot_start : slot_start + rig.index.slot_capacity_tokens].any()
        # Every active row's index cursor advanced exactly once per step.
        assert rig.index.cursor_host[slots].tolist() == [
            position + STEPS for position in positions
        ]
    finally:
        if record is not None:
            _close_record(record)
        weights.close()


def test_row_permutation_reproduces_each_slots_result() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(5512), runtime)
    straight = permuted = None
    try:
        straight = _run_steps(
            runtime,
            weights,
            *CASES["mixed"],
            _case(5512, *CASES["mixed"])[1],
            print_label="straight",
        )
        permuted = _run_steps(
            runtime,
            weights,
            *CASES["permuted"],
            _case(5512, *CASES["permuted"])[1],
            print_label="permuted",
        )
        # The permutation moves whole requests, so each slot's own history must
        # reproduce byte for byte, including its selection.
        for step in range(STEPS):
            for row, slot in enumerate(CASES["mixed"][0]):
                other = CASES["permuted"][0].index(slot)
                assert np.array_equal(
                    straight["selections"][step][row],
                    permuted["selections"][step][other],
                ), f"step {step} slot {slot} selection changed with the row order"
                assert np.array_equal(
                    straight["outputs"][step][row], permuted["outputs"][step][other]
                ), f"step {step} slot {slot} output changed with the row order"
    finally:
        for record in (straight, permuted):
            if record is not None:
                _close_record(record)
        weights.close()


def test_repeat_run_from_the_same_state_is_bit_identical() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(5513), runtime)
    slots, positions = CASES["mixed"]
    mixed_rows = _case(5513, slots, positions)[1]
    first = second = None
    try:
        first = _run_steps(
            runtime, weights, slots, positions, mixed_rows, print_label="first"
        )
        second = _run_steps(
            runtime, weights, slots, positions, mixed_rows, print_label="second"
        )
        for step in range(STEPS):
            assert np.array_equal(first["outputs"][step], second["outputs"][step])
            for row in range(len(slots)):
                assert np.array_equal(
                    first["selections"][step][row], second["selections"][step][row]
                )
    finally:
        for record in (first, second):
            if record is not None:
                _close_record(record)
        weights.close()


def test_invalid_metadata_leaves_the_kv_cache_and_index_untouched() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(5514), runtime)
    slots, positions = [0, 1], [3, 8]
    mixed_rows = _case(5514, slots, positions)[1]
    keys, values = _cpu_kv_table(weights, mixed_rows)
    index_keys = _cpu_index_table(weights, mixed_rows)
    rig = BatchRig(runtime, weights)
    try:
        rig.seed(slots, positions, keys, values, index_keys)
        rig.attention.prepare(slots, positions)
        rig.index.prepare(slots)
        key_before = _download(
            rig.attention.key_cache,
            (rig.attention.total_capacity_tokens, KV_HEADS, HEAD_DIM),
            np.uint16,
            runtime,
        )
        raw_before = _download(
            rig.index.raw_keys,
            (SLOTS * rig.index.slot_capacity_tokens, INDEX_DIM),
            np.float32,
            runtime,
        )
        cursors_before = rig.index.cursor_host.copy()
        mixed = _upload(
            mixed_rows[np.arange(len(slots)), positions], runtime, rig.allocations
        )
        with pytest.raises(ValueError, match="index geometry"):
            run_qwen4_exp_batch_qsa_mixed_token_mixer(
                mixed.ptr,
                weights.device,
                attention_state=rig.attention,
                index_state=rig.index,
                selection=rig.selection,
                scratch=rig.scratch,
                rows=len(slots),
                hidden=HIDDEN,
                query_heads=QUERY_HEADS,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                rotary_dim=ROTARY_DIM,
                theta=THETA,
                index_heads=INDEX_HEADS + 1,
                index_dim=INDEX_DIM,
                index_rotary_dim=INDEX_ROTARY_DIM,
                runtime=runtime,
            )
        runtime.device_synchronize()
        key_after = _download(
            rig.attention.key_cache,
            (rig.attention.total_capacity_tokens, KV_HEADS, HEAD_DIM),
            np.uint16,
            runtime,
        )
        raw_after = _download(
            rig.index.raw_keys,
            (SLOTS * rig.index.slot_capacity_tokens, INDEX_DIM),
            np.float32,
            runtime,
        )
        assert np.array_equal(key_after, key_before)
        assert np.array_equal(raw_after, raw_before)
        assert np.array_equal(rig.index.cursor_host, cursors_before)
        assert rig.selection.counts_host.tolist() == [0] * SLOTS
    finally:
        rig.close()
        weights.close()


def test_failed_step_is_recovered_by_rollback_and_a_rerun(monkeypatch) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(5515), runtime)
    slots, positions = [0, 1], [3, 8]
    mixed_rows = _case(5515, slots, positions)[1]
    keys, values = _cpu_kv_table(weights, mixed_rows)
    index_keys = _cpu_index_table(weights, mixed_rows)
    clean = failing = None
    try:
        clean = _run_steps(
            runtime, weights, slots, positions, mixed_rows, print_label="clean"
        )
        # A twin that fails inside the step, after the index append committed.
        rig = BatchRig(runtime, weights)
        rig.seed(slots, positions, keys, values, index_keys)
        mixed = _upload(
            mixed_rows[np.arange(len(slots)), positions], runtime, rig.allocations
        )
        import hipengine.runtime.qwen4_exp_batch_qsa as batch_qsa

        def explode(*args, **kwargs):
            raise RuntimeError("injected KV write failure")

        rig.attention.prepare(slots, positions)
        rig.index.prepare(slots)
        token = rig.index.begin_rollback()
        monkeypatch.setattr(batch_qsa, "qwen35_write_paged_kv_f32_batch_spans", explode)
        with pytest.raises(RuntimeError, match="injected KV write failure"):
            run_qwen4_exp_batch_qsa_mixed_token_mixer(
                mixed.ptr,
                weights.device,
                attention_state=rig.attention,
                index_state=rig.index,
                selection=rig.selection,
                scratch=rig.scratch,
                rows=len(slots),
                hidden=HIDDEN,
                query_heads=QUERY_HEADS,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                rotary_dim=ROTARY_DIM,
                theta=THETA,
                index_heads=INDEX_HEADS,
                index_dim=INDEX_DIM,
                index_rotary_dim=INDEX_ROTARY_DIM,
                runtime=runtime,
            )
        # The append committed before the failure; the token is what undoes it.
        assert rig.index.cursor_host[slots].tolist() == [
            position + 1 for position in positions
        ]
        rig.index.rollback(token)
        monkeypatch.undo()
        assert rig.index.cursor_host[slots].tolist() == list(positions)
        rig.attention.prepare(slots, positions)
        rig.index.prepare(slots)
        output, _ = rig.step(slots, mixed_rows)
        np.testing.assert_allclose(
            output, clean["outputs"][0], rtol=0.0, atol=0.0,
            err_msg="the recovered step is not identical to an uninterrupted run",
        )
        failing = {"rig": rig, "references": [], "outputs": [output]}
    finally:
        if clean is not None:
            _close_record(clean)
        if failing is not None:
            failing["rig"].close()
        weights.close()
