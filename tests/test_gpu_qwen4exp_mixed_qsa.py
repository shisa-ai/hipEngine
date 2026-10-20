"""GPU: the mixed QSA adapter's depth waves vs a sequential same-request oracle.

``run_qwen4_exp_mixed_qsa_attention`` walks a mixed layout in depth waves: wave
``d`` gathers every request's row at depth ``d``, publishes that wave's own slots
and positions, runs the committed mixed-mixer attention stage once, scatters each
wave's gate output to its physical row, and then runs one output projection over
every physical row of the pass. The reference for that orchestration is the same
committed mixer run the way a single request reaches it: one row per step, one
request at a time, on a separate owner pair seeded with the same committed
prefix. Both sides write the same rows in the same order, so a difference can
only come from the adapter itself — the wave gather, the per-wave publish, the
physical scatter, or the pass-wide projection.

Contracts, asserted separately per ``docs/EXECUTION-PROFILES.md``:

* **Control (exact).** Each physical row carries its own request's result, under
  a row permutation as well as in order; a committed prefix and a slot outside
  the layout keep their index cursors, pooled counters, and KV bytes; abort
  restores every layout slot's cursor and pooled counter to its pre-pass value,
  and the re-run reproduces the pass byte for byte.
* **Arithmetic.** Measured, not assumed: the adapter's gathered shared
  projections and its one full-row output projection are N-row GEMM variants of
  the single-row ones, so the comparison against the sequential reference is
  bounded and printed, and one dense and one sparse row are also compared
  against the CPU layer oracle.
* **Accepted-prefix commit.** A chain rejected down to its root row leaves the
  committed cursor at ``start_position + accepted + 1`` with the pooled-prefix
  counter clamped to that cursor's complete blocks, and the state continues
  correctly: one more row decoded at the committed cursor reproduces the
  sequential reference for that same committed prefix. That step is sparse, so
  it reads exactly the pooled keys the clamp withdrew.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
import pytest

from hipengine.core.memory import free
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaMixerPlan,
    Qwen4ExpBatchQsaSelectionScratch,
    run_qwen4_exp_batch_qsa_input_projections,
    run_qwen4_exp_batch_qsa_mixed_token_mixer,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_mixed_layout import (
    Qwen4ExpMixedLayout,
    Qwen4ExpMixedRequest,
    compile_mixed_layout,
)
from hipengine.runtime.qwen4_exp_mixed_qsa import (
    Qwen4ExpMixedQsaRun,
    Qwen4ExpMixedQsaScratch,
    abort_qwen4_exp_mixed_qsa_attention,
    commit_qwen4_exp_mixed_qsa_attention,
    run_qwen4_exp_mixed_qsa_attention,
)
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpQSAScratch
from tests.test_gpu_qwen4exp_batch_qsa_mixed import (
    BLOCK_SIZE,
    BUDGET,
    C1_ATOL,
    HEAD_DIM,
    HIDDEN,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_ROTARY_DIM,
    KV_HEADS,
    KV_WIDTH,
    MAX_POSITIONS,
    ORACLE_ATOL,
    ORACLE_RTOL,
    OUTPUT_STRIDE,
    Q_WIDTH,
    QUERY_HEADS,
    RATIO,
    ROTARY_DIM,
    SLOTS,
    THETA,
    Weights,
    _cpu_kv_table,
    _cpu_index_table,
    _cpu_selection,
    _download,
    _upload,
)
from tests.test_gpu_qwen4exp_batch_qsa_mixer import _fill_prefix

# The adapter gathers the caller's shared projections and runs one full-row
# output projection, so both sides of the comparison are different GEMM widths.
# The mixed-mixer GPU test bounds the same difference at ``C1_ATOL`` across a
# strictly wider gap (it also swaps the attention kernel), so this is the same
# bound, not a looser one.
ADAPTER_ATOL = C1_ATOL

# The batch owner hands out ``1..N`` from ``admit()`` and never reuses a token,
# so one admission per slot is the whole ownership table these passes see.
OWNER_TOKENS = {slot: slot + 1 for slot in range(SLOTS)}


def _owner_of(slot: int) -> int:
    return OWNER_TOKENS[slot]


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


# -- cases ------------------------------------------------------------------

# ``(slot, start_position, candidate count)`` per request. Rows of a request run
# from ``start_position`` to ``start_position + candidates``.
#
# Chains of three, one, and two rows: mixed lengths plus an AR peer, all inside
# the dense-equivalence bound, so every row attends its whole context.
DENSE_SPEC = ((0, 0, 2), (1, 0, 0), (2, 0, 1))
# A permutation that keeps every chain's rows in parent-first order (the layout
# requires it) while interleaving the three requests, so no wave is contiguous.
DENSE_PERMUTATION = (4, 0, 3, 5, 1, 2)
# Nine rows on one slot cross the bound mid-chain (dense through position
# ``OUTPUT_STRIDE - 1``, sparse after it) while an AR peer stays dense, so one
# pass holds both kinds and a request's sparse rows select from keys its earlier
# dense waves captured.
BOUNDARY_SPEC = ((0, 0, 8), (1, 0, 0))
BOUNDARY_SPARSE_ROW = (0, OUTPUT_STRIDE)
BOUNDARY_DENSE_ROW = (0, 2)
# Committed prefixes of four and six rows, plus a slot outside the layout.
PREFIXED_SPEC = ((0, 4, 2), (1, 6, 0))
UNRELATED_SLOT = SLOTS - 1
UNRELATED_PREFIX = 3
# Both requests commit down to their root row, so the committed cursor is
# position 7 and the follow-up row decoded there is sparse. Slot 0's pass pooled
# a block that covered a rejected row, so the commit withdraws it.
COMMIT_SPEC = ((0, 6, 3), (1, 6, 0))
COMMIT_POSITION = 7


# -- fixtures ---------------------------------------------------------------


def _build(
    seed: int,
    spec: tuple[tuple[int, int, int], ...],
    *,
    physical_row_order=None,
    extra_slots: tuple[int, ...] = (),
    extra_prefix: int = 0,
) -> tuple[Qwen4ExpMixedLayout, dict[int, np.ndarray]]:
    """One layout plus each slot's hidden history, drawn per slot.

    The history is keyed by slot and drawn in ``spec`` order, so the same seed
    replays the same request in a different physical row and a mismatch can only
    be a row-addressing bug.
    """

    rng = np.random.default_rng(seed)
    requests: list[Qwen4ExpMixedRequest] = []
    histories: dict[int, np.ndarray] = {}
    for request_id, (slot, start_position, candidates) in enumerate(spec):
        requests.append(
            Qwen4ExpMixedRequest(
                request_id=request_id,
                slot=slot,
                owner_token=OWNER_TOKENS[slot],
                start_position=start_position,
                root_token=100 + slot,
                candidates=tuple(
                    200 + slot * 10 + depth for depth in range(candidates)
                ),
            )
        )
        histories[slot] = rng.normal(
            0.0, 0.2, size=(start_position + candidates + 1, HIDDEN)
        ).astype(np.float32)
    for slot in extra_slots:
        histories[slot] = rng.normal(
            0.0, 0.2, size=(extra_prefix + 1, HIDDEN)
        ).astype(np.float32)
    return compile_mixed_layout(requests, physical_row_order=physical_row_order), histories


def _history_tables(weights: Weights, histories: dict[int, np.ndarray]) -> dict:
    """CPU K/V and raw index keys for every token of every slot's history."""

    slots = sorted(histories)
    width = max(history.shape[0] for history in histories.values())
    rows = np.zeros((len(slots), width, HIDDEN), dtype=np.float32)
    for index, slot in enumerate(slots):
        rows[index, : histories[slot].shape[0]] = histories[slot]
    keys, values = _cpu_kv_table(weights, rows)
    index_keys = _cpu_index_table(weights, rows)
    return {
        slot: (keys[index], values[index], index_keys[index])
        for index, slot in enumerate(slots)
    }


@dataclass(frozen=True)
class _Seed:
    """Tokens ``0 .. position - 1`` of one slot, as the committed prefix."""

    slot: int
    position: int
    keys: np.ndarray
    values: np.ndarray
    index_keys: np.ndarray


def _seed_prefix(attention, index, seeds, *, runtime, allocations) -> None:
    """Install every seed's committed prefix in one cache fill plus its appends.

    The cache fill rewrites the whole K/V slab, so every seed has to be in the
    same call; the index keys then append one per token through the slot's own
    page table, which is how a committed prefix reaches the index owner.
    """

    seeds = [seed for seed in seeds if seed.position]
    if not seeds:
        return
    _fill_prefix(
        attention,
        blocks_of=attention.blocks_for_slot,
        total_blocks=attention.total_capacity_tokens // BLOCK_SIZE,
        row_slots=[seed.slot for seed in seeds],
        positions=[seed.position for seed in seeds],
        keys=[seed.keys for seed in seeds],
        values=[seed.values for seed in seeds],
    )
    stride = INDEX_DIM * 4
    for seed in seeds:
        device = _upload(seed.index_keys[: seed.position], runtime, allocations)
        for token in range(seed.position):
            attention.prepare([seed.slot], [token])
            index.prepare([seed.slot])
            index.append_active(device.ptr + token * stride)


def _physical_rows(attention, spans: dict[int, range]) -> list[int]:
    """Physical cache rows each slot's logical positions live at."""

    rows: list[int] = []
    for slot, positions in spans.items():
        blocks = attention.block_host[slot]
        for position in positions:
            rows.append(
                int(blocks[position // BLOCK_SIZE]) * BLOCK_SIZE + position % BLOCK_SIZE
            )
    return rows


def _raw_index_keys(attention, index, slot: int, positions, runtime) -> np.ndarray:
    """One slot's raw index keys, read back from the owner that wrote them.

    The CPU selection oracle is fed these bytes rather than CPU-computed keys, so
    a selection mismatch is a selection bug and never a projection rounding
    difference.
    """

    raw = _download(
        index.raw_keys,
        (attention.state_slots * index.slot_capacity_tokens, INDEX_DIM),
        np.float32,
        runtime,
    )
    blocks = attention.block_host[slot]
    rows = [
        int(blocks[position // BLOCK_SIZE]) * BLOCK_SIZE + position % BLOCK_SIZE
        for position in positions
    ]
    return raw[rows]


def _key_snapshot(attention, runtime) -> np.ndarray:
    return _download(
        attention.key_cache,
        (attention.total_capacity_tokens, KV_HEADS, HEAD_DIM),
        np.uint16,
        runtime,
    )


def _decode_row(
    runtime,
    weights: Weights,
    *,
    attention_state,
    index_state,
    selection,
    scratch,
    slot: int,
    position: int,
    hidden_row: np.ndarray,
    allocations: list,
) -> np.ndarray:
    """Decode one row for one slot through the committed mixed mixer."""

    attention_state.prepare([slot], [position])
    index_state.prepare([slot])
    device = _upload(hidden_row[None], runtime, allocations)
    output = run_qwen4_exp_batch_qsa_mixed_token_mixer(
        device.ptr,
        weights.device,
        attention_state=attention_state,
        index_state=index_state,
        selection=selection,
        scratch=scratch,
        rows=1,
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
    runtime.device_synchronize()
    return _download(output, (HIDDEN,), np.float32, runtime)


# -- owners -----------------------------------------------------------------


class Rig:
    """The batch owner pair one mixed pass runs on."""

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

    def close(self) -> None:
        self.index.close()
        self.attention.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


class SequentialOracle:
    """One request at a time, one row per step: the same-request reference."""

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
            rows=1,
            stride=OUTPUT_STRIDE,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=runtime,
        )
        self.scratch = Qwen4ExpQSAScratch.allocate(
            rows=1,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=runtime,
        )

    def step(self, slot: int, position: int, hidden_row: np.ndarray) -> np.ndarray:
        return _decode_row(
            self.runtime,
            self.weights,
            attention_state=self.attention,
            index_state=self.index,
            selection=self.selection,
            scratch=self.scratch,
            slot=slot,
            position=position,
            hidden_row=hidden_row,
            allocations=self.allocations,
        )

    def close(self) -> None:
        self.scratch.close()
        self.selection.close()
        self.index.close()
        self.attention.close()
        for allocation in reversed(self.allocations):
            free(allocation, runtime=self.runtime)


@dataclass
class PassResult:
    """One mixed pass plus the caller-owned scratch it ran through."""

    run: Qwen4ExpMixedQsaRun
    output: np.ndarray
    shared: Qwen4ExpQSAScratch
    scratch: Qwen4ExpMixedQsaScratch

    def close(self) -> None:
        self.scratch.close()
        self.shared.close()


class Staged:
    """A layout, its owners, its committed prefixes, and its reference."""

    def __init__(
        self,
        runtime,
        weights: Weights,
        seed: int,
        spec,
        *,
        physical_row_order=None,
        extra_slots: tuple[int, ...] = (),
        extra_prefix: int = 0,
    ) -> None:
        self.runtime = runtime
        self.weights = weights
        self.layout, self.histories = _build(
            seed,
            spec,
            physical_row_order=physical_row_order,
            extra_slots=extra_slots,
            extra_prefix=extra_prefix,
        )
        self.tables = _history_tables(weights, self.histories)
        self.rig = Rig(runtime, weights)
        self.oracle = SequentialOracle(runtime, weights)
        self.queries: dict[tuple[int, int], np.ndarray] = {}
        self._reference: dict[tuple[int, int], np.ndarray] | None = None
        seeds = [
            _Seed(request.slot, request.start_position, *self.tables[request.slot])
            for request in self.layout.requests
        ]
        _seed_prefix(
            self.rig.attention,
            self.rig.index,
            seeds + [
                _Seed(slot, extra_prefix, *self.tables[slot]) for slot in extra_slots
            ],
            runtime=runtime,
            allocations=self.rig.allocations,
        )
        # The reference sees only the layout's own slots, from the same prefix.
        _seed_prefix(
            self.oracle.attention,
            self.oracle.index,
            seeds,
            runtime=runtime,
            allocations=self.oracle.allocations,
        )

    @property
    def limit(self) -> int:
        """Largest context the budgeted selection is equivalent to dense."""

        return self.rig.index.dense_equivalent_limit

    def hidden_rows(self) -> np.ndarray:
        """The pass's shared input hidden, one row per physical row."""

        return np.stack(
            [self.histories[row.slot][row.position] for row in self.layout.rows]
        )

    def raw_index_keys(self, slot: int, positions) -> np.ndarray:
        """The reference owner's own raw index key bytes for one slot."""

        return _raw_index_keys(
            self.oracle.attention, self.oracle.index, slot, positions, self.runtime
        )

    def run_pass(self) -> PassResult:
        """Project every physical row, run the pass, and read its result back."""

        layout = self.layout
        total = layout.total_rows
        shared = Qwen4ExpQSAScratch.allocate(
            rows=total,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=self.runtime,
        )
        try:
            scratch = Qwen4ExpMixedQsaScratch.allocate(
                layout=layout,
                attention_state=self.rig.attention,
                index_state=self.rig.index,
                hidden=HIDDEN,
                query_heads=QUERY_HEADS,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                runtime=self.runtime,
            )
        except BaseException:
            shared.close()
            raise
        try:
            mixed = _upload(self.hidden_rows(), self.runtime, self.rig.allocations)
            # The caller's side of the contract: one projection over every
            # physical row, which is what the pass gathers its wave rows from.
            run_qwen4_exp_batch_qsa_input_projections(
                Qwen4ExpBatchQsaMixerPlan(
                    self.runtime, total, Q_WIDTH, KV_WIDTH, (), (), 0
                ),
                mixed.ptr,
                self.weights.device,
                scratch=shared,
                hidden=HIDDEN,
                index_dim=INDEX_DIM,
            )
            run = run_qwen4_exp_mixed_qsa_attention(
                mixed.ptr,
                self.weights.device,
                layout=layout,
                shared=shared,
                attention_state=self.rig.attention,
                index_state=self.rig.index,
                scratch=scratch,
                owner_of=_owner_of,
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
            output = _download(run.output, (total, HIDDEN), np.float32, self.runtime)
        except BaseException:
            scratch.close()
            shared.close()
            raise
        return PassResult(run=run, output=output, shared=shared, scratch=scratch)

    def reference(self) -> dict[tuple[int, int], np.ndarray]:
        """Run every request's chain one row at a time on the reference owners."""

        if self._reference is None:
            outputs: dict[tuple[int, int], np.ndarray] = {}
            for request in self.layout.requests:
                for depth in range(request.rows):
                    position = request.start_position + depth
                    outputs[(request.slot, position)] = self.oracle.step(
                        request.slot, position, self.histories[request.slot][position]
                    )
                    if position + 1 > self.limit:
                        # The sparse step's prepared index query, which the CPU
                        # selection oracle needs and the buffer overwrites next.
                        self.queries[(request.slot, position)] = _download(
                            self.oracle.scratch.index_query,
                            (INDEX_HEADS, INDEX_DIM),
                            np.float32,
                            self.runtime,
                        )
            self._reference = outputs
        return self._reference

    def close(self) -> None:
        self.oracle.close()
        self.rig.close()


def _assert_matches_reference(result: PassResult, staged: Staged, *, label: str) -> None:
    reference = staged.reference()
    for row in staged.layout.rows:
        expected = reference[(row.slot, row.position)]
        actual = result.output[row.physical_row]
        print(
            f"{label} physical={row.physical_row} slot={row.slot} depth={row.depth} "
            f"position={row.position} context={row.context} "
            f"max|adapter-sequential|={np.abs(actual - expected).max():.3e}"
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=ADAPTER_ATOL,
            err_msg=(
                f"{label}: physical row {row.physical_row} (slot {row.slot}, "
                f"position {row.position}) differs from the sequential "
                "same-request reference"
            ),
        )


def _cpu_layer(weights: Weights, staged: Staged, slot: int, position: int) -> np.ndarray:
    """The CPU layer oracle for one row, at its own selected positions."""

    keys, values, _ = staged.tables[slot]
    if position + 1 <= staged.limit:
        selected = np.arange(position + 1)
    else:
        selected = _cpu_selection(
            weights,
            staged.raw_index_keys(slot, range(position + 1)),
            staged.queries[(slot, position)],
            position,
        )
    return weights.cpu_layer(
        staged.histories[slot][position],
        position,
        keys[: position + 1],
        values[: position + 1],
        selected,
    )


# -- tests ------------------------------------------------------------------


def test_dense_chains_and_an_ar_peer_match_the_sequential_reference() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(7201), runtime)
    staged = Staged(runtime, weights, 7201, DENSE_SPEC)
    result = None
    try:
        assert [request.rows for request in staged.layout.requests] == [3, 1, 2]
        assert staged.layout.autoregressive_request_ids == (1,)
        assert max(row.context for row in staged.layout.rows) <= staged.limit
        result = staged.run_pass()
        assert result.run.waves == 3
        assert result.run.wave_rows == (3, 2, 1)
        _assert_matches_reference(result, staged, label="dense")
        # Every row of every chain advanced exactly once.
        slots = [request.slot for request in staged.layout.requests]
        assert staged.rig.index.cursor_host[slots].tolist() == [
            request.start_position + request.rows for request in staged.layout.requests
        ]
        for slot, position in ((0, 2), (2, 1)):
            np.testing.assert_allclose(
                staged.reference()[(slot, position)],
                _cpu_layer(weights, staged, slot, position),
                rtol=ORACLE_RTOL,
                atol=ORACLE_ATOL,
                err_msg=f"the reference itself misses the CPU layer at slot {slot}",
            )
    finally:
        if result is not None:
            result.close()
        staged.close()
        weights.close()


def test_a_physical_row_permutation_moves_rows_not_results() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(7202), runtime)
    straight = Staged(runtime, weights, 7202, DENSE_SPEC)
    permuted = Staged(
        runtime, weights, 7202, DENSE_SPEC, physical_row_order=DENSE_PERMUTATION
    )
    straight_result = permuted_result = None
    try:
        straight_result = straight.run_pass()
        permuted_result = permuted.run_pass()
        # The permutation rewrites physical order only: the same requests, the
        # same histories, and therefore the same reference for both layouts.
        assert [request.slot for request in straight.layout.requests] == [
            request.slot for request in permuted.layout.requests
        ]
        assert [row.semantic_row for row in permuted.layout.rows] == list(
            DENSE_PERMUTATION
        )
        _assert_matches_reference(straight_result, straight, label="straight")
        _assert_matches_reference(permuted_result, permuted, label="permuted")
        # Each slot's own result follows its request to the permuted row.
        for row in straight.layout.rows:
            other = permuted.layout.physical_row(row.request_id, row.depth)
            assert np.array_equal(
                permuted_result.output[other],
                straight_result.output[row.physical_row],
            ), (
                f"slot {row.slot} depth {row.depth} changed with the row order: "
                f"physical row {row.physical_row} became {other}"
            )
    finally:
        for result in (straight_result, permuted_result):
            if result is not None:
                result.close()
        permuted.close()
        straight.close()
        weights.close()


def test_a_chain_crossing_the_dense_bound_matches_the_sequential_reference() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(7203), runtime)
    staged = Staged(runtime, weights, 7203, BOUNDARY_SPEC)
    result = None
    try:
        result = staged.run_pass()
        # The pass really crossed the bound inside one chain: its deepest wave is
        # sparse while the peer's only row is dense.
        slot, position = BOUNDARY_SPARSE_ROW
        assert position + 1 > staged.limit
        assert BOUNDARY_DENSE_ROW[1] + 1 <= staged.limit
        assert result.run.waves == BOUNDARY_SPEC[0][2] + 1
        _assert_matches_reference(result, staged, label="boundary")
        for row_slot, row_position in (BOUNDARY_DENSE_ROW, BOUNDARY_SPARSE_ROW):
            np.testing.assert_allclose(
                staged.reference()[(row_slot, row_position)],
                _cpu_layer(weights, staged, row_slot, row_position),
                rtol=ORACLE_RTOL,
                atol=ORACLE_ATOL,
                err_msg=(
                    f"the reference itself misses the CPU layer at slot {row_slot} "
                    f"position {row_position}"
                ),
            )
        # The sparse row's selection is a strict subset of its context, which is
        # what makes its wave exercise the index pipeline and not just the dense
        # all-visible path.
        selected = _cpu_selection(
            weights,
            staged.raw_index_keys(slot, range(position + 1)),
            staged.queries[(slot, position)],
            position,
        )
        assert 0 < len(selected) < position + 1
    finally:
        if result is not None:
            result.close()
        staged.close()
        weights.close()


def test_committing_the_accepted_prefix_leaves_a_usable_sparse_state() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(7204), runtime)
    staged = Staged(runtime, weights, 7204, COMMIT_SPEC)
    result = None
    follow = reference = None
    allocations: list = []
    try:
        slots = [request.slot for request in staged.layout.requests]
        result = staged.run_pass()
        # The pass appended every row of every chain.
        assert staged.rig.index.cursor_host[slots].tolist() == [
            request.start_position + request.rows for request in staged.layout.requests
        ]
        pooled_before = staged.rig.index.pooled_count_host[slots].copy()
        commit = commit_qwen4_exp_mixed_qsa_attention(
            result.run,
            (0, 0),
            index_state=staged.rig.index,
            owner_of=_owner_of,
        )
        assert commit.accepted_counts == (0, 0)
        assert commit.commit_positions == (COMMIT_SPEC[0][1], COMMIT_SPEC[1][1])
        # Each request keeps its root row and withdraws the rest of its chain.
        assert staged.rig.index.cursor_host[slots].tolist() == [
            COMMIT_POSITION,
            COMMIT_POSITION,
        ]
        # Slot 0 pooled a block that covered a rejected row, so the commit
        # withdraws it; slot 1's pooled blocks all end at its committed cursor.
        assert pooled_before.tolist() == [5, 3]
        assert staged.rig.index.pooled_count_host[slots].tolist() == [
            COMMIT_POSITION // RATIO,
            COMMIT_POSITION // RATIO,
        ]
        # The committed state continues: one more row at the committed cursor,
        # which is sparse, reproduces the reference for the same committed prefix.
        follow = Qwen4ExpBatchQsaSelectionScratch.allocate(
            rows=1,
            stride=OUTPUT_STRIDE,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=runtime,
        )
        follow_scratch = Qwen4ExpQSAScratch.allocate(
            rows=1,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=runtime,
        )
        try:
            reference = SequentialOracle(runtime, weights)
            _seed_prefix(
                reference.attention,
                reference.index,
                [
                    _Seed(slot, COMMIT_POSITION, *staged.tables[slot])
                    for slot in slots
                ],
                runtime=runtime,
                allocations=reference.allocations,
            )
            for slot in slots:
                hidden_row = np.random.default_rng(72040 + slot).normal(
                    0.0, 0.2, size=HIDDEN,
                ).astype(np.float32)
                expected = reference.step(slot, COMMIT_POSITION, hidden_row)
                actual = _decode_row(
                    runtime,
                    weights,
                    attention_state=staged.rig.attention,
                    index_state=staged.rig.index,
                    selection=follow,
                    scratch=follow_scratch,
                    slot=slot,
                    position=COMMIT_POSITION,
                    hidden_row=hidden_row,
                    allocations=allocations,
                )
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=0.0,
                    atol=ADAPTER_ATOL,
                    err_msg=(
                        f"slot {slot} does not continue from its committed prefix: "
                        f"the row decoded at position {COMMIT_POSITION} differs from "
                        "the sequential reference"
                    ),
                )
        finally:
            follow_scratch.close()
    finally:
        if reference is not None:
            reference.close()
        if follow is not None:
            follow.close()
        if result is not None:
            result.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        staged.close()
        weights.close()


def test_abort_restores_the_prefix_and_the_rerun_reproduces_the_pass() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    weights = Weights(np.random.default_rng(7205), runtime)
    staged = Staged(
        runtime,
        weights,
        7205,
        PREFIXED_SPEC,
        extra_slots=(UNRELATED_SLOT,),
        extra_prefix=UNRELATED_PREFIX,
    )
    first = second = None
    try:
        slots = [request.slot for request in staged.layout.requests]
        cursors_before = staged.rig.index.cursor_host.copy()
        pooled_before = staged.rig.index.pooled_count_host.copy()
        prefix_rows = _physical_rows(
            staged.rig.attention,
            {request.slot: range(request.start_position) for request in staged.layout.requests},
        )
        unrelated_rows = _physical_rows(
            staged.rig.attention, {UNRELATED_SLOT: range(UNRELATED_PREFIX)}
        )
        key_before = _key_snapshot(staged.rig.attention, runtime)
        first = staged.run_pass()
        # Only the layout's own slots advanced; the unrelated slot kept both of
        # its counters and every one of its committed cache rows.
        assert staged.rig.index.cursor_host[slots].tolist() == [
            request.start_position + request.rows for request in staged.layout.requests
        ]
        assert staged.rig.index.cursor_host[UNRELATED_SLOT] == UNRELATED_PREFIX
        assert (
            staged.rig.index.pooled_count_host[UNRELATED_SLOT]
            == pooled_before[UNRELATED_SLOT]
        )
        key_after = _key_snapshot(staged.rig.attention, runtime)
        assert np.array_equal(key_after[prefix_rows], key_before[prefix_rows]), (
            "the pass rewrote a committed prefix row's K/V"
        )
        assert np.array_equal(
            key_after[unrelated_rows], key_before[unrelated_rows]
        ), "the pass rewrote a slot outside the layout"
        _assert_matches_reference(first, staged, label="first")
        abort_qwen4_exp_mixed_qsa_attention(first.run, index_state=staged.rig.index)
        # Abort rewinds the logical visibility of every layout slot and nothing
        # else: the unrelated slot was already back at its own values.
        assert np.array_equal(staged.rig.index.cursor_host, cursors_before)
        assert np.array_equal(staged.rig.index.pooled_count_host, pooled_before)
        second = staged.run_pass()
        np.testing.assert_allclose(
            second.output,
            first.output,
            rtol=0.0,
            atol=0.0,
            err_msg="the re-run after abort is not identical to the first pass",
        )
        _assert_matches_reference(second, staged, label="rerun")
    finally:
        for result in (first, second):
            if result is not None:
                result.close()
        staged.close()
        weights.close()
