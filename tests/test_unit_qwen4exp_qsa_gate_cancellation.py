"""Cancellation arm for the QSA retrieval gate, against a behavioral fake runner.

The fake keeps the state the arm reads (cursor, per-layer index counts, pooled
counts, K/V and pooled-key bytes, the private pending-append triple) and does no
arithmetic of its own. The arm's real logic — boundary assertions, private
close, rewind, replay and reference — runs unmodified, so a regression in its
expectations shows up here without a GPU.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.core.memory import host_array_ptr
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpQSAIndexDeviceState
from scripts import qwen4_exp_qsa_retrieval_gate as gate

RATIO = 4
BLOCK_SIZE = 4
BLOCKS = 4
ROWS = BLOCKS * BLOCK_SIZE
ROW_BYTES = 4
BASE = 12
LAYER_BLOCKS = ([0, 1, 3, 2], [2, 3, 0, 1])


class _Device:
    """One flat byte array standing in for device memory."""

    def __init__(self, size: int) -> None:
        self.bytes = np.zeros(size, dtype=np.uint8)
        self.base = host_array_ptr(self.bytes)

    def buffer(self, offset: int, nbytes: int) -> SimpleNamespace:
        return SimpleNamespace(ptr=self.base + offset, nbytes=nbytes, offset=offset)

    def memcpy(self, dst: int, src: int, nbytes: int, kind: object) -> None:
        ctypes.memmove(dst, int(src), nbytes)

    def fill(self, offset: int, nbytes: int, value: int) -> None:
        self.bytes[offset : offset + nbytes] = value


class _Attention:
    def __init__(self, device: _Device, offset: int, *, block_host: list[int]) -> None:
        self.block_size = BLOCK_SIZE
        self.block_host = np.asarray(block_host, dtype=np.int64)
        self.max_positions = len(block_host) * BLOCK_SIZE
        size = len(block_host) * BLOCK_SIZE * ROW_BYTES
        self.key_cache = device.buffer(offset, size)
        self.value_cache = device.buffer(offset + size, size)


def _state(device: _Device, offset: int, *, dense_limit: int) -> Mock:
    state = Mock(spec=Qwen4ExpQSAIndexDeviceState)
    state.capacity = BLOCKS * RATIO
    state.compression_ratio = RATIO
    state.dense_equivalent_limit = dense_limit
    state.pooled_keys = device.buffer(offset, BLOCKS * ROW_BYTES)
    state.count = 0
    state.pooled_count = 0
    return state


class _Context:
    def run(self, fn: object) -> object:
        return fn()  # type: ignore[operator]


class _Steps:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.log.append("close")


class _FakeRunner:
    """Two QSA layers over permuted block tables; a chunk writes all of its rows."""

    def __init__(
        self,
        *,
        base: int = BASE,
        suspended_salt: int = 0,
        divergent_replay: bool = False,
        clamp_pooled: bool = True,
        suspend_returns_result: bool = False,
        mid_layer_stop: bool = False,
        attention_states: tuple[object, ...] | None = None,
        dense_mode: bool = False,
        no_pool_advance: bool = False,
        zero_kv: bool = False,
    ) -> None:
        self.position = 0
        self.max_sequence_length = 64
        self.prefill_chunk_size = 8
        self.ratio = RATIO
        self.suspended_salt = suspended_salt
        self.divergent_replay = divergent_replay
        self.clamp_pooled = clamp_pooled
        self.suspend_returns_result = suspend_returns_result
        self.mid_layer_stop = mid_layer_stop
        self.no_pool_advance = no_pool_advance
        self.close_log: list[str] = []
        self.append_calls = 0
        self._pending_prefill_append = None
        attention_bytes = 2 * 2 * ROWS * ROW_BYTES
        self.device = _Device(attention_bytes + 2 * BLOCKS * ROW_BYTES)
        self.attention_states = (
            (
                _Attention(self.device, 0, block_host=LAYER_BLOCKS[0]),
                _Attention(self.device, 2 * ROWS * ROW_BYTES, block_host=LAYER_BLOCKS[1]),
            )
            if attention_states is None
            else attention_states
        )
        dense_limit = 64 if dense_mode else 8
        self.index_states = (
            _state(self.device, attention_bytes, dense_limit=dense_limit),
            _state(
                self.device,
                attention_bytes + BLOCKS * ROW_BYTES,
                dense_limit=dense_limit,
            ),
        )
        if zero_kv:
            for attention in self.attention_states:
                attention.key_cache.nbytes = 0
                attention.value_cache.nbytes = 0
        self.qsa_bindings = {
            1: SimpleNamespace(qsa_state_index=0),
            3: SimpleNamespace(qsa_state_index=1),
        }
        self.runtime = self.device
        self.position = base
        for state in self.index_states:
            state.count = base
            state.pooled_count = base // self.ratio

    # -- state helpers -----------------------------------------------------
    def snapshot(self) -> SimpleNamespace:
        # The real snapshot carries the cursor and non-append state only; the
        # pooled cursor is rewound by restore_count's clamp, not by the snapshot.
        return SimpleNamespace(
            position=self.position,
            counts=[state.count for state in self.index_states],
        )

    def restore(self, snapshot: SimpleNamespace) -> None:
        self.position = snapshot.position
        for state, count in zip(self.index_states, snapshot.counts):
            state.count = count
            if self.clamp_pooled:
                state.pooled_count = min(state.pooled_count, count // self.ratio)

    # -- device writes -----------------------------------------------------
    def _write_chunk(
        self, start: int, count: int, salt: int, layers: int | None = None
    ) -> None:
        for layer, attention in enumerate(self.attention_states):
            if layers is not None and layer >= layers:
                break
            rows = gate._physical_rows(attention, start, count)
            for name in ("key_cache", "value_cache"):
                buffer = getattr(attention, name)
                bias = 0 if name == "key_cache" else 1
                for index, row in enumerate(rows):
                    self.device.fill(
                        buffer.offset + row * ROW_BYTES,
                        ROW_BYTES,
                        (layer * 31 + (start + index) * 17 + bias + salt) & 0xFF,
                    )
        first = start // self.ratio
        last = (start + count) // self.ratio
        for layer, state in enumerate(self.index_states):
            if layers is not None and layer >= layers:
                break
            for block in range(first, last):
                self.device.fill(
                    state.pooled_keys.offset + block * ROW_BYTES,
                    ROW_BYTES,
                    (layer * 53 + block * 7 + salt) & 0xFF,
                )

    def _commit_index(self, start: int, count: int, layers: int | None = None) -> None:
        for layer, state in enumerate(self.index_states):
            if layers is not None and layer >= layers:
                break
            state.count = start + count
            if not self.no_pool_advance:
                state.pooled_count = (start + count) // self.ratio

    # -- the two prefill entries the arm uses ------------------------------
    def prefill_append_resumable(self, tokens: object, *, time_budget_seconds: float):
        assert time_budget_seconds > 0
        start = self.position
        reached = 1 if self.mid_layer_stop else None
        self._write_chunk(start, len(tokens), self.suspended_salt, layers=reached)
        self._commit_index(start, len(tokens), layers=reached)
        self._pending_prefill_append = (_Context(), _Steps(self.close_log), tuple(tokens))
        if self.suspend_returns_result:
            return SimpleNamespace(logits=np.zeros(2, dtype=np.float32))
        return None

    def prefill_append(self, tokens: object) -> SimpleNamespace:
        start = self.position
        self._write_chunk(start, len(tokens), 0)
        self._commit_index(start, len(tokens))
        self.position = start + len(tokens)
        self.append_calls += 1
        logits = np.asarray(
            [
                start + len(tokens),
                sum(state.count for state in self.index_states),
                sum(state.pooled_count for state in self.index_states),
            ],
            dtype=np.float32,
        )
        if self.divergent_replay and self.append_calls == 1:
            logits = logits + np.float32(1.0)
        return SimpleNamespace(logits=logits)


def _arm(runner: _FakeRunner, probe: tuple[int, ...] = (5, 6, 7, 8)) -> dict[str, object]:
    baseline = runner.snapshot()
    return gate.cancellation_arm(
        runner, baseline, probe, ratio=RATIO, progress=lambda message: None
    )


def test_cancellation_arm_suspends_at_the_all_layer_boundary_then_replays_clean() -> None:
    runner = _FakeRunner()

    report = _arm(runner)

    assert report["status"] == "exercised"
    assert report["suspended_at"] == "all_layer_flush"
    assert report["chunk_tokens"] == 4
    assert report["cursor_before_rewind"] == BASE
    assert report["cursor_uncommitted"] is True
    # Every layer appended the chunk while the cursor stayed uncommitted.
    assert report["index_appended_all"] is True
    assert report["index_counts_before_rewind"] == {"layers": 2, "unique": [BASE + 4]}
    assert report["pooled_blocks_before_rewind"] == {
        "layers": 2,
        "unique": [(BASE + 4) // RATIO],
        "blocks_completed": 1,
        "mode": "sparse",
        "expected_frontier": (BASE + 4) // RATIO,
    }
    # The abandoned writes are byte-identical to a committed chunk's writes.
    assert report["kv_rows_compared"] == 4
    assert report["kv_rows_exact"] is True
    assert report["pooled_keys_exact"] is True
    assert report["pooled_blocks_compared"] == 2
    # The private close ran, the cursor and pooled counters rewound, and the
    # replay reproduced the reference logits.
    assert runner.close_log == ["close"]
    assert report["cursor_after_rewind"] == BASE
    assert report["index_counts_after_rewind"] == {"layers": 2, "unique": [BASE]}
    assert report["rewind_exact"] is True
    assert report["replay_exact"] is True
    assert report["boundary_verified"] is True
    assert report["pooled_consistent"] is True
    assert report["coverage_gated"] is True
    assert report["kv_captures_complete"] is True
    assert report["attention_states"] == 2
    assert report["kv_evidence"] == "row_overwrite_equality_shared_storage"
    assert report["clean"] is True
    assert runner.position == BASE  # the arm hands the baseline back
    coverage = report["coverage"]
    assert coverage["total"] == 2
    assert coverage["qsa_states"] == 2
    assert coverage["bound_indices"] == [0, 1]
    assert coverage["covers_all"] is True
    assert len(coverage["types"]) == 1


def test_cancellation_arm_detects_suspended_writes_that_differ_from_committed() -> None:
    runner = _FakeRunner(suspended_salt=9)

    report = _arm(runner)

    assert report["kv_rows_exact"] is False
    assert report["pooled_keys_exact"] is False
    assert report["clean"] is False


def test_cancellation_arm_detects_a_rewind_that_leaves_the_pooled_cursor() -> None:
    runner = _FakeRunner(clamp_pooled=False)

    report = _arm(runner)

    assert report["rewind_exact"] is False
    assert report["clean"] is False


def test_cancellation_arm_detects_a_divergent_replay() -> None:
    runner = _FakeRunner(divergent_replay=True)

    report = _arm(runner)

    assert report["replay_exact"] is False
    assert report["clean"] is False


def test_cancellation_arm_requires_a_suspension_not_a_completed_chunk() -> None:
    runner = _FakeRunner(suspend_returns_result=True)

    with pytest.raises(RuntimeError, match="expected a suspension"):
        _arm(runner)


def test_cancellation_arm_refuses_probes_that_cannot_fit() -> None:
    runner = _FakeRunner()

    with pytest.raises(ValueError, match="prefill chunk size"):
        _arm(runner, probe=tuple(range(9)))
    with pytest.raises(ValueError, match="at least one token"):
        _arm(runner, probe=())


def test_cancellation_arm_gates_incomplete_binding_coverage() -> None:
    runner = _FakeRunner()
    runner.qsa_bindings = {1: SimpleNamespace(qsa_state_index=0)}

    report = _arm(runner)

    # The bound layer's evidence is fine; the coverage gate is what fails.
    assert report["suspended_at"] == "all_layer_flush"
    assert report["coverage_gated"] is False
    assert report["clean"] is False


def test_cancellation_arm_requires_nonempty_kv_captures() -> None:
    runner = _FakeRunner(attention_states=())

    report = _arm(runner)

    assert report["attention_states"] == 0
    assert report["kv_rows_compared"] == 0
    assert report["kv_captures_complete"] is False
    assert report["kv_rows_exact"] is False
    assert report["clean"] is False


def test_cancellation_arm_marks_a_mid_layer_stop_unverified() -> None:
    runner = _FakeRunner(mid_layer_stop=True)

    report = _arm(runner)

    assert report["cursor_uncommitted"] is True
    assert report["index_appended_all"] is False
    assert report["pooled_consistent"] is False
    assert report["boundary_verified"] is False
    assert report["suspended_at"] == "unverified"
    assert report["clean"] is False


def test_cancellation_arm_rejects_a_sparse_frontier_that_does_not_advance() -> None:
    runner = _FakeRunner(no_pool_advance=True)

    report = _arm(runner)

    # The chunk appends keys that complete a block, so a sparse selection must
    # have pooled it. Not advancing is not a pass, and the stale pooled bytes are
    # not compared at all.
    assert report["pooled_blocks_before_rewind"]["mode"] == "sparse"
    assert report["pooled_blocks_before_rewind"]["expected_frontier"] == (BASE + 4) // RATIO
    assert report["pooled_consistent"] is False
    assert report["boundary_verified"] is False
    assert report["suspended_at"] == "unverified"
    assert report["pooled_blocks_compared"] == 0
    assert report["pooled_keys_exact"] == "not_exercised"
    assert report["clean"] is False


def test_cancellation_arm_marks_dense_pooled_bytes_not_exercised() -> None:
    runner = _FakeRunner(dense_mode=True)

    report = _arm(runner)

    # Dense rows append raw keys without pooled blocks: the frontier is not this
    # chunk's work, so nothing is claimed about pooled bytes.
    assert report["pooled_blocks_before_rewind"]["mode"] == "dense"
    assert report["pooled_blocks_before_rewind"]["expected_frontier"] is None
    assert report["pooled_consistent"] is True
    assert report["boundary_verified"] is True
    assert report["pooled_blocks_compared"] == 0
    assert report["pooled_keys_exact"] == "not_exercised"
    assert report["clean"] is True


def test_cancellation_arm_requires_nonzero_kv_capture_bytes() -> None:
    runner = _FakeRunner(zero_kv=True)

    report = _arm(runner)

    # The capture dict has the right shape but holds no bytes, so it proves
    # nothing and must not count as a complete capture.
    assert report["kv_rows_compared"] == 4
    assert report["kv_captures_complete"] is False
    assert report["kv_rows_exact"] is False
    assert report["clean"] is False


def test_kv_geometry_requires_whole_token_rows() -> None:
    attention = _Attention(_Device(4096), 0, block_host=[0, 1])
    attention.key_cache.nbytes = attention.key_cache.nbytes + 1

    with pytest.raises(RuntimeError, match="whole number of"):
        gate._kv_geometry(attention)


def test_kv_geometry_rejects_an_over_reserved_block_table() -> None:
    attention = _Attention(_Device(4096), 0, block_host=[0, 1])
    attention.max_positions = attention.max_positions - BLOCK_SIZE

    with pytest.raises(RuntimeError, match="over-reserves"):
        gate._kv_geometry(attention)


def test_kv_geometry_rejects_a_table_that_cannot_cover_the_capacity() -> None:
    attention = _Attention(_Device(4096), 0, block_host=[0, 1])
    attention.max_positions = attention.max_positions + BLOCK_SIZE

    with pytest.raises(RuntimeError, match="covers"):
        gate._kv_geometry(attention)


def test_kv_geometry_accepts_the_paged_page_tail() -> None:
    attention = _Attention(_Device(4096), 0, block_host=[0, 1, 2])
    attention.max_positions = 2 * BLOCK_SIZE + 1  # one page of tail

    slots, row_bytes = gate._kv_geometry(attention)

    assert slots == 3 * BLOCK_SIZE
    assert row_bytes == ROW_BYTES


def test_qsa_layer_states_rejects_an_index_state_of_another_type() -> None:
    runner = _FakeRunner()
    runner.index_states = (runner.index_states[0], SimpleNamespace(count=0))

    with pytest.raises(RuntimeError, match="not Qwen4ExpQSAIndexDeviceState"):
        gate._qsa_layer_states(runner)


def test_index_state_coverage_reports_a_binding_gap() -> None:
    runner = _FakeRunner()
    assert gate._index_state_coverage(runner)["covers_all"] is True

    runner.qsa_bindings = {1: SimpleNamespace(qsa_state_index=0)}

    coverage = gate._index_state_coverage(runner)
    assert coverage["covers_all"] is False
    assert coverage["bound_indices"] == [0]
    assert coverage["total"] == 2


def test_physical_rows_follow_the_block_table_not_the_identity() -> None:
    attention = _Attention(_Device(4096), 0, block_host=[1, 0])

    assert gate._physical_rows(attention, 0, 2) == [4, 5]
    assert gate._physical_rows(attention, 3, 3) == [7, 0, 1]


def test_retrieval_pass_gates_an_exercised_cancellation_arm() -> None:
    report = {
        "retrieval_exact": True,
        "tokenizer_match": True,
        "cpu_index_oracle": {"exact_positions": True},
        "transactional": {"repeat_exact": True, "isolation_exact": True},
        "memory_after_close": {"current_allocated_bytes": 0},
    }

    assert gate.retrieval_passed({**report, "cancellation": {"status": "not_exercised"}}, None) is True
    assert gate.retrieval_passed({**report, "cancellation": {"status": "exercised", "clean": True}}, None) is True
    assert gate.retrieval_passed({**report, "cancellation": {"status": "exercised", "clean": False}}, None) is False
    # The existing arms still decide on their own.
    assert gate.retrieval_passed({**report, "retrieval_exact": False}, None) is False


def test_cancel_arm_is_on_by_default_and_off_by_flag() -> None:
    parser = gate.build_parser()
    base = ["model", "--json-out", "out.json"]

    assert parser.parse_args(base).cancel_arm is True
    assert parser.parse_args(base + ["--no-cancel-arm"]).cancel_arm is False
    assert parser.parse_args(base + ["--cancel-arm"]).cancel_arm is True
