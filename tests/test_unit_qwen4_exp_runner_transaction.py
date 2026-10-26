from __future__ import annotations

from types import MappingProxyType, MethodType, SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_runner as runner_module
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGGUFResidentModelRunner,
    Qwen4ExpTargetVerifyResult,
)


class _Snapshot:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _State:
    def __init__(self) -> None:
        self.snapshot = _Snapshot()
        self.restored = []

    def device_snapshot(self, snapshot=None):
        return self.snapshot if snapshot is None else snapshot

    def restore_device_snapshot(self, snapshot) -> None:
        self.restored.append(snapshot)


class _Attention:
    def __init__(self) -> None:
        self.positions = []

    def set_position(self, position: int) -> None:
        self.positions.append(int(position))


class _Index:
    def __init__(self) -> None:
        self.counts = []

    def restore_count(self, count: int) -> None:
        self.counts.append(int(count))


def _runner() -> Qwen4ExpGGUFResidentModelRunner:
    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    runner.state = _State()
    runner.position = 7
    runner._ple_hash_states = {1: "before"}
    runner.attention_states = (_Attention(), _Attention())
    runner.index_states = (_Index(), _Index())
    runner._device_transaction_snapshot = None
    runner._device_transaction_lease = False
    runner._shared_position_context = None
    runner._shared_position_context_host = None
    return runner


class _CompactAttention:
    """``Qwen4ExpDenseAttentionState`` surface ``adopt_prefix_cursor`` touches.

    Mirrors the real state's checks and its two per-state 8-byte uploads, so a
    test can count uploads on both the compact and the per-state route.
    """

    def __init__(self, *, max_positions: int = 64, closed: bool = False) -> None:
        self.max_positions = max_positions
        self.closed = closed
        self.runtime = None
        self.position = DeviceBuffer(0x100, 8)
        self.context = DeviceBuffer(0x108, 8)
        self.position_host = np.zeros(1, dtype=np.int64)
        self.context_host = np.ones(1, dtype=np.int64)
        self.positions: list[int] = []

    def set_position(self, value: int) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp attention state is closed")
        position = int(value)
        if not 0 <= position < self.max_positions:
            raise ValueError("attention position exceeds state capacity")
        self.positions.append(position)
        self.position_host[0] = position
        self.context_host[0] = position + 1
        runner_module.copy_host_array_to_device(
            self.position, self.position_host, runtime=self.runtime
        )
        runner_module.copy_host_array_to_device(
            self.context, self.context_host, runtime=self.runtime
        )


class _CompactIndex:
    """``Qwen4ExpQSAIndexDeviceState.restore_count`` without device buffers."""

    def __init__(
        self, *, capacity: int = 64, closed: bool = False, compression_ratio: int = 4
    ) -> None:
        self.capacity = capacity
        self.closed = closed
        self.compression_ratio = compression_ratio
        self.count = 0
        self.pooled_count = 0
        self.counts: list[int] = []

    def restore_count(self, count: int) -> None:
        if self.closed:
            raise RuntimeError("QSA index state is closed")
        restored = int(count)
        if restored < 0 or restored > self.capacity:
            raise ValueError("QSA index restore count exceeds capacity")
        self.counts.append(restored)
        self.count = restored
        self.pooled_count = min(self.pooled_count, restored // self.compression_ratio)


@pytest.fixture
def uploads(monkeypatch):
    """Record every host-to-device copy the cursor path issues."""

    calls: list[tuple[int, int]] = []

    def record(buffer, array, nbytes=None, *, runtime=None):
        calls.append((int(buffer.ptr), array.nbytes if nbytes is None else int(nbytes)))

    monkeypatch.setattr(runner_module, "copy_host_array_to_device", record)
    return calls


def _cursor_runner(
    *,
    states: int = 2,
    shared: bool = True,
    max_positions: int = 64,
    capacity: int = 64,
    attention_closed: bool = False,
    index_closed: bool = False,
) -> Qwen4ExpGGUFResidentModelRunner:
    runner = _runner()
    runner.runtime = None
    runner.attention_states = tuple(
        _CompactAttention(max_positions=max_positions, closed=attention_closed)
        for _ in range(states)
    )
    runner.index_states = tuple(
        _CompactIndex(capacity=capacity, closed=index_closed) for _ in range(states)
    )
    if shared:
        nbytes = 2 * states * 8
        runner._shared_position_context = DeviceBuffer(0x2000, nbytes)
        runner._shared_position_context_host = np.zeros(2 * states, dtype=np.int64)
        runner._shared_position_context_host[1::2] = 1
    return runner


def test_qwen4_exp_device_transaction_rolls_back_all_owned_cursors() -> None:
    runner = _runner()
    transaction = runner.begin_device_transaction()
    assert transaction.position == 7
    assert transaction.ple_hash_states == MappingProxyType({1: "before"})

    runner.position = 11
    runner._ple_hash_states = {2: "after"}
    runner.rollback_device_transaction(transaction)

    assert runner.position == 7
    assert runner._ple_hash_states == {1: "before"}
    assert runner.state.restored == [transaction.decode_state]
    assert [row.positions for row in runner.attention_states] == [[6], [6]]
    assert [row.counts for row in runner.index_states] == [[7], [7]]
    assert transaction.rolled_back and not transaction.committed
    assert transaction.closed
    with pytest.raises(RuntimeError, match="already finalized"):
        runner.rollback_device_transaction(transaction)


def test_qwen4_exp_serial_verify_oracle_returns_every_row() -> None:
    runner = _runner()
    calls = []

    def step(self, token_id, **kwargs):
        calls.append((int(token_id), dict(kwargs)))
        self.position += 1
        return SimpleNamespace(
            token_id=int(token_id) + 10,
            logits=np.asarray([token_id, token_id + 1], dtype=np.float32),
            hidden_seeds=np.full((1, 3), token_id, dtype=np.float32),
        )

    runner.step = MethodType(step, runner)
    result = runner.verify_target_block_serial_exact(
        (1, 2, 3), capture_logits=True, capture_hidden_seeds=True
    )

    assert result.token_ids == (11, 12, 13)
    assert len(result.logits) == 3
    assert result.hidden_seeds.shape == (3, 3)
    assert runner.position == 10
    assert [row[0] for row in calls] == [1, 2, 3]
    assert all(row[1]["capture_target_hidden"] for row in calls)
    with pytest.raises(ValueError, match="rows must be in 1..8"):
        runner.verify_target_block_serial_exact(())


def test_qwen4_exp_reusable_device_transaction_leases_one_snapshot() -> None:
    runner = _runner()
    first = runner.begin_device_transaction(reuse_snapshot=True)
    pooled = first.decode_state
    with pytest.raises(RuntimeError, match="already leased"):
        runner.begin_device_transaction(reuse_snapshot=True)
    runner.commit_device_transaction(first)
    assert not pooled.closed

    second = runner.begin_device_transaction(reuse_snapshot=True)
    assert second.decode_state is pooled
    runner.rollback_device_transaction(second)
    assert not pooled.closed
    assert not runner._device_transaction_lease


@pytest.mark.parametrize("rejection_depth", (0, 1, 2, 3, None))
def test_qwen4_exp_acceptance_replays_only_rejected_prefix(rejection_depth) -> None:
    runner = _runner()
    candidates = (11, 12, 13, 14)
    truths = list(candidates)
    if rejection_depth is not None:
        truths[rejection_depth] += 100
    block = Qwen4ExpTargetVerifyResult(
        token_ids=tuple(truths), logits=None, hidden_seeds=None
    )
    calls = []
    transaction = SimpleNamespace()
    runner.begin_device_transaction = lambda **kwargs: (
        calls.append(("begin", kwargs)) or transaction
    )
    runner.verify_target_block_deferred_head = lambda inputs: (
        calls.append(("verify", tuple(inputs))) or block
    )
    runner.commit_device_transaction = lambda tx: calls.append(("commit", tx))
    runner.rollback_device_transaction = lambda tx: calls.append(("rollback", tx))

    def replay(inputs, **kwargs):
        calls.append(("replay", tuple(inputs), kwargs))
        count = len(inputs)
        return Qwen4ExpTargetVerifyResult(
            token_ids=tuple(truths[:count]), logits=None, hidden_seeds=None
        )

    runner.verify_target_block_serial_exact = replay
    result = runner.verify_target_block_with_acceptance(
        (10, 11, 12, 13), candidates
    )

    accepted = 4 if rejection_depth is None else rejection_depth
    consumed = 4 if rejection_depth is None else rejection_depth + 1
    assert result.accepted == accepted
    assert result.consumed == consumed
    assert result.replayed == (rejection_depth is not None and consumed < 4)
    assert result.verify.token_ids == tuple(truths[:consumed])
    if consumed < 4:
        assert ("rollback", transaction) in calls
        assert any(call[0] == "replay" and len(call[1]) == consumed for call in calls)
    else:
        assert ("commit", transaction) in calls


def test_qwen4_exp_adopt_prefix_cursor_uploads_the_compact_slots_once(uploads) -> None:
    runner = _cursor_runner(states=3)

    runner.adopt_prefix_cursor(7, {1: "before"})

    # One interleaved [position, context] copy for all three QSA states.
    assert uploads == [(0x2000, 48)]
    assert runner._shared_position_context_host.tolist() == [6, 7, 6, 7, 6, 7]
    for state in runner.attention_states:
        assert state.positions == [], "the compact route must not run per-state copies"
        assert state.position_host[0] == 6
        assert state.context_host[0] == 7
    assert [index.counts for index in runner.index_states] == [[7], [7], [7]]
    assert runner.position == 7
    assert runner._ple_hash_states == {1: "before"}


def test_qwen4_exp_adopt_prefix_cursor_keeps_the_per_state_path(uploads) -> None:
    runner = _cursor_runner(states=2, shared=False)

    runner.adopt_prefix_cursor(7, {1: "before"})

    # Two 8-byte copies per state, as before, when there is no shared staging.
    assert len(uploads) == 4
    assert [state.positions for state in runner.attention_states] == [[6], [6]]
    for state in runner.attention_states:
        assert state.position_host[0] == 6
        assert state.context_host[0] == 7
    assert [index.counts for index in runner.index_states] == [[7], [7]]
    assert runner.position == 7


def test_qwen4_exp_adopt_prefix_cursor_owns_index_ple_and_zero_cursor(uploads) -> None:
    runner = _cursor_runner(states=2)
    states = {1: "before"}

    runner.adopt_prefix_cursor(3, states)
    states[2] = "later"

    assert runner._ple_hash_states == {1: "before"}
    assert runner._ple_hash_states is not states
    assert [index.count for index in runner.index_states] == [3, 3]
    # Dense rows restore a raw count; pooled blocks can only be clamped.
    assert [index.pooled_count for index in runner.index_states] == [0, 0]

    uploads.clear()
    runner.adopt_prefix_cursor(0, {})
    assert uploads == [(0x2000, 32)]
    assert runner.position == 0
    assert runner._ple_hash_states == {}
    assert [state.position_host[0] for state in runner.attention_states] == [0, 0]
    assert [state.context_host[0] for state in runner.attention_states] == [1, 1]


@pytest.mark.parametrize(
    "case,position,kwargs,error,match",
    [
        ("closed_attention", 4, {"attention_closed": True}, RuntimeError,
         "attention state is closed"),
        ("closed_index", 4, {"index_closed": True}, RuntimeError,
         "QSA index state is closed"),
        ("attention_bounds", 8, {"max_positions": 4}, ValueError,
         "attention position exceeds state capacity"),
        ("index_bounds", 5, {"capacity": 4}, ValueError,
         "QSA index restore count exceeds capacity"),
        ("negative_position", -1, {}, ValueError,
         "QSA index restore count exceeds capacity"),
    ],
)
def test_qwen4_exp_adopt_prefix_cursor_refuses_before_uploading(
    uploads, case, position, kwargs, error, match
) -> None:
    runner = _cursor_runner(states=2, **kwargs)

    with pytest.raises(error, match=match):
        runner.adopt_prefix_cursor(position, {1: "before"})

    assert uploads == [], f"{case} must refuse before any device write"
    assert runner.position == 7
    assert runner._ple_hash_states == {1: "before"}
    assert all(state.positions == [] for state in runner.attention_states)
    assert all(index.counts == [] for index in runner.index_states)


def test_qwen4_exp_adopt_prefix_cursor_refuses_a_closed_runner(uploads) -> None:
    runner = _cursor_runner(states=2)
    runner.closed = True

    with pytest.raises(RuntimeError, match="closed"):
        runner.adopt_prefix_cursor(4, {1: "before"})

    assert uploads == []
    assert runner.position == 7


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("consumed", [1, 2, 3])
def test_partial_verify_commit_publishes_cursors_through_shared_helper(
    uploads, monkeypatch, shared, consumed
):
    runner = _cursor_runner(states=3, shared=shared)
    runner._pending_verify_block = ((10, 11, 12, 13), 7, {1: "start"}, 4)
    runner.state.ple_conv = DeviceBuffer(0x3000, 32)
    runner.ple_prefill_scratch = object()
    runner._verify_rows_buffers = {"ple_conv": DeviceBuffer(0x4000, 32)}
    runner._state_row_strides = lambda: (16, 8)
    runner.gdn_bindings = {}
    copies = []
    runner.runtime = SimpleNamespace(memcpy_async=lambda *args: copies.append(args))
    runner.config = SimpleNamespace(
        ple_layers=(), ple_eos_token_id=0, ple_layer_multipliers=(),
        ple_head_offsets=(), ple_head_vocab_sizes=(), ple_heads_per_ngram=1,
        ple_ngram_size=3,
    )
    hashes = {2: "accepted"}
    hash_calls = []
    def hash_rows(tokens, **kwargs):
        hash_calls.append((tokens, kwargs))
        return None, hashes
    monkeypatch.setattr(runner_module, "ple_hash_rows", hash_rows)

    runner.commit_target_block_rows(consumed)

    assert runner._pending_verify_block is None
    assert runner.position == 7 + consumed
    assert runner.ple_hash_states == hashes
    assert hash_calls[0][0] == [10, 11, 12][:consumed]
    assert hash_calls[0][1]["positions"] == list(range(7, 7 + consumed))
    assert hash_calls[0][1]["states"] == {1: "start"}
    assert len(copies) == 1  # PLE history restoration is unchanged.
    assert len(uploads) == (1 if shared else 6)
    if shared:
        assert uploads == [(0x2000, 48)]
    assert all(s.position_host[0] == 6 + consumed for s in runner.attention_states)
    assert all(s.context_host[0] == 7 + consumed for s in runner.attention_states)
    assert all(s.count == 7 + consumed for s in runner.index_states)


def test_qwen4_exp_device_transaction_commit_keeps_current_state() -> None:
    runner = _runner()
    transaction = runner.begin_device_transaction()
    runner.position = 9
    runner._ple_hash_states = {3: "committed"}

    runner.commit_device_transaction(transaction)

    assert runner.position == 9
    assert runner._ple_hash_states == {3: "committed"}
    assert runner.state.restored == []
    assert transaction.committed and not transaction.rolled_back
    assert transaction.closed
    with pytest.raises(RuntimeError, match="already finalized"):
        runner.commit_device_transaction(transaction)
