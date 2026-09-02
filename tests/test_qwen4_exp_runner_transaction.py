from __future__ import annotations

from types import MappingProxyType, MethodType, SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner


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
