from __future__ import annotations

from types import MappingProxyType

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

    def device_snapshot(self):
        return self.snapshot

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
