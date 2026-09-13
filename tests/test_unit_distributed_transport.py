"""CPU tests for collective group protocol and the mock transport.

The mock transport is the executable specification for the enqueue discipline
the RCCL transport must follow: same group shape on every rank, validation
before issue, failure poisons the group, and completion is explicit.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.distributed.mock import MockMemory, MockTransport
from hipengine.distributed.transport import (
    CollectiveKind,
    CollectiveRequest,
    CommunicatorAbortedError,
    EnqueueRecorder,
    TransportError,
    TransportStateError,
)


def _rank_buffers(transport: MockTransport, count: int) -> tuple[list[int], list[int]]:
    send = [transport.memory.alloc(rank, count * 4) for rank in range(transport.world_size)]
    recv = [transport.memory.alloc(rank, count * 4) for rank in range(transport.world_size)]
    return send, recv


def test_mock_all_reduce_sums_across_ranks() -> None:
    transport = MockTransport(world_size=2)
    count = 4
    send, recv = _rank_buffers(transport, count)
    transport.memory.write(0, send[0], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    transport.memory.write(1, send[1], np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))

    transport.group_start()
    for rank in range(2):
        transport.all_reduce_sum(rank, send[rank], recv[rank], count=count, dtype="fp32")
    transport.group_end()
    transport.sync()

    expected = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32)
    for rank in range(2):
        np.testing.assert_array_equal(transport.memory.read(rank, recv[rank], dtype="fp32", count=count), expected)
    assert transport.sequence == 1
    assert transport.groups_committed == 1
    assert transport.bytes_communicated == 2 * count * 4


def test_mock_all_reduce_uneven_counts_across_ranks() -> None:
    """Each rank has its own local shard; the all-reduce count is the full width."""

    transport = MockTransport(world_size=3)
    full = 6
    send = [transport.memory.alloc(rank, full * 4) for rank in range(3)]
    recv = [transport.memory.alloc(rank, full * 4) for rank in range(3)]
    for rank in range(3):
        transport.memory.write(rank, send[rank], np.full(full, float(rank + 1), dtype=np.float32))
    transport.group_start()
    for rank in range(3):
        transport.all_reduce_sum(rank, send[rank], recv[rank], count=full, dtype="fp32")
    transport.group_end()
    transport.sync()
    expected = np.full(full, 6.0, dtype=np.float32)
    for rank in range(3):
        np.testing.assert_array_equal(transport.memory.read(rank, recv[rank], dtype="fp32", count=full), expected)


def test_mock_broadcast_copies_root_payload() -> None:
    transport = MockTransport(world_size=2)
    count = 3
    send = [transport.memory.alloc(rank, count * 4) for rank in range(2)]
    recv = [transport.memory.alloc(rank, count * 4) for rank in range(2)]
    transport.memory.write(1, send[1], np.array([7.0, 8.0, 9.0], dtype=np.float32))
    transport.group_start()
    for rank in range(2):
        transport.broadcast(rank, send[rank], recv[rank], count=count, dtype="fp32", root=1)
    transport.group_end()
    transport.sync()
    for rank in range(2):
        np.testing.assert_array_equal(
            transport.memory.read(rank, recv[rank], dtype="fp32", count=count),
            np.array([7.0, 8.0, 9.0], dtype=np.float32),
        )


def test_enqueue_outside_group_is_rejected() -> None:
    transport = MockTransport(world_size=2)
    send, recv = _rank_buffers(transport, 2)
    with pytest.raises(TransportStateError):
        transport.all_reduce_sum(0, send[0], recv[0], count=2, dtype="fp32")


def test_group_end_without_start_is_rejected() -> None:
    transport = MockTransport(world_size=2)
    with pytest.raises(TransportStateError):
        transport.group_end()


def test_double_group_start_is_rejected() -> None:
    transport = MockTransport(world_size=2)
    send, recv = _rank_buffers(transport, 2)
    transport.group_start()
    with pytest.raises(TransportStateError):
        transport.group_start()
    for rank in range(2):
        transport.all_reduce_sum(rank, send[rank], recv[rank], count=2, dtype="fp32")
    transport.group_end()
    assert transport.sequence == 1


def test_empty_group_is_rejected() -> None:
    transport = MockTransport(world_size=2)
    transport.group_start()
    with pytest.raises(TransportStateError):
        transport.group_end()


def test_missing_rank_collective_is_rejected() -> None:
    transport = MockTransport(world_size=2)
    send, recv = _rank_buffers(transport, 2)
    transport.group_start()
    transport.all_reduce_sum(0, send[0], recv[0], count=2, dtype="fp32")
    with pytest.raises(TransportStateError):
        transport.group_end()
    assert transport.sequence == 0


def test_mismatched_collective_order_is_rejected() -> None:
    transport = MockTransport(world_size=2)
    send, recv = _rank_buffers(transport, 2)
    transport.group_start()
    transport.all_reduce_sum(0, send[0], recv[0], count=2, dtype="fp32")
    transport.broadcast(1, send[1], recv[1], count=2, dtype="fp32", root=0)
    with pytest.raises(TransportStateError):
        transport.group_end()


def test_invalid_request_fields_are_rejected_before_issue() -> None:
    transport = MockTransport(world_size=2)
    send, recv = _rank_buffers(transport, 2)
    transport.group_start()
    with pytest.raises(TransportStateError):
        transport.all_reduce_sum(2, send[0], recv[0], count=2, dtype="fp32")
    with pytest.raises(TransportStateError):
        transport.all_reduce_sum(0, send[0], recv[0], count=2, dtype="int8")
    with pytest.raises(TransportStateError):
        transport.broadcast(0, send[0], recv[0], count=2, dtype="fp32", root=5)
    with pytest.raises(TransportStateError):
        transport.broadcast(0, send[0], recv[0], count=2, dtype="fp32", root=0)
        transport.all_reduce_sum(0, send[0], recv[0], count=-1, dtype="fp32")


def test_injected_group_failure_poisons_transport() -> None:
    transport = MockTransport(world_size=2, fail_at_group_end=0)
    send, recv = _rank_buffers(transport, 2)
    transport.group_start()
    for rank in range(2):
        transport.all_reduce_sum(rank, send[rank], recv[rank], count=2, dtype="fp32")
    with pytest.raises(CommunicatorAbortedError):
        transport.group_end()
    assert transport.poisoned
    with pytest.raises(CommunicatorAbortedError):
        transport.group_start()


def test_rank_failure_during_sync_poisons_transport() -> None:
    transport = MockTransport(world_size=2, fail_rank=1)
    with pytest.raises(CommunicatorAbortedError):
        transport.sync()
    assert transport.poisoned


def test_closed_transport_rejects_work() -> None:
    transport = MockTransport(world_size=2)
    transport.close()
    with pytest.raises(TransportStateError):
        transport.group_start()


def test_request_validate_rejects_bad_root_and_count() -> None:
    request = CollectiveRequest(
        kind=CollectiveKind.BROADCAST,
        rank=0,
        count=1,
        dtype="fp32",
        send_ptr=1,
        recv_ptr=2,
        root=None,
    )
    with pytest.raises(TransportStateError):
        request.validate(world_size=2, sequence=0)
    bad = CollectiveRequest(kind=CollectiveKind.ALL_REDUCE_SUM, rank=0, count=1, dtype="fp32", send_ptr=1, recv_ptr=2, root=1)
    with pytest.raises(TransportStateError):
        bad.validate(world_size=2, sequence=0)


def test_enqueue_recorder_reports_skew() -> None:
    recorder = EnqueueRecorder()
    recorder.record(0, start=1.000, end=1.010)
    recorder.record(1, start=1.004, end=1.030)
    assert recorder.skew_s() == pytest.approx(0.004)
    assert recorder.max_enqueue_s() == pytest.approx(0.026)
    payload = recorder.to_dict()
    assert payload["enqueue_count"] == 2


def test_mock_memory_rejects_unknown_pointer() -> None:
    memory = MockMemory()
    with pytest.raises(TransportStateError):
        memory.read(0, 0xDEAD, dtype="fp32", count=1)


def test_mock_repeated_groups_reuse_the_same_buffers_without_state_leak() -> None:
    """Many groups over one buffer pair must not accumulate or drift."""

    transport = MockTransport(world_size=2)
    memory = transport.memory
    send = [memory.alloc(rank, 16) for rank in range(2)]
    recv = [memory.alloc(rank, 16) for rank in range(2)]
    for group in range(50):
        for rank in range(2):
            memory.fill(rank, send[rank], float(rank + 1), dtype="fp32", count=4)
        transport.group_start()
        for rank in range(2):
            transport.all_reduce_sum(rank, send[rank], recv[rank], count=4, dtype="fp32")
        transport.group_end()
        transport.sync()
        for rank in range(2):
            actual = memory.read(rank, recv[rank], dtype="fp32", count=4)
            assert np.array_equal(actual, np.full(4, 3.0, dtype=np.float32)), f"group {group} rank {rank}"
    assert transport.groups_committed == 50
    assert transport.sequence == 50


def test_mock_cleanup_is_idempotent_and_rejects_later_work() -> None:
    """Closing twice is safe and no later group is accepted."""

    transport = MockTransport(world_size=2)
    memory = transport.memory
    send = [memory.alloc(rank, 16) for rank in range(2)]
    recv = [memory.alloc(rank, 16) for rank in range(2)]
    transport.close()
    transport.close()
    with pytest.raises(TransportStateError):
        transport.group_start()
    with pytest.raises(TransportStateError):
        transport.all_reduce_sum(0, send[0], recv[0], count=4, dtype="fp32")
    with pytest.raises(TransportStateError):
        transport.sync()
    # A context manager also closes cleanly when the body never ran a group.
    with MockTransport(world_size=2) as context_transport:
        assert context_transport.world_size == 2
    with pytest.raises(TransportStateError):
        context_transport.group_start()


def test_mock_abort_mid_group_leaves_no_open_group() -> None:
    """A poisoned group must not leave the transport stuck inside group_start."""

    transport = MockTransport(world_size=2, fail_at_group_end=0)
    memory = transport.memory
    send = [memory.alloc(rank, 16) for rank in range(2)]
    recv = [memory.alloc(rank, 16) for rank in range(2)]
    transport.group_start()
    for rank in range(2):
        transport.all_reduce_sum(rank, send[rank], recv[rank], count=4, dtype="fp32")
    with pytest.raises(CommunicatorAbortedError):
        transport.group_end()
    assert transport.poisoned is True
    # The failed group closed itself, so the next attempt is rejected for being
    # poisoned rather than for leaving a group open.
    with pytest.raises(TransportError):
        transport.group_start()
    with pytest.raises(TransportError):
        transport.sync()
    with pytest.raises(TransportError):
        transport.check_errors()


def test_mock_group_keeps_each_rank_request_on_its_own_buffers() -> None:
    """A group must apply each rank's request to that rank's pointers only."""

    transport = MockTransport(world_size=3)
    memory = transport.memory
    send = [memory.alloc(rank, 12) for rank in range(3)]
    recv = [memory.alloc(rank, 12) for rank in range(3)]
    for rank in range(3):
        memory.fill(rank, send[rank], float(rank + 1), dtype="fp32", count=3)
    transport.group_start()
    for rank in range(3):
        transport.all_reduce_sum(rank, send[rank], recv[rank], count=3, dtype="fp32")
    transport.group_end()
    transport.sync()
    for rank in range(3):
        # Sum of 1 + 2 + 3 on every rank, and the send buffers are untouched.
        assert np.array_equal(memory.read(rank, recv[rank], dtype="fp32", count=3), np.full(3, 6.0))
        assert np.array_equal(memory.read(rank, send[rank], dtype="fp32", count=3), np.full(3, float(rank + 1)))


def test_enqueue_recorder_tracks_repeated_groups_separately() -> None:
    """Recorder state is per group, not cumulative across groups."""

    first = EnqueueRecorder()
    first.record(0, start=0.0, end=0.001)
    first.record(1, start=0.002, end=0.004)
    second = EnqueueRecorder()
    second.record(0, start=10.0, end=10.0005)
    second.record(1, start=10.001, end=10.002)
    assert first.skew_s() == pytest.approx(0.002)
    assert second.skew_s() == pytest.approx(0.001)
    assert first.to_dict()["enqueue_count"] == 2
    assert second.max_enqueue_s() == pytest.approx(0.001)
