"""CPU-only tests for the real RCCL transport wrapper.

``RcclTransport`` is the code that runs in production, but until now its only
coverage was ``tests/test_gpu_distributed_transport.py``, which needs two GPUs
and therefore exercises the happy path on CI hardware only. The failure paths -
a rank issuing a different operation sequence, and a communicator that fails to
initialize - were covered *only* by ``MockTransport``, which is a different
implementation. That is how a transport that issues mismatched operations before
validating them, and leaks resources on partial initialization, passed review.

These tests drive the real wrapper through a fake RCCL binding and a fake HIP
runtime, so no GPU and no ``librccl.so`` are needed. The fake records every call,
which is what lets a test assert *ordering* - that a mismatch is rejected before
the operation reaches RCCL - rather than only that an exception was raised.
"""

from __future__ import annotations

import ctypes

import pytest

from hipengine.core.device import Device
from hipengine.core.runtime import MemcpyKind
from hipengine.distributed.rccl import (
    NCCL_SUCCESS,
    RcclTransport,
    TransportStateError,
    TransportUnavailableError,
)
from hipengine.distributed.transport import CommunicatorAbortedError

WORLD = 2


class FakeRcclBinding:
    """Records the RCCL calls the transport makes, in order."""

    def __init__(self, *, fail_init_ranks: frozenset[int] = frozenset()) -> None:
        self.calls: list[tuple] = []
        self.fail_init_ranks = set(fail_init_ranks)
        self.comm_handles = [0x1000 + rank for rank in range(WORLD)]
        self.aborted: list[int] = []
        self.destroyed: list[int] = []
        self.async_error: dict[int, int] = {}
        self._next_handle = 0x2000

    # -- lifecycle ----------------------------------------------------------

    def version(self) -> int:
        return 22707

    def unique_id(self) -> ctypes.Structure:
        self.calls.append(("unique_id",))
        return ctypes.c_void_p(0xAA)

    def comm_init_rank(self, nranks: int, unique_id: object, rank: int) -> int:
        self.calls.append(("comm_init_rank", int(rank)))
        if int(rank) in self.fail_init_ranks:
            raise TransportUnavailableError(f"injected init failure on rank {rank}")
        handle = self.comm_handles[int(rank)]
        self.async_error[handle] = NCCL_SUCCESS
        return handle

    def comm_async_error(self, comm: int) -> int:
        self.calls.append(("comm_async_error", int(comm)))
        return int(self.async_error.get(int(comm), NCCL_SUCCESS))

    def comm_destroy(self, comm: int) -> None:
        self.calls.append(("comm_destroy", int(comm)))
        self.destroyed.append(int(comm))

    def comm_abort(self, comm: int) -> None:
        self.calls.append(("comm_abort", int(comm)))
        self.aborted.append(int(comm))

    # -- collectives --------------------------------------------------------

    def group_start(self) -> None:
        self.calls.append(("group_start",))

    def group_end(self) -> None:
        self.calls.append(("group_end",))

    def all_reduce(self, *, send: int, recv: int, count: int, dtype: str, comm: int, stream: int) -> None:
        self.calls.append(("all_reduce", int(comm), int(count), str(dtype)))

    def broadcast(
        self, *, send: int, recv: int, count: int, dtype: str, root: int, comm: int, stream: int
    ) -> None:
        self.calls.append(("broadcast", int(comm), int(count), str(dtype), int(root)))

    # -- helpers for assertions --------------------------------------------

    def operations(self) -> list[tuple]:
        return [call for call in self.calls if call[0] in ("all_reduce", "broadcast")]


class FakeHipRuntime:
    """Minimal device runtime: device selection plus stream bookkeeping."""

    def __init__(self) -> None:
        self._current = 0
        self.streams: list[int] = []
        self.destroyed_streams: list[int] = []
        self.queried: list[int] = []
        self._next_stream = 0x5000

    def set_device(self, device: int) -> None:
        self._current = int(device)

    def get_device(self) -> int:
        return self._current

    def stream_create(self, *, nonblocking: bool = True) -> int:
        stream = self._next_stream
        self._next_stream += 1
        self.streams.append(stream)
        return stream

    def stream_destroy(self, stream: int) -> None:
        self.destroyed_streams.append(int(stream))
        if int(stream) in self.streams:
            self.streams.remove(int(stream))

    def stream_query(self, stream: int) -> bool:
        self.queried.append(int(stream))
        return True

    def memcpy(self, *args: object) -> None:  # pragma: no cover - not used here
        raise NotImplementedError

    def memcpy_async(self, *args: object) -> None:  # pragma: no cover - not used here
        raise NotImplementedError


def _devices() -> tuple[Device, ...]:
    return tuple(Device("hip", index) for index in range(WORLD))


def _transport(
    binding: FakeRcclBinding | None = None,
    runtime: FakeHipRuntime | None = None,
) -> tuple[RcclTransport, FakeRcclBinding, FakeHipRuntime]:
    binding = binding or FakeRcclBinding()
    runtime = runtime or FakeHipRuntime()
    transport = RcclTransport(_devices(), binding=binding, runtime=runtime, init_timeout_s=5.0)
    return transport, binding, runtime


def _allocate(transport: RcclTransport) -> tuple[list[int], list[int]]:
    send = [0x10000 + rank * 0x100 for rank in range(WORLD)]
    recv = [0x20000 + rank * 0x100 for rank in range(WORLD)]
    return send, recv


# -- happy path: proves the fakes are faithful -------------------------------


def test_matching_groups_issue_every_rank_and_advance_sequence() -> None:
    transport, binding, _runtime = _transport()
    send, recv = _allocate(transport)

    transport.group_start()
    for rank in range(WORLD):
        transport.all_reduce_sum(rank, send[rank], recv[rank], count=1280, dtype="bf16")
    transport.group_end()
    transport.sync()

    assert transport.sequence == 1
    assert transport.poisoned is False
    # Both ranks' operations reached RCCL, in rank order, inside one group.
    assert binding.operations() == [
        ("all_reduce", binding.comm_handles[0], 1280, "bf16"),
        ("all_reduce", binding.comm_handles[1], 1280, "bf16"),
    ]
    assert binding.calls.count(("group_start",)) == 1
    assert binding.calls.count(("group_end",)) == 1
    transport.close()
    assert binding.destroyed == list(binding.comm_handles)


# -- defect 1: mismatched operations must not be issued ----------------------


def test_mismatched_operation_is_rejected_before_it_reaches_rccl() -> None:
    """A rank issuing a different op must be refused, not queued.

    The mock validates before applying; the real wrapper used to enqueue into
    RCCL first and only compare shapes at ``group_end``, by which point the
    mismatched operation was already in the communicator's queue - a hang or a
    corrupt result instead of an error.
    """

    transport, binding, _runtime = _transport()
    send, recv = _allocate(transport)

    transport.group_start()
    transport.all_reduce_sum(0, send[0], recv[0], count=1280, dtype="bf16")
    with pytest.raises(TransportStateError):
        transport.all_reduce_sum(1, send[1], recv[1], count=2560, dtype="bf16")

    issued = binding.operations()
    assert len(issued) == 1, f"the mismatching operation reached RCCL: {issued}"
    assert issued[0][1] == binding.comm_handles[0]
    assert transport.poisoned is True


def test_mismatched_kind_is_rejected_before_it_reaches_rccl() -> None:
    transport, binding, _runtime = _transport()
    send, recv = _allocate(transport)

    transport.group_start()
    transport.all_reduce_sum(0, send[0], recv[0], count=1280, dtype="bf16")
    with pytest.raises(TransportStateError):
        transport.broadcast(1, send[1], recv[1], count=1280, dtype="bf16", root=0)

    assert len(binding.operations()) == 1
    assert transport.poisoned is True


def test_poisoned_transport_aborts_instead_of_leaving_queued_work() -> None:
    """Failure must be explicit: abort the communicators, do not hang.

    A poisoned group with work already queued would block a later ``sync``
    forever, which the plan forbids ("one-rank failure must not hang the
    process").
    """

    transport, binding, _runtime = _transport()
    send, recv = _allocate(transport)
    transport.group_start()
    transport.all_reduce_sum(0, send[0], recv[0], count=1280, dtype="bf16")
    with pytest.raises(TransportStateError):
        transport.all_reduce_sum(1, send[1], recv[1], count=2560, dtype="bf16")

    transport.close()
    assert sorted(binding.aborted) == sorted(binding.comm_handles)
    assert binding.destroyed == []


def test_group_that_never_registers_every_rank_fails_without_hanging() -> None:
    transport, binding, _runtime = _transport()
    send, recv = _allocate(transport)
    transport.group_start()
    transport.all_reduce_sum(0, send[0], recv[0], count=1280, dtype="bf16")
    transport.group_end()  # rank 1 never registered

    with pytest.raises(TransportStateError):
        transport.sync(timeout_s=0.05)
    assert transport.poisoned is True


# -- defect 2: partial initialization must not leak --------------------------


def test_partial_init_destroys_created_communicators_and_streams() -> None:
    """A failed init must leave nothing behind.

    ``__init__`` creates the streams and then the communicators. When a
    communicator failed to initialize the constructor raised without destroying
    the communicators that did come up, or the streams - and because the
    constructor never returned, no caller could call ``close()``.
    """

    binding = FakeRcclBinding(fail_init_ranks=frozenset({1}))
    runtime = FakeHipRuntime()
    with pytest.raises(TransportUnavailableError):
        RcclTransport(_devices(), binding=binding, runtime=runtime, init_timeout_s=5.0)

    assert sorted(binding.destroyed) == [binding.comm_handles[0]]
    assert binding.aborted == []
    assert sorted(runtime.destroyed_streams) == sorted(runtime.streams) or runtime.streams == []
    assert runtime.streams == [], f"streams leaked: {runtime.streams}"


def test_stream_creation_failure_leaves_no_communicators() -> None:
    binding = FakeRcclBinding()
    runtime = FakeHipRuntime()

    def failing_create(*, nonblocking: bool = True) -> int:
        raise RuntimeError("injected stream failure")

    runtime.stream_create = failing_create  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        RcclTransport(_devices(), binding=binding, runtime=runtime, init_timeout_s=5.0)
    assert binding.calls == []


# -- coverage declaration ----------------------------------------------------

#: Failure modes the transport must handle, and where each is exercised on the
#: *real* wrapper. ``MockTransport`` covers all of them too, but the mock is a
#: different implementation: a mode that only the mock covers is unverified for
#: production. Adding a mode here without a real-wrapper test is a deliberate
#: declaration, not an oversight.
REAL_WRAPPER_COVERAGE = {
    "group_shape_mismatch": "test_mismatched_operation_is_rejected_before_it_reaches_rccl",
    "group_shape_mismatch_kind": "test_mismatched_kind_is_rejected_before_it_reaches_rccl",
    "poisoned_aborts_queued_work": "test_poisoned_transport_aborts_instead_of_leaving_queued_work",
    "incomplete_group": "test_group_that_never_registers_every_rank_fails_without_hanging",
    "partial_init_teardown": "test_partial_init_destroys_created_communicators_and_streams",
    "stream_init_failure": "test_stream_creation_failure_leaves_no_communicators",
    "happy_path": "test_matching_groups_issue_every_rank_and_advance_sequence",
}

#: Modes intentionally left to the mock, with the reason.
MOCK_ONLY = {
    "async_communicator_error": "needs a live communicator that reports an async error",
    "completion_timeout": "needs a device that does not complete; covered by the GPU suite",
}


def test_every_declared_failure_mode_has_a_real_wrapper_test() -> None:
    """The declaration above must name tests that exist in this module."""

    import sys

    module = sys.modules[__name__]
    for mode, test_name in REAL_WRAPPER_COVERAGE.items():
        assert hasattr(module, test_name), f"{mode} points at missing test {test_name}"

    # Modes that need hardware are covered by the guarded GPU suite; keep the
    # list honest by requiring each to name a reason.
    for mode, reason in MOCK_ONLY.items():
        assert reason.strip(), f"{mode} is mock-only without a reason"


def test_mock_and_real_transport_agree_on_shape_mismatch() -> None:
    """Both transports reject a mismatch, at the point each one can.

    The mock holds every rank's requests in one process, so it validates the
    whole group at ``group_end`` before applying anything. The real transport
    cannot: an operation handed to RCCL is queued immediately, so it validates
    each operation against its peers before issuing it. The *outcome* is the same
    exception type; the *timing* differs, and this test pins both so a future
    change cannot quietly move the real transport back to issue-then-validate.
    """

    from hipengine.distributed.mock import MockTransport
    from hipengine.distributed.transport import TransportStateError as StateError

    # Mock: the mismatch surfaces at group_end, before anything is applied.
    mock = MockTransport(world_size=2)
    send = [mock.memory.alloc(rank, 16) for rank in range(2)]
    recv = [mock.memory.alloc(rank, 16) for rank in range(2)]
    mock.group_start()
    mock.all_reduce_sum(0, send[0], recv[0], count=4, dtype="fp32")
    mock.all_reduce_sum(1, send[1], recv[1], count=8, dtype="fp32")  # accepted, then validated
    with pytest.raises(StateError):
        mock.group_end()
    assert mock.groups_committed == 0

    # Real wrapper: the mismatch surfaces on the offending enqueue, and the
    # mismatching operation never reaches RCCL.
    transport, binding, _runtime = _transport()
    send_ptrs, recv_ptrs = _allocate(transport)
    transport.group_start()
    transport.all_reduce_sum(0, send_ptrs[0], recv_ptrs[0], count=4, dtype="fp32")
    with pytest.raises(StateError):
        transport.all_reduce_sum(1, send_ptrs[1], recv_ptrs[1], count=8, dtype="fp32")
    assert len(binding.operations()) == 1
    assert transport.poisoned is True
