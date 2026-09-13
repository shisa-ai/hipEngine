"""CPU mock collective transport for plan/group-protocol tests.

The mock implements the same enqueue discipline as :class:`RcclTransport`
(``group_start`` → per-rank requests → ``group_end`` → ``sync``) over
host-side byte buffers, so protocol violations, atomicity, and failure
injection are testable without ROCm. It is not a performance model: it exists
to make the control/ownership contract executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from hipengine.distributed.transport import (
    CollectiveKind,
    CollectiveRequest,
    CommunicatorAbortedError,
    TimeoutError_,
    TransportStateError,
)

_MOCK_DTYPES = {"fp32": np.float32, "fp16": np.float16}


class MockMemory:
    """Per-rank simulated device memory keyed by fake pointer."""

    def __init__(self) -> None:
        self._buffers: dict[tuple[int, int], bytearray] = {}
        self._next_ptr = 0x10000

    def alloc(self, rank: int, nbytes: int) -> int:
        if int(nbytes) < 0:
            raise ValueError("nbytes must be non-negative")
        ptr = self._next_ptr
        self._next_ptr += max(1, int(nbytes)) + 16
        self._buffers[(int(rank), ptr)] = bytearray(int(nbytes))
        return ptr

    def free(self, rank: int, ptr: int) -> None:
        self._buffers.pop((int(rank), int(ptr)), None)

    def buffer(self, rank: int, ptr: int) -> bytearray:
        try:
            return self._buffers[(int(rank), int(ptr))]
        except KeyError as error:
            raise TransportStateError(f"no simulated buffer for rank {rank} ptr {ptr:#x}") from error

    def write(self, rank: int, ptr: int, array: np.ndarray) -> None:
        data = np.ascontiguousarray(array)
        self.buffer(rank, ptr)[: data.nbytes] = data.tobytes()

    def read(self, rank: int, ptr: int, *, dtype: str, count: int) -> np.ndarray:
        np_dtype = _MOCK_DTYPES[dtype]
        raw = bytes(self.buffer(rank, ptr)[: int(count) * np.dtype(np_dtype).itemsize])
        return np.frombuffer(raw, dtype=np_dtype, count=int(count)).copy()

    def fill(self, rank: int, ptr: int, value: float, *, dtype: str, count: int) -> None:
        np_dtype = _MOCK_DTYPES[dtype]
        array = np.full(int(count), value, dtype=np_dtype)
        self.write(rank, ptr, array)


@dataclass
class MockTransport:
    """Deterministic CPU collective transport with injectable failures."""

    world_size: int
    memory: MockMemory = field(default_factory=MockMemory)
    fail_at_group_end: int | None = None
    fail_rank: int | None = None

    def __post_init__(self) -> None:
        if int(self.world_size) < 1:
            raise TransportStateError("mock transport requires at least one rank")
        self._sequence = 0
        self._poisoned = False
        self._in_group = False
        self._requests: dict[int, list[CollectiveRequest]] = {}
        self._closed = False
        self.groups_committed = 0
        self.bytes_communicated = 0

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def group_start(self) -> None:
        self._require_live()
        if self._in_group:
            raise TransportStateError("collective group already open")
        self._in_group = True
        self._requests = {rank: [] for rank in range(self.world_size)}

    def all_reduce_sum(self, rank: int, send_ptr: int, recv_ptr: int, *, count: int, dtype: str) -> None:
        self._enqueue(
            CollectiveRequest(
                kind=CollectiveKind.ALL_REDUCE_SUM,
                rank=int(rank),
                count=int(count),
                dtype=str(dtype),
                send_ptr=int(send_ptr),
                recv_ptr=int(recv_ptr),
            )
        )

    def broadcast(
        self,
        rank: int,
        send_ptr: int,
        recv_ptr: int,
        *,
        count: int,
        dtype: str,
        root: int,
    ) -> None:
        self._enqueue(
            CollectiveRequest(
                kind=CollectiveKind.BROADCAST,
                rank=int(rank),
                count=int(count),
                dtype=str(dtype),
                send_ptr=int(send_ptr),
                recv_ptr=int(recv_ptr),
                root=int(root),
            )
        )

    def group_end(self) -> None:
        self._require_live()
        if not self._in_group:
            raise TransportStateError("no collective group is open")
        self._validate_group()
        if self.fail_at_group_end is not None and self._sequence == int(self.fail_at_group_end):
            self._poisoned = True
            self._in_group = False
            raise CommunicatorAbortedError("injected group failure")
        self._apply_group()
        self._in_group = False
        self._sequence += 1
        self.groups_committed += 1

    def sync(self, *, timeout_s: float | None = None) -> None:
        self._require_live()
        if self.fail_rank is not None:
            self._poisoned = True
            raise CommunicatorAbortedError(f"injected rank {self.fail_rank} failure")
        if timeout_s is not None and float(timeout_s) <= 0:
            raise TimeoutError_("mock transport deadline already expired")

    def check_errors(self) -> None:
        self._require_live()

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "MockTransport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _enqueue(self, request: CollectiveRequest) -> None:
        self._require_live()
        if not self._in_group:
            raise TransportStateError("collectives must be enqueued inside group_start/group_end")
        request.validate(world_size=self.world_size, sequence=self._sequence)
        if request.dtype not in _MOCK_DTYPES:
            raise TransportStateError(f"mock transport does not implement dtype {request.dtype!r}")
        self._requests[request.rank].append(request)

    def _validate_group(self) -> None:
        counts = {rank: len(requests) for rank, requests in self._requests.items()}
        if len(set(counts.values())) != 1 or not counts:
            raise TransportStateError(f"every rank must enqueue the same number of collectives; got {counts}")
        if next(iter(counts.values()), 0) == 0:
            raise TransportStateError("collective group is empty")
        reference = [(r.kind, r.count, r.dtype, r.root) for r in self._requests[0]]
        for rank, requests in self._requests.items():
            if [(r.kind, r.count, r.dtype, r.root) for r in requests] != reference:
                raise TransportStateError(f"rank {rank} collective order does not match rank 0")

    def _apply_group(self) -> None:
        for index in range(len(self._requests[0])):
            request = self._requests[0][index]
            if request.kind is CollectiveKind.ALL_REDUCE_SUM:
                self._apply_all_reduce(index, request)
            elif request.kind is CollectiveKind.BROADCAST:
                self._apply_broadcast(index, request)

    def _apply_all_reduce(self, index: int, reference: CollectiveRequest) -> None:
        total: np.ndarray | None = None
        for rank in range(self.world_size):
            request = self._requests[rank][index]
            values = self.memory.read(rank, request.send_ptr, dtype=request.dtype, count=request.count)
            total = values if total is None else total + values
        assert total is not None
        for rank in range(self.world_size):
            request = self._requests[rank][index]
            self.memory.write(rank, request.recv_ptr, total.astype(_MOCK_DTYPES[request.dtype]))
            self.bytes_communicated += int(request.count) * np.dtype(_MOCK_DTYPES[request.dtype]).itemsize

    def _apply_broadcast(self, index: int, reference: CollectiveRequest) -> None:
        root = int(reference.root or 0)
        source = self.memory.read(root, self._requests[root][index].send_ptr, dtype=reference.dtype, count=reference.count)
        for rank in range(self.world_size):
            request = self._requests[rank][index]
            self.memory.write(rank, request.recv_ptr, source)
            self.bytes_communicated += int(request.count) * np.dtype(_MOCK_DTYPES[request.dtype]).itemsize

    def _require_live(self) -> None:
        if self._closed:
            raise TransportStateError("transport is closed")
        if self._poisoned:
            raise CommunicatorAbortedError("mock communicator group is poisoned")


def make_mock_plan(devices: Iterable[int], *, hidden_size: int, **kwargs):
    """Convenience: build a plan whose devices are fake HIP indices."""

    from hipengine.distributed.plan import DistributedPlan

    return DistributedPlan.resolve(list(devices), hidden_size=hidden_size, algorithm="mock", **kwargs)
