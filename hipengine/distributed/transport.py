"""Collective transport interface shared by the RCCL and mock implementations.

The transport owns *only* collective communication: rank-group broadcast and
all-reduce-sum, completion/error reporting, and communicator lifetime. It does
not know about models, layers, schedulers, or KV state.

Enqueue discipline (docs/QWEN38-27B-GFX1100-TP2.md "Runtime and communication"):
  * all ranks' collectives for one segment are enqueued inside one group;
  * ``group_end`` closes the group and establishes *enqueue*, not device
    completion; callers must :meth:`sync` before consuming results;
  * dependent kernels are only enqueued after the group has closed on every
    communicator;
  * requests are validated (kind, count, dtype, root, sequence) before they are
    issued, and a poisoned communicator fails closed instead of silently
    recovering from half-committed state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hipengine.distributed.plan import SUPPORTED_COMM_DTYPES


class TransportError(RuntimeError):
    """Base class for collective transport failures."""


class TransportUnavailableError(TransportError):
    """The requested transport library or device support is not present."""


class CommunicatorAbortedError(TransportError):
    """The communicator group was aborted or poisoned; it is not reusable."""


class TransportStateError(TransportError):
    """A request violated the group/enqueue protocol."""


class TimeoutError_(TransportError):
    """Completion did not arrive before the deadline (device may be hung)."""


def require_rows_value(value: object, *, capacity: int, name: str = "rows") -> int:
    """Validate a row count is a real positive integer within ``capacity``.

    ``int(1.5)`` and ``int(True)`` both silently collapse to 1, so a caller
    passing a float or bool would get a single-row transfer while believing it
    asked for a batch. Reject anything that is not an ``int`` (bools are not
    integers here) before any allocation or launch. Shared by the shard
    executor and both staged transports so every boundary agrees.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}"
        )
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value!r}")
    if value > int(capacity):
        raise ValueError(f"{name} {value} exceeds capacity {capacity}")
    return value


class CollectiveKind(enum.Enum):
    BROADCAST = "broadcast"
    ALL_REDUCE_SUM = "all_reduce_sum"


@dataclass(frozen=True)
class CollectiveRequest:
    """One validated collective request for one rank."""

    kind: CollectiveKind
    rank: int
    count: int
    dtype: str
    send_ptr: int
    recv_ptr: int
    root: int | None = None

    def validate(self, *, world_size: int, sequence: int) -> None:
        if self.dtype not in SUPPORTED_COMM_DTYPES:
            raise TransportStateError(f"unsupported collective dtype {self.dtype!r}")
        if int(self.count) < 0:
            raise TransportStateError("collective count must be non-negative")
        if not 0 <= int(self.rank) < int(world_size):
            raise TransportStateError(f"rank {self.rank} is outside world size {world_size}")
        if self.kind is CollectiveKind.BROADCAST:
            if self.root is None:
                raise TransportStateError("broadcast requires a root rank")
            if not 0 <= int(self.root) < int(world_size):
                raise TransportStateError(f"broadcast root {self.root} is outside world size {world_size}")
        elif self.root is not None:
            raise TransportStateError(f"{self.kind.value} must not carry a root")
        if int(sequence) < 0:
            raise TransportStateError("collective sequence must be non-negative")


@runtime_checkable
class CollectiveTransport(Protocol):
    """Structural interface consumed by the distributed runner adapter."""

    @property
    def world_size(self) -> int: ...

    @property
    def sequence(self) -> int: ...

    @property
    def poisoned(self) -> bool: ...

    def group_start(self) -> None: ...

    def group_end(self) -> None: ...

    def all_reduce_sum(self, rank: int, send_ptr: int, recv_ptr: int, *, count: int, dtype: str) -> None: ...

    def broadcast(
        self,
        rank: int,
        send_ptr: int,
        recv_ptr: int,
        *,
        count: int,
        dtype: str,
        root: int,
    ) -> None: ...

    def sync(self, *, timeout_s: float | None = None) -> None: ...

    def close(self) -> None: ...


class EnqueueRecorder:
    """Tracks host-side enqueue order/skew for one group of ranks.

    Rank enqueue skew is first-class evidence in Packet 0/4: sequential replay of
    two ranks can serialize compute even when the collectives themselves are
    asynchronous, so the benchmark records when each rank's enqueue started and
    finished relative to the group.
    """

    def __init__(self) -> None:
        self.entries: list[tuple[int, float, float]] = []

    def record(self, rank: int, *, start: float, end: float) -> None:
        self.entries.append((int(rank), float(start), float(end)))

    def skew_s(self) -> float:
        """Return the spread between the earliest and latest enqueue start."""

        if not self.entries:
            return 0.0
        starts = [start for _, start, _ in self.entries]
        return max(starts) - min(starts)

    def max_enqueue_s(self) -> float:
        return max((end - start for _, start, end in self.entries), default=0.0)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "enqueue_count": len(self.entries),
            "skew_s": self.skew_s(),
            "max_enqueue_s": self.max_enqueue_s(),
        }
