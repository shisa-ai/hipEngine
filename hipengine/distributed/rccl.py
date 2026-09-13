"""ctypes RCCL collective transport (no torch, no ``torch.distributed``).

The binding is configured against the installed RCCL headers' enum values and
verified symbol names; nothing about opaque struct sizes is guessed beyond the
header-declared ``ncclUniqueId`` of 128 bytes. Importing this module does not
load ``librccl.so``; the library is opened by :class:`RcclBinding`.

One process owns every rank (single-process multi-GPU): communicators are
initialized concurrently from one thread per rank, then a single thread enqueues
grouped collectives for all ranks. Streams are nonblocking and per-rank, so the
caller can overlap independent work later without changing the transport.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Final, Sequence

from hipengine.core.device import Device, scoped_current_device
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.distributed.transport import (
    CollectiveKind,
    CollectiveRequest,
    CommunicatorAbortedError,
    EnqueueRecorder,
    TimeoutError_,
    TransportError,
    TransportStateError,
    TransportUnavailableError,
)

DEFAULT_RCCL_LIBRARY: Final[str] = "librccl.so"
NCCL_UNIQUE_ID_BYTES: Final[int] = 128
NCCL_SUCCESS: Final[int] = 0
NCCL_IN_PROGRESS: Final[int] = 7
NCCL_VERSION_MAJOR: Final[int] = 2

#: ``ncclDataType_t`` values from the installed RCCL header.
NCCL_DATA_TYPES: Final[dict[str, int]] = {
    "int8": 0,
    "uint8": 1,
    "int32": 2,
    "uint32": 3,
    "int64": 4,
    "uint64": 5,
    "fp16": 6,
    "fp32": 7,
    "fp64": 8,
    "bf16": 9,
}

NCCL_SUM: Final[int] = 0


class RcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


class RcclError(TransportError):
    """RCCL returned a non-success status code."""

    def __init__(self, code: int, operation: str):
        self.code = int(code)
        self.operation = operation
        super().__init__(f"RCCL {operation} failed with status {self.code}")


@dataclass
class RcclBinding:
    """Typed ``librccl.so`` entry points."""

    library: ctypes.CDLL

    @classmethod
    def load(cls, path: str = DEFAULT_RCCL_LIBRARY) -> "RcclBinding":
        try:
            library = ctypes.CDLL(path)
        except OSError as error:
            raise TransportUnavailableError(f"could not load RCCL library {path!r}: {error}") from error
        binding = cls(library)
        binding._configure()
        return binding

    def version(self) -> int:
        value = ctypes.c_int()
        self._check(self.library.ncclGetVersion(ctypes.byref(value)), "ncclGetVersion")
        return int(value.value)

    def unique_id(self) -> RcclUniqueId:
        unique_id = RcclUniqueId()
        self._check(self.library.ncclGetUniqueId(ctypes.byref(unique_id)), "ncclGetUniqueId")
        return unique_id

    def comm_init_rank(self, nranks: int, unique_id: RcclUniqueId, rank: int) -> int:
        comm = ctypes.c_void_p()
        self._check(
            self.library.ncclCommInitRank(ctypes.byref(comm), ctypes.c_int(int(nranks)), unique_id, ctypes.c_int(int(rank))),
            f"ncclCommInitRank(rank={rank})",
        )
        return 0 if comm.value is None else int(comm.value)

    def comm_async_error(self, comm: int) -> int:
        status = ctypes.c_int(NCCL_SUCCESS)
        self._check(
            self.library.ncclCommGetAsyncError(ctypes.c_void_p(comm), ctypes.byref(status)),
            "ncclCommGetAsyncError",
        )
        return int(status.value)

    def comm_destroy(self, comm: int) -> None:
        self._check(self.library.ncclCommDestroy(ctypes.c_void_p(comm)), "ncclCommDestroy")

    def comm_abort(self, comm: int) -> None:
        self._check(self.library.ncclCommAbort(ctypes.c_void_p(comm)), "ncclCommAbort")

    def group_start(self) -> None:
        self._check(self.library.ncclGroupStart(), "ncclGroupStart")

    def group_end(self) -> None:
        self._check(self.library.ncclGroupEnd(), "ncclGroupEnd")

    def all_reduce(self, *, send: int, recv: int, count: int, dtype: str, comm: int, stream: int) -> None:
        # RCCL signature is ncclAllReduce(sendbuff, recvbuff, ...).
        self._check(
            self.library.ncclAllReduce(
                ctypes.c_void_p(send),
                ctypes.c_void_p(recv),
                ctypes.c_size_t(int(count)),
                ctypes.c_int(NCCL_DATA_TYPES[dtype]),
                ctypes.c_int(NCCL_SUM),
                ctypes.c_void_p(comm),
                ctypes.c_void_p(stream),
            ),
            "ncclAllReduce",
        )

    def broadcast(self, *, send: int, recv: int, count: int, dtype: str, root: int, comm: int, stream: int) -> None:
        self._check(
            self.library.ncclBroadcast(
                ctypes.c_void_p(send),
                ctypes.c_void_p(recv),
                ctypes.c_size_t(int(count)),
                ctypes.c_int(NCCL_DATA_TYPES[dtype]),
                ctypes.c_int(int(root)),
                ctypes.c_void_p(comm),
                ctypes.c_void_p(stream),
            ),
            "ncclBroadcast",
        )

    def _check(self, code: int, operation: str) -> None:
        if int(code) != NCCL_SUCCESS:
            raise RcclError(int(code), operation)

    def _configure(self) -> None:
        signatures = {
            "ncclGetVersion": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "ncclGetUniqueId": ([ctypes.POINTER(RcclUniqueId)], ctypes.c_int),
            "ncclCommInitRank": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, RcclUniqueId, ctypes.c_int],
                ctypes.c_int,
            ),
            "ncclCommGetAsyncError": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "ncclCommDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "ncclCommAbort": ([ctypes.c_void_p], ctypes.c_int),
            "ncclGroupStart": ([], ctypes.c_int),
            "ncclGroupEnd": ([], ctypes.c_int),
            "ncclAllReduce": (
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
            "ncclBroadcast": (
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self.library, name, None)
            if function is None:
                raise TransportUnavailableError(f"RCCL library does not export required symbol {name}")
            function.argtypes = argtypes
            function.restype = restype


class RcclTransport:
    """Group-owned RCCL communicators for one ordered rank list.

    ``devices`` is the rank order. ``N == 1`` still creates a one-rank
    communicator so the code path is uniform in tests, but serving N=1 should use
    the plain TP1 runner instead (see :class:`hipengine.distributed.context.DistributedContext`).
    """

    def __init__(
        self,
        devices: Sequence[Device],
        *,
        binding: RcclBinding | None = None,
        runtime: HipRuntime | None = None,
        init_timeout_s: float = 120.0,
    ) -> None:
        self._devices = tuple(devices)
        if not self._devices:
            raise TransportStateError("RCCL transport requires at least one device")
        if len(set(self._devices)) != len(self._devices):
            raise TransportStateError("RCCL transport requires unique devices")
        if any(device.kind != "hip" for device in self._devices):
            raise TransportStateError("RCCL transport requires hip devices")
        self._binding = binding or RcclBinding.load()
        self._runtime = runtime or get_hip_runtime()
        self._poisoned = False
        self._closed = False
        self._sequence = 0
        self._group_lock = threading.Lock()
        #: sequence -> {rank: shape}. A group is validated as soon as every rank
        #: has registered its shape, which covers both the single-thread grouped
        #: enqueue (all ranks in one call) and one-thread-per-rank enqueue.
        self._group_shapes: dict[int, dict[int, tuple]] = {}
        self._local = threading.local()
        self._streams: list[int] = []
        self._comms: list[int] = []
        self._create_streams()
        self._init_communicators(timeout_s=init_timeout_s)

    # -- properties ---------------------------------------------------------

    @property
    def world_size(self) -> int:
        return len(self._devices)

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def devices(self) -> tuple[Device, ...]:
        return self._devices

    def stream(self, rank: int) -> int:
        self._require_rank(rank)
        return self._streams[int(rank)]

    def communicator(self, rank: int) -> int:
        self._require_rank(rank)
        return self._comms[int(rank)]

    # -- group protocol -----------------------------------------------------

    def group_start(self) -> None:
        self._require_live()
        if self._thread_group() is not None:
            raise TransportStateError("collective group already open on this thread")
        self._binding.group_start()
        self._local.group = {"requests": {}}

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
        group = self._thread_group()
        if group is None:
            raise TransportStateError("no collective group is open on this thread")
        requests: dict[int, list[CollectiveRequest]] = group["requests"]
        if not requests:
            raise TransportStateError("collective group is empty")
        self._binding.group_end()
        self._local.group = None
        with self._group_lock:
            sequence = self._sequence
            bucket = self._group_shapes.setdefault(sequence, {})
            for rank, rank_requests in requests.items():
                if rank in bucket:
                    self._poison(f"rank {rank} registered two groups at sequence {sequence}")
                    raise TransportStateError(f"rank {rank} registered two groups at sequence {sequence}")
                bucket[rank] = tuple((r.kind, r.count, r.dtype, r.root) for r in rank_requests)
            if len(bucket) == self.world_size:
                self._validate_group_shapes(bucket)
                self._group_shapes.pop(sequence, None)
                self._sequence += 1

    def _validate_group_shapes(self, bucket: dict[int, tuple]) -> None:
        shapes = set(bucket.values())
        if len(shapes) != 1:
            self._poison("collective group shape mismatch across ranks")
            raise TransportStateError(
                "every rank must enqueue the same collective sequence; got "
                + ", ".join(f"rank{rank}={shape}" for rank, shape in sorted(bucket.items()))
            )

    def sync(self, *, timeout_s: float | None = None, poll_interval_s: float = 0.0002) -> None:
        """Wait for device completion of every rank, bounded by ``timeout_s``.

        Polls ``hipStreamQuery`` and the communicator async-error state instead of
        blocking forever in ``hipStreamSynchronize``: a hung device must stay
        killable by a supervisor, so a deadline turns into an explicit poisoned
        communicator and a raised :class:`TimeoutError_`.
        """

        self._require_live()
        with self._group_lock:
            incomplete = {sequence: sorted(bucket) for sequence, bucket in self._group_shapes.items()}
        if incomplete:
            self._poison(f"incomplete collective group at sync: {incomplete}")
            raise TransportStateError(
                f"collective group did not register every rank before sync: {incomplete}"
            )
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        pending = set(range(self.world_size))
        while pending:
            for rank in sorted(pending):
                with scoped_current_device(self._runtime, self._devices[rank].index):
                    if self._runtime.stream_query(self._streams[rank]):
                        pending.discard(rank)
            if not pending:
                break
            self._check_async_errors()
            if deadline is not None and time.monotonic() >= deadline:
                self._poison("collective completion deadline exceeded")
                raise TimeoutError_(f"collective completion timed out after {timeout_s}s on ranks {sorted(pending)}")
            time.sleep(poll_interval_s)
        self._check_async_errors()

    def check_errors(self) -> None:
        """Raise if any communicator has reported an asynchronous error."""

        self._require_live()
        self._check_async_errors()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._group_lock:
            self._group_shapes.clear()
        self._local.group = None
        comms = list(self._comms)
        self._comms = []
        for rank, comm in enumerate(comms):
            try:
                with scoped_current_device(self._runtime, self._devices[rank].index):
                    if self._poisoned:
                        self._binding.comm_abort(comm)
                    else:
                        self._binding.comm_destroy(comm)
            except Exception:  # noqa: BLE001 - teardown must attempt every rank
                pass
        for rank, stream in enumerate(self._streams):
            try:
                with scoped_current_device(self._runtime, self._devices[rank].index):
                    self._runtime.stream_destroy(stream)
            except Exception:  # noqa: BLE001 - teardown must attempt every rank
                pass
        self._streams = []

    def __enter__(self) -> "RcclTransport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _create_streams(self) -> None:
        for device in self._devices:
            with scoped_current_device(self._runtime, device.index):
                self._streams.append(self._runtime.stream_create(nonblocking=True))

    def _init_communicators(self, *, timeout_s: float) -> None:
        unique_id = self._binding.unique_id()
        comms: list[int | None] = [None] * self.world_size
        errors: list[BaseException | None] = [None] * self.world_size
        barrier = threading.Barrier(self.world_size + 1, timeout=max(1.0, float(timeout_s)))

        def init_rank(rank: int) -> None:
            try:
                with scoped_current_device(self._runtime, self._devices[rank].index):
                    comms[rank] = self._binding.comm_init_rank(self.world_size, unique_id, rank)
            except BaseException as error:  # noqa: BLE001 - surfaced on the caller thread
                errors[rank] = error
            finally:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass

        threads = [threading.Thread(target=init_rank, args=(rank,), name=f"rccl-init-{rank}") for rank in range(self.world_size)]
        for thread in threads:
            thread.start()
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        for thread in threads:
            thread.join(timeout=max(1.0, float(timeout_s)))
        for rank, error in enumerate(errors):
            if error is not None:
                raise TransportUnavailableError(f"communicator init failed on rank {rank}: {error!r}") from error
        missing = [rank for rank, comm in enumerate(comms) if comm is None]
        if missing:
            raise TransportUnavailableError(f"communicator init did not complete on ranks {missing}")
        self._comms = [int(comm) for comm in comms if comm is not None]
        try:
            self._wait_async_ready(timeout_s=timeout_s)
        except Exception:
            self._poisoned = True
            self.close()
            raise

    def _wait_async_ready(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        pending = set(range(self.world_size))
        while pending:
            for rank in sorted(pending):
                if self._binding.comm_async_error(self._comms[rank]) == NCCL_SUCCESS:
                    pending.discard(rank)
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError_(f"communicator init timed out on ranks {sorted(pending)}")
            time.sleep(0.001)

    def _enqueue(self, request: CollectiveRequest) -> None:
        self._require_live()
        group = self._thread_group()
        if group is None:
            raise TransportStateError("collectives must be enqueued inside group_start/group_end")
        request.validate(world_size=self.world_size, sequence=self._sequence)
        rank = request.rank
        with scoped_current_device(self._runtime, self._devices[rank].index):
            if request.kind is CollectiveKind.ALL_REDUCE_SUM:
                self._binding.all_reduce(
                    send=request.send_ptr,
                    recv=request.recv_ptr,
                    count=request.count,
                    dtype=request.dtype,
                    comm=self._comms[rank],
                    stream=self._streams[rank],
                )
            elif request.kind is CollectiveKind.BROADCAST:
                self._binding.broadcast(
                    send=request.send_ptr,
                    recv=request.recv_ptr,
                    count=request.count,
                    dtype=request.dtype,
                    root=int(request.root),
                    comm=self._comms[rank],
                    stream=self._streams[rank],
                )
            else:  # pragma: no cover - enum is exhaustive
                raise TransportStateError(f"unsupported collective kind {request.kind}")
        group["requests"].setdefault(rank, []).append(request)

    def _thread_group(self) -> dict | None:
        return getattr(self._local, "group", None)

    def _check_async_errors(self) -> None:
        for rank, comm in enumerate(self._comms):
            status = self._binding.comm_async_error(comm)
            if status not in (NCCL_SUCCESS, NCCL_IN_PROGRESS):
                self._poison(f"rank {rank} communicator async error {status}")
                raise CommunicatorAbortedError(f"rank {rank} communicator reported async error {status}")

    def _poison(self, reason: str) -> None:
        self._poisoned = True
        self._poison_reason = reason

    def _require_rank(self, rank: int) -> None:
        if not 0 <= int(rank) < self.world_size:
            raise TransportStateError(f"rank {rank} is outside world size {self.world_size}")

    def _require_live(self) -> None:
        if self._closed:
            raise TransportStateError("transport is closed")
        if self._poisoned:
            raise CommunicatorAbortedError(f"communicator group is poisoned: {getattr(self, '_poison_reason', 'unknown')}")


def record_enqueue(recorder: EnqueueRecorder, rank: int, start: float, end: float) -> None:
    """Convenience wrapper so callers do not import the recorder type."""

    recorder.record(rank, start=start, end=end)
