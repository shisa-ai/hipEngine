"""Rank-bound runtime ownership and distributed context lifecycle.

One process owns every rank. Each rank has one explicit device context, one
nonblocking compute stream, and persistent per-device workspaces; allocation,
launch, and teardown select the owning device explicitly instead of relying on
whatever device happens to be current.

The context is the only object that owns communicator lifetime. It never
resolves model shards, schedules work, or mutates KV state; those belong to the
model plugin and the distributed runner adapter.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from hipengine.core.device import Device, scoped_current_device
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free as _free, malloc as _malloc
from hipengine.distributed.plan import DistributedPlan, PlanError
from hipengine.distributed.transport import CollectiveTransport, TransportStateError


@dataclass
class RankRuntime:
    """Device-bound runtime owner for one rank.

    All memory and stream operations select ``device`` for the duration of the
    call, so two ranks can be driven from one thread without leaking
    thread-local current-device state.
    """

    rank: int
    device: Device
    runtime: HipRuntime

    def activate(self):
        """Return a context manager that selects this rank's device."""

        return scoped_current_device(self.runtime, self.device.index)

    def malloc(self, nbytes: int) -> DeviceBuffer:
        return _malloc(nbytes, runtime=self.runtime, device=self.device)

    def free(self, buffer: DeviceBuffer) -> None:
        _free(buffer, runtime=self.runtime)

    def stream_create(self, *, nonblocking: bool = True) -> int:
        with self.activate():
            return self.runtime.stream_create(nonblocking=nonblocking)

    def stream_destroy(self, stream: int) -> None:
        with self.activate():
            self.runtime.stream_destroy(stream)

    def stream_synchronize(self, stream: int) -> None:
        with self.activate():
            self.runtime.stream_synchronize(stream)

    def event_create(self, *, flags: int = 0) -> int:
        with self.activate():
            return self.runtime.event_create(flags=flags)

    def event_record(self, event: int, stream: int = 0) -> None:
        with self.activate():
            self.runtime.event_record(event, stream)

    def event_synchronize(self, event: int) -> None:
        with self.activate():
            self.runtime.event_synchronize(event)

    def event_destroy(self, event: int) -> None:
        with self.activate():
            self.runtime.event_destroy(event)

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
        with self.activate():
            self.runtime.memcpy_async(dst, src, nbytes, kind, stream)

    def mem_get_info(self) -> tuple[int, int]:
        with self.activate():
            return self.runtime.mem_get_info()


def _validate_transport_identity(
    plan: DistributedPlan,
    transport: CollectiveTransport,
    runtime: HipRuntime,
) -> None:
    """A transport must describe the same ranks, devices, runtime, and algorithm.

    A world-size check alone accepts a transport built for the same *number* of
    ranks in a different order, which pairs a rank's allocations with another
    rank's communicator and stream. Nothing downstream can detect that, because
    every rank still holds a plausible buffer and every collective still
    completes.

    A transport that exposes no device list is a mock or an opaque stand-in; that
    is allowed only when the plan itself declares the mock algorithm, so
    substituting one for a real transport is an explicit choice.
    """

    declared = getattr(transport, "algorithm", None)
    if declared is not None and str(declared) != plan.algorithm:
        raise TransportStateError(
            f"transport algorithm {declared!r} does not match plan algorithm {plan.algorithm!r}"
        )
    devices = getattr(transport, "devices", None)
    if devices is None:
        if plan.algorithm != "mock":
            raise TransportStateError(
                f"transport for algorithm {plan.algorithm!r} must declare its ordered devices; "
                "only a mock plan may use an opaque transport"
            )
        return
    plan_devices = tuple(spec.device for spec in plan.ranks)
    transport_devices = tuple(devices)
    if transport_devices != plan_devices:
        raise TransportStateError(
            f"transport devices {[str(device) for device in transport_devices]} do not match "
            f"plan devices {[str(device) for device in plan_devices]}"
        )
    transport_runtime = getattr(transport, "runtime", None)
    if transport_runtime is not None and transport_runtime is not runtime:
        raise TransportStateError(
            "transport runtime is not the context runtime; one rank binding must not select "
            "devices through two runtimes"
        )


@dataclass
class DistributedContext:
    """Resolved plan plus rank runtimes and (for N>1) one collective transport."""

    plan: DistributedPlan
    ranks: tuple[RankRuntime, ...]
    transport: CollectiveTransport | None = None
    _closed: bool = field(default=False, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        plan: DistributedPlan,
        *,
        transport: CollectiveTransport | None = None,
        runtime: HipRuntime | None = None,
        require_hardware: bool = True,
        validate_devices: bool = True,
    ) -> "DistributedContext":
        """Build rank runtimes and, for N>1, the collective transport.

        N=1 deliberately does not construct a communicator: the existing TP1
        runner is the single-rank path, and building a one-rank RCCL group would
        add startup cost and a failure mode without changing behavior.
        """

        selected_runtime = runtime or get_hip_runtime()
        if validate_devices:
            _validate_plan_devices(plan, selected_runtime, require_hardware=require_hardware)
        ranks = tuple(
            RankRuntime(rank=spec.rank, device=spec.device, runtime=selected_runtime) for spec in plan.ranks
        )
        if plan.is_single_rank:
            if transport is not None:
                raise TransportStateError("N=1 context must not carry a collective transport")
            return cls(plan=plan, ranks=ranks, transport=None)
        if transport is None:
            if plan.algorithm != "rccl":
                raise TransportStateError(
                    f"algorithm {plan.algorithm!r} requires an explicit transport instance for N>1"
                )
            from hipengine.distributed.rccl import RcclTransport

            transport = RcclTransport([spec.device for spec in plan.ranks], runtime=selected_runtime)
        if int(getattr(transport, "world_size", 0)) != plan.world_size:
            if transport is not None:
                transport.close()
            raise TransportStateError(
                f"transport world size {getattr(transport, 'world_size', None)} does not match plan degree {plan.world_size}"
            )
        try:
            _validate_transport_identity(plan, transport, selected_runtime)
        except TransportStateError:
            transport.close()
            raise
        return cls(plan=plan, ranks=ranks, transport=transport)

    # -- accessors ----------------------------------------------------------

    @property
    def world_size(self) -> int:
        return self.plan.world_size

    @property
    def collectives_available(self) -> bool:
        return self.transport is not None

    def rank(self, rank: int) -> RankRuntime:
        try:
            return self.ranks[int(rank)]
        except (IndexError, ValueError) as error:
            raise PlanError(f"rank {rank!r} is outside this context") from error

    # -- lifecycle ----------------------------------------------------------

    def sync(self, *, timeout_s: float | None = None) -> None:
        if self.transport is None:
            return
        self.transport.sync(timeout_s=timeout_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    def __enter__(self) -> "DistributedContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _validate_plan_devices(plan: DistributedPlan, runtime: HipRuntime, *, require_hardware: bool) -> None:
    """Reject impossible plans before any allocation.

    With ``require_hardware`` the host must actually expose every planned device
    index. Without it (CPU tests) only internal consistency is checked, which the
    plan dataclass already enforces.
    """

    if not require_hardware:
        return
    count = runtime.device_count()
    for spec in plan.ranks:
        if spec.device.index >= count:
            raise PlanError(
                f"rank {spec.rank} requests hip:{spec.device.index} but the host exposes {count} device(s)"
            )
