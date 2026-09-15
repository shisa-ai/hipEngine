"""Two-rank page-locked host-staged all-reduce transport.

This is the measured ``staged_exchange_batched`` structure from
``scripts/tp_collective_bench.py``, moved into a reusable runtime component.
Per reduction it stages each rank's partial D2H through pinned host memory,
sums on the host in f32, and copies the reduced vector H2D back to every
rank. It exists because it is the only reduction structure measured on this
host that projects a TP2 win: 42.9 us per reduction against 153.7 us for the
copy-free RCCL per-step chain
(``benchmarks/results/2026-09-14-w7900-tp2-dependent-reduction-chain.json``).

Enqueue discipline (the measured protocol, kept exactly):

* every rank's D2H is submitted before any stream is awaited;
* each rank's stream is then awaited once - the wait covers both copies and
  doubles as the staging-slot reuse guard, because the next write into a slot
  set happens after a wait that is queued after the copy that read it;
* the host sum is a single f32 reduction over a contiguous
  ``(ranks, hidden)`` view of the staging arena;
* every rank's H2D is submitted from a pinned reduced-payload slot with no
  return wait - the next consumer runs on that rank's stream, so stream order
  already places it after the reduction.

Ownership: every allocation (pinned arena, reduced device buffers) belongs to
the device it serves and is freed through a scoped current-device selection.
Failure poisons the transport: after any enqueue/wait error the transport
refuses further reductions and reports poisoned, and it is the caller's job
to fail the whole rank group - a partially advanced session is not reusable.
"""

from __future__ import annotations

import ctypes
from typing import Any, Mapping

import numpy as np

from hipengine.core.device import scoped_current_device
from hipengine.core.runtime import MemcpyKind
from hipengine.distributed.transport import (
    TransportError,
    TransportStateError,
)

#: The staging arena holds ``slots`` slot sets of one f32 row per rank. Two
#: slot sets keep a host write from racing the H2D that reads the reduced
#: payload of the previous call even if a caller pipelines reductions without
#: waiting; under the intended discipline one wait per stream per call already
#: guards reuse (see the module docstring).
_SLOT_SETS = 2


class StagedExchangeTransport:
    """One persistent two-rank host-staged reduction path.

    The transport owns its staging and result buffers for the lifetime of the
    session: nothing is allocated or freed per reduction. ``partial_ptrs`` and
    the returned reduced buffers are device-resident; the host never reads the
    hidden state back except through the pinned partial copy this reduction
    stages, which is the allowed pinned-host reduction boundary.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        devices: tuple[int, ...],
        streams: Mapping[int, int],
        hidden: int,
    ) -> None:
        if not devices:
            raise TransportStateError("a staged exchange needs at least one rank")
        if len(set(devices)) != len(devices):
            raise TransportStateError("staged-exchange ranks must be distinct devices")
        missing = [d for d in devices if d not in streams]
        if missing:
            raise TransportStateError(f"no stream given for ranks {missing}")
        self._runtime = runtime
        self.devices = tuple(int(d) for d in devices)
        self._streams = {int(d): int(streams[d]) for d in devices}
        self.hidden = int(hidden)
        self._slot_nbytes = self.hidden * 4
        self._poisoned = False
        self.reductions = 0

        # Per-rank reduced result buffers: persistent, device-resident, and
        # written by the H2D on the rank's own stream.
        self._reduced_ptrs: dict[int, int] = {}
        for device in self.devices:
            with scoped_current_device(runtime, device):
                self._reduced_ptrs[device] = int(runtime.malloc(self._slot_nbytes))

        # Pinned staging: ``_SLOT_SETS`` slot sets of one f32 row per rank,
        # page-locked so a D2H does not stage through a driver bounce buffer.
        # A separate pinned reduced-payload region holds what the H2D reads;
        # it alternates with the same slot index as the D2H slots so the
        # documented wait guard covers both directions.
        self._arena_nbytes = _SLOT_SETS * len(self.devices) * self._slot_nbytes
        self._host = (ctypes.c_ubyte * self._arena_nbytes)()
        self._arena_ptr = ctypes.addressof(self._host)
        runtime.host_register(self._arena_ptr, self._arena_nbytes)
        self._payload_nbytes = _SLOT_SETS * self._slot_nbytes
        self._payload_host = (ctypes.c_ubyte * self._payload_nbytes)()
        self._payload_ptr = ctypes.addressof(self._payload_host)
        runtime.host_register(self._payload_ptr, self._payload_nbytes)
        self._slot = 0

    # -- state ------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def reduced_ptr(self, device: int) -> int:
        """The device-resident reduced vector for one rank."""

        self._require_live()
        device = int(device)
        if device not in self._reduced_ptrs:
            raise TransportStateError(f"rank {device} is not part of this exchange")
        return self._reduced_ptrs[device]

    # -- the reduction ----------------------------------------------------

    def reduce(self, partial_ptrs: Mapping[int, int]) -> dict[int, int]:
        """Reduce one partial per rank into every rank's reduced buffer.

        Both ranks' D2H copies are submitted on their own streams before any
        wait, each stream is awaited once, the host sums the staging slots in
        f32, and every rank's H2D is submitted with no return wait. Returns
        the reduced device pointer per rank; the value is stream-ordered after
        the H2D on that rank's stream, so the caller's next consumer needs no
        host synchronization.
        """

        self._require_live()
        missing = [d for d in self.devices if int(d) not in partial_ptrs]
        if missing:
            self._poisoned = True
            raise TransportStateError(f"no partial given for ranks {missing}")
        runtime = self._runtime
        try:
            slot = self._slot
            self._slot = 1 - slot
            partial_base = self._arena_ptr + slot * len(self.devices) * self._slot_nbytes

            # Batch every rank's D2H before awaiting any stream: the measured
            # protocol. Each copy rides its own rank's stream, so stream order
            # places it after the producer of that partial.
            for index, device in enumerate(self.devices):
                with scoped_current_device(runtime, device):
                    runtime.memcpy_async(
                        partial_base + index * self._slot_nbytes,
                        int(partial_ptrs[device]),
                        self._slot_nbytes,
                        MemcpyKind.DEVICE_TO_HOST,
                        self._streams[device],
                    )

            # One wait per stream. This is also the slot-reuse guard: the
            # next write into this slot set happens after the caller reaches
            # the next reduce, whose wait is queued after everything this
            # call enqueued on that stream - including the H2D below.
            for device in self.devices:
                with scoped_current_device(runtime, device):
                    runtime.stream_synchronize(self._streams[device])

            # Single f32 reduction over the contiguous (ranks, hidden) view.
            slots = np.frombuffer(self._host, dtype=np.uint8)
            rows = np.frombuffer(
                slots[
                    slot
                    * len(self.devices)
                    * self._slot_nbytes : slot
                    * len(self.devices)
                    * self._slot_nbytes
                    + len(self.devices) * self._slot_nbytes
                ].tobytes(),
                dtype="<f4",
            ).reshape(len(self.devices), self.hidden)
            reduced = rows.sum(axis=0, dtype=np.float32)

            # Publish into this call's pinned payload slot, then submit every
            # rank's H2D from it with no return wait. The next call's write
            # into this slot cannot race this H2D: that write happens after
            # the next call's D2H wait, and that wait is queued on each rank's
            # stream after this H2D.
            payload_base = self._payload_ptr + slot * self._slot_nbytes
            np.frombuffer(self._payload_host, dtype=np.uint8)[
                slot * self._slot_nbytes : (slot + 1) * self._slot_nbytes
            ] = np.ascontiguousarray(reduced, dtype="<f4").view(np.uint8)
            for device in self.devices:
                with scoped_current_device(runtime, device):
                    runtime.memcpy_async(
                        self._reduced_ptrs[device],
                        payload_base,
                        self._slot_nbytes,
                        MemcpyKind.HOST_TO_DEVICE,
                        self._streams[device],
                    )
            self.reductions += 1
            return dict(self._reduced_ptrs)
        except TransportError:
            self._poisoned = True
            raise
        except Exception as error:  # noqa: BLE001 - poison on any failure
            self._poisoned = True
            raise TransportError(f"staged exchange failed: {type(error).__name__}: {error}") from error

    # -- teardown ---------------------------------------------------------

    def close(self) -> None:
        """Free every buffer this transport owns, through its own device."""

        runtime = self._runtime
        for device in self.devices:
            ptr = self._reduced_ptrs.pop(device, None)
            if ptr is not None:
                with scoped_current_device(runtime, device):
                    runtime.free(ptr)
        if self._arena_ptr:
            runtime.host_unregister(self._arena_ptr)
            self._arena_ptr = 0
        if self._payload_ptr:
            runtime.host_unregister(self._payload_ptr)
            self._payload_ptr = 0

    def __enter__(self) -> "StagedExchangeTransport":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_live(self) -> None:
        if self._poisoned:
            raise TransportError(
                "staged exchange is poisoned by an earlier failure; "
                "the rank group is not reusable"
            )
