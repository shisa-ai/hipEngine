"""Compiled host driver for the two-rank device-side staged exchange.

This is the graphed schedule's production reduction path: the per-layer
exchange moves out of the host dependency chain entirely. Each (layer, rank)
pair's captured graph stages its own down partial into a host-mapped slot,
publishes a system-visible flag, and spin-sums the other rank's staged row
against its own device partial in f32, narrowing once to bf16 - the same
widen/add/narrow arithmetic the host driver performs, so the output is
bit-identical to the host-summed payload. The host submits; it never waits
per layer.

Ownership: the handle owns its mapped staging/flag arenas and device step
counters for its lifetime; nothing is allocated or freed per step or per
reduce. Failures poison the handle and it refuses further work - the caller
must fail the whole rank group, exactly as with the other transports.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Mapping

from hipengine.core.build import build_hip
from hipengine.distributed.transport import TransportStateError

_SOURCE = Path(__file__).with_name("device_exchange_host.cpp")

_OK = 0
_CAPTURED = 1  # enqueue-only success; no work submitted for execution


def build_tp2_device_exchange(
    *,
    cache_root: str | Path | None = None,
    compiler: str = "hipcc",
    compiler_version: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    load: bool = True,
) -> Any:
    """Build (or reuse) the hipcc-built device-exchange host driver."""

    return build_hip(
        sources=[_SOURCE],
        family="tp2-device-exchange",
        compiler=compiler,
        compiler_version=compiler_version,
        force=force,
        dry_run=dry_run,
        load=load,
        cache_root=cache_root,
    )


def _bind(library: Any) -> None:
    if getattr(library, "_tp2_dev_exchange_bound", False):
        return
    library.tp2_dev_exchange_create.argtypes = [
        ctypes.POINTER(ctypes.c_int32),  # per-rank devices
        ctypes.c_int32,  # world
        ctypes.POINTER(ctypes.c_uint64),  # per-rank streams
        ctypes.c_int32,  # num exchange slots (layers)
        ctypes.c_int32,  # hidden
        ctypes.c_uint32,  # spin budget per exchange (bounded failure)
        ctypes.POINTER(ctypes.c_int32),  # out error code
    ]
    library.tp2_dev_exchange_create.restype = ctypes.c_void_p
    library.tp2_dev_exchange_step_begin.argtypes = [ctypes.c_void_p]
    library.tp2_dev_exchange_step_begin.restype = ctypes.c_int32
    library.tp2_dev_exchange_reset_timeouts.argtypes = [ctypes.c_void_p]
    library.tp2_dev_exchange_reset_timeouts.restype = ctypes.c_int32
    library.tp2_dev_exchange_bump.argtypes = [ctypes.c_void_p]
    library.tp2_dev_exchange_bump.restype = ctypes.c_int32
    library.tp2_dev_exchange_enqueue_rank.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int32,  # rank
        ctypes.c_void_p,  # own partial (device)
        ctypes.c_int32,  # slot
        ctypes.c_uint64,  # out payload (device)
    ]
    library.tp2_dev_exchange_enqueue_rank.restype = ctypes.c_int32
    library.tp2_dev_exchange_wait.argtypes = [ctypes.c_void_p]
    library.tp2_dev_exchange_wait.restype = ctypes.c_int32
    library.tp2_dev_exchange_last_error.argtypes = [ctypes.c_void_p]
    library.tp2_dev_exchange_last_error.restype = ctypes.c_char_p
    library.tp2_dev_exchange_destroy.argtypes = [ctypes.c_void_p]
    library.tp2_dev_exchange_destroy.restype = None
    library.tp2_dev_exchange_live_allocations.argtypes = []
    library.tp2_dev_exchange_live_allocations.restype = ctypes.c_int64
    library._tp2_dev_exchange_bound = True


def device_exchange_live_allocations(library: Any | None = None) -> int:
    """How many allocations the device-exchange driver currently owns.

    Process-wide ownership accounting, not a size: a create/close cycle and a
    create that failed halfway through must both leave this at zero. It exists
    because the timeout flags were allocated per rank and never freed, and
    nothing else in the process could observe that.
    """

    if library is None:
        library = build_tp2_device_exchange()
    _bind(library)
    return int(library.tp2_dev_exchange_live_allocations())


class CompiledDeviceExchange:
    """One persistent device-side two-rank reduction path, compiled.

    The public surface matches the staged transports for the graphed
    schedule: ``step_begin()`` once per token step, then
    ``enqueue_rank(rank, own_partial, slot, out)`` inside each rank's graph
    capture (or eagerly for the tail). No ``reduce`` waits: the consumer
    kernel's spin is the synchronization.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        devices: tuple[int, ...],
        streams: Mapping[int, int],
        num_layers: int,
        hidden: int,
        library: Any | None = None,
        max_spins: int = 2_000_000,
        rows: int = 1,
    ) -> None:
        if len(devices) != 2:
            raise TransportStateError("the device exchange serves exactly two ranks")
        if len(set(devices)) != len(devices):
            raise TransportStateError("device-exchange ranks must be distinct devices")
        if rows < 1:
            raise TransportStateError(f"rows must be positive, got {rows}")
        missing = [d for d in devices if d not in streams]
        if missing:
            raise TransportStateError(f"no stream given for ranks {missing}")
        self._runtime = runtime
        self.devices = tuple(int(d) for d in devices)
        # ``hidden`` is the row width the driver stages per slot, so a batched
        # caller passes the whole batch as one long row: the spin-add kernel is
        # already generic in its element count, and the per-slot staging and
        # flag arithmetic stay identical to the single-row route.
        self.hidden = int(hidden) * int(rows)
        self.rows = int(rows)
        self.num_layers = int(num_layers)
        self._poisoned = False
        self._handle: int | None = None
        # Step bookkeeping (what the artifact summary reports in device
        # mode); the reduce itself stays allocation-free.
        self.step_begins = 0
        self.enqueues: list[tuple[int, int, int]] = []
        self.waits = 0
        # Steps whose submission has not been observed by a ``wait`` yet.
        # ``step_begin`` and ``reset_timeouts`` both clear the device timeout
        # flags, so a clear that lands while an earlier step is unobserved can
        # erase a spin timeout no one has seen - and because the spin kernel
        # writes nothing on timeout, the erased failure would leave a stale
        # output row passing as a result. Eager enqueues also count, including
        # enqueues after a completed wait without a new bump. The native
        # driver distinguishes them from graph recording via stream status.
        # Capture-only enqueues have not executed and do not count.
        self._unobserved_steps = 0

        if library is None:
            library = build_tp2_device_exchange()
        _bind(library)
        self._library = library

        devices_arr = (ctypes.c_int32 * len(self.devices))(*self.devices)
        streams_arr = (ctypes.c_uint64 * len(self.devices))(
            *[int(streams[d]) for d in self.devices]
        )
        error_code = ctypes.c_int32(0)
        handle = library.tp2_dev_exchange_create(
            devices_arr,
            len(self.devices),
            streams_arr,
            self.num_layers,
            self.hidden,
            int(max_spins),
            ctypes.byref(error_code),
        )
        if not handle:
            detail = library.tp2_dev_exchange_last_error(None)
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(
                f"device exchange create failed (code {error_code.value}): {message}"
            )
        self._handle = int(handle)

    # -- state ------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def has_unobserved_work(self) -> bool:
        """True while submitted work has not been observed by a wait.

        A caller that is about to clear the timeout flags must first observe
        whatever it submitted: the spin kernel writes nothing on timeout, so a
        clear erases the only evidence that a stale row was produced.
        """

        return self._unobserved_steps > 0

    def _require_live(self) -> None:
        if self._poisoned:
            raise TransportStateError("device exchange is poisoned by an earlier failure")
        if self._handle is None:
            raise TransportStateError("device exchange is closed")

    def _refuse_unobserved_clear(self, what: str) -> None:
        """Refuse a flag clear while an earlier step's outcome is unobserved.

        The flags are sticky on purpose (that is what ``bump`` is for), so the
        only safe rule is to refuse the clear until a wait has observed the
        group's outcome rather than to clear and hope the failure was not real.
        """

        if self._unobserved_steps:
            raise TransportStateError(
                f"device exchange cannot {what} while {self._unobserved_steps} "
                "step(s) have been submitted and not waited: clearing the "
                "spin-timeout flags now would erase a timeout no wait has "
                "observed, and the spin kernel writes nothing on timeout, so a "
                "stale output row would pass as a result"
            )

    # -- the reduction ----------------------------------------------------

    def step_begin(self) -> None:
        """Bump both ranks' step counters once, on their own streams.

        This also clears the timeout flags, so it starts a step: it refuses
        while an earlier step's outcome is unobserved, because the clear would
        erase that step's timeout.
        """

        self._require_live()
        self._refuse_unobserved_clear("begin a step")
        code = self._library.tp2_dev_exchange_step_begin(ctypes.c_void_p(self._handle))
        if code != _OK:
            self._poisoned = True
            detail = self._library.tp2_dev_exchange_last_error(
                ctypes.c_void_p(self._handle)
            )
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(f"device exchange step_begin failed: {message}")
        self.step_begins += 1
        self._unobserved_steps += 1

    def reset_timeouts(self) -> None:
        """Clear both ranks' spin-timeout flags without touching the counters.

        A batched group resets once and then bumps per layer, so a timeout in
        any layer stays visible to :meth:`wait` instead of being cleared by the
        next layer's bump. The spin kernel writes nothing on timeout, so a
        cleared flag would let a stale output row pass as a result - which is
        also why this refuses while an earlier step is unobserved: the clear
        would erase a timeout that no wait has reported yet.
        """

        self._require_live()
        self._refuse_unobserved_clear("reset the spin timeouts")
        code = self._library.tp2_dev_exchange_reset_timeouts(
            ctypes.c_void_p(self._handle)
        )
        if code != _OK:
            self._poisoned = True
            detail = self._library.tp2_dev_exchange_last_error(
                ctypes.c_void_p(self._handle)
            )
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(
                f"device exchange reset_timeouts failed: {message}"
            )

    def bump(self) -> None:
        """Advance both ranks' step counters without clearing the timeouts.

        The publish/spin pair compares against the counter, so a batched caller
        that reuses two slots must advance it once per layer; without the bump
        the peer's flag would already satisfy the comparison and the spin would
        read the previous layer's staging.
        """

        self._require_live()
        code = self._library.tp2_dev_exchange_bump(ctypes.c_void_p(self._handle))
        if code != _OK:
            self._poisoned = True
            detail = self._library.tp2_dev_exchange_last_error(
                ctypes.c_void_p(self._handle)
            )
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(f"device exchange bump failed: {message}")
        self.step_begins += 1
        self._unobserved_steps += 1

    def enqueue_rank(
        self,
        rank: int,
        own_partial: int,
        slot: int,
        out_payload: int,
    ) -> int:
        """Enqueue one rank's exchange for one slot; return the output pointer.

        Stream-ordered and host-synchronization-free: this is the capturable
        unit a graphed schedule records inside each rank's layer graph.
        """

        self._require_live()
        code = self._library.tp2_dev_exchange_enqueue_rank(
            ctypes.c_void_p(self._handle),
            int(rank),
            ctypes.c_void_p(int(own_partial)),
            int(slot),
            ctypes.c_uint64(int(out_payload)),
        )
        # Native enqueue returns 1 only for an active stream capture. All
        # other nonzero results are failures; an eager success is unobserved
        # work even when the caller did not start a new counter step.
        if code not in (_OK, _CAPTURED):
            self._poisoned = True
            detail = self._library.tp2_dev_exchange_last_error(
                ctypes.c_void_p(self._handle)
            )
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(f"device exchange enqueue failed: {message}")
        if code == _OK:
            self._unobserved_steps = max(1, self._unobserved_steps)
        self.enqueues.append((int(rank), int(slot), int(own_partial)))
        return int(out_payload)

    def wait(self) -> None:
        """Sync both rank streams once each (the eager tail's only wait)."""

        self._require_live()
        code = self._library.tp2_dev_exchange_wait(ctypes.c_void_p(self._handle))
        if code != _OK:
            self._poisoned = True
            detail = self._library.tp2_dev_exchange_last_error(
                ctypes.c_void_p(self._handle)
            )
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(f"device exchange wait failed: {message}")
        self.waits += 1
        # The step's outcome is now observed: a later clear cannot erase a
        # timeout this wait would have raised on.
        self._unobserved_steps = 0

    # -- teardown ---------------------------------------------------------

    def close(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        self._library.tp2_dev_exchange_destroy(ctypes.c_void_p(handle))

    def __enter__(self) -> "CompiledDeviceExchange":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
