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
    library._tp2_dev_exchange_bound = True


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

    def _require_live(self) -> None:
        if self._poisoned:
            raise TransportStateError("device exchange is poisoned by an earlier failure")
        if self._handle is None:
            raise TransportStateError("device exchange is closed")

    # -- the reduction ----------------------------------------------------

    def step_begin(self) -> None:
        """Bump both ranks' step counters once, on their own streams."""

        self._require_live()
        code = self._library.tp2_dev_exchange_step_begin(ctypes.c_void_p(self._handle))
        if code != _OK:
            self._poisoned = True
            detail = self._library.tp2_dev_exchange_last_error(
                ctypes.c_void_p(self._handle)
            )
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(f"device exchange step_begin failed: {message}")
        self.step_begins += 1

    def reset_timeouts(self) -> None:
        """Clear both ranks' spin-timeout flags without touching the counters.

        A batched group resets once and then bumps per layer, so a timeout in
        any layer stays visible to :meth:`wait` instead of being cleared by the
        next layer's bump. The spin kernel writes nothing on timeout, so a
        cleared flag would let a stale output row pass as a result.
        """

        self._require_live()
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
        if code != _OK:
            self._poisoned = True
            detail = self._library.tp2_dev_exchange_last_error(
                ctypes.c_void_p(self._handle)
            )
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(f"device exchange enqueue failed: {message}")
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
