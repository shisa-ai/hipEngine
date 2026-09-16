"""Compiled host driver for the two-rank staged exchange.

This is the transport the resume lever specifies: drive the staged exchange
from compiled code - the same batched protocol the Python route
(:class:`hipengine.distributed.staged.StagedExchangeTransport`) runs, but
enqueued from a hipcc-built host driver like every other kernel instead of
through per-call ctypes submissions and device-scoped switches - and remove
the H2D return path: the reduced f32 payload lives in mapped pinned host
memory, so both ranks' consumer kernels (the boundary ``f32_to_bf16`` cast)
read it zero-copy over the bus and no host-to-device copy is submitted.

Measured basis (W7900 + RX 7900 XTX, ``benchmarks/results/
2026-09-14-w7900-tp2-staged-exchange-native-ab.json``): the compiled A/B arm
ran the same protocol at 20.8 us per reduction against ~40 us for the Python
loop, and the slice probe attributed ~29 us of the Python route's wall to the
H2D return submissions. The gap is protocol overhead, not copy bandwidth.

Bit-parity with the Python route: for the two-rank world this driver serves,
the host sum is the same single f32 addition per element in the same rank
order (row 0 + row 1), and bf16 staging widens by the same bit shift into the
high half - so a reduced payload is bit-identical to the Python route's. The
driver rejects world != 2 at create; the Python route stays the general
fallback and the registered fallback route.

Ownership: the driver owns its mapped pinned arenas for the transport's
lifetime; nothing is allocated or freed per reduction. Any reduce failure
poisons the transport and it refuses further reductions - the caller must
fail the whole rank group, exactly as with the Python route.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Mapping

from hipengine.core.build import build_hip
from hipengine.distributed.transport import (
    TransportError,
    TransportStateError,
    require_rows_value,
)

_SOURCE = Path(__file__).with_name("staged_exchange_host.cpp")

#: Staging-dtype codes the compiled driver's create validates.
_STAGING_DTYPE_CODES = {"f32": 0, "bf16": 1}

#: Alternating slot sets, the same discipline the Python route pins.
_SLOT_SETS = 2

#: Driver return codes (mirrored from the C source).
_REDUCE_OK = 0


def build_tp2_staged_exchange(
    *,
    cache_root: str | Path | None = None,
    compiler: str = "hipcc",
    compiler_version: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
):
    """Build or reuse the compiled staged-exchange host driver.

    Host-only source: no device code, so no offload arch is required, but the
    shared cache key discipline (source bytes + flags + compiler version)
    applies unchanged. Lazy: nothing compiles or loads at import time.
    """

    return build_hip(
        sources=(_SOURCE,),
        family="tp2-staged-exchange",
        profile="baseline",
        cache_root=cache_root,
        compiler=compiler,
        compiler_version=compiler_version,
        extra_flags=("-std=c++17",),
        force=force,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _bind(library: Any) -> None:
    """Pin the C ABI onto a freshly loaded driver library (idempotent)."""

    if getattr(library, "_tp2_staged_bound", False):
        return
    library.tp2_staged_create.argtypes = [
        ctypes.POINTER(ctypes.c_int32),  # devices
        ctypes.c_int32,  # world
        ctypes.POINTER(ctypes.c_void_p),  # streams
        ctypes.c_int32,  # hidden
        ctypes.c_int32,  # staging dtype code
        ctypes.c_int32,  # slot sets
        ctypes.c_int32,  # capacity rows
        ctypes.POINTER(ctypes.c_int32),  # out error code
    ]
    library.tp2_staged_create.restype = ctypes.c_void_p
    library.tp2_staged_reduce.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_void_p),  # per-rank partial pointers
        ctypes.POINTER(ctypes.c_uint64),  # out payload device pointer
    ]
    library.tp2_staged_reduce.restype = ctypes.c_int32
    library.tp2_staged_reduce_rows.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_void_p),  # per-rank partial pointers
        ctypes.c_int32,  # active rows
        ctypes.POINTER(ctypes.c_uint64),  # out payload device pointer
    ]
    library.tp2_staged_reduce_rows.restype = ctypes.c_int32
    library.tp2_staged_reduce_at.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_void_p),  # per-rank partial pointers
        ctypes.c_int32,  # caller-chosen payload slot
        ctypes.POINTER(ctypes.c_uint64),  # out payload device pointer
    ]
    library.tp2_staged_reduce_at.restype = ctypes.c_int32
    library.tp2_staged_reduce_at_rows.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.POINTER(ctypes.c_void_p),  # per-rank partial pointers
        ctypes.c_int32,  # caller-chosen payload slot
        ctypes.c_int32,  # active rows
        ctypes.POINTER(ctypes.c_uint64),  # out payload device pointer
    ]
    library.tp2_staged_reduce_at_rows.restype = ctypes.c_int32
    library.tp2_staged_payload_base.argtypes = [ctypes.c_void_p]
    library.tp2_staged_payload_base.restype = ctypes.c_uint64
    library.tp2_staged_slot_stride.argtypes = [ctypes.c_void_p]
    library.tp2_staged_slot_stride.restype = ctypes.c_size_t
    library.tp2_staged_last_error.argtypes = [ctypes.c_void_p]
    library.tp2_staged_last_error.restype = ctypes.c_char_p
    library.tp2_staged_destroy.argtypes = [ctypes.c_void_p]
    library.tp2_staged_destroy.restype = None
    library._tp2_staged_bound = True


class CompiledStagedExchangeTransport:
    """One persistent two-rank host-staged reduction path, compiled.

    The transport's public surface matches the Python route's for the shard
    group: ``reduce(partial_ptrs)`` returns, per rank, the device-visible
    address of the reduced f32 row to consume - here the mapped pinned
    payload, the same address for both ranks - stream-ordered behind the
    host sum, so the caller's next consumer needs no synchronization.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        devices: tuple[int, ...],
        streams: Mapping[int, int],
        hidden: int,
        staging_dtype: str = "f32",
        library: Any | None = None,
        slot_sets: int = _SLOT_SETS,
        rows: int = 1,
    ) -> None:
        if not devices:
            raise TransportStateError("a staged exchange needs at least one rank")
        if len(set(devices)) != len(devices):
            raise TransportStateError("staged-exchange ranks must be distinct devices")
        missing = [d for d in devices if d not in streams]
        if missing:
            raise TransportStateError(f"no stream given for ranks {missing}")
        if staging_dtype not in _STAGING_DTYPE_CODES:
            raise TransportStateError(
                f"unsupported staging dtype {staging_dtype!r}; "
                f"expected one of {sorted(_STAGING_DTYPE_CODES)}"
            )
        self._runtime = runtime
        self.devices = tuple(int(d) for d in devices)
        self.hidden = int(hidden)
        # ``rows`` is the capacity of one slot; ``reduce(rows=n)`` stages and
        # sums exactly the first ``n`` rows and zeroes the inactive tail.
        self.rows = require_rows_value(rows, capacity=rows)
        self.staging_dtype = str(staging_dtype)
        self.staging_row_bytes = self.hidden * (2 if staging_dtype == "bf16" else 4)
        self.staging_nbytes = self.rows * self.staging_row_bytes
        self._poisoned = False
        self.reductions = 0
        self._handle: int | None = None

        if library is None:
            library = build_tp2_staged_exchange()
        _bind(library)
        self._library = library

        devices_arr = (ctypes.c_int32 * len(self.devices))(*self.devices)
        streams_arr = (ctypes.c_void_p * len(self.devices))(
            *[int(streams[d]) for d in self.devices]
        )
        error_code = ctypes.c_int32(0)
        slot_sets = int(slot_sets)
        if slot_sets < 2:
            raise TransportStateError("a staged exchange needs at least two slot sets")
        handle = library.tp2_staged_create(
            devices_arr,
            len(self.devices),
            streams_arr,
            self.hidden,
            _STAGING_DTYPE_CODES[staging_dtype],
            slot_sets,
            self.rows,
            ctypes.byref(error_code),
        )
        if not handle:
            detail = library.tp2_staged_last_error(None)
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportStateError(
                f"compiled staged exchange create failed (code {error_code.value}): {message}"
            )
        self._handle = int(handle)
        self._slot_sets = int(slot_sets)
        self._payload_base = int(
            library.tp2_staged_payload_base(ctypes.c_void_p(self._handle))
        )
        self._slot_stride = int(
            library.tp2_staged_slot_stride(ctypes.c_void_p(self._handle))
        )
        # Per-call scratch: one pointer per rank plus the payload out-param,
        # allocated once - the reduce path allocates nothing.
        self._partial_array = (ctypes.c_void_p * len(self.devices))()
        self._payload_out = ctypes.c_uint64(0)

    # -- state ------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def payload_ptr(self, slot: int) -> int:
        """The device-visible address of one fixed payload slot's reduced row."""

        self._require_live()
        slot = int(slot)
        if slot < 0 or slot >= self._slot_sets:
            raise TransportStateError(
                f"payload slot {slot} outside this transport's {self._slot_sets} slot sets"
            )
        return self._payload_base + slot * self._slot_stride

    # -- the reduction ----------------------------------------------------

    def reduce(
        self,
        partial_ptrs: Mapping[int, int],
        *,
        slot: int | None = None,
        rows: int | None = None,
    ) -> dict[int, int]:
        """Reduce the active rows of one partial per rank into the mapped payload.

        Both ranks' D2H copies are submitted before any wait, each stream is
        awaited once, the compiled host sums the staged rows in f32 (reading
        them in this transport's staging dtype), and the payload's
        device-visible address is returned per rank with no H2D copy: every
        rank's consumer reads the mapped host row zero-copy on its own
        stream. Only the first ``rows`` rows are summed; the trailing capacity
        rows of the payload are zeroed. With ``slot`` the reduce publishes into
        that fixed payload slot set and the internal alternation is untouched.

        The capacity-1 single-row route keeps calling the original ABI
        functions unchanged.
        """

        self._require_live()
        active = require_rows_value(self.rows if rows is None else rows, capacity=self.rows)
        missing = [d for d in self.devices if int(d) not in partial_ptrs]
        if missing:
            self._poisoned = True
            raise TransportStateError(f"no partial given for ranks {missing}")
        for index, device in enumerate(self.devices):
            self._partial_array[index] = int(partial_ptrs[device])
        single_row = self.rows == 1 and active == 1
        if slot is None:
            if single_row:
                code = self._library.tp2_staged_reduce(
                    ctypes.c_void_p(self._handle),
                    self._partial_array,
                    ctypes.byref(self._payload_out),
                )
            else:
                code = self._library.tp2_staged_reduce_rows(
                    ctypes.c_void_p(self._handle),
                    self._partial_array,
                    active,
                    ctypes.byref(self._payload_out),
                )
        elif single_row:
            code = self._library.tp2_staged_reduce_at(
                ctypes.c_void_p(self._handle),
                self._partial_array,
                int(slot),
                ctypes.byref(self._payload_out),
            )
        else:
            code = self._library.tp2_staged_reduce_at_rows(
                ctypes.c_void_p(self._handle),
                self._partial_array,
                int(slot),
                active,
                ctypes.byref(self._payload_out),
            )
        if code != _REDUCE_OK:
            self._poisoned = True
            detail = self._library.tp2_staged_last_error(ctypes.c_void_p(self._handle))
            message = detail.decode(errors="replace") if detail else "unknown error"
            raise TransportError(f"compiled staged exchange failed (code {code}): {message}")
        self.reductions += 1
        return {device: int(self._payload_out.value) for device in self.devices}

    # -- teardown ---------------------------------------------------------

    def close(self) -> None:
        """Free the driver's mapped pinned arenas, exactly once."""

        if self._handle is None:
            return
        self._library.tp2_staged_destroy(ctypes.c_void_p(self._handle))
        self._handle = None

    def __enter__(self) -> "CompiledStagedExchangeTransport":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_live(self) -> None:
        if self._handle is None:
            raise TransportError("compiled staged exchange is closed")
        if self._poisoned:
            raise TransportError(
                "staged exchange is poisoned by an earlier failure; the rank group is not reusable"
            )
