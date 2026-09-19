"""The per-layer MLP shard group: both ranks' chains plus the staged exchange.

One :class:`MlpShardGroup` owns the MLP-only TP2 execution schedule for a
model-owning loop (``hipengine.distributed.tp2_generate``): for one layer's
MLP, each rank runs its own shard chain (gate GEMV, up GEMV, SiLU-multiply,
down GEMV to an f32 partial) from its own device-resident input row, the
staged exchange reduces the two partials, and the reduced value is cast to
bf16 on every rank at the declared boundary. The returned per-rank bf16
pointer is what the caller's single residual add consumes - partials are
summed before any residual/norm consumer, and the residual is added once, by
the caller, exactly as in the TP1 schedule.

Ownership and discipline:

* both ranks' chains are enqueued before either is awaited (the measured
  batched protocol; the exchange's host wait covers each rank's copies);
* the group never allocates or frees per call - every buffer is persistent
  per rank, allocated at construction;
* the exchange runs on one of two registered transports selected by ``driver``:
  ``"compiled"`` (default: the hipcc-built host driver with the H2D return
  path removed - consumers read the mapped pinned payload zero-copy; measured
  156 us p50 in-schedule against 201 us for the Python route, with
  bit-identical reduced payloads) or ``"python"`` (the original staged route:
  pinned D2H, host f32 sum, H2D return - the registered fallback); both
  reduce the same partials in the same order, so the reduced f32 payload is
  bit-identical for the two-rank world;
* any exchange failure poisons the transport and the group refuses further
  forwards; the caller must fail the whole rank group - a partially advanced
  session is not reusable;
* ``close`` frees every rank's buffers and weights exactly once.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from hipengine.core.device import scoped_current_device
from hipengine.distributed.device_exchange_compiled import CompiledDeviceExchange
from hipengine.distributed.shard_exec import MlpShardRank
from hipengine.distributed.staged import StagedExchangeTransport
from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport
from hipengine.distributed.transport import TransportError, require_rows_value


class ShardGroupError(RuntimeError):
    """The shard group cannot continue; the rank group must fail."""


class MlpShardGroup:
    """Both ranks' persistent MLP shard state for a set of layers.

    ``weights`` maps ``layer_id -> {device: {role: ShardWeight}}`` (the
    materialized planner shards from
    :func:`hipengine.distributed.shard_weights.upload_mlp_shard_weights`).
    The group builds one :class:`MlpShardRank` per (layer, device) and one
    shared exchange transport (selected by ``driver``); the reduced f32
    buffers are the transport's persistent per-rank buffers (Python route) or
    the mapped pinned payload rows (compiled route), reused layer to layer,
    which is safe because the caller consumes the reduced value on the same
    stream that the next layer's exchange reads.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        devices: tuple[int, ...],
        streams: Mapping[int, int],
        hidden: int,
        per_rank_ffn: int | Mapping[int, int],
        weights: Mapping[int, Mapping[int, Mapping[str, Any]]],
        staging_dtype: str = "f32",
        driver: str = "compiled",
        mlp_decode_variant: str | Mapping[int, str | None] | None = None,
        slot_sets: int = 2,
        rows: int = 1,
        owns_weights: bool = True,
        reduce_mode: str = "host",
    ) -> None:
        if not weights:
            raise ShardGroupError("a shard group needs at least one layer")
        if driver not in {"python", "compiled"}:
            raise ShardGroupError(
                f"unknown exchange driver {driver!r}; expected 'python' or 'compiled'"
            )
        if reduce_mode not in {"host", "device"}:
            raise ShardGroupError(
                f"unknown reduce_mode {reduce_mode!r}; expected 'host' or 'device'"
            )
        self._runtime = runtime
        self.devices = tuple(int(d) for d in devices)
        self.hidden = int(hidden)
        # A scalar applies to every rank (the even split); a mapping gives each
        # rank its own shard width, which an uneven split needs. The widths
        # must match the uploaded payloads - the manifest is the authority, and
        # the rank's own launch validates against it.
        if isinstance(per_rank_ffn, Mapping):
            missing_width = [d for d in self.devices if int(d) not in per_rank_ffn]
            if missing_width:
                raise ShardGroupError(
                    f"per_rank_ffn has no width for ranks {missing_width}"
                )
            self.per_rank_ffn: dict[int, int] = {
                int(d): int(per_rank_ffn[int(d)]) for d in self.devices
            }
        else:
            self.per_rank_ffn = {int(d): int(per_rank_ffn) for d in self.devices}
        if any(width < 1 for width in self.per_rank_ffn.values()):
            raise ShardGroupError(f"per_rank_ffn must be positive, got {self.per_rank_ffn!r}")
        if isinstance(mlp_decode_variant, Mapping):
            self._mlp_decode_variant: dict[int, str | None] = {
                int(d): (
                    str(mlp_decode_variant[int(d)])
                    if mlp_decode_variant.get(int(d)) is not None
                    else None
                )
                for d in self.devices
            }
        else:
            uniform = (
                str(mlp_decode_variant) if mlp_decode_variant is not None else None
            )
            self._mlp_decode_variant = {int(d): uniform for d in self.devices}
        # ``rows`` is the buffer/slot capacity; each ``forward(rows=n)`` runs
        # exactly ``n`` active rows and the exchange zeroes the inactive tail.
        self.rows = require_rows_value(rows, capacity=rows)
        self._streams = {int(d): int(streams[d]) for d in self.devices}
        self._closed = False
        # A bulk-prefill group can share the decode group's uploaded shard
        # weights; ``owns_weights=False`` keeps its ``close`` from freeing
        # them twice.
        self._owns_weights = bool(owns_weights)
        self.exchange_walls_s: list[float] = []
        self.reduce_mode = str(reduce_mode)

        self._ranks: dict[tuple[int, int], MlpShardRank] = {}
        for layer_id, per_device in weights.items():
            missing = [d for d in self.devices if int(d) not in per_device]
            if missing:
                raise ShardGroupError(f"layer {layer_id} has no shard weights for ranks {missing}")
            for device in self.devices:
                self._ranks[(int(layer_id), device)] = MlpShardRank(
                    runtime,
                    device=device,
                    stream=self._streams[device],
                    weights=per_device[device],
                    hidden=hidden,
                    per_rank_ffn=self.per_rank_ffn[int(device)],
                    partial_dtype=staging_dtype,
                    mlp_decode_variant=self._mlp_decode_variant[int(device)],
                    rows=self.rows,
                    owns_weights=self._owns_weights,
                )
        # The device-side reduction, when selected: a peer-spin kernel sums the
        # two ranks' partials into a device buffer, so no f32 payload crosses
        # PCIe and no boundary cast is needed. It needs two ranks and bf16
        # partials (the kernel's widen/add/narrow arithmetic is bf16 in and out).
        self._device_exchange: CompiledDeviceExchange | None = None
        if self.reduce_mode == "device":
            if len(self.devices) != 2:
                raise ShardGroupError(
                    "the device-side reduction serves exactly two ranks; "
                    f"got {len(self.devices)}"
                )
            if str(staging_dtype) != "bf16":
                raise ShardGroupError(
                    "the device-side reduction is bf16 in and out; "
                    f"staging_dtype={staging_dtype!r} would change the arithmetic"
                )
            self._device_exchange = CompiledDeviceExchange(
                runtime,
                devices=self.devices,
                streams=self._streams,
                # Two alternating slots with a counter bump per layer, so the
                # staging stays a fixed size instead of one slot per layer.
                num_layers=2,
                hidden=int(hidden),
                rows=self.rows,
            )
        if driver == "compiled":
            # The compiled host driver: the same batched protocol enqueued
            # from hipcc-built code, with the H2D return path removed (the
            # consumer reads the mapped pinned payload zero-copy). Bit-ident
            # sums for the two-rank world; the Python route stays the
            # registered fallback.
            self._transport: StagedExchangeTransport | CompiledStagedExchangeTransport = (
                CompiledStagedExchangeTransport(
                    runtime,
                    devices=self.devices,
                    streams=self._streams,
                    hidden=hidden,
                    staging_dtype=staging_dtype,
                    slot_sets=slot_sets,
                    rows=self.rows,
                )
            )
        else:
            self._transport = StagedExchangeTransport(
                runtime,
                devices=self.devices,
                streams=self._streams,
                hidden=hidden,
                staging_dtype=staging_dtype,
                rows=self.rows,
            )
        # The rank partials and the transport staging must agree on dtype; a
        # mismatch would stage an f32 partial as bf16 (or the reverse).
        if str(getattr(self._transport, "staging_dtype", staging_dtype)) != str(staging_dtype):
            raise ShardGroupError(
                f"transport staging dtype {self._transport.staging_dtype!r} does not "
                f"match the rank partial dtype {staging_dtype!r}"
            )
        # The bf16 boundary buffer per rank: the reduced f32 value cast to the
        # dtype the TP1 local chain's down projection writes, so the caller's
        # residual add consumes one identical contract on either path. With the
        # device-side reduction the same buffer is the spin-add kernel's output,
        # so the consumer reads it directly and no cast is enqueued.
        self._out_ptrs: dict[int, int] = {}
        for device in self.devices:
            with scoped_current_device(runtime, device):
                self._out_ptrs[device] = int(runtime.malloc(self.rows * self.hidden * 2))

    # -- state -------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return self._transport.poisoned

    @property
    def reductions(self) -> int:
        return self._transport.reductions

    @property
    def mlp_decode_variant(self) -> str | None:
        """The fused gate/up+SiLU variant the ranks resolve, or None.

        With an even split every rank resolves the same variant, so the scalar
        form is kept. An uneven split gives ranks different shard shapes, and
        the shape-qualified policy lookup can resolve differently per rank; in
        that case the per-rank mapping is returned and a scalar answer would be
        a lie.
        """

        resolved = set(self._mlp_decode_variant.values())
        if len(resolved) == 1:
            return next(iter(resolved))
        return self._mlp_decode_variant

    def output_ptr(self, device: int) -> int:
        """This rank's persistent bf16 MLP-output buffer."""

        self._require_live()
        device = int(device)
        if device not in self._out_ptrs:
            raise ShardGroupError(f"rank {device} is not part of this group")
        return self._out_ptrs[device]

    # -- execution ---------------------------------------------------------

    def forward(
        self,
        layer_id: int,
        inputs: Mapping[int, int],
        *,
        rows: int | None = None,
    ) -> Mapping[int, int]:
        """Run one layer's sharded MLP; return the bf16 output pointer per rank.

        The eager composition of :meth:`enqueue_chain`, the transport
        reduction, and :meth:`cast_reduced`.
        """

        partial_ptrs = self.enqueue_chain(layer_id, inputs, rows=rows)
        rows = self._resolve_rows(rows)
        if self._device_exchange is not None:
            return self._forward_device_reduce(layer_id, partial_ptrs, rows=rows)
        started = time.perf_counter()
        reduced = self._transport.reduce(partial_ptrs, rows=rows)
        self.exchange_walls_s.append(time.perf_counter() - started)
        for device in self.devices:
            self.cast_reduced(device, reduced[device])
        return dict(self._out_ptrs)

    def _forward_device_reduce(
        self,
        layer_id: int,
        partial_ptrs: Mapping[int, int],
        *,
        rows: int,
    ) -> Mapping[int, int]:
        """Reduce the partials on the device into the bf16 boundary buffer.

        The counterpart of the staged route's host sum plus cast, with both of
        those removed: the spin-add kernel sums this rank's partial with the
        peer's staged partial in bf16 and writes the boundary buffer directly,
        so nothing is read back over PCIe and no cast kernel is enqueued.

        Each layer bumps the step counter before its exchange. Two slots are
        reused, so without the bump the peer's published flag would already
        satisfy the spin's comparison and the kernel would sum the previous
        layer's staging. The timeout flags are reset once per group (see
        ``begin_device_group``) rather than per layer, so a timeout in any layer
        is still visible to the final ``wait``.
        """

        exchange = self._device_exchange
        assert exchange is not None
        if int(rows) > int(self.rows):
            raise ShardGroupError(
                "the device-side reduction stages a fixed rows x hidden block, "
                f"so it cannot exceed the group capacity ({self.rows} rows), got {rows}"
            )
        # The exchange was built for the group's capacity, so a shorter active
        # count still stages and reduces the whole block. That is correct (each
        # element is independent and the partial buffer is capacity-sized, so
        # the tail is allocated memory the consumer never reads) and only costs
        # the unused fraction. The bulk prefill sizes its workspace to the
        # prompt, so the two agree on the common path.
        exchange.bump()
        slot = int(layer_id) % 2
        for rank, device in enumerate(self.devices):
            exchange.enqueue_rank(
                rank, int(partial_ptrs[device]), slot, self._out_ptrs[device]
            )
        return dict(self._out_ptrs)

    def begin_device_group(self) -> None:
        """Clear the device exchange's spin-timeout flags, once per prefill.

        Called before a group of layers rather than per layer: the flags are
        sticky on purpose, because the spin kernel writes nothing when it times
        out and a cleared flag would let a stale boundary row pass as a result.
        """

        if self._device_exchange is not None:
            self._device_exchange.reset_timeouts()

    def finish_device_group(self) -> None:
        """Sync both ranks once at the end of a device-reduced group.

        The device route enqueues every layer without a host wait, so the group
        needs one wait at its end; it also surfaces a spin timeout as a failure
        instead of leaving a stale row in a boundary buffer.
        """

        if self._device_exchange is not None:
            self._device_exchange.wait()

    def enqueue_chain(
        self,
        layer_id: int,
        inputs: Mapping[int, int],
        *,
        rows: int | None = None,
    ) -> Mapping[int, int]:
        """Enqueue both ranks' shard chains; return the down partial per rank.

        ``inputs`` maps device -> the device-resident bf16 post-attention-norm
        rows that rank's shard consumes. Stream-ordered and
        host-synchronization-free: this is the capturable unit a graphed
        schedule captures (the transport reduction stays host-driven between
        graph segments).
        """

        self._require_live()
        rows = self._resolve_rows(rows)
        if int(layer_id) not in {
            layer for layer, _device in self._ranks
        }:
            raise ShardGroupError(f"the group has no layer {layer_id}")
        layer_ranks = {
            device: self._ranks[(int(layer_id), device)] for device in self.devices
        }
        missing = [d for d in self.devices if int(d) not in inputs]
        if missing:
            raise ShardGroupError(f"no input given for ranks {missing}")
        partial_ptrs: dict[int, int] = {}
        try:
            for device in self.devices:
                layer_ranks[device].write_input_from_device(int(inputs[device]), rows=rows)
            for device in self.devices:
                partial_ptrs[device] = layer_ranks[device].forward_partial(rows=rows)
        except TransportError:
            raise
        except Exception as error:  # noqa: BLE001 - fail the group, not the rank
            raise ShardGroupError(
                f"layer {layer_id} shard chain failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        return partial_ptrs

    def enqueue_rank_chain(
        self,
        layer_id: int,
        device: int,
        input_ptr: int,
        *,
        rows: int | None = None,
    ) -> int:
        """Enqueue one rank's shard chain; return its down partial pointer.

        The per-rank building block of :meth:`enqueue_chain`, used by graphed
        schedules that capture each rank's chain inside its own layer graph.
        """

        self._require_live()
        rows = self._resolve_rows(rows)
        rank = self._ranks.get((int(layer_id), int(device)))
        if rank is None:
            raise ShardGroupError(f"the group has no layer {layer_id} on device {device}")
        try:
            rank.write_input_from_device(int(input_ptr), rows=rows)
            return rank.forward_partial(rows=rows)
        except TransportError:
            raise
        except Exception as error:  # noqa: BLE001 - fail the group, not the rank
            raise ShardGroupError(
                f"layer {layer_id} rank {device} shard chain failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    def reduce_partials(
        self,
        partial_ptrs: Mapping[int, int],
        *,
        slot: int | None = None,
        rows: int | None = None,
    ) -> Mapping[int, int]:
        """Reduce the given down partials; return the reduced pointer per rank.

        With ``slot`` the reduction publishes into that fixed payload slot
        (the graphed schedule's stable per-layer slot); the eager schedule's
        two-slot alternation is untouched.
        """

        self._require_live()
        rows = self._resolve_rows(rows)
        started = time.perf_counter()
        reduced = self._transport.reduce(partial_ptrs, slot=slot, rows=rows)
        self.exchange_walls_s.append(time.perf_counter() - started)
        return reduced

    def reduced_payload_ptr(self, slot: int) -> int:
        """The fixed mapped reduced-row address for one payload slot.

        Compiled-transport only: this is the stable pointer a captured
        graph's deferred consumer reads.
        """

        self._require_live()
        payload_ptr = getattr(self._transport, "payload_ptr", None)
        if payload_ptr is None:
            raise ShardGroupError(
                "fixed payload slots need the compiled staged-exchange driver"
            )
        return int(payload_ptr(slot))

    def reset_exchange_walls(self) -> None:
        """Drop the recorded exchange walls (capture-time bookkeeping)."""

        self.exchange_walls_s.clear()

    def _resolve_rows(self, rows: int | None) -> int:
        """Resolve a caller row count to capacity-checked int (None = capacity)."""

        return require_rows_value(self.rows if rows is None else rows, capacity=self.rows)

    def cast_reduced(self, device: int, reduced_ptr: int) -> None:
        """Cast one rank's reduced f32 rows into the group's bf16 boundary buffer.

        The full capacity is cast: the transport zeroes the inactive tail of
        its reduced payload, so the output buffer's trailing rows are written as
        explicit zeros rather than left holding a stale previous chunk. A
        consumer that reads ``rows * hidden`` still sees only active data.
        """

        self._require_live()
        # Imported lazily from the convert package (whose __init__ re-exports
        # the launcher) so test fakes can patch that binding - a module-top
        # import here would bypass them.
        from hipengine.kernels.hip_gfx1100.convert import (  # noqa: PLC0415
            f32_to_bf16,
        )

        with scoped_current_device(self._runtime, device):
            f32_to_bf16(
                int(reduced_ptr),
                self._out_ptrs[int(device)],
                self.rows * self.hidden,
                stream=self._streams[int(device)],
                runtime=self._runtime,
            )

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Free every rank's buffers and weights, exactly once."""

        if self._closed:
            return
        self._closed = True
        self._transport.close()
        if self._device_exchange is not None:
            self._device_exchange.close()
            self._device_exchange = None
        for rank in self._ranks.values():
            rank.close()
        for device, ptr in self._out_ptrs.items():
            with scoped_current_device(self._runtime, device):
                self._runtime.free(ptr)
        self._ranks.clear()

    def __enter__(self) -> "MlpShardGroup":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _require_live(self) -> None:
        if self._closed:
            raise ShardGroupError("MlpShardGroup is closed; its buffers are freed")
