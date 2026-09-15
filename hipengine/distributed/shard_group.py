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
from hipengine.distributed.shard_exec import MlpShardRank
from hipengine.distributed.staged import StagedExchangeTransport
from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport
from hipengine.distributed.transport import TransportError


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
        per_rank_ffn: int,
        weights: Mapping[int, Mapping[int, Mapping[str, Any]]],
        staging_dtype: str = "f32",
        driver: str = "compiled",
        mlp_decode_variant: str | None = None,
    ) -> None:
        if not weights:
            raise ShardGroupError("a shard group needs at least one layer")
        if driver not in {"python", "compiled"}:
            raise ShardGroupError(
                f"unknown exchange driver {driver!r}; expected 'python' or 'compiled'"
            )
        self._runtime = runtime
        self.devices = tuple(int(d) for d in devices)
        self.hidden = int(hidden)
        self.per_rank_ffn = int(per_rank_ffn)
        self._streams = {int(d): int(streams[d]) for d in self.devices}
        self._closed = False
        self._mlp_decode_variant = (
            str(mlp_decode_variant) if mlp_decode_variant is not None else None
        )
        self.exchange_walls_s: list[float] = []

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
                    per_rank_ffn=per_rank_ffn,
                    partial_dtype=staging_dtype,
                    mlp_decode_variant=mlp_decode_variant,
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
                )
            )
        else:
            self._transport = StagedExchangeTransport(
                runtime,
                devices=self.devices,
                streams=self._streams,
                hidden=hidden,
                staging_dtype=staging_dtype,
            )
        # The bf16 boundary buffer per rank: the reduced f32 value cast to the
        # dtype the TP1 local chain's down projection writes, so the caller's
        # residual add consumes one identical contract on either path.
        self._out_ptrs: dict[int, int] = {}
        for device in self.devices:
            with scoped_current_device(runtime, device):
                self._out_ptrs[device] = int(runtime.malloc(self.hidden * 2))

    # -- state -------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return self._transport.poisoned

    @property
    def reductions(self) -> int:
        return self._transport.reductions

    @property
    def mlp_decode_variant(self) -> str | None:
        """The fused gate/up+SiLU variant every rank resolves, or None."""

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
    ) -> Mapping[int, int]:
        """Run one layer's sharded MLP; return the bf16 output pointer per rank.

        ``inputs`` maps device -> the device-resident bf16 post-attention-norm
        row that rank's shard consumes. Both ranks' chains are enqueued before
        either is awaited; the exchange's per-stream wait is the only host
        synchronization, and every returned pointer is stream-ordered after
        its H2D on that rank's stream.
        """

        self._require_live()
        if int(layer_id) not in {layer for layer, _device in self._ranks}:
            raise ShardGroupError(f"the group has no layer {layer_id}")
        layer_ranks = {device: self._ranks[(int(layer_id), device)] for device in self.devices}
        missing = [d for d in self.devices if int(d) not in inputs]
        if missing:
            raise ShardGroupError(f"no input given for ranks {missing}")
        partial_ptrs: dict[int, int] = {}
        try:
            for device in self.devices:
                layer_ranks[device].write_input_from_device(int(inputs[device]))
            for device in self.devices:
                partial_ptrs[device] = layer_ranks[device].forward_partial()
        except TransportError:
            raise
        except Exception as error:  # noqa: BLE001 - fail the group, not the rank
            raise ShardGroupError(
                f"layer {layer_id} shard chain failed: {type(error).__name__}: {error}"
            ) from error

        started = time.perf_counter()
        reduced = self._transport.reduce(partial_ptrs)
        self.exchange_walls_s.append(time.perf_counter() - started)

        from hipengine.kernels.hip_gfx1100.convert import f32_to_bf16  # noqa: PLC0415

        for device in self.devices:
            with scoped_current_device(self._runtime, device):
                f32_to_bf16(
                    reduced[device],
                    self._out_ptrs[device],
                    self.hidden,
                    stream=self._streams[device],
                    runtime=self._runtime,
                )
        return dict(self._out_ptrs)

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Free every rank's buffers and weights, exactly once."""

        if self._closed:
            return
        self._closed = True
        self._transport.close()
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
