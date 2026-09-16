"""Device-resident MLP shard execution for one TP rank.

This is the correctness-baseline shard chain from
``scripts/tp2_mlp_slice_e2e.py`` (TP2-A, validated against an independent CPU
oracle), moved into a reusable runtime component. One instance owns one
rank's shard: its resident weight payloads, and persistent activation buffers
that are allocated once at construction and reused by every decode step -
never allocated per call.

The chain per rank is the incumbent unfused route (gate GEMV, up GEMV,
SiLU-multiply, down GEMV writing f32 partials) or the fused gate/up+SiLU
candidate at the shard shape. The down GEMV's output is a *partial* - the
rank's contribution to the layer's output - and it stays device-resident on
this rank's stream; the caller reduces it across ranks (for example through
:class:`hipengine.distributed.staged.StagedExchangeTransport`) before the
residual/norm consumer runs. Nothing here reads hidden state back to the
host.

Ownership: every launch, buffer, and free belongs to this rank's device and
runs under a scoped current-device selection. Weights and buffers are freed
exactly once by :meth:`MlpShardRank.close`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hipengine.core.device import Device, scoped_current_device
from hipengine.core.memory import copy_host_to_device
from hipengine.distributed.transport import require_rows_value
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_separate_out_bf16,
)


class MlpShardRankError(RuntimeError):
    """A requested batched shard route is unsupported; fail before launch."""


class ShardWeightSpec:
    """The weight-spec surface ``launch_gguf_linear`` consumes.

    ``layout`` and ``quant_key`` come from the shard materialization plan -
    never from a table here - so the resident payload is always what the
    planner produced for this rank.
    """

    def __init__(self, layout: str, quant_key: str, allocation_names: tuple[str, ...]):
        self.layout = layout
        self.quant_key = quant_key
        self.allocation_names = tuple(allocation_names)
        self.allocations = set(self.allocation_names)


class ShardWeightAllocation:
    """A persistent device allocation holding one rank's resident payload."""

    def __init__(self, name: str, runtime: Any, device: int, payload: np.ndarray):
        self.name = name
        self.nbytes = int(payload.nbytes)
        self._device = int(device)
        self._runtime = runtime
        with scoped_current_device(runtime, device):
            self.buffer = int(runtime.malloc(self.nbytes))
        self._host = np.ascontiguousarray(payload)
        copy_host_to_device(
            _DeviceBufferProxy(self.buffer, self.nbytes, device),
            self._host.ctypes.data,
            self.nbytes,
            runtime=runtime,
        )
        self.tensor = _TensorProxy(self.buffer, payload.shape)

    def free(self) -> None:
        with scoped_current_device(self._runtime, self._device):
            self._runtime.free(self.buffer)


class _DeviceBufferProxy:
    """The buffer surface ``copy_host_to_device`` needs."""

    def __init__(self, ptr: int, nbytes: int, device: int):
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)
        self.device = Device("hip", int(device))


class _TensorProxy:
    def __init__(self, ptr: int, shape: tuple[int, ...]):
        self.ptr = int(ptr)
        self.shape = tuple(shape)


class ShardWeight:
    """A GGUFDeviceWeight stand-in over one rank's resident payload."""

    def __init__(self, layout: str, quant_key: str, allocation: ShardWeightAllocation):
        self.spec = ShardWeightSpec(layout, quant_key, (allocation.name,))
        self.backend = "hip_gfx1100"
        self._allocation = allocation

    def allocation(self, name: str | None = None) -> ShardWeightAllocation:
        if name is not None and name != self._allocation.name:
            raise KeyError(f"shard weight has allocation {self._allocation.name!r}, not {name!r}")
        return self._allocation


def upload_shard_weight(
    runtime: Any,
    *,
    device: int,
    name: str,
    layout: str,
    quant_key: str,
    payload: np.ndarray,
) -> ShardWeight:
    """Upload one rank's shard payload as a persistent resident weight."""

    allocation = ShardWeightAllocation(name, runtime, device, payload)
    return ShardWeight(layout, quant_key, allocation)


class MlpShardRank:
    """One rank's persistent MLP shard execution state.

    Activations are allocated once: the bf16 input row, the gate/up/activated
    intermediates, and the f32 down partial. ``forward_partial`` enqueues the
    chain on this rank's stream and returns the device pointer of the f32
    partial (``hidden`` floats); no host synchronization happens inside, so a
    driver can overlap this rank's chain with the other rank's.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        device: int,
        stream: int,
        weights: Mapping[str, ShardWeight],
        hidden: int,
        per_rank_ffn: int,
        partial_dtype: str = "f32",
        mlp_decode_variant: str | None = None,
        rows: int = 1,
    ) -> None:
        if partial_dtype not in {"f32", "bf16"}:
            raise ValueError(
                f"unsupported partial dtype {partial_dtype!r}; expected 'f32' or 'bf16'"
            )
        self._runtime = runtime
        self.device = int(device)
        self.stream = int(stream)
        self.hidden = int(hidden)
        self.per_rank_ffn = int(per_rank_ffn)
        # ``rows`` is the buffer capacity; ``active_rows`` is how many rows the
        # most recent input staging actually filled. Every consumer must
        # either use exactly the active count or be rejected before launch.
        if isinstance(rows, bool) or not isinstance(rows, int):
            raise ValueError(
                f"rows must be an integer, got {type(rows).__name__} {rows!r}"
            )
        self.rows = rows
        if self.rows < 1:
            raise ValueError(f"rows must be positive, got {rows!r}")
        self.active_rows = 1
        self.partial_dtype = str(partial_dtype)
        self.partial_itemsize = 4 if partial_dtype == "f32" else 2
        # The shape-qualified fused gate/up+SiLU variant the policy table
        # resolves for this rank's shard shape, or None for the unfused
        # chain. The policy lookup happens where the model identity lives
        # (the session over the resident runner); the rank only executes
        # what it is told and falls back to unfused when no variant
        # resolves.
        self.mlp_decode_variant = (
            str(mlp_decode_variant) if mlp_decode_variant is not None else None
        )
        self._closed = False
        self._weights: dict[str, ShardWeight] = dict(weights)
        for role in ("ffn_gate", "ffn_up", "ffn_down"):
            if role not in self._weights:
                raise ValueError(f"MLP shard weights are missing {role!r}")

        # Persistent activations, allocated once for up to ``rows`` rows:
        # the bf16 input, bf16 gate/up/act intermediates, and the down
        # partial the exchange reduces from, in this rank's partial dtype.
        self.x_ptr = self._alloc(self.rows * self.hidden * 2)
        self.gate_ptr = self._alloc(self.rows * self.per_rank_ffn * 2)
        self.up_ptr = self._alloc(self.rows * self.per_rank_ffn * 2)
        self.act_ptr = self._alloc(self.rows * self.per_rank_ffn * 2)
        self.down_partial_ptr = self._alloc(
            self.rows * self.hidden * self.partial_itemsize
        )

    # -- execution --------------------------------------------------------

    def write_input(self, x_bf16_bytes: np.ndarray, rows: int | None = None) -> None:
        """Stage one or more bf16 input rows into this rank's input buffer.

        ``rows`` defaults to the payload's own row count (a ``hidden * 2``
        byte payload is one row). The byte count must match exactly and fit
        the rank's capacity; ``active_rows`` is then fixed so a later
        ``forward_partial`` cannot silently run a different row count.
        """

        self._require_live()
        payload = np.ascontiguousarray(x_bf16_bytes, dtype=np.uint8).reshape(-1)
        if payload.size % (self.hidden * 2) != 0:
            raise ValueError(
                f"input payload is {payload.size} bytes, not a whole number of "
                f"{self.hidden * 2}-byte rows"
            )
        payload_rows = payload.size // (self.hidden * 2)
        rows = require_rows_value(
            payload_rows if rows is None else rows, capacity=self.rows
        )
        if rows != payload_rows:
            raise ValueError(
                f"input payload has {payload_rows} rows but rows={rows} was declared"
            )
        from hipengine.core.memory import copy_host_to_device  # noqa: PLC0415

        copy_host_to_device(
            _DeviceBufferProxy(self.x_ptr, payload.size, self.device),
            payload.ctypes.data,
            payload.size,
            runtime=self._runtime,
        )
        self.active_rows = rows

    def write_input_from_device(self, src_ptr: int, rows: int | None = None) -> None:
        """Copy up to ``rows`` bf16 rows from a device buffer on this rank.

        The producer is a kernel on this rank's device (for example the
        replicated post-attention norm's output rows), so the copy is a
        same-device D2D on this rank's own stream: no host round trip, no
        cross-device visibility assumed. The copy is stream-ordered behind
        that producer and ahead of ``forward_partial``.
        """

        self._require_live()
        rows = self._resolve_rows(rows)
        nbytes = rows * self.hidden * 2
        from hipengine.core.runtime import MemcpyKind  # noqa: PLC0415

        with scoped_current_device(self._runtime, self.device):
            self._runtime.memcpy_async(
                self.x_ptr,
                int(src_ptr),
                nbytes,
                MemcpyKind.DEVICE_TO_DEVICE,
                self.stream,
            )
        self.active_rows = rows

    def _resolve_rows(self, rows: int | None) -> int:
        """Resolve an explicit row count, or the active count, to capacity-checked int."""

        selected = self.active_rows if rows is None else rows
        return require_rows_value(selected, capacity=self.rows)

    def forward_partial(
        self,
        *,
        fused: bool | None = None,
        fused_variant: str | None = None,
        rows: int | None = None,
    ) -> int:
        """Enqueue this rank's MLP chain; return the down partial's device pointer.

        The chain is stream-ordered behind whatever the caller enqueued on
        this rank's stream (including ``write_input``'s copy), and the
        returned pointer is the producer for the cross-rank reduction. The
        partial's dtype is this rank's ``partial_dtype``: f32 where the
        layer's registered down consumer admits it, bf16 where it does not.

        ``rows`` selects a batched (prefill) chain: the same sharded weights
        and the same full-hidden partial, launched as a GEMM over ``rows``
        rows instead of a single-row GEMV. Batched prefill always uses the
        unfused gate/up/SiLU/down chain; the fused decode candidate is a
        single-row route.

        The gate/up+SiLU route: an explicitly passed ``fused`` wins; by
        default the rank runs the fused pair when its construction-time
        ``mlp_decode_variant`` resolved from the policy table, else the
        unfused chain. With ``fused=True`` a variant is required - the
        explicit-candidate path never guesses.
        """

        self._require_live()
        rows = self._resolve_rows(rows)
        if rows != self.active_rows:
            raise ValueError(
                f"forward_partial rows {rows} does not match the {self.active_rows} "
                f"rows staged by write_input; stage and run the same row count"
            )
        if rows > 1:
            self._require_batched_route(rows)
        if fused is None:
            fused = self.mlp_decode_variant is not None
        if rows > 1:
            fused = False
        if fused and fused_variant is None:
            fused_variant = self.mlp_decode_variant
        if fused and fused_variant is None:
            raise ValueError("the fused chain needs its registered variant name")
        use_gemv_decode = rows == 1
        runtime = self._runtime
        from hipengine.runtime.gguf_linear import (  # noqa: PLC0415
            launch_gguf_linear,
        )

        with scoped_current_device(runtime, self.device):
            if fused:
                from hipengine.runtime.gguf_linear import (  # noqa: PLC0415
                    launch_gguf_linear_pair_silu,
                )

                launched = launch_gguf_linear_pair_silu(
                    self._weights["ffn_gate"],
                    self._weights["ffn_up"],
                    self.x_ptr,
                    self.act_ptr,
                    rows,
                    self.hidden,
                    self.per_rank_ffn,
                    use_gemv_decode=use_gemv_decode,
                    registered_decode_variant=fused_variant,
                    stream=self.stream,
                    runtime=runtime,
                )
                if not launched:
                    raise RuntimeError(
                        f"the fused pair+SiLU candidate did not launch at "
                        f"({rows}, {self.hidden}, {self.per_rank_ffn})"
                    )
            else:
                launch_gguf_linear(
                    self._weights["ffn_gate"],
                    self.x_ptr,
                    self.gate_ptr,
                    rows,
                    self.hidden,
                    self.per_rank_ffn,
                    use_gemv_decode=use_gemv_decode,
                    stream=self.stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    self._weights["ffn_up"],
                    self.x_ptr,
                    self.up_ptr,
                    rows,
                    self.hidden,
                    self.per_rank_ffn,
                    use_gemv_decode=use_gemv_decode,
                    stream=self.stream,
                    runtime=runtime,
                )
                silu_mul_separate_out_bf16(
                    self.gate_ptr,
                    self.up_ptr,
                    self.act_ptr,
                    rows,
                    self.per_rank_ffn,
                    stream=self.stream,
                    runtime=runtime,
                )
            down_kwargs = {
                "use_gemv_decode": use_gemv_decode,
                "stream": self.stream,
                "runtime": runtime,
            }
            if self.partial_dtype == "f32":
                down_kwargs["output_dtype"] = "f32"
            launch_gguf_linear(
                self._weights["ffn_down"],
                self.act_ptr,
                self.down_partial_ptr,
                rows,
                self.per_rank_ffn,
                self.hidden,
                **down_kwargs,
            )
        return self.down_partial_ptr

    def synchronize(self) -> None:
        """Wait for this rank's stream (request/transaction boundaries only)."""

        self._require_live()
        with scoped_current_device(self._runtime, self.device):
            self._runtime.stream_synchronize(self.stream)

    def _require_batched_route(self, rows: int) -> None:
        """Fail before launch when the shard quant has no declared batched route.

        This checks that the dispatch surface admits a multi-row launch for the
        weight's layout/activation/output combination (an unsupported
        combination raises there). It does **not** by itself prove the resolved
        leaf is rows-aware: the real launcher rewrites to ``t16_wmma_prefill`` /
        ``t16_gemv_rowtile`` leaves that the surface table does not name, so the
        empirical per-row GPU probe is the rows-aware validation. What this
        prevents is launching a multi-row buffer into a route the surface does
        not admit at all.
        """

        from hipengine.runtime.gguf_linear import (  # noqa: PLC0415
            resolve_gguf_linear_dispatch,
        )

        output_dtype = "f32" if self.partial_dtype == "f32" else "bf16"
        for role in ("ffn_gate", "ffn_up", "ffn_down"):
            weight = self._weights[role]
            try:
                resolve_gguf_linear_dispatch(
                    weight, output_dtype=output_dtype, rows=rows
                )
            except Exception as error:  # noqa: BLE001 - surface the route failure
                raise MlpShardRankError(
                    f"no batched ({rows}-row) {output_dtype}-output route for "
                    f"{weight.spec.layout}/{weight.spec.quant_key} ({role}): "
                    f"{type(error).__name__}: {error}"
                ) from error

    def read_partial(self, rows: int | None = None) -> np.ndarray:
        """Read the down partial back (diagnostics only, never the serving path)."""

        self._require_live()
        rows = self._resolve_rows(rows)
        if rows != self.active_rows:
            raise ValueError(
                f"read_partial rows {rows} does not match the {self.active_rows} "
                f"rows staged by write_input"
            )
        from hipengine.core.memory import copy_device_to_host  # noqa: PLC0415

        nbytes = rows * self.hidden * self.partial_itemsize
        out = np.empty(nbytes, dtype=np.uint8)
        with scoped_current_device(self._runtime, self.device):
            copy_device_to_host(
                out.ctypes.data,
                _DeviceBufferProxy(self.down_partial_ptr, nbytes, self.device),
                nbytes,
                runtime=self._runtime,
            )
        if self.partial_dtype == "f32":
            values = out.view("<f4")
        else:
            bits = out.view("<u2").astype(np.uint32) << 16
            values = bits.view(np.float32)
        return values if rows == 1 else values.reshape(rows, self.hidden)

    def read_input(self, rows: int | None = None) -> np.ndarray:
        """Read the staged bf16 input rows back (diagnostics only)."""

        self._require_live()
        rows = self._resolve_rows(rows)
        if rows != self.active_rows:
            raise ValueError(
                f"read_input rows {rows} does not match the {self.active_rows} "
                f"rows staged by write_input"
            )
        from hipengine.core.memory import copy_device_to_host  # noqa: PLC0415

        nbytes = rows * self.hidden * 2
        out = np.empty(nbytes, dtype=np.uint8)
        with scoped_current_device(self._runtime, self.device):
            copy_device_to_host(
                out.ctypes.data,
                _DeviceBufferProxy(self.x_ptr, nbytes, self.device),
                nbytes,
                runtime=self._runtime,
            )
        return out

    # -- teardown ---------------------------------------------------------

    def close(self) -> None:
        """Free this rank's persistent buffers and weights, exactly once."""

        if self._closed:
            return
        self._closed = True
        for ptr in (self.x_ptr, self.gate_ptr, self.up_ptr, self.act_ptr, self.down_partial_ptr):
            with scoped_current_device(self._runtime, self.device):
                self._runtime.free(ptr)
        for weight in self._weights.values():
            weight.allocation().free()

    def __enter__(self) -> "MlpShardRank":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _alloc(self, nbytes: int) -> int:
        with scoped_current_device(self._runtime, self.device):
            return int(self._runtime.malloc(nbytes))

    def _require_live(self) -> None:
        if self._closed:
            raise RuntimeError("MlpShardRank is closed; its buffers are freed")
