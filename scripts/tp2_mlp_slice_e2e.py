"""TP2-A: execute the MLP shard slice on two GPUs and validate it against an
independent CPU oracle.

This is the correctness baseline for the TP2 MLP design. On each rank's own
device it uploads that rank's shard payloads (the same split the incumbent shard
planner produces, repacked into the same resident t16 layouts), runs the unfused
chain the route resolves to today - gate GEMV, up GEMV, SiLU-multiply, down GEMV
- and copies every intermediate back. On device 0 it also runs the TP1 teacher:
the incumbent fused gate/up+SiLU route at its admitted shape plus the full down
GEMV.

Nothing here is compared against itself. The oracle is computed on the host from
the dequantized GGUF weights by plain matrix arithmetic:

* a float64 *truth* chain, ``down(silu(gate @ x) * (up @ x))``, which is what the
  bf16 slice approximates;
* a *contract* emulation per rank, which rounds each intermediate to bf16 exactly
  where the device chain writes bf16, so a device partial should land on it
  within accumulation-order noise;
* the per-rank partials and their f32 sum, which is what a real reduction sees.

The artifact records every metric for every stage. The gate is the production
numerical envelope, not exactness: mean/tail/max relative error of the summed
TP2 output against the float64 truth and against the TP1 teacher, with the
per-rank partials pinned to the bf16 contract emulation.

Run:

    uv run python3 scripts/tp2_mlp_slice_e2e.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --layer 0 \
        --json benchmarks/results/2026-09-15-w7900-tp2-mlp-slice-e2e.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tp2_mlp_shard_plan_probe import (  # noqa: E402
    MLP_TENSORS,
    _spec_for_slot,
    _t16_repack_for,
    _t16_repack_tiles,
    fused_path_admission,
    resolve_incumbent_plan,
)

#: Full tensor names for one block's MLP, in planner order.
def mlp_tensor_names(layer: int) -> tuple[str, ...]:
    return tuple(f"blk.{int(layer)}.{name}" for name in MLP_TENSORS)


def mlp_roles() -> tuple[str, ...]:
    """Slot roles, e.g. ``ffn_gate`` from ``ffn_gate.weight``."""

    return tuple(name.split(".")[0] for name in MLP_TENSORS)
from hipengine.core.device import Device, scoped_current_device  # noqa: E402
from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.runtime import MemcpyKind  # noqa: E402
from hipengine.core.memory import copy_host_to_device, copy_device_to_host  # noqa: E402
from hipengine.kernels.cpu_reference.maple import f32_to_bf16_bits  # noqa: E402
from hipengine.kernels.cpu_reference.ops import rmsnorm  # noqa: E402
from hipengine.loading.gguf import GGUFReader, scan_gguf  # noqa: E402
from hipengine.loading.qwen35_gguf import (  # noqa: E402
    build_qwen35_gguf_tensor_map,
    qwen35_gguf_config_from_metadata,
)
from hipengine.loading.qwen35_gguf_admission import (  # noqa: E402
    build_qwen35_gguf_role_manifest,
)
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    materialize_qwen35_gguf_weights,
)
from hipengine.loading.qwen35_gguf_shards import (  # noqa: E402
    build_shard_manifest,
    materialize_slice,
    source_payload,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (  # noqa: E402
    silu_mul_separate_out_bf16,
)
from hipengine.quant.gguf import bf16_to_float32, dequantize_gguf_data  # noqa: E402
from hipengine.runtime.gguf_linear import (  # noqa: E402
    launch_gguf_linear,
    launch_gguf_linear_pair_silu,
)

#: The absolute floor under which a relative error is not computed: bf16 has
#: about three significant decimal digits, so differences below 1e-3 of a value
#: near zero are quantization, not arithmetic.
_REL_FLOOR = 1e-3


def silu_f32(values: np.ndarray) -> np.ndarray:
    """SiLU in f32: x * sigmoid(x), computed stably."""

    x = np.asarray(values, dtype=np.float32)
    # sigmoid(x) = 1 / (1 + exp(-x)); exp overflows harmlessly to inf for large
    # negative x, and 1/inf is 0, which is the correct limit.
    return (x / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def bf16_round(values: np.ndarray) -> np.ndarray:
    """Round f32 to bf16 precision, returning f32 (round to nearest even)."""

    bits = f32_to_bf16_bits(values)
    return bf16_to_float32(bits)


def bf16_bytes(values: np.ndarray) -> np.ndarray:
    """The bf16 byte payload of an f32 array, little-endian uint16."""

    return f32_to_bf16_bits(values).astype("<u2").tobytes()


def residual_boundary(
    residual_f32: np.ndarray,
    mlp_out: np.ndarray,
    norm_weight: np.ndarray,
) -> dict[str, np.ndarray]:
    """The next block's input boundary: residual add, then input RMSNorm.

    This is what a replicated next layer consumes on both ranks. The add runs in
    f32 - the incumbent f32-residual decode path - and the norm is the CPU
    reference, so a TP2 boundary is compared against a TP1 boundary through the
    same arithmetic.
    """

    next_hidden = (np.asarray(residual_f32, dtype=np.float32) + np.asarray(mlp_out, dtype=np.float32)).astype(np.float32)
    return {
        "next_hidden": next_hidden,
        "next_norm": rmsnorm(next_hidden, norm_weight).astype(np.float32),
    }


def _relative_errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64).ravel()
    e = np.asarray(expected, dtype=np.float64).ravel()
    diff = np.abs(a - e)
    denom = np.maximum(np.abs(e), _REL_FLOOR)
    rel = diff / denom
    return {
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "p99_abs_err": float(np.quantile(diff, 0.99)),
        "max_rel_err": float(rel.max()),
        "mean_rel_err": float(rel.mean()),
        "p99_rel_err": float(np.quantile(rel, 0.99)),
    }


# ---------------------------------------------------------------------------
# Host oracle
# ---------------------------------------------------------------------------


def dequantized_mlp(reader: GGUFReader, layer: int) -> dict[str, np.ndarray]:
    """Dequantize the layer's gate/up/down payloads to f32 [out, in] arrays."""

    arrays: dict[str, np.ndarray] = {}
    for role in mlp_roles():
        name = f"blk.{int(layer)}.{role}.weight"
        info = reader.tensor_info(name)
        payload = source_payload(
            reader.path, data_offset=info.data_offset, nbytes=info.nbytes
        )
        rows = int(info.shape[0])
        columns = int(np.prod(info.shape[1:])) if len(info.shape) > 1 else 1
        block = dequantize_gguf_data(payload.reshape(rows, -1), info.ggml_type)
        arrays[role] = np.asarray(block, dtype=np.float32).reshape(rows, columns)
    return arrays


def full_truth(
    gate: np.ndarray, up: np.ndarray, down: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """The float64 MLP truth for one token: down(silu(gate @ x) * (up @ x))."""

    g = gate.astype(np.float64) @ x.astype(np.float64)
    u = up.astype(np.float64) @ x.astype(np.float64)
    a = (g / (np.float64(1.0) + np.exp(-g))) * u
    return (down.astype(np.float64) @ a).astype(np.float32)


def contract_rank_partials(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    x_bf16: np.ndarray,
    rank: int,
    world_size: int,
    *,
    down_output_dtype: str = "f32",
) -> dict[str, np.ndarray]:
    """One rank's partials with bf16 rounding exactly where the device writes it.

    Gate/up are column-split on their output axis; the down projection is
    row-split on its input axis. Each intermediate is rounded to bf16 before it
    crosses a kernel boundary, because that is what the device chain writes.
    """

    hidden = gate.shape[1]
    ffn = gate.shape[0]
    if ffn % int(world_size):
        raise ValueError(f"ffn {ffn} does not split across {world_size} ranks")
    columns = slice(rank * (ffn // int(world_size)), (rank + 1) * (ffn // int(world_size)))
    # The down projection is split on its input axis: each rank consumes its own
    # half of the activated intermediate and produces a full-width partial, so
    # the reduction is a plain sum over ranks at the end.
    if down.shape[1] != ffn:
        raise ValueError(f"down expects {down.shape[1]} inputs, gate produced {ffn}")

    g = bf16_round(gate[columns, :] @ x_bf16)
    u = bf16_round(up[columns, :] @ x_bf16)
    a = bf16_round(silu_f32(g) * u)
    if down_output_dtype == "f32":
        # An f32 partial is the kernel's f32 accumulator written out unrounded;
        # the only rounding left is the one the reduction performs.
        y = (down[:, columns] @ a).astype(np.float32)
    else:
        y = bf16_round(down[:, columns] @ a)
    return {"gate": g, "up": u, "activated": a, "down_partial": y}


def rank_shard_payload(
    reader: GGUFReader,
    materialization: Any,
    plan: Any,
    shard: Any,
    *,
    layer: int,
) -> tuple[np.ndarray, str, str]:
    """One rank's resident-layout payload for one planned MLP tensor.

    The layout and quant key come from the incumbent materialization plan for
    this slot - never from a table here - and the payload is the planner's slice
    repacked into that resident layout, which is what the shard plan probe
    already verified against the device-resident weight.
    """

    slot = plan.name.split(".", 2)[2]
    slot = slot[: -len(".weight")] if slot.endswith(".weight") else slot
    spec = _spec_for_slot(materialization, layer=int(layer), slot=slot)
    layout = str(spec.layout)
    quant_key = str(spec.quant_key)
    info = reader.tensor_info(plan.name)
    source = source_payload(
        reader.path, data_offset=info.data_offset, nbytes=info.nbytes
    )
    repack = _t16_repack_for(layout, str(info.ggml_type_name))
    if repack is None:
        raise SystemExit(
            f"{plan.name}: resident layout {layout!r} is raw and cannot be sliced"
        )
    local = materialize_slice(source, shard)
    if plan.kind == "column":
        rows = int(shard.axis_stop - shard.axis_start)
        bytes_per_row = int(plan.source_row_bytes)
    else:
        rows = int(shard.local_shape[0])
        bytes_per_row = int(shard.local_nbytes) // max(1, int(rows))
    tiles = np.ascontiguousarray(
        _t16_repack_tiles(
            local,
            rows=rows,
            bytes_per_row=bytes_per_row,
            quant_type=str(info.ggml_type_name),
            repack=repack,
        )
    )
    return (
        np.ascontiguousarray(tiles).reshape(-1),
        layout,
        quant_key,
    )


# ---------------------------------------------------------------------------
# Device stand-ins
# ---------------------------------------------------------------------------


class _ShardSpec:
    """The weight-spec surface launch_gguf_linear consumes."""

    def __init__(self, layout: str, quant_key: str, allocation_names: tuple[str, ...]):
        self.layout = layout
        self.quant_key = quant_key
        self.allocation_names = allocation_names
        self.allocations = set(allocation_names)


class _ShardAllocation:
    """A device allocation holding one rank's resident-layout payload."""

    def __init__(self, name: str, runtime: Any, device: int, payload: np.ndarray):
        self.name = name
        self.nbytes = int(payload.nbytes)
        with scoped_current_device(runtime, device):
            self.buffer = int(runtime.malloc(self.nbytes))
        self._device = device
        self._host = np.ascontiguousarray(payload)
        copy_host_to_device(
            _DeviceBufferProxy(self.buffer, self.nbytes, device),
            self._host.ctypes.data,
            self.nbytes,
            runtime=runtime,
        )
        self.tensor = _TensorProxy(self.buffer, payload.shape)

    def free(self, *, runtime: Any = None) -> None:
        with scoped_current_device(runtime, self._device):
            runtime.free(self.buffer)


class _DeviceBufferProxy:
    """The buffer surface copy_host_to_device needs."""

    def __init__(self, ptr: int, nbytes: int, device: int):
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)
        self.device = Device("hip", device)


class _TensorProxy:
    def __init__(self, ptr: int, shape: tuple[int, ...]):
        self.ptr = int(ptr)
        self.shape = tuple(shape)


class _ShardWeight:
    """A GGUFDeviceWeight stand-in over one rank's payload."""

    def __init__(self, layout: str, quant_key: str, allocation: _ShardAllocation):
        self.spec = _ShardSpec(layout, quant_key, (allocation.name,))
        self.backend = "hip_gfx1100"
        self._allocation = allocation

    def allocation(self, name: str | None = None) -> _ShardAllocation:
        if name is not None and name != self._allocation.name:
            raise KeyError(f"shard weight has allocation {self._allocation.name!r}, not {name!r}")
        return self._allocation


# ---------------------------------------------------------------------------
# Device execution
# ---------------------------------------------------------------------------


def _alloc(runtime: Any, device: int, nbytes: int) -> int:
    with scoped_current_device(runtime, device):
        return int(runtime.malloc(int(nbytes)))


def _upload(runtime: Any, device: int, ptr: int, payload: np.ndarray) -> None:
    host = np.ascontiguousarray(payload)
    copy_host_to_device(
        _DeviceBufferProxy(ptr, int(host.nbytes), device),
        host.ctypes.data,
        int(host.nbytes),
        runtime=runtime,
    )


def _download(runtime: Any, device: int, ptr: int, nbytes: int) -> np.ndarray:
    host = (ctypes.c_ubyte * int(nbytes))()
    copy_device_to_host(
        ctypes.addressof(host),
        _DeviceBufferProxy(ptr, int(nbytes), device),
        runtime=runtime,
    )
    return np.frombuffer(bytes(host), dtype=np.uint8)


def run_rank_chain(
    runtime: Any,
    *,
    device: int,
    stream: int,
    weights: dict[str, _ShardWeight],
    x_bf16_bytes: np.ndarray,
    hidden: int,
    per_rank_ffn: int,
    output_dtype: str = "f32",
    keep_down_partial: bool = False,
) -> dict[str, Any]:
    """Run the unfused shard chain on one device.

    Returns every intermediate as raw bytes. With ``keep_down_partial`` the down
    buffer is left allocated and its pointer is returned as ``down_ptr`` - the
    exchange D2Hs from it on this rank's stream - and the caller owns the free.
    """

    out_itemsize = 4 if output_dtype == "f32" else 2
    x_ptr = _alloc(runtime, device, hidden * 2)
    gate_ptr = _alloc(runtime, device, per_rank_ffn * 2)
    up_ptr = _alloc(runtime, device, per_rank_ffn * 2)
    act_ptr = _alloc(runtime, device, per_rank_ffn * 2)
    down_ptr = _alloc(runtime, device, hidden * out_itemsize)
    try:
        _upload(runtime, device, x_ptr, x_bf16_bytes)

        # Every launch, and the stream it is issued on, belongs to this rank's
        # device; the current device must be set around the whole sequence.
        with scoped_current_device(runtime, device):
            launch_gguf_linear(
                weights["ffn_gate"],
                x_ptr,
                gate_ptr,
                1,
                hidden,
                per_rank_ffn,
                use_gemv_decode=True,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                weights["ffn_up"],
                x_ptr,
                up_ptr,
                1,
                hidden,
                per_rank_ffn,
                use_gemv_decode=True,
                stream=stream,
                runtime=runtime,
            )
            silu_mul_separate_out_bf16(
                gate_ptr,
                up_ptr,
                act_ptr,
                1,
                per_rank_ffn,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                weights["ffn_down"],
                act_ptr,
                down_ptr,
                1,
                per_rank_ffn,
                hidden,
                use_gemv_decode=True,
                output_dtype=output_dtype,
                stream=stream,
                runtime=runtime,
            )
            runtime.stream_synchronize(stream)
        output = {
            "gate": _download(runtime, device, gate_ptr, per_rank_ffn * 2),
            "up": _download(runtime, device, up_ptr, per_rank_ffn * 2),
            "activated": _download(runtime, device, act_ptr, per_rank_ffn * 2),
            "down_partial": _download(runtime, device, down_ptr, hidden * out_itemsize),
            "down_output_dtype": output_dtype,
        }
        if keep_down_partial:
            output["down_ptr"] = int(down_ptr)
            output["_down_nbytes"] = hidden * out_itemsize
            return output
        with scoped_current_device(runtime, device):
            runtime.free(down_ptr)
        return output
    finally:
        # down_ptr is intentionally absent here when kept: the caller owns it.
        for ptr in (x_ptr, gate_ptr, up_ptr, act_ptr):
            with scoped_current_device(runtime, device):
                runtime.free(ptr)


def run_rank_chain_fused(
    runtime: Any,
    *,
    device: int,
    stream: int,
    weights: dict[str, _ShardWeight],
    x_bf16_bytes: np.ndarray,
    hidden: int,
    per_rank_ffn: int,
    decode_variant: str,
    down_output_dtype: str = "f32",
) -> dict[str, np.ndarray]:
    """The fused gate/up+SiLU candidate at the shard shape.

    The policy table does not admit (1, hidden, per_rank_ffn), so this is a
    *candidate* run, not a production route: it goes through the same registered
    kernel the TP1 teacher uses, at a shape the kernel's own contract accepts.
    Comparing it against :func:`run_rank_chain` decides whether the candidate is
    numerically equivalent to the unfused chain before anyone touches the policy
    table.
    """

    from hipengine.runtime.gguf_linear import launch_gguf_linear_pair_silu  # noqa: PLC0415

    # The candidate's down GEMV writes the same partial dtype as the baseline
    # chain's reduction contract; the buffer must be sized for that output.
    down_itemsize = 4 if down_output_dtype == "f32" else 2
    x_ptr = _alloc(runtime, device, hidden * 2)
    act_ptr = _alloc(runtime, device, per_rank_ffn * 2)
    down_ptr = _alloc(runtime, device, hidden * down_itemsize)
    try:
        _upload(runtime, device, x_ptr, x_bf16_bytes)
        with scoped_current_device(runtime, device):
            launched = launch_gguf_linear_pair_silu(
                weights["ffn_gate"],
                weights["ffn_up"],
                x_ptr,
                act_ptr,
                1,
                hidden,
                per_rank_ffn,
                use_gemv_decode=True,
                registered_decode_variant=decode_variant,
                stream=stream,
                runtime=runtime,
            )
            if not launched:
                raise RuntimeError(
                    f"the fused pair+SiLU candidate did not launch at "
                    f"(1, {hidden}, {per_rank_ffn})"
                )
            launch_gguf_linear(
                weights["ffn_down"],
                act_ptr,
                down_ptr,
                1,
                per_rank_ffn,
                hidden,
                use_gemv_decode=True,
                output_dtype=down_output_dtype,
                stream=stream,
                runtime=runtime,
            )
            runtime.stream_synchronize(stream)
        return {
            "activated": _download(runtime, device, act_ptr, per_rank_ffn * 2),
            "down_partial": _download(
                runtime, device, down_ptr, hidden * down_itemsize
            ),
        }
    finally:
        for ptr in (x_ptr, act_ptr, down_ptr):
            with scoped_current_device(runtime, device):
                runtime.free(ptr)


class PinnedStaging:
    """A page-locked host buffer one reduction stages its partials through.

    The measured transport stages each rank's partial D2H through pinned host
    memory, sums on the host in f32, and copies the reduced vector H2D back to
    every rank. Page-locking matters because a pageable D2H is staged through a
    driver-owned bounce buffer, which is a second, unmeasured copy on the
    critical path.
    """

    def __init__(self, runtime: Any, nbytes: int):
        self._runtime = runtime
        self.nbytes = int(nbytes)
        self._host = (ctypes.c_ubyte * self.nbytes)()
        self.ptr = ctypes.addressof(self._host)
        runtime.host_register(self.ptr, self.nbytes)

    def view(self) -> np.ndarray:
        return np.frombuffer(self._host, dtype=np.uint8)

    def free(self) -> None:
        self._runtime.host_unregister(self.ptr)

    def __enter__(self) -> "PinnedStaging":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.free()


def staged_exchange_reduce(
    runtime: Any,
    *,
    partial_ptrs: dict[int, int],
    reduced_ptrs: dict[int, int],
    streams: dict[int, int],
    staging: PinnedStaging,
    hidden: int,
    devices: list[int],
    partial_dtype: str = "f32",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Both ranks' partials -> pinned host -> f32 sum -> H2D to every rank.

    Ordering: each rank's D2H is issued on that rank's own stream, so stream
    order places it after the GEMV that produced the partial; the host waits on
    that stream before reading the slot. Each rank's H2D of the reduced vector
    is likewise issued on its stream, so any later work on that rank - the next
    consumer - is ordered after the reduction without a host round trip.

    Returns the reduced f32 vector and the bytes each stage moved, including a
    read-back of every rank's reduced buffer so the H2D is verified, not assumed.

    ``partial_dtype`` is the dtype each rank's down projection wrote: ``f32``
    stages the unrounded accumulator, ``bf16`` stages the rounded partial and
    widens it on the host before the f32 sum. The reduced vector handed back is
    f32 either way - that is what the next consumer's residual add reads.
    """

    if partial_dtype not in {"f32", "bf16"}:
        raise ValueError(f"unsupported partial dtype {partial_dtype!r}")
    slot_nbytes = hidden * (2 if partial_dtype == "bf16" else 4)
    reduced = np.zeros(hidden, dtype=np.float32)
    record: dict[str, Any] = {
        "d2h_bytes": 0,
        "h2d_bytes": 0,
        "verified_ranks": [],
        "protocol": "batched: both D2H submitted before either wait, one wait per stream, H2D submitted without a return wait",
    }
    staging_view = staging.view()
    # Batch the submissions first: issuing both D2H copies before waiting on
    # either is the measured protocol (rank batching), and the host wait covers
    # both copies instead of paying stream latency twice in sequence.
    for device in devices:
        with scoped_current_device(runtime, device):
            runtime.memcpy_async(
                staging.ptr + device * slot_nbytes,
                partial_ptrs[device],
                slot_nbytes,
                MemcpyKind.DEVICE_TO_HOST,
                streams[device],
            )
        record["d2h_bytes"] += slot_nbytes
    for device in devices:
        with scoped_current_device(runtime, device):
            runtime.stream_synchronize(streams[device])
    # One pass over both staging slots instead of a frombuffer per rank: the
    # slots are contiguous, so the sum is a single reduction over a (ranks,
    # hidden) view. bf16 partials widen on the host first; the buffer is
    # already f32 in the f32 case, so no astype is needed there.
    raw = staging_view[: len(devices) * slot_nbytes].tobytes()
    if partial_dtype == "bf16":
        both = bf16_to_float32(
            np.frombuffer(raw, dtype="<u2")
        ).reshape(len(devices), hidden)
    else:
        both = np.frombuffer(raw, dtype="<f4").reshape(len(devices), hidden)
    reduced = both.sum(axis=0, dtype=np.float32)

    payload = np.ascontiguousarray(reduced, dtype="<f4")
    for device in devices:
        # The H2D is issued on the rank's stream without a host wait: the next
        # consumer runs on that stream, so stream order already places it after
        # the reduction, and a host sync here would buy nothing.
        with scoped_current_device(runtime, device):
            runtime.memcpy_async(
                reduced_ptrs[device],
                payload.ctypes.data,
                slot_nbytes,
                MemcpyKind.HOST_TO_DEVICE,
                streams[device],
            )
        record["h2d_bytes"] += slot_nbytes
    return reduced, record


def measure_tp1_segment(
    runtime: Any,
    *,
    resident_gate: Any,
    resident_up: Any,
    resident_down: Any,
    x_ptr: int,
    iterations: int,
    warmup: int,
    hidden: int,
    ffn: int,
    decode_variant: str,
) -> dict[str, Any]:
    """Wall and device time of the TP1 MLP segment: fused pair+SiLU, then down.

    Launches are async and the step is synchronized once, which is how the
    engine drives a decode step; the per-stage numbers come from HIP events on
    the same stream, so they are device time and their sum is comparable to the
    step wall.
    """

    inter_ptr = _alloc(runtime, 0, ffn * 2)
    y_ptr = _alloc(runtime, 0, hidden * 4)
    start, mid, stop = (runtime.event_create() for _ in range(3))
    walls: list[float] = []
    pair_us: list[float] = []
    down_us: list[float] = []
    try:
        with scoped_current_device(runtime, 0):
            for i in range(warmup + iterations):
                t0 = time.perf_counter()
                runtime.event_record(start, 0)
                launched = launch_gguf_linear_pair_silu(
                    resident_gate,
                    resident_up,
                    x_ptr,
                    inter_ptr,
                    1,
                    hidden,
                    ffn,
                    use_gemv_decode=True,
                    registered_decode_variant=decode_variant,
                    runtime=runtime,
                )
                if not launched:
                    raise RuntimeError("the TP1 fused pair+SiLU route did not launch")
                runtime.event_record(mid, 0)
                launch_gguf_linear(
                    resident_down,
                    inter_ptr,
                    y_ptr,
                    1,
                    ffn,
                    hidden,
                    use_gemv_decode=True,
                    output_dtype="f32",
                    runtime=runtime,
                )
                runtime.event_record(stop, 0)
                runtime.event_synchronize(stop)
                wall_us = (time.perf_counter() - t0) * 1e6
                if i >= warmup:
                    walls.append(wall_us)
                    pair_us.append(runtime.event_elapsed_time_ms(start, mid) * 1e3)
                    down_us.append(runtime.event_elapsed_time_ms(mid, stop) * 1e3)
    finally:
        for event in (start, mid, stop):
            runtime.event_destroy(event)
        for ptr in (inter_ptr, y_ptr):
            with scoped_current_device(runtime, 0):
                runtime.free(ptr)
    return {
        "iterations": int(iterations),
        "step_us_p50": float(np.quantile(walls, 0.5)),
        "step_us_mean": float(np.mean(walls)),
        "pair_silu_us_p50": float(np.quantile(pair_us, 0.5)),
        "down_us_p50": float(np.quantile(down_us, 0.5)),
        "device_sum_us_p50": float(np.quantile(np.asarray(pair_us) + np.asarray(down_us), 0.5)),
    }


def measure_tp2_segment(
    runtime: Any,
    *,
    rank_weights: list[dict[str, _ShardWeight]],
    x_ptr_by_rank: dict[int, int],
    down_ptrs: dict[int, int],
    reduced_ptrs: dict[int, int],
    streams: dict[int, int],
    staging: "PinnedStaging",
    hidden: int,
    per_rank_ffn: int,
    fused: bool,
    decode_variant: str | None,
    iterations: int,
    warmup: int,
    partial_dtype: str = "f32",
) -> dict[str, Any]:
    """Wall of the complete TP2 MLP step on both cards.

    One step is: both ranks' chains launched concurrently on their own devices
    and streams, both streams synchronized, then the staged exchange (per-rank
    D2H into pinned host, host f32 sum, H2D to every rank, synchronized). The
    residual add is deliberately excluded: it is the same single 5120-vector add
    on both paths and is not part of the segment under comparison.
    """

    from hipengine.runtime.gguf_linear import launch_gguf_linear_pair_silu  # noqa: PLC0415

    # Separate gate/up/activated slabs per rank: the unfused chain writes two
    # bf16 GEMV outputs and then a third buffer for the SiLU product, mirroring
    # the correctness chain's layout.
    gate_ptrs = {r: _alloc(runtime, r, per_rank_ffn * 2) for r in streams}
    up_ptrs = {r: _alloc(runtime, r, per_rank_ffn * 2) for r in streams}
    act_ptrs = {r: _alloc(runtime, r, per_rank_ffn * 2) for r in streams}
    chain_walls: list[float] = []
    exchange_walls: list[float] = []
    step_walls: list[float] = []
    slot = hidden * (2 if partial_dtype == "bf16" else 4)
    try:
        for i in range(warmup + iterations):
            t0 = time.perf_counter()
            for rank, stream in streams.items():
                weights = rank_weights[rank]
                with scoped_current_device(runtime, rank):
                    if fused:
                        launch_gguf_linear_pair_silu(
                            weights["ffn_gate"],
                            weights["ffn_up"],
                            x_ptr_by_rank[rank],
                            act_ptrs[rank],
                            1,
                            hidden,
                            per_rank_ffn,
                            use_gemv_decode=True,
                            registered_decode_variant=decode_variant,
                            stream=stream,
                            runtime=runtime,
                        )
                    else:
                        launch_gguf_linear(
                            weights["ffn_gate"],
                            x_ptr_by_rank[rank],
                            gate_ptrs[rank],
                            1,
                            hidden,
                            per_rank_ffn,
                            use_gemv_decode=True,
                            stream=stream,
                            runtime=runtime,
                        )
                        launch_gguf_linear(
                            weights["ffn_up"],
                            x_ptr_by_rank[rank],
                            up_ptrs[rank],
                            1,
                            hidden,
                            per_rank_ffn,
                            use_gemv_decode=True,
                            stream=stream,
                            runtime=runtime,
                        )
                        silu_mul_separate_out_bf16(
                            gate_ptrs[rank],
                            up_ptrs[rank],
                            act_ptrs[rank],
                            1,
                            per_rank_ffn,
                            stream=stream,
                            runtime=runtime,
                        )
                    launch_gguf_linear(
                        weights["ffn_down"],
                        act_ptrs[rank],
                        down_ptrs[rank],
                        1,
                        per_rank_ffn,
                        hidden,
                        use_gemv_decode=True,
                        output_dtype=partial_dtype,
                        stream=stream,
                        runtime=runtime,
                    )
            for rank, stream in streams.items():
                with scoped_current_device(runtime, rank):
                    runtime.stream_synchronize(stream)
            chain_us = (time.perf_counter() - t0) * 1e6

            t1 = time.perf_counter()
            reduced = np.zeros(hidden, dtype=np.float32)
            view = staging.view()
            for rank in streams:
                with scoped_current_device(runtime, rank):
                    runtime.memcpy_async(
                        staging.ptr + rank * slot,
                        down_ptrs[rank],
                        slot,
                        MemcpyKind.DEVICE_TO_HOST,
                        streams[rank],
                    )
                    runtime.stream_synchronize(streams[rank])
                raw = view[rank * slot : (rank + 1) * slot].tobytes()
                if partial_dtype == "bf16":
                    reduced += bf16_to_float32(np.frombuffer(raw, dtype="<u2"))
                else:
                    reduced += np.frombuffer(raw, dtype="<f4").astype(np.float32)
            payload = np.ascontiguousarray(reduced.astype("<f4"))
            for rank in streams:
                with scoped_current_device(runtime, rank):
                    runtime.memcpy_async(
                        reduced_ptrs[rank],
                        payload.ctypes.data,
                        slot,
                        MemcpyKind.HOST_TO_DEVICE,
                        streams[rank],
                    )
                    runtime.stream_synchronize(streams[rank])
            exchange_us = (time.perf_counter() - t1) * 1e6
            if i >= warmup:
                chain_walls.append(chain_us)
                exchange_walls.append(exchange_us)
                step_walls.append((time.perf_counter() - t0) * 1e6)
    finally:
        for ptrs in (gate_ptrs, up_ptrs, act_ptrs):
            for rank, ptr in ptrs.items():
                with scoped_current_device(runtime, rank):
                    runtime.free(ptr)
    return {
        "fused": bool(fused),
        "iterations": int(iterations),
        "chain_us_p50": float(np.quantile(chain_walls, 0.5)),
        "exchange_us_p50": float(np.quantile(exchange_walls, 0.5)),
        "step_us_p50": float(np.quantile(step_walls, 0.5)),
        "step_us_mean": float(np.mean(step_walls)),
    }


def profile_exchange_parts(
    runtime: Any,
    *,
    partial_ptrs: dict[int, int],
    reduced_ptrs: dict[int, int],
    streams: dict[int, int],
    staging: "PinnedStaging",
    hidden: int,
    devices: list[int],
    iterations: int = 300,
    warmup: int = 20,
    partial_dtype: str = "f32",
) -> dict[str, Any]:
    """Where the exchange wall goes, measured part by part.

    The parts are timed in isolation at steady state: the gather (both D2H
    submits, both waits, host sum) and the full exchange including the return
    copies. The difference isolates what the H2D return path costs, which is
    the quantity the transport decision turns on: a return path that doubles
    the exchange wall changes which protocols are worth driving.
    """

    slot = hidden * (2 if partial_dtype == "bf16" else 4)
    # The seed only has to be a valid partial-shaped pattern for the timing
    # loop, so build it in the staging dtype: f32 normals, or their upper 16
    # bits (a truncation to bf16) for the bf16 arm.
    seed_f32 = np.ascontiguousarray(
        np.random.default_rng(7).standard_normal(hidden, dtype="<f4")
    )
    if partial_dtype == "bf16":
        seed = np.ascontiguousarray(
            np.frombuffer(seed_f32.tobytes(), dtype="<u2")[1::2].astype("<u2")
        )
    else:
        seed = seed_f32
    for rank in devices:
        with scoped_current_device(runtime, rank):
            runtime.memcpy(
                partial_ptrs[rank],
                seed.ctypes.data,
                slot,
                MemcpyKind.HOST_TO_DEVICE,
            )
            runtime.stream_synchronize(streams[rank])

    def gather() -> np.ndarray:
        for rank in devices:
            with scoped_current_device(runtime, rank):
                runtime.memcpy_async(
                    staging.ptr + rank * slot,
                    partial_ptrs[rank],
                    slot,
                    MemcpyKind.DEVICE_TO_HOST,
                    streams[rank],
                )
        for rank in devices:
            with scoped_current_device(runtime, rank):
                runtime.stream_synchronize(streams[rank])
        raw = staging.view()[: len(devices) * slot].tobytes()
        if partial_dtype == "bf16":
            both = bf16_to_float32(np.frombuffer(raw, dtype="<u2")).reshape(
                len(devices), hidden
            )
        else:
            both = np.frombuffer(raw, dtype="<f4").reshape(len(devices), hidden)
        return both.sum(axis=0, dtype=np.float32)

    def full() -> None:
        reduced = gather()
        payload = np.ascontiguousarray(reduced, dtype="<f4")
        for rank in devices:
            with scoped_current_device(runtime, rank):
                runtime.memcpy_async(
                    reduced_ptrs[rank],
                    payload.ctypes.data,
                    slot,
                    MemcpyKind.HOST_TO_DEVICE,
                    streams[rank],
                )
        for rank in devices:
            with scoped_current_device(runtime, rank):
                runtime.stream_synchronize(streams[rank])

    def timed(fn) -> dict[str, float]:
        fn()
        fn()
        samples = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e6)
        return {
            "p50_us": float(np.quantile(samples, 0.5)),
            "p99_us": float(np.quantile(samples, 0.99)),
            "mean_us": float(np.mean(samples)),
        }

    gather_stats = timed(gather)
    full_stats = timed(full)
    return {
        "gather_p50_us": gather_stats["p50_us"],
        "gather_p99_us": gather_stats["p99_us"],
        "full_p50_us": full_stats["p50_us"],
        "full_p99_us": full_stats["p99_us"],
        "return_path_us_p50": full_stats["p50_us"] - gather_stats["p50_us"],
        "note": (
            "the H2D return copies to both ranks cost about as much as the whole "
            "gather; the transport A/B's native arm measured 20.8 us per reduction "
            "for the same batched protocol driven from compiled code"
        ),
    }


def run_tp1_teacher(
    runtime: Any,
    *,
    resident_gate: Any,
    resident_up: Any,
    resident_down: Any,
    x_bf16_bytes: np.ndarray,
    stream: int,
    hidden: int,
    ffn: int,
    decode_variant: str,
    output_dtype: str = "bf16",
) -> dict[str, np.ndarray]:
    """The TP1 incumbent: fused pair+SiLU at its admitted shape, then down."""

    out_itemsize = 4 if output_dtype == "f32" else 2
    x_ptr = _alloc(runtime, 0, hidden * 2)
    inter_ptr = _alloc(runtime, 0, ffn * 2)
    y_ptr = _alloc(runtime, 0, hidden * out_itemsize)
    try:
        _upload(runtime, 0, x_ptr, x_bf16_bytes)
        with scoped_current_device(runtime, 0):
            launched = launch_gguf_linear_pair_silu(
                resident_gate,
                resident_up,
                x_ptr,
                inter_ptr,
                1,
                hidden,
                ffn,
                use_gemv_decode=True,
                registered_decode_variant=decode_variant,
                stream=stream,
                runtime=runtime,
            )
            if not launched:
                raise RuntimeError("the TP1 fused pair+SiLU route did not launch")
            launch_gguf_linear(
                resident_down,
                inter_ptr,
                y_ptr,
                1,
                ffn,
                hidden,
                use_gemv_decode=True,
                output_dtype=output_dtype,
                stream=stream,
                runtime=runtime,
            )
            runtime.stream_synchronize(stream)
        return {
            "intermediate": _download(runtime, 0, inter_ptr, ffn * 2),
            "down_partial": _download(runtime, 0, y_ptr, hidden * out_itemsize),
        }
    finally:
        for ptr in (x_ptr, inter_ptr, y_ptr):
            with scoped_current_device(runtime, 0):
                runtime.free(ptr)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    *,
    model: Path,
    layer: int,
    world_size: int,
    seed: int = 20260915,
    iterations: int = 200,
    warmup: int = 20,
    down_output_dtype: str = "f32",
) -> dict[str, Any]:
    import hashlib  # noqa: PLC0415

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    reader = GGUFReader(str(model))
    info = scan_gguf(str(model))
    config = qwen35_gguf_config_from_metadata(info)
    hidden = int(config.hidden_size)
    ffn = int(config.feed_forward_length)
    backend = "hip_gfx1100"

    if ffn % int(world_size):
        raise SystemExit(f"feed_forward_length {ffn} does not split across {world_size}")

    names = mlp_tensor_names(int(layer))
    materialization, plan_context = resolve_incumbent_plan(info)
    model_map = build_qwen35_gguf_tensor_map(info)
    fingerprint = build_qwen35_gguf_role_manifest(model_map).fingerprint
    manifest = build_shard_manifest(info, world_size=int(world_size), model_hash=fingerprint)
    by_name = {plan.name: plan for plan in manifest.tensors}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"shard manifest has no plan for {missing}")

    admission = fused_path_admission(
        geometry=config,
        file_type_name=getattr(info, "file_type_name", None),
        world_size=int(world_size),
    )

    # -- host oracle ---------------------------------------------------------
    full = dequantized_mlp(reader, int(layer))
    rng = np.random.default_rng(seed)
    x_f32 = rng.standard_normal(hidden, dtype=np.float32) * np.float32(0.05)
    x_bf16 = bf16_round(x_f32)
    x_bf16_bytes = np.frombuffer(bf16_bytes(x_f32), dtype=np.uint8)

    truth = full_truth(full["ffn_gate"], full["ffn_up"], full["ffn_down"], x_bf16)
    per_rank = ffn // int(world_size)
    # The residual entering this block's MLP tail and the next block's input
    # norm weight: both are replicated on every rank, so the boundary comparison
    # below is per-rank independent of the shard.
    residual_f32 = bf16_round(rng.standard_normal(hidden, dtype=np.float32) * np.float32(0.05))
    norm_weight = (rng.standard_normal(hidden, dtype=np.float32) * np.float32(0.1) + np.float32(1.0)).astype(np.float32)
    contract = [
        contract_rank_partials(
            full["ffn_gate"],
            full["ffn_up"],
            full["ffn_down"],
            x_bf16,
            rank,
            world_size,
            down_output_dtype=down_output_dtype,
        )
        for rank in range(int(world_size))
    ]
    contract_sum = np.sum([c["down_partial"] for c in contract], axis=0, dtype=np.float32)

    # -- device ---------------------------------------------------------------
    runtime = get_hip_runtime()
    device_count = runtime.device_count()
    if device_count < int(world_size):
        raise SystemExit(
            f"need {world_size} devices for ranks, found {device_count}"
        )
    streams = []
    device_names = []
    for device in range(int(world_size)):
        with scoped_current_device(runtime, device):
            streams.append(runtime.stream_create())
            device_names.append(runtime.device_get_name(device))

    # Rank shard payloads, built from the same planner slices the probe verified.
    plans = {name: by_name[name] for name in names}
    rank_weights: list[dict[str, _ShardWeight]] = []
    rank_outputs: list[dict[str, Any]] = []
    staging: PinnedStaging | None = None
    resident: Any = None
    reduced_ptrs: dict[int, int] = {}
    exchange_record: dict[str, Any] = {}
    reduced: np.ndarray | None = None
    teacher: dict[str, Any] = {}
    exchange_profile: dict[str, Any] = {}
    try:
        for rank in range(int(world_size)):
            weights: dict[str, _ShardWeight] = {}
            for name in names:
                role = name.split(".")[2]
                payload, layout, quant_key = rank_shard_payload(
                    reader,
                    materialization,
                    plans[name],
                    plans[name].slice_for(rank),
                    layer=int(layer),
                )
                allocation = _ShardAllocation("tiles", runtime, rank, payload)
                weights[role] = _ShardWeight(layout, quant_key, allocation)
            rank_weights.append(weights)

        # The down projection writes f32 partials by default: the kernel's f32
        # accumulator leaves the rank unrounded, so the reduction is a plain f32
        # sum with no per-rank bf16 step in front of it. ``--down-output-dtype
        # bf16`` is the shipped TP2 schedule's rounded partial (one bf16 rounding
        # per rank before the sum) and exists so the two arithmetic options are
        # compared against the same f64 oracle. Gate/up and the activation stay
        # bf16, which is the incumbent activation contract.
        down_dtype = str(down_output_dtype)
        rank_outputs = []
        for rank in range(int(world_size)):
            rank_outputs.append(
                run_rank_chain(
                    runtime,
                    device=rank,
                    stream=streams[rank],
                    weights=rank_weights[rank],
                    x_bf16_bytes=x_bf16_bytes,
                    hidden=hidden,
                    per_rank_ffn=per_rank,
                    output_dtype=down_dtype,
                    keep_down_partial=True,
                )
            )
        down_ptrs = {rank: int(out["down_ptr"]) for rank, out in enumerate(rank_outputs)}
        # One reduced buffer per rank: after the reduction both ranks hold the
        # same hidden state, which is what the replicated next layer consumes.
        reduced_ptrs = {rank: _alloc(runtime, rank, hidden * 4) for rank in range(int(world_size))}
        staging = PinnedStaging(runtime, int(world_size) * hidden * 4)
        reduced, exchange_record = staged_exchange_reduce(
            runtime,
            partial_ptrs=down_ptrs,
            reduced_ptrs=reduced_ptrs,
            streams={rank: streams[rank] for rank in range(int(world_size))},
            staging=staging,
            hidden=hidden,
            devices=list(range(int(world_size))),
            partial_dtype=down_dtype,
        )
        # Verify each rank's on-device reduced vector once, outside any timed
        # path: sync both streams and read the buffers back.
        payload_check = np.ascontiguousarray(reduced.astype("<f4"))
        for rank in range(int(world_size)):
            with scoped_current_device(runtime, rank):
                runtime.stream_synchronize(streams[rank])
                got = np.frombuffer(
                    _download(runtime, rank, reduced_ptrs[rank], hidden * 4).tobytes(),
                    dtype="<f4",
                )
            exchange_record.setdefault("verified_ranks", []).append(
                {
                    "device": rank,
                    "h2d_roundtrip_max_abs": float(np.abs(got - payload_check).max()),
                }
            )

        # The fused candidate, run only when the policy resolves a TP1 variant
        # whose kernel contract admits the shard shape. It is compared against
        # the unfused baseline above; it does not replace it as the record.
        fused_candidate = None
        tp1_variant = admission.get("tp1_variant")
        if tp1_variant and admission.get("shard_shape_error") is None:
            try:
                fused_candidate = [
                    run_rank_chain_fused(
                        runtime,
                        device=rank,
                        stream=streams[rank],
                        weights=rank_weights[rank],
                        x_bf16_bytes=x_bf16_bytes,
                        hidden=hidden,
                        per_rank_ffn=per_rank,
                        decode_variant=str(tp1_variant),
                        down_output_dtype=down_dtype,
                    )
                    for rank in range(int(world_size))
                ]
            except (RuntimeError, ValueError) as error:
                fused_candidate = {"error": f"{type(error).__name__}: {error}"}

        # TP1 teacher on device 0 through the incumbent resident weights. The
        # resident weights stay alive through the segment measurement below,
        # which reuses them, and are freed in the outer finally.
        resident = materialize_qwen35_gguf_weights(
            str(model),
            selected_slots=[f"layers.{int(layer)}.ffn_gate", f"layers.{int(layer)}.ffn_up", f"layers.{int(layer)}.ffn_down"],
            device=Device("hip", 0),
            backend=backend,
        )
        lw = next(
            entry for entry in resident.layers if int(entry.layer_id) == int(layer)
        )
        decode_variant = admission["tp1_variant"]
        if decode_variant is None:
            raise SystemExit(
                "no TP1 fused decode variant resolved; the teacher route is unavailable"
            )
        teacher = run_tp1_teacher(
            runtime,
            resident_gate=lw.weight("ffn_gate"),
            resident_up=lw.weight("ffn_up"),
            resident_down=lw.weight("ffn_down"),
            x_bf16_bytes=x_bf16_bytes,
            stream=streams[0],
            hidden=hidden,
            ffn=ffn,
            decode_variant=str(decode_variant),
            output_dtype="f32",
        )
        x_ptrs: dict[int, int] = {}
        try:
            x_ptrs = {rank: _alloc(runtime, rank, hidden * 2) for rank in range(int(world_size))}
            for rank in range(int(world_size)):
                _upload(runtime, rank, x_ptrs[rank], x_bf16_bytes)
            tp2_unfused = measure_tp2_segment(
                runtime,
                rank_weights=rank_weights,
                x_ptr_by_rank=x_ptrs,
                down_ptrs=down_ptrs,
                reduced_ptrs=reduced_ptrs,
                streams={rank: streams[rank] for rank in range(int(world_size))},
                staging=staging,
                hidden=hidden,
                per_rank_ffn=per_rank,
                fused=False,
                decode_variant=None,
                iterations=iterations,
                warmup=warmup,
                partial_dtype=down_dtype,
            )
            tp2_fused = (
                measure_tp2_segment(
                    runtime,
                    rank_weights=rank_weights,
                    x_ptr_by_rank=x_ptrs,
                    down_ptrs=down_ptrs,
                    reduced_ptrs=reduced_ptrs,
                    streams={rank: streams[rank] for rank in range(int(world_size))},
                    staging=staging,
                    hidden=hidden,
                    per_rank_ffn=per_rank,
                    fused=True,
                    decode_variant=str(tp1_variant),
                    iterations=iterations,
                    warmup=warmup,
                    partial_dtype=down_dtype,
                )
                if tp1_variant
                else None
            )
            tp1 = measure_tp1_segment(
                runtime,
                resident_gate=lw.weight("ffn_gate"),
                resident_up=lw.weight("ffn_up"),
                resident_down=lw.weight("ffn_down"),
                x_ptr=x_ptrs[0],
                iterations=iterations,
                warmup=warmup,
                hidden=hidden,
                ffn=ffn,
                decode_variant=str(decode_variant),
            )
            exchange_profile = profile_exchange_parts(
                runtime,
                partial_ptrs=down_ptrs,
                reduced_ptrs=reduced_ptrs,
                streams={rank: streams[rank] for rank in range(int(world_size))},
                staging=staging,
                hidden=hidden,
                devices=list(range(int(world_size))),
                partial_dtype=down_dtype,
            )
        finally:
            for rank, ptr in x_ptrs.items():
                with scoped_current_device(runtime, rank):
                    runtime.free(ptr)
    finally:
        for rank, weights in enumerate(rank_weights):
            for weight in weights.values():
                weight._allocation.free(runtime=runtime)
        for rank, out in enumerate(rank_outputs):
            if "down_ptr" in out:
                with scoped_current_device(runtime, rank):
                    runtime.free(int(out["down_ptr"]))
        for rank, ptr in reduced_ptrs.items():
            with scoped_current_device(runtime, rank):
                runtime.free(ptr)
        if staging is not None:
            staging.free()
        if resident is not None:
            resident.free()
        for stream in streams:
            runtime.stream_destroy(stream)

    # -- checks ---------------------------------------------------------------

    def as_f32(raw: np.ndarray, count: int, dtype: str) -> np.ndarray:
        flat = np.frombuffer(raw.tobytes(), dtype=np.uint8)
        if dtype == "f32":
            return flat.view("<f4").astype(np.float32)
        return bf16_to_float32(flat.view("<u2"))

    per_rank_report = []
    summed = np.zeros(hidden, dtype=np.float32)
    for rank, output in enumerate(rank_outputs):
        gate_dev = as_f32(output["gate"], per_rank, "bf16")
        up_dev = as_f32(output["up"], per_rank, "bf16")
        act_dev = as_f32(output["activated"], per_rank, "bf16")
        partial_dev = as_f32(output["down_partial"], hidden, output["down_output_dtype"])
        c = contract[rank]
        summed += partial_dev
        per_rank_report.append(
            {
                "rank": rank,
                "device": rank,
                "gate_vs_contract": _relative_errors(gate_dev, c["gate"]),
                "up_vs_contract": _relative_errors(up_dev, c["up"]),
                "activated_vs_contract": _relative_errors(act_dev, c["activated"]),
                "down_partial_vs_contract": _relative_errors(partial_dev, c["down_partial"]),
                "down_partial_max_abs": float(np.abs(partial_dev - c["down_partial"]).max()),
            }
        )

    teacher_y = as_f32(teacher["down_partial"], hidden, "f32")

    # The residual/next-consumer boundary: what the replicated next layer would
    # consume on each rank after the exchange. The TP2 path reduces in f32 and
    # adds the residual in f32; the TP1 teacher writes an f32 down output; the
    # truth chain has no rounding at all.
    assert reduced is not None
    boundary_tp2 = residual_boundary(residual_f32, reduced, norm_weight)
    boundary_tp1 = residual_boundary(residual_f32, teacher_y, norm_weight)
    boundary_truth = residual_boundary(residual_f32, truth, norm_weight)

    fused_report: dict[str, Any] | None = None
    if isinstance(fused_candidate, list):
        fused_sum = np.zeros(hidden, dtype=np.float32)
        fused_rank_rows = []
        for rank, output in enumerate(fused_candidate):
            act_dev = as_f32(output["activated"], per_rank, "bf16")
            partial_dev = as_f32(
                output["down_partial"], hidden, str(down_dtype)
            )
            fused_sum += partial_dev
            c = contract[rank]
            unfused_partial = as_f32(
                rank_outputs[rank]["down_partial"],
                hidden,
                rank_outputs[rank]["down_output_dtype"],
            )
            fused_rank_rows.append(
                {
                    "rank": rank,
                    "activated_vs_unfused": _relative_errors(act_dev, as_f32(rank_outputs[rank]["activated"], per_rank, "bf16")),
                    "down_partial_vs_unfused": _relative_errors(partial_dev, unfused_partial),
                    "activated_vs_contract": _relative_errors(act_dev, c["activated"]),
                }
            )
        fused_report = {
            "decode_variant": str(tp1_variant),
            "shape": [1, hidden, per_rank],
            "policy_admitted": False,
            "note": (
                "a candidate at a shape the policy table does not list; compared "
                "against the unfused baseline, not added to production admission"
            ),
            "per_rank": fused_rank_rows,
            "sum_vs_unfused_sum": _relative_errors(fused_sum, summed),
            "sum_vs_truth": _relative_errors(fused_sum, truth),
        }
    elif isinstance(fused_candidate, dict):
        fused_report = fused_candidate

    report = {
        "schema_version": 1,
        "kind": "tp2-mlp-slice-e2e",
        "generated_at": started,
        "model": str(model),
        "layer": int(layer),
        "world_size": int(world_size),
        "hidden_size": hidden,
        "feed_forward_length": ffn,
        "per_rank_ffn": per_rank,
        "seed": int(seed),
        "manifest_hash": manifest.manifest_hash(),
        "plan_context": plan_context,
        "devices": device_names,
        "fused_path_admission": admission,
        "reduction": {
            "down_output_dtype": str(down_output_dtype),
            "sum_dtype": "f32",
            "transport": "staged exchange: per-rank D2H into pinned host, host f32 sum, H2D to every rank",
            "decision": (
                "f32 partials: the down kernel's f32 accumulator leaves the rank "
                "unrounded, so no per-rank bf16 rounding precedes the sum, and the "
                "f32 reduced vector feeds the incumbent f32-residual decode path "
                "without a conversion"
            ) if str(down_output_dtype) == "f32" else (
                "bf16 partials (shipped TP2 schedule): each rank's down output is "
                "rounded to bf16 once before the f32 staged sum"
            ),
            "exchange": exchange_record,
            "exchange_profile": exchange_profile,
        },
        "stages": {
            "tp2_sum_vs_truth": _relative_errors(summed, truth),
            "tp2_sum_vs_contract_sum": _relative_errors(summed, contract_sum),
            "tp2_sum_vs_tp1_teacher": _relative_errors(summed, teacher_y),
            "tp1_teacher_vs_truth": _relative_errors(teacher_y, truth),
            "boundary_next_hidden_tp2_vs_tp1": _relative_errors(
                boundary_tp2["next_hidden"], boundary_tp1["next_hidden"]
            ),
            "boundary_next_hidden_tp2_vs_truth": _relative_errors(
                boundary_tp2["next_hidden"], boundary_truth["next_hidden"]
            ),
            "boundary_next_norm_tp2_vs_tp1": _relative_errors(
                boundary_tp2["next_norm"], boundary_tp1["next_norm"]
            ),
            "boundary_next_norm_tp2_vs_truth": _relative_errors(
                boundary_tp2["next_norm"], boundary_truth["next_norm"]
            ),
        },
        "per_rank": per_rank_report,
        "segment_walls": {
            "protocol": {
                "iterations": int(iterations),
                "warmup": int(warmup),
                "note": (
                    "host wall around one synchronized step, launches async; "
                    "per-stage numbers are HIP event device time on the same "
                    "stream. The residual add is excluded on both paths: it is "
                    "one identical 5120-vector add."
                ),
            },
            "tp1": tp1,
            "tp2_unfused": tp2_unfused,
            "tp2_fused": tp2_fused,
        },
        "per_rank": per_rank_report,
        "fused_candidate": fused_report,
        "input": {
            "x_abs_max": float(np.abs(x_f32).max()),
            "residual_abs_max": float(np.abs(residual_f32).max()),
            "x_checksum": hashlib.sha256(x_bf16_bytes.tobytes()).hexdigest()[:16],
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--down-output-dtype",
        default="f32",
        choices=("f32", "bf16"),
        help=(
            "dtype the rank's down projection writes: f32 is the unrounded "
            "accumulator (no per-rank rounding before the staged sum), bf16 "
            "rounds each rank's partial once"
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(
        model=args.model,
        layer=args.layer,
        world_size=args.world_size,
        seed=args.seed,
        iterations=args.iterations,
        warmup=args.warmup,
        down_output_dtype=args.down_output_dtype,
    )
    text = json.dumps(report, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    gates = report["stages"]
    for name, metrics in gates.items():
        print(
            f"{name}: max_abs={metrics['max_abs_err']:.4e} "
            f"max_rel={metrics['max_rel_err']:.4e} mean_rel={metrics['mean_rel_err']:.4e}"
        )
    walls = report["segment_walls"]
    tp1 = walls["tp1"]
    print(
        f"TP1 step: {tp1['step_us_p50']:.1f} us wall "
        f"(pair+silu {tp1['pair_silu_us_p50']:.1f}, down {tp1['down_us_p50']:.1f})"
    )
    for name in ("tp2_unfused", "tp2_fused"):
        tp2 = walls[name]
        if tp2 is None:
            continue
        print(
            f"{name}: {tp2['step_us_p50']:.1f} us wall "
            f"(chains {tp2['chain_us_p50']:.1f}, exchange {tp2['exchange_us_p50']:.1f})"
        )
    if walls["tp2_unfused"]:
        saving = tp1["step_us_p50"] - walls["tp2_unfused"]["step_us_p50"]
        print(f"segment saving vs unfused TP2: {saving:+.1f} us")
    if walls["tp2_fused"]:
        saving = tp1["step_us_p50"] - walls["tp2_fused"]["step_us_p50"]
        print(f"segment saving vs fused TP2: {saving:+.1f} us")
    fused = report.get("fused_candidate")
    if fused and "sum_vs_unfused_sum" in fused:
        metrics = fused["sum_vs_unfused_sum"]
        print(
            f"fused vs unfused sum: max_abs={metrics['max_abs_err']:.4e} "
            f"mean_rel={metrics['mean_rel_err']:.4e} max_rel={metrics['max_rel_err']:.4e}"
        )
    elif fused:
        print(f"fused candidate did not run: {fused.get('error')}")
    if args.json is not None:
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
