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
from hipengine.core.memory import copy_host_to_device, copy_device_to_host  # noqa: E402
from hipengine.kernels.cpu_reference.maple import f32_to_bf16_bits  # noqa: E402
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
    output_dtype: str = "bf16",
) -> dict[str, np.ndarray]:
    """Run the unfused shard chain on one device and return every intermediate."""

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
        return {
            "gate": _download(runtime, device, gate_ptr, per_rank_ffn * 2),
            "up": _download(runtime, device, up_ptr, per_rank_ffn * 2),
            "activated": _download(runtime, device, act_ptr, per_rank_ffn * 2),
            "down_partial": _download(runtime, device, down_ptr, hidden * out_itemsize),
        }
    finally:
        for ptr in (x_ptr, gate_ptr, up_ptr, act_ptr, down_ptr):
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

    x_ptr = _alloc(runtime, device, hidden * 2)
    act_ptr = _alloc(runtime, device, per_rank_ffn * 2)
    down_ptr = _alloc(runtime, device, hidden * 2)
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
                stream=stream,
                runtime=runtime,
            )
            runtime.stream_synchronize(stream)
        return {
            "activated": _download(runtime, device, act_ptr, per_rank_ffn * 2),
            "down_partial": _download(runtime, device, down_ptr, hidden * 2),
        }
    finally:
        for ptr in (x_ptr, act_ptr, down_ptr):
            with scoped_current_device(runtime, device):
                runtime.free(ptr)


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
        with scoped_current_device(runtime, 0):
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
    output_dtype: str = "bf16",
    seed: int = 20260915,
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
    contract = [
        contract_rank_partials(
            full["ffn_gate"], full["ffn_up"], full["ffn_down"], x_bf16, rank, world_size
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
                    output_dtype=output_dtype,
                )
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
                    )
                    for rank in range(int(world_size))
                ]
            except (RuntimeError, ValueError) as error:
                fused_candidate = {"error": f"{type(error).__name__}: {error}"}

        # TP1 teacher on device 0 through the incumbent resident weights.
        resident = materialize_qwen35_gguf_weights(
            str(model),
            selected_slots=[f"layers.{int(layer)}.ffn_gate", f"layers.{int(layer)}.ffn_up", f"layers.{int(layer)}.ffn_down"],
            device=Device("hip", 0),
            backend=backend,
        )
        try:
            layer_weights = next(
                entry for entry in resident.layers if int(entry.layer_id) == int(layer)
            )
            decode_variant = admission["tp1_variant"]
            if decode_variant is None:
                raise SystemExit(
                    "no TP1 fused decode variant resolved; the teacher route is unavailable"
                )
            teacher = run_tp1_teacher(
                runtime,
                resident_gate=layer_weights.weight("ffn_gate"),
                resident_up=layer_weights.weight("ffn_up"),
                resident_down=layer_weights.weight("ffn_down"),
                x_bf16_bytes=x_bf16_bytes,
                stream=streams[0],
                hidden=hidden,
                ffn=ffn,
                decode_variant=str(decode_variant),
                output_dtype=output_dtype,
            )
        finally:
            resident.free()
    finally:
        for rank, weights in enumerate(rank_weights):
            for weight in weights.values():
                weight._allocation.free(runtime=runtime)
        for stream in streams:
            runtime.stream_destroy(stream)

    # -- checks ---------------------------------------------------------------
    itemsize = 4 if output_dtype == "f32" else 2

    def as_f32(raw: np.ndarray, count: int) -> np.ndarray:
        flat = np.frombuffer(raw.tobytes(), dtype=np.uint8)
        if itemsize == 4:
            return flat.view("<f4").astype(np.float32)
        return bf16_to_float32(flat.view("<u2"))

    per_rank_report = []
    summed = np.zeros(hidden, dtype=np.float32)
    for rank, output in enumerate(rank_outputs):
        gate_dev = as_f32(output["gate"], per_rank)
        up_dev = as_f32(output["up"], per_rank)
        act_dev = as_f32(output["activated"], per_rank)
        partial_dev = as_f32(output["down_partial"], hidden)
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

    teacher_y = as_f32(teacher["down_partial"], hidden)

    fused_report: dict[str, Any] | None = None
    if isinstance(fused_candidate, list):
        fused_sum = np.zeros(hidden, dtype=np.float32)
        fused_rank_rows = []
        for rank, output in enumerate(fused_candidate):
            act_dev = as_f32(output["activated"], per_rank)
            partial_dev = as_f32(output["down_partial"], hidden)
            fused_sum += partial_dev
            c = contract[rank]
            fused_rank_rows.append(
                {
                    "rank": rank,
                    "activated_vs_unfused": _relative_errors(act_dev, as_f32(rank_outputs[rank]["activated"], per_rank)),
                    "down_partial_vs_unfused": _relative_errors(partial_dev, as_f32(rank_outputs[rank]["down_partial"], hidden)),
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
        "output_dtype": output_dtype,
        "seed": int(seed),
        "manifest_hash": manifest.manifest_hash(),
        "plan_context": plan_context,
        "devices": device_names,
        "fused_path_admission": admission,
        "stages": {
            "tp2_sum_vs_truth": _relative_errors(summed, truth),
            "tp2_sum_vs_contract_sum": _relative_errors(summed, contract_sum),
            "tp2_sum_vs_tp1_teacher": _relative_errors(summed, teacher_y),
            "tp1_teacher_vs_truth": _relative_errors(teacher_y, truth),
        },
        "per_rank": per_rank_report,
        "fused_candidate": fused_report,
        "input": {
            "x_abs_max": float(np.abs(x_f32).max()),
            "x_checksum": hashlib.sha256(x_bf16_bytes.tobytes()).hexdigest()[:16],
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output-dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(
        model=args.model,
        layer=args.layer,
        world_size=args.world_size,
        output_dtype=args.output_dtype,
        seed=args.seed,
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
