"""Torch-free pinned EVIE-4.5B checkpoint validation and materialization.

Loads the ``tencent/EVIE-4.5B`` safetensors checkpoint (BF16 storage) and
uploads FP32 device tensors. BF16 payloads are read as raw storage bytes and
converted on the host (NumPy has no bfloat16 dtype), so no torch is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hipengine.core.device import Device
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import memory_stats
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    load_host_array_to_device,
)
from hipengine.loading.safetensors import (
    TensorInfo,
    WeightIndex,
    load_weight_index,
    read_tensor_storage_bytes,
)
from hipengine.models.evie import (
    PINNED_EVIE_MODEL_ID,
    EvieModelSpec,
    expected_evie_weight_shapes,
    parse_evie_model_spec,
    validate_evie_weight_index,
)


@dataclass(frozen=True)
class EvieLoadedModel:
    """Validated model metadata plus all resident FP32 device weights."""

    spec: EvieModelSpec
    index: WeightIndex
    weights: DeviceWeightMap
    baseline_allocated_bytes: int
    baseline_active_allocations: int

    @property
    def fp32_weight_bytes(self) -> int:
        return sum(
            allocation.buffer.nbytes
            for allocation in self.weights.tensors.values()
            if allocation.owns_buffer
        )

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.weights.free(runtime=runtime)


def convert_evie_weight_to_fp32(name: str, info: TensorInfo) -> np.ndarray:
    """Read one BF16 checkpoint tensor and convert it to contiguous FP32."""

    raw = read_tensor_storage_bytes(info)
    if info.dtype == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        host = bits.view(np.float32)
    elif info.dtype == "F32":
        host = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"EVIE weight {name} dtype {info.dtype} not supported")
    count = 1
    for dim in info.shape:
        count *= dim
    if host.size != count:
        raise ValueError(f"EVIE weight {name} element count mismatch")
    host = host.reshape(info.shape or ())
    if not bool(np.isfinite(host).all()):
        raise ValueError(f"EVIE weight {name} must contain only finite values")
    return np.ascontiguousarray(host)


def evie_gemm_weight(name: str) -> bool:
    """True when a weight feeds a rocBLAS GEMM (fp16 production candidate).

    All GEMM weights convert to fp16 in fp16 mode, including the GDN
    projections: full f16-in/f16-out GDN passes the oracle fixture
    (query cos 0.9999986, maxsim +0.01%). An earlier collapse attributed
    to f16 GDN output rounding was a stale-edit artifact, not numerics.
    """

    if not name.endswith(".weight"):
        return False
    return not any(
        marker in name
        for marker in ("norm", "embed_tokens", "pos_embed", "conv1d")
    )


def materialize_evie_weights(
    index: WeightIndex,
    spec: EvieModelSpec,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None,
    precision: str = "fp32",
) -> DeviceWeightMap:
    """Upload weights converted from BF16 storage.

    ``precision="fp32"`` (strict) uploads FP32 tensors. ``precision="fp16"``
    uploads GEMM weights (see :func:`evie_gemm_weight`) as FP16 with FP32
    tails; everything norm/embed/pos-related stays FP32 because the fp32
    kernels read those buffers directly.
    """

    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be 'fp32' or 'fp16'")
    validate_evie_weight_index(index, spec)
    target_device = device or Device("hip", 0)
    expected = expected_evie_weight_shapes(spec)
    allocations: dict[str, DeviceTensorAllocation] = {}
    # rocBLAS tiles can over-read past operand ends; keep every weight's
    # tail inside mapped memory so heap layout cannot fault.
    weight_pad = 1 << 20
    try:
        for name in sorted(expected):
            info = index.tensors[name]
            host = convert_evie_weight_to_fp32(name, info)
            if precision == "fp16" and evie_gemm_weight(name):
                host16 = np.ascontiguousarray(
                    host.reshape(-1).astype(np.float16)
                )
                padded16 = np.zeros(
                    host16.nbytes + weight_pad, dtype=np.float16
                )
                padded16[: host16.size] = host16
                prepared = load_host_array_to_device(
                    name,
                    padded16,
                    device=target_device,
                    runtime=runtime,
                )
                allocations[name] = DeviceTensorAllocation(
                    name=name,
                    source=info,
                    buffer=prepared.buffer,
                    tensor=prepared.tensor,
                )
                continue
            padded = np.empty(host.nbytes + weight_pad, dtype=np.float32)
            padded[: host.size] = host.reshape(-1)
            prepared = load_host_array_to_device(
                name,
                padded,
                device=target_device,
                runtime=runtime,
            )
            allocations[name] = DeviceTensorAllocation(
                name=name,
                source=info,
                buffer=prepared.buffer,
                tensor=prepared.tensor,
            )
    except Exception:
        DeviceWeightMap(allocations).free(runtime=runtime)
        raise
    return DeviceWeightMap(dict(sorted(allocations.items())))


def load_evie_model(
    model_path: str | Path,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None,
    precision: str = "fp32",
) -> EvieLoadedModel:
    """Validate the pinned snapshot and take ownership of all resident FP32 weights.

    ``model_path`` may be the pinned Hugging Face repo id (resolved from the
    local cache only; hipEngine never downloads during load), a local snapshot
    directory, or a ``model.safetensors`` file with its ``config.json`` beside it.
    """

    baseline = memory_stats()
    index = load_weight_index(model_path)
    spec = parse_evie_model_spec(index.config)
    weights = materialize_evie_weights(
        index,
        spec,
        device=device,
        runtime=runtime,
        precision=precision,
    )
    return EvieLoadedModel(
        spec=spec,
        index=index,
        weights=weights,
        baseline_allocated_bytes=baseline["current_allocated_bytes"],
        baseline_active_allocations=baseline["active_allocations"],
    )


__all__ = [
    "PINNED_EVIE_MODEL_ID",
    "EvieLoadedModel",
    "convert_evie_weight_to_fp32",
    "load_evie_model",
    "materialize_evie_weights",
]
