"""Torch-free pinned TimesFM 2.5 200M checkpoint validation and materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import memory_stats
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    load_host_array_to_device,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex, load_weight_index
from hipengine.models.timesfm import (
    PINNED_TIMESFM_MODEL_ID,
    TimesFMModelSpec,
    expected_timesfm_weight_shapes,
    parse_timesfm_model_spec,
    validate_timesfm_weight_index,
)


@dataclass(frozen=True)
class TimesFMLoadedModel:
    """Validated model metadata plus all resident FP32 device weights."""

    spec: TimesFMModelSpec
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


def convert_timesfm_weight_to_fp32(name: str, value: Any) -> np.ndarray:
    """Validate one stored F32 weight without any dtype conversion."""

    source = np.asarray(value)
    if source.dtype != np.float32:
        raise ValueError(f"TimesFM weight {name} source dtype must be float32")
    if not bool(np.isfinite(source).all()):
        raise ValueError(f"TimesFM weight {name} must contain only finite values")
    return np.ascontiguousarray(source)


def _source_allocation(
    info: TensorInfo,
    array: np.ndarray,
    *,
    device: Device,
    runtime: HipRuntime | None,
) -> DeviceTensorAllocation:
    prepared = load_host_array_to_device(
        info.name,
        array,
        device=device,
        runtime=runtime,
    )
    return DeviceTensorAllocation(
        name=info.name,
        source=info,
        buffer=prepared.buffer,
        tensor=prepared.tensor,
    )


def materialize_timesfm_weights(
    index: WeightIndex,
    spec: TimesFMModelSpec,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> DeviceWeightMap:
    """Upload fixed-address FP32 weights (stored dtype, no conversion)."""

    validate_timesfm_weight_index(spec, index)
    target_device = device or Device("hip", 0)
    expected = expected_timesfm_weight_shapes(spec)
    names_by_shard: dict[Path, list[str]] = {}
    for name in expected:
        names_by_shard.setdefault(index.tensors[name].shard_path, []).append(name)

    allocations: dict[str, DeviceTensorAllocation] = {}
    try:
        for shard in sorted(names_by_shard):
            with safe_open(str(shard), framework="numpy") as handle:
                for name in sorted(names_by_shard[shard]):
                    info = index.tensors[name]
                    source = handle.get_tensor(name)
                    if source.shape != info.shape:
                        raise ValueError(
                            f"TimesFM weight {name} changed shape while loading: "
                            f"{source.shape} != {info.shape}"
                        )
                    validated = convert_timesfm_weight_to_fp32(name, source)
                    allocations[name] = _source_allocation(
                        info,
                        validated,
                        device=target_device,
                        runtime=runtime,
                    )
    except Exception:
        DeviceWeightMap(allocations).free(runtime=runtime)
        raise
    weights = DeviceWeightMap(dict(sorted(allocations.items())))
    owned_bytes = sum(
        allocation.buffer.nbytes
        for allocation in weights.tensors.values()
        if allocation.owns_buffer
    )
    expected_bytes = spec.parameter_count * 4
    if owned_bytes != expected_bytes:
        weights.free(runtime=runtime)
        raise ValueError(
            f"resident TimesFM FP32 bytes {owned_bytes} != contract {expected_bytes}"
        )
    return weights


def load_timesfm_model(
    model_path: str | Path,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> TimesFMLoadedModel:
    """Validate the pinned snapshot and take ownership of all resident FP32 weights.

    ``model_path`` may be the pinned Hugging Face repo id (resolved from the
    local cache only; hipEngine never downloads during load), a local snapshot
    directory, or a ``model.safetensors`` file with its ``config.json`` beside it.
    """

    baseline = memory_stats()
    index = load_weight_index(model_path)
    spec = parse_timesfm_model_spec(index.config)
    weights = materialize_timesfm_weights(
        index,
        spec,
        device=device,
        runtime=runtime,
    )
    return TimesFMLoadedModel(
        spec=spec,
        index=index,
        weights=weights,
        baseline_allocated_bytes=baseline["current_allocated_bytes"],
        baseline_active_allocations=baseline["active_allocations"],
    )


__all__ = [
    "PINNED_TIMESFM_MODEL_ID",
    "TimesFMLoadedModel",
    "convert_timesfm_weight_to_fp32",
    "load_timesfm_model",
    "materialize_timesfm_weights",
]
