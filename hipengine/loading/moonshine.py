"""Torch-free pinned Moonshine checkpoint validation and FP16 materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import memory_stats
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    alias_device_allocation,
    load_host_array_to_device,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex, load_weight_index
from hipengine.models.moonshine import (
    MoonshineModelSpec,
    expected_moonshine_weight_shapes,
    parse_moonshine_model_spec,
    validate_moonshine_weight_index,
)


@dataclass(frozen=True)
class MoonshineLoadedModel:
    """Validated model metadata plus all resident FP16 device weights."""

    spec: MoonshineModelSpec
    index: WeightIndex
    weights: DeviceWeightMap
    baseline_allocated_bytes: int
    baseline_active_allocations: int

    @property
    def owned_weight_bytes(self) -> int:
        return sum(
            allocation.buffer.nbytes
            for allocation in self.weights.tensors.values()
            if allocation.owns_buffer
        )

    @property
    def owned_weight_allocations(self) -> int:
        return sum(
            allocation.owns_buffer for allocation in self.weights.tensors.values()
        )

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.weights.free(runtime=runtime)


def convert_moonshine_weight_to_fp16(name: str, value: Any) -> np.ndarray:
    """Perform the one allowed load-time F32 -> FP16 conversion."""

    source = np.asarray(value)
    if source.dtype != np.float32:
        raise ValueError(f"Moonshine weight {name} source dtype must be float32")
    if not bool(np.isfinite(source).all()):
        raise ValueError(f"Moonshine weight {name} must contain only finite values")
    converted = np.ascontiguousarray(source, dtype=np.float16)
    if not bool(np.isfinite(converted).all()):
        raise ValueError(f"Moonshine weight {name} is non-finite after FP16 conversion")
    return converted


def read_generation_config(model_path: Path) -> dict[str, Any]:
    path = model_path / "generation_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"generation_config.json not found under {model_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("generation_config.json must contain an object")
    return value


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


def materialize_moonshine_weights(
    index: WeightIndex,
    spec: MoonshineModelSpec,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> DeviceWeightMap:
    """Convert every stored F32 tensor once and upload fixed-address FP16 weights."""

    validate_moonshine_weight_index(spec, index)
    target_device = device or Device("hip", 0)
    expected = expected_moonshine_weight_shapes(spec)
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
                            f"Moonshine weight {name} changed shape while loading: "
                            f"{source.shape} != {info.shape}"
                        )
                    converted = convert_moonshine_weight_to_fp16(name, source)
                    allocations[name] = _source_allocation(
                        info,
                        converted,
                        device=target_device,
                        runtime=runtime,
                    )
        owner = allocations[spec.embedding_weight_name]
        allocations[spec.lm_head_alias_name] = alias_device_allocation(
            spec.lm_head_alias_name,
            owner,
            owner.tensor.shape,
            DType.FP16,
            device=target_device,
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
    if owned_bytes != spec.runtime_weight_bytes:
        weights.free(runtime=runtime)
        raise ValueError(
            f"resident Moonshine FP16 bytes {owned_bytes} != contract {spec.runtime_weight_bytes}"
        )
    return weights


def load_moonshine_model(
    model_path: str | Path,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> MoonshineLoadedModel:
    """Validate the pinned snapshot and take ownership of all resident FP16 weights."""

    baseline = memory_stats()
    index = load_weight_index(model_path)
    generation = read_generation_config(index.model_path)
    spec = parse_moonshine_model_spec(index.config, generation)
    validate_moonshine_weight_index(spec, index)
    weights = materialize_moonshine_weights(
        index,
        spec,
        device=device,
        runtime=runtime,
    )
    return MoonshineLoadedModel(
        spec=spec,
        index=index,
        weights=weights,
        baseline_allocated_bytes=baseline["current_allocated_bytes"],
        baseline_active_allocations=baseline["active_allocations"],
    )


__all__ = [
    "MoonshineLoadedModel",
    "convert_moonshine_weight_to_fp16",
    "load_moonshine_model",
    "materialize_moonshine_weights",
    "read_generation_config",
]
