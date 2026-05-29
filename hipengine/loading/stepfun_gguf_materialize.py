"""Device materialization for split StepFun GGUF tensor maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.loading.gguf import (
    GGUFSplitModelInfo,
    GGUFSplitTensorInfo,
    MissingGGUFTensorError,
    scan_gguf_splits,
)
from hipengine.loading.materialize import DeviceTensorAllocation, load_host_array_to_device_as_dtype
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_RAW_GGUF,
)
from hipengine.loading.stepfun_gguf import (
    StepFunGGUFConfig,
    StepFunGGUFLayerMap,
    StepFunGGUFModelMap,
    build_stepfun_gguf_tensor_map,
)
from hipengine.quant.gguf import GGMLQuantizationType, numpy_storage_dtype


@dataclass(frozen=True)
class StepFunGGUFWeightSpec:
    """One planned resident StepFun GGUF weight record."""

    slot_path: str
    source: GGUFSplitTensorInfo
    quant_key: str
    layout: str
    allocation_names: tuple[str, ...]


@dataclass(frozen=True)
class StepFunGGUFMaterializationPlan:
    """Resident-weight layout plan derived from a validated StepFun tensor map."""

    config: StepFunGGUFConfig
    root_specs: Mapping[str, StepFunGGUFWeightSpec]
    layer_specs: tuple[Mapping[str, StepFunGGUFWeightSpec], ...]

    @property
    def specs(self) -> tuple[StepFunGGUFWeightSpec, ...]:
        specs: list[StepFunGGUFWeightSpec] = []
        seen: set[tuple[Path, str, str]] = set()
        for spec in self.root_specs.values():
            key = (spec.source.source_path, spec.source.name, spec.layout)
            if key not in seen:
                seen.add(key)
                specs.append(spec)
        for layer in self.layer_specs:
            for spec in layer.values():
                key = (spec.source.source_path, spec.source.name, spec.layout)
                if key not in seen:
                    seen.add(key)
                    specs.append(spec)
        return tuple(specs)

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(spec.source.name for spec in self.specs)

    @property
    def total_nbytes(self) -> int:
        return sum(spec.source.nbytes for spec in self.specs)

    @property
    def quant_counts(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for spec in self.specs:
            counts[spec.quant_key] = counts.get(spec.quant_key, 0) + 1
        return MappingProxyType(counts)


@dataclass(frozen=True)
class StepFunGGUFDeviceWeight:
    """Owned device allocations for one logical StepFun GGUF weight."""

    spec: StepFunGGUFWeightSpec
    allocations: Mapping[str, DeviceTensorAllocation]

    def allocation(self, name: str = "raw") -> DeviceTensorAllocation:
        return self.allocations[name]

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for allocation in reversed(tuple(self.allocations.values())):
            allocation.free(runtime=runtime)


@dataclass(frozen=True)
class StepFunGGUFResidentLayerWeights:
    layer_id: int
    attention_type: str
    mlp_type: str
    weights: Mapping[str, StepFunGGUFDeviceWeight]

    def weight(self, slot: str) -> StepFunGGUFDeviceWeight:
        return self.weights[slot]


@dataclass(frozen=True)
class StepFunGGUFResidentWeights:
    """Device-resident StepFun GGUF weights assembled from split shards."""

    config: StepFunGGUFConfig
    root_weights: Mapping[str, StepFunGGUFDeviceWeight]
    layers: tuple[StepFunGGUFResidentLayerWeights, ...]

    def root(self, slot: str) -> StepFunGGUFDeviceWeight:
        return self.root_weights[slot]

    def layer(self, layer_id: int) -> StepFunGGUFResidentLayerWeights:
        return self.layers[layer_id]

    @property
    def weights(self) -> tuple[StepFunGGUFDeviceWeight, ...]:
        weights: list[StepFunGGUFDeviceWeight] = []
        seen: set[int] = set()
        for weight in self.root_weights.values():
            if id(weight) not in seen:
                seen.add(id(weight))
                weights.append(weight)
        for layer in self.layers:
            for weight in layer.weights.values():
                if id(weight) not in seen:
                    seen.add(id(weight))
                    weights.append(weight)
        return tuple(weights)

    @property
    def allocated_nbytes(self) -> int:
        return sum(
            allocation.buffer.nbytes
            for weight in self.weights
            for allocation in weight.allocations.values()
        )

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for weight in reversed(self.weights):
            weight.free(runtime=runtime)


def stepfun_split_tensor_data(tensor: GGUFSplitTensorInfo):
    """Return a read-only NumPy memmap for a split StepFun tensor payload."""

    import numpy as np

    return np.memmap(
        tensor.source_path,
        mode="r",
        dtype=numpy_storage_dtype(tensor.ggml_type),
        offset=tensor.data_offset,
        shape=tensor.byte_shape,
    )


def plan_stepfun_gguf_materialization(
    model_map: StepFunGGUFModelMap,
) -> StepFunGGUFMaterializationPlan:
    root_specs = {
        slot: _spec_for_tensor(f"root.{slot}", tensor)
        for slot, tensor in model_map.root_tensors.items()
    }
    layer_specs = tuple(_plan_layer(layer) for layer in model_map.layers)
    return StepFunGGUFMaterializationPlan(
        config=model_map.config,
        root_specs=MappingProxyType(root_specs),
        layer_specs=tuple(MappingProxyType(layer) for layer in layer_specs),
    )


def materialize_stepfun_gguf_weights(
    info_or_paths: GGUFSplitModelInfo | Sequence[str | Path],
    *,
    selected_slots: Iterable[str] | None = None,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> StepFunGGUFResidentWeights:
    """Materialize a validated StepFun split GGUF map to resident device records.

    ``selected_slots`` is a test/debug hook using slot paths such as
    ``root.output_norm`` or ``layers.0.attn_q``. Production callers leave it
    unset to materialize the full model.
    """

    info = info_or_paths if isinstance(info_or_paths, GGUFSplitModelInfo) else scan_gguf_splits(info_or_paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    plan = plan_stepfun_gguf_materialization(model_map)
    selected = None if selected_slots is None else set(selected_slots)
    if selected is not None:
        known = {spec.slot_path for spec in plan.specs}
        missing = sorted(selected - known)
        if missing:
            preview = ", ".join(missing[:8])
            more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise MissingGGUFTensorError(f"unknown StepFun materialization slots: {preview}{more}")
    materialized: dict[tuple[Path, str, str], StepFunGGUFDeviceWeight] = {}
    try:
        root_weights = {
            slot: _materialize_or_alias(spec, materialized, selected, device=device, runtime=runtime)
            for slot, spec in plan.root_specs.items()
            if selected is None or spec.slot_path in selected
        }
        layers = tuple(
            StepFunGGUFResidentLayerWeights(
                layer_id=layer.layer_id,
                attention_type=layer.attention_type,
                mlp_type=layer.mlp_type,
                weights=MappingProxyType(
                    {
                        slot: _materialize_or_alias(
                            plan.layer_specs[layer.layer_id][slot],
                            materialized,
                            selected,
                            device=device,
                            runtime=runtime,
                        )
                        for slot in plan.layer_specs[layer.layer_id]
                        if selected is None or plan.layer_specs[layer.layer_id][slot].slot_path in selected
                    }
                ),
            )
            for layer in model_map.layers
        )
    except Exception:
        for weight in reversed(tuple(materialized.values())):
            weight.free(runtime=runtime)
        raise
    return StepFunGGUFResidentWeights(
        config=plan.config,
        root_weights=MappingProxyType(root_weights),
        layers=layers,
    )


def _plan_layer(layer: StepFunGGUFLayerMap) -> dict[str, StepFunGGUFWeightSpec]:
    return {
        slot: _spec_for_tensor(f"layers.{layer.layer_id}.{slot}", tensor)
        for slot, tensor in layer.tensors.items()
    }


def _spec_for_tensor(slot_path: str, tensor: GGUFSplitTensorInfo) -> StepFunGGUFWeightSpec:
    qtype = GGMLQuantizationType(tensor.ggml_type)
    if qtype == GGMLQuantizationType.F32:
        return StepFunGGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="f32",
            layout=LAYOUT_DENSE_F32,
            allocation_names=("raw",),
        )
    if qtype == GGMLQuantizationType.BF16:
        return StepFunGGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="bf16",
            layout=LAYOUT_DENSE_BF16,
            allocation_names=("raw",),
        )
    if qtype in {
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q8_0,
    }:
        return StepFunGGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key=f"gguf_{tensor.ggml_type_name.lower()}",
            layout=LAYOUT_RAW_GGUF,
            allocation_names=("raw",),
        )
    raise ValueError(f"unsupported StepFun GGUF tensor type {tensor.ggml_type_name!r}: {tensor.name}")


def _materialize_or_alias(
    spec: StepFunGGUFWeightSpec,
    materialized: dict[tuple[Path, str, str], StepFunGGUFDeviceWeight],
    selected: set[str] | None,
    *,
    device: Device | None,
    runtime: HipRuntime | None,
) -> StepFunGGUFDeviceWeight:
    del selected  # selection is handled by callers before materialization.
    key = (spec.source.source_path, spec.source.name, spec.layout)
    weight = materialized.get(key)
    if weight is None:
        weight = _materialize_spec(spec, device=device, runtime=runtime)
        materialized[key] = weight
    return weight


def _materialize_spec(
    spec: StepFunGGUFWeightSpec,
    *,
    device: Device | None,
    runtime: HipRuntime | None,
) -> StepFunGGUFDeviceWeight:
    import numpy as np

    raw = np.ascontiguousarray(stepfun_split_tensor_data(spec.source))
    if spec.layout == LAYOUT_RAW_GGUF:
        allocation = load_host_array_to_device_as_dtype(
            spec.source.name,
            raw,
            DType.INT8,
            source_dtype="I8",
            device=device,
            runtime=runtime,
        )
    elif spec.layout == LAYOUT_DENSE_F32:
        allocation = load_host_array_to_device_as_dtype(
            spec.source.name,
            raw,
            DType.FP32,
            source_dtype="F32",
            device=device,
            runtime=runtime,
        )
    elif spec.layout == LAYOUT_DENSE_BF16:
        allocation = load_host_array_to_device_as_dtype(
            spec.source.name,
            raw,
            DType.BF16,
            source_dtype="BF16",
            device=device,
            runtime=runtime,
        )
    else:
        raise ValueError(f"unsupported StepFun materialization layout {spec.layout!r}")
    return StepFunGGUFDeviceWeight(
        spec=spec,
        allocations=MappingProxyType({"raw": allocation}),
    )


__all__ = [
    "StepFunGGUFDeviceWeight",
    "StepFunGGUFMaterializationPlan",
    "StepFunGGUFResidentLayerWeights",
    "StepFunGGUFResidentWeights",
    "StepFunGGUFWeightSpec",
    "materialize_stepfun_gguf_weights",
    "plan_stepfun_gguf_materialization",
    "stepfun_split_tensor_data",
]
