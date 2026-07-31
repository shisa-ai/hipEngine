"""Torch-free pinned Moonshine checkpoint validation and FP16 materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    load_host_array_to_device_as_dtype,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex, load_weight_index
from hipengine.models.moonshine import (
    MoonshineModelSpec,
    expected_moonshine_weight_shapes,
    parse_moonshine_model_spec,
    validate_moonshine_weight_index,
)
from hipengine.quant.w8a16 import W8A16_LAYOUT, quantize_w8a16_per_output

MOONSHINE_W8A16_FAMILY_ORDER = ("lm_head", "mlp", "attention")
MOONSHINE_W8A16_COMPONENT_ORDER = (
    "lm_head",
    "mlp_fc1",
    "mlp_fc2",
    "self_attention",
    "cross_attention",
)
_MOONSHINE_W8A16_ALIASES = {
    "mlp": ("mlp_fc1", "mlp_fc2"),
    "attention": ("self_attention", "cross_attention"),
}


@dataclass(frozen=True)
class MoonshineW8A16Tensor:
    """One resident row-major INT8 weight and its FP32 output-row scales."""

    source_name: str
    family: str
    qweight: DeviceTensorAllocation
    scales: DeviceTensorAllocation
    source_fp16_nbytes: int
    layout: str = W8A16_LAYOUT

    @property
    def packed_nbytes(self) -> int:
        return self.qweight.buffer.nbytes + self.scales.buffer.nbytes

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.scales.free(runtime=runtime)
        self.qweight.free(runtime=runtime)


@dataclass(frozen=True)
class MoonshineW8A16Weights:
    """Selected decoder W8A16 sidecars; FP16 fallback weights remain separate."""

    families: tuple[str, ...]
    tensors: Mapping[str, MoonshineW8A16Tensor]

    def __contains__(self, source_name: str) -> bool:
        return source_name in self.tensors

    def __getitem__(self, source_name: str) -> MoonshineW8A16Tensor:
        return self.tensors[source_name]

    @property
    def source_fp16_nbytes(self) -> int:
        return sum(tensor.source_fp16_nbytes for tensor in self.tensors.values())

    @property
    def qweight_nbytes(self) -> int:
        return sum(tensor.qweight.buffer.nbytes for tensor in self.tensors.values())

    @property
    def scale_nbytes(self) -> int:
        return sum(tensor.scales.buffer.nbytes for tensor in self.tensors.values())

    @property
    def packed_nbytes(self) -> int:
        return self.qweight_nbytes + self.scale_nbytes

    def contract(self) -> dict[str, object]:
        return {
            "enabled": True,
            "families": list(self.families),
            "tensor_count": len(self.tensors),
            "layout": W8A16_LAYOUT,
            "weight_dtype": "int8",
            "activation_dtype": "float16",
            "accumulation_dtype": "float32",
            "output_dtype": "float16",
            "scale_dtype": "float32",
            "scale_granularity": "per_output_channel",
            "symmetric_range": [-127, 127],
            "source_fp16_nbytes": self.source_fp16_nbytes,
            "qweight_nbytes": self.qweight_nbytes,
            "scale_nbytes": self.scale_nbytes,
            "packed_nbytes": self.packed_nbytes,
            "active_read_byte_reduction": self.source_fp16_nbytes - self.packed_nbytes,
        }

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for tensor in reversed(tuple(self.tensors.values())):
            tensor.free(runtime=runtime)


@dataclass(frozen=True)
class MoonshineLoadedModel:
    """Validated model metadata plus all resident FP16 device weights."""

    spec: MoonshineModelSpec
    index: WeightIndex
    weights: DeviceWeightMap
    baseline_allocated_bytes: int
    baseline_active_allocations: int
    w8a16: MoonshineW8A16Weights | None = None

    @property
    def fp16_weight_bytes(self) -> int:
        return sum(
            allocation.buffer.nbytes
            for allocation in self.weights.tensors.values()
            if allocation.owns_buffer
        )

    @property
    def owned_weight_bytes(self) -> int:
        sidecar = 0 if self.w8a16 is None else self.w8a16.packed_nbytes
        return self.fp16_weight_bytes + sidecar

    @property
    def owned_weight_allocations(self) -> int:
        fp16 = sum(allocation.owns_buffer for allocation in self.weights.tensors.values())
        sidecar = 0 if self.w8a16 is None else 2 * len(self.w8a16.tensors)
        return fp16 + sidecar

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self.w8a16 is not None:
            self.w8a16.free(runtime=runtime)
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


def normalize_moonshine_w8a16_families(
    families: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalize selected families into the required candidate order."""

    if families is None:
        requested: tuple[str, ...] = ()
    elif isinstance(families, str):
        requested = tuple(part.strip() for part in families.split(",") if part.strip())
    else:
        requested = tuple(str(part).strip() for part in families if str(part).strip())
    allowed = set(MOONSHINE_W8A16_COMPONENT_ORDER) | set(_MOONSHINE_W8A16_ALIASES)
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"unknown Moonshine W8A16 families/components: {unknown}")
    expanded = set(requested)
    for alias, components in _MOONSHINE_W8A16_ALIASES.items():
        if alias in expanded:
            expanded.update(components)
    return tuple(component for component in MOONSHINE_W8A16_COMPONENT_ORDER if component in expanded)


def moonshine_w8a16_source_names(
    spec: MoonshineModelSpec,
    families: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Return exact stored weight names selected for W8A16 conversion."""

    selected = normalize_moonshine_w8a16_families(families)
    names: list[str] = []
    if "lm_head" in selected:
        names.append(spec.embedding_weight_name)
    for layer in range(spec.decoder_layers):
        prefix = f"model.decoder.layers.{layer}"
        if "mlp_fc1" in selected:
            names.append(f"{prefix}.mlp.fc1.weight")
        if "mlp_fc2" in selected:
            names.append(f"{prefix}.mlp.fc2.weight")
        if "self_attention" in selected:
            names.extend(
                f"{prefix}.self_attn.{projection}.weight"
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        if "cross_attention" in selected:
            names.extend(
                f"{prefix}.encoder_attn.{projection}.weight"
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
    return tuple(names)


def _moonshine_w8a16_family(source_name: str, spec: MoonshineModelSpec) -> str:
    if source_name == spec.embedding_weight_name:
        return "lm_head"
    if source_name.endswith(".mlp.fc1.weight"):
        return "mlp_fc1"
    if source_name.endswith(".mlp.fc2.weight"):
        return "mlp_fc2"
    if ".self_attn." in source_name:
        return "self_attention"
    if ".encoder_attn." in source_name:
        return "cross_attention"
    raise ValueError(f"weight {source_name!r} is not a Moonshine W8A16 candidate")


def materialize_moonshine_w8a16_weights(
    index: WeightIndex,
    spec: MoonshineModelSpec,
    families: str | Iterable[str] | None,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> MoonshineW8A16Weights | None:
    """Build selected load-time W8A16 sidecars without changing FP16 fallbacks."""

    selected = normalize_moonshine_w8a16_families(families)
    if not selected:
        return None
    validate_moonshine_weight_index(spec, index)
    target_device = device or Device("hip", 0)
    names = moonshine_w8a16_source_names(spec, selected)
    names_by_shard: dict[Path, list[str]] = {}
    for name in names:
        names_by_shard.setdefault(index.tensors[name].shard_path, []).append(name)

    tensors: dict[str, MoonshineW8A16Tensor] = {}
    try:
        for shard in sorted(names_by_shard):
            with safe_open(str(shard), framework="numpy") as handle:
                for name in sorted(names_by_shard[shard]):
                    info = index.tensors[name]
                    source = handle.get_tensor(name)
                    if source.shape != info.shape:
                        raise ValueError(
                            f"Moonshine weight {name} changed shape while W8A16 loading: "
                            f"{source.shape} != {info.shape}"
                        )
                    source_fp16 = convert_moonshine_weight_to_fp16(name, source)
                    packed = quantize_w8a16_per_output(source_fp16)
                    qweight = load_host_array_to_device_as_dtype(
                        f"{name}.w8a16.qweight",
                        packed.qweight,
                        DType.INT8,
                        source_dtype="I8",
                        device=target_device,
                        runtime=runtime,
                    )
                    try:
                        scales = load_host_array_to_device_as_dtype(
                            f"{name}.w8a16.scale",
                            packed.scales,
                            DType.FP32,
                            source_dtype="F32",
                            device=target_device,
                            runtime=runtime,
                        )
                    except Exception:
                        qweight.free(runtime=runtime)
                        raise
                    tensors[name] = MoonshineW8A16Tensor(
                        source_name=name,
                        family=_moonshine_w8a16_family(name, spec),
                        qweight=qweight,
                        scales=scales,
                        source_fp16_nbytes=packed.source_fp16_nbytes,
                    )
    except Exception:
        MoonshineW8A16Weights(selected, dict(sorted(tensors.items()))).free(
            runtime=runtime
        )
        raise
    return MoonshineW8A16Weights(selected, dict(sorted(tensors.items())))


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
    w8a16_families: str | Iterable[str] | None = None,
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
    try:
        w8a16 = materialize_moonshine_w8a16_weights(
            index,
            spec,
            w8a16_families,
            device=device,
            runtime=runtime,
        )
    except Exception:
        weights.free(runtime=runtime)
        raise
    return MoonshineLoadedModel(
        spec=spec,
        index=index,
        weights=weights,
        baseline_allocated_bytes=baseline["current_allocated_bytes"],
        baseline_active_allocations=baseline["active_allocations"],
        w8a16=w8a16,
    )


__all__ = [
    "MOONSHINE_W8A16_COMPONENT_ORDER",
    "MOONSHINE_W8A16_FAMILY_ORDER",
    "MoonshineLoadedModel",
    "MoonshineW8A16Tensor",
    "MoonshineW8A16Weights",
    "convert_moonshine_weight_to_fp16",
    "load_moonshine_model",
    "materialize_moonshine_w8a16_weights",
    "materialize_moonshine_weights",
    "moonshine_w8a16_source_names",
    "normalize_moonshine_w8a16_families",
    "read_generation_config",
]
