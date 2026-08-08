"""Torch-free loader for deepgrove/maple-preview-2bit-mlx.

The official checkpoint is an MLX safetensors artifact, but its packed storage is
framework-independent. This loader validates all exact-path names, shapes, and
dtypes, preserves uint32 ternary/affine bit patterns, and materializes them to
HIP device memory without importing torch or MLX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    load_host_array_to_device_as_dtype,
)
from hipengine.loading.safetensors import (
    TensorInfo,
    WeightIndex,
    load_weight_index,
    read_tensor_storage_bytes,
)
from hipengine.models.maple import MapleModelSpec, parse_maple_model_spec

if TYPE_CHECKING:
    from collections.abc import Iterable

MAPLE_EXACT_TENSOR_COUNT = 463
MAPLE_EXACT_WEIGHT_BYTES = 5_308_186_624
MAPLE_FLASH_HEAD_PREFIX = "lm_head_flash."


class MapleLayoutError(ValueError):
    """The checkpoint does not match Maple's exact packed layout."""


@dataclass(frozen=True)
class MapleTensorRequirement:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        itemsize = {"BF16": 2, "U32": 4}[self.dtype]
        return itemsize * int(np.prod(self.shape, dtype=np.int64))


@dataclass(frozen=True)
class MapleLayoutValidation:
    spec: MapleModelSpec
    exact_tensor_names: tuple[str, ...]
    ignored_flash_head_names: tuple[str, ...]
    exact_weight_bytes: int


@dataclass(frozen=True)
class MapleCheckpoint:
    index: WeightIndex
    validation: MapleLayoutValidation

    @property
    def spec(self) -> MapleModelSpec:
        return self.validation.spec

    @property
    def exact_tensor_names(self) -> tuple[str, ...]:
        return self.validation.exact_tensor_names


@dataclass(frozen=True)
class MapleTernaryDeviceWeight:
    weight: Tensor
    row_alpha: Tensor


@dataclass(frozen=True)
class MapleAffine4DeviceWeight:
    weight: Tensor
    scales: Tensor
    biases: Tensor
    group_size: int = 64


@dataclass(frozen=True)
class MapleLayerDeviceWeights:
    input_layernorm: Tensor
    post_attention_layernorm: Tensor
    q_norm: Tensor
    k_norm: Tensor
    router: Tensor
    q_proj: MapleTernaryDeviceWeight
    k_proj: MapleTernaryDeviceWeight
    v_proj: MapleTernaryDeviceWeight
    o_proj: MapleTernaryDeviceWeight
    expert_gate_proj: MapleTernaryDeviceWeight
    expert_up_proj: MapleTernaryDeviceWeight
    expert_down_proj: MapleTernaryDeviceWeight


@dataclass(frozen=True)
class MapleDeviceWeights:
    """Structured exact-path tensors plus their owning allocation map."""

    checkpoint: MapleCheckpoint
    embeddings: MapleAffine4DeviceWeight
    layers: tuple[MapleLayerDeviceWeights, ...]
    final_norm: Tensor
    lm_head: MapleAffine4DeviceWeight
    allocations: DeviceWeightMap

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.allocations.free(runtime=runtime)


def maple_tensor_requirements(spec: MapleModelSpec) -> tuple[MapleTensorRequirement, ...]:
    """Return the complete exact-head tensor manifest in runtime load order."""

    h = spec.hidden_size
    q = spec.q_size
    kv = spec.kv_size
    e = spec.num_experts
    m = spec.moe_intermediate_size
    vocab = spec.vocab_size
    requirements: list[MapleTensorRequirement] = [
        MapleTensorRequirement("model.word_embeddings.weight", "U32", (vocab, h // 8)),
        MapleTensorRequirement("model.word_embeddings.scales", "BF16", (vocab, h // 64)),
        MapleTensorRequirement("model.word_embeddings.biases", "BF16", (vocab, h // 64)),
        MapleTensorRequirement("model.norm.weight", "BF16", (h,)),
    ]
    for layer in range(spec.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        requirements.extend(
            (
                MapleTensorRequirement(f"{prefix}.input_layernorm.weight", "BF16", (h,)),
                MapleTensorRequirement(
                    f"{prefix}.post_attention_layernorm.weight", "BF16", (h,)
                ),
                MapleTensorRequirement(
                    f"{prefix}.self_attn.q_norm.weight", "BF16", (spec.head_dim,)
                ),
                MapleTensorRequirement(
                    f"{prefix}.self_attn.k_norm.weight", "BF16", (spec.head_dim,)
                ),
                MapleTensorRequirement(f"{prefix}.mlp.gate.weight", "BF16", (e, h)),
                *_ternary_requirements(f"{prefix}.self_attn.q_proj", (q, h // 16), (q,)),
                *_ternary_requirements(f"{prefix}.self_attn.k_proj", (kv, h // 16), (kv,)),
                *_ternary_requirements(f"{prefix}.self_attn.v_proj", (kv, h // 16), (kv,)),
                *_ternary_requirements(f"{prefix}.self_attn.o_proj", (h, q // 16), (h,)),
                *_ternary_requirements(
                    f"{prefix}.mlp.switch_mlp.gate_proj", (e, m, h // 16), (e, m)
                ),
                *_ternary_requirements(
                    f"{prefix}.mlp.switch_mlp.up_proj", (e, m, h // 16), (e, m)
                ),
                *_ternary_requirements(
                    f"{prefix}.mlp.switch_mlp.down_proj", (e, h, m // 16), (e, h)
                ),
            )
        )
    requirements.extend(
        (
            MapleTensorRequirement("lm_head.weight", "U32", (vocab, h // 8)),
            MapleTensorRequirement("lm_head.scales", "BF16", (vocab, h // 64)),
            MapleTensorRequirement("lm_head.biases", "BF16", (vocab, h // 64)),
        )
    )
    return tuple(requirements)


def validate_maple_weight_index(index: WeightIndex) -> MapleLayoutValidation:
    """Validate the official exact-head layout; optional FlashHead tensors are ignored."""

    spec = parse_maple_model_spec(index.config)
    requirements = maple_tensor_requirements(spec)
    required_by_name = {requirement.name: requirement for requirement in requirements}
    missing = sorted(set(required_by_name) - set(index.tensors))
    if missing:
        raise MapleLayoutError(_preview("missing Maple tensors", missing))

    errors: list[str] = []
    for name, requirement in required_by_name.items():
        info = index.tensors[name]
        if info.dtype != requirement.dtype:
            errors.append(f"{name}: dtype {info.dtype}, expected {requirement.dtype}")
        if info.shape != requirement.shape:
            errors.append(f"{name}: shape {info.shape}, expected {requirement.shape}")
        if info.nbytes != requirement.nbytes:
            errors.append(f"{name}: {info.nbytes} bytes, expected {requirement.nbytes}")
    if errors:
        raise MapleLayoutError(_preview("invalid Maple tensor metadata", errors))

    ignored = tuple(sorted(name for name in index.tensors if name.startswith(MAPLE_FLASH_HEAD_PREFIX)))
    unexpected = sorted(set(index.tensors) - set(required_by_name) - set(ignored))
    if unexpected:
        raise MapleLayoutError(_preview("unexpected Maple tensors", unexpected))

    exact_bytes = sum(requirement.nbytes for requirement in requirements)
    if len(requirements) != MAPLE_EXACT_TENSOR_COUNT:
        raise AssertionError(
            f"Maple manifest drift: {len(requirements)} tensors, expected {MAPLE_EXACT_TENSOR_COUNT}"
        )
    if exact_bytes != MAPLE_EXACT_WEIGHT_BYTES:
        raise AssertionError(
            f"Maple manifest drift: {exact_bytes} bytes, expected {MAPLE_EXACT_WEIGHT_BYTES}"
        )
    return MapleLayoutValidation(
        spec=spec,
        exact_tensor_names=tuple(requirement.name for requirement in requirements),
        ignored_flash_head_names=ignored,
        exact_weight_bytes=exact_bytes,
    )


def load_maple_checkpoint(model_path: str | Path) -> MapleCheckpoint:
    index = load_weight_index(model_path)
    return MapleCheckpoint(index=index, validation=validate_maple_weight_index(index))


def read_maple_tensor(info: TensorInfo) -> np.ndarray:
    """Read one packed/BF16 tensor as raw NumPy storage without value conversion."""

    dtype = {"U32": np.dtype("<u4"), "BF16": np.dtype("<u2")}.get(info.dtype)
    if dtype is None:
        raise MapleLayoutError(
            f"exact Maple tensor {info.name!r} has unsupported dtype {info.dtype!r}"
        )
    payload = read_tensor_storage_bytes(info)
    return np.frombuffer(payload, dtype=dtype).reshape(info.shape)


def load_maple_tensor_to_device(
    info: TensorInfo,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> DeviceTensorAllocation:
    """Byte-preserving U32/BF16 upload; U32 is represented as runtime INT32 bits."""

    array = read_maple_tensor(info)
    runtime_dtype = DType.INT32 if info.dtype == "U32" else DType.BF16
    prepared = load_host_array_to_device_as_dtype(
        info.name,
        array,
        runtime_dtype,
        source_dtype=info.dtype,
        device=device,
        runtime=runtime,
    )
    return DeviceTensorAllocation(
        name=info.name,
        source=info,
        buffer=prepared.buffer,
        tensor=prepared.tensor,
    )


def materialize_maple_weights(
    checkpoint: MapleCheckpoint,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> MapleDeviceWeights:
    """Upload all 5.31 GB exact-path tensors and return structured layer views."""

    allocations: dict[str, DeviceTensorAllocation] = {}
    try:
        for name in checkpoint.exact_tensor_names:
            allocations[name] = load_maple_tensor_to_device(
                checkpoint.index.tensors[name], device=device, runtime=runtime
            )
        owners = DeviceWeightMap(allocations)
        layers = tuple(
            _layer_device_weights(owners, layer)
            for layer in range(checkpoint.spec.num_hidden_layers)
        )
        return MapleDeviceWeights(
            checkpoint=checkpoint,
            embeddings=_affine4_device_weight(owners, "model.word_embeddings"),
            layers=layers,
            final_norm=owners["model.norm.weight"],
            lm_head=_affine4_device_weight(owners, "lm_head"),
            allocations=owners,
        )
    except Exception:
        DeviceWeightMap(allocations).free(runtime=runtime)
        raise


def _ternary_requirements(
    prefix: str,
    weight_shape: tuple[int, ...],
    alpha_shape: tuple[int, ...],
) -> tuple[MapleTensorRequirement, MapleTensorRequirement]:
    return (
        MapleTensorRequirement(f"{prefix}.weight", "U32", weight_shape),
        MapleTensorRequirement(f"{prefix}.row_alpha", "BF16", alpha_shape),
    )


def _ternary_device_weight(owners: DeviceWeightMap, prefix: str) -> MapleTernaryDeviceWeight:
    return MapleTernaryDeviceWeight(
        weight=owners[f"{prefix}.weight"],
        row_alpha=owners[f"{prefix}.row_alpha"],
    )


def _affine4_device_weight(owners: DeviceWeightMap, prefix: str) -> MapleAffine4DeviceWeight:
    return MapleAffine4DeviceWeight(
        weight=owners[f"{prefix}.weight"],
        scales=owners[f"{prefix}.scales"],
        biases=owners[f"{prefix}.biases"],
    )


def _layer_device_weights(owners: DeviceWeightMap, layer: int) -> MapleLayerDeviceWeights:
    prefix = f"model.layers.{layer}"
    return MapleLayerDeviceWeights(
        input_layernorm=owners[f"{prefix}.input_layernorm.weight"],
        post_attention_layernorm=owners[f"{prefix}.post_attention_layernorm.weight"],
        q_norm=owners[f"{prefix}.self_attn.q_norm.weight"],
        k_norm=owners[f"{prefix}.self_attn.k_norm.weight"],
        router=owners[f"{prefix}.mlp.gate.weight"],
        q_proj=_ternary_device_weight(owners, f"{prefix}.self_attn.q_proj"),
        k_proj=_ternary_device_weight(owners, f"{prefix}.self_attn.k_proj"),
        v_proj=_ternary_device_weight(owners, f"{prefix}.self_attn.v_proj"),
        o_proj=_ternary_device_weight(owners, f"{prefix}.self_attn.o_proj"),
        expert_gate_proj=_ternary_device_weight(
            owners, f"{prefix}.mlp.switch_mlp.gate_proj"
        ),
        expert_up_proj=_ternary_device_weight(owners, f"{prefix}.mlp.switch_mlp.up_proj"),
        expert_down_proj=_ternary_device_weight(
            owners, f"{prefix}.mlp.switch_mlp.down_proj"
        ),
    )


def _preview(label: str, values: Iterable[str], *, limit: int = 8) -> str:
    values = tuple(values)
    shown = ", ".join(values[:limit])
    more = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
    return f"{label}: {shown}{more}"


__all__ = [
    "MAPLE_EXACT_TENSOR_COUNT",
    "MAPLE_EXACT_WEIGHT_BYTES",
    "MAPLE_FLASH_HEAD_PREFIX",
    "MapleAffine4DeviceWeight",
    "MapleCheckpoint",
    "MapleDeviceWeights",
    "MapleLayerDeviceWeights",
    "MapleLayoutError",
    "MapleLayoutValidation",
    "MapleTensorRequirement",
    "MapleTernaryDeviceWeight",
    "load_maple_checkpoint",
    "load_maple_tensor_to_device",
    "maple_tensor_requirements",
    "materialize_maple_weights",
    "read_maple_tensor",
    "validate_maple_weight_index",
]
