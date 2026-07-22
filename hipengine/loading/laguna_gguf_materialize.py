"""Dry resident-layout and unified-memory admission planning for Laguna GGUF."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.laguna_gguf import LagunaGGUFConfig, LagunaGGUFModelMap
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import GGUF_Q4_K_TILE16_BLOCK_BYTES
from hipengine.quant.gguf_t16 import GGUF_Q6_K_T16_BLOCK_BYTES

LAYOUT_DENSE_F16 = "dense_f16"
LAYOUT_DENSE_F32 = "dense_f32"
LAYOUT_RAW_GGUF = "raw_gguf"
LAYOUT_Q4_K_PACK8 = "q4_k_pack8"
LAYOUT_GGUF_Q4_K_T16 = "gguf_q4_k_t16_v1"
LAYOUT_GGUF_Q6_K_T16 = "gguf_q6_k_t16_v1"

DEFAULT_LAGUNA_SCRATCH_BYTES = 2 * 2**30
DEFAULT_LAGUNA_SAFETY_RESERVE_BYTES = 8 * 2**30
_GGUF_K_BLOCK = 256
_T16_COLUMNS = 16


class LagunaMemoryAdmissionError(MemoryError):
    """Raised before allocation when a Laguna resident plan exceeds its budget."""


@dataclass(frozen=True)
class LagunaGGUFWeightSpec:
    """One source tensor and its planned replacement resident allocations."""

    slot_path: str
    source: GGUFTensorInfo
    quant_key: str
    layout: str
    resident_dtype: str
    allocation_nbytes: Mapping[str, int]

    @property
    def allocation_names(self) -> tuple[str, ...]:
        return tuple(self.allocation_nbytes)

    @property
    def resident_nbytes(self) -> int:
        return sum(int(value) for value in self.allocation_nbytes.values())

    @property
    def loader_transient_nbytes(self) -> int:
        """Worst host source+replacement bytes while this tensor is converted."""

        return int(self.source.nbytes) + self.resident_nbytes


@dataclass(frozen=True)
class LagunaGGUFMaterializationPlan:
    """Validated resident replacement layouts for every Laguna weight."""

    config: LagunaGGUFConfig
    root_specs: Mapping[str, LagunaGGUFWeightSpec]
    layer_specs: tuple[Mapping[str, LagunaGGUFWeightSpec], ...]

    @property
    def specs(self) -> tuple[LagunaGGUFWeightSpec, ...]:
        return (
            *tuple(self.root_specs.values()),
            *tuple(spec for layer in self.layer_specs for spec in layer.values()),
        )

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(spec.source.name for spec in self.specs)

    @property
    def source_nbytes(self) -> int:
        return sum(int(spec.source.nbytes) for spec in self.specs)

    @property
    def resident_nbytes(self) -> int:
        return sum(spec.resident_nbytes for spec in self.specs)

    @property
    def max_loader_transient_nbytes(self) -> int:
        return max((spec.loader_transient_nbytes for spec in self.specs), default=0)

    @property
    def layout_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(sorted(Counter(spec.layout for spec in self.specs).items())))

    @property
    def precision_contractions(self) -> tuple[str, ...]:
        contractions: list[str] = []
        for spec in self.specs:
            source_type = GGMLQuantizationType(spec.source.ggml_type)
            if source_type == GGMLQuantizationType.F16 and spec.resident_dtype != "fp16":
                contractions.append(spec.slot_path)
            if source_type == GGMLQuantizationType.F32 and spec.resident_dtype != "fp32":
                contractions.append(spec.slot_path)
        return tuple(contractions)


@dataclass(frozen=True)
class LagunaKVMemoryPlan:
    context_length: int
    global_layer_count: int
    sliding_layer_count: int
    global_tokens_per_layer: int
    sliding_tokens_per_layer: int
    bytes_per_layer_token: int
    resident_nbytes: int
    storage_dtype: str


@dataclass(frozen=True)
class LagunaMemoryAdmissionPlan:
    weights: LagunaGGUFMaterializationPlan
    kv: LagunaKVMemoryPlan
    available_nbytes: int
    scratch_nbytes: int
    safety_reserve_nbytes: int
    loader_transient_nbytes: int
    peak_required_nbytes: int
    headroom_bytes: int

    @property
    def passed(self) -> bool:
        return self.headroom_bytes >= 0


def plan_laguna_gguf_materialization(
    model_map: LagunaGGUFModelMap,
) -> LagunaGGUFMaterializationPlan:
    """Plan all resident weights without reading payload bytes or allocating."""

    model_map.validation.raise_for_errors()
    roots = {
        slot: _spec_for_tensor(f"root.{slot}", tensor)
        for slot, tensor in model_map.root_tensors.items()
    }
    layers = tuple(
        MappingProxyType(
            {
                slot: _spec_for_tensor(f"layers.{layer.layer_id}.{slot}", tensor)
                for slot, tensor in layer.tensors.items()
            }
        )
        for layer in model_map.layers
    )
    plan = LagunaGGUFMaterializationPlan(
        config=model_map.config,
        root_specs=MappingProxyType(roots),
        layer_specs=layers,
    )
    expected = set(model_map.tensor_names)
    actual = set(plan.tensor_names)
    if len(plan.tensor_names) != len(actual) or actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Laguna materialization plan must consume every mapped tensor exactly once; "
            f"missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    if plan.precision_contractions:
        raise ValueError(
            "Laguna materialization plan contracts source F16/F32 tensors: "
            f"{plan.precision_contractions[:4]}"
        )
    return plan


def plan_laguna_memory_admission(
    weights: LagunaGGUFMaterializationPlan,
    *,
    context_length: int,
    available_bytes: int,
    storage_dtype: str = "bf16",
    scratch_nbytes: int = DEFAULT_LAGUNA_SCRATCH_BYTES,
    safety_reserve_nbytes: int = DEFAULT_LAGUNA_SAFETY_RESERVE_BYTES,
    honor_sliding_window: bool = True,
) -> LagunaMemoryAdmissionPlan:
    """Calculate peak UMA demand and reject over-budget plans before allocation."""

    context = int(context_length)
    available = int(available_bytes)
    scratch = int(scratch_nbytes)
    reserve = int(safety_reserve_nbytes)
    if context <= 0 or context > weights.config.context_length:
        raise ValueError(
            f"context_length must be within [1, {weights.config.context_length}]"
        )
    if available <= 0:
        raise ValueError("available_bytes must be positive")
    if scratch < 0 or reserve < 0:
        raise ValueError("scratch and safety reserve must be non-negative")
    dtype = str(storage_dtype).lower()
    if dtype not in {"bf16", "fp16"}:
        raise ValueError("initial Laguna KV storage_dtype must be 'bf16' or 'fp16'")
    if (
        not honor_sliding_window
        and weights.config.sliding_window > 0
        and weights.config.layer_types.count("sliding_attention") > 0
    ):
        raise ValueError(
            "Laguna all-layers-full-KV planning is rejected; SWA layers must use the ring"
        )

    kv = _plan_kv_memory(weights.config, context_length=context, storage_dtype=dtype)
    transient = weights.max_loader_transient_nbytes
    peak = (
        weights.resident_nbytes
        + kv.resident_nbytes
        + scratch
        + reserve
        + transient
    )
    headroom = available - peak
    result = LagunaMemoryAdmissionPlan(
        weights=weights,
        kv=kv,
        available_nbytes=available,
        scratch_nbytes=scratch,
        safety_reserve_nbytes=reserve,
        loader_transient_nbytes=transient,
        peak_required_nbytes=peak,
        headroom_bytes=headroom,
    )
    if not result.passed:
        raise LagunaMemoryAdmissionError(
            "Laguna peak memory plan exceeds available UMA before allocation: "
            f"required={peak} available={available} deficit={-headroom}"
        )
    return result


def _plan_kv_memory(
    config: LagunaGGUFConfig,
    *,
    context_length: int,
    storage_dtype: str,
) -> LagunaKVMemoryPlan:
    global_layers = config.layer_types.count("full_attention")
    sliding_layers = config.layer_types.count("sliding_attention")
    sliding_tokens = min(context_length, config.sliding_window) if sliding_layers else 0
    element_nbytes = 2
    bytes_per_layer_token = (
        config.head_count_kv * (config.key_length + config.value_length) * element_nbytes
    )
    resident = bytes_per_layer_token * (
        global_layers * context_length + sliding_layers * sliding_tokens
    )
    return LagunaKVMemoryPlan(
        context_length=context_length,
        global_layer_count=global_layers,
        sliding_layer_count=sliding_layers,
        global_tokens_per_layer=context_length,
        sliding_tokens_per_layer=sliding_tokens,
        bytes_per_layer_token=bytes_per_layer_token,
        resident_nbytes=resident,
        storage_dtype=storage_dtype,
    )


def _spec_for_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> LagunaGGUFWeightSpec:
    qtype = GGMLQuantizationType(tensor.ggml_type)
    if qtype == GGMLQuantizationType.F32:
        return _spec(
            slot_path,
            tensor,
            quant_key="f32",
            layout=LAYOUT_DENSE_F32,
            resident_dtype="fp32",
            allocations={"raw": tensor.nbytes},
        )
    if qtype == GGMLQuantizationType.F16:
        return _spec(
            slot_path,
            tensor,
            quant_key="fp16",
            layout=LAYOUT_DENSE_F16,
            resident_dtype="fp16",
            allocations={"raw": tensor.nbytes},
        )
    if qtype == GGMLQuantizationType.Q4_K:
        if slot_path == "root.token_embedding":
            return _spec(
                slot_path,
                tensor,
                quant_key="gguf_q4_k",
                layout=LAYOUT_RAW_GGUF,
                resident_dtype="q4_k",
                allocations={"raw": tensor.nbytes},
            )
        if len(tensor.shape) == 3:
            allocations = {"tiles": _q4_k_t16_nbytes(tensor)}
            return _spec(
                slot_path,
                tensor,
                quant_key="gguf_q4_k_t16_v1",
                layout=LAYOUT_GGUF_Q4_K_T16,
                resident_dtype="q4_k_t16",
                allocations=allocations,
            )
        if len(tensor.shape) == 2:
            return _spec(
                slot_path,
                tensor,
                quant_key="gguf_q4_k",
                layout=LAYOUT_Q4_K_PACK8,
                resident_dtype="q4_k_pack8",
                allocations=_q4_k_pack8_allocations(tensor),
            )
    if qtype == GGMLQuantizationType.Q6_K:
        if slot_path == "root.lm_head" or len(tensor.shape) == 3:
            return _spec(
                slot_path,
                tensor,
                quant_key="gguf_q6_k_t16_v1",
                layout=LAYOUT_GGUF_Q6_K_T16,
                resident_dtype="q6_k_t16",
                allocations={"tiles": _q6_k_t16_nbytes(tensor)},
            )
        if len(tensor.shape) == 2:
            return _spec(
                slot_path,
                tensor,
                quant_key="gguf_q6_k",
                layout=LAYOUT_RAW_GGUF,
                resident_dtype="q6_k",
                allocations={"raw": tensor.nbytes},
            )
    raise ValueError(
        f"unsupported Laguna resident tensor contract {tensor.ggml_type_name} "
        f"shape={tensor.shape}: {slot_path}"
    )


def _spec(
    slot_path: str,
    tensor: GGUFTensorInfo,
    *,
    quant_key: str,
    layout: str,
    resident_dtype: str,
    allocations: Mapping[str, int],
) -> LagunaGGUFWeightSpec:
    if any(int(value) <= 0 for value in allocations.values()):
        raise ValueError(f"Laguna allocation sizes must be positive: {slot_path}")
    return LagunaGGUFWeightSpec(
        slot_path=slot_path,
        source=tensor,
        quant_key=quant_key,
        layout=layout,
        resident_dtype=resident_dtype,
        allocation_nbytes=MappingProxyType(
            {name: int(value) for name, value in allocations.items()}
        ),
    )


def _q4_k_pack8_allocations(tensor: GGUFTensorInfo) -> Mapping[str, int]:
    out_features, in_features = tensor.shape
    if out_features % 8 or in_features % 32:
        raise ValueError(
            f"Q4_K pack8 requires out%8==0 and in%32==0: {tensor.name} {tensor.shape}"
        )
    return {
        "qweight": (out_features // 8) * in_features * 4,
        "scales": (in_features // 32) * out_features * 4,
        "mins": (in_features // 32) * out_features * 4,
    }


def _q4_k_t16_nbytes(tensor: GGUFTensorInfo) -> int:
    experts, out_features, in_features = tensor.shape
    if out_features % _T16_COLUMNS or in_features % _GGUF_K_BLOCK:
        raise ValueError(f"Q4_K T16 shape unsupported: {tensor.name} {tensor.shape}")
    return (
        experts
        * (out_features // _T16_COLUMNS)
        * (in_features // _GGUF_K_BLOCK)
        * GGUF_Q4_K_TILE16_BLOCK_BYTES
    )


def _q6_k_t16_nbytes(tensor: GGUFTensorInfo) -> int:
    if len(tensor.shape) == 2:
        experts = 1
        out_features, in_features = tensor.shape
    elif len(tensor.shape) == 3:
        experts, out_features, in_features = tensor.shape
    else:
        raise ValueError(f"Q6_K T16 requires rank 2 or 3: {tensor.name}")
    if out_features % _T16_COLUMNS or in_features % _GGUF_K_BLOCK:
        raise ValueError(f"Q6_K T16 shape unsupported: {tensor.name} {tensor.shape}")
    return (
        experts
        * (out_features // _T16_COLUMNS)
        * (in_features // _GGUF_K_BLOCK)
        * GGUF_Q6_K_T16_BLOCK_BYTES
    )


__all__ = [
    "DEFAULT_LAGUNA_SAFETY_RESERVE_BYTES",
    "DEFAULT_LAGUNA_SCRATCH_BYTES",
    "LAYOUT_DENSE_F16",
    "LAYOUT_DENSE_F32",
    "LAYOUT_GGUF_Q4_K_T16",
    "LAYOUT_GGUF_Q6_K_T16",
    "LAYOUT_Q4_K_PACK8",
    "LAYOUT_RAW_GGUF",
    "LagunaGGUFMaterializationPlan",
    "LagunaGGUFWeightSpec",
    "LagunaKVMemoryPlan",
    "LagunaMemoryAdmissionError",
    "LagunaMemoryAdmissionPlan",
    "plan_laguna_gguf_materialization",
    "plan_laguna_memory_admission",
]
