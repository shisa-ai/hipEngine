"""Device materialization for Qwen3.5 GGUF tensor maps."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipError, HipRuntime
from hipengine.core.memory import DeviceBuffer, DeviceMemoryArena, free, malloc
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.policy import GGUFModelGeometry
from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo
from hipengine.loading.materialize import (
    DeviceBufferAllocator,
    DeviceTensorAllocation,
    float_array_to_bf16_bits,
    load_host_array_to_device_as_dtype as _load_host_array_to_device_as_dtype,
)
from hipengine.loading.qwen35_gguf import (
    Qwen35GGUFConfig,
    Qwen35GGUFLayerMap,
    Qwen35GGUFModelMap,
    build_qwen35_gguf_tensor_map,
)
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.quant.gguf_q4_k import (
    GGUF_Q4_K_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_QMICRO_BLOCK_BYTES,
    repack_gguf_q4_k_pack8,
    repack_gguf_q4_k_tile16,
    repack_gguf_q4_k_tile16_qmicro,
)
from hipengine.quant.gguf_t16 import (
    GGUF_Q5_K_BLOCK_BYTES,
    GGUF_Q5_K_T16_BLOCK_BYTES,
    GGUF_Q6_K_BLOCK_BYTES,
    GGUF_Q6_K_T16_BLOCK_BYTES,
    GGUF_Q8_0_BLOCK_BYTES,
    GGUF_Q8_0_T16_BLOCK_BYTES,
    GGUF_T16_COLS,
    repack_gguf_q5_k_qmicro_tile16,
    repack_gguf_q5_k_tile16,
    repack_gguf_q6_k_tile16,
    repack_gguf_q6_k_tile16_qmicro_planar,
    repack_gguf_q8_0_tile16,
)
from hipengine.quant.gguf_x8 import repack_gguf_q4_k_x8, repack_gguf_q5_k_x8, repack_gguf_q6_k_x8

LAYOUT_DENSE_F32 = "dense_f32"
LAYOUT_DENSE_BF16 = "dense_bf16"
LAYOUT_RAW_GGUF = "raw_gguf"
LAYOUT_Q4_K_PACK8 = "q4_k_pack8"
LAYOUT_GGUF_EXPERT_PACK8_SIDECAR = "gguf_expert_pack8_v1"
LAYOUT_GGUF_Q4_K_T16 = "gguf_q4_k_t16_v1"
LAYOUT_GGUF_Q4_K_QMICRO_T16 = "gguf_q4_k_qmicro_t16_v1"
LAYOUT_GGUF_Q4_K_X8 = "gguf_q4_k_x8_v1"
LAYOUT_GGUF_Q5_K_T16 = "gguf_q5_k_t16_v1"
LAYOUT_GGUF_Q5_K_QMICRO_T16 = "gguf_q5_k_qmicro_t16_v1"
LAYOUT_GGUF_Q6_K_T16 = "gguf_q6_k_t16_v1"
LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR = "gguf_q6_k_t16_qmicro_planar_v1"
LAYOUT_GGUF_Q8_0_T16 = "gguf_q8_0_t16_v1"
LAYOUT_GGUF_Q5_K_X8 = "gguf_q5_k_x8_v1"
LAYOUT_GGUF_Q6_K_X8 = "gguf_q6_k_x8_v1"
HIPENGINE_GGUF_DECODE_REPACK_ENV = "HIPENGINE_GGUF_DECODE_REPACK"
HIPENGINE_GGUF_SELECTED_X8_REPACK_ENV = "HIPENGINE_GGUF_SELECTED_X8_REPACK"
Q4_T16_DECODE_TILES = "decode_tiles"
Q4_T16_DECODE_TILES_R3PLUS = "decode_tiles_r3plus"
HIPENGINE_GGUF_SELECTED_DOWN_RAW_ENV = "HIPENGINE_GGUF_SELECTED_DOWN_RAW"
HIPENGINE_GGUF_SELECTED_GATE_UP_RAW_ENV = "HIPENGINE_GGUF_SELECTED_GATE_UP_RAW"
HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV = "HIPENGINE_GGUF_SELECTED_GATE_UP_X8"
HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV = "HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"
HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL_ENV = "HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL"
HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR_ENV = "HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR"
GGUF_SELECTIVE_WEIGHT_ARENA_MAX_ALLOCATION_BYTES = 16 * 1024 * 1024
GGUF_SELECTIVE_WEIGHT_ARENA_ALIGNMENT = 4096


@dataclass(frozen=True)
class Qwen35GGUFWeightSpec:
    """One planned resident GGUF weight record."""

    slot_path: str
    source: GGUFTensorInfo
    quant_key: str
    layout: str
    allocation_names: tuple[str, ...]
    sidecar_layouts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Qwen35GGUFPrecisionContraction:
    """Diagnostic record for a source GGUF tensor planned at lower precision."""

    slot_path: str
    source_name: str
    source_type: str
    resident_layout: str
    resident_quant_key: str
    llama_cpp_contract: str
    hipengine_contract: str


@dataclass(frozen=True)
class Qwen35GGUFMaterializationPlan:
    """Resident-weight layout plan derived from a validated tensor map."""

    config: Qwen35GGUFConfig
    root_specs: Mapping[str, Qwen35GGUFWeightSpec]
    layer_specs: tuple[Mapping[str, Qwen35GGUFWeightSpec], ...]

    @property
    def specs(self) -> tuple[Qwen35GGUFWeightSpec, ...]:
        specs: list[Qwen35GGUFWeightSpec] = []
        seen: set[tuple[str, str]] = set()
        for spec in self.root_specs.values():
            key = (spec.source.name, spec.layout)
            if key not in seen:
                seen.add(key)
                specs.append(spec)
        for layer in self.layer_specs:
            for spec in layer.values():
                key = (spec.source.name, spec.layout)
                if key not in seen:
                    seen.add(key)
                    specs.append(spec)
        return tuple(specs)

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(spec.source.name for spec in self.specs)


@dataclass(frozen=True)
class Qwen35GGUFSelectiveWeightArenaPlan:
    """Exact metadata-derived plan for the bounded small-weight owner."""

    supported: bool
    alignment: int
    max_allocation_bytes: int
    requested_bytes: int
    capacity_bytes: int
    allocation_count: int
    dedicated_requested_bytes: int
    dedicated_allocation_count: int
    allocation_nbytes: tuple[tuple[str, str, int], ...]
    reason: str | None = None


@dataclass(frozen=True)
class Qwen35GGUFDeviceWeight:
    """Owned device allocations for one logical GGUF weight."""

    spec: Qwen35GGUFWeightSpec
    allocations: Mapping[str, DeviceTensorAllocation]
    backend: str

    def allocation(self, name: str = "raw") -> DeviceTensorAllocation:
        return self.allocations[name]

    def has_allocation(self, name: str) -> bool:
        return name in self.allocations

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for allocation in reversed(tuple(self.allocations.values())):
            allocation.free(runtime=runtime)


@dataclass(frozen=True)
class Qwen35GGUFResidentLayerWeights:
    layer_id: int
    layer_type: str
    weights: Mapping[str, Qwen35GGUFDeviceWeight]

    def weight(self, slot: str) -> Qwen35GGUFDeviceWeight:
        return self.weights[slot]


@dataclass(frozen=True)
class Qwen35GGUFResidentWeights:
    """Device-resident Qwen3.5 GGUF weights.

    ``root_weights['lm_head']`` aliases ``root_weights['token_embedding']`` for
    the local tied-output GGUF. Selective routes keep non-owning small-weight
    views and free their one arena owner here; larger allocations remain owning.
    """

    config: Qwen35GGUFConfig
    root_weights: Mapping[str, Qwen35GGUFDeviceWeight]
    layers: tuple[Qwen35GGUFResidentLayerWeights, ...]
    backend: str
    geometry: GGUFModelGeometry | None = None
    model_name: str | None = None
    file_type_name: str | None = None
    allocation_arena: DeviceMemoryArena | None = None
    allocation_mode: str = "dedicated"
    allocation_arena_reason: str | None = None

    def root(self, slot: str) -> Qwen35GGUFDeviceWeight:
        return self.root_weights[slot]

    def layer(self, layer_id: int) -> Qwen35GGUFResidentLayerWeights:
        return self.layers[layer_id]

    @property
    def weights(self) -> tuple[Qwen35GGUFDeviceWeight, ...]:
        weights: list[Qwen35GGUFDeviceWeight] = []
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

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for weight in reversed(self.weights):
            weight.free(runtime=runtime)
        if self.allocation_arena is not None:
            self.allocation_arena.close()


class _SelectiveWeightAllocator(DeviceBufferAllocator):
    """Route only bounded allocations into one arena; keep large owners dedicated."""

    def __init__(
        self,
        arena: DeviceMemoryArena,
        *,
        max_allocation_bytes: int,
        runtime: HipRuntime | None,
    ) -> None:
        self.arena = arena
        self.max_allocation_bytes = int(max_allocation_bytes)
        self.runtime = runtime

    def allocate(self, nbytes: int) -> DeviceBuffer:
        if int(nbytes) <= self.max_allocation_bytes:
            return self.arena.allocate(nbytes)
        return malloc(nbytes, runtime=self.runtime)

    def owns(self, buffer: DeviceBuffer) -> bool:
        return self.arena.owns(buffer)

    def release(self, buffer: DeviceBuffer) -> None:
        if self.owns(buffer):
            self.arena.release(buffer)
        else:
            free(buffer, runtime=self.runtime)


def plan_qwen35_gguf_materialization(
    model_map: Qwen35GGUFModelMap,
    *,
    decode_repack: bool | None = None,
    dense_q4_t16: bool = False,
    dense_q4_qmicro_t16_gate_up: bool = False,
    dense_q4_t16_attn_q_08b: bool = False,
    dense_q5_t16_ssm_out: bool = False,
    dense_q5_raw_mmq_ssm_out: bool = False,
    dense_q5_t16_ssm_out_08b: bool = False,
    dense_q5_t16_qkv: bool = False,
    dense_q5_t16_h5120: bool = False,
    dense_q6_qmicro_planar: bool = False,
    dense_q6_qmicro_planar_excluded_slots: Iterable[str] = (),
) -> Qwen35GGUFMaterializationPlan:
    requested_decode_repack = gguf_decode_repack_enabled(decode_repack)
    q6_planar_excluded = frozenset(
        str(slot) for slot in dense_q6_qmicro_planar_excluded_slots
    )
    contract_q3_f32_linear = any(
        GGMLQuantizationType(tensor.ggml_type)
        in {
            GGMLQuantizationType.IQ2_XS,
            GGMLQuantizationType.IQ3_XXS,
            GGMLQuantizationType.IQ4_XS,
        }
        for layer in model_map.layers
        for tensor in layer.tensors.values()
    )
    # Raw-IQ models' selected kernels consume compressed rank-3 GGUF layouts.
    # Keep one compatible resident plan instead of silently mixing it with the
    # Q4-oriented T16 decode residents.
    use_decode_repack = requested_decode_repack and not contract_q3_f32_linear
    root_specs = {
        slot: _spec_for_tensor(
            f"root.{slot}",
            tensor,
            decode_repack=use_decode_repack,
            contract_f32_linear=contract_q3_f32_linear,
            dense_q4_t16=bool(dense_q4_t16),
            dense_q4_qmicro_t16_gate_up=bool(dense_q4_qmicro_t16_gate_up),
            dense_q4_t16_attn_q_08b=bool(dense_q4_t16_attn_q_08b),
            dense_q5_t16_ssm_out=bool(dense_q5_t16_ssm_out),
            dense_q5_raw_mmq_ssm_out=bool(dense_q5_raw_mmq_ssm_out),
            dense_q5_t16_ssm_out_08b=bool(dense_q5_t16_ssm_out_08b),
            dense_q5_t16_qkv=bool(dense_q5_t16_qkv),
            dense_q5_t16_h5120=bool(dense_q5_t16_h5120),
            dense_q6_qmicro_planar=bool(dense_q6_qmicro_planar),
            dense_q6_qmicro_planar_excluded_slots=q6_planar_excluded,
        )
        for slot, tensor in model_map.root_tensors.items()
    }
    layer_specs = tuple(
        _plan_layer(
            layer,
            decode_repack=use_decode_repack,
            contract_f32_linear=contract_q3_f32_linear,
            dense_q4_t16=bool(dense_q4_t16),
            dense_q4_qmicro_t16_gate_up=bool(dense_q4_qmicro_t16_gate_up),
            dense_q4_t16_attn_q_08b=bool(dense_q4_t16_attn_q_08b),
            dense_q5_t16_ssm_out=bool(dense_q5_t16_ssm_out),
            dense_q5_raw_mmq_ssm_out=bool(dense_q5_raw_mmq_ssm_out),
            dense_q5_t16_ssm_out_08b=bool(dense_q5_t16_ssm_out_08b),
            dense_q5_t16_qkv=bool(dense_q5_t16_qkv),
            dense_q5_t16_h5120=bool(dense_q5_t16_h5120),
            dense_q6_qmicro_planar=bool(dense_q6_qmicro_planar),
            dense_q6_qmicro_planar_excluded_slots=q6_planar_excluded,
        )
        for layer in model_map.layers
    )
    return Qwen35GGUFMaterializationPlan(
        config=model_map.config,
        root_specs=MappingProxyType(root_specs),
        layer_specs=tuple(MappingProxyType(layer) for layer in layer_specs),
    )


def plan_qwen35_gguf_selective_weight_arena(
    plan: Qwen35GGUFMaterializationPlan,
    *,
    selected_slots: Iterable[str] | None = None,
    deferred_device_slots: Iterable[str] | None = None,
    alignment: int = GGUF_SELECTIVE_WEIGHT_ARENA_ALIGNMENT,
    max_allocation_bytes: int = GGUF_SELECTIVE_WEIGHT_ARENA_MAX_ALLOCATION_BYTES,
) -> Qwen35GGUFSelectiveWeightArenaPlan:
    """Plan only bounded small-weight views and fail closed on unknown layouts."""

    parsed_alignment = _validate_arena_alignment(alignment)
    parsed_max = int(max_allocation_bytes)
    if parsed_max <= 0:
        raise ValueError("selective weight arena max allocation must be positive")
    selected = None if selected_slots is None else {str(slot) for slot in selected_slots}
    deferred = set() if deferred_device_slots is None else {str(slot) for slot in deferred_device_slots}
    records: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    requested_bytes = 0
    capacity_bytes = 0
    dedicated_requested_bytes = 0
    dedicated_allocation_count = 0
    specs = (
        *tuple(plan.root_specs.values()),
        *(spec for layer in plan.layer_specs for spec in layer.values()),
    )
    try:
        for spec in specs:
            if selected is not None and spec.slot_path not in selected:
                continue
            if spec.slot_path in deferred:
                continue
            key = (spec.source.name, spec.layout)
            if key in seen:
                continue
            seen.add(key)
            if spec.layout == LAYOUT_Q4_K_PACK8:
                raise ValueError(
                    f"unsupported selective arena layout {spec.layout!r} for {spec.slot_path}"
                )
            for allocation_name, nbytes in _planned_weight_allocation_nbytes(spec):
                if nbytes <= parsed_max:
                    capacity_bytes = _align_arena(capacity_bytes, parsed_alignment) + nbytes
                    requested_bytes += nbytes
                    records.append((spec.source.name, allocation_name, int(nbytes)))
                else:
                    dedicated_requested_bytes += nbytes
                    dedicated_allocation_count += 1
    except (TypeError, ValueError) as exc:
        return Qwen35GGUFSelectiveWeightArenaPlan(
            supported=False,
            alignment=parsed_alignment,
            max_allocation_bytes=parsed_max,
            requested_bytes=0,
            capacity_bytes=0,
            allocation_count=0,
            dedicated_requested_bytes=0,
            dedicated_allocation_count=0,
            allocation_nbytes=(),
            reason=str(exc),
        )
    return Qwen35GGUFSelectiveWeightArenaPlan(
        supported=True,
        alignment=parsed_alignment,
        max_allocation_bytes=parsed_max,
        requested_bytes=requested_bytes,
        capacity_bytes=_align_arena(capacity_bytes, parsed_alignment),
        allocation_count=len(records),
        dedicated_requested_bytes=dedicated_requested_bytes,
        dedicated_allocation_count=dedicated_allocation_count,
        allocation_nbytes=tuple(records),
    )


def planned_qwen35_gguf_weight_allocation_nbytes(
    spec: Qwen35GGUFWeightSpec,
) -> tuple[tuple[str, int], ...]:
    """Return exact metadata-only device bytes for one resident weight spec."""

    source = spec.source
    primary_nbytes: int
    if spec.layout == LAYOUT_Q4_K_PACK8:
        if len(source.shape) != 2:
            raise ValueError(f"Q4 pack8 plan requires rank-2 storage for {spec.slot_path}")
        out_features, in_features = (int(dim) for dim in source.shape)
        if out_features % 8 or in_features % 256:
            raise ValueError(f"Q4 pack8 shape is not tile-aligned for {spec.slot_path}")
        pack8_nbytes = {
            "qweight": out_features * in_features // 2,
            "scales": out_features * in_features // 8,
            "mins": out_features * in_features // 8,
        }
        records: list[tuple[str, int]] = []
        for allocation_name in spec.allocation_names:
            if allocation_name in pack8_nbytes:
                nbytes = pack8_nbytes[allocation_name]
            elif allocation_name in {Q4_T16_DECODE_TILES, Q4_T16_DECODE_TILES_R3PLUS}:
                nbytes = _planned_t16_nbytes(
                    source,
                    block_bytes=GGUF_Q4_K_BLOCK_BYTES,
                    tile_block_bytes=GGUF_Q4_K_TILE16_BLOCK_BYTES,
                    slot_path=spec.slot_path,
                )
            else:
                raise ValueError(
                    f"unsupported Q4 pack8 allocation {allocation_name!r} for {spec.slot_path}"
                )
            records.append((allocation_name, int(nbytes)))
        return tuple(records)
    if spec.layout == LAYOUT_DENSE_F32:
        primary_nbytes = int(source.n_elements) * DType.FP32.itemsize
    elif spec.layout == LAYOUT_DENSE_BF16:
        primary_nbytes = int(source.n_elements) * DType.BF16.itemsize
    elif spec.layout == LAYOUT_GGUF_Q4_K_T16:
        primary_nbytes = _planned_t16_nbytes(
            source,
            block_bytes=GGUF_Q4_K_BLOCK_BYTES,
            tile_block_bytes=GGUF_Q4_K_TILE16_BLOCK_BYTES,
            slot_path=spec.slot_path,
        )
    elif spec.layout == LAYOUT_GGUF_Q4_K_QMICRO_T16:
        primary_nbytes = _planned_t16_nbytes(
            source,
            block_bytes=GGUF_Q4_K_BLOCK_BYTES,
            tile_block_bytes=GGUF_Q4_K_TILE16_QMICRO_BLOCK_BYTES,
            slot_path=spec.slot_path,
        )
    elif spec.layout == LAYOUT_GGUF_Q5_K_T16:
        primary_nbytes = _planned_t16_nbytes(
            source,
            block_bytes=GGUF_Q5_K_BLOCK_BYTES,
            tile_block_bytes=GGUF_Q5_K_T16_BLOCK_BYTES,
            slot_path=spec.slot_path,
        )
    elif spec.layout == LAYOUT_GGUF_Q6_K_T16:
        primary_nbytes = _planned_t16_nbytes(
            source,
            block_bytes=GGUF_Q6_K_BLOCK_BYTES,
            tile_block_bytes=GGUF_Q6_K_T16_BLOCK_BYTES,
            slot_path=spec.slot_path,
        )
    elif spec.layout == LAYOUT_GGUF_Q8_0_T16:
        primary_nbytes = _planned_t16_nbytes(
            source,
            block_bytes=GGUF_Q8_0_BLOCK_BYTES,
            tile_block_bytes=GGUF_Q8_0_T16_BLOCK_BYTES,
            slot_path=spec.slot_path,
        )
    elif spec.layout in {
        LAYOUT_RAW_GGUF,
        LAYOUT_GGUF_Q4_K_X8,
        LAYOUT_GGUF_Q5_K_QMICRO_T16,
        LAYOUT_GGUF_Q5_K_X8,
        LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        LAYOUT_GGUF_Q6_K_X8,
    }:
        primary_nbytes = int(source.nbytes)
    else:
        raise ValueError(f"unsupported resident layout {spec.layout!r} for {spec.slot_path}")

    records = []
    for allocation_name in spec.allocation_names:
        if allocation_name in {"tiles", "raw"}:
            nbytes = (
                primary_nbytes
                if allocation_name == spec.allocation_names[0]
                else int(source.nbytes)
            )
        elif allocation_name == "x8":
            nbytes = int(source.nbytes)
        else:
            raise ValueError(
                f"unsupported resident allocation {allocation_name!r} for {spec.slot_path}"
            )
        records.append((allocation_name, int(nbytes)))
    return tuple(records)


def _planned_weight_allocation_nbytes(
    spec: Qwen35GGUFWeightSpec,
) -> tuple[tuple[str, int], ...]:
    """Compatibility alias used by the selective arena planner."""

    return planned_qwen35_gguf_weight_allocation_nbytes(spec)


def _planned_t16_nbytes(
    source: GGUFTensorInfo,
    *,
    block_bytes: int,
    tile_block_bytes: int,
    slot_path: str,
) -> int:
    byte_shape = tuple(int(dim) for dim in source.byte_shape)
    if len(byte_shape) == 2:
        experts = 1
        out_features, bytes_per_row = byte_shape
    elif len(byte_shape) == 3:
        experts, out_features, bytes_per_row = byte_shape
    else:
        raise ValueError(f"T16 plan requires rank-2 or rank-3 storage for {slot_path}")
    if out_features % GGUF_T16_COLS or bytes_per_row % int(block_bytes):
        raise ValueError(f"T16 shape is not tile-aligned for {slot_path}")
    return (
        experts
        * (out_features // GGUF_T16_COLS)
        * (bytes_per_row // int(block_bytes))
        * int(tile_block_bytes)
    )


def _validate_arena_alignment(alignment: int) -> int:
    parsed = int(alignment)
    if parsed <= 0 or parsed & (parsed - 1):
        raise ValueError("selective weight arena alignment must be a positive power of two")
    return parsed


def _align_arena(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def audit_qwen35_gguf_precision_contractions(
    plan: Qwen35GGUFMaterializationPlan,
) -> tuple[Qwen35GGUFPrecisionContraction, ...]:
    """Return source F32 GGUF tensors intentionally planned as BF16 residents.

    This is a parity diagnostic, not a failure by itself: current kernels may
    require BF16 resident inputs, while llama.cpp's GGML graph consumes these
    GGUF F32 tensors as F32 graph tensors.  The audit lets target-AR triage
    name those contractions explicitly before changing math or kernels.
    """

    findings: list[Qwen35GGUFPrecisionContraction] = []
    for spec in plan.specs:
        if spec.layout != LAYOUT_DENSE_BF16:
            continue
        if GGMLQuantizationType(spec.source.ggml_type) != GGMLQuantizationType.F32:
            continue
        findings.append(
            Qwen35GGUFPrecisionContraction(
                slot_path=spec.slot_path,
                source_name=spec.source.name,
                source_type=spec.source.ggml_type_name,
                resident_layout=spec.layout,
                resident_quant_key=spec.quant_key,
                llama_cpp_contract="GGUF F32 tensor participates in llama.cpp's F32 GGML graph",
                hipengine_contract=_precision_contraction_contract(spec.slot_path),
            )
        )
    return tuple(findings)


def materialize_qwen35_gguf_weights(
    reader_or_path: GGUFReader | str | Path,
    *,
    selected_slots: Iterable[str] | None = None,
    deferred_device_slots: Iterable[str] | None = None,
    decode_repack: bool | None = None,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    backend: str = "hip_gfx1100",
    use_selective_weight_arena: bool = False,
    selective_weight_max_allocation_bytes: int = GGUF_SELECTIVE_WEIGHT_ARENA_MAX_ALLOCATION_BYTES,
) -> Qwen35GGUFResidentWeights:
    """Materialize a validated Qwen3.5 GGUF map to resident device records.

    ``selected_slots`` is a test/debug hook using slot paths such as
    ``root.output_norm`` or ``layers.0.attn_qkv``. Production callers leave it
    unset to materialize the full model. ``deferred_device_slots`` retains the
    validated weight specs but performs no device allocation for those slots;
    callers must materialize them before passing the records to a kernel.
    """

    reader = reader_or_path if isinstance(reader_or_path, GGUFReader) else GGUFReader(reader_or_path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    file_type_name = getattr(reader.info, "file_type_name", None)
    raw_qmicro_file_types = backend_package_capability(
        backend,
        "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES",
        (),
    )
    qmicro_file_types = (
        frozenset(str(item) for item in raw_qmicro_file_types)
        if isinstance(raw_qmicro_file_types, (tuple, list, set, frozenset))
        else frozenset()
    )
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=decode_repack,
        dense_q4_t16=bool(
            backend_package_capability(
                backend,
                "GGUF_DENSE_Q4_T16",
                False,
            )
        ),
        dense_q4_qmicro_t16_gate_up=(
            bool(
                backend_package_capability(
                    backend,
                    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
                    False,
                )
            )
            and file_type_name in qmicro_file_types
        ),
        dense_q4_t16_attn_q_08b=bool(
            backend_package_capability(
                backend,
                "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
                False,
            )
        ),
        dense_q5_t16_ssm_out=bool(
            backend_package_capability(
                backend,
                "GGUF_DENSE_Q5_T16_SSM_OUT",
                False,
            )
        ),
        dense_q5_raw_mmq_ssm_out=(
            os.environ.get("HIPENGINE_GGUF_C8_Q5_RAW_MMQ", "1")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and bool(
                backend_package_capability(
                    backend,
                    "GGUF_C8_Q5_RAW_MMQ_SSM_OUT",
                    False,
                )
            )
        ),
        dense_q5_t16_ssm_out_08b=bool(
            backend_package_capability(
                backend,
                "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
                False,
            )
        ),
        dense_q5_t16_qkv=bool(
            backend_package_capability(
                backend,
                "GGUF_DENSE_Q5_T16_QKV",
                False,
            )
        ),
        dense_q5_t16_h5120=bool(
            backend_package_capability(
                backend,
                "GGUF_DENSE_Q5_T16_H5120",
                False,
            )
        ),
        dense_q6_qmicro_planar=bool(
            backend_package_capability(
                backend,
                "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
                False,
            )
        ),
        dense_q6_qmicro_planar_excluded_slots=tuple(
            backend_package_capability(
                backend,
                "GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS",
                (),
            )
        ),
    )
    selected = None if selected_slots is None else set(selected_slots)
    deferred = set() if deferred_device_slots is None else {str(slot) for slot in deferred_device_slots}
    known_slots = {spec.slot_path for spec in plan.specs}
    unknown_deferred = tuple(sorted(deferred - known_slots))
    if unknown_deferred:
        raise ValueError(
            "unknown deferred GGUF device slot(s): " + ", ".join(unknown_deferred)
        )
    materialized: dict[tuple[str, str], Qwen35GGUFDeviceWeight] = {}
    allocation_arena: DeviceMemoryArena | None = None
    allocator: _SelectiveWeightAllocator | None = None
    allocation_mode = "dedicated"
    allocation_arena_reason: str | None = None
    arena_plan: Qwen35GGUFSelectiveWeightArenaPlan | None = None
    if use_selective_weight_arena:
        arena_plan = plan_qwen35_gguf_selective_weight_arena(
            plan,
            selected_slots=selected,
            deferred_device_slots=deferred,
            max_allocation_bytes=selective_weight_max_allocation_bytes,
        )
        if not arena_plan.supported:
            allocation_mode = "dedicated_selective_arena_unsupported"
            allocation_arena_reason = arena_plan.reason
        elif arena_plan.capacity_bytes > 0:
            try:
                allocation_arena = DeviceMemoryArena.create(
                    arena_plan.capacity_bytes,
                    runtime=runtime,
                    alignment=arena_plan.alignment,
                )
                allocator = _SelectiveWeightAllocator(
                    allocation_arena,
                    max_allocation_bytes=arena_plan.max_allocation_bytes,
                    runtime=runtime,
                )
                allocation_mode = "selective_small_weight_arena"
            except (HipError, MemoryError) as exc:
                allocation_mode = "dedicated_selective_arena_denied"
                allocation_arena_reason = str(exc)

    def load(spec: Qwen35GGUFWeightSpec) -> Qwen35GGUFDeviceWeight:
        if spec.slot_path in deferred:
            return Qwen35GGUFDeviceWeight(
                spec=spec,
                allocations=MappingProxyType({}),
                backend=backend,
            )
        return _materialize_or_alias(
            spec,
            reader,
            materialized,
            selected,
            device=device,
            runtime=runtime,
            backend=backend,
            allocator=allocator,
        )

    try:
        root_weights = {
            slot: load(spec)
            for slot, spec in plan.root_specs.items()
            if selected is None or spec.slot_path in selected
        }
        layers = tuple(
            Qwen35GGUFResidentLayerWeights(
                layer_id=layer.layer_id,
                layer_type=layer.layer_type,
                weights=MappingProxyType(
                    {
                        slot: load(plan.layer_specs[layer.layer_id][slot])
                        for slot in plan.layer_specs[layer.layer_id]
                        if selected is None or plan.layer_specs[layer.layer_id][slot].slot_path in selected
                    }
                ),
            )
            for layer in model_map.layers
        )
        if allocation_arena is not None and arena_plan is not None:
            dedicated_allocations = tuple(
                allocation
                for weight in materialized.values()
                for allocation in weight.allocations.values()
                if allocation.owns_buffer
            )
            actual_signature = (
                allocation_arena.allocation_count,
                allocation_arena.requested_bytes,
                allocation_arena.capacity_bytes,
                len(dedicated_allocations),
                sum(int(allocation.buffer.nbytes) for allocation in dedicated_allocations),
            )
            planned_signature = (
                arena_plan.allocation_count,
                arena_plan.requested_bytes,
                arena_plan.capacity_bytes,
                arena_plan.dedicated_allocation_count,
                arena_plan.dedicated_requested_bytes,
            )
            if actual_signature != planned_signature:
                raise RuntimeError(
                    "GGUF selective weight arena plan changed during materialization: "
                    f"actual={actual_signature} planned={planned_signature}"
                )
    except Exception:
        for weight in reversed(tuple(materialized.values())):
            weight.free(runtime=runtime)
        if allocation_arena is not None:
            allocation_arena.close()
        raise
    metadata = getattr(reader.info, "metadata", {})
    model_name = None
    if isinstance(metadata, Mapping) and metadata.get("general.name") is not None:
        model_name = str(metadata["general.name"])
    return Qwen35GGUFResidentWeights(
        config=plan.config,
        root_weights=MappingProxyType(root_weights),
        layers=layers,
        backend=backend,
        geometry=GGUFModelGeometry.try_from_config(plan.config),
        model_name=model_name,
        file_type_name=(None if file_type_name is None else str(file_type_name)),
        allocation_arena=allocation_arena,
        allocation_mode=allocation_mode,
        allocation_arena_reason=allocation_arena_reason,
    )


def _plan_layer(
    layer: Qwen35GGUFLayerMap,
    *,
    decode_repack: bool,
    contract_f32_linear: bool = False,
    dense_q4_t16: bool = False,
    dense_q4_qmicro_t16_gate_up: bool = False,
    dense_q4_t16_attn_q_08b: bool = False,
    dense_q5_t16_ssm_out: bool = False,
    dense_q5_raw_mmq_ssm_out: bool = False,
    dense_q5_t16_ssm_out_08b: bool = False,
    dense_q5_t16_qkv: bool = False,
    dense_q5_t16_h5120: bool = False,
    dense_q6_qmicro_planar: bool = False,
    dense_q6_qmicro_planar_excluded_slots: frozenset[str] = frozenset(),
) -> dict[str, Qwen35GGUFWeightSpec]:
    return {
        slot: _spec_for_tensor(
            f"layers.{layer.layer_id}.{slot}",
            tensor,
            decode_repack=decode_repack,
            contract_f32_linear=contract_f32_linear,
            dense_q4_t16=dense_q4_t16,
            dense_q4_qmicro_t16_gate_up=dense_q4_qmicro_t16_gate_up,
            dense_q4_t16_attn_q_08b=dense_q4_t16_attn_q_08b,
            dense_q5_t16_ssm_out=dense_q5_t16_ssm_out,
            dense_q5_raw_mmq_ssm_out=dense_q5_raw_mmq_ssm_out,
            dense_q5_t16_ssm_out_08b=dense_q5_t16_ssm_out_08b,
            dense_q5_t16_qkv=dense_q5_t16_qkv,
            dense_q5_t16_h5120=dense_q5_t16_h5120,
            dense_q6_qmicro_planar=dense_q6_qmicro_planar,
            dense_q6_qmicro_planar_excluded_slots=(
                dense_q6_qmicro_planar_excluded_slots
            ),
        )
        for slot, tensor in layer.tensors.items()
    }


def gguf_decode_repack_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    raw = os.environ.get(HIPENGINE_GGUF_DECODE_REPACK_ENV, "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def gguf_selected_x8_repack_mode(value: bool | str | None = None) -> str:
    """Return selected-down X8 repack mode: ``off``, ``q5``, ``q6``, or ``both``."""

    raw = os.environ.get(HIPENGINE_GGUF_SELECTED_X8_REPACK_ENV, "")
    if value is not None:
        raw = str(value)
    mode = raw.strip().lower()
    if mode in {"", "0", "false", "no", "off"}:
        return "off"
    if mode in {"1", "true", "yes", "on", "both", "all"}:
        return "both"
    if mode in {"q5", "q5_k", "gguf_q5_k"}:
        return "q5"
    if mode in {"q6", "q6_k", "gguf_q6_k"}:
        return "q6"
    raise ValueError(
        f"{HIPENGINE_GGUF_SELECTED_X8_REPACK_ENV} must be off, q5, q6, both, or a boolean value"
    )


def gguf_selected_x8_repack_enabled(value: bool | str | None = None) -> bool:
    return gguf_selected_x8_repack_mode(value) != "off"


def _gguf_selected_x8_repack_enabled_for(quant: str) -> bool:
    mode = gguf_selected_x8_repack_mode()
    return mode == "both" or mode == quant


def gguf_selected_down_raw_mode(value: bool | str | None = None) -> str:
    """Return selected-down raw-GGUF mode: ``off``, ``q5``, ``q6``, or ``both``."""

    if value is None:
        raw = os.environ.get(HIPENGINE_GGUF_SELECTED_DOWN_RAW_ENV, "")
    elif isinstance(value, bool):
        raw = "both" if value else ""
    else:
        raw = str(value)
    normalized = raw.strip().lower()
    if normalized in {"", "0", "false", "off", "no"}:
        return "off"
    if normalized in {"1", "true", "yes", "on", "both", "all"}:
        return "both"
    if normalized in {"q5", "q5_k", "gguf_q5_k"}:
        return "q5"
    if normalized in {"q6", "q6_k", "gguf_q6_k"}:
        return "q6"
    raise ValueError(
        f"{HIPENGINE_GGUF_SELECTED_DOWN_RAW_ENV} must be one of off, q5, q6, or both; got {raw!r}"
    )


def gguf_selected_down_raw_enabled(value: bool | str | None = None) -> bool:
    return gguf_selected_down_raw_mode(value) != "off"


def _gguf_selected_down_raw_enabled_for(quant: str) -> bool:
    mode = gguf_selected_down_raw_mode()
    return mode == "both" or mode == quant


def gguf_selected_gate_up_raw_enabled(value: bool | str | None = None) -> bool:
    if value is None:
        raw = os.environ.get(HIPENGINE_GGUF_SELECTED_GATE_UP_RAW_ENV, "")
    elif isinstance(value, bool):
        raw = "1" if value else ""
    else:
        raw = str(value)
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def gguf_selected_gate_up_x8_enabled(value: bool | str | None = None) -> bool:
    if value is None:
        raw = os.environ.get(HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV, "")
    elif isinstance(value, bool):
        raw = "1" if value else ""
    else:
        raw = str(value)
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def gguf_q8_0_raw_sidecar_enabled(value: bool | str | None = None) -> bool:
    """Return whether T16 dense Q8_0 residents should retain raw GGUF bytes too.

    This is a default-off llama-compat diagnostic sidecar. The pair-only route
    retains raw bytes only for linear-attention ``attn_qkv`` and ``attn_gate``;
    ``HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL=1`` broadens the sidecar to every dense
    Q8_0 T16 tensor so verifier Q/K/V and singleton projections can be tested.
    """

    if value is None:
        raw = os.environ.get(HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV, "")
    elif isinstance(value, bool):
        raw = "1" if value else ""
    else:
        raw = str(value)
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def gguf_q8_0_raw_sidecar_all_enabled() -> bool:
    raw = os.environ.get(HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL_ENV, "")
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def gguf_lm_head_q6_x8_sidecar_enabled(value: bool | str | None = None) -> bool:
    """Return whether the Q6_K lm-head T16 resident keeps an X8 top-1 sidecar."""

    if value is None:
        raw = os.environ.get(HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR_ENV, "")
    elif isinstance(value, bool):
        raw = "1" if value else ""
    else:
        raw = str(value)
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def plan_qwen35_gguf_weight_spec(
    slot_path: str,
    tensor: GGUFTensorInfo,
    *,
    decode_repack: bool = False,
    dense_q4_t16: bool = False,
    dense_q4_qmicro_t16_gate_up: bool = False,
    dense_q4_t16_attn_q_08b: bool = False,
    dense_q5_t16_ssm_out: bool = False,
    dense_q5_raw_mmq_ssm_out: bool = False,
    dense_q5_t16_ssm_out_08b: bool = False,
    dense_q5_t16_qkv: bool = False,
    dense_q5_t16_h5120: bool = False,
    dense_q6_qmicro_planar: bool = False,
    dense_q6_qmicro_planar_excluded_slots: Iterable[str] = (),
) -> Qwen35GGUFWeightSpec:
    """Plan one canonical GGUF weight for AR or draft-model materialization."""

    return _spec_for_tensor(
        slot_path,
        tensor,
        decode_repack=bool(decode_repack),
        dense_q4_t16=bool(dense_q4_t16),
        dense_q4_qmicro_t16_gate_up=bool(dense_q4_qmicro_t16_gate_up),
        dense_q4_t16_attn_q_08b=bool(dense_q4_t16_attn_q_08b),
        dense_q5_t16_ssm_out=bool(dense_q5_t16_ssm_out),
        dense_q5_raw_mmq_ssm_out=bool(dense_q5_raw_mmq_ssm_out),
        dense_q5_t16_ssm_out_08b=bool(dense_q5_t16_ssm_out_08b),
        dense_q5_t16_qkv=bool(dense_q5_t16_qkv),
        dense_q5_t16_h5120=bool(dense_q5_t16_h5120),
        dense_q6_qmicro_planar=bool(dense_q6_qmicro_planar),
        dense_q6_qmicro_planar_excluded_slots=frozenset(
            str(slot) for slot in dense_q6_qmicro_planar_excluded_slots
        ),
    )


def _spec_for_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
    *,
    decode_repack: bool,
    contract_f32_linear: bool = False,
    dense_q4_t16: bool = False,
    dense_q4_qmicro_t16_gate_up: bool = False,
    dense_q4_t16_attn_q_08b: bool = False,
    dense_q5_t16_ssm_out: bool = False,
    dense_q5_raw_mmq_ssm_out: bool = False,
    dense_q5_t16_ssm_out_08b: bool = False,
    dense_q5_t16_qkv: bool = False,
    dense_q5_t16_h5120: bool = False,
    dense_q6_qmicro_planar: bool = False,
    dense_q6_qmicro_planar_excluded_slots: frozenset[str] = frozenset(),
) -> Qwen35GGUFWeightSpec:
    qtype = GGMLQuantizationType(tensor.ggml_type)
    if qtype == GGMLQuantizationType.F32:
        bf16_linear_weight = contract_f32_linear and slot_path.endswith(
            (".ffn_gate_inp", ".ffn_gate_inp_shexp", ".ssm_alpha", ".ssm_beta")
        )
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="bf16" if bf16_linear_weight else "f32",
            layout=LAYOUT_DENSE_BF16 if bf16_linear_weight else LAYOUT_DENSE_F32,
            allocation_names=("raw",),
        )
    if _is_token_embedding_slot(slot_path) and qtype in (
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        GGMLQuantizationType.Q8_0,
    ):
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key=f"gguf_{tensor.ggml_type_name.lower()}",
            layout=LAYOUT_RAW_GGUF,
            allocation_names=("raw",),
        )
    if qtype == GGMLQuantizationType.Q4_K:
        if len(tensor.shape) != 2:
            if decode_repack and _is_selected_expert_tensor(slot_path, tensor):
                if gguf_selected_gate_up_raw_enabled() and not _is_selected_down_expert_tensor(slot_path, tensor):
                    return Qwen35GGUFWeightSpec(
                        slot_path=slot_path,
                        source=tensor,
                        quant_key="gguf_q4_k",
                        layout=LAYOUT_RAW_GGUF,
                        allocation_names=("raw",),
                    )
                if gguf_selected_gate_up_x8_enabled() and not _is_selected_down_expert_tensor(slot_path, tensor):
                    return Qwen35GGUFWeightSpec(
                        slot_path=slot_path,
                        source=tensor,
                        quant_key="gguf_q4_k_x8_v1",
                        layout=LAYOUT_GGUF_Q4_K_X8,
                        allocation_names=("tiles",),
                    )
                return Qwen35GGUFWeightSpec(
                    slot_path=slot_path,
                    source=tensor,
                    quant_key="gguf_q4_k_t16_v1",
                    layout=LAYOUT_GGUF_Q4_K_T16,
                    allocation_names=("tiles",),
                )
            return Qwen35GGUFWeightSpec(
                slot_path=slot_path,
                source=tensor,
                quant_key="gguf_q4_k",
                layout=LAYOUT_RAW_GGUF,
                allocation_names=("raw",),
                sidecar_layouts=_sidecar_layouts_for_tensor(slot_path, tensor),
            )
        if (
            decode_repack
            and dense_q4_qmicro_t16_gate_up
            and _is_dense_q4_qmicro_t16_gate_up_tensor(slot_path, tensor)
        ):
            return Qwen35GGUFWeightSpec(
                slot_path=slot_path,
                source=tensor,
                quant_key="gguf_q4_k_qmicro_t16_v1",
                layout=LAYOUT_GGUF_Q4_K_QMICRO_T16,
                allocation_names=("tiles",),
            )
        if decode_repack and (
            (
                dense_q4_t16
                and _dense_q4_t16_sidecar_allocation_name(slot_path, tensor) is not None
            )
            or (
                dense_q4_t16_attn_q_08b
                and _is_dense_q4_t16_attn_q_08b_tensor(slot_path, tensor)
            )
        ):
            return Qwen35GGUFWeightSpec(
                slot_path=slot_path,
                source=tensor,
                quant_key="gguf_q4_k_t16_v1",
                layout=LAYOUT_GGUF_Q4_K_T16,
                allocation_names=("tiles",),
            )
        allocation_names = ("qweight", "scales", "mins")
        q4_t16_sidecar = (
            _dense_q4_t16_sidecar_allocation_name(slot_path, tensor)
            if decode_repack
            else None
        )
        if q4_t16_sidecar is not None:
            allocation_names += (q4_t16_sidecar,)
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="gguf_q4_k",
            layout=LAYOUT_Q4_K_PACK8,
            allocation_names=allocation_names,
        )
    if qtype == GGMLQuantizationType.Q5_K:
        if decode_repack and (
            (
                dense_q5_t16_ssm_out
                and _is_dense_q5_t16_ssm_out_tensor(slot_path, tensor)
            )
            or (
                dense_q5_t16_ssm_out_08b
                and _is_dense_q5_t16_ssm_out_08b_tensor(slot_path, tensor)
            )
            or (
                dense_q5_t16_qkv
                and _is_dense_q5_t16_qkv_tensor(slot_path, tensor)
            )
            or (
                dense_q5_t16_h5120
                and _is_dense_h5120_q5_t16_tensor(slot_path, tensor)
            )
        ):
            return Qwen35GGUFWeightSpec(
                slot_path=slot_path,
                source=tensor,
                quant_key="gguf_q5_k_t16_v1",
                layout=LAYOUT_GGUF_Q5_K_T16,
                allocation_names=(
                    ("tiles", "raw")
                    if dense_q5_raw_mmq_ssm_out
                    and _is_dense_q5_t16_ssm_out_tensor(slot_path, tensor)
                    else ("tiles",)
                ),
            )
        if decode_repack and _is_selected_expert_tensor(slot_path, tensor):
            if _gguf_selected_down_raw_enabled_for("q5") and _is_selected_down_expert_tensor(slot_path, tensor):
                return Qwen35GGUFWeightSpec(
                    slot_path=slot_path,
                    source=tensor,
                    quant_key="gguf_q5_k",
                    layout=LAYOUT_RAW_GGUF,
                    allocation_names=("raw",),
                )
            if _gguf_selected_x8_repack_enabled_for("q5") and _is_selected_down_expert_tensor(slot_path, tensor):
                return Qwen35GGUFWeightSpec(
                    slot_path=slot_path,
                    source=tensor,
                    quant_key="gguf_q5_k_x8_v1",
                    layout=LAYOUT_GGUF_Q5_K_X8,
                    allocation_names=("tiles",),
                )
            return Qwen35GGUFWeightSpec(
                slot_path=slot_path,
                source=tensor,
                quant_key="gguf_q5_k_qmicro_t16_v1",
                layout=LAYOUT_GGUF_Q5_K_QMICRO_T16,
                allocation_names=("tiles",),
            )
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="gguf_q5_k",
            layout=LAYOUT_RAW_GGUF if len(tensor.shape) != 2 else LAYOUT_DENSE_BF16,
            allocation_names=("raw",),
            sidecar_layouts=_sidecar_layouts_for_tensor(slot_path, tensor),
        )
    if qtype == GGMLQuantizationType.Q6_K and decode_repack and slot_path == "root.lm_head" and len(tensor.shape) == 2:
        keep_x8_sidecar = gguf_lm_head_q6_x8_sidecar_enabled()
        use_qmicro_planar = (
            dense_q6_qmicro_planar
            and "lm_head" not in dense_q6_qmicro_planar_excluded_slots
            and not keep_x8_sidecar
            and tuple(map(int, tensor.shape)) == (248_320, 5_120)
        )
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key=(
                "gguf_q6_k_t16_qmicro_planar_v1"
                if use_qmicro_planar
                else "gguf_q6_k_t16_v1"
            ),
            layout=(
                LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
                if use_qmicro_planar
                else LAYOUT_GGUF_Q6_K_T16
            ),
            allocation_names=("tiles", "x8") if keep_x8_sidecar else ("tiles",),
        )
    if qtype == GGMLQuantizationType.Q6_K and slot_path.startswith("layers."):
        slot_name = slot_path.rsplit(".", 1)[-1]
        use_qmicro_planar = (
            dense_q6_qmicro_planar
            and slot_name not in dense_q6_qmicro_planar_excluded_slots
        )
        if (
            decode_repack
            and use_qmicro_planar
            and _is_narrow_q6_attn_v_tensor(slot_path, tensor)
        ):
            return Qwen35GGUFWeightSpec(
                slot_path=slot_path,
                source=tensor,
                quant_key="gguf_q6_k_t16_qmicro_planar_v1",
                layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
                allocation_names=("tiles",),
            )
        if decode_repack and _is_wide_rank2_q6_t16_tensor(slot_path, tensor):
            return Qwen35GGUFWeightSpec(
                slot_path=slot_path,
                source=tensor,
                quant_key=(
                    "gguf_q6_k_t16_qmicro_planar_v1"
                    if use_qmicro_planar
                    else "gguf_q6_k_t16_v1"
                ),
                layout=(
                    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
                    if use_qmicro_planar
                    else LAYOUT_GGUF_Q6_K_T16
                ),
                allocation_names=("tiles",),
            )
        if decode_repack and _is_selected_expert_tensor(slot_path, tensor):
            if _gguf_selected_down_raw_enabled_for("q6") and _is_selected_down_expert_tensor(slot_path, tensor):
                return Qwen35GGUFWeightSpec(
                    slot_path=slot_path,
                    source=tensor,
                    quant_key="gguf_q6_k",
                    layout=LAYOUT_RAW_GGUF,
                    allocation_names=("raw",),
                )
            if _gguf_selected_x8_repack_enabled_for("q6") and _is_selected_down_expert_tensor(slot_path, tensor):
                return Qwen35GGUFWeightSpec(
                    slot_path=slot_path,
                    source=tensor,
                    quant_key="gguf_q6_k_x8_v1",
                    layout=LAYOUT_GGUF_Q6_K_X8,
                    allocation_names=("tiles",),
                )
            return Qwen35GGUFWeightSpec(
                slot_path=slot_path,
                source=tensor,
                quant_key="gguf_q6_k_t16_v1",
                layout=LAYOUT_GGUF_Q6_K_T16,
                allocation_names=("tiles",),
            )
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="gguf_q6_k",
            layout=LAYOUT_RAW_GGUF if len(tensor.shape) != 2 else LAYOUT_DENSE_BF16,
            allocation_names=("raw",),
            sidecar_layouts=_sidecar_layouts_for_tensor(slot_path, tensor),
        )
    if qtype == GGMLQuantizationType.Q8_0 and decode_repack and slot_path.startswith("layers.") and len(tensor.shape) == 2:
        allocation_names = ("tiles",)
        if gguf_q8_0_raw_sidecar_enabled() and (
            gguf_q8_0_raw_sidecar_all_enabled() or _is_linear_attention_q8_pair_tensor(slot_path, tensor)
        ):
            allocation_names = ("tiles", "raw")
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="gguf_q8_0_t16_v1",
            layout=LAYOUT_GGUF_Q8_0_T16,
            allocation_names=allocation_names,
        )
    if qtype in (GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_0):
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key=f"gguf_{tensor.ggml_type_name.lower()}",
            layout=LAYOUT_RAW_GGUF,
            allocation_names=("raw",),
        )
    if qtype in (
        GGMLQuantizationType.IQ3_XXS,
        GGMLQuantizationType.Q3_K,
    ) or (
        qtype in (GGMLQuantizationType.IQ2_XS, GGMLQuantizationType.IQ4_XS)
        and _is_selected_expert_tensor(slot_path, tensor)
    ):
        # Native selected GEMV keeps routed rank-3 IQ2/IQ3/IQ4 experts raw.
        # Rank-2 IQ2_XS/IQ4_XS tensors keep the dense-BF16 fallback below;
        # rank-2 IQ3_XXS/Q3_K remain unsupported rather than silently expanding.
        if not _is_selected_expert_tensor(slot_path, tensor):
            raise ValueError(
                f"unsupported Qwen3.5 GGUF tensor type {tensor.ggml_type_name!r} outside "
                f"rank-3 expert slots: {tensor.name}"
            )
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key=f"gguf_{tensor.ggml_type_name.lower()}",
            layout=LAYOUT_RAW_GGUF,
            allocation_names=("raw",),
        )
    if qtype in (
        GGMLQuantizationType.Q4_1,
        GGMLQuantizationType.IQ2_XS,
        GGMLQuantizationType.IQ4_XS,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
    ):
        quant_key = (
            "fp16" if qtype == GGMLQuantizationType.F16 else f"gguf_{tensor.ggml_type_name.lower()}"
        )
        if qtype == GGMLQuantizationType.BF16:
            quant_key = "bf16"
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key=quant_key,
            layout=LAYOUT_DENSE_BF16,
            allocation_names=("raw",),
        )
    raise ValueError(f"unsupported Qwen3.5 GGUF tensor type {tensor.ggml_type_name!r}: {tensor.name}")


def _precision_contraction_contract(slot_path: str) -> str:
    known = {}
    for suffix, contract in known.items():
        if slot_path.endswith(suffix):
            return contract
    return "source F32 tensor is stored as BF16 by the current resident materialization plan"


def _gguf_ssm_a_to_kernel_a_log(raw: object):
    """Convert GGUF Qwen3.5 ``ssm_a`` coefficients to the GDN kernel ABI.

    llama.cpp treats GGUF ``blk.*.ssm_a`` as the direct negative decay
    coefficient used in ``exp(ssm_a * softplus(alpha + dt_bias))``.  The shared
    hipEngine GDN kernels are also used by PARO, where the ABI is ``A_log`` and
    the kernel computes ``exp(-exp(A_log) * softplus(...))``.  Materialize GGUF
    ``ssm_a`` as ``log(-ssm_a)`` so the existing kernel math is exactly the same
    as llama.cpp without changing the PARO ABI.
    """

    import numpy as np

    coeff = np.asarray(raw, dtype=np.float32)
    if not np.all(np.isfinite(coeff)):
        raise ValueError("GGUF qwen35 ssm_a contains non-finite values")
    if np.any(coeff >= 0.0):
        raise ValueError("GGUF qwen35 ssm_a must contain negative decay coefficients")
    return np.ascontiguousarray(np.log(-coeff), dtype=np.float32)


def _is_token_embedding_slot(slot_path: str) -> bool:
    return slot_path == "root.token_embedding" or slot_path.endswith(".embed_tokens")


_DENSE_Q4_T16_SIDECAR_POLICY = (
    ("attn_gate", (6_144, 5_120), Q4_T16_DECODE_TILES),
    ("attn_k", (1_024, 5_120), Q4_T16_DECODE_TILES),
    ("attn_output", (5_120, 6_144), Q4_T16_DECODE_TILES),
    ("attn_q", (12_288, 5_120), Q4_T16_DECODE_TILES),
    ("attn_qkv", (10_240, 5_120), Q4_T16_DECODE_TILES_R3PLUS),
    ("attn_v", (1_024, 5_120), Q4_T16_DECODE_TILES_R3PLUS),
    ("ffn_down", (5_120, 17_408), Q4_T16_DECODE_TILES),
    ("ffn_gate", (17_408, 5_120), Q4_T16_DECODE_TILES),
    ("ffn_up", (17_408, 5_120), Q4_T16_DECODE_TILES),
)


def _dense_q4_t16_sidecar_allocation_name(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> str | None:
    """Return the measured dense-H5120 compact-T16 verifier policy."""

    if len(tensor.shape) != 2:
        return None
    shape = tuple(map(int, tensor.shape))
    for role, expected_shape, allocation_name in _DENSE_Q4_T16_SIDECAR_POLICY:
        if shape == expected_shape and slot_path.endswith(f".{role}"):
            return allocation_name
    return None


def _is_dense_q4_qmicro_t16_gate_up_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> bool:
    """Select the measured Qwen3.8-27B dense gate/up replacement payload."""

    return (
        len(tensor.shape) == 2
        and tuple(map(int, tensor.shape)) == (17_408, 5_120)
        and slot_path.startswith("layers.")
        and slot_path.endswith((".ffn_gate", ".ffn_up"))
    )


def _is_dense_q4_t16_attn_q_08b_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> bool:
    """Select only the measured Qwen3.5-0.8B full-attention Q projection."""

    return (
        len(tensor.shape) == 2
        and tuple(map(int, tensor.shape)) == (4_096, 1_024)
        and slot_path.startswith("layers.")
        and slot_path.endswith(".attn_q")
    )


def _is_dense_q5_t16_qkv_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> bool:
    """Select only the measured Qwen3.5-0.8B linear-attention QKV projection."""

    return (
        len(tensor.shape) == 2
        and tuple(map(int, tensor.shape)) == (6_144, 1_024)
        and slot_path.startswith("layers.")
        and slot_path.endswith(".attn_qkv")
    )


def _is_dense_h5120_q5_t16_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> bool:
    """Select Q4_K_S Q5 roles with operation-complete dense-H5120 consumers."""

    shape = tuple(map(int, tensor.shape))
    return (
        len(shape) == 2
        and slot_path.startswith("layers.")
        and (
            (slot_path.endswith(".ffn_down") and shape == (5_120, 17_408))
            or (slot_path.endswith(".attn_qkv") and shape == (10_240, 5_120))
            or (slot_path.endswith(".attn_v") and shape == (1_024, 5_120))
        )
    )


def _is_dense_q5_t16_ssm_out_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> bool:
    """Select only the measured dense-H5120 recurrent output projection."""

    return (
        len(tensor.shape) == 2
        and tuple(map(int, tensor.shape)) == (5_120, 6_144)
        and slot_path.startswith("layers.")
        and slot_path.endswith(".ssm_out")
    )


def _is_dense_q5_t16_ssm_out_08b_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> bool:
    """Select only the measured Qwen3.5-0.8B recurrent output projection."""

    return (
        len(tensor.shape) == 2
        and tuple(map(int, tensor.shape)) == (1_024, 2_048)
        and slot_path.startswith("layers.")
        and slot_path.endswith(".ssm_out")
    )


def _is_wide_rank2_q6_t16_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> bool:
    """Select measured wide dense projections without regressing narrow V."""

    return (
        len(tensor.shape) == 2
        and int(tensor.shape[0]) >= 5_120
        and slot_path.endswith((".ffn_down", ".attn_qkv"))
    )


def _is_narrow_q6_attn_v_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
) -> bool:
    """Select the measured dense-H5120 full-attention V projection."""

    return (
        len(tensor.shape) == 2
        and tuple(map(int, tensor.shape)) == (1_024, 5_120)
        and slot_path.startswith("layers.")
        and slot_path.endswith(".attn_v")
    )


def _is_selected_expert_tensor(slot_path: str, tensor: GGUFTensorInfo) -> bool:
    return len(tensor.shape) == 3 and slot_path.endswith((".ffn_gate_exps", ".ffn_up_exps", ".ffn_down_exps"))


def _is_selected_down_expert_tensor(slot_path: str, tensor: GGUFTensorInfo) -> bool:
    return len(tensor.shape) == 3 and slot_path.endswith(".ffn_down_exps")


def _is_linear_attention_q8_pair_tensor(slot_path: str, tensor: GGUFTensorInfo) -> bool:
    return (
        len(tensor.shape) == 2
        and GGMLQuantizationType(tensor.ggml_type) == GGMLQuantizationType.Q8_0
        and slot_path.endswith((".attn_qkv", ".attn_gate"))
    )


def _sidecar_layouts_for_tensor(slot_path: str, tensor: GGUFTensorInfo) -> tuple[str, ...]:
    if (
        _is_selected_expert_tensor(slot_path, tensor)
        and GGMLQuantizationType(tensor.ggml_type)
        in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K)
    ):
        return (LAYOUT_GGUF_EXPERT_PACK8_SIDECAR,)
    return ()


def _materialize_or_alias(
    spec: Qwen35GGUFWeightSpec,
    reader: GGUFReader,
    materialized: dict[tuple[str, str], Qwen35GGUFDeviceWeight],
    selected: set[str] | None,
    *,
    device: Device | None,
    runtime: HipRuntime | None,
    backend: str,
    allocator: DeviceBufferAllocator | None = None,
) -> Qwen35GGUFDeviceWeight:
    del selected  # selection is handled by callers before materialization.
    key = (spec.source.name, spec.layout)
    weight = materialized.get(key)
    if weight is None:
        weight = _materialize_spec(
            spec,
            reader,
            device=device,
            runtime=runtime,
            backend=backend,
            allocator=allocator,
        )
        materialized[key] = weight
    return weight


def materialize_qwen35_gguf_weight_spec(
    spec: Qwen35GGUFWeightSpec,
    reader: GGUFReader,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    backend: str = "hip_gfx1100",
) -> Qwen35GGUFDeviceWeight:
    """Materialize one planned GGUF weight for AR or draft-model ownership."""

    return _materialize_spec(
        spec,
        reader,
        device=device,
        runtime=runtime,
        backend=str(backend),
    )


def _materialize_spec(
    spec: Qwen35GGUFWeightSpec,
    reader: GGUFReader,
    *,
    device: Device | None,
    runtime: HipRuntime | None,
    backend: str,
    allocator: DeviceBufferAllocator | None = None,
) -> Qwen35GGUFDeviceWeight:
    import numpy as np

    def load_host_array_to_device_as_dtype(*args, **kwargs):
        return _load_host_array_to_device_as_dtype(
            *args,
            **kwargs,
            allocator=allocator,
        )

    raw = np.ascontiguousarray(reader.tensor_data(spec.source.name))
    if spec.slot_path.endswith(".ssm_a"):
        raw = _gguf_ssm_a_to_kernel_a_log(raw)
    allocations: dict[str, DeviceTensorAllocation]
    if spec.layout == LAYOUT_Q4_K_PACK8:
        packed = repack_gguf_q4_k_pack8(raw)
        allocations = {
            "qweight": load_host_array_to_device_as_dtype(
                f"{spec.source.name}.pack8.qweight",
                packed.qweight,
                DType.INT32,
                source_dtype="I32",
                device=device,
                runtime=runtime,
            ),
            "scales": load_host_array_to_device_as_dtype(
                f"{spec.source.name}.pack8.scales",
                packed.scales,
                DType.FP32,
                source_dtype="F32",
                device=device,
                runtime=runtime,
            ),
            "mins": load_host_array_to_device_as_dtype(
                f"{spec.source.name}.pack8.mins",
                packed.mins,
                DType.FP32,
                source_dtype="F32",
                device=device,
                runtime=runtime,
            ),
        }
        q4_t16_sidecar_names = tuple(
            name
            for name in (
                Q4_T16_DECODE_TILES,
                Q4_T16_DECODE_TILES_R3PLUS,
            )
            if name in spec.allocation_names
        )
        if q4_t16_sidecar_names:
            decode_tiles = repack_gguf_q4_k_tile16(
                raw if raw.ndim == 3 else raw[None, ...]
            ).tiles
            for sidecar_name in q4_t16_sidecar_names:
                allocations[sidecar_name] = load_host_array_to_device_as_dtype(
                    f"{spec.source.name}.t16_decode_sidecar",
                    decode_tiles,
                    DType.INT8,
                    source_dtype="I8",
                    device=device,
                    runtime=runtime,
                )
    elif spec.layout in {
        LAYOUT_GGUF_Q4_K_T16,
        LAYOUT_GGUF_Q4_K_QMICRO_T16,
        LAYOUT_GGUF_Q4_K_X8,
        LAYOUT_GGUF_Q5_K_T16,
        LAYOUT_GGUF_Q5_K_QMICRO_T16,
        LAYOUT_GGUF_Q6_K_T16,
        LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        LAYOUT_GGUF_Q8_0_T16,
        LAYOUT_GGUF_Q5_K_X8,
        LAYOUT_GGUF_Q6_K_X8,
    }:
        if spec.layout == LAYOUT_GGUF_Q4_K_T16:
            packed = repack_gguf_q4_k_tile16(
                raw if raw.ndim == 3 else raw[None, ...]
            )
        elif spec.layout == LAYOUT_GGUF_Q4_K_QMICRO_T16:
            packed = repack_gguf_q4_k_tile16_qmicro(
                raw if raw.ndim == 3 else raw[None, ...]
            )
        elif spec.layout == LAYOUT_GGUF_Q4_K_X8:
            packed = repack_gguf_q4_k_x8(raw)
        elif spec.layout == LAYOUT_GGUF_Q5_K_T16:
            packed = repack_gguf_q5_k_tile16(
                raw if raw.ndim == 3 else raw[None, ...]
            )
        elif spec.layout == LAYOUT_GGUF_Q5_K_QMICRO_T16:
            packed = repack_gguf_q5_k_qmicro_tile16(raw)
        elif spec.layout == LAYOUT_GGUF_Q6_K_T16:
            packed = repack_gguf_q6_k_tile16(raw if raw.ndim == 3 else raw[None, ...])
        elif spec.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR:
            packed = repack_gguf_q6_k_tile16_qmicro_planar(
                raw if raw.ndim == 3 else raw[None, ...]
            )
        elif spec.layout == LAYOUT_GGUF_Q5_K_X8:
            packed = repack_gguf_q5_k_x8(raw)
        elif spec.layout == LAYOUT_GGUF_Q6_K_X8:
            packed = repack_gguf_q6_k_x8(raw)
        else:
            packed = repack_gguf_q8_0_tile16(raw)
        allocations = {
            "tiles": load_host_array_to_device_as_dtype(
                f"{spec.source.name}.t16.tiles",
                packed.tiles,
                DType.INT8,
                source_dtype="I8",
                device=device,
                runtime=runtime,
            )
        }
        if "x8" in spec.allocation_names:
            if spec.layout != LAYOUT_GGUF_Q6_K_T16:
                raise ValueError("X8 sidecar is only supported for Q6_K T16 residents")
            x8_packed = repack_gguf_q6_k_x8(raw if raw.ndim == 3 else raw[None, ...])
            x8_tiles = x8_packed.tiles[0] if raw.ndim == 2 else x8_packed.tiles
            allocations["x8"] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.x8_sidecar",
                x8_tiles,
                DType.INT8,
                source_dtype="I8",
                device=device,
                runtime=runtime,
            )
        if "raw" in spec.allocation_names:
            allocations["raw"] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.raw_sidecar",
                raw,
                DType.INT8,
                source_dtype="I8",
                device=device,
                runtime=runtime,
            )
    elif spec.layout == LAYOUT_RAW_GGUF:
        allocations = {
            "raw": load_host_array_to_device_as_dtype(
                spec.source.name,
                raw,
                DType.INT8,
                source_dtype="I8",
                device=device,
                runtime=runtime,
            )
        }
    elif spec.layout == LAYOUT_DENSE_F32:
        allocations = {
            "raw": load_host_array_to_device_as_dtype(
                spec.source.name,
                raw,
                DType.FP32,
                source_dtype="F32",
                device=device,
                runtime=runtime,
            )
        }
    elif spec.layout == LAYOUT_DENSE_BF16:
        if GGMLQuantizationType(spec.source.ggml_type) == GGMLQuantizationType.BF16:
            bf16 = raw
        else:
            bf16 = float_array_to_bf16_bits(dequantize_gguf_data(raw, spec.source.ggml_type))
        allocations = {
            "raw": load_host_array_to_device_as_dtype(
                f"{spec.source.name}.dense_bf16_fallback",
                bf16,
                DType.BF16,
                source_dtype="BF16",
                device=device,
                runtime=runtime,
            )
        }
    else:
        raise ValueError(f"unsupported materialization layout {spec.layout!r}")
    return Qwen35GGUFDeviceWeight(
        spec=spec,
        allocations=MappingProxyType(allocations),
        backend=backend,
    )


__all__ = [
    "LAYOUT_DENSE_BF16",
    "LAYOUT_DENSE_F32",
    "HIPENGINE_GGUF_DECODE_REPACK_ENV",
    "HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL_ENV",
    "HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR_ENV",
    "HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV",
    "HIPENGINE_GGUF_SELECTED_DOWN_RAW_ENV",
    "HIPENGINE_GGUF_SELECTED_GATE_UP_RAW_ENV",
    "HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV",
    "HIPENGINE_GGUF_SELECTED_X8_REPACK_ENV",
    "GGUF_SELECTIVE_WEIGHT_ARENA_ALIGNMENT",
    "GGUF_SELECTIVE_WEIGHT_ARENA_MAX_ALLOCATION_BYTES",
    "LAYOUT_GGUF_EXPERT_PACK8_SIDECAR",
    "LAYOUT_GGUF_Q4_K_QMICRO_T16",
    "LAYOUT_GGUF_Q4_K_T16",
    "LAYOUT_GGUF_Q4_K_X8",
    "LAYOUT_GGUF_Q5_K_QMICRO_T16",
    "LAYOUT_GGUF_Q5_K_T16",
    "LAYOUT_GGUF_Q5_K_X8",
    "LAYOUT_GGUF_Q6_K_T16",
    "LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR",
    "LAYOUT_GGUF_Q6_K_X8",
    "LAYOUT_GGUF_Q8_0_T16",
    "LAYOUT_Q4_K_PACK8",
    "LAYOUT_RAW_GGUF",
    "Qwen35GGUFDeviceWeight",
    "Qwen35GGUFMaterializationPlan",
    "Qwen35GGUFPrecisionContraction",
    "Qwen35GGUFSelectiveWeightArenaPlan",
    "Qwen35GGUFResidentLayerWeights",
    "Qwen35GGUFResidentWeights",
    "Qwen35GGUFWeightSpec",
    "_gguf_ssm_a_to_kernel_a_log",
    "audit_qwen35_gguf_precision_contractions",
    "gguf_decode_repack_enabled",
    "gguf_lm_head_q6_x8_sidecar_enabled",
    "gguf_q8_0_raw_sidecar_all_enabled",
    "gguf_q8_0_raw_sidecar_enabled",
    "gguf_selected_down_raw_enabled",
    "gguf_selected_down_raw_mode",
    "gguf_selected_gate_up_raw_enabled",
    "gguf_selected_gate_up_x8_enabled",
    "gguf_selected_x8_repack_enabled",
    "gguf_selected_x8_repack_mode",
    "materialize_qwen35_gguf_weights",
    "plan_qwen35_gguf_materialization",
    "plan_qwen35_gguf_selective_weight_arena",
    "plan_qwen35_gguf_weight_spec",
    "planned_qwen35_gguf_weight_allocation_nbytes",
]
