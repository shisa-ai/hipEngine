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
    GGUF_Q4_K_TILE16_COLS,
    repack_gguf_q4_k_pack8,
    repack_gguf_q4_k_tile16,
)
from hipengine.quant.gguf_t16 import (
    GGUF_Q5_K_BLOCK_BYTES,
    GGUF_Q5_K_T16_BLOCK_BYTES,
    GGUF_T16_COLS,
    repack_gguf_q5_k_qmicro_tile16,
    repack_gguf_q5_k_tile16,
    repack_gguf_q6_k_tile16,
    repack_gguf_q8_0_tile16,
)
from hipengine.quant.gguf_x8 import repack_gguf_q4_k_x8, repack_gguf_q5_k_x8, repack_gguf_q6_k_x8

LAYOUT_DENSE_F32 = "dense_f32"
LAYOUT_DENSE_BF16 = "dense_bf16"
LAYOUT_RAW_GGUF = "raw_gguf"
LAYOUT_Q4_K_PACK8 = "q4_k_pack8"
LAYOUT_GGUF_EXPERT_PACK8_SIDECAR = "gguf_expert_pack8_v1"
LAYOUT_GGUF_Q4_K_T16 = "gguf_q4_k_t16_v1"
LAYOUT_GGUF_Q4_K_X8 = "gguf_q4_k_x8_v1"
LAYOUT_GGUF_Q5_K_T16 = "gguf_q5_k_t16_v1"
LAYOUT_GGUF_Q5_K_QMICRO_T16 = "gguf_q5_k_qmicro_t16_v1"
LAYOUT_GGUF_Q6_K_T16 = "gguf_q6_k_t16_v1"
LAYOUT_GGUF_Q8_0_T16 = "gguf_q8_0_t16_v1"
LAYOUT_GGUF_Q5_K_X8 = "gguf_q5_k_x8_v1"
LAYOUT_GGUF_Q6_K_X8 = "gguf_q6_k_x8_v1"
HIPENGINE_GGUF_DECODE_REPACK_ENV = "HIPENGINE_GGUF_DECODE_REPACK"
HIPENGINE_GGUF_SELECTED_X8_REPACK_ENV = "HIPENGINE_GGUF_SELECTED_X8_REPACK"
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
) -> Qwen35GGUFMaterializationPlan:
    requested_decode_repack = gguf_decode_repack_enabled(decode_repack)
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
        )
        for slot, tensor in model_map.root_tensors.items()
    }
    layer_specs = tuple(
        _plan_layer(
            layer,
            decode_repack=use_decode_repack,
            contract_f32_linear=contract_q3_f32_linear,
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


def _planned_weight_allocation_nbytes(
    spec: Qwen35GGUFWeightSpec,
) -> tuple[tuple[str, int], ...]:
    source = spec.source
    if spec.layout == LAYOUT_Q4_K_PACK8:
        raise ValueError(f"unsupported selective arena layout {spec.layout!r} for {spec.slot_path}")
    if spec.layout == LAYOUT_DENSE_F32:
        primary_nbytes = int(source.n_elements) * DType.FP32.itemsize
    elif spec.layout == LAYOUT_DENSE_BF16:
        primary_nbytes = int(source.n_elements) * DType.BF16.itemsize
    elif spec.layout == LAYOUT_GGUF_Q4_K_T16:
        if len(source.byte_shape) != 3:
            raise ValueError(f"Q4 T16 arena plan requires rank-3 storage for {spec.slot_path}")
        experts, out_features, bytes_per_row = (int(dim) for dim in source.byte_shape)
        if out_features % GGUF_Q4_K_TILE16_COLS or bytes_per_row % GGUF_Q4_K_BLOCK_BYTES:
            raise ValueError(f"Q4 T16 arena shape is not tile-aligned for {spec.slot_path}")
        primary_nbytes = (
            experts
            * (out_features // GGUF_Q4_K_TILE16_COLS)
            * (bytes_per_row // GGUF_Q4_K_BLOCK_BYTES)
            * GGUF_Q4_K_TILE16_BLOCK_BYTES
        )
    elif spec.layout == LAYOUT_GGUF_Q5_K_T16:
        if len(source.byte_shape) != 3:
            raise ValueError(f"Q5 T16 arena plan requires rank-3 storage for {spec.slot_path}")
        experts, out_features, bytes_per_row = (int(dim) for dim in source.byte_shape)
        if out_features % GGUF_T16_COLS or bytes_per_row % GGUF_Q5_K_BLOCK_BYTES:
            raise ValueError(f"Q5 T16 arena shape is not tile-aligned for {spec.slot_path}")
        primary_nbytes = (
            experts
            * (out_features // GGUF_T16_COLS)
            * (bytes_per_row // GGUF_Q5_K_BLOCK_BYTES)
            * GGUF_Q5_K_T16_BLOCK_BYTES
        )
    elif spec.layout in {
        LAYOUT_RAW_GGUF,
        LAYOUT_GGUF_Q4_K_X8,
        LAYOUT_GGUF_Q5_K_QMICRO_T16,
        LAYOUT_GGUF_Q5_K_X8,
        LAYOUT_GGUF_Q6_K_T16,
        LAYOUT_GGUF_Q6_K_X8,
        LAYOUT_GGUF_Q8_0_T16,
    }:
        primary_nbytes = int(source.nbytes)
    else:
        raise ValueError(f"unsupported selective arena layout {spec.layout!r} for {spec.slot_path}")

    records: list[tuple[str, int]] = []
    for allocation_name in spec.allocation_names:
        if allocation_name == "tiles" or allocation_name == "raw":
            nbytes = (
                primary_nbytes
                if allocation_name == spec.allocation_names[0]
                else int(source.nbytes)
            )
        elif allocation_name == "x8":
            nbytes = int(source.nbytes)
        else:
            raise ValueError(
                f"unsupported selective arena allocation {allocation_name!r} "
                f"for {spec.slot_path}"
            )
        records.append((allocation_name, int(nbytes)))
    return tuple(records)


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
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=decode_repack)
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
    return Qwen35GGUFResidentWeights(
        config=plan.config,
        root_weights=MappingProxyType(root_weights),
        layers=layers,
        backend=backend,
        allocation_arena=allocation_arena,
        allocation_mode=allocation_mode,
        allocation_arena_reason=allocation_arena_reason,
    )


def _plan_layer(
    layer: Qwen35GGUFLayerMap,
    *,
    decode_repack: bool,
    contract_f32_linear: bool = False,
) -> dict[str, Qwen35GGUFWeightSpec]:
    return {
        slot: _spec_for_tensor(
            f"layers.{layer.layer_id}.{slot}",
            tensor,
            decode_repack=decode_repack,
            contract_f32_linear=contract_f32_linear,
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
) -> Qwen35GGUFWeightSpec:
    """Plan one canonical GGUF weight for AR or draft-model materialization."""

    return _spec_for_tensor(slot_path, tensor, decode_repack=bool(decode_repack))


def _spec_for_tensor(
    slot_path: str,
    tensor: GGUFTensorInfo,
    *,
    decode_repack: bool,
    contract_f32_linear: bool = False,
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
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="gguf_q4_k",
            layout=LAYOUT_Q4_K_PACK8,
            allocation_names=("qweight", "scales", "mins"),
        )
    if qtype == GGMLQuantizationType.Q5_K:
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
        allocation_names = ("tiles",)
        if gguf_lm_head_q6_x8_sidecar_enabled():
            allocation_names = ("tiles", "x8")
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="gguf_q6_k_t16_v1",
            layout=LAYOUT_GGUF_Q6_K_T16,
            allocation_names=allocation_names,
        )
    if qtype == GGMLQuantizationType.Q6_K and slot_path.startswith("layers."):
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
    elif spec.layout in {
        LAYOUT_GGUF_Q4_K_T16,
        LAYOUT_GGUF_Q4_K_X8,
        LAYOUT_GGUF_Q5_K_T16,
        LAYOUT_GGUF_Q5_K_QMICRO_T16,
        LAYOUT_GGUF_Q6_K_T16,
        LAYOUT_GGUF_Q8_0_T16,
        LAYOUT_GGUF_Q5_K_X8,
        LAYOUT_GGUF_Q6_K_X8,
    }:
        if spec.layout == LAYOUT_GGUF_Q4_K_T16:
            packed = repack_gguf_q4_k_tile16(raw)
        elif spec.layout == LAYOUT_GGUF_Q4_K_X8:
            packed = repack_gguf_q4_k_x8(raw)
        elif spec.layout == LAYOUT_GGUF_Q5_K_T16:
            packed = repack_gguf_q5_k_tile16(raw)
        elif spec.layout == LAYOUT_GGUF_Q5_K_QMICRO_T16:
            packed = repack_gguf_q5_k_qmicro_tile16(raw)
        elif spec.layout == LAYOUT_GGUF_Q6_K_T16:
            packed = repack_gguf_q6_k_tile16(raw if raw.ndim == 3 else raw[None, ...])
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
    "LAYOUT_GGUF_Q4_K_T16",
    "LAYOUT_GGUF_Q4_K_X8",
    "LAYOUT_GGUF_Q5_K_QMICRO_T16",
    "LAYOUT_GGUF_Q5_K_T16",
    "LAYOUT_GGUF_Q5_K_X8",
    "LAYOUT_GGUF_Q6_K_T16",
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
]
