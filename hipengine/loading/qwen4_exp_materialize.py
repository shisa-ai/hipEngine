"""One-layout Qwen4Exp residency planning and sparse PLE mmap ownership."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import mmap
import os
import resource
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    load_host_array_to_device_as_dtype,
)
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
)
from hipengine.loading.qwen4_exp_gguf import (
    GDN,
    Qwen4ExpGGUFConfig,
    Qwen4ExpGGUFModelMap,
    Qwen4ExpGGUFTensorRef,
)
from hipengine.quant.gguf import dequantization_supported, dequantize_gguf_data

LAYOUT_RAW_GGUF = "raw_gguf"
LAYOUT_PLE_SPARSE_MMAP = "ple_sparse_mmap"


@dataclass(frozen=True)
class Qwen4ExpGGUFWeightSpec:
    slot_path: str
    source_ref: Qwen4ExpGGUFTensorRef
    quant_key: str
    layout: str
    allocation_names: tuple[str, ...]
    device_resident: bool
    device_nbytes: int

    @property
    def source(self) -> GGUFTensorInfo:
        return self.source_ref.tensor


@dataclass(frozen=True)
class Qwen4ExpResidencyPlan:
    config: Qwen4ExpGGUFConfig
    root_specs: Mapping[str, Qwen4ExpGGUFWeightSpec]
    layer_specs: tuple[Mapping[str, Qwen4ExpGGUFWeightSpec], ...]
    ple_spec: Qwen4ExpGGUFWeightSpec
    raw_payload_bytes: int
    device_weight_bytes: int
    replacement_payload_bytes: int
    alternate_layout_bytes: int
    ple_mmap_bytes: int
    staging_buffer_count: int
    staging_row_capacity: int
    staging_bytes: int
    tensor_bytes_by_type: Mapping[str, int]

    @property
    def specs(self) -> tuple[Qwen4ExpGGUFWeightSpec, ...]:
        return (
            *self.root_specs.values(),
            *(spec for layer in self.layer_specs for spec in layer.values()),
            self.ple_spec,
        )

    @property
    def device_specs(self) -> tuple[Qwen4ExpGGUFWeightSpec, ...]:
        return tuple(spec for spec in self.specs if spec.device_resident)


@dataclass(frozen=True)
class Qwen4ExpMemoryAdmissionPlan:
    available_device_bytes: int
    required_bytes: int
    device_weight_bytes: int
    staging_bytes: int
    kv_bytes: int
    index_bytes: int
    runtime_state_bytes: int
    scratch_bytes: int
    reserve_bytes: int
    context_tokens: int
    resident_capacity: int

    @property
    def passed(self) -> bool:
        return self.required_bytes <= self.available_device_bytes

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.required_bytes - self.available_device_bytes)


def plan_qwen4_exp_residency(
    model_map: Qwen4ExpGGUFModelMap,
    *,
    staging_token_capacity: int = 256,
    ple_rows_per_token: int = 16,
) -> Qwen4ExpResidencyPlan:
    """Plan one raw device owner per hot tensor and one sparse mmap PLE owner."""

    if not model_map.validation.passed:
        raise ValueError("qwen4exp tensor map must pass before residency planning")
    if model_map.ple_table is None:
        raise ValueError("qwen4exp tensor map has no PLE table")
    token_capacity = int(staging_token_capacity)
    rows_per_token = int(ple_rows_per_token)
    if token_capacity <= 0 or rows_per_token <= 0:
        raise ValueError("staging capacities must be positive")

    def device_spec(slot_path: str, ref: Qwen4ExpGGUFTensorRef) -> Qwen4ExpGGUFWeightSpec:
        quant_key, layout = _qwen4_exp_runtime_layout(ref.tensor)
        return Qwen4ExpGGUFWeightSpec(
            slot_path=slot_path,
            source_ref=ref,
            quant_key=quant_key,
            layout=layout,
            allocation_names=("raw",),
            device_resident=True,
            device_nbytes=int(ref.tensor.nbytes),
        )

    roots = {
        slot: device_spec(f"root.{slot}", ref)
        for slot, ref in model_map.roots.items()
    }
    layers = tuple(
        MappingProxyType(
            {
                slot: device_spec(f"layers.{layer.layer_id}.{slot}", ref)
                for slot, ref in layer.slots.items()
            }
        )
        for layer in model_map.layers
    )
    ple_spec = Qwen4ExpGGUFWeightSpec(
        slot_path="ple.table",
        source_ref=model_map.ple_table,
        quant_key="ple_mmap",
        layout=LAYOUT_PLE_SPARSE_MMAP,
        allocation_names=(),
        device_resident=False,
        device_nbytes=0,
    )
    specs = (*roots.values(), *(spec for layer in layers for spec in layer.values()), ple_spec)
    source_names = [spec.source.name for spec in specs]
    if len(source_names) != len(set(source_names)):
        raise ValueError("residency plan would create duplicate logical tensor owners")
    type_bytes: Counter[str] = Counter()
    for spec in specs:
        type_bytes[spec.source.ggml_type_name] += int(spec.source.nbytes)
    raw_payload = sum(int(spec.source.nbytes) for spec in specs)
    device_bytes = sum(spec.device_nbytes for spec in specs)
    row_capacity = token_capacity * rows_per_token
    staging_bytes = 2 * row_capacity * model_map.config.ple_row_width * 4
    return Qwen4ExpResidencyPlan(
        config=model_map.config,
        root_specs=MappingProxyType(roots),
        layer_specs=layers,
        ple_spec=ple_spec,
        raw_payload_bytes=raw_payload,
        device_weight_bytes=device_bytes,
        replacement_payload_bytes=0,
        alternate_layout_bytes=0,
        ple_mmap_bytes=int(ple_spec.source.nbytes),
        staging_buffer_count=2,
        staging_row_capacity=row_capacity,
        staging_bytes=staging_bytes,
        tensor_bytes_by_type=MappingProxyType(dict(sorted(type_bytes.items()))),
    )


def plan_qwen4_exp_memory_admission(
    residency: Qwen4ExpResidencyPlan,
    *,
    available_device_bytes: int,
    context_tokens: int,
    resident_capacity: int = 1,
    scratch_bytes: int = 4 * 1024**3,
    reserve_bytes: int = 4 * 1024**3,
) -> Qwen4ExpMemoryAdmissionPlan:
    """Account complete resident, KV/index, recurrent, scratch, and reserve bytes."""

    available = int(available_device_bytes)
    context = int(context_tokens)
    capacity = int(resident_capacity)
    scratch = int(scratch_bytes)
    reserve = int(reserve_bytes)
    if available < 0 or scratch < 0 or reserve < 0:
        raise ValueError("byte counts must be non-negative")
    if context <= 0 or context > residency.config.context_length:
        raise ValueError("context_tokens must be in 1..native context length")
    if capacity <= 0:
        raise ValueError("resident_capacity must be positive")
    kv_bytes = capacity * context * residency.config.bf16_kv_bytes_per_token
    index_bytes = capacity * _qsa_index_state_bytes(residency.config, context)
    runtime_state = capacity * _runtime_state_bytes_per_request(residency.config)
    required = (
        residency.device_weight_bytes
        + residency.staging_bytes
        + kv_bytes
        + index_bytes
        + runtime_state
        + scratch
        + reserve
    )
    return Qwen4ExpMemoryAdmissionPlan(
        available_device_bytes=available,
        required_bytes=required,
        device_weight_bytes=residency.device_weight_bytes,
        staging_bytes=residency.staging_bytes,
        kv_bytes=kv_bytes,
        index_bytes=index_bytes,
        runtime_state_bytes=runtime_state,
        scratch_bytes=scratch,
        reserve_bytes=reserve,
        context_tokens=context,
        resident_capacity=capacity,
    )


def _qsa_index_state_bytes(config: Qwen4ExpGGUFConfig, context_tokens: int) -> int:
    """Return exact device bytes for the current raw-FP32 QSA index owner."""

    context = int(context_tokens)
    ratio = config.qsa_compression_ratio
    complete_blocks = context // ratio
    per_layer = (
        context * config.indexer_key_length * 4
        + complete_blocks * ratio * 4  # physical member indices
        + complete_blocks * 8  # logical block starts
        + complete_blocks * config.indexer_key_length * 4  # pooled FP32 keys
        + complete_blocks * 4  # scores for one c1 query
        + config.qsa_block_budget * 8  # selected complete-block starts
        + 4  # selected count
        + 8  # current query position
        + config.qsa_dense_equivalent_max_tokens * 8  # expanded blocks + tail
    )
    return config.qsa_layer_count * per_layer


def _runtime_state_bytes_per_request(config: Qwen4ExpGGUFConfig) -> int:
    gdn_layers = config.layer_types.count(GDN)
    fp32_bytes = 4
    matrix_state = (
        gdn_layers * config.gdn_inner_size * config.gdn_state_size * fp32_bytes
    )
    conv_channels = (
        config.gdn_inner_size + 2 * config.gdn_group_count * config.gdn_state_size
    )
    conv_state = (
        gdn_layers
        * config.gdn_conv_kernel
        * conv_channels
        * fp32_bytes
    )
    ple_history = (
        (config.ple_conv_kernel - 1)
        * config.ple_ngram_size
        * config.residual_width
        * fp32_bytes
    )
    bf16_residual = config.residual_width * 2
    return matrix_state + conv_state + ple_history + bf16_residual


@dataclass(frozen=True)
class Qwen4ExpDeviceWeight:
    spec: Qwen4ExpGGUFWeightSpec
    backend: str
    allocations: Mapping[str, DeviceTensorAllocation]

    def allocation(self, name: str | None = None) -> DeviceTensorAllocation:
        key = self.spec.allocation_names[0] if name is None else str(name)
        return self.allocations[key]

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for allocation in reversed(tuple(self.allocations.values())):
            allocation.free(runtime=runtime)


@dataclass
class Qwen4ExpResidentWeights:
    """Physical Qwen4Exp owners with one device allocation per hot tensor."""

    plan: Qwen4ExpResidencyPlan
    device_weights: Mapping[str, Any]
    ple_table: "Qwen4ExpPLEMMapTable"
    ple_staging: "Qwen4ExpPLEStagingRing"
    runtime: Any | None = None
    closed: bool = False

    def weight(self, slot_path: str) -> Any:
        if self.closed:
            raise RuntimeError("Qwen4Exp resident weights are closed")
        return self.device_weights[slot_path]

    def close(self) -> None:
        if self.closed:
            return
        self.ple_staging.close()
        self.ple_table.close()
        for allocation in reversed(tuple(self.device_weights.values())):
            allocation.free(runtime=self.runtime)
        self.device_weights = MappingProxyType({})
        self.closed = True


def materialize_qwen4_exp_weights(
    readers: Sequence[Any],
    *,
    plan: Qwen4ExpResidencyPlan,
    runtime: HipRuntime | None = None,
    device_loader: Any | None = None,
    pin_ple_staging: bool = True,
    backend: str = "hip_gfx1100",
) -> Qwen4ExpResidentWeights:
    """Materialize hot raw weights once and create the sparse PLE owner."""

    reader_parts = tuple(readers)
    expected_parts = max(spec.source_ref.part_index for spec in plan.specs) + 1
    if len(reader_parts) != expected_parts:
        raise ValueError(
            f"reader part count {len(reader_parts)} does not match plan {expected_parts}"
        )
    load = device_loader or materialize_qwen4_exp_raw_weight
    active_runtime = runtime
    if pin_ple_staging and active_runtime is None:
        active_runtime = get_hip_runtime()
    allocations: dict[str, Any] = {}
    table: Qwen4ExpPLEMMapTable | None = None
    ring: Qwen4ExpPLEStagingRing | None = None
    try:
        for spec in plan.device_specs:
            reader = reader_parts[spec.source_ref.part_index]
            allocation = load(spec, reader, runtime=active_runtime)
            allocations[spec.slot_path] = Qwen4ExpDeviceWeight(
                spec=spec,
                backend=str(backend),
                allocations=MappingProxyType({"raw": allocation}),
            )
        ple_ref = plan.ple_spec.source_ref
        table = Qwen4ExpPLEMMapTable(
            reader_parts[ple_ref.part_index],
            ple_ref.tensor,
            semantic_rows=plan.config.ple_row_count,
        )
        ring = Qwen4ExpPLEStagingRing.create(
            table,
            row_capacity=plan.staging_row_capacity,
            runtime=active_runtime if pin_ple_staging else None,
        )
    except Exception:
        if ring is not None:
            ring.close()
        if table is not None:
            table.close()
        for allocation in reversed(tuple(allocations.values())):
            allocation.free(runtime=active_runtime)
        raise
    return Qwen4ExpResidentWeights(
        plan=plan,
        device_weights=MappingProxyType(allocations),
        ple_table=table,
        ple_staging=ring,
        runtime=active_runtime,
    )


def _qwen4_exp_runtime_layout(tensor: GGUFTensorInfo) -> tuple[str, str]:
    if tensor.ggml_type_name == "F32":
        return "f32", LAYOUT_DENSE_F32
    if tensor.ggml_type_name == "BF16":
        return "bf16", LAYOUT_DENSE_BF16
    return f"gguf_{tensor.ggml_type_name.lower()}", LAYOUT_RAW_GGUF


def materialize_qwen4_exp_raw_weight(
    spec: Qwen4ExpGGUFWeightSpec,
    reader: Any,
    *,
    runtime: HipRuntime | None = None,
) -> DeviceTensorAllocation:
    raw = reader.tensor_data(spec.source.name)
    if spec.source.ggml_type_name == "F32":
        dtype, source_dtype = DType.FP32, "F32"
    elif spec.source.ggml_type_name == "F16":
        dtype, source_dtype = DType.FP16, "F16"
    elif spec.source.ggml_type_name == "BF16":
        dtype, source_dtype = DType.BF16, "BF16"
    else:
        dtype, source_dtype = DType.INT8, "I8"
    return load_host_array_to_device_as_dtype(
        spec.source.name,
        raw,
        dtype,
        source_dtype=source_dtype,
        runtime=runtime,
    )


class Qwen4ExpPLEMMapTable:
    """One lazy GGUF memmap owner that dequantizes only requested PLE rows."""

    def __init__(self, reader: Any, tensor: GGUFTensorInfo, *, semantic_rows: int):
        if tensor.name != "per_layer_token_embd.weight":
            raise ValueError("tensor must be per_layer_token_embd.weight")
        if len(tensor.shape) != 2:
            raise ValueError("PLE tensor must be rank two")
        semantic = int(semantic_rows)
        if semantic <= 0 or semantic > tensor.shape[0]:
            raise ValueError("semantic_rows must be in 1..physical rows")
        if not dequantization_supported(tensor.ggml_type):
            raise ValueError(f"PLE qtype {tensor.ggml_type_name} has no CPU dequantizer")
        self.reader = reader
        self.tensor = tensor
        self.semantic_rows = semantic
        self.row_width = int(tensor.shape[1])
        self.rows_gathered = 0
        self._raw: Any | None = reader.tensor_data(tensor.name)
        self._telemetry: dict[str, Any] | None = None
        self._telemetry_rows: set[int] = set()
        self._telemetry_pages: set[int] = set()
        self._cache_mode = "unadvised"
        self._cache_range: dict[str, int] | None = None
        self._random_access_mode = "off"
        self._random_access_requested_mode = "off"
        self._row_prefetch_ranges: list[dict[str, int]] = []

    def enable_telemetry(self) -> None:
        """Enable opt-in cumulative PLE I/O telemetry."""

        self._telemetry = {
            "calls": 0,
            "requested_rows": 0,
            "requested_source_bytes": 0,
            "gather_dequant_wall_ns": 0,
            "staging_copy_wall_ns": 0,
            "h2d_wall_ns": 0,
            "h2d_bytes": 0,
            "minor_faults_proxy": 0,
            "major_faults_proxy": 0,
            "cache_mode": self._cache_mode,
            "random_access_mode": self._random_access_mode,
            "prefetch_ranges": (
                [dict(item) for item in self._row_prefetch_ranges]
                if self._row_prefetch_ranges
                else [dict(self._cache_range)] if self._cache_range else []
            ),
        }
        self._telemetry_rows.clear()
        self._telemetry_pages.clear()

    def telemetry(self) -> dict[str, Any] | None:
        if self._telemetry is None:
            return None
        pages = sorted(self._telemetry_pages)
        ranges: list[dict[str, int]] = []
        for page in pages:
            if ranges and page == ranges[-1]["last_page"] + 1:
                ranges[-1]["last_page"] = page
                ranges[-1]["page_count"] += 1
            else:
                ranges.append({"first_page": page, "last_page": page, "page_count": 1})
        return {
            **self._telemetry,
            "unique_rows": len(self._telemetry_rows),
            "unique_pages": len(pages),
            "adjacent_page_pairs": sum(item["page_count"] - 1 for item in ranges),
            "page_range_count": len(ranges),
            "page_ranges_sample": ranges[:32],
        }

    def _record_stage_copy(self, wall_ns: int) -> None:
        if self._telemetry is not None:
            self._telemetry["staging_copy_wall_ns"] += int(wall_ns)

    def record_h2d(self, *, nbytes: int, wall_ns: int) -> None:
        if self._telemetry is not None:
            self._telemetry["h2d_bytes"] += int(nbytes)
            self._telemetry["h2d_wall_ns"] += int(wall_ns)

    def gather_rows(self, row_indices: Any) -> np.ndarray:
        if self._raw is None:
            raise RuntimeError("PLE mmap table is closed")
        indices = np.asarray(row_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("row_indices must have shape [rows]")
        if indices.size and (
            int(np.min(indices)) < 0 or int(np.max(indices)) >= self.semantic_rows
        ):
            raise IndexError("PLE row index is outside semantic rows")
        if self._random_access_requested_mode != "off":
            self.prefetch_rows(
                indices, random_access=self._random_access_requested_mode
            )
        started = time.perf_counter_ns() if self._telemetry is not None else 0
        before_faults = resource.getrusage(resource.RUSAGE_SELF) if started else None
        selected = np.asarray(self._raw[indices])
        values = dequantize_gguf_data(selected, self.tensor.ggml_type).astype(np.float32)
        values = values.reshape(indices.size, self.row_width)
        self.rows_gathered += int(indices.size)
        if self._telemetry is not None:
            row_bytes = int(self.tensor.byte_shape[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            for index in indices.tolist():
                row = int(index)
                self._telemetry_rows.add(row)
                first = (int(self.tensor.data_offset) + row * row_bytes) // page_size
                last = (int(self.tensor.data_offset) + (row + 1) * row_bytes - 1) // page_size
                self._telemetry_pages.update(range(first, last + 1))
            after_faults = resource.getrusage(resource.RUSAGE_SELF)
            self._telemetry["calls"] += 1
            self._telemetry["requested_rows"] += int(indices.size)
            self._telemetry["requested_source_bytes"] += int(indices.size) * row_bytes
            self._telemetry["gather_dequant_wall_ns"] += time.perf_counter_ns() - started
            assert before_faults is not None
            self._telemetry["minor_faults_proxy"] += int(after_faults.ru_minflt - before_faults.ru_minflt)
            self._telemetry["major_faults_proxy"] += int(after_faults.ru_majflt - before_faults.ru_majflt)
        return values

    def configure_random_access(self, mode: str) -> None:
        """Configure default-off per-gather sparse prefetch advice."""

        selected = str(mode)
        if selected not in {"off", "auto", "on"}:
            raise ValueError("PLE random access mode must be off, auto, or on")
        self._random_access_requested_mode = selected

    def prefetch_rows(
        self, row_indices: Any, *, random_access: str = "auto"
    ) -> dict[str, Any]:
        """Apply sparse access advice and page-aligned WILLNEED row ranges."""

        if self._raw is None:
            raise RuntimeError("PLE mmap table is closed")
        requested = str(random_access)
        if requested not in {"off", "auto", "on"}:
            raise ValueError("PLE random access mode must be off, auto, or on")
        indices = np.asarray(row_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("row_indices must have shape [rows]")
        if indices.size and (
            int(np.min(indices)) < 0 or int(np.max(indices)) >= self.semantic_rows
        ):
            raise IndexError("PLE row index is outside semantic rows")
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        row_bytes = int(self.tensor.byte_shape[1])
        pages: set[int] = set()
        for index in indices.tolist():
            start = int(self.tensor.data_offset) + int(index) * row_bytes
            pages.update(range(start // page_size, (start + row_bytes - 1) // page_size + 1))
        ranges: list[dict[str, int]] = []
        for page in sorted(pages):
            offset = page * page_size
            if ranges and offset == ranges[-1]["offset"] + ranges[-1]["nbytes"]:
                ranges[-1]["nbytes"] += page_size
            else:
                ranges.append({"offset": offset, "nbytes": page_size})
        unique_rows = len({int(item) for item in indices.tolist()})
        useful_bytes = unique_rows * row_bytes
        prefetched_bytes = sum(item["nbytes"] for item in ranges)
        selected = requested
        if requested == "auto":
            selected = "on" if prefetched_bytes and useful_bytes * 2 < prefetched_bytes else "off"
        mapping = getattr(self._raw, "_mmap", None)
        mapping_applied = False
        if mapping is not None and hasattr(mapping, "madvise"):
            mapping.madvise(
                mmap.MADV_RANDOM if selected == "on" else mmap.MADV_NORMAL
            )
            mapping_applied = True
        source = getattr(self.reader, "path", None)
        fadvise = getattr(os, "posix_fadvise", None)
        applied = False
        if source is not None and callable(fadvise):
            descriptor = os.open(os.fspath(source), os.O_RDONLY)
            try:
                fadvise(
                    descriptor,
                    int(self.tensor.data_offset),
                    int(self.tensor.nbytes),
                    os.POSIX_FADV_RANDOM if selected == "on" else os.POSIX_FADV_NORMAL,
                )
                for item in ranges:
                    fadvise(
                        descriptor,
                        item["offset"],
                        item["nbytes"],
                        os.POSIX_FADV_WILLNEED,
                    )
                applied = True
            finally:
                os.close(descriptor)
        self._random_access_mode = selected
        self._row_prefetch_ranges = [dict(item) for item in ranges]
        if self._telemetry is not None:
            self._telemetry["random_access_mode"] = selected
            self._telemetry["prefetch_ranges"] = [dict(item) for item in ranges]
        return {
            "random_access_requested": requested,
            "random_access_selected": selected,
            "page_size": page_size,
            "unique_rows": unique_rows,
            "unique_pages": len(pages),
            "ranges": ranges,
            "file_advice_applied": applied,
            "mapping_advice_applied": mapping_applied,
        }

    def advise_cache(self, mode: str) -> dict[str, Any]:
        """Apply file-scoped warm/cold advice to only the PLE tensor range."""

        if self._raw is None:
            raise RuntimeError("PLE mmap table is closed")
        selected = str(mode)
        if selected not in {"warm", "cold"}:
            raise ValueError("PLE cache mode must be warm or cold")
        mmap_advice = (
            mmap.MADV_WILLNEED if selected == "warm" else mmap.MADV_DONTNEED
        )
        fadvise_advice = (
            os.POSIX_FADV_WILLNEED
            if selected == "warm"
            else os.POSIX_FADV_DONTNEED
        )
        mapping = getattr(self._raw, "_mmap", None)
        mapping_applied = False
        remapped = False
        if mapping is not None and hasattr(mapping, "madvise"):
            mapping.madvise(mmap_advice)
            mapping_applied = True
        if selected == "cold" and mapping is not None and hasattr(mapping, "close"):
            mapping.close()
            self._raw = None
            remapped = True
        file_applied = False
        source = getattr(self.reader, "path", None)
        fadvise = getattr(os, "posix_fadvise", None)
        if source is not None and callable(fadvise):
            descriptor = os.open(os.fspath(source), os.O_RDONLY)
            try:
                fadvise(
                    descriptor,
                    int(self.tensor.data_offset),
                    int(self.tensor.nbytes),
                    fadvise_advice,
                )
                file_applied = True
            finally:
                os.close(descriptor)
        if remapped:
            self._raw = self.reader.tensor_data(self.tensor.name)
        self._cache_mode = selected
        self._cache_range = {
            "offset": int(self.tensor.data_offset),
            "nbytes": int(self.tensor.nbytes),
        }
        if self._telemetry is not None:
            self._telemetry["cache_mode"] = selected
            self._telemetry["prefetch_ranges"] = [dict(self._cache_range)]
        return {
            "mode": selected,
            "scope": "ple_tensor_file_range",
            "offset": int(self.tensor.data_offset),
            "nbytes": int(self.tensor.nbytes),
            "file_advice_applied": file_applied,
            "mapping_advice_applied": mapping_applied,
            "mapping_reopened": remapped,
        }

    def close(self) -> None:
        if self._raw is None:
            return
        mapping = getattr(self._raw, "_mmap", None)
        if mapping is not None:
            mapping.close()
        self._raw = None


class Qwen4ExpPLEStagingRing:
    """Bounded double-buffered host row staging, optionally HIP page-locked."""

    def __init__(
        self,
        table: Qwen4ExpPLEMMapTable,
        buffers: tuple[np.ndarray, np.ndarray],
        *,
        runtime: Any | None,
        registered_ptrs: tuple[int, ...],
    ) -> None:
        self.table = table
        self._buffers = buffers
        self.runtime = runtime
        self._registered_ptrs = registered_ptrs
        self._active = 0
        self._closed = False

    @classmethod
    def create(
        cls,
        table: Qwen4ExpPLEMMapTable,
        *,
        row_capacity: int,
        runtime: Any | None = None,
    ) -> "Qwen4ExpPLEStagingRing":
        capacity = int(row_capacity)
        if capacity <= 0:
            raise ValueError("row_capacity must be positive")
        buffers = (
            np.empty((capacity, table.row_width), dtype=np.float32),
            np.empty((capacity, table.row_width), dtype=np.float32),
        )
        registered: list[int] = []
        if runtime is not None:
            try:
                for buffer in buffers:
                    ptr = int(buffer.ctypes.data)
                    runtime.host_register(ptr, int(buffer.nbytes))
                    registered.append(ptr)
            except Exception:
                for ptr in reversed(registered):
                    runtime.host_unregister(ptr)
                raise
        return cls(
            table,
            buffers,
            runtime=runtime,
            registered_ptrs=tuple(registered),
        )

    @property
    def row_capacity(self) -> int:
        return 0 if self._closed else int(self._buffers[0].shape[0])

    @property
    def pinned(self) -> bool:
        return bool(self._registered_ptrs) and not self._closed

    def stage(self, row_indices: Any) -> np.ndarray:
        if self._closed:
            raise RuntimeError("PLE staging ring is closed")
        indices = np.asarray(row_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("row_indices must have shape [rows]")
        if indices.size > self.row_capacity:
            raise ValueError("PLE row request exceeds staging capacity")
        buffer = self._buffers[self._active]
        values = self.table.gather_rows(indices)
        started = time.perf_counter_ns() if self.table._telemetry is not None else 0
        np.copyto(buffer[: indices.size], values)
        if started:
            self.table._record_stage_copy(time.perf_counter_ns() - started)
        result = buffer[: indices.size]
        self._active = 1 - self._active
        return result

    def record_h2d(self, *, nbytes: int, wall_ns: int) -> None:
        self.table.record_h2d(nbytes=nbytes, wall_ns=wall_ns)

    def close(self) -> None:
        if self._closed:
            return
        if self.runtime is not None:
            for ptr in reversed(self._registered_ptrs):
                self.runtime.host_unregister(ptr)
        self._registered_ptrs = ()
        self._buffers = ()  # type: ignore[assignment]
        self._closed = True


__all__ = [
    "LAYOUT_PLE_SPARSE_MMAP",
    "LAYOUT_RAW_GGUF",
    "Qwen4ExpDeviceWeight",
    "Qwen4ExpGGUFWeightSpec",
    "Qwen4ExpMemoryAdmissionPlan",
    "Qwen4ExpPLEMMapTable",
    "Qwen4ExpPLEStagingRing",
    "Qwen4ExpResidencyPlan",
    "Qwen4ExpResidentWeights",
    "materialize_qwen4_exp_raw_weight",
    "materialize_qwen4_exp_weights",
    "plan_qwen4_exp_memory_admission",
    "plan_qwen4_exp_residency",
]
