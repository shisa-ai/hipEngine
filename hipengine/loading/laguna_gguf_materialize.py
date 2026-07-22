"""Dry resident-layout and unified-memory admission planning for Laguna GGUF."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import resource
import time
from types import MappingProxyType
from typing import Any, Mapping

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo
from hipengine.loading.laguna_gguf import (
    LagunaGGUFConfig,
    LagunaGGUFModelMap,
    build_laguna_gguf_tensor_map,
)
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    load_host_array_to_device_as_dtype,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import (
    GGUF_Q4_K_TILE16_BLOCK_BYTES,
    repack_gguf_q4_k_pack8,
    repack_gguf_q4_k_tile16,
)
from hipengine.quant.gguf_t16 import (
    GGUF_Q6_K_T16_BLOCK_BYTES,
    repack_gguf_q6_k_tile16,
)

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
class LagunaGGUFMaterializationProfile:
    """Natural-path phase and process-counter deltas for one resident weight."""

    slot_path: str
    tensor_name: str
    layout: str
    source_nbytes: int
    resident_nbytes: int
    source_map_seconds: float
    repack_seconds: float
    allocation_seconds: float
    upload_seconds: float
    other_seconds: float
    total_seconds: float
    allocation_count: int
    upload_count: int
    allocated_nbytes: int
    uploaded_nbytes: int
    source_map_minor_faults: int
    source_map_major_faults: int
    source_map_read_bytes: int | None
    repack_minor_faults: int
    repack_major_faults: int
    repack_read_bytes: int | None
    upload_minor_faults: int
    upload_major_faults: int
    upload_read_bytes: int | None
    minor_faults: int
    major_faults: int
    read_bytes: int | None
    rss_bytes: int
    max_rss_bytes: int


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


@dataclass(frozen=True)
class LagunaGGUFDeviceWeight:
    """Owned device allocations for one Laguna logical weight."""

    spec: LagunaGGUFWeightSpec
    allocations: Mapping[str, DeviceTensorAllocation]
    backend: str

    def allocation(self, name: str | None = None) -> DeviceTensorAllocation:
        key = next(iter(self.allocations)) if name is None else name
        return self.allocations[key]

    @property
    def resident_nbytes(self) -> int:
        return sum(allocation.buffer.nbytes for allocation in self.allocations.values())

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for allocation in reversed(tuple(self.allocations.values())):
            allocation.free(runtime=runtime)


@dataclass(frozen=True)
class LagunaGGUFResidentLayerWeights:
    layer_id: int
    attention_type: str
    mlp_type: str
    weights: Mapping[str, LagunaGGUFDeviceWeight]

    def weight(self, slot: str) -> LagunaGGUFDeviceWeight:
        return self.weights[slot]


@dataclass(frozen=True)
class LagunaGGUFResidentWeights:
    """Device-resident selected or full Laguna weight set with owned teardown."""

    config: LagunaGGUFConfig
    root_weights: Mapping[str, LagunaGGUFDeviceWeight]
    layers: tuple[LagunaGGUFResidentLayerWeights, ...]
    backend: str
    admission: LagunaMemoryAdmissionPlan

    def root(self, slot: str) -> LagunaGGUFDeviceWeight:
        return self.root_weights[slot]

    def layer(self, layer_id: int) -> LagunaGGUFResidentLayerWeights:
        return self.layers[layer_id]

    @property
    def weights(self) -> tuple[LagunaGGUFDeviceWeight, ...]:
        return (
            *tuple(self.root_weights.values()),
            *tuple(weight for layer in self.layers for weight in layer.weights.values()),
        )

    @property
    def resident_nbytes(self) -> int:
        return sum(weight.resident_nbytes for weight in self.weights)

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for weight in reversed(self.weights):
            weight.free(runtime=runtime)


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
        raise ValueError(f"context_length must be within [1, {weights.config.context_length}]")
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
    peak = weights.resident_nbytes + kv.resident_nbytes + scratch + reserve + transient
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


def materialize_laguna_gguf_weights(
    reader_or_path: GGUFReader | str | Path,
    *,
    selected_slots: Iterable[str] | None = None,
    context_length: int = 4_096,
    available_bytes: int | None = None,
    storage_dtype: str = "bf16",
    scratch_nbytes: int = DEFAULT_LAGUNA_SCRATCH_BYTES,
    safety_reserve_nbytes: int = DEFAULT_LAGUNA_SAFETY_RESERVE_BYTES,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    backend: str = "hip_gfx1151",
    progress: Callable[[int, int, LagunaGGUFWeightSpec], None] | None = None,
    profile: Callable[[LagunaGGUFMaterializationProfile], None] | None = None,
) -> LagunaGGUFResidentWeights:
    """Stream selected or all planned Laguna weights into owned device buffers."""

    reader = (
        GGUFReader(reader_or_path) if isinstance(reader_or_path, (str, Path)) else reader_or_path
    )
    model_map = build_laguna_gguf_tensor_map(reader.info)
    plan = plan_laguna_gguf_materialization(model_map)
    active_runtime = runtime if runtime is not None else get_hip_runtime()
    if available_bytes is None:
        try:
            available_bytes = int(active_runtime.mem_get_info()[0])
        except AttributeError as exc:
            raise ValueError(
                "available_bytes is required when the runtime has no mem_get_info()"
            ) from exc
    admission = plan_laguna_memory_admission(
        plan,
        context_length=context_length,
        available_bytes=available_bytes,
        storage_dtype=storage_dtype,
        scratch_nbytes=scratch_nbytes,
        safety_reserve_nbytes=safety_reserve_nbytes,
    )

    specs_by_path = {spec.slot_path: spec for spec in plan.specs}
    selected = None if selected_slots is None else {str(item) for item in selected_slots}
    if selected is not None:
        unknown = tuple(sorted(selected - set(specs_by_path)))
        if unknown:
            raise ValueError(f"unknown selected Laguna slots: {unknown}")
    selected_count = len(plan.specs) if selected is None else len(selected)
    completed: list[LagunaGGUFDeviceWeight] = []
    complete_count = 0
    try:
        root_weights: dict[str, LagunaGGUFDeviceWeight] = {}
        for slot, spec in plan.root_specs.items():
            if selected is not None and spec.slot_path not in selected:
                continue
            weight = _materialize_spec(
                spec,
                reader,
                device=device,
                runtime=active_runtime,
                backend=backend,
                profile=profile,
            )
            root_weights[slot] = weight
            completed.append(weight)
            complete_count += 1
            if progress is not None:
                progress(complete_count, selected_count, spec)

        resident_layers: list[LagunaGGUFResidentLayerWeights] = []
        for layer in model_map.layers:
            layer_weights: dict[str, LagunaGGUFDeviceWeight] = {}
            for slot, spec in plan.layer_specs[layer.layer_id].items():
                if selected is not None and spec.slot_path not in selected:
                    continue
                weight = _materialize_spec(
                    spec,
                    reader,
                    device=device,
                    runtime=active_runtime,
                    backend=backend,
                    profile=profile,
                )
                layer_weights[slot] = weight
                completed.append(weight)
                complete_count += 1
                if progress is not None:
                    progress(complete_count, selected_count, spec)
            resident_layers.append(
                LagunaGGUFResidentLayerWeights(
                    layer_id=layer.layer_id,
                    attention_type=layer.attention_type,
                    mlp_type=layer.mlp_type,
                    weights=MappingProxyType(layer_weights),
                )
            )
    except Exception:
        for weight in reversed(completed):
            weight.free(runtime=active_runtime)
        raise

    return LagunaGGUFResidentWeights(
        config=plan.config,
        root_weights=MappingProxyType(root_weights),
        layers=tuple(resident_layers),
        backend=backend,
        admission=admission,
    )


def _materialize_spec(
    spec: LagunaGGUFWeightSpec,
    reader: GGUFReader,
    *,
    device: Device | None,
    runtime: HipRuntime,
    backend: str,
    profile: Callable[[LagunaGGUFMaterializationProfile], None] | None = None,
) -> LagunaGGUFDeviceWeight:
    import numpy as np

    total_started = time.perf_counter()
    total_before = _process_counters() if profile is not None else None
    source_started = time.perf_counter()
    source_before = _process_counters() if profile is not None else None
    raw = np.ascontiguousarray(reader.tensor_data(spec.source.name))
    source_map_seconds = time.perf_counter() - source_started
    source_after = _process_counters() if profile is not None else None
    source_delta = _counter_delta(source_before, source_after)
    repack_seconds = 0.0
    repack_delta = _ProcessCounterDelta()
    timed_runtime = _TimedUploadRuntime(runtime) if profile is not None else runtime

    def measured_repack(operation: Callable[[], Any]) -> Any:
        nonlocal repack_seconds, repack_delta
        if profile is None:
            return operation()
        before = _process_counters()
        started = time.perf_counter()
        result = operation()
        repack_seconds += time.perf_counter() - started
        repack_delta = repack_delta + _counter_delta(before, _process_counters())
        return result

    allocations: dict[str, DeviceTensorAllocation] = {}
    try:
        if spec.layout == LAYOUT_DENSE_F16:
            allocations["raw"] = load_host_array_to_device_as_dtype(
                spec.source.name,
                raw,
                DType.FP16,
                source_dtype="F16",
                device=device,
                runtime=timed_runtime,
            )
        elif spec.layout == LAYOUT_DENSE_F32:
            allocations["raw"] = load_host_array_to_device_as_dtype(
                spec.source.name,
                raw,
                DType.FP32,
                source_dtype="F32",
                device=device,
                runtime=timed_runtime,
            )
        elif spec.layout == LAYOUT_RAW_GGUF:
            allocations["raw"] = load_host_array_to_device_as_dtype(
                spec.source.name,
                raw,
                DType.INT8,
                source_dtype="I8",
                device=device,
                runtime=timed_runtime,
            )
        elif spec.layout == LAYOUT_Q4_K_PACK8:
            packed = measured_repack(lambda: repack_gguf_q4_k_pack8(raw))
            allocations["qweight"] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.pack8.qweight",
                packed.qweight,
                DType.INT32,
                source_dtype="I32",
                device=device,
                runtime=timed_runtime,
            )
            allocations["scales"] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.pack8.scales",
                packed.scales,
                DType.FP32,
                source_dtype="F32",
                device=device,
                runtime=timed_runtime,
            )
            allocations["mins"] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.pack8.mins",
                packed.mins,
                DType.FP32,
                source_dtype="F32",
                device=device,
                runtime=timed_runtime,
            )
        elif spec.layout == LAYOUT_GGUF_Q4_K_T16:
            packed = measured_repack(lambda: repack_gguf_q4_k_tile16(raw))
            allocations["tiles"] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.q4_t16.tiles",
                packed.tiles,
                DType.INT8,
                source_dtype="I8",
                device=device,
                runtime=timed_runtime,
            )
        elif spec.layout == LAYOUT_GGUF_Q6_K_T16:
            packed = measured_repack(
                lambda: repack_gguf_q6_k_tile16(raw if raw.ndim == 3 else raw[None, ...])
            )
            allocations["tiles"] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.q6_t16.tiles",
                packed.tiles,
                DType.INT8,
                source_dtype="I8",
                device=device,
                runtime=timed_runtime,
            )
        else:
            raise ValueError(f"unsupported Laguna materialization layout {spec.layout!r}")

        actual_names = tuple(allocations)
        if actual_names != spec.allocation_names:
            raise ValueError(
                f"Laguna allocation names differ for {spec.slot_path}: "
                f"planned={spec.allocation_names} actual={actual_names}"
            )
        for name, allocation in allocations.items():
            planned_nbytes = int(spec.allocation_nbytes[name])
            if allocation.buffer.nbytes != planned_nbytes:
                raise ValueError(
                    f"Laguna allocation bytes differ for {spec.slot_path}.{name}: "
                    f"planned={planned_nbytes} actual={allocation.buffer.nbytes}"
                )
    except Exception:
        for allocation in reversed(tuple(allocations.values())):
            allocation.free(runtime=runtime)
        raise

    weight = LagunaGGUFDeviceWeight(
        spec=spec,
        allocations=MappingProxyType(allocations),
        backend=backend,
    )
    if profile is not None:
        assert isinstance(timed_runtime, _TimedUploadRuntime)
        total_seconds = time.perf_counter() - total_started
        total_delta = _counter_delta(total_before, _process_counters())
        measured_seconds = (
            source_map_seconds
            + repack_seconds
            + timed_runtime.allocation_seconds
            + timed_runtime.upload_seconds
        )
        rss_bytes = _rss_bytes()
        max_rss_bytes = max(rss_bytes, _max_rss_bytes())
        record = LagunaGGUFMaterializationProfile(
            slot_path=spec.slot_path,
            tensor_name=spec.source.name,
            layout=spec.layout,
            source_nbytes=spec.source.nbytes,
            resident_nbytes=spec.resident_nbytes,
            source_map_seconds=source_map_seconds,
            repack_seconds=repack_seconds,
            allocation_seconds=timed_runtime.allocation_seconds,
            upload_seconds=timed_runtime.upload_seconds,
            other_seconds=max(0.0, total_seconds - measured_seconds),
            total_seconds=total_seconds,
            allocation_count=timed_runtime.allocation_count,
            upload_count=timed_runtime.upload_count,
            allocated_nbytes=timed_runtime.allocated_nbytes,
            uploaded_nbytes=timed_runtime.uploaded_nbytes,
            source_map_minor_faults=source_delta.minor_faults,
            source_map_major_faults=source_delta.major_faults,
            source_map_read_bytes=source_delta.read_bytes,
            repack_minor_faults=repack_delta.minor_faults,
            repack_major_faults=repack_delta.major_faults,
            repack_read_bytes=repack_delta.read_bytes,
            upload_minor_faults=timed_runtime.upload_delta.minor_faults,
            upload_major_faults=timed_runtime.upload_delta.major_faults,
            upload_read_bytes=timed_runtime.upload_delta.read_bytes,
            minor_faults=total_delta.minor_faults,
            major_faults=total_delta.major_faults,
            read_bytes=total_delta.read_bytes,
            rss_bytes=rss_bytes,
            max_rss_bytes=max_rss_bytes,
        )
        try:
            profile(record)
        except Exception:
            weight.free(runtime=runtime)
            raise
    return weight


@dataclass(frozen=True)
class _ProcessCounters:
    minor_faults: int
    major_faults: int
    read_bytes: int | None


@dataclass(frozen=True)
class _ProcessCounterDelta:
    minor_faults: int = 0
    major_faults: int = 0
    read_bytes: int | None = 0

    def __add__(self, other: _ProcessCounterDelta) -> _ProcessCounterDelta:
        if self.read_bytes is None or other.read_bytes is None:
            read_bytes = None
        else:
            read_bytes = self.read_bytes + other.read_bytes
        return _ProcessCounterDelta(
            minor_faults=self.minor_faults + other.minor_faults,
            major_faults=self.major_faults + other.major_faults,
            read_bytes=read_bytes,
        )


class _TimedUploadRuntime:
    def __init__(self, runtime: HipRuntime) -> None:
        self.runtime = runtime
        self.allocation_seconds = 0.0
        self.upload_seconds = 0.0
        self.allocation_count = 0
        self.upload_count = 0
        self.allocated_nbytes = 0
        self.uploaded_nbytes = 0
        self.upload_delta = _ProcessCounterDelta()

    def malloc(self, nbytes: int) -> int:
        started = time.perf_counter()
        try:
            return self.runtime.malloc(nbytes)
        finally:
            self.allocation_seconds += time.perf_counter() - started
            self.allocation_count += 1
            self.allocated_nbytes += int(nbytes)

    def memcpy(self, dst: int, src: int, count: int, kind: Any) -> None:
        before = _process_counters()
        started = time.perf_counter()
        try:
            self.runtime.memcpy(dst, src, count, kind)
        finally:
            self.upload_seconds += time.perf_counter() - started
            self.upload_count += 1
            self.uploaded_nbytes += int(count)
            self.upload_delta = self.upload_delta + _counter_delta(before, _process_counters())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)


def _process_counters() -> _ProcessCounters:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    read_bytes: int | None = None
    try:
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            if line.startswith("read_bytes:"):
                read_bytes = int(line.split(":", 1)[1].strip())
                break
    except (FileNotFoundError, OSError, ValueError):
        pass
    return _ProcessCounters(
        minor_faults=int(usage.ru_minflt),
        major_faults=int(usage.ru_majflt),
        read_bytes=read_bytes,
    )


def _counter_delta(
    before: _ProcessCounters | None,
    after: _ProcessCounters | None,
) -> _ProcessCounterDelta:
    if before is None or after is None:
        return _ProcessCounterDelta(read_bytes=None)
    read_bytes = (
        None
        if before.read_bytes is None or after.read_bytes is None
        else max(0, after.read_bytes - before.read_bytes)
    )
    return _ProcessCounterDelta(
        minor_faults=max(0, after.minor_faults - before.minor_faults),
        major_faults=max(0, after.major_faults - before.major_faults),
        read_bytes=read_bytes,
    )


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1_024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1_024


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
        raise ValueError(f"Q4_K pack8 requires out%8==0 and in%32==0: {tensor.name} {tensor.shape}")
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
    "LagunaGGUFDeviceWeight",
    "LagunaGGUFMaterializationPlan",
    "LagunaGGUFMaterializationProfile",
    "LagunaGGUFResidentLayerWeights",
    "LagunaGGUFResidentWeights",
    "LagunaGGUFWeightSpec",
    "LagunaKVMemoryPlan",
    "LagunaMemoryAdmissionError",
    "LagunaMemoryAdmissionPlan",
    "materialize_laguna_gguf_weights",
    "plan_laguna_gguf_materialization",
    "plan_laguna_memory_admission",
]
