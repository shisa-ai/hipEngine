"""Dry resident-layout and unified-memory admission planning for Laguna GGUF."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time
from types import MappingProxyType
from typing import Any, Mapping

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.backends import backend_package_capability
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
    convert_gguf_q6_k_tile16_to_qmicro,
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
_LAGUNA_REPACKED_CACHE_SCHEMA = 1
_LAGUNA_REPACKED_CACHE_LAYOUT_VERSION = "laguna-repacked-v1"
_REPACKED_CACHE_LAYOUTS = frozenset({LAYOUT_Q4_K_PACK8, LAYOUT_GGUF_Q4_K_T16, LAYOUT_GGUF_Q6_K_T16})


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
    source_kind: str
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
class LagunaGGUFRepackedCache:
    """Validated versioned host artifact containing replacement-layout arrays."""

    path: Path
    manifest: Mapping[str, Any]
    plan_fingerprint: str

    def has(self, spec: LagunaGGUFWeightSpec) -> bool:
        return spec.slot_path in self.manifest["entries"]

    def payloads(self, spec: LagunaGGUFWeightSpec) -> Mapping[str, _ResidentPayload]:
        import numpy as np

        try:
            entry = self.manifest["entries"][spec.slot_path]
        except KeyError as exc:
            raise KeyError(f"Laguna repacked cache has no entry for {spec.slot_path}") from exc
        if entry["layout"] != spec.layout:
            raise ValueError(
                f"Laguna repacked cache layout mismatch for {spec.slot_path}: "
                f"cache={entry['layout']} plan={spec.layout}"
            )
        payloads: dict[str, _ResidentPayload] = {}
        for name in spec.allocation_names:
            try:
                metadata = entry["allocations"][name]
            except KeyError as exc:
                raise ValueError(
                    f"Laguna repacked cache allocation missing: {spec.slot_path}.{name}"
                ) from exc
            relative = Path(metadata["file"])
            artifact = (self.path / relative).resolve()
            if not artifact.is_relative_to(self.path.resolve()):
                raise ValueError(f"Laguna repacked cache path escapes root: {relative}")
            if not artifact.is_file():
                raise FileNotFoundError(f"Laguna repacked cache payload missing: {artifact}")
            # Read each replacement tensor eagerly into a bounded host buffer.
            # Uploading directly from an mmap forces HIP's synchronous copy to
            # fault tens of GiB one page at a time and is much slower on UMA.
            array = np.load(artifact, allow_pickle=False)
            expected_shape = tuple(int(value) for value in metadata["shape"])
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"Laguna repacked cache shape mismatch for {spec.slot_path}.{name}: "
                    f"cache={array.shape} expected={expected_shape}"
                )
            if array.dtype.str != metadata["numpy_dtype"]:
                raise ValueError(
                    f"Laguna repacked cache dtype mismatch for {spec.slot_path}.{name}: "
                    f"cache={array.dtype.str} expected={metadata['numpy_dtype']}"
                )
            if int(array.nbytes) != int(spec.allocation_nbytes[name]):
                raise ValueError(
                    f"Laguna repacked cache byte mismatch for {spec.slot_path}.{name}: "
                    f"cache={array.nbytes} expected={spec.allocation_nbytes[name]}"
                )
            payloads[name] = _ResidentPayload(
                array=array,
                dtype=DType.parse(metadata["runtime_dtype"]),
                source_dtype=str(metadata["source_dtype"]),
            )
        return MappingProxyType(payloads)


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
    source_abs_max: float | None = None
    source_row_l2_max: float | None = None

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
    q6_qmicro: bool = False

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


def build_laguna_repacked_cache(
    reader_or_path: GGUFReader | str | Path,
    cache_path: str | Path,
    *,
    selected_slots: Iterable[str] | None = None,
    source_sha256: str | None = None,
    progress: Callable[[int, int, LagunaGGUFWeightSpec], None] | None = None,
) -> dict[str, Any]:
    """Build an atomic host cache for only the transformed replacement layouts."""

    import numpy as np

    reader = (
        GGUFReader(reader_or_path) if isinstance(reader_or_path, (str, Path)) else reader_or_path
    )
    model_map = build_laguna_gguf_tensor_map(reader.info)
    plan = plan_laguna_gguf_materialization(model_map)
    specs_by_path = {spec.slot_path: spec for spec in plan.specs}
    selected = None if selected_slots is None else {str(item) for item in selected_slots}
    if selected is not None:
        unknown = tuple(sorted(selected - set(specs_by_path)))
        if unknown:
            raise ValueError(f"unknown selected Laguna cache slots: {unknown}")
    cache_specs = tuple(
        spec
        for spec in plan.specs
        if spec.layout in _REPACKED_CACHE_LAYOUTS
        and (selected is None or spec.slot_path in selected)
    )
    if not cache_specs:
        raise ValueError("Laguna repacked cache selection contains no transformed layouts")

    target = Path(cache_path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Laguna repacked cache already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    required_nbytes = sum(spec.resident_nbytes for spec in cache_specs)
    free_nbytes = shutil.disk_usage(target.parent).free
    if free_nbytes < required_nbytes + 64 * 2**20:
        raise OSError(
            "insufficient disk space for Laguna repacked cache: "
            f"required={required_nbytes + 64 * 2**20} free={free_nbytes}"
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    entries: dict[str, Any] = {}
    try:
        for index, spec in enumerate(cache_specs, start=1):
            raw = np.ascontiguousarray(reader.tensor_data(spec.source.name))
            payloads = _resident_payloads(spec, raw)
            allocation_entries: dict[str, Any] = {}
            slot_digest = hashlib.sha256(spec.slot_path.encode("utf-8")).hexdigest()[:16]
            for name, payload in payloads.items():
                filename = f"{index:04d}-{slot_digest}-{name}.npy"
                artifact = temporary / filename
                np.save(artifact, payload.array, allow_pickle=False)
                allocation_entries[name] = {
                    "file": filename,
                    "shape": [int(value) for value in payload.array.shape],
                    "numpy_dtype": payload.array.dtype.str,
                    "runtime_dtype": payload.dtype.value,
                    "source_dtype": payload.source_dtype,
                    "nbytes": int(payload.array.nbytes),
                }
            entries[spec.slot_path] = {
                "tensor_name": spec.source.name,
                "layout": spec.layout,
                "source_nbytes": int(spec.source.nbytes),
                "resident_nbytes": spec.resident_nbytes,
                "allocations": allocation_entries,
            }
            if progress is not None:
                progress(index, len(cache_specs), spec)

        manifest = {
            "schema": _LAGUNA_REPACKED_CACHE_SCHEMA,
            "layout_version": _LAGUNA_REPACKED_CACHE_LAYOUT_VERSION,
            "source": {**_source_identity(reader), "sha256": source_sha256},
            "plan_fingerprint": _materialization_plan_fingerprint(plan),
            "cacheable_layouts": sorted(_REPACKED_CACHE_LAYOUTS),
            "entry_count": len(entries),
            "resident_nbytes": required_nbytes,
            "entries": entries,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def open_laguna_repacked_cache(
    cache_path: str | Path,
    reader_or_path: GGUFReader | str | Path,
    *,
    source_sha256: str | None = None,
) -> LagunaGGUFRepackedCache:
    """Open and validate a cache against the current source and layout plan."""

    reader = (
        GGUFReader(reader_or_path) if isinstance(reader_or_path, (str, Path)) else reader_or_path
    )
    cache_root = Path(cache_path).expanduser().resolve()
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Laguna repacked cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema", -1)) != _LAGUNA_REPACKED_CACHE_SCHEMA:
        raise ValueError(f"unsupported Laguna repacked cache schema: {manifest.get('schema')}")
    if manifest.get("layout_version") != _LAGUNA_REPACKED_CACHE_LAYOUT_VERSION:
        raise ValueError(
            f"unsupported Laguna repacked cache layout: {manifest.get('layout_version')}"
        )

    plan = plan_laguna_gguf_materialization(build_laguna_gguf_tensor_map(reader.info))
    fingerprint = _materialization_plan_fingerprint(plan)
    if manifest.get("plan_fingerprint") != fingerprint:
        raise ValueError("Laguna repacked cache plan fingerprint does not match source")
    expected_source = _source_identity(reader)
    cached_source = manifest.get("source", {})
    for field in ("size_bytes", "mtime_ns"):
        expected = expected_source.get(field)
        cached = cached_source.get(field)
        if expected is not None and cached is not None and int(expected) != int(cached):
            raise ValueError(
                f"Laguna repacked cache source {field} mismatch: cache={cached} source={expected}"
            )
    if source_sha256 is not None and cached_source.get("sha256") != source_sha256:
        raise ValueError("Laguna repacked cache source SHA-256 mismatch")

    specs = {spec.slot_path: spec for spec in plan.specs}
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Laguna repacked cache has no entries")
    for slot_path, entry in entries.items():
        spec = specs.get(slot_path)
        if spec is None or spec.layout not in _REPACKED_CACHE_LAYOUTS:
            raise ValueError(f"Laguna repacked cache has unexpected slot: {slot_path}")
        if entry.get("layout") != spec.layout:
            raise ValueError(f"Laguna repacked cache layout mismatch for {slot_path}")
        if set(entry.get("allocations", {})) != set(spec.allocation_names):
            raise ValueError(f"Laguna repacked cache allocation names mismatch for {slot_path}")
        for name in spec.allocation_names:
            metadata = entry["allocations"][name]
            if int(metadata.get("nbytes", -1)) != int(spec.allocation_nbytes[name]):
                raise ValueError(
                    f"Laguna repacked cache allocation bytes mismatch for {slot_path}.{name}"
                )
    return LagunaGGUFRepackedCache(
        path=cache_root,
        manifest=MappingProxyType(manifest),
        plan_fingerprint=fingerprint,
    )


def _source_identity(reader: GGUFReader) -> dict[str, Any]:
    path = Path(reader.info.path).expanduser()
    try:
        stat = path.stat()
    except (FileNotFoundError, OSError):
        return {"path": str(path), "size_bytes": None, "mtime_ns": None}
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _materialization_plan_fingerprint(plan: LagunaGGUFMaterializationPlan) -> str:
    rows = []
    for spec in plan.specs:
        rows.append(
            {
                "slot_path": spec.slot_path,
                "tensor_name": spec.source.name,
                "ggml_type": int(spec.source.ggml_type),
                "shape": [int(value) for value in spec.source.shape],
                "data_offset": int(spec.source.data_offset),
                "source_nbytes": int(spec.source.nbytes),
                "layout": spec.layout,
                "resident_dtype": spec.resident_dtype,
                "allocations": dict(spec.allocation_nbytes),
            }
        )
    payload = json.dumps(
        {
            "layout_version": _LAGUNA_REPACKED_CACHE_LAYOUT_VERSION,
            "specs": rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    repacked_cache: LagunaGGUFRepackedCache | str | Path | None = None,
    repacked_cache_source_sha256: str | None = None,
    q6_qmicro: bool | None = None,
) -> LagunaGGUFResidentWeights:
    """Stream selected or all planned Laguna weights into owned device buffers."""

    reader = (
        GGUFReader(reader_or_path) if isinstance(reader_or_path, (str, Path)) else reader_or_path
    )
    model_map = build_laguna_gguf_tensor_map(reader.info)
    plan = plan_laguna_gguf_materialization(model_map)
    cache = (
        open_laguna_repacked_cache(
            repacked_cache,
            reader,
            source_sha256=repacked_cache_source_sha256,
        )
        if isinstance(repacked_cache, (str, Path))
        else repacked_cache
    )
    if repacked_cache_source_sha256 is not None and cache is not None:
        cached_sha256 = cache.manifest["source"].get("sha256")
        if cached_sha256 != repacked_cache_source_sha256:
            raise ValueError("Laguna repacked cache source SHA-256 mismatch")
    active_runtime = runtime if runtime is not None else get_hip_runtime()
    selected_q6_qmicro = bool(
        backend_package_capability(backend, "LAGUNA_Q6_QMICRO", False)
        if q6_qmicro is None
        else q6_qmicro
    )
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
                q6_qmicro=selected_q6_qmicro,
                profile=profile,
                repacked_cache=cache,
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
                    q6_qmicro=selected_q6_qmicro,
                    profile=profile,
                    repacked_cache=cache,
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
        q6_qmicro=selected_q6_qmicro,
    )


@dataclass(frozen=True)
class _ResidentPayload:
    array: Any
    dtype: DType
    source_dtype: str


def _resident_payloads(
    spec: LagunaGGUFWeightSpec,
    raw: Any,
) -> Mapping[str, _ResidentPayload]:
    if spec.layout == LAYOUT_DENSE_F16:
        payloads = {"raw": _ResidentPayload(raw, DType.FP16, "F16")}
    elif spec.layout == LAYOUT_DENSE_F32:
        payloads = {"raw": _ResidentPayload(raw, DType.FP32, "F32")}
    elif spec.layout == LAYOUT_RAW_GGUF:
        payloads = {"raw": _ResidentPayload(raw, DType.INT8, "I8")}
    elif spec.layout == LAYOUT_Q4_K_PACK8:
        packed = repack_gguf_q4_k_pack8(raw)
        payloads = {
            "qweight": _ResidentPayload(packed.qweight, DType.INT32, "I32"),
            "scales": _ResidentPayload(packed.scales, DType.FP32, "F32"),
            "mins": _ResidentPayload(packed.mins, DType.FP32, "F32"),
        }
    elif spec.layout == LAYOUT_GGUF_Q4_K_T16:
        packed = repack_gguf_q4_k_tile16(raw)
        payloads = {"tiles": _ResidentPayload(packed.tiles, DType.INT8, "I8")}
    elif spec.layout == LAYOUT_GGUF_Q6_K_T16:
        packed = repack_gguf_q6_k_tile16(raw if raw.ndim == 3 else raw[None, ...])
        payloads = {"tiles": _ResidentPayload(packed.tiles, DType.INT8, "I8")}
    else:
        raise ValueError(f"unsupported Laguna materialization layout {spec.layout!r}")
    if tuple(payloads) != spec.allocation_names:
        raise ValueError(
            f"Laguna payload names differ for {spec.slot_path}: "
            f"planned={spec.allocation_names} actual={tuple(payloads)}"
        )
    return MappingProxyType(payloads)


def _materialize_spec(
    spec: LagunaGGUFWeightSpec,
    reader: GGUFReader,
    *,
    device: Device | None,
    runtime: HipRuntime,
    backend: str,
    q6_qmicro: bool | None = None,
    profile: Callable[[LagunaGGUFMaterializationProfile], None] | None = None,
    repacked_cache: LagunaGGUFRepackedCache | None = None,
) -> LagunaGGUFDeviceWeight:
    import numpy as np

    total_started = time.perf_counter()
    total_before = _process_counters() if profile is not None else None
    source_started = time.perf_counter()
    source_before = _process_counters() if profile is not None else None
    cache_hit = repacked_cache is not None and repacked_cache.has(spec)
    source_kind = "repacked_cache" if cache_hit else "gguf"
    if cache_hit:
        assert repacked_cache is not None
        payloads = repacked_cache.payloads(spec)
        raw = None
    else:
        raw = np.ascontiguousarray(reader.tensor_data(spec.source.name))
        payloads = None
    source_map_seconds = time.perf_counter() - source_started
    source_after = _process_counters() if profile is not None else None
    source_delta = _counter_delta(source_before, source_after)
    repack_seconds = 0.0
    repack_delta = _ProcessCounterDelta()
    timed_runtime = _TimedUploadRuntime(runtime) if profile is not None else runtime
    selected_q6_qmicro = bool(
        backend_package_capability(backend, "LAGUNA_Q6_QMICRO", False)
        if q6_qmicro is None
        else q6_qmicro
    )

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

    if payloads is None:
        assert raw is not None
        if spec.layout in _REPACKED_CACHE_LAYOUTS:
            payloads = measured_repack(lambda: _resident_payloads(spec, raw))
        else:
            payloads = _resident_payloads(spec, raw)
    if (
        selected_q6_qmicro
        and spec.layout == LAYOUT_GGUF_Q6_K_T16
        and spec.slot_path.endswith(".ffn_down_exps")
    ):
        legacy_payload = payloads["tiles"]
        qmicro = measured_repack(
            lambda: convert_gguf_q6_k_tile16_to_qmicro(
                legacy_payload.array
            )
        )
        payloads = MappingProxyType(
            {
                "tiles": _ResidentPayload(
                    qmicro.tiles,
                    legacy_payload.dtype,
                    legacy_payload.source_dtype,
                )
            }
        )

    allocations: dict[str, DeviceTensorAllocation] = {}
    source_abs_max: float | None = None
    source_row_l2_max: float | None = None
    if spec.slot_path.endswith(".attn_norm"):
        source_values = np.asarray(payloads["raw"].array, dtype=np.float32)
        source_abs_max = (
            float(np.max(np.abs(source_values)))
            if bool(np.isfinite(source_values).all())
            else float("inf")
        )
    if spec.slot_path.endswith((".attn_v", ".attn_gate")):
        source_values = np.asarray(payloads["raw"].array, dtype=np.float64)
        if bool(np.isfinite(source_values).all()):
            source_row_l2_max = float(
                np.max(np.linalg.norm(source_values, axis=1))
            )
        else:
            source_row_l2_max = float("inf")
    try:
        for name, payload in payloads.items():
            allocations[name] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.{name}",
                payload.array,
                payload.dtype,
                source_dtype=payload.source_dtype,
                device=device,
                runtime=timed_runtime,
            )

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
        source_abs_max=source_abs_max,
        source_row_l2_max=source_row_l2_max,
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
            source_kind=source_kind,
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
    raw_quant_keys = {
        GGMLQuantizationType.Q5_K: "gguf_q5_k",
        GGMLQuantizationType.Q8_0: "gguf_q8_0",
        GGMLQuantizationType.IQ2_XS: "gguf_iq2_xs",
        GGMLQuantizationType.IQ3_XXS: "gguf_iq3_xxs",
        GGMLQuantizationType.IQ4_XS: "gguf_iq4_xs",
    }
    if qtype in raw_quant_keys:
        return _spec(
            slot_path,
            tensor,
            quant_key=raw_quant_keys[qtype],
            layout=LAYOUT_RAW_GGUF,
            resident_dtype=qtype.name.lower(),
            allocations={"raw": tensor.nbytes},
        )
    if qtype == GGMLQuantizationType.Q4_K:
        if slot_path in {"root.token_embedding", "root.lm_head"}:
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
    "LagunaGGUFRepackedCache",
    "LagunaGGUFResidentLayerWeights",
    "LagunaGGUFResidentWeights",
    "LagunaGGUFWeightSpec",
    "LagunaKVMemoryPlan",
    "LagunaMemoryAdmissionError",
    "LagunaMemoryAdmissionPlan",
    "build_laguna_repacked_cache",
    "materialize_laguna_gguf_weights",
    "open_laguna_repacked_cache",
    "plan_laguna_gguf_materialization",
    "plan_laguna_memory_admission",
]
