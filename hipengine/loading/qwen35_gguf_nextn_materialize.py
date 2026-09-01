"""Device materialization for the separate trailing GGUF NextN draft block."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.policy import GGUFModelGeometry
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.gguf_mtp_hot_vocab import (
    GGUFHotVocabSelection,
    default_gguf_hot_vocab_path,
    load_gguf_hot_vocab_selection,
)
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    load_host_array_to_device_as_dtype,
)
from hipengine.loading.qwen35_gguf import FULL_ATTENTION
from hipengine.loading.qwen35_gguf_materialize import (
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFResidentLayerWeights,
    Qwen35GGUFResidentWeights,
    Qwen35GGUFWeightSpec,
    materialize_qwen35_gguf_weight_spec,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
from hipengine.loading.qwen35_gguf_nextn import (
    Qwen35GGUFNextNMap,
    build_qwen35_gguf_nextn_tensor_map,
)


@dataclass(frozen=True)
class Qwen35GGUFNextNMaterializationPlan:
    """Resident layout plan for one draft block and its target fallbacks."""

    model_map: Qwen35GGUFNextNMap
    layer_specs: Mapping[str, Qwen35GGUFWeightSpec]
    nextn_specs: Mapping[str, Qwen35GGUFWeightSpec]
    fallback_specs: Mapping[str, Qwen35GGUFWeightSpec]

    @property
    def draft_specs(self) -> tuple[Qwen35GGUFWeightSpec, ...]:
        return tuple((*self.layer_specs.values(), *self.nextn_specs.values()))

    @property
    def specs(self) -> tuple[Qwen35GGUFWeightSpec, ...]:
        specs: list[Qwen35GGUFWeightSpec] = []
        seen: set[tuple[str, str]] = set()
        for spec in (*self.draft_specs, *self.fallback_specs.values()):
            key = (spec.source.name, spec.layout)
            if key not in seen:
                seen.add(key)
                specs.append(spec)
        return tuple(specs)


@dataclass(frozen=True)
class Qwen35GGUFNextNHotVocab:
    """Owned compact proposal head and compact-to-full token map."""

    selection: GGUFHotVocabSelection
    lm_head: Qwen35GGUFDeviceWeight
    token_ids: DeviceTensorAllocation

    @property
    def size(self) -> int:
        return self.selection.size

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.token_ids.free(runtime=runtime)
        self.lm_head.free(runtime=runtime)


@dataclass(frozen=True)
class Qwen35GGUFNextNResidentWeights:
    """Owned draft weights plus target embedding/output fallback records."""

    plan: Qwen35GGUFNextNMaterializationPlan
    layer_weights: Mapping[str, Qwen35GGUFDeviceWeight]
    nextn_weights: Mapping[str, Qwen35GGUFDeviceWeight]
    fallback_weights: Mapping[str, Qwen35GGUFDeviceWeight]
    owned_weights: tuple[Qwen35GGUFDeviceWeight, ...]
    backend: str
    hot_vocab: Qwen35GGUFNextNHotVocab | None = None

    @property
    def config(self):
        return self.plan.model_map.config

    @property
    def block_id(self) -> int:
        return int(self.plan.model_map.block_id)

    def layer(self, slot: str) -> Qwen35GGUFDeviceWeight:
        return self.layer_weights[slot]

    def nextn(self, slot: str) -> Qwen35GGUFDeviceWeight:
        return self.nextn_weights[slot]

    def fallback(self, slot: str) -> Qwen35GGUFDeviceWeight:
        return self.fallback_weights[slot]

    def as_full_stack_weights(self) -> Qwen35GGUFResidentWeights:
        """Adapt only blk.N to the existing one-layer full-attention executor."""

        draft_config = replace(
            self.config,
            block_count=1,
            declared_block_count=1,
            ignored_block_ids=(),
            layer_types=(FULL_ATTENTION,),
        )
        root_weights = MappingProxyType(
            {
                "token_embedding": self.fallback("token_embedding"),
                "output_norm": self.fallback("output_norm"),
                "lm_head": self.fallback("lm_head"),
            }
        )
        layer = Qwen35GGUFResidentLayerWeights(
            layer_id=0,
            layer_type=FULL_ATTENTION,
            weights=MappingProxyType(dict(self.layer_weights)),
        )
        return Qwen35GGUFResidentWeights(
            config=draft_config,
            root_weights=root_weights,
            layers=(layer,),
            backend=self.backend,
            geometry=GGUFModelGeometry.from_config(draft_config),
        )

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self.hot_vocab is not None:
            self.hot_vocab.free(runtime=runtime)
        for weight in reversed(self.owned_weights):
            weight.free(runtime=runtime)


def _materialize_hot_vocab(
    reader: GGUFReader,
    spec: Qwen35GGUFWeightSpec,
    selection: GGUFHotVocabSelection,
    *,
    device: Device | None,
    runtime: HipRuntime | None,
    backend: str,
) -> Qwen35GGUFNextNHotVocab:
    """Pack individually selected raw Q6 rows into a compact planar-T16 head."""

    import numpy as np

    if GGMLQuantizationType(spec.source.ggml_type) != GGMLQuantizationType.Q6_K:
        raise ValueError("GGUF MTP hot vocabulary requires a Q6_K output head")
    if spec.quant_key != "gguf_q6_k_t16_qmicro_planar_v1":
        raise ValueError("GGUF MTP hot vocabulary requires the planar Q6 T16 output layout")
    raw = np.asarray(reader.tensor_data(spec.source.name))
    if raw.ndim != 2 or int(raw.shape[0]) != len(
        reader.info.metadata["tokenizer.ggml.tokens"]
    ):
        raise ValueError("GGUF MTP hot-vocabulary output head has an unexpected shape")
    selected_raw = np.ascontiguousarray(
        raw[np.asarray(selection.token_ids, dtype=np.int64)],
        dtype=np.uint8,
    )
    packed = repack_gguf_q6_k_tile16_qmicro_planar(selected_raw[None, ...])
    head_allocation: DeviceTensorAllocation | None = None
    token_allocation: DeviceTensorAllocation | None = None
    try:
        head_allocation = load_host_array_to_device_as_dtype(
            f"{spec.source.name}.mtp_hot_vocab{selection.size}.tiles",
            packed.tiles,
            DType.INT8,
            source_dtype="I8",
            device=device,
            runtime=runtime,
        )
        token_allocation = load_host_array_to_device_as_dtype(
            f"{spec.source.name}.mtp_hot_vocab{selection.size}.token_ids",
            np.asarray(selection.token_ids, dtype=np.int32),
            DType.INT32,
            source_dtype="I32",
            device=device,
            runtime=runtime,
        )
    except Exception:
        if token_allocation is not None:
            token_allocation.free(runtime=runtime)
        if head_allocation is not None:
            head_allocation.free(runtime=runtime)
        raise
    hot_spec = replace(spec, slot_path="draft.hot_lm_head")
    return Qwen35GGUFNextNHotVocab(
        selection=selection,
        lm_head=Qwen35GGUFDeviceWeight(
            spec=hot_spec,
            allocations=MappingProxyType({"tiles": head_allocation}),
            backend=str(backend),
        ),
        token_ids=token_allocation,
    )


def plan_qwen35_gguf_nextn_materialization(
    model_map: Qwen35GGUFNextNMap,
    *,
    decode_repack: bool = False,
    dense_q4_t16: bool = False,
    dense_q5_t16_ssm_out: bool = False,
    dense_q5_t16_h5120: bool = False,
    dense_q6_qmicro_planar: bool = False,
) -> Qwen35GGUFNextNMaterializationPlan:
    """Plan blk.N independently from the unchanged AR weight plan."""

    plan_kwargs = {
        "decode_repack": bool(decode_repack),
        "dense_q4_t16": bool(dense_q4_t16),
        "dense_q5_t16_ssm_out": bool(dense_q5_t16_ssm_out),
        "dense_q5_t16_h5120": bool(dense_q5_t16_h5120),
        "dense_q6_qmicro_planar": bool(dense_q6_qmicro_planar),
    }
    layer_specs = {
        slot: plan_qwen35_gguf_weight_spec(
            f"draft.layer.{slot}", tensor, **plan_kwargs
        )
        for slot, tensor in model_map.layer_tensors.items()
    }
    nextn_specs = {
        slot: plan_qwen35_gguf_weight_spec(
            f"draft.nextn.{slot}", tensor, **plan_kwargs
        )
        for slot, tensor in model_map.nextn_tensors.items()
    }
    fallback_specs = {
        slot: plan_qwen35_gguf_weight_spec(f"root.{slot}", tensor, **plan_kwargs)
        for slot, tensor in model_map.fallback_tensors.items()
    }
    return Qwen35GGUFNextNMaterializationPlan(
        model_map=model_map,
        layer_specs=MappingProxyType(layer_specs),
        nextn_specs=MappingProxyType(nextn_specs),
        fallback_specs=MappingProxyType(fallback_specs),
    )


def materialize_qwen35_gguf_nextn_weights(
    reader_or_path: GGUFReader | str | Path,
    *,
    borrowed_fallback_weights: Mapping[str, Qwen35GGUFDeviceWeight] | None = None,
    hot_vocab_path: str | Path | None = None,
    decode_repack: bool = True,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    backend: str = "hip_gfx1100",
) -> Qwen35GGUFNextNResidentWeights:
    """Materialize the draft block without adding it to the 40-layer AR map.

    When target root weights are supplied, the draft resident borrows those
    records and owns only its layer/NextN allocations. The caller must keep the
    target resident alive until this draft resident is closed.
    """

    reader = reader_or_path if isinstance(reader_or_path, GGUFReader) else GGUFReader(reader_or_path)
    model_map = build_qwen35_gguf_nextn_tensor_map(reader.info)
    plan = plan_qwen35_gguf_nextn_materialization(
        model_map,
        decode_repack=decode_repack,
        dense_q4_t16=bool(
            backend_package_capability(backend, "GGUF_DENSE_Q4_T16", False)
        ),
        dense_q5_t16_ssm_out=bool(
            backend_package_capability(
                backend, "GGUF_DENSE_Q5_T16_SSM_OUT", False
            )
        ),
        dense_q5_t16_h5120=bool(
            backend_package_capability(
                backend, "GGUF_DENSE_Q5_T16_H5120", False
            )
        ),
        dense_q6_qmicro_planar=bool(
            backend_package_capability(
                backend, "GGUF_DENSE_Q6_T16_QMICRO_PLANAR", False
            )
        ),
    )
    borrowed: dict[str, Qwen35GGUFDeviceWeight] | None = None
    if borrowed_fallback_weights is not None:
        borrowed = dict(borrowed_fallback_weights)
        expected_slots = set(plan.fallback_specs)
        extra = sorted(set(borrowed) - expected_slots)
        if extra:
            raise ValueError(f"borrowed fallback slots are not in the NextN map: extra={extra}")
        borrowed_specs = dict(plan.fallback_specs)
        for slot, weight in borrowed.items():
            expected_spec = plan.fallback_specs[slot]
            borrowed_spec = weight.spec
            expected_source = expected_spec.source
            borrowed_source = borrowed_spec.source
            if (
                borrowed_source.name != expected_source.name
                or tuple(borrowed_source.shape) != tuple(expected_source.shape)
                or int(borrowed_source.ggml_type) != int(expected_source.ggml_type)
            ):
                raise ValueError(f"borrowed fallback {slot!r} does not match the NextN source tensor")
            if str(weight.backend) != str(backend):
                raise ValueError(
                    f"borrowed fallback {slot!r} uses backend {weight.backend!r}, expected {backend!r}"
                )
            borrowed_specs[slot] = borrowed_spec
        plan = replace(plan, fallback_specs=MappingProxyType(borrowed_specs))

    materialized: dict[tuple[str, str], Qwen35GGUFDeviceWeight] = {}

    def load(spec: Qwen35GGUFWeightSpec) -> Qwen35GGUFDeviceWeight:
        key = (spec.source.name, spec.layout)
        weight = materialized.get(key)
        if weight is None:
            weight = materialize_qwen35_gguf_weight_spec(
                spec,
                reader,
                device=device,
                runtime=runtime,
                backend=backend,
            )
            materialized[key] = weight
        return weight

    try:
        layer_weights = {slot: load(spec) for slot, spec in plan.layer_specs.items()}
        nextn_weights = {slot: load(spec) for slot, spec in plan.nextn_specs.items()}
        fallback_weights = {
            slot: borrowed[slot] if borrowed is not None and slot in borrowed else load(spec)
            for slot, spec in plan.fallback_specs.items()
        }
        resolved_hot_vocab_path = (
            default_gguf_hot_vocab_path(reader.info)
            if hot_vocab_path == "auto"
            else hot_vocab_path
        )
        hot_vocab = (
            _materialize_hot_vocab(
                reader,
                plan.fallback_specs["lm_head"],
                load_gguf_hot_vocab_selection(resolved_hot_vocab_path, reader.info),
                device=device,
                runtime=runtime,
                backend=str(backend),
            )
            if resolved_hot_vocab_path is not None
            else None
        )
    except Exception:
        for weight in reversed(tuple(materialized.values())):
            weight.free(runtime=runtime)
        raise
    return Qwen35GGUFNextNResidentWeights(
        plan=plan,
        layer_weights=MappingProxyType(layer_weights),
        nextn_weights=MappingProxyType(nextn_weights),
        fallback_weights=MappingProxyType(fallback_weights),
        owned_weights=tuple(materialized.values()),
        backend=str(backend),
        hot_vocab=hot_vocab,
    )


__all__ = [
    "Qwen35GGUFNextNHotVocab",
    "Qwen35GGUFNextNMaterializationPlan",
    "Qwen35GGUFNextNResidentWeights",
    "materialize_qwen35_gguf_nextn_weights",
    "plan_qwen35_gguf_nextn_materialization",
]
