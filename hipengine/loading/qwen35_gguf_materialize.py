"""Device materialization for Qwen3.5 GGUF tensor maps."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    float_array_to_bf16_bits,
    load_host_array_to_device_as_dtype,
)
from hipengine.loading.qwen35_gguf import (
    Qwen35GGUFConfig,
    Qwen35GGUFLayerMap,
    Qwen35GGUFModelMap,
    build_qwen35_gguf_tensor_map,
)
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8, repack_gguf_q4_k_tile16
from hipengine.quant.gguf_t16 import (
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
LAYOUT_GGUF_Q6_K_T16 = "gguf_q6_k_t16_v1"
LAYOUT_GGUF_Q8_0_T16 = "gguf_q8_0_t16_v1"
LAYOUT_GGUF_Q5_K_X8 = "gguf_q5_k_x8_v1"
LAYOUT_GGUF_Q6_K_X8 = "gguf_q6_k_x8_v1"
HIPENGINE_GGUF_DECODE_REPACK_ENV = "HIPENGINE_GGUF_DECODE_REPACK"
HIPENGINE_GGUF_SELECTED_X8_REPACK_ENV = "HIPENGINE_GGUF_SELECTED_X8_REPACK"
HIPENGINE_GGUF_SELECTED_DOWN_RAW_ENV = "HIPENGINE_GGUF_SELECTED_DOWN_RAW"
HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV = "HIPENGINE_GGUF_SELECTED_GATE_UP_X8"
HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV = "HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"


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
class Qwen35GGUFDeviceWeight:
    """Owned device allocations for one logical GGUF weight."""

    spec: Qwen35GGUFWeightSpec
    allocations: Mapping[str, DeviceTensorAllocation]

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

    The map owns all device buffers. ``root_weights['lm_head']`` aliases
    ``root_weights['token_embedding']`` for the local tied-output GGUF.
    """

    config: Qwen35GGUFConfig
    root_weights: Mapping[str, Qwen35GGUFDeviceWeight]
    layers: tuple[Qwen35GGUFResidentLayerWeights, ...]

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


def plan_qwen35_gguf_materialization(
    model_map: Qwen35GGUFModelMap,
    *,
    decode_repack: bool | None = None,
) -> Qwen35GGUFMaterializationPlan:
    use_decode_repack = gguf_decode_repack_enabled(decode_repack)
    root_specs = {
        slot: _spec_for_tensor(f"root.{slot}", tensor, decode_repack=use_decode_repack)
        for slot, tensor in model_map.root_tensors.items()
    }
    layer_specs = tuple(_plan_layer(layer, decode_repack=use_decode_repack) for layer in model_map.layers)
    return Qwen35GGUFMaterializationPlan(
        config=model_map.config,
        root_specs=MappingProxyType(root_specs),
        layer_specs=tuple(MappingProxyType(layer) for layer in layer_specs),
    )


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
    decode_repack: bool | None = None,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> Qwen35GGUFResidentWeights:
    """Materialize a validated Qwen3.5 GGUF map to resident device records.

    ``selected_slots`` is a test/debug hook using slot paths such as
    ``root.output_norm`` or ``layers.0.attn_qkv``. Production callers leave it
    unset to materialize the full model.
    """

    reader = reader_or_path if isinstance(reader_or_path, GGUFReader) else GGUFReader(reader_or_path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=decode_repack)
    selected = None if selected_slots is None else set(selected_slots)
    materialized: dict[tuple[str, str], Qwen35GGUFDeviceWeight] = {}
    try:
        root_weights = {
            slot: _materialize_or_alias(spec, reader, materialized, selected, device=device, runtime=runtime)
            for slot, spec in plan.root_specs.items()
            if selected is None or spec.slot_path in selected
        }
        layers = tuple(
            Qwen35GGUFResidentLayerWeights(
                layer_id=layer.layer_id,
                layer_type=layer.layer_type,
                weights=MappingProxyType(
                    {
                        slot: _materialize_or_alias(
                            plan.layer_specs[layer.layer_id][slot],
                            reader,
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
    return Qwen35GGUFResidentWeights(
        config=plan.config,
        root_weights=MappingProxyType(root_weights),
        layers=layers,
    )


def _plan_layer(layer: Qwen35GGUFLayerMap, *, decode_repack: bool) -> dict[str, Qwen35GGUFWeightSpec]:
    return {
        slot: _spec_for_tensor(f"layers.{layer.layer_id}.{slot}", tensor, decode_repack=decode_repack)
        for slot, tensor in layer.tensors.items()
    }


def gguf_decode_repack_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    raw = os.environ.get(HIPENGINE_GGUF_DECODE_REPACK_ENV, "")
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

    This is a default-off llama-compat diagnostic sidecar. The current consumer is
    the dense-Q8 q8_1/dp4a verifier probe, which only needs linear-attention
    ``attn_qkv`` and ``attn_gate`` raw bytes.
    """

    if value is None:
        raw = os.environ.get(HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV, "")
    elif isinstance(value, bool):
        raw = "1" if value else ""
    else:
        raw = str(value)
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def _spec_for_tensor(slot_path: str, tensor: GGUFTensorInfo, *, decode_repack: bool) -> Qwen35GGUFWeightSpec:
    qtype = GGMLQuantizationType(tensor.ggml_type)
    if qtype == GGMLQuantizationType.F32:
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="f32",
            layout=LAYOUT_DENSE_F32,
            allocation_names=("raw",),
        )
    if qtype == GGMLQuantizationType.Q4_K:
        if len(tensor.shape) != 2:
            if decode_repack and _is_selected_expert_tensor(slot_path, tensor):
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
                quant_key="gguf_q5_k_t16_v1",
                layout=LAYOUT_GGUF_Q5_K_T16,
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
        return Qwen35GGUFWeightSpec(
            slot_path=slot_path,
            source=tensor,
            quant_key="gguf_q6_k_t16_v1",
            layout=LAYOUT_GGUF_Q6_K_T16,
            allocation_names=("tiles",),
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
        if gguf_q8_0_raw_sidecar_enabled() and _is_linear_attention_q8_pair_tensor(slot_path, tensor):
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
        GGMLQuantizationType.Q4_1,
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
) -> Qwen35GGUFDeviceWeight:
    del selected  # selection is handled by callers before materialization.
    key = (spec.source.name, spec.layout)
    weight = materialized.get(key)
    if weight is None:
        weight = _materialize_spec(spec, reader, device=device, runtime=runtime)
        materialized[key] = weight
    return weight


def _materialize_spec(
    spec: Qwen35GGUFWeightSpec,
    reader: GGUFReader,
    *,
    device: Device | None,
    runtime: HipRuntime | None,
) -> Qwen35GGUFDeviceWeight:
    import numpy as np

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
    return Qwen35GGUFDeviceWeight(spec=spec, allocations=MappingProxyType(allocations))


__all__ = [
    "LAYOUT_DENSE_BF16",
    "LAYOUT_DENSE_F32",
    "HIPENGINE_GGUF_DECODE_REPACK_ENV",
    "HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV",
    "HIPENGINE_GGUF_SELECTED_DOWN_RAW_ENV",
    "HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV",
    "HIPENGINE_GGUF_SELECTED_X8_REPACK_ENV",
    "LAYOUT_GGUF_EXPERT_PACK8_SIDECAR",
    "LAYOUT_GGUF_Q4_K_T16",
    "LAYOUT_GGUF_Q4_K_X8",
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
    "Qwen35GGUFResidentLayerWeights",
    "Qwen35GGUFResidentWeights",
    "Qwen35GGUFWeightSpec",
    "_gguf_ssm_a_to_kernel_a_log",
    "audit_qwen35_gguf_precision_contractions",
    "gguf_decode_repack_enabled",
    "gguf_q8_0_raw_sidecar_enabled",
    "gguf_selected_down_raw_enabled",
    "gguf_selected_down_raw_mode",
    "gguf_selected_gate_up_x8_enabled",
    "gguf_selected_x8_repack_enabled",
    "gguf_selected_x8_repack_mode",
    "materialize_qwen35_gguf_weights",
    "plan_qwen35_gguf_materialization",
]
