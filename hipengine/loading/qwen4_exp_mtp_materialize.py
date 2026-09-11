"""One-layout materialization for a Qwen4Exp MTP-only GGUF sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.loading.qwen4_exp_materialize import (
    LAYOUT_RAW_GGUF,
    Qwen4ExpDeviceWeight,
    Qwen4ExpGGUFWeightSpec,
    materialize_qwen4_exp_raw_weight,
)
from hipengine.loading.qwen4_exp_mtp_gguf import (
    Qwen4ExpMTPGGUFConfig,
    Qwen4ExpMTPGGUFMap,
)
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
)


@dataclass(frozen=True)
class Qwen4ExpMTPResidencyPlan:
    config: Qwen4ExpMTPGGUFConfig
    weight_specs: Mapping[str, Qwen4ExpGGUFWeightSpec]
    raw_payload_bytes: int
    device_weight_bytes: int
    alternate_layout_bytes: int = 0
    replacement_payload_bytes: int = 0

    @property
    def specs(self) -> tuple[Qwen4ExpGGUFWeightSpec, ...]:
        return tuple(self.weight_specs.values())


@dataclass
class Qwen4ExpMTPResidentWeights:
    plan: Qwen4ExpMTPResidencyPlan
    device_weights: Mapping[str, Qwen4ExpDeviceWeight]
    runtime: HipRuntime
    closed: bool = False

    def weight(self, slot_path: str) -> Qwen4ExpDeviceWeight:
        if self.closed:
            raise RuntimeError("Qwen4Exp MTP resident weights are closed")
        try:
            return self.device_weights[str(slot_path)]
        except KeyError as exc:
            raise KeyError(f"unknown Qwen4Exp MTP resident slot {slot_path!r}") from exc

    def close(self) -> None:
        if self.closed:
            return
        for weight in reversed(tuple(self.device_weights.values())):
            weight.free(runtime=self.runtime)
        self.device_weights = MappingProxyType({})
        self.closed = True


def _runtime_layout(type_name: str) -> tuple[str, str]:
    if type_name == "F32":
        return "f32", LAYOUT_DENSE_F32
    if type_name == "BF16":
        return "bf16", LAYOUT_DENSE_BF16
    return f"gguf_{type_name.lower()}", LAYOUT_RAW_GGUF


def plan_qwen4_exp_mtp_residency(
    model_map: Qwen4ExpMTPGGUFMap,
) -> Qwen4ExpMTPResidencyPlan:
    if not model_map.validation.passed:
        raise ValueError("Qwen4Exp MTP map must pass before residency planning")
    specs: dict[str, Qwen4ExpGGUFWeightSpec] = {}
    for slot, ref in model_map.weights.items():
        quant_key, layout = _runtime_layout(ref.tensor.ggml_type_name)
        specs[slot] = Qwen4ExpGGUFWeightSpec(
            slot_path=slot,
            source_ref=ref,
            quant_key=quant_key,
            layout=layout,
            allocation_names=("raw",),
            device_resident=True,
            device_nbytes=int(ref.tensor.nbytes),
        )
    payload = sum(spec.source.nbytes for spec in specs.values())
    return Qwen4ExpMTPResidencyPlan(
        config=model_map.config,
        weight_specs=MappingProxyType(specs),
        raw_payload_bytes=int(payload),
        device_weight_bytes=int(payload),
    )


def materialize_qwen4_exp_mtp_weights(
    readers: Sequence[Any],
    *,
    plan: Qwen4ExpMTPResidencyPlan,
    backend: str = "hip_gfx1151",
    runtime: HipRuntime | None = None,
    device_loader: Any | None = None,
) -> Qwen4ExpMTPResidentWeights:
    parts = tuple(readers)
    expected_parts = max(spec.source_ref.part_index for spec in plan.specs) + 1
    if len(parts) != expected_parts:
        raise ValueError(
            f"reader part count {len(parts)} does not match MTP plan {expected_parts}"
        )
    active_runtime = runtime or get_hip_runtime()
    load = device_loader or materialize_qwen4_exp_raw_weight
    weights: dict[str, Qwen4ExpDeviceWeight] = {}
    try:
        for spec in plan.specs:
            allocation = load(
                spec,
                parts[spec.source_ref.part_index],
                runtime=active_runtime,
            )
            weights[spec.slot_path] = Qwen4ExpDeviceWeight(
                spec=spec,
                backend=str(backend),
                allocations=MappingProxyType({"raw": allocation}),
            )
    except Exception:
        for weight in reversed(tuple(weights.values())):
            weight.free(runtime=active_runtime)
        raise
    return Qwen4ExpMTPResidentWeights(
        plan=plan,
        device_weights=MappingProxyType(weights),
        runtime=active_runtime,
    )


__all__ = [
    "Qwen4ExpMTPResidencyPlan",
    "Qwen4ExpMTPResidentWeights",
    "materialize_qwen4_exp_mtp_weights",
    "plan_qwen4_exp_mtp_residency",
]
