"""One-layout Qwen4Exp BF16/F32 vision weight materialization."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_DENSE_BF16, LAYOUT_DENSE_F32
from hipengine.loading.qwen4_exp_materialize import (
    Qwen4ExpDeviceWeight,
    Qwen4ExpGGUFWeightSpec,
    materialize_qwen4_exp_raw_weight,
)
from hipengine.loading.qwen4_exp_vision_gguf import Qwen4ExpVisionConfig, Qwen4ExpVisionMap


@dataclass(frozen=True)
class Qwen4ExpVisionResidencyPlan:
    config: Qwen4ExpVisionConfig
    weight_specs: Mapping[str, Qwen4ExpGGUFWeightSpec]
    device_weight_bytes: int

    @property
    def specs(self):
        return tuple(self.weight_specs.values())


@dataclass
class Qwen4ExpVisionResidentWeights:
    plan: Qwen4ExpVisionResidencyPlan
    device_weights: Mapping[str, Qwen4ExpDeviceWeight]
    runtime: HipRuntime
    closed: bool = False

    def weight(self, slot: str) -> Qwen4ExpDeviceWeight:
        if self.closed:
            raise RuntimeError("Qwen4Exp vision weights are closed")
        return self.device_weights[str(slot)]

    def close(self) -> None:
        if self.closed:
            return
        for weight in reversed(tuple(self.device_weights.values())):
            weight.free(runtime=self.runtime)
        self.device_weights = MappingProxyType({})
        self.closed = True


def plan_qwen4_exp_vision_residency(model_map: Qwen4ExpVisionMap) -> Qwen4ExpVisionResidencyPlan:
    if not model_map.validation.passed:
        raise ValueError("Qwen4Exp vision map must pass before materialization")
    specs = {}
    for slot, ref in model_map.weights.items():
        layout = LAYOUT_DENSE_BF16 if ref.tensor.ggml_type_name == "BF16" else LAYOUT_DENSE_F32
        quant = "bf16" if ref.tensor.ggml_type_name == "BF16" else "f32"
        specs[slot] = Qwen4ExpGGUFWeightSpec(
            slot_path=slot,
            source_ref=ref,
            quant_key=quant,
            layout=layout,
            allocation_names=("raw",),
            device_resident=True,
            device_nbytes=int(ref.tensor.nbytes),
        )
    total = sum(spec.device_nbytes for spec in specs.values())
    return Qwen4ExpVisionResidencyPlan(
        config=model_map.config,
        weight_specs=MappingProxyType(specs),
        device_weight_bytes=total,
    )


def materialize_qwen4_exp_vision_weights(
    readers: Sequence[Any],
    *,
    plan: Qwen4ExpVisionResidencyPlan,
    backend: str = "hip_gfx1151",
    runtime: HipRuntime | None = None,
) -> Qwen4ExpVisionResidentWeights:
    parts = tuple(readers)
    active = runtime or get_hip_runtime()
    weights = {}
    try:
        for spec in plan.specs:
            allocation = materialize_qwen4_exp_raw_weight(
                spec, parts[spec.source_ref.part_index], runtime=active
            )
            weights[spec.slot_path] = Qwen4ExpDeviceWeight(
                spec=spec,
                backend=str(backend),
                allocations=MappingProxyType({"raw": allocation}),
            )
    except Exception:
        for weight in reversed(tuple(weights.values())):
            weight.free(runtime=active)
        raise
    return Qwen4ExpVisionResidentWeights(
        plan=plan,
        device_weights=MappingProxyType(weights),
        runtime=active,
    )


__all__ = [
    "Qwen4ExpVisionResidencyPlan",
    "Qwen4ExpVisionResidentWeights",
    "materialize_qwen4_exp_vision_weights",
    "plan_qwen4_exp_vision_residency",
]
