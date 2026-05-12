"""FP16 quant plugin scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.quant.registry import register_quant


@dataclass(frozen=True)
class FP16Quant:
    name: str = "fp16"
    weight_storage: str = "fp16"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp16"
    scale_granularity: str = "none"
    calibration_artifact: str = "none"
    kernel_family: str = "fp16"


FP16 = register_quant(FP16Quant())
