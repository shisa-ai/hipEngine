"""BF16 unquantized compute plugin."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.quant.registry import register_quant


@dataclass(frozen=True)
class BF16Quant:
    name: str = "bf16"
    weight_storage: str = "bf16"
    activation_preprocess: str = "none"
    compute_dtype: str = "bf16"
    scale_granularity: str = "none"
    calibration_artifact: str = "none"
    kernel_family: str = "bf16"


BF16 = register_quant(BF16Quant())
