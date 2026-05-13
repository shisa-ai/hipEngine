"""PARO W4A16 quantization plugin."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.quant.registry import register_quant


@dataclass(frozen=True)
class W4ParoQuant:
    """PARO/AWQ 4-bit weight, BF16 activation quantization contract.

    This records the six orthogonal quant axes used by the registry. Weight tensor loading
    and layout materialization live in the loader layer; this plugin is only metadata for
    dispatch/planning.
    """

    name: str = "w4_paro"
    weight_storage: str = "uint4_pack8_awq"
    activation_preprocess: str = "bf16_pairwise_rotation"
    compute_dtype: str = "bf16"
    scale_granularity: str = "group128_per_output_channel"
    calibration_artifact: str = "paroquant_theta_pairs_scales"
    kernel_family: str = "paro_awq_pack8"


W4_PARO = register_quant(W4ParoQuant())
