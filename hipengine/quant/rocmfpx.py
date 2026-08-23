"""CIRU ROCmFPX GGUF quant identities used by Kairic Edge.

The first authority adapter deliberately expands Q4_0_ROCMFP4 to lossless BF16
weights and Q6_0_ROCMFPX to lossless F32 weights at load time. Native compressed
execution remains a separate kernel campaign; this module only makes the source
formats explicit registry plugins instead of treating custom GGUF IDs as an
anonymous loader exception.
"""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.quant.registry import register_quant


@dataclass(frozen=True)
class GGUFROCmFP4Quant:
    name: str = "gguf_q4_0_rocmfp4"
    weight_storage: str = "gguf_block32_rocmfp4_codebook10_dual_ue4m3"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block32_half16_ue4m3"
    calibration_artifact: str = "gguf"
    kernel_family: str = "dense_bf16_authority_fallback"


@dataclass(frozen=True)
class GGUFROCmFP6Quant:
    name: str = "gguf_q6_0_rocmfpx"
    weight_storage: str = "gguf_block32_signed_magnitude6_dual_ue4m3"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block32_half16_ue4m3"
    calibration_artifact: str = "gguf"
    kernel_family: str = "dense_f32_authority_fallback"


GGUF_Q4_0_ROCMFP4 = register_quant(GGUFROCmFP4Quant())
GGUF_Q6_0_ROCMFPX = register_quant(GGUFROCmFP6Quant())


__all__ = [
    "GGUFROCmFP4Quant",
    "GGUFROCmFP6Quant",
    "GGUF_Q4_0_ROCMFP4",
    "GGUF_Q6_0_ROCMFPX",
]
