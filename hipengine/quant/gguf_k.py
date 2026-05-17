"""GGUF K-family/simple quant plugin metadata for native/fallback bring-up."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.quant.registry import register_quant


@dataclass(frozen=True)
class GGUFQ80Quant:
    """GGUF block_q8_0 weight-only quantization contract."""

    name: str = "gguf_q8_0"
    weight_storage: str = "gguf_block_q8_0"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block32_scale"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_k_gemv"


@dataclass(frozen=True)
class GGUFQ41Quant:
    """GGUF block_q4_1 weight-only quantization contract."""

    name: str = "gguf_q4_1"
    weight_storage: str = "gguf_block_q4_1"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block32_scale_min"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_dense_bf16_fallback"


@dataclass(frozen=True)
class GGUFQ5KQuant:
    """GGUF block_q5_K weight-only quantization contract."""

    name: str = "gguf_q5_k"
    weight_storage: str = "gguf_block_q5_k"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock32_scale_min"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_k_gemv"


@dataclass(frozen=True)
class GGUFIQ4XSQuant:
    """GGUF iq4_xs weight-only quantization contract."""

    name: str = "gguf_iq4_xs"
    weight_storage: str = "gguf_iq4_xs"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_iq4_xs"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_dense_bf16_fallback"


@dataclass(frozen=True)
class GGUFQ6KQuant:
    """GGUF block_q6_K weight-only quantization contract."""

    name: str = "gguf_q6_k"
    weight_storage: str = "gguf_block_q6_k"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock16_scale"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_k_gemv"


GGUF_Q8_0 = register_quant(GGUFQ80Quant())
GGUF_Q4_1 = register_quant(GGUFQ41Quant())
GGUF_Q5_K = register_quant(GGUFQ5KQuant())
GGUF_Q6_K = register_quant(GGUFQ6KQuant())
GGUF_IQ4_XS = register_quant(GGUFIQ4XSQuant())


__all__ = [
    "GGUFQ80Quant",
    "GGUFQ41Quant",
    "GGUFQ5KQuant",
    "GGUFQ6KQuant",
    "GGUFIQ4XSQuant",
    "GGUF_IQ4_XS",
    "GGUF_Q4_1",
    "GGUF_Q5_K",
    "GGUF_Q6_K",
    "GGUF_Q8_0",
]
