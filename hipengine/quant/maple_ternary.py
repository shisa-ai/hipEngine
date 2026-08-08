"""Maple 2-bit ternary (MLX checkpoint) quant plugin."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.quant.registry import register_quant


@dataclass(frozen=True)
class MapleTernary2Quant:
    """Maple ternary weight format from the official MLX 2-bit checkpoint.

    Projections are uint32-packed 2-bit codes (16 per word, LSB first, value = code-1)
    with one bf16 row scale (``row_alpha``) per output row. Embeddings and the lm_head
    use MLX affine 4-bit (group 64). The router stays dense bf16 and computes in fp32.
    """

    name: str = "maple_ternary2"
    weight_storage: str = "u32_packed_2bit_row_alpha"
    activation_preprocess: str = "none"
    compute_dtype: str = "bf16"
    scale_granularity: str = "per_row"
    calibration_artifact: str = "quantization_aware_training"
    kernel_family: str = "maple_ternary"


MAPLE_TERNARY2 = register_quant(MapleTernary2Quant())
