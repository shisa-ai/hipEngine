"""GGUF Q4_K quantization plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.quant.registry import register_quant


@dataclass(frozen=True)
class GGUFQ4KQuant:
    """GGUF block_q4_K weight-only quantization contract.

    The GGUF tensor layout is block-256 with eight 32-value subblocks.  Each
    block carries fp16 ``d``/``dmin`` plus packed 6-bit scale/min metadata; the
    HIP kernels preserve that math instead of translating to PARO/AWQ zeros.
    """

    name: str = "gguf_q4_k"
    weight_storage: str = "gguf_block_q4_k"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock32_scale_min"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_q4_k_gemv"


GGUF_Q4_K = register_quant(GGUFQ4KQuant())
