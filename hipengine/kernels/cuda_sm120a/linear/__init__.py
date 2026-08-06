"""CUDA ``sm_120a`` Moonshine dense projection and LM-head kernels."""

from hipengine.kernels.cuda_sm120a.linear.lm_head import (
    build_moonshine_lm_head,
    lm_head_argmax_scratch_elements,
    moonshine_lm_head_argmax_fp16,
    plan_moonshine_lm_head_build,
    register_moonshine_lm_head_kernels,
)
from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
    build_moonshine_projection,
    moonshine_f16_lm_head_projection,
    moonshine_f16_lm_head_projection_wave8,
    moonshine_f16_projection,
    moonshine_f16_projection_bias,
    moonshine_f16_projection_bias_gated_silu,
    moonshine_f16_projection_bias_residual,
    moonshine_f16_projection_pair,
    moonshine_f16_projection_pair_head_major,
    moonshine_f16_projection_triple,
    plan_moonshine_projection_build,
    register_moonshine_projection_kernels,
)

__all__ = [
    "build_moonshine_lm_head",
    "build_moonshine_projection",
    "lm_head_argmax_scratch_elements",
    "moonshine_f16_lm_head_projection",
    "moonshine_f16_lm_head_projection_wave8",
    "moonshine_f16_projection",
    "moonshine_f16_projection_bias",
    "moonshine_f16_projection_bias_gated_silu",
    "moonshine_f16_projection_bias_residual",
    "moonshine_f16_projection_pair",
    "moonshine_f16_projection_pair_head_major",
    "moonshine_f16_projection_triple",
    "moonshine_lm_head_argmax_fp16",
    "plan_moonshine_lm_head_build",
    "plan_moonshine_projection_build",
    "register_moonshine_lm_head_kernels",
    "register_moonshine_projection_kernels",
]
