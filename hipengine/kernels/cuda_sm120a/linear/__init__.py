"""CUDA ``sm_120a`` Moonshine dense projection kernels."""

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
    "build_moonshine_projection",
    "moonshine_f16_lm_head_projection",
    "moonshine_f16_lm_head_projection_wave8",
    "moonshine_f16_projection",
    "moonshine_f16_projection_bias",
    "moonshine_f16_projection_bias_gated_silu",
    "moonshine_f16_projection_bias_residual",
    "moonshine_f16_projection_pair",
    "moonshine_f16_projection_pair_head_major",
    "moonshine_f16_projection_triple",
    "plan_moonshine_projection_build",
    "register_moonshine_projection_kernels",
]
