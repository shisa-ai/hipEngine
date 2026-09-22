"""gfx1100 Gemma 4 kernel wrappers."""

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    build_gemma4_norm,
    gemma4_add_rmsnorm_scale_bf16,
    gemma4_branch_add_bf16,
    gemma4_expert_weight_scale_f32,
    gemma4_head_rmsnorm_f32w_bf16,
    gemma4_rmsnorm_f32w_bf16,
    gemma4_rmsnorm_f32w_f32,
    gemma4_rmsnorm_weightless_bf16,
    gemma4_router_prescale_bf16,
    plan_gemma4_norm_build,
    register_gemma4_norm_kernels,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rope import (
    GEMMA4_ROPE_DEFAULT_TYPE,
    GEMMA4_ROPE_PROPORTIONAL_TYPE,
    gemma4_rope_angles,
    gemma4_rope_cos_sin_tables,
    gemma4_rope_inverse_frequencies,
    gemma4_rotate_split_half,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import (
    build_gemma4_rotary,
    gemma4_k_to_v_bf16,
    gemma4_partial_rotary_bf16,
    gemma4_partial_rotary_f32,
    plan_gemma4_rotary_build,
    register_gemma4_rotary_kernels,
)

__all__ = [
    "GEMMA4_ROPE_DEFAULT_TYPE",
    "GEMMA4_ROPE_PROPORTIONAL_TYPE",
    "build_gemma4_norm",
    "build_gemma4_rotary",
    "gemma4_add_rmsnorm_scale_bf16",
    "gemma4_branch_add_bf16",
    "gemma4_expert_weight_scale_f32",
    "gemma4_head_rmsnorm_f32w_bf16",
    "gemma4_k_to_v_bf16",
    "gemma4_partial_rotary_bf16",
    "gemma4_partial_rotary_f32",
    "gemma4_rmsnorm_f32w_bf16",
    "gemma4_rmsnorm_f32w_f32",
    "gemma4_rmsnorm_weightless_bf16",
    "gemma4_rope_angles",
    "gemma4_rope_cos_sin_tables",
    "gemma4_rope_inverse_frequencies",
    "gemma4_rotate_split_half",
    "gemma4_router_prescale_bf16",
    "plan_gemma4_norm_build",
    "plan_gemma4_rotary_build",
    "register_gemma4_norm_kernels",
    "register_gemma4_rotary_kernels",
]
