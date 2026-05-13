"""gfx1100 fused kernel wrappers."""

from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    build_paro_combine,
    plan_paro_combine_build,
    register_paro_combine_kernels,
    shared_gate_combine_out_bf16,
    shared_gate_combine_residual_out_bf16,
    weighted_sum_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    plan_paro_silu_build,
    register_paro_silu_kernels,
    silu_mul_dual_out_bf16,
    silu_mul_dual_rotate_out_bf16,
    silu_mul_pair_rotate_out_bf16,
)

__all__ = [
    "build_paro_combine",
    "build_paro_silu",
    "plan_paro_combine_build",
    "plan_paro_silu_build",
    "register_paro_combine_kernels",
    "register_paro_silu_kernels",
    "shared_gate_combine_out_bf16",
    "shared_gate_combine_residual_out_bf16",
    "silu_mul_dual_out_bf16",
    "silu_mul_dual_rotate_out_bf16",
    "silu_mul_pair_rotate_out_bf16",
    "weighted_sum_out_bf16_f32w",
    "weighted_sum_shared_gate_combine_residual_out_bf16_f32w",
]
