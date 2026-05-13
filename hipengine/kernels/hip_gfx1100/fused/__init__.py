"""gfx1100 fused kernel wrappers."""

from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    plan_paro_silu_build,
    register_paro_silu_kernels,
    silu_mul_dual_out_bf16,
    silu_mul_dual_rotate_out_bf16,
    silu_mul_pair_rotate_out_bf16,
)

__all__ = [
    "build_paro_silu",
    "plan_paro_silu_build",
    "register_paro_silu_kernels",
    "silu_mul_dual_out_bf16",
    "silu_mul_dual_rotate_out_bf16",
    "silu_mul_pair_rotate_out_bf16",
]
