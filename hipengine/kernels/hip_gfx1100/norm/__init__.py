"""gfx1100 normalization kernel wrappers."""

from hipengine.kernels.hip_gfx1100.norm.rmsnorm import (
    build_qwen35_rmsnorm,
    paro_add_rmsnorm_out_bf16,
    paro_add_rmsnorm_out_fp16,
    paro_rmsnorm_out_bf16,
    paro_rmsnorm_out_fp16,
    plan_qwen35_rmsnorm_build,
    qwen35_add_rmsnorm_bf16,
    qwen35_add_rmsnorm_f32_bf16,
    qwen35_head_rmsnorm_f32_bf16,
    qwen35_rmsnorm_bf16,
    register_qwen35_rmsnorm_kernels,
)

__all__ = [
    "build_qwen35_rmsnorm",
    "paro_add_rmsnorm_out_bf16",
    "paro_add_rmsnorm_out_fp16",
    "paro_rmsnorm_out_bf16",
    "paro_rmsnorm_out_fp16",
    "plan_qwen35_rmsnorm_build",
    "qwen35_add_rmsnorm_bf16",
    "qwen35_add_rmsnorm_f32_bf16",
    "qwen35_head_rmsnorm_f32_bf16",
    "qwen35_rmsnorm_bf16",
    "register_qwen35_rmsnorm_kernels",
]
