"""gfx1100 linear-attention kernel wrappers."""

from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    build_qwen35_linear_attn_conv,
    plan_qwen35_linear_attn_conv_build,
    qwen35_linear_attn_conv_decode_bf16,
    qwen35_linear_attn_conv_decode_f32,
    register_qwen35_linear_attn_conv_kernels,
)

__all__ = [
    "build_qwen35_linear_attn_conv",
    "plan_qwen35_linear_attn_conv_build",
    "qwen35_linear_attn_conv_decode_bf16",
    "qwen35_linear_attn_conv_decode_f32",
    "register_qwen35_linear_attn_conv_kernels",
]
