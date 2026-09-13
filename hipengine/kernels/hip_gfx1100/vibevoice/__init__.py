"""gfx1100-tree VibeVoice-ASR kernels (built per-arch for gfx1100/gfx1151)."""

from hipengine.kernels.hip_gfx1100.vibevoice.encoder import (
    build_vibevoice_encoder,
    conv_rows_out,
    f32_to_bf16_bits,
    plan_vibevoice_encoder_build,
    transpose_conv_weight_t,
    vv_add_scaled_noise_bf16,
    vv_add_bias_bf16,
    vv_conv_gemm_bf16,
    vv_depthwise_conv_bf16,
    vv_gelu_bf16,
    vv_rmsnorm_bf16,
    vv_scale_residual_bf16,
)

__all__ = [
    "build_vibevoice_encoder",
    "conv_rows_out",
    "f32_to_bf16_bits",
    "plan_vibevoice_encoder_build",
    "transpose_conv_weight_t",
    "vv_add_scaled_noise_bf16",
    "vv_conv_gemm_bf16",
    "vv_depthwise_conv_bf16",
    "vv_gelu_bf16",
    "vv_rmsnorm_bf16",
    "vv_scale_residual_bf16",
]
