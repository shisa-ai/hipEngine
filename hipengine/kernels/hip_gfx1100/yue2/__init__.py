"""gfx1100-tree YuE2 kernels (built per-arch for gfx1100/gfx1151)."""

from hipengine.kernels.hip_gfx1100.yue2.nar import (
    build_yue2_nar,
    nar_add_broadcast_bf16,
    nar_attention_f32,
    nar_attention_wmma,
    nar_gather_add_bf16,
    nar_state_update_bf16,
    plan_yue2_nar_build,
)
from hipengine.kernels.hip_gfx1100.yue2.vae import (
    build_yue2_vae,
    plan_yue2_vae_build,
    vae_add_f32,
    vae_conv1d_f32,
    vae_conv_transpose1d_f32,
    vae_snake_beta_f32,
)

__all__ = [
    "build_yue2_nar",
    "build_yue2_vae",
    "nar_add_broadcast_bf16",
    "nar_attention_f32",
    "nar_attention_wmma",
    "nar_gather_add_bf16",
    "nar_state_update_bf16",
    "plan_yue2_nar_build",
    "plan_yue2_vae_build",
    "vae_add_f32",
    "vae_conv1d_f32",
    "vae_conv_transpose1d_f32",
    "vae_snake_beta_f32",
]
