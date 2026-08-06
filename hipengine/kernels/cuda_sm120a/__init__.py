"""Architecture-qualified CUDA ``sm_120a`` backend scaffold."""

from __future__ import annotations

from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.cuda_sm120a.attention import register_moonshine_attention_kernels
from hipengine.kernels.cuda_sm120a.fused import (
    register_moonshine_glue_kernels,
    register_moonshine_mlp_kernels,
)
from hipengine.kernels.cuda_sm120a.linear import (
    register_moonshine_lm_head_kernels,
    register_moonshine_projection_kernels,
)
from hipengine.kernels.cuda_sm120a.norm import register_moonshine_layernorm_kernels
from hipengine.kernels.cuda_sm120a.smoke import register_smoke_add_kernel
from hipengine.kernels.registry import KernelKey, is_registered

BACKEND = "cuda_sm120a"
TARGET_ARCH = cuda_target_arch_for_backend(BACKEND)


def register_backend_kernels(*, replace: bool = True) -> None:
    """Register only CUDA keys implemented independently in this peer package."""

    smoke_key = KernelKey(BACKEND, "smoke_add", "fp32")
    if replace or not is_registered(smoke_key):
        register_smoke_add_kernel(replace=replace)
    glue_key = KernelKey(BACKEND, "moonshine_embedding", "fp16", "lookup_i64")
    if replace or not is_registered(glue_key):
        register_moonshine_glue_kernels(replace=replace)
    layernorm_key = KernelKey(
        BACKEND,
        "moonshine_layernorm",
        "fp16",
        "fp32_stats",
    )
    if replace or not is_registered(layernorm_key):
        register_moonshine_layernorm_kernels(replace=replace)
    projection_key = KernelKey(
        BACKEND,
        "moonshine_projection",
        "fp16",
        "single_fp32_accum",
    )
    if replace or not is_registered(projection_key):
        register_moonshine_projection_kernels(replace=replace)
    mlp_key = KernelKey(BACKEND, "moonshine_gated_silu", "fp16", "value_gate_split")
    if replace or not is_registered(mlp_key):
        register_moonshine_mlp_kernels(replace=replace)
    lm_head_key = KernelKey(
        BACKEND,
        "moonshine_lm_head",
        "fp16",
        "fused_argmax_fp32_accum",
    )
    if replace or not is_registered(lm_head_key):
        register_moonshine_lm_head_kernels(replace=replace)
    attention_key = KernelKey(
        BACKEND,
        "moonshine_self_attention",
        "fp16",
        "fixed_cache_logical_dim",
    )
    if replace or not is_registered(attention_key):
        register_moonshine_attention_kernels(replace=replace)


register_backend_kernels()

__all__ = ["BACKEND", "TARGET_ARCH", "register_backend_kernels"]
