"""CUDA ``sm_120a`` Moonshine fused and glue primitives."""

from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
    build_moonshine_glue,
    moonshine_argmax_fp16,
    moonshine_embedding_lookup_fp16,
    moonshine_partial_rope_cache_append_fp16,
    moonshine_partial_rope_fp16,
    moonshine_residual_fp16,
    moonshine_self_cache_append_fp16,
    plan_moonshine_glue_build,
    register_moonshine_glue_kernels,
)
from hipengine.kernels.cuda_sm120a.fused.moonshine_mlp import (
    build_moonshine_mlp,
    moonshine_gated_silu_fp16,
    plan_moonshine_mlp_build,
    register_moonshine_mlp_kernels,
)

__all__ = [
    "build_moonshine_glue",
    "build_moonshine_mlp",
    "moonshine_argmax_fp16",
    "moonshine_embedding_lookup_fp16",
    "moonshine_gated_silu_fp16",
    "moonshine_partial_rope_cache_append_fp16",
    "moonshine_partial_rope_fp16",
    "moonshine_residual_fp16",
    "moonshine_self_cache_append_fp16",
    "plan_moonshine_glue_build",
    "plan_moonshine_mlp_build",
    "register_moonshine_glue_kernels",
    "register_moonshine_mlp_kernels",
]
