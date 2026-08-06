"""CUDA ``sm_120a`` Moonshine attention kernels."""

from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
    build_moonshine_attention,
    moonshine_cross_attention_fp16,
    moonshine_cross_attention_grouped_fp16,
    moonshine_cross_attention_parallel_fp16,
    moonshine_self_attention_fp16,
    moonshine_self_attention_parallel_fp16,
    plan_moonshine_attention_build,
    register_moonshine_attention_kernels,
)

__all__ = [
    "build_moonshine_attention",
    "moonshine_cross_attention_fp16",
    "moonshine_cross_attention_grouped_fp16",
    "moonshine_cross_attention_parallel_fp16",
    "moonshine_self_attention_fp16",
    "moonshine_self_attention_parallel_fp16",
    "plan_moonshine_attention_build",
    "register_moonshine_attention_kernels",
]
