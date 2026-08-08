"""CUDA ``sm_120a`` Maple router and MoE kernels."""

from hipengine.kernels.cuda_sm120a.moe.group_scatter import (
    build_qwen35_moe_group_scatter,
    plan_qwen35_moe_group_scatter_build,
    qwen35_moe_group_compact_active_i32_parallel,
    register_qwen35_moe_group_scatter_kernels,
)
from hipengine.kernels.cuda_sm120a.moe.maple_moe import (
    build_maple_moe,
    plan_maple_moe_build,
    register_maple_moe_kernels,
)

__all__ = [
    "build_maple_moe",
    "build_qwen35_moe_group_scatter",
    "plan_maple_moe_build",
    "plan_qwen35_moe_group_scatter_build",
    "qwen35_moe_group_compact_active_i32_parallel",
    "register_maple_moe_kernels",
    "register_qwen35_moe_group_scatter_kernels",
]
