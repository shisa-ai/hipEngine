"""gfx1100 WMMA kernel wrappers."""

from hipengine.kernels.hip_gfx1100.wmma.paro_awq_wmma import (
    build_paro_awq_wmma,
    gemm_awq_selected_dual_pack8_wmma_compact_bf16,
    gemm_awq_selected_dual_pack8_wmma_compact_fp16,
    gemm_awq_selected_pack8_wmma_compact_bf16,
    gemm_awq_selected_pack8_wmma_compact_fp16,
    plan_paro_awq_wmma_build,
    register_paro_awq_wmma_kernels,
)

__all__ = [
    "build_paro_awq_wmma",
    "gemm_awq_selected_dual_pack8_wmma_compact_bf16",
    "gemm_awq_selected_dual_pack8_wmma_compact_fp16",
    "gemm_awq_selected_pack8_wmma_compact_bf16",
    "gemm_awq_selected_pack8_wmma_compact_fp16",
    "plan_paro_awq_wmma_build",
    "register_paro_awq_wmma_kernels",
]
