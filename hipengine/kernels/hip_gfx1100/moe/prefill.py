"""Registry entry points for Qwen3.5/PARO MoE prefill orchestrators."""

from __future__ import annotations

from hipengine.core.dtype import DType
from hipengine.kernels.registry import KernelKey, register


def qwen35_moe_prefill_grouped_compact(state, hidden, residual, *, scratch=None, tokens: int = 1, **kwargs):
    """Run the retained grouped/compact MoE prefill route on a decode-state object."""

    if getattr(hidden, "dtype", None) is DType.FP16:
        return state.run_moe_grouped_compact_fp16(hidden, residual, scratch=scratch, tokens=tokens, **kwargs)
    return state.run_moe_grouped_compact_bf16(hidden, residual, scratch=scratch, tokens=tokens, **kwargs)


def qwen35_moe_prefill_selected_c1_rows(state, hidden, residual, *, scratch=None, tokens: int = 1, **kwargs):
    """Run the selected-row c1 MoE oracle/fallback route for tests only."""

    if getattr(hidden, "dtype", None) is DType.FP16:
        return state.run_moe_c1_fp16(hidden, residual, scratch=scratch, tokens=tokens, **kwargs)
    return state.run_moe_c1_bf16(hidden, residual, scratch=scratch, tokens=tokens, **kwargs)


def register_qwen35_moe_prefill_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "moe_prefill", "w4_paro", "qwen35_grouped_compact"),
        qwen35_moe_prefill_grouped_compact,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_prefill", "w4_paro", "qwen35_selected_c1_rows"),
        qwen35_moe_prefill_selected_c1_rows,
        replace=replace,
    )


register_qwen35_moe_prefill_kernels()
