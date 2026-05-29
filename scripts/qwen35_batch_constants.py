"""Shared Qwen3.5/PARO batch diagnostic constants.

Keep evidence-gating deny lists in one import-light module so c-sweep,
retained-bench, and artifact-schema validation cannot drift.
"""

from __future__ import annotations

PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS = (
    "serial",
    "fallback",
    "per_row",
    "per-row",
    "selected_c1",
    "selected-c1",
    "batch_gemv",
    "batch-gemv",
    "splitk",
    "split_k",
    "split-k",
)


__all__ = ["PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS"]
