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

RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_EVIDENCE_FRAGMENTS = (
    "scripts/qwen35_batch_hidden_bisect.py",
    "qwen35_batch_hidden_bisect.py",
    "hidden-bisect",
    "hidden_bisect",
)

RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_TRACE_FIELD_NAMES = (
    "first_hidden_mismatch",
    "first_tolerance_hidden_mismatch",
    "first_strict_hidden_bit_drift",
    "first_failing_layer_transition",
    "first_hidden_mismatch_focus",
    "first_hidden_mismatch_linear_state_focus",
    "decode_linear_handoff_summary",
    "decode_linear_input_bit_drift_summary",
    "decode_linear_stage_bit_drift_summary",
    "decode_full_attention_bit_drift_summary",
    "decode_full_context_kv_prefix_failure_summary",
    "decode_full_context_oracle_failure_summary",
    "decode_full_kv_current_source_failure_summary",
    "prefill_full_kv_prefix_failure_summary",
)

RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_TRACE_FIELD_FRAGMENTS = (
    "hidden_mismatch",
    "hidden_bit_drift",
    "bit_drift_summary",
    "kv_prefix_failure_summary",
    "kv_current_source_failure_summary",
    "context_oracle_failure_summary",
)

RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES = (
    "failed",
    "invalid_json",
    "missing",
    "rejected",
    "rejected_correctness",
)

RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_COMMAND_FRAGMENTS = (
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_PROJECTIONS",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_STATE",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT",
    "HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE=0",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN",
    "HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR",
    "HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN",
    "--batch-decode-moe-path selected_c1",
    "--batch-decode-moe-path=selected_c1",
    "--batch-decode-linear-path per_row",
    "--batch-decode-linear-path=per_row",
    "--batch-decode-linear-projection-path selected_c1",
    "--batch-decode-linear-projection-path=selected_c1",
    "--batch-decode-linear-state-path selected_c1",
    "--batch-decode-linear-state-path=selected_c1",
    "--batch-decode-linear-output-path selected_c1",
    "--batch-decode-linear-output-path=selected_c1",
    "--batch-decode-linear-output-path batch_gemv",
    "--batch-decode-linear-output-path=batch_gemv",
    "--batch-decode-full-attn-path per_row",
    "--batch-decode-full-attn-path=per_row",
    "--batch-decode-attn-input-path per_row",
    "--batch-decode-attn-input-path=per_row",
    "--batch-decode-post-attn-path per_row",
    "--batch-decode-post-attn-path=per_row",
    "--batch-prefill-linear-path per_segment",
    "--batch-prefill-linear-path=per_segment",
    "--batch-prefill-full-attn-path per_segment",
    "--batch-prefill-full-attn-path=per_segment",
)


__all__ = [
    "PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS",
    "RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_COMMAND_FRAGMENTS",
    "RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_EVIDENCE_FRAGMENTS",
    "RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_TRACE_FIELD_FRAGMENTS",
    "RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_TRACE_FIELD_NAMES",
    "RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES",
]
