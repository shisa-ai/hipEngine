"""Shared Qwen3.5/PARO batch diagnostic constants.

Keep evidence-gating deny lists in one import-light module so c-sweep,
retained-bench, and artifact-schema validation cannot drift.
"""

from __future__ import annotations

from types import MappingProxyType

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

RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES = (
    "attention",
    "moe",
    "projection",
    "sampling",
    "graph_replay",
    "other",
)

RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES = (
    "load",
    "prefill",
    "warmup_decode",
    "decode",
    "validation",
    "other",
)

RETAINED_ARTIFACT_PROFILER_TRACE_KERNEL_NAME_COLUMNS = ("Kernel_Name", "KernelName", "Name")
RETAINED_ARTIFACT_PROFILER_TRACE_START_COLUMNS = ("Start_Timestamp", "StartTimestamp", "StartNs", "Start")
RETAINED_ARTIFACT_PROFILER_TRACE_END_COLUMNS = ("End_Timestamp", "EndTimestamp", "EndNs", "End")
RETAINED_ARTIFACT_PROFILER_TRACE_DURATION_COLUMNS = ("DurationNs", "Duration_NS", "Duration_Ns", "Duration")
RETAINED_ARTIFACT_ROCPROF_EXECUTABLE = "rocprofv3"
RETAINED_ARTIFACT_ROCPROF_COMMAND_FLAGS = ("--kernel-trace", "--output-format", "-d")
RETAINED_ARTIFACT_ROCPROF_OUTPUT_FORMAT = "csv"
RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_SCRIPT = "scripts/qwen35_batch_correctness.py"
RETAINED_ARTIFACT_SERIAL_BRIDGE_SCRIPT = "scripts/qwen35_batch_serial_bench.py"
RETAINED_ARTIFACT_LEGACY_NATIVE_BENCH_SCRIPT = "scripts/qwen35_paro_bench.py"
RETAINED_ARTIFACT_INT8_DIAGNOSTIC_SCRIPT = "scripts/qwen35_batch_int8_diagnostic.py"
RETAINED_ARTIFACT_GGUF_DIAGNOSTIC_SCRIPT = "scripts/qwen35_batch_gguf_diagnostic.py"
RETAINED_ARTIFACT_GGUF_E2E_CORRECTNESS_SCRIPT = "scripts/qwen35_gguf_e2e_correctness.py"
RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT = "scripts/qwen35_batch_retained_bench.py"
RETAINED_ARTIFACT_VALIDATION_SUMMARY_TYPE = "qwen35_retained_validation_summary"
RETAINED_ARTIFACT_ACCEPTED_MODE = "qwen35_paro_native_retained_bench"
RETAINED_ARTIFACT_ACCEPTED_SUMMARY = "Qwen3.5/PARO scheduler compact native c>N benchmark"
RETAINED_ARTIFACT_ACCEPTED_DECISION_REASON = "correctness/protocol passed"
RETAINED_ARTIFACT_ACCEPTED_NOTES = (
    "Native retained c>N path uses packed prompt slabs and step_batch_native for decode.",
    "Batch split-K decode remains out of scope; this accepted protocol keeps context < 1024.",
)

RETAINED_ARTIFACT_PROFILER_TRACE_SYNTHESIZED_FIELDS = (
    "trace_kernel_names",
    "kernel_durations_ns",
    "total_kernel_duration_ns",
    "kernel_duration_shares",
    "kernel_duration_categories_ns",
    "kernel_duration_category_shares",
)
RETAINED_ARTIFACT_PROFILER_SYNTHESIZED_FIELDS = RETAINED_ARTIFACT_PROFILER_TRACE_SYNTHESIZED_FIELDS + (
    "output_format",
    "trace_dir",
)

RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA = 1
RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED = 1234
RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS = MappingProxyType(
    {
        "block_size": 256,
        "max_context_len": 4,
        "num_q_heads": 4,
        "num_kv_heads": 1,
        "head_dim": 8,
    }
)
RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT = 2e-5

RETAINED_ARTIFACT_REQUIRED_SCALING_BASELINES = (
    "c1_baseline",
    "serial_bridge_baseline",
)

RETAINED_ARTIFACT_REQUIRED_SCALING_RATIOS = (
    "aggregate_vs_c1",
    "per_request_vs_c1",
    "aggregate_vs_serial_bridge",
    "per_request_vs_serial_bridge",
)

RETAINED_ARTIFACT_RETAINED_PRECONDITION_KINDS = (
    "primitive_correctness",
    "c1_baseline",
    "serial_bridge",
    "profiler_summary",
)
RETAINED_ARTIFACT_RETAINED_POSTCONDITION_KINDS = (
    "retained_profiler_synthesis",
)
RETAINED_ARTIFACT_RETAINED_CONDITION_STATUS_LABELS = (
    "passed",
    "failed",
)
RETAINED_ARTIFACT_SWEEP_COMMAND_CATEGORIES = (
    "primitive",
    "serial_bridge",
    "native_diagnostic",
    "int8_native_diagnostic",
    "gguf_native_diagnostic",
)
RETAINED_ARTIFACT_SWEEP_COMMAND_STATUS_LABELS = (
    "planned",
    "passed",
    "skipped",
    "failed",
)
RETAINED_ARTIFACT_RETAINED_GATE_FLAGS = (
    "--c1-baseline-json",
    "--serial-bridge-json",
    "--primitive-correctness-json",
    "--profiler-json",
)
RETAINED_ARTIFACT_RETAINED_GATE_LABELS = (
    "c1_baseline_json",
    "serial_bridge_json",
    "primitive_correctness_json",
    "profiler_json",
)
RETAINED_ARTIFACT_INT8_PRIMITIVE_GATE_FLAGS = (
    "--int8-kv-primitive-cpu-json",
    "--int8-kv-primitive-hip-json",
)
RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_UNIQUE_FLAGS = ("--rows", "--seed")
RETAINED_ARTIFACT_CORRECTNESS_REFERENCE_UNIQUE_FLAGS = ("--rows", "--seed", "--json")
RETAINED_ARTIFACT_CORRECTNESS_SCRIPT_ALLOWED_FLAGS = RETAINED_ARTIFACT_CORRECTNESS_REFERENCE_UNIQUE_FLAGS

RETAINED_ARTIFACT_RETAINED_BENCH_UNIQUE_FLAGS = (
    "--model",
    "--fixture",
    "--batch-size",
    "--prompt-length",
    "--decode-tokens",
    "--warmup-decode-tokens",
    "--max-layers",
    "--json",
    *RETAINED_ARTIFACT_RETAINED_GATE_FLAGS,
    *RETAINED_ARTIFACT_INT8_PRIMITIVE_GATE_FLAGS,
    "--projection-dispatch-artifact",
    "--compiler-version-file",
    "--require-cached-build",
)
RETAINED_ARTIFACT_RETAINED_KV_POLICY_FLAGS = (
    "--kv-storage",
    "--kv-scale-dtype",
    "--kv-scale-granularity",
)
RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS = tuple(
    dict.fromkeys(RETAINED_ARTIFACT_RETAINED_BENCH_UNIQUE_FLAGS + RETAINED_ARTIFACT_RETAINED_KV_POLICY_FLAGS)
)
RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_VALUE_FLAGS = (
    "--model",
    "--fixture",
    "--batch-size",
    "--prompt-length",
    "--decode-tokens",
    "--warmup-decode-tokens",
    "--max-layers",
    "--json",
    *RETAINED_ARTIFACT_RETAINED_GATE_FLAGS,
    *RETAINED_ARTIFACT_INT8_PRIMITIVE_GATE_FLAGS,
    "--projection-dispatch-artifact",
    "--compiler-version-file",
    *RETAINED_ARTIFACT_RETAINED_KV_POLICY_FLAGS,
)

RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES = (
    "failed",
    "invalid_json",
    "missing",
    "rejected",
    "rejected_correctness",
)

RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS = (
    "--skip-generated-equality",
)

RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_COMMAND_FRAGMENTS = (
    *RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS,
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE",
    "HIPENGINE_QWEN35_SHARED_EXPERT_PARO_W4_FORCE_GEMV",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_PROJECTIONS",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_AB",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_LINEAR_PROJECTIONS",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_STATE",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT",
    "HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE=0",
    "HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_SIZE",
    "HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_LAYERS",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_CONTEXT",
    "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN",
    "HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR",
    "HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN",
    "--batch-decode-moe-path selected_c1",
    "--batch-decode-moe-path=selected_c1",
    "--batch-decode-linear-path per_row",
    "--batch-decode-linear-path=per_row",
    "--batch-decode-linear-projection-path selected_c1",
    "--batch-decode-linear-projection-path=selected_c1",
    "--batch-decode-linear-projection-path selected_qkv_z",
    "--batch-decode-linear-projection-path=selected_qkv_z",
    "--batch-decode-linear-projection-path selected_ab",
    "--batch-decode-linear-projection-path=selected_ab",
    "--batch-decode-linear-projection-path batch_gemv_selected_ab",
    "--batch-decode-linear-projection-path=batch_gemv_selected_ab",
    "--batch-decode-linear-projection-path batch_gemv",
    "--batch-decode-linear-projection-path=batch_gemv",
    "--batch-decode-linear-state-path selected_c1",
    "--batch-decode-linear-state-path=selected_c1",
    "--batch-decode-linear-output-path selected_c1",
    "--batch-decode-linear-output-path=selected_c1",
    "--batch-decode-full-attn-row-chunk-layers ",
    "--batch-decode-full-attn-row-chunk-layers=",
    "--batch-decode-linear-output-path batch_gemv",
    "--batch-decode-linear-output-path=batch_gemv",
    "--batch-decode-full-attn-path per_row",
    "--batch-decode-full-attn-path=per_row",
    "--batch-decode-full-attn-row-chunk-size",
    "--batch-decode-attn-input-path per_row",
    "--batch-decode-attn-input-path=per_row",
    "--batch-decode-attn-context-path per_row",
    "--batch-decode-attn-context-path=per_row",
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
    "RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT",
    "RETAINED_ARTIFACT_PROFILER_TRACE_DURATION_COLUMNS",
    "RETAINED_ARTIFACT_PROFILER_TRACE_END_COLUMNS",
    "RETAINED_ARTIFACT_PROFILER_TRACE_KERNEL_NAME_COLUMNS",
    "RETAINED_ARTIFACT_PROFILER_TRACE_START_COLUMNS",
    "RETAINED_ARTIFACT_PROFILER_TRACE_SYNTHESIZED_FIELDS",
    "RETAINED_ARTIFACT_ROCPROF_EXECUTABLE",
    "RETAINED_ARTIFACT_ROCPROF_COMMAND_FLAGS",
    "RETAINED_ARTIFACT_ROCPROF_OUTPUT_FORMAT",
    "RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_SCRIPT",
    "RETAINED_ARTIFACT_SERIAL_BRIDGE_SCRIPT",
    "RETAINED_ARTIFACT_LEGACY_NATIVE_BENCH_SCRIPT",
    "RETAINED_ARTIFACT_INT8_DIAGNOSTIC_SCRIPT",
    "RETAINED_ARTIFACT_GGUF_DIAGNOSTIC_SCRIPT",
    "RETAINED_ARTIFACT_GGUF_E2E_CORRECTNESS_SCRIPT",
    "RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT",
    "RETAINED_ARTIFACT_CORRECTNESS_REFERENCE_UNIQUE_FLAGS",
    "RETAINED_ARTIFACT_CORRECTNESS_SCRIPT_ALLOWED_FLAGS",
    "RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_UNIQUE_FLAGS",
    "RETAINED_ARTIFACT_PROFILER_SYNTHESIZED_FIELDS",
    "RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA",
    "RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED",
    "RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS",
    "RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES",
    "RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES",
    "RETAINED_ARTIFACT_REQUIRED_SCALING_BASELINES",
    "RETAINED_ARTIFACT_RETAINED_PRECONDITION_KINDS",
    "RETAINED_ARTIFACT_RETAINED_POSTCONDITION_KINDS",
    "RETAINED_ARTIFACT_RETAINED_CONDITION_STATUS_LABELS",
    "RETAINED_ARTIFACT_SWEEP_COMMAND_CATEGORIES",
    "RETAINED_ARTIFACT_SWEEP_COMMAND_STATUS_LABELS",
    "RETAINED_ARTIFACT_RETAINED_GATE_FLAGS",
    "RETAINED_ARTIFACT_RETAINED_GATE_LABELS",
    "RETAINED_ARTIFACT_INT8_PRIMITIVE_GATE_FLAGS",
    "RETAINED_ARTIFACT_RETAINED_BENCH_UNIQUE_FLAGS",
    "RETAINED_ARTIFACT_RETAINED_KV_POLICY_FLAGS",
    "RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS",
    "RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_VALUE_FLAGS",
    "RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS",
    "RETAINED_ARTIFACT_REQUIRED_SCALING_RATIOS",
    "RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES",
]
