#!/usr/bin/env python3
"""Retained Qwen3.5/PARO compact c>N benchmark.

This is the accepted-path companion to ``qwen35_batch_serial_bench.py``.  It
uses scheduler-owned compact native prefill plus ``step_batch_native`` decode,
then (unless skipped) compares generated token ids against independent c=1
resident runs before marking a row accepted.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
from collections.abc import Mapping
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import memory_stats
from hipengine.dispatch import (
    BatchSamplerMode,
    ProjectionDispatchEvidence,
    batch_sampler_equality_payload_blockers,
    plan_batch_sampler_dispatch,
    projection_dispatch_candidates_from_artifact,
    projection_dispatch_evidence_payload_blockers,
)
from hipengine.generation import GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS, GeneratedToken, GraphBucketCache, ResidentBatchScheduler
from hipengine.kvcache import ResolvedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_artifact_schema import (
    DECODE_EXECUTION_DIAGNOSTIC_TRACE_FIELDS,
    _load_payload,
    validate_cn_diagnostic_artifact_payload,
)
from scripts.qwen35_batch_constants import (
    PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS,
    RETAINED_ARTIFACT_ACCEPTED_DECISION_REASON,
    RETAINED_ARTIFACT_ACCEPTED_MODE,
    RETAINED_ARTIFACT_ACCEPTED_NOTES,
    RETAINED_ARTIFACT_ACCEPTED_SUMMARY,
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT,
    RETAINED_ARTIFACT_LEGACY_NATIVE_BENCH_SCRIPT,
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_SCRIPT,
    RETAINED_ARTIFACT_PROFILER_TRACE_DURATION_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_END_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_KERNEL_NAME_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_START_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_SYNTHESIZED_FIELDS,
    RETAINED_ARTIFACT_ROCPROF_COMMAND_FLAGS,
    RETAINED_ARTIFACT_ROCPROF_EXECUTABLE,
    RETAINED_ARTIFACT_ROCPROF_OUTPUT_FORMAT,
    RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT,
    RETAINED_ARTIFACT_RETAINED_GATE_FLAGS,
    RETAINED_ARTIFACT_RETAINED_GATE_LABELS,
    RETAINED_ARTIFACT_RETAINED_KV_POLICY_FLAGS,
    RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS,
    RETAINED_ARTIFACT_SERIAL_BRIDGE_SCRIPT,
    RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS,
    RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES,
    RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES,
    RETAINED_ARTIFACT_REQUIRED_SCALING_BASELINES,
    RETAINED_ARTIFACT_REQUIRED_SCALING_RATIOS,
    RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES,
)
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = "/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16"
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"
_PROFILER_KERNEL_DURATION_CATEGORIES = RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES
_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES = RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES
_PROFILER_TRACE_KERNEL_NAME_COLUMNS = RETAINED_ARTIFACT_PROFILER_TRACE_KERNEL_NAME_COLUMNS
_PROFILER_TRACE_START_COLUMNS = RETAINED_ARTIFACT_PROFILER_TRACE_START_COLUMNS
_PROFILER_TRACE_END_COLUMNS = RETAINED_ARTIFACT_PROFILER_TRACE_END_COLUMNS
_PROFILER_TRACE_DURATION_COLUMNS = RETAINED_ARTIFACT_PROFILER_TRACE_DURATION_COLUMNS
_ROCPROF_COMMAND_FLAGS = RETAINED_ARTIFACT_ROCPROF_COMMAND_FLAGS
_ROCPROF_EXECUTABLE = RETAINED_ARTIFACT_ROCPROF_EXECUTABLE
_ROCPROF_OUTPUT_FORMAT = RETAINED_ARTIFACT_ROCPROF_OUTPUT_FORMAT
_RETAINED_BENCH_SCRIPT = RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT
_PROJECTION_DISPATCH_ARTIFACT_ENV = "HIPENGINE_QWEN35_PROJECTION_DISPATCH_ARTIFACT"
_DEFAULT_PROJECTION_DISPATCH_ARTIFACT = (
    "benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-projection-dispatch-catalog/summary.json"
)
_RETAINED_GATE_FLAGS = RETAINED_ARTIFACT_RETAINED_GATE_FLAGS
_RETAINED_GATE_LABELS = RETAINED_ARTIFACT_RETAINED_GATE_LABELS
_RETAINED_KV_POLICY_FLAGS = RETAINED_ARTIFACT_RETAINED_KV_POLICY_FLAGS
_PRIMITIVE_CORRECTNESS_SCRIPT = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_SCRIPT
_LEGACY_NATIVE_BENCH_SCRIPT = RETAINED_ARTIFACT_LEGACY_NATIVE_BENCH_SCRIPT
_SERIAL_BRIDGE_SCRIPT = RETAINED_ARTIFACT_SERIAL_BRIDGE_SCRIPT
_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS = RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS
_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS = RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS
_BATCH_SAMPLE_COMMAND_FLAGS = (
    "--batch-sample-mode",
    "--batch-sample-norm-path",
    "--batch-sample-cast-path",
    "--batch-sample-argmax-mode",
    "--batch-sample-argmax-audit",
    "--batch-sample-lm-head-audit",
    "--batch-sample-lm-head-kernel-fence",
    "--batch-sample-final-norm-audit",
    "--batch-sample-final-norm-kernel-fence",
    "--batch-sample-final-rmsnorm-kernel-fence",
    "--batch-sample-final-rmsnorm-temp-fence",
    "--batch-sample-final-cast-temp-fence",
    "--batch-sample-final-cast-tiny-fence",
    "--batch-sample-final-cast-elems-fence",
    "--batch-sample-stabilize-cast-elems",
    "--batch-sample-sync-fence",
    "--batch-sample-suffix-fence",
    "--batch-sample-suffix-kernel-fence",
    "--batch-sample-eq-ok",
    "--batch-sample-eq-artifact",
    "--batch-sample-eq-rows",
)
_DEFAULT_BATCH_SAMPLE_EQ_ARTIFACT_TEMPLATE = (
    "benchmarks/results/2026-06-02-hipengine-qwen35-c{rows}-native-batch-sampler-equality.json"
)
_DEFAULT_BATCH_SAMPLE_EQ_ROWS = (2, 4, 8)
_DEFAULT_BATCH_SAMPLE_STABILIZE_CAST_ELEMS = {8: 256}
_DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS = PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS
_ACCEPTED_MODE = RETAINED_ARTIFACT_ACCEPTED_MODE
_ACCEPTED_SUMMARY = RETAINED_ARTIFACT_ACCEPTED_SUMMARY
_ACCEPTED_DECISION_REASON = RETAINED_ARTIFACT_ACCEPTED_DECISION_REASON
_ACCEPTED_NOTES = RETAINED_ARTIFACT_ACCEPTED_NOTES
_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS
_REQUIRED_PRIMITIVE_CORRECTNESS_SEED = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED
_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT
_UNUSABLE_SCALING_REFERENCE_STATUSES = RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES
_COMMAND_ENV_KEYS = ("HIP_VISIBLE_DEVICES",)


def _argv_has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _apply_default_batch_sample_evidence(args: argparse.Namespace, argv: Sequence[str]) -> None:
    """Enable the row-aware batched LM-head only when canonical evidence exists.

    The active c=2/c=4/c=8 retained diagnostic gates have repo-retained
    generated-token equality artifacts for the row-aware sampler.  Use them as
    the no-flag diagnostic default for covered rows while preserving an explicit
    user override and the existing evidence gate.  C2 is included because the
    serial LM-head default is now the observed token-104 intermittent path, while
    evidenced batched LM-head c2 repeats stayed green.  The c=8 default also
    enables a post-sampler cast stabilizer because the unstabilized / undersized
    batched sampler path remains intermittent under retained evidence runs.
    """

    if any(_argv_has_flag(argv, flag) for flag in _BATCH_SAMPLE_COMMAND_FLAGS):
        return
    rows = getattr(args, "batch_size", None)
    if isinstance(rows, bool) or not isinstance(rows, int) or rows not in _DEFAULT_BATCH_SAMPLE_EQ_ROWS:
        return
    artifact = _DEFAULT_BATCH_SAMPLE_EQ_ARTIFACT_TEMPLATE.format(rows=rows)
    path = Path(artifact)
    if not path.exists():
        return
    decision = plan_batch_sampler_dispatch(
        rows=rows,
        requested_mode=BatchSamplerMode.BATCHED_LM_HEAD,
        c2_equality_green=True,
        equality_artifact=artifact,
        equality_rows=rows,
    )
    if decision.mode is not BatchSamplerMode.BATCHED_LM_HEAD or decision.blockers:
        return
    args.batch_sample_mode = BatchSamplerMode.BATCHED_LM_HEAD.value
    args.batch_sample_eq_ok = True
    args.batch_sample_eq_artifact = path
    args.batch_sample_eq_rows = rows
    stabilize_elems = _DEFAULT_BATCH_SAMPLE_STABILIZE_CAST_ELEMS.get(rows)
    if stabilize_elems is not None:
        args.batch_sample_stabilize_cast_elems = int(stabilize_elems)


def _load_json_path(path: Path) -> Mapping[str, Any]:
    return _load_payload(path)


def _required_primitive_context_lens(rows: int) -> list[int]:
    max_context_len = _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["max_context_len"]
    return [(idx % max_context_len) + 1 for idx in range(rows)]


def _primitive_context_lens_matches(value: Any, rows: int) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and value == _required_primitive_context_lens(rows)
    )


def _is_zero_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


_PROFILER_SYNTHESIZED_FIELDS = RETAINED_ARTIFACT_PROFILER_SYNTHESIZED_FIELDS


def _load_prompt_slices(path: Path, *, prompt_length: int, batch_size: int) -> list[list[int]]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    fixture = _load_json_path(path)
    tokens = [int(token) for token in fixture["prompt_ids"]]
    needed = int(prompt_length) * int(batch_size)
    if len(tokens) < needed:
        raise ValueError(f"fixture contains {len(tokens)} tokens, need at least {needed}")
    return [tokens[row * prompt_length : (row + 1) * prompt_length] for row in range(batch_size)]


def _result_payload(result) -> dict[str, Any]:
    return {"token_id": int(result.token_id), "token_text": result.token_text, "logit": float(result.logit)}


def _shape_key_payload(key) -> dict[str, Any]:
    return {
        "mode": key.mode.value,
        "active_c": key.active_c,
        "context_bucket": key.context_bucket,
        "active_mask": list(key.active_mask),
        "kv_storage_dtype": key.kv_storage_dtype,
        "layer_plan": key.layer_plan,
        "top_k": key.top_k,
        "experts_per_token": key.experts_per_token,
        "replay_steps": key.replay_steps,
        "draft_depth": key.draft_depth,
        "tree_shape": list(key.tree_shape),
    }


def _record_decode_graph_bucket_metadata(
    scheduler: ResidentBatchScheduler,
    scheduler_metadata: dict[str, Any],
    *,
    kv_storage_dtype: str = "bf16",
    layer_plan: str = "all",
) -> None:
    key = scheduler.shape_key(
        mode="decode",
        top_k=0,
        experts_per_token=0,
        replay_steps=1,
        kv_storage_dtype=kv_storage_dtype,
        layer_plan=layer_plan,
    )
    scheduler_metadata["decode_shape_key"] = _shape_key_payload(key)
    scheduler.graph_buckets.get_or_create(key, _shape_key_payload)
    scheduler.graph_buckets.get(key)
    scheduler_metadata["graph_bucket_stats"] = scheduler.graph_buckets.stats.to_json_dict()


def _profiler_graph_kernel_time_histogram(profiler: Mapping[str, Any]) -> dict[str, int] | None:
    kernel_durations = profiler.get("kernel_durations_ns")
    if not isinstance(kernel_durations, Mapping):
        return None
    cache = GraphBucketCache()
    for duration_ns in kernel_durations.values():
        if not _is_finite_positive_number(duration_ns):
            continue
        numeric_duration = float(duration_ns)
        if not numeric_duration.is_integer():
            continue
        cache.record_kernel_time_ns(int(numeric_duration))
    histogram = cache.stats.kernel_time_histogram_ns
    if not histogram:
        return None
    return {bucket: int(histogram.get(bucket, 0)) for bucket in GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS}


def _attach_profiler_graph_kernel_time_histogram(scheduler_metadata: dict[str, Any], profiler: Mapping[str, Any]) -> None:
    profiler_histogram = _profiler_graph_kernel_time_histogram(profiler)
    if profiler_histogram is None:
        return
    graph_stats = scheduler_metadata.get("graph_bucket_stats")
    if not isinstance(graph_stats, Mapping):
        return
    updated_stats = dict(graph_stats)
    existing_histogram = updated_stats.get("kernel_time_histogram_ns")
    merged_histogram = dict(existing_histogram) if isinstance(existing_histogram, Mapping) else {}
    for bucket, count in profiler_histogram.items():
        current_count = merged_histogram.get(bucket, 0)
        if isinstance(current_count, bool) or not isinstance(current_count, int) or current_count < 0:
            current_count = 0
        merged_histogram[bucket] = int(current_count) + int(count)
    fixed_histogram: dict[str, int] = {}
    for bucket in GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS:
        count = merged_histogram.get(bucket, 0)
        fixed_histogram[bucket] = int(count) if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 0
    updated_stats["kernel_time_histogram_ns"] = fixed_histogram
    scheduler_metadata["graph_bucket_stats"] = updated_stats


def _decode_shape_key_blockers(
    scheduler_metadata: Mapping[str, Any],
    *,
    concurrency: int,
    prompt_length: int,
    kv_storage_dtype: str | None = None,
    layer_plan: str | None = None,
) -> list[str]:
    decode_shape_key = scheduler_metadata.get("decode_shape_key")
    if not isinstance(decode_shape_key, Mapping):
        return ["execution.scheduler_metadata.decode_shape_key is missing"]
    blockers: list[str] = []
    if decode_shape_key.get("mode") != "decode":
        blockers.append("execution.scheduler_metadata.decode_shape_key.mode must be decode")
    active_c = decode_shape_key.get("active_c")
    if active_c != int(concurrency):
        blockers.append("execution.scheduler_metadata.decode_shape_key.active_c must match workload.concurrency")
    active_mask = decode_shape_key.get("active_mask")
    if not isinstance(active_mask, list) or not active_mask or any(not isinstance(item, bool) for item in active_mask):
        blockers.append("execution.scheduler_metadata.decode_shape_key.active_mask must be a non-empty bool list")
    else:
        if len(active_mask) != int(concurrency):
            blockers.append("execution.scheduler_metadata.decode_shape_key.active_mask length must match workload.concurrency")
        if sum(1 for active in active_mask if active) != int(concurrency):
            blockers.append("execution.scheduler_metadata.decode_shape_key.active_mask true count must match workload.concurrency")
    context_bucket = decode_shape_key.get("context_bucket")
    if isinstance(context_bucket, bool) or not isinstance(context_bucket, int) or context_bucket <= 0:
        blockers.append("execution.scheduler_metadata.decode_shape_key.context_bucket must be a positive int")
    elif context_bucket < int(prompt_length):
        blockers.append("execution.scheduler_metadata.decode_shape_key.context_bucket must cover workload.prompt_tokens_per_request")
    key_kv_dtype = decode_shape_key.get("kv_storage_dtype")
    if not isinstance(key_kv_dtype, str) or not key_kv_dtype.strip():
        blockers.append("execution.scheduler_metadata.decode_shape_key.kv_storage_dtype must be a non-empty string")
    elif kv_storage_dtype is not None and key_kv_dtype != str(kv_storage_dtype):
        blockers.append("execution.scheduler_metadata.decode_shape_key.kv_storage_dtype must match workload.kv_storage_dtype")
    key_layer_plan = decode_shape_key.get("layer_plan")
    if not isinstance(key_layer_plan, str) or not key_layer_plan.strip():
        blockers.append("execution.scheduler_metadata.decode_shape_key.layer_plan must be a non-empty string")
    elif layer_plan is not None and key_layer_plan != str(layer_plan):
        blockers.append("execution.scheduler_metadata.decode_shape_key.layer_plan must match workload layer plan")
    for field in ("top_k", "experts_per_token", "draft_depth"):
        value = decode_shape_key.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            blockers.append(f"execution.scheduler_metadata.decode_shape_key.{field} must be a non-negative int")
    replay_steps = decode_shape_key.get("replay_steps")
    if isinstance(replay_steps, bool) or not isinstance(replay_steps, int) or replay_steps <= 0:
        blockers.append("execution.scheduler_metadata.decode_shape_key.replay_steps must be a positive int")
    tree_shape = decode_shape_key.get("tree_shape")
    if not isinstance(tree_shape, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tree_shape):
        blockers.append("execution.scheduler_metadata.decode_shape_key.tree_shape must be a list of non-negative ints")
    return blockers


def _graph_replay_stats_blockers(scheduler_metadata: Mapping[str, Any]) -> list[str]:
    graph_stats = scheduler_metadata.get("graph_bucket_stats")
    if not isinstance(graph_stats, Mapping):
        return ["execution.scheduler_metadata.graph_bucket_stats is missing"]
    blockers: list[str] = []
    integer_fields: dict[str, int] = {}
    for field in ("entries", "hits", "misses"):
        value = graph_stats.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            blockers.append(f"execution.scheduler_metadata.graph_bucket_stats.{field} is unavailable or non-integer")
            continue
        integer_fields[field] = int(value)
    replay_kernel_hits = graph_stats.get("replay_kernel_hits", 0)
    if isinstance(replay_kernel_hits, bool) or not isinstance(replay_kernel_hits, int) or replay_kernel_hits < 0:
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.replay_kernel_hits must be a non-negative int")
    elif "hits" in integer_fields and int(replay_kernel_hits) > integer_fields["hits"]:
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.replay_kernel_hits must not exceed hits")
    entries = integer_fields.get("entries")
    hits = integer_fields.get("hits")
    misses = integer_fields.get("misses")
    if entries is not None and entries <= 0:
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.entries must be positive")
    if hits is not None and hits <= 0:
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.hits must be positive")
    replay_hit_rate = graph_stats.get("replay_hit_rate")
    replay_hit_rate_valid = _is_finite_positive_number(replay_hit_rate) and float(replay_hit_rate) <= 1.0
    if not replay_hit_rate_valid:
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.replay_hit_rate must be finite positive <= 1")
    elif hits is not None and misses is not None and hits + misses > 0:
        expected_replay_hit_rate = float(hits) / float(hits + misses)
        if abs(float(replay_hit_rate) - expected_replay_hit_rate) > 1e-9:
            blockers.append("execution.scheduler_metadata.graph_bucket_stats.replay_hit_rate must match hits / (hits + misses)")
    if entries is not None and hits is not None and misses is not None and entries > hits + misses:
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.entries must be covered by hits plus misses")
    miss_reasons = graph_stats.get("miss_reasons")
    if not isinstance(miss_reasons, Mapping):
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.miss_reasons is missing")
    else:
        miss_reason_total = 0
        miss_reason_total_valid = True
        for reason, count in miss_reasons.items():
            if not isinstance(reason, str) or not reason:
                blockers.append("execution.scheduler_metadata.graph_bucket_stats.miss_reasons keys must be non-empty strings")
                miss_reason_total_valid = False
                break
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                blockers.append(f"execution.scheduler_metadata.graph_bucket_stats.miss_reasons.{reason} is unavailable or non-integer")
                miss_reason_total_valid = False
                break
            miss_reason_total += int(count)
        if misses is not None and misses > 0 and not miss_reasons:
            blockers.append("execution.scheduler_metadata.graph_bucket_stats.miss_reasons must be non-empty when misses is positive")
        if misses is not None and miss_reason_total_valid and miss_reason_total != misses:
            blockers.append("execution.scheduler_metadata.graph_bucket_stats.miss_reasons counts must sum to misses")
    return blockers


def _graph_kernel_time_histogram_blockers(scheduler_metadata: Mapping[str, Any]) -> list[str]:
    graph_stats = scheduler_metadata.get("graph_bucket_stats")
    if not isinstance(graph_stats, Mapping):
        return ["execution.scheduler_metadata.graph_bucket_stats is missing"]
    histogram = graph_stats.get("kernel_time_histogram_ns")
    if not isinstance(histogram, Mapping):
        return ["execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns is missing"]
    total_observations = 0
    blockers: list[str] = []
    allowed_buckets = set(GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS)
    if set(histogram) != allowed_buckets:
        allowed = ", ".join(GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS)
        blockers.append(f"execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns must include exactly the fixed buckets {allowed}")
    for bucket, count in histogram.items():
        if not isinstance(bucket, str) or bucket not in allowed_buckets:
            blockers.append(f"execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns.{bucket} is not a known bucket")
            continue
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            blockers.append(f"execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns.{bucket} is unavailable or non-integer")
            continue
        total_observations += int(count)
    if total_observations <= 0:
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns has no observations")
    hits = graph_stats.get("hits")
    if isinstance(hits, int) and not isinstance(hits, bool) and hits > 0 and total_observations < hits:
        blockers.append("execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns observation count must cover graph_bucket_stats.hits")
    return blockers


def _graph_bucket_evidence_blockers(scheduler_metadata: Mapping[str, Any]) -> list[str]:
    blockers = _graph_replay_stats_blockers(scheduler_metadata)
    for blocker in _graph_kernel_time_histogram_blockers(scheduler_metadata):
        if blocker not in blockers:
            blockers.append(blocker)
    return blockers


def _graph_kernel_time_histogram_bucket_ns(duration_ns: int) -> str:
    if duration_ns <= 10_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[0]
    if duration_ns <= 100_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[1]
    if duration_ns <= 1_000_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[2]
    if duration_ns <= 10_000_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[3]
    return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[4]


def _graph_replay_profiler_evidence_blockers(
    scheduler_metadata: Mapping[str, Any], profiler: Mapping[str, Any]
) -> list[str]:
    graph_stats = scheduler_metadata.get("graph_bucket_stats")
    if not isinstance(graph_stats, Mapping):
        return []
    replay_kernel_hits = graph_stats.get("replay_kernel_hits")
    if isinstance(replay_kernel_hits, int) and not isinstance(replay_kernel_hits, bool):
        if replay_kernel_hits <= 0:
            return []
    else:
        hits = graph_stats.get("hits")
        if not isinstance(hits, int) or isinstance(hits, bool) or hits <= 0:
            return []
    blockers: list[str] = []
    duration_categories = profiler.get("kernel_duration_categories_ns")
    graph_replay_duration = duration_categories.get("graph_replay") if isinstance(duration_categories, Mapping) else None
    if not _is_finite_positive_number(graph_replay_duration):
        blockers.append(
            "profiler.kernel_duration_categories_ns.graph_replay must be positive when graph_bucket_stats.hits is positive"
        )
    category_shares = profiler.get("kernel_duration_category_shares")
    graph_replay_share = category_shares.get("graph_replay") if isinstance(category_shares, Mapping) else None
    if not _is_finite_positive_number(graph_replay_share):
        blockers.append(
            "profiler.kernel_duration_category_shares.graph_replay must be positive when graph_bucket_stats.hits is positive"
        )
    expected_kernel_names = profiler.get("expected_kernel_names")
    if isinstance(expected_kernel_names, list) and not any(
        isinstance(kernel_name, str) and _profiler_kernel_duration_category(kernel_name) == "graph_replay"
        for kernel_name in expected_kernel_names
    ):
        blockers.append(
            "profiler.expected_kernel_names must include a graph/replay kernel when graph_bucket_stats.hits is positive"
        )
    kernel_durations = profiler.get("kernel_durations_ns")
    if isinstance(kernel_durations, Mapping):
        has_graph_replay_kernel_duration = any(
            isinstance(kernel_name, str)
            and _profiler_kernel_duration_category(kernel_name) == "graph_replay"
            and _is_finite_positive_number(duration_ns)
            for kernel_name, duration_ns in kernel_durations.items()
        )
        if not has_graph_replay_kernel_duration:
            blockers.append(
                "profiler.kernel_durations_ns must include a positive graph/replay duration when graph_bucket_stats.hits is positive"
            )
        histogram = graph_stats.get("kernel_time_histogram_ns") if isinstance(graph_stats, Mapping) else None
        if isinstance(histogram, Mapping):
            profiler_integer_duration_count = 0
            expected_bucket_counts: dict[str, int] = {}
            for kernel_name, duration_ns in kernel_durations.items():
                if (
                    not isinstance(kernel_name, str)
                    or not kernel_name
                    or _has_disallowed_profiler_kernel_name_fragment(kernel_name)
                    or not _is_finite_positive_number(duration_ns)
                    or not float(duration_ns).is_integer()
                ):
                    continue
                profiler_integer_duration_count += 1
                bucket = _graph_kernel_time_histogram_bucket_ns(int(duration_ns))
                expected_bucket_counts[bucket] = expected_bucket_counts.get(bucket, 0) + 1
            histogram_total = 0
            histogram_counts_valid = True
            for count in histogram.values():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    histogram_counts_valid = False
                    break
                histogram_total += int(count)
            if histogram_counts_valid:
                if profiler_integer_duration_count > 0 and histogram_total < profiler_integer_duration_count:
                    blockers.append(
                        "execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns observation count must cover profiler.kernel_durations_ns"
                    )
                for bucket, expected_count in sorted(expected_bucket_counts.items()):
                    observed_count = histogram.get(bucket, 0)
                    if isinstance(observed_count, bool) or not isinstance(observed_count, int) or observed_count < expected_count:
                        blockers.append(
                            "execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns bucket counts must cover profiler.kernel_durations_ns"
                        )
                        break
    return blockers


def _summarize_samples(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(sample) for sample in samples]
    if not values:
        return {"samples": [], "median": None, "p95": None, "min": None, "max": None, "stdev": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": values,
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _request_id_sort_key(request_id: Any) -> tuple[int, int | str]:
    text = str(request_id)
    return (0, int(text)) if text.isdigit() else (1, text)


def _all_finite(rows: Iterable[dict[str, Any]]) -> bool:
    return all(math.isfinite(float(row["logit"])) for row in rows)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_finite_rate_or_none(value: Any) -> float | None:
    if not _is_number(value):
        return None
    rate = float(value)
    return rate if math.isfinite(rate) and rate > 0.0 else None


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    num = _positive_finite_rate_or_none(numerator)
    denom = _positive_finite_rate_or_none(denominator)
    if num is None or denom is None:
        return None
    return num / denom


def _extract_decode_rates(payload: Mapping[str, Any]) -> tuple[float | None, float | None]:
    measurements = payload.get("measurements")
    aggregate = None
    per_request = None
    if isinstance(measurements, Mapping):
        if _is_number(measurements.get("decode_tok_s_aggregate")):
            aggregate = float(measurements["decode_tok_s_aggregate"])
        if _is_number(measurements.get("decode_tok_s_per_request")):
            per_request = float(measurements["decode_tok_s_per_request"])
    throughput = payload.get("throughput")
    if isinstance(throughput, Mapping) and _is_number(throughput.get("warmed_decode_tok_s")):
        aggregate = float(throughput["warmed_decode_tok_s"])
        per_request = float(throughput["warmed_decode_tok_s"])
    workload = payload.get("workload")
    if aggregate is not None and per_request is None and isinstance(workload, Mapping):
        concurrency = workload.get("concurrency")
        if isinstance(concurrency, int) and concurrency > 0:
            per_request = aggregate / concurrency
    return aggregate, per_request


def _visible_device_env_assignments(payload: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    hardware = payload.get("hardware")
    visible_device = hardware.get("visible_device") if isinstance(hardware, Mapping) else None
    env = visible_device.get("env") if isinstance(visible_device, Mapping) else None
    assignments: dict[str, str] = {}
    reasons: list[str] = []
    gpu_name = hardware.get("gpu") if isinstance(hardware, Mapping) else None
    device_name = visible_device.get("device_name") if isinstance(visible_device, Mapping) else None
    if isinstance(gpu_name, str) and gpu_name and isinstance(device_name, str) and device_name and gpu_name != device_name:
        reasons.append("hardware.gpu does not match hardware.visible_device.device_name")
    if not isinstance(env, Mapping):
        return assignments, reasons
    for key in _COMMAND_ENV_KEYS:
        value = env.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            reasons.append(f"hardware.visible_device.env.{key} is not a non-empty string when present")
        elif not value.strip():
            reasons.append(f"hardware.visible_device.env.{key} is not a non-blank string when present")
        else:
            assignments[key] = value
    return assignments, reasons


def _scaling_reference_software_reasons(payload: Mapping[str, Any]) -> list[str]:
    software = payload.get("software")
    if not isinstance(software, Mapping):
        return ["software provenance is missing for device-stamped scaling reference"]
    reasons: list[str] = []
    for field in ("python", "hipcc_version", "hipengine_commit"):
        value = software.get(field)
        if not isinstance(value, str) or not value:
            reasons.append(f"software.{field} is missing or not a non-empty string")
    dirty = software.get("hipengine_dirty")
    if not isinstance(dirty, bool):
        reasons.append("software.hipengine_dirty is missing or not a bool")
    return reasons


def _scaling_reference_command_env_assignments(
    payload: Mapping[str, Any],
    *,
    required_env: Mapping[str, str] | None = None,
    require_command_label: bool = False,
    expected_command_script: str | None = None,
    expected_inputs: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    commands = payload.get("commands")
    command = commands.get("benchmark") if isinstance(commands, Mapping) else payload.get("command")
    if command is None:
        reasons = [
            f"commands.benchmark is missing while retained command env sets {key}"
            for key, value in (required_env or {}).items()
            if require_command_label and value
        ]
        return {}, reasons
    if not isinstance(command, str) or not command:
        return {}, ["commands.benchmark is not a non-empty string when present"]
    try:
        argv = shlex.split(command)
    except ValueError:
        return {}, ["commands.benchmark is not shell-parseable"]
    raw_assignments = _command_device_env_assignments(argv)
    assignments: dict[str, str] = {}
    reasons: list[str] = []
    for key, value in raw_assignments.items():
        if not value.strip():
            reasons.append(f"commands.benchmark device env {key} is not a non-blank string when present")
        else:
            assignments[key] = value
    reasons.extend(
        f"commands.benchmark device env {key} is missing while retained command env sets it"
        for key, value in (required_env or {}).items()
        if value and key not in raw_assignments
    )
    if expected_command_script is not None:
        launch = _strip_command_env_prefix(argv)
        if len(launch) < 2 or not Path(launch[0]).name.startswith("python") or launch[1] != expected_command_script:
            reasons.append(f"commands.benchmark must launch {expected_command_script}")
    for label, flag in (("model", "--model"), ("fixture", "--fixture")):
        expected_value = (expected_inputs or {}).get(label)
        if not expected_value:
            continue
        command_value = _command_arg_value(command, flag)
        if command_value is None:
            reasons.append(f"commands.benchmark must include {flag} matching retained {label}")
        elif command_value != expected_value:
            reasons.append(f"commands.benchmark {flag} does not match retained {label}")
    return assignments, reasons


def _scaling_reference(
    path: Path | None,
    *,
    default_workload_concurrency: int | None = None,
    expected_command_script: str | None = None,
    expected_inputs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if path is None:
        return {
            "artifact_path": None,
            "status": "missing",
            "decode_tok_s_aggregate": None,
            "decode_tok_s_per_request": None,
            "reason": "no artifact path provided",
        }
    path = Path(path)
    if not path.exists():
        return {
            "artifact_path": str(path),
            "status": "missing",
            "decode_tok_s_aggregate": None,
            "decode_tok_s_per_request": None,
            "reason": "artifact path does not exist",
        }
    try:
        payload = _load_json_path(path)
    except Exception as exc:
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "decode_tok_s_aggregate": None,
            "decode_tok_s_per_request": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "decode_tok_s_aggregate": None,
            "decode_tok_s_per_request": None,
            "reason": "artifact root is not an object",
        }
    reasons: list[str] = []
    reference_artifact_path = None
    source_artifact_path = payload.get("artifact_path")
    if not isinstance(source_artifact_path, str) or not source_artifact_path:
        reasons.append("artifact_path is missing or not a non-empty string")
    elif source_artifact_path != str(path):
        reference_artifact_path = source_artifact_path
        reasons.append("artifact_path does not match scaling reference artifact path")
    else:
        reference_artifact_path = source_artifact_path
    reference_device_env, reference_device_env_reasons = _visible_device_env_assignments(payload)
    reasons.extend(reference_device_env_reasons)
    retained_device_env = _current_command_device_env_assignments()
    if reference_device_env:
        for key, value in reference_device_env.items():
            if retained_device_env.get(key) != value:
                reasons.append(f"hardware.visible_device.env.{key} does not match retained command env")
    reference_command_env, reference_command_env_reasons = _scaling_reference_command_env_assignments(
        payload,
        required_env=retained_device_env,
        require_command_label=bool(reference_device_env),
        expected_command_script=expected_command_script,
        expected_inputs=expected_inputs,
    )
    reasons.extend(reference_command_env_reasons)
    if reference_command_env:
        for key, value in reference_command_env.items():
            if retained_device_env.get(key) != value:
                reasons.append(f"commands.benchmark device env {key} does not match retained command env")
        if not reasons:
            for key, value in reference_command_env.items():
                if retained_device_env.get(key) == value and key not in reference_device_env:
                    reasons.append(f"hardware.visible_device.env.{key} is missing while benchmark command env sets it")
    if reference_device_env and not reasons:
        reasons.extend(_scaling_reference_software_reasons(payload))
    aggregate, per_request = _extract_decode_rates(payload)
    throughput_missing = aggregate is None or per_request is None
    workload = payload.get("workload")
    workload_concurrency = None
    prompt_tokens_per_request = None
    gen_tokens_per_request = None
    if isinstance(workload, Mapping):
        concurrency = workload.get("concurrency")
        if isinstance(concurrency, int) and not isinstance(concurrency, bool):
            workload_concurrency = concurrency
        prompt_tokens = workload.get("prompt_tokens_per_request", workload.get("prompt_length"))
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            prompt_tokens_per_request = prompt_tokens
        gen_tokens = workload.get("gen_tokens_per_request", workload.get("decode_tokens"))
        if isinstance(gen_tokens, int) and not isinstance(gen_tokens, bool):
            gen_tokens_per_request = gen_tokens
    if prompt_tokens_per_request is None:
        prompt_tokens = payload.get("prompt_length")
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            prompt_tokens_per_request = prompt_tokens
    if gen_tokens_per_request is None:
        gen_tokens = payload.get("decode_tokens")
        if isinstance(gen_tokens, int) and not isinstance(gen_tokens, bool):
            gen_tokens_per_request = gen_tokens
    if workload_concurrency is None and default_workload_concurrency is not None:
        workload_concurrency = int(default_workload_concurrency)
    status = str(payload.get("status") or "loaded")
    if status in _UNUSABLE_SCALING_REFERENCE_STATUSES:
        reasons.append(f"status={status!r} is not usable as a scaling reference")
    reference_reason = payload.get("reason")
    if reference_reason is not None:
        reasons.append(f"scaling reference reason is non-null: {reference_reason}")
    if throughput_missing:
        reasons.append("decode throughput fields missing")
    elif not _is_finite_positive_number(aggregate) or not _is_finite_positive_number(per_request):
        reasons.append("decode throughput fields must be positive finite numbers")
    elif workload_concurrency is not None:
        expected_aggregate = float(per_request) * int(workload_concurrency)
        if abs(float(aggregate) - expected_aggregate) > max(1e-9, expected_aggregate * 1e-6):
            reasons.append("decode aggregate rate does not match per-request rate times concurrency")
    if reasons:
        aggregate = None
        per_request = None
    reason = None if not reasons else "; ".join(reasons)
    return {
        "artifact_path": str(path),
        "reference_artifact_path": reference_artifact_path,
        "status": status,
        "run_tag": payload.get("run_tag"),
        "workload_concurrency": workload_concurrency,
        "prompt_tokens_per_request": prompt_tokens_per_request,
        "gen_tokens_per_request": gen_tokens_per_request,
        "decode_tok_s_aggregate": aggregate,
        "decode_tok_s_per_request": per_request,
        "reason": reason,
    }


def _primitive_device_metadata_blockers(device: Any) -> list[str]:
    if not isinstance(device, Mapping):
        return ["device metadata is missing or not an object"]
    blockers: list[str] = []
    env = device.get("env")
    if not isinstance(env, Mapping):
        blockers.append("device.env is missing or not an object")
    else:
        for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            value = env.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                blockers.append(f"device.env.{key} is not a non-empty string when present")
    for field in ("hipGetDeviceCount_error", "hipGetDevice_error", "hipDeviceGetName_error"):
        value = device.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            blockers.append(f"device.{field} is missing or not integer zero")
    visible_count = device.get("visible_device_count")
    if not isinstance(visible_count, int) or isinstance(visible_count, bool) or visible_count <= 0:
        blockers.append("device.visible_device_count is missing or not a positive int")
    current_device = device.get("current_device")
    if not isinstance(current_device, int) or isinstance(current_device, bool) or current_device < 0:
        blockers.append("device.current_device is missing or not a non-negative int")
    device_name = device.get("device_name")
    if not isinstance(device_name, str) or not device_name:
        blockers.append("device.device_name is missing or not a non-empty string")
    return blockers


def _primitive_correctness_reference(path: Path | None, *, rows: int) -> dict[str, Any]:
    if path is None:
        return {
            "artifact_path": None,
            "status": "missing",
            "passed": False,
            "reason": "no primitive correctness artifact path provided",
        }
    path = Path(path)
    if not path.exists():
        return {
            "artifact_path": str(path),
            "status": "missing",
            "passed": False,
            "reason": "artifact path does not exist",
        }
    try:
        payload = _load_json_path(path)
    except Exception as exc:
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "passed": False,
            "reason": "artifact root is not an object",
        }
    reasons: list[str] = []
    artifact_schema = payload.get("schema")
    if not isinstance(artifact_schema, int) or isinstance(artifact_schema, bool) or artifact_schema != 1:
        reasons.append("schema is missing or not 1")
    source_artifact_path = payload.get("artifact_path")
    if not isinstance(source_artifact_path, str) or not source_artifact_path:
        source_artifact_path = None
        reasons.append("artifact_path is missing or not a non-empty string")
    elif source_artifact_path != str(path):
        reasons.append("artifact_path does not match primitive correctness artifact path")
    artifact_rows = payload.get("rows")
    if not isinstance(artifact_rows, int) or isinstance(artifact_rows, bool) or artifact_rows != int(rows):
        reasons.append(f"artifact rows={artifact_rows!r} does not match batch_size={rows}")
    artifact_seed = payload.get("seed")
    if (
        not isinstance(artifact_seed, int)
        or isinstance(artifact_seed, bool)
        or artifact_seed != _REQUIRED_PRIMITIVE_CORRECTNESS_SEED
    ):
        reasons.append("seed is missing or not 1234")
    for field, expected_value in _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS.items():
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected_value:
            reasons.append(f"{field} is missing or not {expected_value}")
    if not _primitive_context_lens_matches(payload.get("context_lens"), int(rows)):
        reasons.append("context_lens is missing or does not match fixture coverage")
    if payload.get("passed") is not True:
        reasons.append("primitive correctness payload did not pass")
    for field in ("append_key_mismatch", "append_value_mismatch"):
        if not _is_zero_int(payload.get(field)):
            reasons.append(f"{field} is missing or not integer zero")
    for field in ("append_batch_aa_key_mismatch", "append_batch_aa_value_mismatch"):
        if not _is_zero_int(payload.get(field)):
            reasons.append(f"{field} is missing or not integer zero")
    attn_batch_aa = payload.get("attn_batch_aa_max_abs")
    if not _is_number(attn_batch_aa) or float(attn_batch_aa) != 0.0:
        reasons.append("attn_batch_aa_max_abs is missing or not 0.0")
    if payload.get("aa_passed") is not True:
        reasons.append("aa_passed is not true")
    reasons.extend(_primitive_device_metadata_blockers(payload.get("device")))
    attn_vs_c1 = payload.get("attn_batch_vs_c1_max_abs")
    if not _is_number(attn_vs_c1) or float(attn_vs_c1) != 0.0:
        reasons.append("attn_batch_vs_c1_max_abs is missing or not 0.0")
    attn_vs_numpy = payload.get("attn_batch_vs_numpy_max_abs")
    if (
        not _is_number(attn_vs_numpy)
        or not math.isfinite(float(attn_vs_numpy))
        or float(attn_vs_numpy) < 0.0
        or float(attn_vs_numpy) > _PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT
    ):
        reasons.append("attn_batch_vs_numpy_max_abs is missing, non-finite, negative, or above 2e-5")
    return {
        "artifact_path": str(path),
        "source_artifact_path": source_artifact_path,
        "status": "loaded",
        "schema": payload.get("schema"),
        "rows": payload.get("rows"),
        "seed": payload.get("seed"),
        "block_size": payload.get("block_size"),
        "max_context_len": payload.get("max_context_len"),
        "num_q_heads": payload.get("num_q_heads"),
        "num_kv_heads": payload.get("num_kv_heads"),
        "head_dim": payload.get("head_dim"),
        "context_lens": payload.get("context_lens"),
        "passed": not reasons,
        "append_key_mismatch": payload.get("append_key_mismatch"),
        "append_value_mismatch": payload.get("append_value_mismatch"),
        "append_batch_aa_key_mismatch": payload.get("append_batch_aa_key_mismatch"),
        "append_batch_aa_value_mismatch": payload.get("append_batch_aa_value_mismatch"),
        "attn_batch_vs_c1_max_abs": attn_vs_c1,
        "attn_batch_vs_numpy_max_abs": payload.get("attn_batch_vs_numpy_max_abs"),
        "attn_batch_aa_max_abs": attn_batch_aa,
        "aa_passed": payload.get("aa_passed"),
        "device": payload.get("device"),
        "reason": None if not reasons else "; ".join(reasons),
    }


def _int8_kv_primitive_layer_accuracy_reference(
    path: Path | None,
    *,
    device: str,
    prompt_length: int,
    scale_dtype: str,
) -> dict[str, Any]:
    if path is None:
        return {
            "artifact_path": None,
            "status": "missing",
            "device": device,
            "passed": False,
            "reason": f"no INT8 KV primitive {device} artifact path provided",
        }
    path = Path(path)
    if not path.exists():
        return {
            "artifact_path": str(path),
            "status": "missing",
            "device": device,
            "passed": False,
            "reason": "artifact path does not exist",
        }
    try:
        payload = _load_json_path(path)
    except Exception as exc:
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "device": device,
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "device": device,
            "passed": False,
            "reason": "artifact root is not an object",
        }

    reasons: list[str] = []
    if payload.get("schema") != 1 or isinstance(payload.get("schema"), bool):
        reasons.append("schema is missing or not 1")
    if payload.get("status") != "accepted":
        reasons.append("status is not accepted")
    if payload.get("passed") is not True:
        reasons.append("payload did not pass")
    if payload.get("mode") != "qwen35_kv_int8_layer_accuracy":
        reasons.append("mode is not qwen35_kv_int8_layer_accuracy")
    if payload.get("device") != device:
        reasons.append(f"device is not {device}")
    source_artifact_path = payload.get("artifact_path")
    if not isinstance(source_artifact_path, str) or not source_artifact_path:
        source_artifact_path = None
        reasons.append("artifact_path is missing or not a non-empty string")
    elif source_artifact_path != str(path):
        reasons.append("artifact_path does not match INT8 KV primitive artifact path")
    source_self_path = payload.get("source_artifact_path")
    if not isinstance(source_self_path, str) or not source_self_path:
        source_self_path = None
        reasons.append("source_artifact_path is missing or not a non-empty string")
    elif source_self_path != str(path):
        reasons.append("source_artifact_path does not match INT8 KV primitive artifact path")

    shape = payload.get("shape")
    if not isinstance(shape, Mapping):
        reasons.append("shape is missing or not an object")
        shape = {}
    expected_contexts = [int(prompt_length), int(prompt_length) + 1]
    if shape.get("contexts") != expected_contexts:
        reasons.append(f"shape.contexts does not match {expected_contexts}")
    for field, expected_value in (("block_size", 256), ("num_q_heads", 16), ("num_kv_heads", 2), ("head_dim", 256)):
        if shape.get(field) != expected_value or isinstance(shape.get(field), bool):
            reasons.append(f"shape.{field} is missing or not {expected_value}")
    if shape.get("scale_dtype") != str(scale_dtype):
        reasons.append(f"shape.scale_dtype is not {scale_dtype}")

    kv_policy = payload.get("kv_policy")
    if not isinstance(kv_policy, Mapping):
        reasons.append("kv_policy is missing or not an object")
    else:
        if kv_policy.get("storage_dtype") != "int8_per_token_head":
            reasons.append("kv_policy.storage_dtype is not int8_per_token_head")
        scale_format = kv_policy.get("scale_metadata_format")
        if not isinstance(scale_format, Mapping) or scale_format.get("scale_dtype") != str(scale_dtype):
            reasons.append(f"kv_policy.scale_metadata_format.scale_dtype is not {scale_dtype}")

    for field in ("blocked_reasons", "correctness_failures"):
        value = payload.get(field)
        if not isinstance(value, list):
            reasons.append(f"{field} is missing or not a list")
        elif value:
            reasons.append(f"{field} is not empty")
    command = payload.get("command")
    if not isinstance(command, str) or "scripts/qwen35_kv_int8_accuracy.py" not in command:
        reasons.append("command is missing scripts/qwen35_kv_int8_accuracy.py")
    elif f"--device {device}" not in command or f"--contexts {prompt_length},{prompt_length + 1}" not in command:
        reasons.append("command does not match retained INT8 KV primitive device/contexts")
    if device == "hip" and isinstance(command, str) and "--require-int8-hip" not in command:
        reasons.append("HIP INT8 KV primitive command is missing --require-int8-hip")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        reasons.append("cases is missing or empty")
    else:
        for case_index, case in enumerate(cases):
            if not isinstance(case, Mapping):
                reasons.append(f"cases[{case_index}] is not an object")
                continue
            paths = case.get("paths")
            int8_path = paths.get("int8_per_token_head") if isinstance(paths, Mapping) else None
            if not isinstance(int8_path, Mapping) or int8_path.get("passed") is not True:
                reasons.append(f"cases[{case_index}].paths.int8_per_token_head did not pass")

    return {
        "artifact_path": str(path),
        "source_artifact_path": source_self_path,
        "status": "loaded",
        "artifact_status": payload.get("status"),
        "schema": payload.get("schema"),
        "mode": payload.get("mode"),
        "device": payload.get("device"),
        "command": payload.get("command"),
        "passed": not reasons,
        "shape": payload.get("shape"),
        "kv_policy": payload.get("kv_policy"),
        "blocked_reasons": payload.get("blocked_reasons"),
        "correctness_failures": payload.get("correctness_failures"),
        "reason": None if not reasons else "; ".join(reasons),
    }


def _int8_kv_primitive_layer_accuracy_blockers(reference: Mapping[str, Any], *, label: str) -> list[str]:
    if reference.get("passed") is True:
        return []
    reason = reference.get("reason")
    return [f"INT8 KV primitive {label} gate did not pass: {reason}"]


def _is_finite_nonnegative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_finite_positive_number(value: Any) -> bool:
    return _is_finite_nonnegative_number(value) and float(value) > 0.0


def _is_stripped_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _is_retained_artifact_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if (
        path.is_absolute()
        or len(path.parts) < 3
        or path.parts[:2] != ("benchmarks", "results")
        or ".." in path.parts
    ):
        return False
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    try:
        return (Path.cwd() / path).resolve().is_relative_to(results_root)
    except OSError:
        return False


def _retained_json_artifact_path_blockers(label: str, path: Any) -> list[str]:
    if path is None:
        return [f"{label} must be provided under benchmarks/results"]
    path_text = str(path)
    blockers: list[str] = []
    if not _is_retained_artifact_path(path_text):
        blockers.append(f"{label} must be a repo-relative path under benchmarks/results")
    if Path(path_text).suffix.lower() != ".json":
        blockers.append(f"{label} must point to a .json artifact")
    return blockers


def _retained_output_artifact_blockers(path: Any) -> list[str]:
    return _retained_json_artifact_path_blockers("artifact_path", path)


def _cached_build_provenance_blockers(args: argparse.Namespace) -> list[str]:
    blockers: list[str] = []
    compiler_version_file = getattr(args, "compiler_version_file", None)
    if compiler_version_file is None or not str(compiler_version_file).strip():
        blockers.append("compiler_version_file must be provided for retained promotion")
    elif not _is_retained_artifact_path(str(compiler_version_file)):
        blockers.append("compiler_version_file must be a repo-relative path under benchmarks/results")
    if getattr(args, "require_cached_build", False) is not True:
        blockers.append("require_cached_build must be true for retained promotion")
    return blockers


def _retained_command_label_provenance_blockers(args: argparse.Namespace, command: str) -> list[str]:
    blockers: list[str] = []
    for field, flag in (("model", "--model"), ("fixture", "--fixture")):
        command_value = _command_arg_value(command, flag)
        expected_value = str(getattr(args, field, ""))
        if not command_value:
            blockers.append(f"commands.benchmark must include {flag} for retained promotion")
        elif expected_value and command_value != expected_value:
            blockers.append(f"commands.benchmark {flag} must match retained {field}")
    try:
        parts = shlex.split(command)
    except ValueError:
        blockers.append("commands.benchmark must be shell-parseable for retained promotion")
        return blockers
    command_device_env = _command_device_env_assignments(parts)
    blank_command_device_env_keys = {key for key, value in command_device_env.items() if not value.strip()}
    for key in sorted(blank_command_device_env_keys):
        blockers.append(f"commands.benchmark {key} must be non-blank for retained promotion")
    for key, expected_value in _current_command_device_env_assignments().items():
        command_value = command_device_env.get(key)
        if command_value is None:
            blockers.append(f"commands.benchmark must include {key}={expected_value} for retained promotion")
        elif key not in blank_command_device_env_keys and command_value != expected_value:
            blockers.append(f"commands.benchmark {key} must match retained command env")
    return blockers


def _retained_command_provenance_blockers(args: argparse.Namespace, argv: Sequence[str] | None) -> list[str]:
    return _retained_command_label_provenance_blockers(args, _command(argv))


def _synthesized_profiler_total_kernel_duration(profiler: Mapping[str, Any]) -> float | None:
    kernel_durations = profiler.get("kernel_durations_ns")
    if not isinstance(kernel_durations, Mapping):
        return None
    total = 0.0
    saw_duration = False
    for duration_ns in kernel_durations.values():
        if not _is_finite_positive_number(duration_ns):
            continue
        total += float(duration_ns)
        saw_duration = True
    return total if saw_duration else None


def _synthesized_profiler_kernel_duration_shares(profiler: Mapping[str, Any]) -> dict[str, float] | None:
    kernel_durations = profiler.get("kernel_durations_ns")
    total_duration = profiler.get("total_kernel_duration_ns")
    if not isinstance(kernel_durations, Mapping) or not _is_finite_positive_number(total_duration):
        return None
    shares = {
        str(kernel_name): float(duration_ns) / float(total_duration)
        for kernel_name, duration_ns in kernel_durations.items()
        if isinstance(kernel_name, str) and kernel_name and _is_finite_positive_number(duration_ns)
    }
    return shares or None


def _has_disallowed_profiler_kernel_name_fragment(kernel_name: str) -> bool:
    lowered = kernel_name.lower()
    return any(fragment in lowered for fragment in _DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS)


def _has_native_batch_profiler_kernel_name(kernel_names: list[Any]) -> bool:
    return any(_is_stripped_non_empty_string(name) and "batch" in name.lower() for name in kernel_names)


def _profiler_kernel_duration_category(kernel_name: str) -> str:
    lowered = kernel_name.lower()
    if "graph" in lowered or "replay" in lowered:
        return "graph_replay"
    if "moe" in lowered or "expert" in lowered or "router" in lowered:
        return "moe"
    if "attn" in lowered or "attention" in lowered or "paged" in lowered or "kv" in lowered:
        return "attention"
    if "lm_head" in lowered or "sample" in lowered or "argmax" in lowered:
        return "sampling"
    projection_fragments = ("projection", "linear", "matmul", "gemm", "gemv", "mmq", "wmma")
    if any(fragment in lowered for fragment in projection_fragments):
        return "projection"
    return "other"


def _synthesized_profiler_kernel_duration_categories(profiler: Mapping[str, Any]) -> dict[str, float] | None:
    kernel_durations = profiler.get("kernel_durations_ns")
    if not isinstance(kernel_durations, Mapping):
        return None
    categories = dict.fromkeys(_PROFILER_KERNEL_DURATION_CATEGORIES, 0.0)
    saw_duration = False
    for kernel_name, duration_ns in kernel_durations.items():
        if not isinstance(kernel_name, str) or not kernel_name or not _is_finite_positive_number(duration_ns):
            continue
        categories[_profiler_kernel_duration_category(kernel_name)] += float(duration_ns)
        saw_duration = True
    return categories if saw_duration else None


def _synthesized_profiler_kernel_duration_category_shares(profiler: Mapping[str, Any]) -> dict[str, float] | None:
    duration_categories = profiler.get("kernel_duration_categories_ns")
    total_duration = profiler.get("total_kernel_duration_ns")
    if not isinstance(duration_categories, Mapping) or not _is_finite_positive_number(total_duration):
        return None
    return {
        category: float(duration_categories.get(category, 0.0)) / float(total_duration)
        for category in _PROFILER_KERNEL_DURATION_CATEGORIES
    }


def _cpu_side_bottlenecks_from_bench(bench: Mapping[str, Any]) -> dict[str, float] | None:
    durations = {
        "load": bench.get("load_seconds"),
        "prefill": bench.get("prefill_seconds"),
        "warmup_decode": bench.get("warmup_seconds"),
        "decode": bench.get("decode_seconds"),
        "validation": 0.0,
        "other": 0.0,
    }
    if not all(_is_finite_nonnegative_number(duration) for duration in durations.values()):
        return None
    return {category: float(durations[category]) for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES}


def _attach_profiler_cpu_side_bottlenecks(profiler: Mapping[str, Any], bench: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(profiler)
    if "cpu_side_bottlenecks_seconds" not in result:
        durations = _cpu_side_bottlenecks_from_bench(bench)
        if durations is not None:
            result["cpu_side_bottlenecks_seconds"] = durations
    durations = result.get("cpu_side_bottlenecks_seconds")
    if "cpu_side_total_seconds" not in result and isinstance(durations, Mapping):
        total_seconds = sum(
            float(duration_seconds)
            for duration_seconds in durations.values()
            if _is_finite_nonnegative_number(duration_seconds)
        )
        if total_seconds > 0.0:
            result["cpu_side_total_seconds"] = total_seconds
    total_seconds = result.get("cpu_side_total_seconds")
    if (
        "cpu_side_bottleneck_shares" not in result
        and isinstance(durations, Mapping)
        and _is_finite_positive_number(total_seconds)
    ):
        result["cpu_side_bottleneck_shares"] = {
            category: float(durations.get(category, 0.0)) / float(total_seconds)
            for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES
        }
    return result


def _command_arg_value(command: str, flag: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    prefix = f"{flag}="
    for idx, part in enumerate(parts):
        if part == flag and idx + 1 < len(parts):
            return parts[idx + 1]
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def _profiler_command_label(profiler: Mapping[str, Any], payload: Mapping[str, Any] | None) -> str | None:
    for source in (profiler, payload):
        if not isinstance(source, Mapping):
            continue
        for key in ("command", "profiler_command"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
        commands = source.get("commands")
        if isinstance(commands, Mapping):
            value = commands.get("profiler")
            if isinstance(value, str) and value:
                return value
    return None


def _resolve_profiler_trace_file(trace_file: str, *, profiler_path: Path) -> Path:
    path = Path(trace_file)
    if path.is_absolute():
        return path
    parent_relative = profiler_path.parent / path
    if parent_relative.exists():
        return parent_relative
    return path


def _profiler_trace_row_kernel_name(row: Mapping[str, Any]) -> str:
    for column in _PROFILER_TRACE_KERNEL_NAME_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _profiler_trace_row_duration_ns(row: Mapping[str, Any]) -> float | None:
    for column in _PROFILER_TRACE_DURATION_COLUMNS:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0.0 and math.isfinite(duration):
            return duration
    start = None
    end = None
    for column in _PROFILER_TRACE_START_COLUMNS:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            start = float(value)
            break
        except (TypeError, ValueError):
            continue
    for column in _PROFILER_TRACE_END_COLUMNS:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            end = float(value)
            break
        except (TypeError, ValueError):
            continue
    if start is None or end is None:
        return None
    duration = end - start
    return duration if duration > 0.0 and math.isfinite(duration) else None


def _read_profiler_trace_kernel_names(trace_file: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    try:
        with trace_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = _profiler_trace_row_kernel_name(row)
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
    except OSError:
        return []
    return names


def _read_profiler_trace_kernel_durations(trace_file: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    try:
        with trace_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = _profiler_trace_row_kernel_name(row)
                duration_ns = _profiler_trace_row_duration_ns(row)
                if name and duration_ns is not None:
                    durations[name] = durations.get(name, 0.0) + duration_ns
    except OSError:
        return {}
    return durations


def _synthesized_profiler_trace_kernel_names(profiler: Mapping[str, Any], *, profiler_path: Path) -> list[str] | None:
    trace_files = profiler.get("trace_files")
    if not isinstance(trace_files, list) or not trace_files:
        return None
    names: list[str] = []
    seen: set[str] = set()
    for trace_file in trace_files:
        if not isinstance(trace_file, str) or not trace_file:
            continue
        for kernel_name in _read_profiler_trace_kernel_names(_resolve_profiler_trace_file(trace_file, profiler_path=profiler_path)):
            if kernel_name not in seen:
                names.append(kernel_name)
                seen.add(kernel_name)
    return names or None


def _synthesized_profiler_kernel_durations_from_traces(profiler: Mapping[str, Any], *, profiler_path: Path) -> dict[str, float] | None:
    trace_files = profiler.get("trace_files")
    if not isinstance(trace_files, list) or not trace_files:
        return None
    durations: dict[str, float] = {}
    for trace_file in trace_files:
        if not isinstance(trace_file, str) or not trace_file:
            continue
        for kernel_name, duration_ns in _read_profiler_trace_kernel_durations(
            _resolve_profiler_trace_file(trace_file, profiler_path=profiler_path)
        ).items():
            durations[kernel_name] = durations.get(kernel_name, 0.0) + duration_ns
    return durations or None


def _profiler_reference(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_captured", "notes": "E2E retained c>N row; profiler trace not captured in this iteration."}
    path = Path(path)
    if not path.exists():
        return {"artifact_path": str(path), "status": "missing", "reason": "artifact path does not exist"}
    try:
        payload = _load_json_path(path)
    except Exception as exc:
        return {"artifact_path": str(path), "status": "invalid_json", "reason": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, Mapping):
        return {"artifact_path": str(path), "status": "invalid_json", "reason": "artifact root is not an object"}
    profiler = payload.get("profiler") if isinstance(payload.get("profiler"), Mapping) else payload
    if not isinstance(profiler, Mapping):
        return {"artifact_path": str(path), "status": "invalid_json", "reason": "profiler summary is not an object"}
    result = dict(profiler)
    profiler_source_artifact_path = result.get("artifact_path")
    result["source_artifact_path"] = (
        profiler_source_artifact_path
        if isinstance(profiler_source_artifact_path, str) and profiler_source_artifact_path
        else None
    )
    result["artifact_path"] = str(path)
    synthesized_fields: set[str] = set()
    if "kernel_durations_ns" not in result:
        kernel_durations = _synthesized_profiler_kernel_durations_from_traces(result, profiler_path=path)
        if kernel_durations is not None:
            result["kernel_durations_ns"] = kernel_durations
            synthesized_fields.add("kernel_durations_ns")
    if "total_kernel_duration_ns" not in result:
        total_kernel_duration_ns = _synthesized_profiler_total_kernel_duration(result)
        if total_kernel_duration_ns is not None:
            result["total_kernel_duration_ns"] = total_kernel_duration_ns
            synthesized_fields.add("total_kernel_duration_ns")
    if "kernel_duration_shares" not in result:
        kernel_duration_shares = _synthesized_profiler_kernel_duration_shares(result)
        if kernel_duration_shares is not None:
            result["kernel_duration_shares"] = kernel_duration_shares
            synthesized_fields.add("kernel_duration_shares")
    if "kernel_duration_categories_ns" not in result:
        kernel_duration_categories_ns = _synthesized_profiler_kernel_duration_categories(result)
        if kernel_duration_categories_ns is not None:
            result["kernel_duration_categories_ns"] = kernel_duration_categories_ns
            synthesized_fields.add("kernel_duration_categories_ns")
    if "kernel_duration_category_shares" not in result:
        kernel_duration_category_shares = _synthesized_profiler_kernel_duration_category_shares(result)
        if kernel_duration_category_shares is not None:
            result["kernel_duration_category_shares"] = kernel_duration_category_shares
            synthesized_fields.add("kernel_duration_category_shares")
    if "trace_kernel_names" not in result:
        trace_kernel_names = _synthesized_profiler_trace_kernel_names(result, profiler_path=path)
        if trace_kernel_names is not None:
            result["trace_kernel_names"] = trace_kernel_names
            synthesized_fields.add("trace_kernel_names")
    profiler_command = _profiler_command_label(profiler, payload)
    if profiler_command is not None:
        if "output_format" not in result:
            output_format = _command_arg_value(profiler_command, _ROCPROF_COMMAND_FLAGS[1])
            if output_format is not None:
                result["output_format"] = output_format
                synthesized_fields.add("output_format")
        if "trace_dir" not in result:
            trace_dir = _command_arg_value(profiler_command, _ROCPROF_COMMAND_FLAGS[2])
            if trace_dir is not None:
                result["trace_dir"] = trace_dir
                synthesized_fields.add("trace_dir")
    if "synthesized_fields" not in result:
        result["synthesized_fields"] = [field for field in _PROFILER_SYNTHESIZED_FIELDS if field in synthesized_fields]
    result.setdefault("artifact_path", str(path))
    result.setdefault("status", "loaded")
    return result


def _payload_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)


def _profiled_command(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    profiler: Mapping[str, Any] | None = None,
) -> str | None:
    explicit = getattr(args, "profiler_command", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    if getattr(args, "profiler_json", None) is None:
        return None
    trace_dir = profiler.get("trace_dir") if isinstance(profiler, Mapping) else None
    if not isinstance(trace_dir, str) or not trace_dir:
        trace_dir = "<profile-dir>"
    return (
        f"{_ROCPROF_EXECUTABLE} {_ROCPROF_COMMAND_FLAGS[0]} {_ROCPROF_COMMAND_FLAGS[1]} {_ROCPROF_OUTPUT_FORMAT} "
        f"{_ROCPROF_COMMAND_FLAGS[2]} {shlex.quote(trace_dir)} -- {_command(argv)}"
    )


def _reference_with_retained_workload_shape(
    reference: Mapping[str, Any],
    *,
    expected_workload_concurrency: int | None,
    prompt_tokens_per_request: int | None,
    gen_tokens_per_request: int | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if expected_workload_concurrency is not None:
        value = reference.get("workload_concurrency")
        if not isinstance(value, int) or isinstance(value, bool):
            reasons.append("workload concurrency label is missing")
        elif value != expected_workload_concurrency:
            reasons.append("workload concurrency label does not match retained workload")
    if prompt_tokens_per_request is not None:
        value = reference.get("prompt_tokens_per_request")
        if not isinstance(value, int) or isinstance(value, bool):
            reasons.append("prompt token count label is missing")
        elif value != prompt_tokens_per_request:
            reasons.append("prompt token count label does not match retained workload")
    if gen_tokens_per_request is not None:
        value = reference.get("gen_tokens_per_request")
        if not isinstance(value, int) or isinstance(value, bool):
            reasons.append("decode token count label is missing")
        elif value != gen_tokens_per_request:
            reasons.append("decode token count label does not match retained workload")
    if not reasons:
        return dict(reference)
    result = dict(reference)
    prior_reason = result.get("reason")
    reason_parts = [str(prior_reason)] if prior_reason else []
    reason_parts.extend(reasons)
    result["reason"] = "; ".join(reason_parts)
    result["decode_tok_s_aggregate"] = None
    result["decode_tok_s_per_request"] = None
    return result


def _scaling_performance_blockers(scaling: Mapping[str, Any]) -> list[str]:
    """Return blockers for retained throughput claims that do not beat baselines."""

    ratios = scaling.get("ratios")
    if not isinstance(ratios, Mapping):
        return ["scaling.ratios must be present before a retained performance claim"]
    blockers: list[str] = []
    for field in ("aggregate_vs_c1", "aggregate_vs_serial_bridge"):
        value = ratios.get(field)
        if not _is_finite_positive_number(value):
            blockers.append(f"scaling.ratios.{field} must be positive numeric for retained performance claim")
        elif float(value) <= 1.0:
            blockers.append(f"scaling.ratios.{field} must be > 1.0 for retained performance claim")
    return blockers


def _build_scaling_comparison(
    args: argparse.Namespace,
    *,
    native_decode_tok_s_aggregate: float | None,
    native_decode_tok_s_per_request: float | None,
) -> dict[str, Any]:
    expected_model = str(getattr(args, "model", ""))
    expected_fixture = str(getattr(args, "fixture", ""))
    c1_expected_inputs = {"model": expected_model} if expected_model else {}
    serial_expected_inputs = dict(c1_expected_inputs)
    if expected_fixture:
        serial_expected_inputs["fixture"] = expected_fixture
    c1 = _scaling_reference(
        getattr(args, "c1_baseline_json", None),
        default_workload_concurrency=1,
        expected_command_script=_LEGACY_NATIVE_BENCH_SCRIPT,
        expected_inputs=c1_expected_inputs,
    )
    serial = _scaling_reference(
        getattr(args, "serial_bridge_json", None),
        expected_command_script=_SERIAL_BRIDGE_SCRIPT,
        expected_inputs=serial_expected_inputs,
    )
    prompt_tokens_per_request = getattr(args, "prompt_length", None)
    if isinstance(prompt_tokens_per_request, bool) or not isinstance(prompt_tokens_per_request, int):
        prompt_tokens_per_request = None
    gen_tokens_per_request = getattr(args, "decode_tokens", None)
    if isinstance(gen_tokens_per_request, bool) or not isinstance(gen_tokens_per_request, int):
        gen_tokens_per_request = None
    batch_size = getattr(args, "batch_size", None)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        batch_size = None
    c1 = _reference_with_retained_workload_shape(
        c1,
        expected_workload_concurrency=1,
        prompt_tokens_per_request=prompt_tokens_per_request,
        gen_tokens_per_request=gen_tokens_per_request,
    )
    serial = _reference_with_retained_workload_shape(
        serial,
        expected_workload_concurrency=batch_size,
        prompt_tokens_per_request=prompt_tokens_per_request,
        gen_tokens_per_request=gen_tokens_per_request,
    )
    native_decode_tok_s_aggregate = _positive_finite_rate_or_none(native_decode_tok_s_aggregate)
    native_decode_tok_s_per_request = _positive_finite_rate_or_none(native_decode_tok_s_per_request)
    ratios = {
        "aggregate_vs_c1": _safe_ratio(native_decode_tok_s_aggregate, c1.get("decode_tok_s_aggregate")),
        "per_request_vs_c1": _safe_ratio(native_decode_tok_s_per_request, c1.get("decode_tok_s_per_request")),
        "aggregate_vs_serial_bridge": _safe_ratio(native_decode_tok_s_aggregate, serial.get("decode_tok_s_aggregate")),
        "per_request_vs_serial_bridge": _safe_ratio(native_decode_tok_s_per_request, serial.get("decode_tok_s_per_request")),
    }
    complete = all(ratios.get(field) is not None for field in RETAINED_ARTIFACT_REQUIRED_SCALING_RATIOS)
    scaling = {
        "complete": complete,
        "native": {
            "decode_tok_s_aggregate": native_decode_tok_s_aggregate,
            "decode_tok_s_per_request": native_decode_tok_s_per_request,
        },
        "c1_baseline": c1,
        "serial_bridge_baseline": serial,
        "ratios": ratios,
    }
    assert all(field in scaling for field in RETAINED_ARTIFACT_REQUIRED_SCALING_BASELINES)
    return scaling


def _merged_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _merged_mapping(current, value)
        else:
            result[key] = value
    return result


def _allocator_memory_evidence(stats: Mapping[str, Any]) -> dict[str, Any]:
    evidence_stats = {
        key: int(value)
        for key, value in stats.items()
        if _is_finite_nonnegative_number(value)
    }
    peak = evidence_stats.get("peak_allocated_bytes")
    return {
        "allocator_reserved_peak_bytes": peak,
        "allocator_memory_stats": evidence_stats,
    }


def _stable_block_id_evidence(args: argparse.Namespace, bench: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(bench, Mapping):
        return {"passed": False, "audit": "native bench evidence is unavailable"}
    scheduler_metadata = bench.get("scheduler_metadata")
    batch_execution = bench.get("batch_execution")
    completed = bench.get("completed")
    if not isinstance(scheduler_metadata, Mapping):
        return {"passed": False, "audit": "scheduler metadata is unavailable"}
    if not isinstance(batch_execution, Mapping):
        return {"passed": False, "audit": "batch execution metadata is unavailable"}
    expected_count = getattr(args, "batch_size", None)
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        return {"passed": False, "audit": "batch_size is unavailable"}
    expected_request_ids = list(range(int(expected_count)))
    reasons: list[str] = []

    def _int_list(value: Any) -> list[int] | None:
        if not isinstance(value, list):
            return None
        result: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                return None
            result.append(int(item))
        return result

    request_ids = _int_list(scheduler_metadata.get("request_ids"))
    admitted = _int_list(scheduler_metadata.get("admitted"))
    if request_ids != expected_request_ids:
        reasons.append("scheduler request ids do not match expected stable ids")
    if admitted != expected_request_ids:
        reasons.append("admitted request ids do not match expected stable ids")
    slot_to_request_after_admit = scheduler_metadata.get("slot_to_request_after_admit")
    request_to_slot: dict[int, int] = {}
    if not isinstance(slot_to_request_after_admit, list) or len(slot_to_request_after_admit) < int(expected_count):
        reasons.append("slot_to_request_after_admit is missing or too short")
    else:
        for slot, request_id in enumerate(slot_to_request_after_admit):
            if request_id is None:
                continue
            if isinstance(request_id, bool) or not isinstance(request_id, int):
                reasons.append("slot_to_request_after_admit contains a non-integer request id")
                break
            rid = int(request_id)
            if rid in request_to_slot:
                reasons.append("slot_to_request_after_admit contains duplicate request ids")
                break
            request_to_slot[rid] = int(slot)
        if [request_to_slot.get(rid) for rid in expected_request_ids] != expected_request_ids:
            reasons.append("admitted request ids are not mapped to stable compact slots")
    active_count_after_admit = scheduler_metadata.get("active_count_after_admit")
    if active_count_after_admit != int(expected_count):
        reasons.append("active_count_after_admit does not match batch_size")

    prefill_slabs = scheduler_metadata.get("prefill_slabs")
    covered_prefill_requests: list[int] = []
    if not isinstance(prefill_slabs, list) or not prefill_slabs:
        reasons.append("prefill slab metadata is missing")
    else:
        for index, slab in enumerate(prefill_slabs):
            if not isinstance(slab, Mapping):
                reasons.append(f"prefill slab {index} is not an object")
                continue
            slab_request_ids = _int_list(slab.get("request_ids"))
            slab_slot_ids = _int_list(slab.get("slot_ids"))
            block_count = slab.get("block_count")
            if slab_request_ids is None or slab_slot_ids is None or len(slab_request_ids) != len(slab_slot_ids):
                reasons.append(f"prefill slab {index} request/slot ids are invalid")
                continue
            if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count <= 0:
                reasons.append(f"prefill slab {index} block_count is invalid")
            covered_prefill_requests.extend(slab_request_ids)
            for rid, slot in zip(slab_request_ids, slab_slot_ids, strict=True):
                if request_to_slot.get(rid) != slot:
                    reasons.append(f"prefill slab {index} slot does not match stable admitted slot")
                    break
    if sorted(covered_prefill_requests) != expected_request_ids:
        reasons.append("prefill slabs do not cover each request id exactly once")

    decode_execution = batch_execution.get("decode_execution")
    if not isinstance(decode_execution, Mapping):
        reasons.append("decode execution metadata is missing")
        decode_slots = None
    else:
        decode_slots = _int_list(decode_execution.get("slots"))
        if decode_slots != expected_request_ids:
            reasons.append("decode slots do not match stable compact slots")

    if not isinstance(completed, list):
        reasons.append("completed request metadata is missing")
        completed_ids = None
    else:
        completed_ids = []
        for row in completed:
            if not isinstance(row, Mapping):
                completed_ids = None
                reasons.append("completed request row is not an object")
                break
            request_id = row.get("request_id")
            if isinstance(request_id, bool) or not isinstance(request_id, int):
                completed_ids = None
                reasons.append("completed request id is invalid")
                break
            completed_ids.append(int(request_id))
        if completed_ids is not None and sorted(completed_ids) != expected_request_ids:
            reasons.append("completed request ids do not match admitted request ids")
    active_count_after_completion = scheduler_metadata.get("active_count_after_completion")
    if active_count_after_completion != 0:
        reasons.append("active_count_after_completion is not zero")
    slot_to_request_after_completion = scheduler_metadata.get("slot_to_request_after_completion")
    if not isinstance(slot_to_request_after_completion, list) or any(item is not None for item in slot_to_request_after_completion):
        reasons.append("slot_to_request_after_completion is not fully reclaimed")

    if reasons:
        return {"passed": False, "audit": "; ".join(reasons)}
    return {
        "passed": True,
        "audit": (
            f"fixed-session stable block identity verified: request_ids={expected_request_ids}, "
            f"slot_map={request_to_slot}, decode_slots={decode_slots}, "
            f"prefill_slabs={len(prefill_slabs) if isinstance(prefill_slabs, list) else 0}"
        ),
    }


def _retained_memory_payload(args: argparse.Namespace, kv_policy: ResolvedKVPolicy, bench: Mapping[str, Any] | None = None) -> dict[str, Any]:
    memory = {
        "max_batch_size": args.batch_size,
        "max_sequence_length": args.prompt_length + args.warmup_decode_tokens + args.decode_tokens + 1,
        "kv_policy": kv_policy_json(kv_policy),
        "kv_storage_dtype": kv_policy.storage_dtype.value,
        "allocator_reserved_peak_bytes": None,
        "dynamic_pool": {
            "enabled": False,
            "evidence": "resident retained bench still uses fixed session allocation; C4 pool counters are unavailable here",
            "grow_events": 0,
            "shrink_events": 0,
            "pool_counters": {
                "current_bytes": 0,
                "high_water_observed_bytes": 0,
                "grow_events": 0,
                "grow_failures": 0,
                "shrink_events": 0,
                "free_pages": 0,
                "refcounted_pages": 0,
            },
        },
        "stable_block_id": {"passed": False, "audit": "not captured in retained bench"},
        "prefix_sharing": {"enabled": False, "savings_bytes": 0},
    }
    bench_memory = bench.get("memory") if isinstance(bench, Mapping) else None
    if isinstance(bench_memory, Mapping):
        memory = _merged_mapping(memory, bench_memory)
    stable_block_id = memory.get("stable_block_id")
    if not isinstance(stable_block_id, Mapping) or stable_block_id.get("passed") is not True:
        memory["stable_block_id"] = _stable_block_id_evidence(args, bench)
    return memory


def _decode_layer_execution_blockers(
    decode_execution: Mapping[str, Any],
    *,
    expected_concurrency: int | None = None,
    expected_prompt_length: int | None = None,
) -> list[str]:
    blockers: list[str] = []
    layer_executions = decode_execution.get("layer_executions")
    if not isinstance(layer_executions, list) or not layer_executions:
        return ["execution.batch_execution.decode_execution.layer_executions must be a non-empty list"]
    decode_slots = decode_execution.get("slots")
    native_full_attention_layers = decode_execution.get("native_full_attention_layers")
    moe_grouped_compact_layers = decode_execution.get("moe_grouped_compact_layers")
    traced_native_full_attention_layers = 0
    traced_grouped_moe_layers = 0
    for index, layer in enumerate(layer_executions):
        label = f"execution.batch_execution.decode_execution.layer_executions[{index}]"
        if not isinstance(layer, Mapping):
            blockers.append(f"{label} must be an object")
            continue
        layer_index = layer.get("layer_index")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
            blockers.append(f"{label}.layer_index must be a non-negative int")
        layer_type = layer.get("layer_type")
        if layer_type not in {"linear_attention", "full_attention"}:
            blockers.append(f"{label}.layer_type must be linear_attention or full_attention")
        rows = layer.get("rows")
        if expected_concurrency is not None:
            if isinstance(rows, bool) or not isinstance(rows, int):
                blockers.append(f"{label}.rows must be an int")
            elif rows != int(expected_concurrency):
                blockers.append(f"{label}.rows must match workload.concurrency")
        slots = layer.get("slots")
        if isinstance(decode_slots, list) and slots != decode_slots:
            blockers.append(f"{label}.slots must match decode_execution.slots")
        elif not isinstance(slots, list):
            blockers.append(f"{label}.slots must be a list")
        if layer.get("native_caware_decode") is not True:
            blockers.append(f"{label}.native_caware_decode must be true")
        moe_path = layer.get("moe_decode_path")
        if moe_path != "grouped_compact":
            blockers.append(f"{label}.moe_decode_path must be grouped_compact")
        else:
            traced_grouped_moe_layers += 1
        full_attention_path = layer.get("full_attention_decode_path")
        if layer_type == "full_attention":
            if full_attention_path != "native_batch":
                blockers.append(f"{label}.full_attention_decode_path must be native_batch")
            else:
                traced_native_full_attention_layers += 1
            max_context = layer.get("max_context")
            if isinstance(max_context, bool) or not isinstance(max_context, int):
                blockers.append(f"{label}.max_context must be an int")
            elif expected_prompt_length is not None and max_context < int(expected_prompt_length):
                blockers.append(f"{label}.max_context must cover workload.prompt_tokens_per_request")
            elif max_context >= 1024:
                blockers.append(f"{label}.max_context must be < 1024 until row-aware split-K native decode lands")
            if "num_splits_per_row" in layer:
                blockers.append(f"{label}.num_splits_per_row must be absent for native retained decode")
            if "full_attention_input_decode_path" in layer:
                blockers.append(f"{label}.full_attention_input_decode_path must be absent for native retained decode")
            if "full_attention_context_decode_path" in layer:
                blockers.append(f"{label}.full_attention_context_decode_path must be absent for native retained decode")
            if "post_attention_decode_path" in layer:
                blockers.append(f"{label}.post_attention_decode_path must be absent for native retained decode")
            if "attn_context_trace_source" in layer:
                blockers.append(f"{label}.attn_context_trace_source must be absent for native retained decode")
        elif layer_type == "linear_attention":
            if full_attention_path != "not_applicable":
                blockers.append(f"{label}.full_attention_decode_path must be not_applicable")
            linear_decode_path = layer.get("linear_attention_decode_path")
            if linear_decode_path not in {None, "native_batch_segments"}:
                blockers.append(f"{label}.linear_attention_decode_path must be native_batch_segments or absent")
            linear_projection_path = layer.get("linear_attention_projection_path")
            if linear_projection_path not in {None, "native_batch"}:
                blockers.append(f"{label}.linear_attention_projection_path must be native_batch or absent")
            linear_state_path = layer.get("linear_attention_state_path")
            if linear_state_path not in {None, "native_segments"}:
                blockers.append(f"{label}.linear_attention_state_path must be native_segments or absent")
            linear_output_path = layer.get("linear_attention_output_path")
            if linear_output_path not in {None, "native_batch", "batch_gemv"}:
                blockers.append(f"{label}.linear_attention_output_path must be native_batch, batch_gemv, or absent")
    if isinstance(native_full_attention_layers, int) and not isinstance(native_full_attention_layers, bool):
        if traced_native_full_attention_layers != native_full_attention_layers:
            blockers.append("execution.batch_execution.decode_execution.layer_executions native full-attention count must match native_full_attention_layers")
    if isinstance(moe_grouped_compact_layers, int) and not isinstance(moe_grouped_compact_layers, bool):
        if traced_grouped_moe_layers != moe_grouped_compact_layers:
            blockers.append("execution.batch_execution.decode_execution.layer_executions grouped MoE count must match moe_grouped_compact_layers")
    return blockers


def _batch_execution_blockers(
    batch_execution: Mapping[str, Any],
    *,
    expected_max_layers: int | None = None,
    expected_concurrency: int | None = None,
    expected_prompt_length: int | None = None,
) -> list[str]:
    blockers: list[str] = []
    path = batch_execution.get("path")
    if not isinstance(path, str) or not path:
        blockers.append("execution.batch_execution.path must be a non-empty string")
    elif path != "scheduler_native_compact_batch" or "serial" in path:
        blockers.append("execution.batch_execution.path must be scheduler_native_compact_batch")
    if batch_execution.get("scheduler_owned") is not True:
        blockers.append("execution.batch_execution.scheduler_owned must be true")
    if batch_execution.get("blockers") != []:
        blockers.append("execution.batch_execution.blockers must be empty")
    row_execution = batch_execution.get("row_execution")
    if not isinstance(row_execution, str) or not row_execution:
        blockers.append("execution.batch_execution.row_execution must be a non-empty string")
    elif "serial" in row_execution or "fallback" in row_execution:
        blockers.append("execution.batch_execution.row_execution must not contain serial or fallback")
    if batch_execution.get("native_compact_prefill") is not True:
        blockers.append("execution.batch_execution.native_compact_prefill must be true")
    native_prefill_plan = batch_execution.get("native_prefill_plan")
    if not isinstance(native_prefill_plan, Mapping):
        blockers.append("execution.batch_execution.native_prefill_plan is missing")
    else:
        if native_prefill_plan.get("path") != "single_request_native_full":
            blockers.append("execution.batch_execution.native_prefill_plan.path must be single_request_native_full")
        if native_prefill_plan.get("full_layer_limit_native") is not True:
            blockers.append("execution.batch_execution.native_prefill_plan.full_layer_limit_native must be true")
        if "first_unsupported_layer" not in native_prefill_plan or native_prefill_plan.get("first_unsupported_layer") is not None:
            blockers.append("execution.batch_execution.native_prefill_plan.first_unsupported_layer must be null")
        if "first_unsupported_type" not in native_prefill_plan or native_prefill_plan.get("first_unsupported_type") is not None:
            blockers.append("execution.batch_execution.native_prefill_plan.first_unsupported_type must be null")
        if expected_max_layers is not None:
            layer_limit = native_prefill_plan.get("layer_limit")
            if isinstance(layer_limit, bool) or not isinstance(layer_limit, int):
                blockers.append("execution.batch_execution.native_prefill_plan.layer_limit must be an int")
            elif layer_limit != int(expected_max_layers):
                blockers.append("execution.batch_execution.native_prefill_plan.layer_limit must match workload.max_layers")
        if native_prefill_plan.get("blockers") != []:
            blockers.append("execution.batch_execution.native_prefill_plan.blockers must be empty")
    if batch_execution.get("native_caware_decode") is not True:
        blockers.append("execution.batch_execution.native_caware_decode must be true")
    for diagnostic_field in DECODE_EXECUTION_DIAGNOSTIC_TRACE_FIELDS:
        if diagnostic_field in batch_execution:
            blockers.append(f"execution.batch_execution.{diagnostic_field} must be absent for native retained decode")
    decode_execution = batch_execution.get("decode_execution")
    if not isinstance(decode_execution, Mapping):
        blockers.append("execution.batch_execution.decode_execution is missing")
    else:
        max_context = decode_execution.get("max_full_attention_context")
        max_context_valid = isinstance(max_context, int) and not isinstance(max_context, bool)
        if expected_prompt_length is not None:
            if not max_context_valid:
                blockers.append("execution.batch_execution.decode_execution.max_full_attention_context must be an int")
            elif max_context < int(expected_prompt_length):
                blockers.append("execution.batch_execution.decode_execution.max_full_attention_context must cover workload.prompt_tokens_per_request")
        if max_context_valid and max_context >= 1024:
            blockers.append("execution.batch_execution.decode_execution.max_full_attention_context must be < 1024 until row-aware split-K native decode lands")
        native_full_attention_layers = decode_execution.get("native_full_attention_layers")
        if isinstance(native_full_attention_layers, bool) or not isinstance(native_full_attention_layers, int) or native_full_attention_layers <= 0:
            blockers.append("execution.batch_execution.decode_execution.native_full_attention_layers must be a positive int")
        if expected_concurrency is not None:
            decode_rows = decode_execution.get("rows")
            if isinstance(decode_rows, bool) or not isinstance(decode_rows, int):
                blockers.append("execution.batch_execution.decode_execution.rows must be an int")
            elif decode_rows != int(expected_concurrency):
                blockers.append("execution.batch_execution.decode_execution.rows must match workload.concurrency")
            decode_slots = decode_execution.get("slots")
            if not isinstance(decode_slots, list):
                blockers.append("execution.batch_execution.decode_execution.slots must be a list")
            else:
                if len(decode_slots) != int(expected_concurrency):
                    blockers.append("execution.batch_execution.decode_execution.slots length must match workload.concurrency")
                elif not all(isinstance(slot, int) and not isinstance(slot, bool) and slot >= 0 for slot in decode_slots):
                    blockers.append("execution.batch_execution.decode_execution.slots entries must be non-negative ints")
                elif len(set(decode_slots)) != len(decode_slots):
                    blockers.append("execution.batch_execution.decode_execution.slots entries must be unique")
            moe_decode_rows = decode_execution.get("moe_decode_rows")
            if isinstance(moe_decode_rows, bool) or not isinstance(moe_decode_rows, int):
                blockers.append("execution.batch_execution.decode_execution.moe_decode_rows must be an int")
            elif moe_decode_rows != int(expected_concurrency):
                blockers.append("execution.batch_execution.decode_execution.moe_decode_rows must match workload.concurrency")
        moe_grouped_compact_layers = decode_execution.get("moe_grouped_compact_layers")
        if isinstance(moe_grouped_compact_layers, bool) or not isinstance(moe_grouped_compact_layers, int) or moe_grouped_compact_layers <= 0:
            blockers.append("execution.batch_execution.decode_execution.moe_grouped_compact_layers must be a positive int")
        if decode_execution.get("moe_selected_c1_fallback_layers") != 0:
            blockers.append("execution.batch_execution.decode_execution.moe_selected_c1_fallback_layers must be zero")
        if decode_execution.get("moe_decode_path") != "grouped_compact":
            blockers.append("execution.batch_execution.decode_execution.moe_decode_path must be grouped_compact for retained c>N MoE decode")
        if decode_execution.get("full_attention_decode_path") != "native_batch":
            blockers.append("execution.batch_execution.decode_execution.full_attention_decode_path must be native_batch")
        full_attention_input_path = decode_execution.get("full_attention_input_decode_path")
        if full_attention_input_path not in {None, "native_batch"}:
            blockers.append("execution.batch_execution.decode_execution.full_attention_input_decode_path must be native_batch or absent")
        full_attention_context_path = decode_execution.get("full_attention_context_decode_path")
        if full_attention_context_path not in {None, "native_batch"}:
            blockers.append("execution.batch_execution.decode_execution.full_attention_context_decode_path must be native_batch or absent")
        post_attention_path = decode_execution.get("post_attention_decode_path")
        if post_attention_path not in {None, "native_batch"}:
            blockers.append("execution.batch_execution.decode_execution.post_attention_decode_path must be native_batch or absent")
        linear_projection_path = decode_execution.get("linear_attention_projection_path")
        if linear_projection_path not in {None, "native_batch"}:
            blockers.append("execution.batch_execution.decode_execution.linear_attention_projection_path must be native_batch or absent")
        linear_state_path = decode_execution.get("linear_attention_state_path")
        if linear_state_path not in {None, "native_segments"}:
            blockers.append("execution.batch_execution.decode_execution.linear_attention_state_path must be native_segments or absent")
        linear_output_path = decode_execution.get("linear_attention_output_path")
        if linear_output_path not in {None, "native_batch", "batch_gemv"}:
            blockers.append("execution.batch_execution.decode_execution.linear_attention_output_path must be native_batch, batch_gemv, or absent")
        if decode_execution.get("native_caware_decode") is not True:
            blockers.append("execution.batch_execution.decode_execution.native_caware_decode must be true")
        for diagnostic_field in DECODE_EXECUTION_DIAGNOSTIC_TRACE_FIELDS:
            if diagnostic_field in decode_execution:
                blockers.append(f"execution.batch_execution.decode_execution.{diagnostic_field} must be absent for native retained decode")
        blockers.extend(
            _decode_layer_execution_blockers(
                decode_execution,
                expected_concurrency=expected_concurrency,
                expected_prompt_length=expected_prompt_length,
            )
        )
        if decode_execution.get("blockers") != []:
            blockers.append("execution.batch_execution.decode_execution.blockers must be empty")
    return blockers


def _path_has_retained_results_symlink_parent(path: Path) -> bool:
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    current = path.parent
    while True:
        try:
            current_resolved = current.resolve()
        except OSError:
            return False
        if not current_resolved.is_relative_to(results_root):
            return False
        if current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent


def _load_retained_json_artifact(value: str) -> tuple[Mapping[str, Any] | None, str | None]:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.suffix.lower() != ".json":
        return None, "artifact_path must point to a .json artifact"
    if path.is_symlink():
        return None, "artifact_path must point to a regular JSON artifact, not a symlink"
    if _path_has_retained_results_symlink_parent(path):
        return None, "artifact_path parent directories must not be symlinks"
    if not path.exists():
        return None, "artifact_path must point to an existing JSON artifact"
    if not path.is_file():
        return None, "artifact_path must point to a regular JSON artifact"
    try:
        payload = _load_json_path(path)
    except OSError as exc:
        return None, f"artifact_path must point to a readable JSON artifact: {exc}"
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"artifact_path must point to a valid JSON artifact: {exc}"
    if not isinstance(payload, Mapping):
        return None, "artifact_path must point to a JSON object artifact"
    return payload, None


def _retained_artifact_row_count(payload: Mapping[str, Any]) -> Any:
    rows = payload.get("rows")
    if rows is not None:
        return rows
    workload = payload.get("workload")
    if isinstance(workload, Mapping):
        return workload.get("concurrency")
    return None


def _retained_artifact_accepted(payload: Mapping[str, Any]) -> bool:
    if payload.get("accepted") is True or payload.get("passed") is True or payload.get("status") == "accepted":
        return True
    decision = payload.get("decision")
    return isinstance(decision, Mapping) and decision.get("accepted") is True


def _projection_evidence_artifact_blockers(evidence: Mapping[str, Any], *, concurrency: int) -> list[str]:
    artifact_path = evidence.get("artifact_path")
    if not _is_retained_artifact_path(artifact_path):
        return []
    payload, error = _load_retained_json_artifact(str(artifact_path))
    if error is not None:
        return [f"execution.batch_execution.projection_dispatch.evidence.{error}"]
    if payload is None:
        return []
    blockers: list[str] = []
    if not _retained_artifact_accepted(payload):
        blockers.append("execution.batch_execution.projection_dispatch.evidence.artifact_path artifact must be accepted")
    try:
        parsed_evidence = ProjectionDispatchEvidence.from_json_dict(evidence)
    except ValueError:
        parsed_evidence = None
    if parsed_evidence is not None:
        blockers.extend(
            projection_dispatch_evidence_payload_blockers(
                payload,
                parsed_evidence,
                rows=int(concurrency),
                label="execution.batch_execution.projection_dispatch.evidence.artifact_path",
            )
        )
    else:
        artifact_rows = _retained_artifact_row_count(payload)
        if isinstance(artifact_rows, bool) or not isinstance(artifact_rows, int):
            blockers.append("execution.batch_execution.projection_dispatch.evidence.artifact_path rows must be an int")
        elif artifact_rows != int(concurrency):
            blockers.append("execution.batch_execution.projection_dispatch.evidence.artifact_path rows must match workload.concurrency")
    return blockers


def _projection_dispatch_blockers(
    batch_execution: Mapping[str, Any],
    *,
    concurrency: int,
    candidates: Any = None,
) -> list[str]:
    projection_dispatch = batch_execution.get("projection_dispatch")
    if not isinstance(projection_dispatch, Mapping):
        return ["execution.batch_execution.projection_dispatch is missing"]
    blockers: list[str] = []
    rows = projection_dispatch.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 1:
        blockers.append("execution.batch_execution.projection_dispatch.rows must be an int > 1")
    elif rows != int(concurrency):
        blockers.append("execution.batch_execution.projection_dispatch.rows must match workload.concurrency")
    if projection_dispatch.get("path") != "benchmark_accepted_caware_projection":
        blockers.append("execution.batch_execution.projection_dispatch.path must be benchmark_accepted_caware_projection")
    selected_candidate = projection_dispatch.get("selected_candidate")
    if not isinstance(selected_candidate, str) or not selected_candidate:
        blockers.append("execution.batch_execution.projection_dispatch.selected_candidate is missing")
    elif selected_candidate == "row_gemv":
        blockers.append("execution.batch_execution.projection_dispatch.selected_candidate must not be row_gemv")
    if projection_dispatch.get("throughput_claim_eligible") is not True:
        blockers.append("execution.batch_execution.projection_dispatch.throughput_claim_eligible must be true")
    if projection_dispatch.get("blockers") != []:
        blockers.append("execution.batch_execution.projection_dispatch.blockers must be empty")
    selection = projection_dispatch.get("selection")
    if not isinstance(selection, Mapping):
        blockers.append("execution.batch_execution.projection_dispatch.selection is missing")
    else:
        for field in ("layer", "quant", "variant"):
            if not isinstance(selection.get(field), str) or not selection.get(field):
                blockers.append(f"execution.batch_execution.projection_dispatch.selection.{field} is missing")
        if selection.get("variant") == "row_gemv":
            blockers.append("execution.batch_execution.projection_dispatch.selection.variant must not be row_gemv")
    evidence = projection_dispatch.get("evidence")
    if not isinstance(evidence, Mapping):
        blockers.append("execution.batch_execution.projection_dispatch.evidence is missing")
    else:
        if evidence.get("accepted") is not True:
            blockers.append("execution.batch_execution.projection_dispatch.evidence.accepted must be true")
        if not _is_retained_artifact_path(evidence.get("artifact_path")):
            blockers.append("execution.batch_execution.projection_dispatch.evidence.artifact_path must be under benchmarks/results")
        else:
            blockers.extend(_projection_evidence_artifact_blockers(evidence, concurrency=concurrency))
        for field in ("aggregate_vs_row_gemv", "per_request_vs_row_gemv"):
            value = evidence.get(field)
            if not _is_finite_positive_number(value) or float(value) <= 1.0:
                blockers.append(f"execution.batch_execution.projection_dispatch.evidence.{field} must be > 1.0")
    selected_candidate_entry: Mapping[str, Any] | None = None
    if not isinstance(candidates, list) or not candidates:
        blockers.append("projection_dispatch_candidates must include selected projection candidate")
    elif isinstance(selected_candidate, str) and selected_candidate:
        matches = [candidate for candidate in candidates if isinstance(candidate, Mapping) and candidate.get("name") == selected_candidate]
        if not matches:
            blockers.append("projection_dispatch_candidates must include selected_candidate")
        else:
            selected_candidate_entry = matches[0]
    if selected_candidate_entry is not None:
        min_rows = selected_candidate_entry.get("min_rows", 2)
        max_rows = selected_candidate_entry.get("max_rows")
        row_bounds_valid = True
        if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows <= 0:
            blockers.append("projection_dispatch_candidates selected_candidate.min_rows must be a positive int")
            row_bounds_valid = False
        if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0):
            blockers.append("projection_dispatch_candidates selected_candidate.max_rows must be a positive int or null")
            row_bounds_valid = False
        if row_bounds_valid and isinstance(rows, int) and not isinstance(rows, bool):
            if rows < int(min_rows) or (isinstance(max_rows, int) and rows > int(max_rows)):
                blockers.append("projection_dispatch_candidates selected_candidate row bounds must include projection_dispatch.rows")
        candidate_selection = selected_candidate_entry.get("selection")
        if not isinstance(candidate_selection, Mapping):
            blockers.append("projection_dispatch_candidates selected_candidate.selection is missing")
        elif isinstance(selection, Mapping):
            expected_selection = {field: candidate_selection.get(field) for field in ("layer", "quant", "variant")}
            actual_selection = {field: selection.get(field) for field in ("layer", "quant", "variant")}
            if expected_selection != actual_selection:
                blockers.append("execution.batch_execution.projection_dispatch.selection must match selected projection_dispatch_candidates entry")
        candidate_evidence = selected_candidate_entry.get("evidence")
        if not isinstance(candidate_evidence, Mapping):
            blockers.append("projection_dispatch_candidates selected_candidate.evidence is missing")
        elif isinstance(evidence, Mapping):
            for field in ("artifact_path", "accepted"):
                if candidate_evidence.get(field) != evidence.get(field):
                    blockers.append("execution.batch_execution.projection_dispatch.evidence must match selected projection_dispatch_candidates entry")
                    break
            else:
                for field in ("aggregate_vs_row_gemv", "per_request_vs_row_gemv"):
                    candidate_value = candidate_evidence.get(field)
                    evidence_value = evidence.get(field)
                    if not (_is_finite_nonnegative_number(candidate_value) and _is_finite_nonnegative_number(evidence_value)) or float(candidate_value) != float(evidence_value):
                        blockers.append("execution.batch_execution.projection_dispatch.evidence must match selected projection_dispatch_candidates entry")
                        break
    return blockers


def _is_path_relative_to(child: str, parent: str) -> bool:
    try:
        child_path = Path(child).expanduser().resolve(strict=False)
        parent_path = Path(parent).expanduser().resolve(strict=False)
        child_path.relative_to(parent_path)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _command_int_arg_matches(command: str, flag: str, expected: int) -> bool:
    value = _command_arg_value(command, flag)
    if value is None:
        return False
    try:
        return int(value) == int(expected)
    except ValueError:
        return False


def _command_string_arg_matches(command: str, flag: str, expected: str) -> bool:
    return _command_arg_value(command, flag) == expected


def _command_has_flag(command: str, flag: str) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return flag in parts


def _split_command_parts(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _profiled_command_segment(command: str) -> list[str] | None:
    parts = _split_command_parts(command)
    if "--" not in parts:
        return None
    return parts[parts.index("--") + 1 :]


def _rocprof_command_prefix(command: str) -> list[str]:
    parts = _split_command_parts(command)
    if "--" not in parts:
        return parts
    return parts[: parts.index("--")]


def _join_command_parts(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _is_env_assignment_token(token: str) -> bool:
    key, sep, _value = token.partition("=")
    return bool(
        sep
        and key
        and (key[0].isalpha() or key[0] == "_")
        and all(ch.isalnum() or ch == "_" for ch in key)
    )


def _strip_command_env_prefix(parts: Sequence[str]) -> list[str]:
    idx = 0
    if parts and Path(parts[0]).name == "env":
        idx = 1
    while idx < len(parts) and _is_env_assignment_token(parts[idx]):
        idx += 1
    if idx == 0:
        return list(parts)
    return list(parts[idx:])


def _command_device_env_assignments(parts: Sequence[str]) -> dict[str, str]:
    idx = 0
    if parts and Path(parts[0]).name == "env":
        idx = 1
    assignments: dict[str, str] = {}
    while idx < len(parts) and _is_env_assignment_token(parts[idx]):
        key, _sep, value = parts[idx].partition("=")
        if key in _COMMAND_ENV_KEYS:
            assignments[key] = value
        idx += 1
    return assignments


def _current_command_device_env_assignments() -> dict[str, str]:
    return _command_device_env_assignments(_command_env_prefix_parts())


def _command_flag_count(parts: Sequence[str], flag: str) -> int:
    prefix = f"{flag}="
    return sum(1 for part in parts if part == flag or part.startswith(prefix))


def _profiler_command_provenance_blockers(
    command: str,
    *,
    trace_dir: str | None,
    profiler_artifact_path: str | None,
    retained_artifact_path: str | None,
    expected_workload: Mapping[str, int] | None,
    expected_inputs: Mapping[str, str] | None,
    expected_build: Mapping[str, Any] | None,
    expected_references: Mapping[str, Any] | None,
    expected_kv_policy: Mapping[str, str] | None,
    expected_sampler: Mapping[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    command_parts = _split_command_parts(command)
    if command_parts.count("--") != 1:
        blockers.append("profiler command must include exactly one rocprof separator")
    rocprof_prefix = _rocprof_command_prefix(command)
    rocprof_prefix_command = _join_command_parts(rocprof_prefix)
    if not rocprof_prefix or Path(rocprof_prefix[0]).name != _ROCPROF_EXECUTABLE:
        blockers.append("profiler command must start with rocprofv3")
    if not any(Path(part).name == _ROCPROF_EXECUTABLE for part in rocprof_prefix):
        blockers.append("profiler command must include rocprofv3")
    for flag in _ROCPROF_COMMAND_FLAGS:
        if _command_flag_count(rocprof_prefix, flag) > 1:
            blockers.append(f"profiler command {flag} must be unique before rocprof separator")
    if _ROCPROF_COMMAND_FLAGS[0] not in rocprof_prefix:
        blockers.append("profiler command must include --kernel-trace")
    if _RETAINED_BENCH_SCRIPT not in command:
        blockers.append("profiler command must target scripts/qwen35_batch_retained_bench.py")
    profiled_segment = _profiled_command_segment(command)
    retained_command = command
    if profiled_segment is None:
        blockers.append("profiler command must include rocprof -- separator")
    else:
        retained_command = _join_command_parts(profiled_segment)
        for flag in tuple(dict.fromkeys(_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS + _BATCH_SAMPLE_COMMAND_FLAGS)):
            if _command_flag_count(profiled_segment, flag) > 1:
                blockers.append(f"profiler command {flag} must be unique after rocprof separator")
        for flag in _RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS:
            if _command_has_flag(retained_command, flag):
                blockers.append(f"profiler command must not include {flag}")
        profiled_env = _command_device_env_assignments(profiled_segment)
        current_env = _current_command_device_env_assignments()
        if any(not value.strip() for value in (*profiled_env.values(), *current_env.values())):
            blockers.append("profiler command device env prefix values must be non-blank")
        elif profiled_env != current_env:
            blockers.append("profiler command device env prefix must match retained command")
        profiled_launch_segment = _strip_command_env_prefix(profiled_segment)
        if (
            len(profiled_launch_segment) < 2
            or not Path(profiled_launch_segment[0]).name.startswith("python")
            or profiled_launch_segment[1] != _RETAINED_BENCH_SCRIPT
        ):
            blockers.append("profiler command must launch retained bench after rocprof separator")
    if _command_arg_value(rocprof_prefix_command, _ROCPROF_COMMAND_FLAGS[1]) != _ROCPROF_OUTPUT_FORMAT:
        blockers.append("profiler command must include --output-format csv")
    if trace_dir is not None and _command_arg_value(rocprof_prefix_command, _ROCPROF_COMMAND_FLAGS[2]) != trace_dir:
        blockers.append("profiler command -d must match profiler.trace_dir")
    profiler_json_flag = _RETAINED_GATE_FLAGS[3]
    if profiler_artifact_path is not None and _command_arg_value(retained_command, profiler_json_flag) != profiler_artifact_path:
        blockers.append(f"profiler command {profiler_json_flag} must match profiler.artifact_path")
    if retained_artifact_path is not None and _command_arg_value(retained_command, "--json") != retained_artifact_path:
        blockers.append("profiler command --json must match retained artifact path")
    if expected_workload is not None:
        for key, flag, default_value in (
            ("batch_size", "--batch-size", None),
            ("prompt_length", "--prompt-length", None),
            ("decode_tokens", "--decode-tokens", None),
            ("warmup_decode_tokens", "--warmup-decode-tokens", 8),
            ("max_layers", "--max-layers", None),
        ):
            expected_value = expected_workload.get(key)
            if isinstance(expected_value, int) and not isinstance(expected_value, bool):
                command_value = _command_arg_value(retained_command, flag)
                if command_value is None and default_value is not None and int(expected_value) == int(default_value):
                    continue
                if not _command_int_arg_matches(retained_command, flag, expected_value):
                    blockers.append(f"profiler command {flag} must match retained workload")
    if expected_inputs is not None:
        for key, flag in (("model", "--model"), ("fixture", "--fixture")):
            expected_value = expected_inputs.get(key)
            if isinstance(expected_value, str) and expected_value and not _command_string_arg_matches(retained_command, flag, expected_value):
                blockers.append(f"profiler command {flag} must match retained {key}")
    if expected_build is not None:
        compiler_version_file = expected_build.get("compiler_version_file")
        if not isinstance(compiler_version_file, str) or not compiler_version_file:
            blockers.append("retained command must include --compiler-version-file")
        elif _command_arg_value(retained_command, "--compiler-version-file") != compiler_version_file:
            blockers.append("profiler command --compiler-version-file must match retained compiler-version-file")
        if expected_build.get("require_cached_build") is not True:
            blockers.append("retained command must include --require-cached-build")
        elif not _command_has_flag(retained_command, "--require-cached-build"):
            blockers.append("profiler command must include --require-cached-build")
    if expected_references is not None:
        for key, flag in zip(_RETAINED_GATE_LABELS[:3], _RETAINED_GATE_FLAGS[:3]):
            expected_value = expected_references.get(key)
            if not isinstance(expected_value, str) or not expected_value:
                blockers.append(f"retained command must include {flag}")
            elif _command_arg_value(retained_command, flag) != expected_value:
                blockers.append(f"profiler command {flag} must match retained reference artifact")
    if expected_kv_policy is not None:
        for key, flag, default_value in (
            ("kv_storage", _RETAINED_KV_POLICY_FLAGS[0], "auto"),
            ("kv_scale_dtype", _RETAINED_KV_POLICY_FLAGS[1], "fp16"),
            ("kv_scale_granularity", _RETAINED_KV_POLICY_FLAGS[2], "per_token_head"),
        ):
            expected_value = expected_kv_policy.get(key)
            command_value = _command_arg_value(retained_command, flag)
            if isinstance(expected_value, str) and expected_value:
                if command_value is None and expected_value == default_value:
                    continue
                if command_value != expected_value:
                    blockers.append(f"profiler command {flag} must match retained KV policy")
    if expected_sampler is not None:
        expected_mode = expected_sampler.get("batch_sample_mode")
        if isinstance(expected_mode, str) and expected_mode:
            command_mode = _command_arg_value(retained_command, "--batch-sample-mode")
            if command_mode is None and expected_mode == "serial_lm_head":
                command_mode = "serial_lm_head"
            if command_mode != expected_mode:
                blockers.append("profiler command --batch-sample-mode must match retained sampler mode")
        expected_norm_path = expected_sampler.get("batch_sample_norm_path")
        if isinstance(expected_norm_path, str) and expected_norm_path:
            command_norm_path = _command_arg_value(retained_command, "--batch-sample-norm-path")
            if command_norm_path is None and expected_norm_path == "batch":
                command_norm_path = "batch"
            if command_norm_path != expected_norm_path:
                blockers.append("profiler command --batch-sample-norm-path must match retained sampler norm path")
        expected_cast_path = expected_sampler.get("batch_sample_cast_path")
        if isinstance(expected_cast_path, str) and expected_cast_path:
            command_cast_path = _command_arg_value(retained_command, "--batch-sample-cast-path")
            if command_cast_path is None and expected_cast_path == "auto":
                command_cast_path = "auto"
            if command_cast_path != expected_cast_path:
                blockers.append("profiler command --batch-sample-cast-path must match retained sampler cast path")
        if expected_mode == "batched_lm_head":
            if expected_sampler.get("batch_sample_eq_ok") is not True:
                blockers.append("retained command must include --batch-sample-eq-ok for batched_lm_head")
            elif not _command_has_flag(retained_command, "--batch-sample-eq-ok"):
                blockers.append("profiler command must include --batch-sample-eq-ok")
            expected_artifact = expected_sampler.get("batch_sample_eq_artifact")
            if not isinstance(expected_artifact, str) or not expected_artifact:
                blockers.append("retained command must include --batch-sample-eq-artifact for batched_lm_head")
            elif _command_arg_value(retained_command, "--batch-sample-eq-artifact") != expected_artifact:
                blockers.append("profiler command --batch-sample-eq-artifact must match retained sampler equality artifact")
            expected_rows = expected_sampler.get("batch_sample_eq_rows")
            if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows <= 0:
                blockers.append("retained command must include --batch-sample-eq-rows for batched_lm_head")
            elif not _command_int_arg_matches(retained_command, "--batch-sample-eq-rows", int(expected_rows)):
                blockers.append("profiler command --batch-sample-eq-rows must match retained sampler equality rows")
    return blockers


def _profiler_provenance_blockers(
    profiler: Mapping[str, Any],
    *,
    profiled_command: str | None = None,
    retained_artifact_path: str | None = None,
    expected_workload: Mapping[str, int] | None = None,
    expected_inputs: Mapping[str, str] | None = None,
    expected_build: Mapping[str, Any] | None = None,
    expected_references: Mapping[str, Any] | None = None,
    expected_kv_policy: Mapping[str, str] | None = None,
    expected_sampler: Mapping[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    profiler_artifact_path = profiler.get("artifact_path")
    retained_profiler_artifact_path = profiler_artifact_path if _is_retained_artifact_path(profiler_artifact_path) else None
    if retained_profiler_artifact_path is None:
        blockers.append("profiler.artifact_path must be under benchmarks/results")
    profiler_source_artifact_path = profiler.get("source_artifact_path")
    if not isinstance(profiler_source_artifact_path, str) or not profiler_source_artifact_path:
        blockers.append("profiler.source_artifact_path must be a non-empty string")
    elif retained_profiler_artifact_path is not None and profiler_source_artifact_path != retained_profiler_artifact_path:
        blockers.append("profiler.source_artifact_path must match profiler.artifact_path")
    if profiler.get("output_format") != _ROCPROF_OUTPUT_FORMAT:
        blockers.append("profiler.output_format must be csv")
    trace_dir = profiler.get("trace_dir")
    if not isinstance(trace_dir, str) or not trace_dir:
        blockers.append("profiler.trace_dir must be a non-empty string")
        trace_dir = None
    elif "<" in trace_dir or ">" in trace_dir:
        blockers.append("profiler.trace_dir must be a concrete path")
    trace_files = profiler.get("trace_files")
    if not isinstance(trace_files, list) or not trace_files or not all(isinstance(trace_file, str) and trace_file for trace_file in trace_files):
        blockers.append("profiler.trace_files must be a non-empty string list")
        return blockers
    if any("<" in trace_file or ">" in trace_file for trace_file in trace_files):
        blockers.append("profiler.trace_files entries must be concrete paths")
    if len(set(trace_files)) != len(trace_files):
        blockers.append("profiler.trace_files entries must be unique")
    if not any(Path(trace_file).name.endswith("kernel_trace.csv") for trace_file in trace_files):
        blockers.append("profiler.trace_files must include a kernel-trace CSV")
    for trace_file in trace_files:
        trace_path = Path(trace_file)
        if trace_path.suffix.lower() != ".csv":
            blockers.append("profiler.trace_files entries must be CSV paths")
            break
        if trace_dir is not None and not _is_path_relative_to(trace_file, trace_dir):
            blockers.append("profiler.trace_files entries must be under profiler.trace_dir")
            break
    artifact_commands = [
        command
        for command in (profiler.get("command"), profiler.get("profiler_command"))
        if isinstance(command, str) and command
    ]
    if retained_profiler_artifact_path is not None and not artifact_commands:
        blockers.append("profiler artifact must include command or profiler_command")
    command_candidates: list[str] = []
    if isinstance(profiled_command, str) and profiled_command and "<profile-dir>" not in profiled_command:
        command_candidates.append(profiled_command)
    command_candidates.extend(artifact_commands)
    unique_command_candidates = list(dict.fromkeys(command_candidates))
    if not unique_command_candidates:
        blockers.append("profiler command must include rocprofv3 --kernel-trace retained bench command")
    else:
        for command in unique_command_candidates:
            blockers.extend(
                _profiler_command_provenance_blockers(
                    command,
                    trace_dir=trace_dir,
                    profiler_artifact_path=retained_profiler_artifact_path,
                    retained_artifact_path=retained_artifact_path,
                    expected_workload=expected_workload,
                    expected_inputs=expected_inputs,
                    expected_build=expected_build,
                    expected_references=expected_references,
                    expected_kv_policy=expected_kv_policy,
                    expected_sampler=expected_sampler,
                )
            )
    return blockers


def _profiler_cpu_side_bottleneck_blockers(profiler: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    cpu_total = profiler.get("cpu_side_total_seconds")
    total_valid = _is_finite_positive_number(cpu_total)
    if not total_valid:
        blockers.append("profiler.cpu_side_total_seconds must be positive numeric")
    expected_keys = set(_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
    durations = profiler.get("cpu_side_bottlenecks_seconds")
    duration_values_valid = False
    duration_sum = 0.0
    if not isinstance(durations, Mapping) or not durations:
        blockers.append("profiler.cpu_side_bottlenecks_seconds must be a non-empty object")
    else:
        duration_keys = {key for key in durations if isinstance(key, str) and key}
        if len(duration_keys) != len(durations):
            blockers.append("profiler.cpu_side_bottlenecks_seconds keys must be non-empty strings")
        elif duration_keys != expected_keys:
            blockers.append("profiler.cpu_side_bottlenecks_seconds keys must match known categories")
        else:
            duration_values_valid = True
            for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES:
                duration_seconds = durations[category]
                if not _is_finite_nonnegative_number(duration_seconds):
                    blockers.append(f"profiler.cpu_side_bottlenecks_seconds.{category} must be finite nonnegative numeric")
                    duration_values_valid = False
                    break
                duration_sum += float(duration_seconds)
            if duration_values_valid and total_valid and not math.isclose(
                duration_sum, float(cpu_total), rel_tol=1e-6, abs_tol=1e-9
            ):
                blockers.append("profiler.cpu_side_bottlenecks_seconds must sum to profiler.cpu_side_total_seconds")
    shares = profiler.get("cpu_side_bottleneck_shares")
    share_values_valid = False
    share_sum = 0.0
    if not isinstance(shares, Mapping) or not shares:
        blockers.append("profiler.cpu_side_bottleneck_shares must be a non-empty object")
    else:
        share_keys = {key for key in shares if isinstance(key, str) and key}
        if len(share_keys) != len(shares):
            blockers.append("profiler.cpu_side_bottleneck_shares keys must be non-empty strings")
        elif share_keys != expected_keys:
            blockers.append("profiler.cpu_side_bottleneck_shares keys must match known categories")
        else:
            share_values_valid = True
            for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES:
                share = shares[category]
                if not _is_finite_nonnegative_number(share):
                    blockers.append(f"profiler.cpu_side_bottleneck_shares.{category} must be finite nonnegative numeric")
                    share_values_valid = False
                    break
                share_sum += float(share)
                if total_valid and duration_values_valid:
                    expected_share = float(durations[category]) / float(cpu_total)
                    if not math.isclose(float(share), expected_share, rel_tol=1e-6, abs_tol=1e-9):
                        blockers.append(f"profiler.cpu_side_bottleneck_shares.{category} must match duration/total")
                        share_values_valid = False
                        break
            if share_values_valid and not math.isclose(share_sum, 1.0, rel_tol=1e-6, abs_tol=1e-9):
                blockers.append("profiler.cpu_side_bottleneck_shares must sum to 1.0")
    return blockers


def _profiler_synthesized_fields_blockers(profiler: Mapping[str, Any]) -> list[str]:
    synthesized_fields = profiler.get("synthesized_fields")
    if not isinstance(synthesized_fields, list) or not all(isinstance(field, str) for field in synthesized_fields):
        return ["profiler.synthesized_fields must be a string list"]
    blockers: list[str] = []
    if len(set(synthesized_fields)) != len(synthesized_fields):
        blockers.append("profiler.synthesized_fields must not contain duplicates")
    unknown_fields = sorted(set(synthesized_fields) - set(_PROFILER_SYNTHESIZED_FIELDS))
    if unknown_fields:
        blockers.append("profiler.synthesized_fields must only name known synthesized profiler fields")
    return blockers


def _profiler_kernel_evidence_blockers(profiler: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if profiler.get("status") != "captured":
        blockers.append("profiler.status must be captured")
    if profiler.get("expected_kernels_present") is not True:
        blockers.append("profiler.expected_kernels_present must be true")
    trace_kernel_names = profiler.get("trace_kernel_names")
    if not isinstance(trace_kernel_names, list) or not any(isinstance(name, str) and name for name in trace_kernel_names):
        blockers.append("profiler.trace_kernel_names must be a non-empty string list")
    elif not all(_is_stripped_non_empty_string(name) for name in trace_kernel_names):
        blockers.append("profiler.trace_kernel_names entries must be non-empty strings")
    elif len(set(trace_kernel_names)) != len(trace_kernel_names):
        blockers.append("profiler.trace_kernel_names entries must be unique")
    elif any(_has_disallowed_profiler_kernel_name_fragment(name) for name in trace_kernel_names):
        blockers.append("profiler.trace_kernel_names must not include serial/per-row/fallback kernel names")
    elif not _has_native_batch_profiler_kernel_name(trace_kernel_names):
        blockers.append("profiler.trace_kernel_names must include at least one native batch kernel name")
    expected_kernel_names = profiler.get("expected_kernel_names")
    if not isinstance(expected_kernel_names, list) or not any(isinstance(name, str) and name for name in expected_kernel_names):
        blockers.append("profiler.expected_kernel_names must be a non-empty string list")
    elif not all(_is_stripped_non_empty_string(name) for name in expected_kernel_names):
        blockers.append("profiler.expected_kernel_names entries must be non-empty strings")
    elif len(set(expected_kernel_names)) != len(expected_kernel_names):
        blockers.append("profiler.expected_kernel_names entries must be unique")
    elif any(_has_disallowed_profiler_kernel_name_fragment(name) for name in expected_kernel_names):
        blockers.append("profiler.expected_kernel_names must not include serial/per-row/fallback kernel names")
    elif not _has_native_batch_profiler_kernel_name(expected_kernel_names):
        blockers.append("profiler.expected_kernel_names must include at least one native batch kernel name")
    kernel_durations = profiler.get("kernel_durations_ns")
    if not isinstance(kernel_durations, Mapping) or not kernel_durations:
        blockers.append("profiler.kernel_durations_ns must be a non-empty object")
        return blockers
    duration_keys_valid = True
    durations_valid = True
    duration_total = 0.0
    for kernel_name, duration_ns in kernel_durations.items():
        if not _is_stripped_non_empty_string(kernel_name):
            blockers.append("profiler.kernel_durations_ns keys must be non-empty strings")
            duration_keys_valid = False
            break
        if _has_disallowed_profiler_kernel_name_fragment(kernel_name):
            blockers.append("profiler.kernel_durations_ns must not include serial/per-row/fallback kernel names")
            duration_keys_valid = False
            break
        if not _is_finite_positive_number(duration_ns):
            blockers.append(f"profiler.kernel_durations_ns.{kernel_name} must be positive numeric")
            durations_valid = False
            break
        duration_total += float(duration_ns)
    total_kernel_duration = profiler.get("total_kernel_duration_ns")
    total_duration_valid = _is_finite_positive_number(total_kernel_duration)
    if not total_duration_valid:
        blockers.append("profiler.total_kernel_duration_ns must be positive numeric")
    elif durations_valid and not math.isclose(float(total_kernel_duration), duration_total, rel_tol=1e-6, abs_tol=1e-3):
        blockers.append("profiler.total_kernel_duration_ns must equal sum(profiler.kernel_durations_ns)")
    kernel_duration_shares = profiler.get("kernel_duration_shares")
    if not isinstance(kernel_duration_shares, Mapping) or not kernel_duration_shares:
        blockers.append("profiler.kernel_duration_shares must be a non-empty object")
    elif duration_keys_valid:
        duration_key_set = {key for key in kernel_durations if _is_stripped_non_empty_string(key)}
        share_key_set = {key for key in kernel_duration_shares if _is_stripped_non_empty_string(key)}
        if len(share_key_set) != len(kernel_duration_shares):
            blockers.append("profiler.kernel_duration_shares keys must be non-empty strings")
        elif share_key_set != duration_key_set:
            blockers.append("profiler.kernel_duration_shares keys must match profiler.kernel_durations_ns")
        else:
            for kernel_name in sorted(share_key_set):
                share = kernel_duration_shares[kernel_name]
                if not _is_finite_nonnegative_number(share):
                    blockers.append(f"profiler.kernel_duration_shares.{kernel_name} must be finite nonnegative numeric")
                    break
                if total_duration_valid and durations_valid:
                    expected_share = float(kernel_durations[kernel_name]) / float(total_kernel_duration)
                    if not math.isclose(float(share), expected_share, rel_tol=1e-6, abs_tol=1e-9):
                        blockers.append(f"profiler.kernel_duration_shares.{kernel_name} must match duration/total")
                        break
    category_key_set = set(_PROFILER_KERNEL_DURATION_CATEGORIES)
    expected_categories: dict[str, float] | None = None
    if duration_keys_valid and durations_valid:
        expected_categories = dict.fromkeys(_PROFILER_KERNEL_DURATION_CATEGORIES, 0.0)
        for kernel_name, duration_ns in kernel_durations.items():
            expected_categories[_profiler_kernel_duration_category(kernel_name)] += float(duration_ns)
    kernel_duration_categories = profiler.get("kernel_duration_categories_ns")
    category_values_valid = False
    if not isinstance(kernel_duration_categories, Mapping) or not kernel_duration_categories:
        blockers.append("profiler.kernel_duration_categories_ns must be a non-empty object")
    else:
        category_keys = {key for key in kernel_duration_categories if isinstance(key, str) and key}
        if len(category_keys) != len(kernel_duration_categories):
            blockers.append("profiler.kernel_duration_categories_ns keys must be non-empty strings")
        elif category_keys != category_key_set:
            blockers.append("profiler.kernel_duration_categories_ns keys must match known categories")
        else:
            category_values_valid = True
            for category in _PROFILER_KERNEL_DURATION_CATEGORIES:
                category_value = kernel_duration_categories[category]
                if not _is_finite_nonnegative_number(category_value):
                    blockers.append(f"profiler.kernel_duration_categories_ns.{category} must be finite nonnegative numeric")
                    category_values_valid = False
                    break
                if expected_categories is not None and not math.isclose(
                    float(category_value), expected_categories[category], rel_tol=1e-6, abs_tol=1e-3
                ):
                    blockers.append(f"profiler.kernel_duration_categories_ns.{category} must match categorized kernel_durations_ns")
                    category_values_valid = False
                    break
    kernel_duration_category_shares = profiler.get("kernel_duration_category_shares")
    if not isinstance(kernel_duration_category_shares, Mapping) or not kernel_duration_category_shares:
        blockers.append("profiler.kernel_duration_category_shares must be a non-empty object")
    else:
        category_share_keys = {key for key in kernel_duration_category_shares if isinstance(key, str) and key}
        if len(category_share_keys) != len(kernel_duration_category_shares):
            blockers.append("profiler.kernel_duration_category_shares keys must be non-empty strings")
        elif category_share_keys != category_key_set:
            blockers.append("profiler.kernel_duration_category_shares keys must match known categories")
        else:
            for category in _PROFILER_KERNEL_DURATION_CATEGORIES:
                category_share = kernel_duration_category_shares[category]
                if not _is_finite_nonnegative_number(category_share):
                    blockers.append(f"profiler.kernel_duration_category_shares.{category} must be finite nonnegative numeric")
                    break
                if total_duration_valid and category_values_valid:
                    expected_share = float(kernel_duration_categories[category]) / float(total_kernel_duration)
                    if not math.isclose(float(category_share), expected_share, rel_tol=1e-6, abs_tol=1e-9):
                        blockers.append(f"profiler.kernel_duration_category_shares.{category} must match category/total")
                        break
    if isinstance(trace_kernel_names, list):
        trace_name_set = {name for name in trace_kernel_names if _is_stripped_non_empty_string(name)}
        missing_trace_names = [name for name in kernel_durations if _is_stripped_non_empty_string(name) and name not in trace_name_set]
        if missing_trace_names:
            blockers.append("profiler.trace_kernel_names must include profiler.kernel_durations_ns keys")
    if isinstance(expected_kernel_names, list):
        trace_name_set = set()
        if isinstance(trace_kernel_names, list):
            trace_name_set = {name for name in trace_kernel_names if _is_stripped_non_empty_string(name)}
        missing_expected_trace_name = False
        for kernel_name in expected_kernel_names:
            if _is_stripped_non_empty_string(kernel_name):
                if trace_name_set and kernel_name not in trace_name_set and not missing_expected_trace_name:
                    blockers.append("profiler.trace_kernel_names must include profiler.expected_kernel_names")
                    missing_expected_trace_name = True
                if not _is_finite_positive_number(kernel_durations.get(kernel_name)):
                    blockers.append(f"profiler.kernel_durations_ns.{kernel_name} must be positive numeric")
                    break
    return blockers


def _projection_dispatch_profiler_blockers(batch_execution: Mapping[str, Any], profiler: Mapping[str, Any]) -> list[str]:
    projection_dispatch = batch_execution.get("projection_dispatch")
    if not isinstance(projection_dispatch, Mapping):
        return []
    fragments: list[str] = []
    selected_candidate = projection_dispatch.get("selected_candidate")
    if isinstance(selected_candidate, str) and selected_candidate and selected_candidate != "row_gemv":
        fragments.append(selected_candidate.lower())
    selection = projection_dispatch.get("selection")
    variant = selection.get("variant") if isinstance(selection, Mapping) else None
    if isinstance(variant, str) and variant and variant != "row_gemv":
        fragments.append(variant.lower())
    if not fragments:
        return []
    blockers: list[str] = []
    expected_kernel_names = profiler.get("expected_kernel_names")
    if isinstance(expected_kernel_names, list):
        expected_lower_names = [name.lower() for name in expected_kernel_names if isinstance(name, str) and name]
        if not any(fragment in name for fragment in fragments for name in expected_lower_names):
            blockers.append("profiler.expected_kernel_names must include selected projection_dispatch candidate or variant")
    trace_kernel_names = profiler.get("trace_kernel_names")
    if isinstance(trace_kernel_names, list):
        trace_lower_names = [name.lower() for name in trace_kernel_names if isinstance(name, str) and name]
        if not any(fragment in name for fragment in fragments for name in trace_lower_names):
            blockers.append("profiler.trace_kernel_names must include selected projection_dispatch candidate or variant")
    kernel_durations = profiler.get("kernel_durations_ns")
    if isinstance(kernel_durations, Mapping):
        duration_lower_names = [
            name.lower()
            for name, duration_ns in kernel_durations.items()
            if isinstance(name, str) and name and _is_finite_positive_number(duration_ns)
        ]
        if not any(fragment in name for fragment in fragments for name in duration_lower_names):
            blockers.append("profiler.kernel_durations_ns must include a positive selected projection_dispatch candidate or variant duration")
    profiler_names: list[str] = []
    for field in ("expected_kernel_names", "trace_kernel_names"):
        names = profiler.get(field)
        if isinstance(names, list):
            profiler_names.extend(name for name in names if isinstance(name, str) and name)
    if isinstance(kernel_durations, Mapping):
        profiler_names.extend(name for name in kernel_durations if isinstance(name, str) and name)
    lowered_names = [name.lower() for name in profiler_names]
    if not lowered_names or not any(fragment in name for fragment in fragments for name in lowered_names):
        blockers.append("profiler kernel names must include selected projection_dispatch candidate or variant")
    return blockers


def _load_sampler_equality_artifact(value: str) -> tuple[Mapping[str, Any] | None, str | None]:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.suffix.lower() != ".json":
        return None, "equality_artifact must point to a .json artifact"
    if path.is_symlink():
        return None, "equality_artifact must point to a regular JSON artifact, not a symlink"
    if _path_has_retained_results_symlink_parent(path):
        return None, "equality_artifact parent directories must not be symlinks"
    if not path.exists():
        return None, "equality_artifact must point to an existing JSON artifact"
    if not path.is_file():
        return None, "equality_artifact must point to a regular JSON artifact"
    try:
        payload = _load_json_path(path)
    except OSError as exc:
        return None, f"equality_artifact must point to a readable JSON artifact: {exc}"
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"equality_artifact must point to a valid JSON artifact: {exc}"
    if not isinstance(payload, Mapping):
        return None, "equality_artifact must point to a JSON object artifact"
    return payload, None


def _sampler_profiler_name_matches(kernel_name: str) -> bool:
    lowered = kernel_name.lower()
    if any(fragment in lowered for fragment in ("serial", "per_row", "per-row", "fallback")):
        return False
    return "batch" in lowered and _profiler_kernel_duration_category(kernel_name) == "sampling"


def _sampler_execution_profiler_blockers(batch_execution: Mapping[str, Any], profiler: Mapping[str, Any]) -> list[str]:
    decode_execution = batch_execution.get("decode_execution")
    sampler_execution = decode_execution.get("sampler_execution") if isinstance(decode_execution, Mapping) else None
    if not isinstance(sampler_execution, Mapping):
        return []
    if sampler_execution.get("mode") != "batched_lm_head" or sampler_execution.get("native_row_aware_lm_head") is not True:
        return []
    blockers: list[str] = []
    expected_kernel_names = profiler.get("expected_kernel_names")
    if isinstance(expected_kernel_names, list) and not any(
        isinstance(kernel_name, str) and _sampler_profiler_name_matches(kernel_name)
        for kernel_name in expected_kernel_names
    ):
        blockers.append("profiler.expected_kernel_names must include a native batch sampler/lm_head kernel")
    trace_kernel_names = profiler.get("trace_kernel_names")
    if isinstance(trace_kernel_names, list) and not any(
        isinstance(kernel_name, str) and _sampler_profiler_name_matches(kernel_name)
        for kernel_name in trace_kernel_names
    ):
        blockers.append("profiler.trace_kernel_names must include a native batch sampler/lm_head kernel")
    kernel_durations = profiler.get("kernel_durations_ns")
    if isinstance(kernel_durations, Mapping) and not any(
        isinstance(kernel_name, str)
        and _sampler_profiler_name_matches(kernel_name)
        and _is_finite_positive_number(duration_ns)
        for kernel_name, duration_ns in kernel_durations.items()
    ):
        blockers.append("profiler.kernel_durations_ns must include a positive native batch sampler/lm_head duration")
    return blockers


def _sampler_execution_blockers(batch_execution: Mapping[str, Any], *, expected_concurrency: int | None = None) -> list[str]:
    decode_execution = batch_execution.get("decode_execution")
    if not isinstance(decode_execution, Mapping):
        return ["execution.batch_execution.decode_execution is missing"]
    sampler_execution = decode_execution.get("sampler_execution")
    if not isinstance(sampler_execution, Mapping):
        return ["execution.batch_execution.decode_execution.sampler_execution is missing"]
    blockers: list[str] = []
    if expected_concurrency is not None:
        rows = sampler_execution.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int):
            blockers.append("execution.batch_execution.decode_execution.sampler_execution.rows must be an int")
        elif rows != int(expected_concurrency):
            blockers.append("execution.batch_execution.decode_execution.sampler_execution.rows must match workload.concurrency")
    if sampler_execution.get("requested_mode") != "batched_lm_head":
        blockers.append("execution.batch_execution.decode_execution.sampler_execution.requested_mode must be batched_lm_head")
    if sampler_execution.get("native_row_aware_lm_head") is not True:
        blockers.append("execution.batch_execution.decode_execution.sampler_execution.native_row_aware_lm_head must be true")
    if sampler_execution.get("mode") != "batched_lm_head":
        blockers.append("execution.batch_execution.decode_execution.sampler_execution.mode must be batched_lm_head")
    if sampler_execution.get("c2_equality_green") is not True:
        blockers.append("execution.batch_execution.decode_execution.sampler_execution.c2_equality_green must be true")
    if expected_concurrency is not None:
        equality_rows = sampler_execution.get("equality_rows")
        if isinstance(equality_rows, bool) or not isinstance(equality_rows, int):
            blockers.append("execution.batch_execution.decode_execution.sampler_execution.equality_rows must be an int")
        elif equality_rows != int(expected_concurrency):
            blockers.append("execution.batch_execution.decode_execution.sampler_execution.equality_rows must match workload.concurrency")
    equality_artifact = sampler_execution.get("equality_artifact")
    if not _is_retained_artifact_path(equality_artifact):
        blockers.append("execution.batch_execution.decode_execution.sampler_execution.equality_artifact must be under benchmarks/results")
    else:
        equality_artifact_payload, equality_artifact_error = _load_sampler_equality_artifact(str(equality_artifact))
        if equality_artifact_error is not None:
            blockers.append(f"execution.batch_execution.decode_execution.sampler_execution.{equality_artifact_error}")
        elif equality_artifact_payload is not None:
            artifact_expected_rows = expected_concurrency
            sampler_rows = sampler_execution.get("rows")
            if artifact_expected_rows is None and isinstance(sampler_rows, int) and not isinstance(sampler_rows, bool):
                artifact_expected_rows = sampler_rows
            if artifact_expected_rows is not None:
                blockers.extend(
                    batch_sampler_equality_payload_blockers(
                        equality_artifact_payload,
                        rows=int(artifact_expected_rows),
                        label="execution.batch_execution.decode_execution.sampler_execution.equality_artifact",
                        expected_artifact_path=str(equality_artifact),
                    )
                )
    if sampler_execution.get("blockers") != []:
        blockers.append("execution.batch_execution.decode_execution.sampler_execution.blockers must be empty")
    return blockers


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return None
    return [int(item) for item in value]


_NATIVE_C_GT_ONE_BF16_CONTEXT_BLOCKER = (
    "native c>N decode currently supports compact physical-slot-ordered rows; "
    "full-attention batch context is native only for BF16 KV and context < 1024"
)
_NATIVE_C_GT_ONE_GENERATED_EQUALITY_BLOCKER = (
    "native c>N decode is experimental and blocked until generated-token equality passes"
)


def _batch_execution_has_native_full_attention_context(
    batch_execution: Mapping[str, Any], *, kv_storage_dtype: str | None
) -> bool:
    decode_execution = batch_execution.get("decode_execution")
    if not isinstance(decode_execution, Mapping):
        return False
    if decode_execution.get("full_attention_decode_path") != "native_batch":
        return False
    max_context = decode_execution.get("max_full_attention_context")
    return (
        kv_storage_dtype == "bf16"
        and isinstance(max_context, int)
        and not isinstance(max_context, bool)
        and max_context < 1024
    )


def _batch_execution_with_satisfied_correctness_gates(
    batch_execution: Mapping[str, Any], *, equality_passed: bool, kv_storage_dtype: str | None
) -> dict[str, Any]:
    """Remove stale native c>N blockers once this artifact proves the gate."""

    sanitized = dict(batch_execution)
    if not equality_passed:
        return sanitized
    blockers = sanitized.get("blockers")
    if isinstance(blockers, list):
        stale_blockers = {_NATIVE_C_GT_ONE_GENERATED_EQUALITY_BLOCKER}
        if _batch_execution_has_native_full_attention_context(sanitized, kv_storage_dtype=kv_storage_dtype):
            stale_blockers.add(_NATIVE_C_GT_ONE_BF16_CONTEXT_BLOCKER)
        sanitized["blockers"] = [blocker for blocker in blockers if blocker not in stale_blockers]
    projection_dispatch = sanitized.get("projection_dispatch")
    projection_eligible = isinstance(projection_dispatch, Mapping) and projection_dispatch.get("throughput_claim_eligible") is True
    if (
        sanitized.get("native_compact_prefill") is True
        and sanitized.get("native_caware_decode") is True
        and _batch_execution_has_native_full_attention_context(sanitized, kv_storage_dtype=kv_storage_dtype)
        and sanitized.get("blockers") == []
        and projection_eligible
    ):
        sanitized["throughput_claim_eligible"] = True
    return sanitized


def _generated_token_equality_blockers(
    equality: Any,
    *,
    expected_concurrency: int,
    expected_decode_tokens: int,
    expected_warmup_decode_tokens: int,
) -> list[str]:
    if not isinstance(equality, Mapping):
        return ["correctness.generated_token_equality is missing"]
    blockers: list[str] = []
    if equality.get("passed") is not True:
        blockers.append("correctness.generated_token_equality.passed must be true")
    if equality.get("skipped") is not False:
        blockers.append("correctness.generated_token_equality.skipped must be false")
    equality_comparison = equality.get("comparison")
    if not isinstance(equality_comparison, str) or not equality_comparison:
        blockers.append("correctness.generated_token_equality.comparison must be a non-empty string")
    elif equality_comparison != "native_batch_vs_independent_c1":
        blockers.append("correctness.generated_token_equality.comparison must be native_batch_vs_independent_c1")
    equality_rows = equality.get("rows")
    if not isinstance(equality_rows, int) or isinstance(equality_rows, bool):
        blockers.append("correctness.generated_token_equality.rows must be an int")
    elif equality_rows != expected_concurrency:
        blockers.append("correctness.generated_token_equality.rows must match expected concurrency")
    expected_tokens = 1 + int(expected_warmup_decode_tokens) + int(expected_decode_tokens)
    tokens_per_sequence = equality.get("tokens_per_sequence")
    if not isinstance(tokens_per_sequence, int) or isinstance(tokens_per_sequence, bool):
        blockers.append("correctness.generated_token_equality.tokens_per_sequence must be an int")
    elif tokens_per_sequence != expected_tokens:
        blockers.append("correctness.generated_token_equality.tokens_per_sequence must match seed plus warmup plus decode tokens")
    equality_warmup_tokens = equality.get("warmup_decode_tokens")
    if not isinstance(equality_warmup_tokens, int) or isinstance(equality_warmup_tokens, bool):
        blockers.append("correctness.generated_token_equality.warmup_decode_tokens must be an int")
    elif equality_warmup_tokens != expected_warmup_decode_tokens:
        blockers.append("correctness.generated_token_equality.warmup_decode_tokens must match expected warmup decode tokens")
    equality_decode_tokens = equality.get("gen_tokens_per_request")
    if not isinstance(equality_decode_tokens, int) or isinstance(equality_decode_tokens, bool):
        blockers.append("correctness.generated_token_equality.gen_tokens_per_request must be an int")
    elif equality_decode_tokens != expected_decode_tokens:
        blockers.append("correctness.generated_token_equality.gen_tokens_per_request must match expected decode tokens")

    def _validate_sequences(field: str) -> Any:
        sequences = equality.get(field)
        label = f"correctness.generated_token_equality.{field}"
        if not isinstance(sequences, list):
            blockers.append(f"{label} is not a list")
            return sequences
        if len(sequences) != expected_concurrency:
            blockers.append(f"{label} length does not match expected concurrency")
        for row_index, sequence in enumerate(sequences):
            row_label = f"{label}[{row_index}]"
            if not isinstance(sequence, list):
                blockers.append(f"{row_label} is not a token id list")
                continue
            if len(sequence) != expected_tokens:
                blockers.append(f"{row_label} length does not match seed plus warmup plus decode tokens")
            if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in sequence):
                blockers.append(f"{row_label} contains a non-token id")
        return sequences

    batch_sequences = _validate_sequences("batch_sequences")
    c1_sequences = _validate_sequences("c1_sequences")
    if isinstance(batch_sequences, list) and isinstance(c1_sequences, list) and batch_sequences != c1_sequences:
        blockers.append("correctness.generated_token_equality.batch_sequences must equal c1_sequences")
    mismatches = equality.get("mismatches")
    if not isinstance(mismatches, list):
        blockers.append("correctness.generated_token_equality.mismatches is not a list")
    elif mismatches:
        blockers.append("correctness.generated_token_equality.mismatches must be empty")
    return blockers


def _token_payload_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    ids: list[int] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        token_id = item.get("token_id")
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            return None
        ids.append(int(token_id))
    return ids


def _execution_token_evidence_blockers(
    seed_tokens: Any,
    generated_tokens: Any,
    equality: Mapping[str, Any],
    *,
    expected_concurrency: int,
    expected_decode_tokens: int,
) -> list[str]:
    blockers: list[str] = []
    expected_keys = {str(request_id) for request_id in range(expected_concurrency)}
    batch_sequences = equality.get("batch_sequences")
    if not isinstance(seed_tokens, Mapping):
        blockers.append("execution.seed_tokens is missing")
    else:
        if set(seed_tokens.keys()) != expected_keys:
            blockers.append("execution.seed_tokens keys do not match expected row ids")
        for request_id in range(expected_concurrency):
            row = seed_tokens.get(str(request_id))
            label = f"execution.seed_tokens.{request_id}"
            if not isinstance(row, Mapping):
                blockers.append(f"{label} is not an object")
                continue
            token_id = row.get("token_id")
            if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
                blockers.append(f"{label}.token_id is not a non-negative int")
                continue
            if (
                isinstance(batch_sequences, list)
                and request_id < len(batch_sequences)
                and isinstance(batch_sequences[request_id], list)
                and batch_sequences[request_id]
                and token_id != batch_sequences[request_id][0]
            ):
                blockers.append(f"{label}.token_id does not match generated-token equality seed")
    if not isinstance(generated_tokens, Mapping):
        blockers.append("execution.generated_tokens is missing")
    else:
        if set(generated_tokens.keys()) != expected_keys:
            blockers.append("execution.generated_tokens keys do not match expected row ids")
        for request_id in range(expected_concurrency):
            label = f"execution.generated_tokens.{request_id}"
            token_ids = _token_payload_ids(generated_tokens.get(str(request_id)))
            if token_ids is None:
                blockers.append(f"{label} is not a token payload list")
                continue
            if len(token_ids) != expected_decode_tokens:
                blockers.append(f"{label} length does not match expected decode tokens")
            if (
                isinstance(batch_sequences, list)
                and request_id < len(batch_sequences)
                and isinstance(batch_sequences[request_id], list)
                and len(batch_sequences[request_id]) >= expected_decode_tokens
            ):
                expected_suffix = batch_sequences[request_id][-expected_decode_tokens:]
                suffix_is_ints = all(isinstance(token, int) and not isinstance(token, bool) for token in expected_suffix)
                if suffix_is_ints and token_ids != [int(token) for token in expected_suffix]:
                    blockers.append(f"{label} does not match generated-token equality decode suffix")
    return blockers


def _timing_measurement_blockers(
    bench: Mapping[str, Any],
    *,
    expected_decode_tokens: int,
    expected_warmup_decode_tokens: int,
) -> list[str]:
    blockers: list[str] = []
    for field in ("prefill_seconds", "decode_seconds"):
        if not _is_finite_positive_number(bench.get(field)):
            blockers.append(f"measurements.{field} is not positive")
    for field in ("load_seconds", "warmup_seconds"):
        if not _is_finite_nonnegative_number(bench.get(field)):
            blockers.append(f"measurements.{field} is not finite non-negative")

    def _validate_samples(field: str, expected_len: int, parent_seconds: str) -> None:
        samples = bench.get(field)
        if not isinstance(samples, list):
            blockers.append(f"measurements.{field} is not a list")
            return
        if len(samples) != expected_len:
            blockers.append(f"measurements.{field} length does not match expected token count")
        if any(not _is_finite_positive_number(sample) for sample in samples):
            blockers.append(f"measurements.{field} samples are not all positive")
            return
        parent_value = bench.get(parent_seconds)
        if _is_finite_nonnegative_number(parent_value):
            parent_float = float(parent_value)
            sample_sum = sum(float(sample) for sample in samples)
            if sample_sum - parent_float > max(1e-9, parent_float * 1e-6):
                blockers.append(f"measurements.{field} samples exceed {parent_seconds}")

    _validate_samples("decode_step_seconds", expected_decode_tokens, "decode_seconds")
    _validate_samples("warmup_step_seconds", expected_warmup_decode_tokens, "warmup_seconds")
    return blockers


def _bucket_key_axis(bucket_key: str, axis: str) -> str | None:
    prefix = f"{axis}="
    for segment in bucket_key.split(":"):
        if segment.startswith(prefix):
            value = segment[len(prefix) :]
            return value if value else None
    return None


def _request_observability_blockers(
    per_request: Any,
    *,
    expected_concurrency: int,
    expected_mode: str | None = None,
    expected_context_bucket: int | None = None,
    expected_context_buckets: set[int] | None = None,
    expected_active_mask: str | None = None,
    expected_kv_storage_dtype: str | None = None,
    expected_layer_plan: str | None = None,
    expected_top_k: int | None = None,
    expected_experts_per_token: int | None = None,
    expected_replay_steps: int | None = None,
    expected_draft_depth: int | None = None,
) -> list[str]:
    if not isinstance(per_request, Mapping):
        return ["observability.per_request is missing"]
    blockers: list[str] = []
    allowed_context_buckets: set[int] = set()
    if expected_context_bucket is not None:
        allowed_context_buckets.add(int(expected_context_bucket))
    if expected_context_buckets is not None:
        allowed_context_buckets.update(int(bucket) for bucket in expected_context_buckets)
    expected_keys = {str(request_id) for request_id in range(expected_concurrency)}
    if set(per_request.keys()) != expected_keys:
        blockers.append("observability.per_request keys do not match expected row ids")
    required_fields = (
        "queue_seconds",
        "prefill_seconds",
        "decode_seconds",
        "kv_pages_owned",
        "kv_pages_peak",
        "bucket_key",
        "admission_blocked_reason",
        "finish_reason",
        "admitted_timestamp",
        "completion_timestamp",
    )
    for request_id, row in per_request.items():
        label = f"observability.per_request.{request_id}"
        if not isinstance(row, Mapping):
            blockers.append(f"{label} is not an object")
            continue
        for field in required_fields:
            if field not in row:
                blockers.append(f"{label}.{field} is missing")
        for field in ("queue_seconds", "prefill_seconds", "decode_seconds", "admitted_timestamp", "completion_timestamp"):
            if field in row and not _is_finite_nonnegative_number(row.get(field)):
                blockers.append(f"{label}.{field} is unavailable or non-finite")
        admitted_timestamp = row.get("admitted_timestamp")
        completion_timestamp = row.get("completion_timestamp")
        if _is_finite_nonnegative_number(admitted_timestamp) and _is_finite_nonnegative_number(completion_timestamp):
            latency = float(completion_timestamp) - float(admitted_timestamp)
            if latency <= 0.0:
                blockers.append(f"{label}.completion_timestamp is not greater than admitted_timestamp")
            else:
                timing_components = (
                    row.get("queue_seconds"),
                    row.get("prefill_seconds"),
                    row.get("decode_seconds"),
                )
                if all(_is_finite_nonnegative_number(component) for component in timing_components):
                    component_total = sum(float(component) for component in timing_components)
                    if component_total - latency > max(1e-9, latency * 1e-6):
                        blockers.append(f"{label}.timing components exceed completion latency")
        for field in ("kv_pages_owned", "kv_pages_peak"):
            value = row.get(field)
            if field in row and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                blockers.append(f"{label}.{field} is not a non-negative int")
        kv_pages_owned = row.get("kv_pages_owned")
        kv_pages_peak = row.get("kv_pages_peak")
        if (
            isinstance(kv_pages_owned, int)
            and not isinstance(kv_pages_owned, bool)
            and kv_pages_owned >= 0
            and isinstance(kv_pages_peak, int)
            and not isinstance(kv_pages_peak, bool)
            and kv_pages_peak >= 0
            and kv_pages_peak < kv_pages_owned
        ):
            blockers.append(f"{label}.kv_pages_peak is below kv_pages_owned")
        bucket_key = row.get("bucket_key")
        if bucket_key is not None and (not isinstance(bucket_key, str) or not bucket_key.strip()):
            blockers.append(f"{label}.bucket_key is not a non-empty string or null")
        elif isinstance(bucket_key, str):
            mode_axis = bucket_key.split(":", 1)[0]
            c_axis = _bucket_key_axis(bucket_key, "c")
            ctx_axis = _bucket_key_axis(bucket_key, "ctx")
            mask_axis = _bucket_key_axis(bucket_key, "mask")
            kv_axis = _bucket_key_axis(bucket_key, "kv")
            layer_axis = _bucket_key_axis(bucket_key, "layers")
            top_k_axis = _bucket_key_axis(bucket_key, "top_k")
            experts_axis = _bucket_key_axis(bucket_key, "experts")
            replay_axis = _bucket_key_axis(bucket_key, "replay")
            draft_axis = _bucket_key_axis(bucket_key, "draft")
            if expected_mode is not None and mode_axis != str(expected_mode):
                blockers.append(f"{label}.bucket_key mode must match scheduler decode shape key")
            if c_axis is None or ctx_axis is None or mask_axis is None:
                blockers.append(f"{label}.bucket_key must include c, context, and active-mask axes")
            else:
                if c_axis != str(expected_concurrency):
                    blockers.append(f"{label}.bucket_key c axis must match expected concurrency")
                if allowed_context_buckets and ctx_axis not in {str(bucket) for bucket in allowed_context_buckets}:
                    blockers.append(f"{label}.bucket_key context axis must match scheduler observed decode shape key")
                if expected_active_mask is not None and mask_axis != str(expected_active_mask):
                    blockers.append(f"{label}.bucket_key active-mask axis must match scheduler decode shape key")
            if kv_axis is None or layer_axis is None:
                blockers.append(f"{label}.bucket_key must include kv and layer-plan axes")
            else:
                if expected_kv_storage_dtype is not None and kv_axis != str(expected_kv_storage_dtype):
                    blockers.append(f"{label}.bucket_key kv axis must match expected KV storage dtype")
                if expected_layer_plan is not None and layer_axis != str(expected_layer_plan):
                    blockers.append(f"{label}.bucket_key layer-plan axis must match expected layer plan")
            if top_k_axis is None or experts_axis is None or replay_axis is None or draft_axis is None:
                blockers.append(f"{label}.bucket_key must include top-k, experts, replay, and draft axes")
            else:
                if expected_top_k is not None and top_k_axis != str(expected_top_k):
                    blockers.append(f"{label}.bucket_key top-k axis must match scheduler decode shape key")
                if expected_experts_per_token is not None and experts_axis != str(expected_experts_per_token):
                    blockers.append(f"{label}.bucket_key experts axis must match scheduler decode shape key")
                if expected_replay_steps is not None and replay_axis != str(expected_replay_steps):
                    blockers.append(f"{label}.bucket_key replay axis must match scheduler decode shape key")
                if expected_draft_depth is not None and draft_axis != str(expected_draft_depth):
                    blockers.append(f"{label}.bucket_key draft axis must match scheduler decode shape key")
        admission_blocked_reason = row.get("admission_blocked_reason")
        if admission_blocked_reason is not None and (not isinstance(admission_blocked_reason, str) or not admission_blocked_reason.strip()):
            blockers.append(f"{label}.admission_blocked_reason is not a non-empty string or null")
        finish_reason = row.get("finish_reason")
        if finish_reason is not None and (not isinstance(finish_reason, str) or not finish_reason.strip()):
            blockers.append(f"{label}.finish_reason is not a non-empty string")
    return blockers


def _completed_execution_blockers(
    completed: Any,
    *,
    expected_concurrency: int,
    expected_prompt_lengths: Sequence[int],
    expected_decode_tokens: int,
    expected_warmup_decode_tokens: int,
    generated_tokens: Any,
    per_request_observability: Any,
) -> list[str]:
    if not isinstance(completed, list):
        return ["execution.completed is missing"]
    blockers: list[str] = []
    if len(completed) != expected_concurrency:
        blockers.append("execution.completed length does not match expected concurrency")
    generated_by_request = generated_tokens if isinstance(generated_tokens, Mapping) else {}
    per_request = per_request_observability if isinstance(per_request_observability, Mapping) else {}
    seen_request_ids: set[int] = set()
    for index, row in enumerate(completed):
        label = f"execution.completed[{index}]"
        if not isinstance(row, Mapping):
            blockers.append(f"{label} is not an object")
            continue
        request_id = row.get("request_id")
        if not isinstance(request_id, int) or isinstance(request_id, bool) or request_id < 0 or request_id >= expected_concurrency:
            blockers.append(f"{label}.request_id is not in expected range")
            continue
        if request_id in seen_request_ids:
            blockers.append("execution.completed request_id values are not unique")
        seen_request_ids.add(request_id)
        prompt_tokens = _int_list(row.get("prompt_tokens"))
        if prompt_tokens is None:
            blockers.append(f"{label}.prompt_tokens is not an int list")
        elif request_id < len(expected_prompt_lengths) and len(prompt_tokens) != int(expected_prompt_lengths[request_id]):
            blockers.append(f"{label}.prompt_tokens length does not match expected prompt length")
        completed_tokens = _int_list(row.get("generated_tokens"))
        if completed_tokens is None:
            blockers.append(f"{label}.generated_tokens is not an int list")
        else:
            expected_completed_tokens = int(expected_warmup_decode_tokens) + int(expected_decode_tokens)
            if len(completed_tokens) != expected_completed_tokens:
                blockers.append(f"{label}.generated_tokens length does not match expected warmup+decode tokens")
            expected_generated = _token_payload_ids(generated_by_request.get(str(request_id)))
            if expected_generated is not None and completed_tokens[-len(expected_generated):] != expected_generated:
                blockers.append(f"{label}.generated_tokens suffix does not match execution generated_tokens")
        if row.get("finished") is not True:
            blockers.append(f"{label}.finished is not true")
        finish_reason = row.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            blockers.append(f"{label}.finish_reason is missing")
        else:
            observed = per_request.get(str(request_id))
            if isinstance(observed, Mapping):
                observed_finish_reason = observed.get("finish_reason")
                if isinstance(observed_finish_reason, str) and observed_finish_reason != finish_reason:
                    blockers.append(f"{label}.finish_reason does not match observability")
    missing_request_ids = [str(request_id) for request_id in range(expected_concurrency) if request_id not in seen_request_ids]
    if missing_request_ids:
        blockers.append("execution.completed does not include every request_id")
    return blockers


def _memory_evidence_blockers(
    memory: Mapping[str, Any],
    *,
    expected_batch_size: int | None = None,
    expected_sequence_length: int | None = None,
    expected_kv_policy: Mapping[str, Any] | None = None,
    expected_kv_storage_dtype: str | None = None,
) -> list[str]:
    blockers: list[str] = []
    if expected_batch_size is not None and memory.get("max_batch_size") != expected_batch_size:
        blockers.append("memory.max_batch_size does not match expected batch size")
    max_sequence_length = memory.get("max_sequence_length")
    if expected_sequence_length is not None:
        if not isinstance(max_sequence_length, int) or isinstance(max_sequence_length, bool) or max_sequence_length < expected_sequence_length:
            blockers.append("memory.max_sequence_length does not cover expected sequence length")
    if expected_kv_storage_dtype is not None and memory.get("kv_storage_dtype") != expected_kv_storage_dtype:
        blockers.append("memory.kv_storage_dtype does not match expected KV storage dtype")
    memory_kv_policy = memory.get("kv_policy")
    if expected_kv_policy is not None:
        if not isinstance(memory_kv_policy, Mapping):
            blockers.append("memory.kv_policy is missing")
        elif dict(memory_kv_policy) != dict(expected_kv_policy):
            blockers.append("memory.kv_policy does not match expected KV policy")
    allocator_peak = memory.get("allocator_reserved_peak_bytes")
    if not _is_finite_nonnegative_number(allocator_peak):
        blockers.append("memory.allocator_reserved_peak_bytes is unavailable or non-finite")
    allocator_stats = memory.get("allocator_memory_stats")
    if not isinstance(allocator_stats, Mapping):
        blockers.append("memory.allocator_memory_stats is missing")
    else:
        stats_current = allocator_stats.get("current_allocated_bytes")
        stats_peak = allocator_stats.get("peak_allocated_bytes")
        if not _is_finite_nonnegative_number(stats_current):
            blockers.append("memory.allocator_memory_stats.current_allocated_bytes is unavailable or non-finite")
        if not _is_finite_nonnegative_number(stats_peak):
            blockers.append("memory.allocator_memory_stats.peak_allocated_bytes is unavailable or non-finite")
        elif _is_finite_nonnegative_number(allocator_peak) and int(stats_peak) != int(allocator_peak):
            blockers.append("memory.allocator_memory_stats.peak_allocated_bytes does not match allocator_reserved_peak_bytes")
        if _is_finite_nonnegative_number(stats_current) and _is_finite_nonnegative_number(stats_peak) and float(stats_current) > float(stats_peak):
            blockers.append("memory.allocator_memory_stats.current_allocated_bytes is above peak_allocated_bytes")
        total_allocated = allocator_stats.get("total_allocated_bytes")
        total_freed = allocator_stats.get("total_freed_bytes")
        for field, value in (("total_allocated_bytes", total_allocated), ("total_freed_bytes", total_freed)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                blockers.append(f"memory.allocator_memory_stats.{field} is not a non-negative int")
        active_allocations = allocator_stats.get("active_allocations")
        peak_allocations = allocator_stats.get("peak_allocations")
        for field, value in (("active_allocations", active_allocations), ("peak_allocations", peak_allocations)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                blockers.append(f"memory.allocator_memory_stats.{field} is not a non-negative int")
        if (
            isinstance(active_allocations, int)
            and not isinstance(active_allocations, bool)
            and active_allocations >= 0
            and isinstance(peak_allocations, int)
            and not isinstance(peak_allocations, bool)
            and peak_allocations >= 0
            and active_allocations > peak_allocations
        ):
            blockers.append("memory.allocator_memory_stats.active_allocations is above peak_allocations")
    dynamic_pool = memory.get("dynamic_pool")
    if not isinstance(dynamic_pool, Mapping):
        blockers.append("memory.dynamic_pool evidence is missing")
    else:
        if not isinstance(dynamic_pool.get("enabled"), bool):
            blockers.append("memory.dynamic_pool.enabled is not bool")
        evidence = dynamic_pool.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            blockers.append("memory.dynamic_pool.evidence is missing")
        pool_counters = dynamic_pool.get("pool_counters")
        required_counters = (
            "current_bytes",
            "high_water_observed_bytes",
            "grow_events",
            "grow_failures",
            "shrink_events",
            "free_pages",
            "refcounted_pages",
        )
        if not isinstance(pool_counters, Mapping):
            blockers.append("memory.dynamic_pool.pool_counters is missing")
        else:
            for field in required_counters:
                value = pool_counters.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    blockers.append(f"memory.dynamic_pool.pool_counters.{field} is not a non-negative int")
            current_bytes = pool_counters.get("current_bytes")
            high_water_bytes = pool_counters.get("high_water_observed_bytes")
            if (
                _is_finite_nonnegative_number(current_bytes)
                and _is_finite_nonnegative_number(high_water_bytes)
                and float(high_water_bytes) < float(current_bytes)
            ):
                blockers.append("memory.dynamic_pool.pool_counters.high_water_observed_bytes is below current_bytes")
        for field in ("grow_events", "shrink_events"):
            value = dynamic_pool.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                blockers.append(f"memory.dynamic_pool.{field} is not a non-negative int")
            elif isinstance(pool_counters, Mapping):
                counter_value = pool_counters.get(field)
                if isinstance(counter_value, int) and not isinstance(counter_value, bool) and counter_value >= 0 and value != counter_value:
                    blockers.append(f"memory.dynamic_pool.{field} does not match pool_counters.{field}")
    stable_block_id = memory.get("stable_block_id")
    if not isinstance(stable_block_id, Mapping):
        blockers.append("memory.stable_block_id evidence is missing")
    else:
        if stable_block_id.get("passed") is not True:
            blockers.append("memory.stable_block_id.passed is not true")
        audit = stable_block_id.get("audit")
        if not isinstance(audit, str) or not audit.strip():
            blockers.append("memory.stable_block_id.audit is missing")
    prefix_sharing = memory.get("prefix_sharing")
    if not isinstance(prefix_sharing, Mapping):
        blockers.append("memory.prefix_sharing evidence is missing")
    else:
        prefix_enabled = prefix_sharing.get("enabled")
        prefix_savings = prefix_sharing.get("savings_bytes")
        if not isinstance(prefix_enabled, bool):
            blockers.append("memory.prefix_sharing.enabled is not bool")
        if not _is_finite_nonnegative_number(prefix_savings):
            blockers.append("memory.prefix_sharing.savings_bytes is unavailable or non-finite")
        elif prefix_enabled is False and float(prefix_savings) != 0.0:
            blockers.append("memory.prefix_sharing.savings_bytes is nonzero while prefix sharing is disabled")
    return blockers


def _run_capture(command: Sequence[str], *, timeout: float = 5.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": proc.returncode,
            "output": proc.stdout.strip(),
        }
    except Exception as exc:  # pragma: no cover - best-effort environment capture
        return {
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": None,
            "output": f"{type(exc).__name__}: {exc}",
        }


def _software_context() -> dict[str, Any]:
    commit = _run_capture(["git", "rev-parse", "HEAD"])
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False).returncode != 0
    return {
        "python": sys.version.split()[0],
        "hipcc_version": _run_capture(["hipcc", "--version"], timeout=10.0)["output"],
        "hipengine_commit": commit["output"],
        "hipengine_dirty": dirty,
        "torch_rocm": _run_capture(
            ["python3", "-c", "import torch; print(torch.__version__, torch.version.hip)"],
            timeout=10.0,
        ),
    }


def _visible_hip_device_context() -> dict[str, Any]:
    env_keys = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
    visible_env: dict[str, str] = {}
    for key in env_keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            visible_env[key] = value
    context: dict[str, Any] = {"env": visible_env}
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        hip.hipGetDeviceCount.restype = ctypes.c_int
        count = ctypes.c_int()
        count_error = int(hip.hipGetDeviceCount(ctypes.byref(count)))
        context["hipGetDeviceCount_error"] = count_error
        context["visible_device_count"] = int(count.value)
        if count_error != 0 or count.value <= 0:
            return context

        hip.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        hip.hipGetDevice.restype = ctypes.c_int
        current_device = ctypes.c_int()
        device_error = int(hip.hipGetDevice(ctypes.byref(current_device)))
        context["hipGetDevice_error"] = device_error
        device_index = int(current_device.value) if device_error == 0 else 0
        context["current_device"] = device_index

        hip.hipDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        hip.hipDeviceGetName.restype = ctypes.c_int
        name = ctypes.create_string_buffer(256)
        name_error = int(hip.hipDeviceGetName(name, ctypes.c_int(len(name)), ctypes.c_int(device_index)))
        context["hipDeviceGetName_error"] = name_error
        if name_error == 0:
            context["device_name"] = name.value.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - best-effort benchmark provenance.
        context["error"] = f"{type(exc).__name__}: {exc}"
    return context


def _hardware_context() -> dict[str, Any]:
    visible_device = _visible_hip_device_context()
    visible_device_name = visible_device.get("device_name")
    gpu_name = visible_device_name if isinstance(visible_device_name, str) and visible_device_name else "AMD Radeon Pro W7900"
    return {
        "gpu": gpu_name,
        "arch": "gfx1100",
        "default_hardware": gpu_name == "AMD Radeon Pro W7900",
        "visible_device": visible_device,
        "rocminfo": _run_capture(["bash", "-lc", "rocminfo | grep -E 'Name:|gfx' | head -4"], timeout=10.0),
        "rocm_smi": _run_capture(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"], timeout=10.0),
    }


def _command_env_prefix_parts() -> list[str]:
    assignments = [
        f"{key}={value}"
        for key in _COMMAND_ENV_KEYS
        if (value := os.environ.get(key)) is not None and value.strip()
    ]
    return ["env", *assignments] if assignments else []


def _command(argv: Sequence[str] | None) -> str:
    parts = [*_command_env_prefix_parts(), "python3", _RETAINED_BENCH_SCRIPT]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _primitive_correctness_command(path: Path | None, *, rows: int, seed: int = 1234) -> str:
    parts = [
        *_command_env_prefix_parts(),
        "python3",
        _PRIMITIVE_CORRECTNESS_SCRIPT,
        "--rows",
        str(rows),
        "--seed",
        str(seed),
        "--json",
    ]
    parts.append(str(path) if path is not None else "<primitive-correctness-json>")
    return " ".join(shlex.quote(part) for part in parts)


def _compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text()


def _decode_scheduler_step_native(
    session: Qwen35ParoResidentSession,
    scheduler: ResidentBatchScheduler,
    next_token_by_request: dict[int, int],
    generated_by_request: dict[int, list[dict[str, Any]]],
    *,
    count_output: bool,
    kv_storage_dtype: str = "bf16",
    layer_plan: str = "all",
    scheduler_metadata: dict[str, Any] | None = None,
    device_resident: bool = False,
) -> tuple[int, bool]:
    work = scheduler.next_decode_work(
        kv_storage_dtype=kv_storage_dtype,
        layer_plan=layer_plan,
    )
    if work is None:
        raise RuntimeError("scheduler did not emit decode work")
    if scheduler_metadata is not None:
        shape_payload = _shape_key_payload(
            scheduler.shape_key(
                mode="decode",
                kv_storage_dtype=kv_storage_dtype,
                layer_plan=layer_plan,
            )
        )
        observed_shapes = scheduler_metadata.setdefault("decode_shape_keys_observed", [])
        if isinstance(observed_shapes, list) and shape_payload not in observed_shapes:
            observed_shapes.append(shape_payload)
    request_ids = tuple(request_id for request_id in work.request_ids if request_id in next_token_by_request)
    slots = [scheduler.active_batch.slot_for(request_id) for request_id in request_ids]
    if tuple(slots) != tuple(range(len(slots))):
        raise RuntimeError(f"native retained benchmark requires compact slots, got {slots!r}")
    results = session.step_batch_native(
        [next_token_by_request[request_id] for request_id in request_ids],
        positions=[scheduler.active_batch.requests[request_id].context_len for request_id in request_ids],
        slots=slots,
        sample=True,
        device_resident=device_resident,
    )
    generated: list[GeneratedToken] = []
    for request_id, result in zip(request_ids, results, strict=True):
        if result is None:
            raise RuntimeError("decode did not produce a token")
        next_token_by_request[request_id] = result.token_id
        if count_output:
            generated_by_request[request_id].append(_result_payload(result))
        generated.append(GeneratedToken(request_id, result.token_id))
    scheduler.record_generated(generated)
    return len(results), True


def _run_native_bench(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    max_layers: int,
    warmup_decode_tokens: int,
    decode_tokens: int,
    compiler_version: str | None,
    require_cached_build: bool,
    kv_policy: ResolvedKVPolicy,
    device_resident: bool = False,
) -> dict[str, Any]:
    batch_size = len(prompts)
    prompt_lengths = {len(prompt) for prompt in prompts}
    if len(prompt_lengths) != 1:
        raise ValueError("current benchmark expects equal prompt lengths")
    prompt_length = prompt_lengths.pop()
    max_sequence_length = prompt_length + warmup_decode_tokens + decode_tokens + 1
    scheduler = ResidentBatchScheduler(capacity=batch_size)
    request_ids = [scheduler.submit(prompt, max_new_tokens=warmup_decode_tokens + decode_tokens) for prompt in prompts]
    admitted = scheduler.admit_pending()
    if admitted != tuple(request_ids):
        raise RuntimeError(f"unexpected admitted request ids {admitted!r}")

    seed_by_request: dict[int, Any] = {}
    generated_by_request: dict[int, list[dict[str, Any]]] = {request_id: [] for request_id in request_ids}
    measured_step_seconds: list[float] = []
    warmup_step_seconds: list[float] = []
    scheduler_metadata: dict[str, Any] = {
        "request_ids": list(request_ids),
        "admitted": list(admitted),
        "slot_to_request_after_admit": list(scheduler.active_batch.slot_to_request),
        "active_count_after_admit": scheduler.active_count,
        "prefill_slabs": [],
        "decode_native_steps": 0,
    }

    load_start = time.perf_counter()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        max_batch_size=batch_size,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        load_seconds = time.perf_counter() - load_start

        prefill_start = time.perf_counter()
        slabs = scheduler.next_compact_prefill_slabs(chunk_size=prompt_length, block_size=session.block_size)
        for slab in slabs:
            scheduler_metadata["prefill_slabs"].append(
                {
                    "request_ids": list(slab.request_ids),
                    "slot_ids": list(slab.physical_slot_ids),
                    "rows": slab.rows,
                    "request_count": slab.request_count,
                    "block_count": slab.block_count,
                }
            )
            results = session.prefill_native_packed(slab, sample=True)
            for request_id, result in zip(slab.request_ids, results, strict=True):
                if result is None:
                    raise RuntimeError("prefill did not produce a seed token")
                seed_by_request[request_id] = result
        prefill_seconds = time.perf_counter() - prefill_start

        if set(seed_by_request) != set(request_ids):
            raise RuntimeError("missing one or more prefill seed tokens")

        _record_decode_graph_bucket_metadata(
            scheduler,
            scheduler_metadata,
            kv_storage_dtype=kv_policy.storage_dtype.value,
            layer_plan=f"max_layers={int(max_layers)}",
        )
        next_token_by_request = {request_id: seed_by_request[request_id].token_id for request_id in request_ids}
        warmup_start = time.perf_counter()
        for _ in range(warmup_decode_tokens):
            step_start = time.perf_counter()
            _count, native = _decode_scheduler_step_native(
                session,
                scheduler,
                next_token_by_request,
                generated_by_request,
                count_output=False,
                kv_storage_dtype=kv_policy.storage_dtype.value,
                layer_plan=f"max_layers={int(max_layers)}",
                scheduler_metadata=scheduler_metadata,
                device_resident=device_resident,
            )
            scheduler_metadata["decode_native_steps"] += int(native)
            warmup_step_seconds.append(time.perf_counter() - step_start)
        warmup_seconds = time.perf_counter() - warmup_start

        decode_start = time.perf_counter()
        for _ in range(decode_tokens):
            step_start = time.perf_counter()
            _count, native = _decode_scheduler_step_native(
                session,
                scheduler,
                next_token_by_request,
                generated_by_request,
                count_output=True,
                kv_storage_dtype=kv_policy.storage_dtype.value,
                layer_plan=f"max_layers={int(max_layers)}",
                scheduler_metadata=scheduler_metadata,
                device_resident=device_resident,
            )
            scheduler_metadata["decode_native_steps"] += int(native)
            measured_step_seconds.append(time.perf_counter() - step_start)
        decode_seconds = time.perf_counter() - decode_start
        completed = list(scheduler.completed.values())
        scheduler_metadata["active_count_after_completion"] = scheduler.active_count
        scheduler_metadata["slot_to_request_after_completion"] = list(scheduler.active_batch.slot_to_request)
        scheduler_metadata["graph_bucket_stats"] = scheduler.graph_buckets.stats.to_json_dict()
        batch_execution = session.batch_execution_metadata(
            scheduler_owned=True,
            native_decode=True,
            active_rows=batch_size,
        ).to_json_dict()

    completed_payload = [done.to_json_dict() for done in completed]
    request_observability = {
        str(done.request_id): done.observability.to_json_dict()
        for done in completed
    }
    seed_rows = [_result_payload(seed_by_request[request_id]) for request_id in request_ids]
    generated_rows = [row for rows in generated_by_request.values() for row in rows]
    finite_logits = _all_finite(seed_rows) and _all_finite(generated_rows)
    return {
        "load_seconds": load_seconds,
        "prefill_seconds": prefill_seconds,
        "warmup_seconds": warmup_seconds,
        "decode_seconds": decode_seconds,
        "warmup_step_seconds": warmup_step_seconds,
        "decode_step_seconds": measured_step_seconds,
        "seed_tokens": {str(request_id): _result_payload(seed_by_request[request_id]) for request_id in request_ids},
        "generated_tokens": {str(request_id): generated_by_request[request_id] for request_id in request_ids},
        "scheduler_metadata": scheduler_metadata,
        "batch_execution": batch_execution,
        "completed": completed_payload,
        "request_observability": request_observability,
        "finite_logits": finite_logits,
        "memory": _allocator_memory_evidence(memory_stats()),
    }


def _run_c1_reference_tokens(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    total_decode_tokens: int,
    max_layers: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
    kv_policy: ResolvedKVPolicy,
) -> list[list[int]]:
    rows: list[list[int]] = []
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        for prompt in prompts:
            scheduler = ResidentBatchScheduler(capacity=1)
            request_id = scheduler.submit(prompt, max_new_tokens=total_decode_tokens)
            admitted = scheduler.admit_pending()
            if admitted != (request_id,):
                raise RuntimeError(f"unexpected c=1 admitted request ids {admitted!r}")
            slabs = scheduler.next_compact_prefill_slabs(chunk_size=len(prompt), block_size=session.block_size)
            if len(slabs) != 1:
                raise RuntimeError("c=1 reference expected one compact prefill slab")
            seed = session.prefill_native_packed(slabs[0], sample=True)[0]
            if seed is None:
                raise RuntimeError("c=1 prefill did not produce a seed token")
            token_ids = [int(seed.token_id)]
            next_token = int(seed.token_id)
            for offset in range(total_decode_tokens):
                result = session.step_batch_native(
                    [next_token],
                    positions=[len(prompt) + offset],
                    slots=[0],
                    sample=True,
                )[0]
                if result is None:
                    raise RuntimeError("c=1 decode did not produce a token")
                next_token = int(result.token_id)
                token_ids.append(next_token)
            rows.append(token_ids)
            session.reset()
    return rows


def _generated_sequences_from_bench(bench: dict[str, Any], request_ids: Sequence[int]) -> list[list[int]]:
    rows: list[list[int]] = []
    completed_by_id = {int(row["request_id"]): row for row in bench.get("completed", [])}
    for request_id in request_ids:
        seed = int(bench["seed_tokens"][str(request_id)]["token_id"])
        if request_id in completed_by_id:
            generated = [int(token) for token in completed_by_id[request_id]["generated_tokens"]]
        else:
            generated = [int(item["token_id"]) for item in bench["generated_tokens"][str(request_id)]]
        rows.append([seed, *generated])
    return rows


def _equal_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    prefix = 0
    for left_token, right_token in zip(left, right, strict=False):
        if int(left_token) != int(right_token):
            break
        prefix += 1
    return prefix


def _token_window(sequence: Sequence[int], center: int, *, radius: int = 5) -> list[int]:
    if center < 0:
        center = 0
    start = max(0, int(center) - int(radius))
    end = min(len(sequence), int(center) + int(radius) + 1)
    return [int(token) for token in sequence[start:end]]


def _generated_token_equality_prefix_summary(
    batch_sequences: Sequence[Sequence[int]],
    c1_sequences: Sequence[Sequence[int]],
) -> dict[str, Any]:
    prefixes: list[int] = []
    first_mismatch_indices: list[int | None] = []
    for batch, c1 in zip(batch_sequences, c1_sequences, strict=False):
        prefix = _equal_prefix_length(batch, c1)
        prefixes.append(prefix)
        first_mismatch_indices.append(None if prefix == min(len(batch), len(c1)) and len(batch) == len(c1) else prefix)
    return {
        "prefix_lengths": prefixes,
        "min_equal_prefix_tokens": min(prefixes) if prefixes else 0,
        "first_mismatch_indices": first_mismatch_indices,
    }


def _generated_token_equality_mismatch_summaries(
    batch_sequences: Sequence[Sequence[int]],
    c1_sequences: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    c1_rows = [[int(token) for token in row] for row in c1_sequences]
    batch_rows = [[int(token) for token in row] for row in batch_sequences]
    for row, (batch, c1) in enumerate(zip(batch_rows, c1_rows, strict=False)):
        if batch == c1:
            continue
        first = _equal_prefix_length(batch, c1)
        batch_prefixes_by_c1_row = [
            {"row": c1_row, "equal_prefix_tokens": _equal_prefix_length(batch, other_c1)}
            for c1_row, other_c1 in enumerate(c1_rows)
        ]
        batch_prefixes_by_c1_row.sort(key=lambda item: (-int(item["equal_prefix_tokens"]), int(item["row"])))
        summaries.append(
            {
                "row": row,
                "first_mismatch_index": first,
                "batch_token_at_mismatch": int(batch[first]) if first < len(batch) else None,
                "c1_token_at_mismatch": int(c1[first]) if first < len(c1) else None,
                "batch_window": _token_window(batch, first),
                "c1_window": _token_window(c1, first),
                "batch_matches_c1_rows": [c1_row for c1_row, other_c1 in enumerate(c1_rows) if batch == other_c1],
                "batch_matches_other_batch_rows": [
                    batch_row for batch_row, other_batch in enumerate(batch_rows) if batch_row != row and batch == other_batch
                ],
                "batch_prefixes_by_c1_row": batch_prefixes_by_c1_row,
            }
        )
    return summaries


def _default_projection_dispatch_artifact_arg() -> str | None:
    path = Path.cwd() / _DEFAULT_PROJECTION_DISPATCH_ARTIFACT
    if not path.is_file() or path.is_symlink():
        return None
    return _DEFAULT_PROJECTION_DISPATCH_ARTIFACT


def _projection_dispatch_artifact_arg(args: argparse.Namespace) -> str | None:
    artifact = getattr(args, "projection_dispatch_artifact", None)
    if artifact is None:
        raw_artifact = os.environ.get(_PROJECTION_DISPATCH_ARTIFACT_ENV)
        if raw_artifact is None or not raw_artifact.strip():
            artifact = _default_projection_dispatch_artifact_arg()
            if artifact is None:
                return None
        else:
            artifact = raw_artifact.strip()
    artifact_text = str(artifact).strip()
    if not artifact_text:
        return None
    if not _is_retained_artifact_path(artifact_text):
        raise ValueError(
            f"invalid projection dispatch artifact {artifact_text}: must be a relative path under benchmarks/results"
        )
    return artifact_text


def _projection_dispatch_artifact_file_path(artifact: str) -> Path:
    path = Path(artifact)
    check_path = path if path.is_absolute() else Path.cwd() / path
    root_path = Path.cwd() / "benchmarks" / "results"
    parent = check_path.parent
    while True:
        if parent.is_symlink():
            raise ValueError(
                f"invalid projection dispatch artifact {artifact}: parent directories must not be symlinks"
            )
        if parent == root_path or parent == parent.parent:
            break
        parent = parent.parent
    if check_path.is_symlink():
        raise ValueError(f"invalid projection dispatch artifact {artifact}: must not be a symlink")
    if not check_path.is_file():
        raise ValueError(f"invalid projection dispatch artifact {artifact}: must point to a regular JSON artifact")
    return check_path


def _resolved_batch_decode_linear_projection_path(args: argparse.Namespace) -> str:
    path = str(getattr(args, "batch_decode_linear_projection_path", "auto"))
    if path != "auto":
        return path
    batch_size = getattr(args, "batch_size", 2)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        batch_size = 2
    # The c>N no-selected batch projection path is generated-token green for
    # c=2/c=4/c=8 after the FP16 QKV/Z batch-GEMV reduction moved to 128
    # threads. Keep larger, unproven row counts on the older full selected-c1
    # replay fallback.
    return "batch" if int(batch_size) <= 8 else "selected_c1"


def _resolved_batch_decode_moe_path(args: argparse.Namespace) -> str:
    if not hasattr(args, "batch_decode_moe_path"):
        return "grouped_compact"
    path = str(getattr(args, "batch_decode_moe_path", "grouped_compact"))
    if path != "auto":
        return path
    # The retained-claim gate requires grouped-compact MoE, and the current
    # correctness frontier is generated-token green for c=2/c=4/c=8 with grouped-
    # compact. Keep selected-c1 as an explicit speed/diagnostic override rather
    # than the correctness-first auto default.
    return "grouped_compact"


def _resolved_batch_decode_full_attn_path(args: argparse.Namespace) -> str:
    if not hasattr(args, "batch_decode_full_attn_path"):
        return "native_batch"
    path = str(getattr(args, "batch_decode_full_attn_path", "native_batch"))
    if path != "auto":
        return path
    # Native batch full-attention is generated-token green for c=2 as one
    # batch. For c>=3, prompt/window-sensitive rows can fail when three or
    # more rows share one native full-attention group, so the correctness-first
    # auto path keeps native kernels but caps row groups below.
    return "native_batch"


def _resolved_batch_decode_full_attn_row_chunk_size(args: argparse.Namespace) -> int:
    explicit = int(getattr(args, "batch_decode_full_attn_row_chunk_size", 0) or 0)
    if explicit != 0:
        return explicit
    if str(getattr(args, "batch_decode_full_attn_path", "native_batch")) != "auto":
        return 0
    batch_size = getattr(args, "batch_size", 0)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        batch_size = 0
    # c=3..c=8 diagnostics show native full-attention is generated-token green
    # when split into <=2-row native chunks, while larger groups reproduce
    # prompt/window-sensitive failures. c=4 has a narrow no-rowchunk selected-
    # dense-context repair for the first-four fixture, but the derived rows4..7
    # fixture is only green with rowchunk2, so correctness-first auto rowchunks
    # c=4 as well.
    return 2 if int(batch_size) in {3, 4, 5, 6, 7, 8} else 0


def _resolved_batch_decode_full_attn_row_chunk_layers(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "batch_decode_full_attn_row_chunk_layers", "") or "").strip()
    if explicit:
        return explicit
    if str(getattr(args, "batch_decode_full_attn_path", "native_batch")) != "auto":
        return ""
    batch_size = getattr(args, "batch_size", 0)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        batch_size = 0
    # Original c3/c5/c6 evidence shows rowchunking only the first four full-
    # attention producer layers preserves generated-token equality for those
    # correctness-first auto row counts while smaller tested subsets remain red.
    # C4 layer 11 alone is repeat/profile-green on the primary first-four
    # fixture, but the hard rows4..7 fixture invalidated it as a prompt-stable
    # default. Current primary and hard rows4..7 probes both pass when only the
    # final four full-attention producer layers stay native, so c4 uses first-
    # six. A current c8 first-nine/drop39 profiler recaptured row3/token118 red
    # in the older all-layer scope, but the newer drop27/drop31 default keeps
    # three explicit drop39 repeats green; prior broader drop35 evidence is red.
    # Keep c7 all-layer until its selected-layer path is profiler-stable; use
    # the c8 drop27/drop31/drop39 rowchunk diagnostic as the current narrowest
    # repeated-green c8 scope.
    if int(batch_size) in {3, 5, 6}:
        return "3,7,11,15"
    if int(batch_size) == 4:
        return "3,15"
    if int(batch_size) == 8:
        return "3,7,11,15,19,23"
    return ""


def _resolved_batch_decode_full_attn_output_path(args: argparse.Namespace) -> str:
    path = str(getattr(args, "batch_decode_full_attn_output_path", "batch"))
    if path != "batch":
        return path
    if _resolved_batch_decode_full_attn_row_chunk_layers(args):
        # Layer-scoped c4 rowchunk leaves later full-attention layers on the
        # native path, so keep the already accepted row-aware batch-GEMV O
        # projection for all full-attention layers in that narrowed diagnostic.
        return "batch_gemv"
    return path


def _resolved_batch_decode_attn_context_path(args: argparse.Namespace) -> str:
    path = str(getattr(args, "batch_decode_attn_context_path", "batch"))
    if path != "batch":
        return path
    return path


def _resolved_batch_decode_attn_dense_context_batch_gate_layers(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "batch_decode_attn_dense_context_batch_gate_layers", "") or "").strip()
    if explicit:
        return explicit
    if str(getattr(args, "batch_decode_full_attn_path", "native_batch")) != "auto":
        return ""
    if str(getattr(args, "batch_decode_attn_context_path", "batch")) != "batch":
        return ""
    # Auto correctness now prefers rowchunk2 for c=4/c=8 hard windows. Keep
    # selected dense-context layers as an explicit diagnostic override only.
    return ""


def _apply_runtime_env_args(args: argparse.Namespace) -> None:
    os.environ["HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"] = "1"
    batch_decode_moe_path = _resolved_batch_decode_moe_path(args)
    force_selected_c1_moe = batch_decode_moe_path == "selected_c1"
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE"] = "1" if force_selected_c1_moe else "0"
    os.environ["HIPENGINE_QWEN35_SHARED_EXPERT_PARO_W4_FORCE_GEMV"] = "1" if force_selected_c1_moe else "0"
    os.environ["HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR"] = (
        "1" if getattr(args, "batch_prefill_linear_path", "packed_segments") == "per_segment" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR"] = (
        "1" if getattr(args, "batch_decode_linear_path", "batch_segments") == "per_row" else "0"
    )
    linear_projection_path = _resolved_batch_decode_linear_projection_path(args)
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_PROJECTIONS"] = (
        "1" if linear_projection_path == "selected_c1" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ"] = (
        "1" if linear_projection_path == "selected_qkv_z" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ_INPUT"] = (
        "1" if linear_projection_path == "selected_qkv_z_input" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKV"] = (
        "1" if linear_projection_path == "selected_qkv" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_Z"] = (
        "1" if linear_projection_path == "selected_z" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_AB"] = (
        "1" if linear_projection_path in {"selected_ab", "batch_gemv_selected_ab"} else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_LINEAR_PROJECTIONS"] = (
        "1" if linear_projection_path in {"batch_gemv", "batch_gemv_selected_ab"} else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_STATE"] = (
        "1" if getattr(args, "batch_decode_linear_state_path", "batch_segments") == "selected_c1" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR_MOE"] = (
        "1" if (not force_selected_c1_moe and getattr(args, "batch_decode_linear_moe_path", "grouped_compact") == "per_row_c1") else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT"] = str(
        getattr(args, "batch_decode_linear_output_path", "batch_gemv")
    )
    batch_decode_full_attn_path = _resolved_batch_decode_full_attn_path(args)
    os.environ["HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE"] = "0" if batch_decode_full_attn_path == "per_row" else "1"
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_SIZE"] = str(
        _resolved_batch_decode_full_attn_row_chunk_size(args)
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_LAYERS"] = _resolved_batch_decode_full_attn_row_chunk_layers(args)
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT"] = (
        "1" if getattr(args, "batch_decode_attn_input_path", "batch") == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_QKV"] = (
        "1" if getattr(args, "batch_decode_attn_qkv_path", "batch") == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SCRATCH"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_BATCH_SCRATCH"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "per_row_batch_scratch" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_MOE"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "per_row_attn_batch_moe" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_POST_MOE"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "per_row_attn_batch_post_moe" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_O_POST_MOE"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "per_row_attn_batch_o_post_moe" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_BATCH_CONTEXT_O_POST_MOE"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "per_row_preqkv_append_batch_context_o_post_moe" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_CONTEXT_BATCH_GATE_O_POST_MOE"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "per_row_preqkv_append_context_batch_gate_o_post_moe" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_CONTEXT_GATE_BATCH_O_POST_MOE"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "per_row_preqkv_append_context_gate_batch_o_post_moe" else "0"
    )
    batch_decode_attn_context_path = _resolved_batch_decode_attn_context_path(args)
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PERSISTENT_SCRATCH"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") in {"persistent_c1", "persistent_c1_no_batch_setup"} else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SKIP_BATCH_SETUP"] = (
        "1" if getattr(args, "batch_decode_attn_scratch_path", "batch") == "persistent_c1_no_batch_setup" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_CONTEXT"] = (
        "1" if batch_decode_attn_context_path == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_CONTEXT_ONLY"] = (
        "1" if batch_decode_attn_context_path == "per_row_context_only" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_ONLY"] = (
        "1" if batch_decode_attn_context_path == "per_row_dense_context_only" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_BATCH_GATE"] = (
        "1" if batch_decode_attn_context_path == "per_row_dense_context_batch_gate" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_BATCH_GATE_LAYERS"] = (
        _resolved_batch_decode_attn_dense_context_batch_gate_layers(args)
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PAGED_CONTEXT_ONLY"] = (
        "1" if batch_decode_attn_context_path == "per_row_paged_context_only" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_BATCH_TEMP_FULL_ATTN_CONTEXT"] = (
        "1" if batch_decode_attn_context_path == "batch_temp_output" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_BATCH_COMPACT_FULL_ATTN_CONTEXT"] = (
        "1" if batch_decode_attn_context_path == "batch_compact_cache" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_GATE"] = (
        "1" if getattr(args, "batch_decode_attn_gate_path", "batch") == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_KV_APPEND"] = (
        "1" if getattr(args, "batch_decode_full_attn_kv_append_path", "batch") == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_APPEND_CONTEXT"] = (
        "1" if getattr(args, "batch_decode_attn_append_context_order", "phased") == "interleaved" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SUFFIX"] = (
        "1" if getattr(args, "batch_decode_attn_suffix_order", "phased") == "interleaved" else "0"
    )
    batch_full_attention_output_path = _resolved_batch_decode_full_attn_output_path(args)
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_OUTPUT"] = (
        "1" if batch_full_attention_output_path == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_FULL_ATTN_OUTPUT"] = (
        "1" if batch_full_attention_output_path == "batch_gemv" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_NATIVE_FULL_ATTN_OUTPUT"] = (
        "1" if batch_full_attention_output_path == "native" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_NATIVE_ROW_CHUNK_FULL_ATTN_OUTPUT"] = (
        "1" if batch_full_attention_output_path == "native_row_chunk" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_LAYER_COPY"] = (
        "1" if getattr(args, "batch_decode_full_attn_layer_copy", "batch") == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_MOE"] = (
        "1" if (not force_selected_c1_moe and getattr(args, "batch_decode_full_attn_moe_path", "grouped_compact") == "per_row_c1") else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN"] = (
        "1" if getattr(args, "batch_decode_post_attn_path", "batch") == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_MODE"] = str(getattr(args, "batch_sample_mode", "serial_lm_head"))
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_NORM_PATH"] = str(getattr(args, "batch_sample_norm_path", "batch"))
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_CAST_PATH"] = str(getattr(args, "batch_sample_cast_path", "auto"))
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_ARGMAX_MODE"] = str(getattr(args, "batch_sample_argmax_mode", "batch"))
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_ARGMAX_AUDIT"] = "1" if getattr(args, "batch_sample_argmax_audit", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_LM_HEAD_AUDIT"] = "1" if getattr(args, "batch_sample_lm_head_audit", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_LM_HEAD_KERNEL_FENCE"] = "1" if getattr(args, "batch_sample_lm_head_kernel_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_NORM_AUDIT"] = "1" if getattr(args, "batch_sample_final_norm_audit", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_NORM_KERNEL_FENCE"] = "1" if getattr(args, "batch_sample_final_norm_kernel_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_RMSNORM_KERNEL_FENCE"] = "1" if getattr(args, "batch_sample_final_rmsnorm_kernel_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_RMSNORM_TEMP_FENCE"] = "1" if getattr(args, "batch_sample_final_rmsnorm_temp_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_TEMP_FENCE"] = "1" if getattr(args, "batch_sample_final_cast_temp_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_TINY_FENCE"] = "1" if getattr(args, "batch_sample_final_cast_tiny_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_ELEMS_FENCE"] = str(max(0, int(getattr(args, "batch_sample_final_cast_elems_fence", 0) or 0)))
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_STABILIZE_CAST_ELEMS"] = str(max(0, int(getattr(args, "batch_sample_stabilize_cast_elems", 0) or 0)))
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_SYNC_FENCE"] = "1" if getattr(args, "batch_sample_sync_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_SUFFIX_FENCE"] = "1" if getattr(args, "batch_sample_suffix_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_SUFFIX_KERNEL_FENCE"] = "1" if getattr(args, "batch_sample_suffix_kernel_fence", False) else "0"
    os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_C2_EQ_OK"] = (
        "1" if getattr(args, "batch_sample_eq_ok", False) else "0"
    )
    batch_sample_eq_artifact = getattr(args, "batch_sample_eq_artifact", None)
    if batch_sample_eq_artifact is None:
        os.environ.pop("HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ARTIFACT", None)
    else:
        os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ARTIFACT"] = str(batch_sample_eq_artifact)
    batch_sample_eq_rows = getattr(args, "batch_sample_eq_rows", None)
    if batch_sample_eq_rows is None:
        os.environ.pop("HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ROWS", None)
    else:
        os.environ["HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ROWS"] = str(batch_sample_eq_rows)
    projection_dispatch_artifact = _projection_dispatch_artifact_arg(args)
    if projection_dispatch_artifact is not None:
        os.environ[_PROJECTION_DISPATCH_ARTIFACT_ENV] = projection_dispatch_artifact


def _projection_candidate_evidence_artifact_errors(candidate: Any) -> list[str]:
    evidence = getattr(candidate, "evidence", None)
    if evidence is None:
        return []
    payload, error = _load_retained_json_artifact(evidence.artifact_path)
    if error is not None:
        return [error]
    if payload is None:
        return []
    errors: list[str] = []
    if not _retained_artifact_accepted(payload):
        errors.append("artifact_path artifact must be accepted")
    artifact_rows = _retained_artifact_row_count(payload)
    if isinstance(artifact_rows, bool) or not isinstance(artifact_rows, int):
        errors.append("artifact_path rows must be an int")
    elif not candidate.applies_to(artifact_rows):
        errors.append("artifact_path rows must be within candidate row bounds")
    else:
        errors.extend(
            projection_dispatch_evidence_payload_blockers(
                payload,
                evidence,
                rows=artifact_rows,
                label="artifact_path",
            )
        )
    return errors


def _projection_dispatch_candidates_for_payload(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    artifact = _projection_dispatch_artifact_arg(args)
    if artifact is None:
        return None
    path = _projection_dispatch_artifact_file_path(artifact)
    try:
        payload = _load_json_path(path)
    except OSError as exc:
        raise ValueError(f"invalid projection dispatch artifact {artifact}: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid projection dispatch artifact {artifact}: must be valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid projection dispatch artifact {artifact}: root must be a JSON object")
    try:
        candidates = projection_dispatch_candidates_from_artifact(payload)
    except ValueError as exc:
        raise ValueError(f"invalid projection dispatch artifact {artifact}: {exc}") from exc
    if not candidates:
        raise ValueError(
            f"invalid projection dispatch artifact {artifact}: must include projection_dispatch_candidates"
        )
    evidence_errors: list[str] = []
    for candidate in candidates:
        for error in _projection_candidate_evidence_artifact_errors(candidate):
            evidence_errors.append(f"{candidate.name} evidence {error}")
    if evidence_errors:
        raise ValueError(
            f"invalid projection dispatch artifact {artifact}: " + "; ".join(evidence_errors)
        )
    return [candidate.to_json_dict() for candidate in candidates]


def _build_payload(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    bench: dict[str, Any],
    prompt_lengths: list[int],
    equality: dict[str, Any],
) -> dict[str, Any]:
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    aggregate_prefill_tokens = args.batch_size * args.prompt_length
    aggregate_decode_tokens = args.batch_size * args.decode_tokens
    prefill_tok_s = aggregate_prefill_tokens / bench["prefill_seconds"] if bench["prefill_seconds"] > 0 else None
    decode_tok_s = aggregate_decode_tokens / bench["decode_seconds"] if bench["decode_seconds"] > 0 and aggregate_decode_tokens else None
    decode_tok_s_per_request = decode_tok_s / args.batch_size if decode_tok_s is not None else None
    scaling = _build_scaling_comparison(
        args,
        native_decode_tok_s_aggregate=decode_tok_s,
        native_decode_tok_s_per_request=decode_tok_s_per_request,
    )
    primitive_correctness_path = getattr(args, "primitive_correctness_json", None)
    primitive_correctness = _primitive_correctness_reference(
        primitive_correctness_path,
        rows=args.batch_size,
    )
    int8_kv_primitive_layer_accuracy: dict[str, Any] | None = None
    int8_kv_primitive_blockers: list[str] = []
    if kv_policy.storage_dtype.value == "int8_per_token_head":
        cpu_reference = _int8_kv_primitive_layer_accuracy_reference(
            getattr(args, "int8_kv_primitive_cpu_json", None),
            device="cpu",
            prompt_length=args.prompt_length,
            scale_dtype=str(args.kv_scale_dtype),
        )
        hip_gate = _int8_kv_primitive_layer_accuracy_reference(
            getattr(args, "int8_kv_primitive_hip_json", None),
            device="hip",
            prompt_length=args.prompt_length,
            scale_dtype=str(args.kv_scale_dtype),
        )
        int8_kv_primitive_layer_accuracy = {
            "cpu_reference": cpu_reference,
            "hip_gate": hip_gate,
        }
        int8_kv_primitive_blockers.extend(
            _int8_kv_primitive_layer_accuracy_blockers(cpu_reference, label="cpu_reference")
        )
        int8_kv_primitive_blockers.extend(
            _int8_kv_primitive_layer_accuracy_blockers(hip_gate, label="hip_gate")
        )
    primitive_seed = primitive_correctness.get("seed")
    correctness_reference_seed = primitive_seed if isinstance(primitive_seed, int) and not isinstance(primitive_seed, bool) else 1234
    correctness_reference_command = _primitive_correctness_command(
        primitive_correctness_path,
        rows=args.batch_size,
        seed=correctness_reference_seed,
    )
    profiler = _attach_profiler_cpu_side_bottlenecks(
        _profiler_reference(getattr(args, "profiler_json", None)),
        bench,
    )
    profiled_command = _profiled_command(args, argv, profiler)
    retained_artifact_path = str(args.json) if args.json is not None else None
    scheduler_metadata = dict(bench["scheduler_metadata"])
    _attach_profiler_graph_kernel_time_histogram(scheduler_metadata, profiler)
    profiler_captured = profiler.get("status") == "captured" and profiler.get("expected_kernels_present") is True
    profiler_blockers = _profiler_provenance_blockers(
        profiler,
        profiled_command=profiled_command,
        retained_artifact_path=retained_artifact_path,
        expected_workload={
            "batch_size": args.batch_size,
            "prompt_length": args.prompt_length,
            "decode_tokens": args.decode_tokens,
            "warmup_decode_tokens": args.warmup_decode_tokens,
            "max_layers": args.max_layers,
        },
        expected_inputs={
            key: value
            for key, value in {
                "model": str(getattr(args, "model", "")),
                "fixture": str(getattr(args, "fixture", "")),
            }.items()
            if value
        },
        expected_build={
            "compiler_version_file": str(args.compiler_version_file) if getattr(args, "compiler_version_file", None) is not None else None,
            "require_cached_build": bool(getattr(args, "require_cached_build", False)),
        },
        expected_references={
            "c1_baseline_json": str(args.c1_baseline_json) if getattr(args, "c1_baseline_json", None) is not None else None,
            "serial_bridge_json": str(args.serial_bridge_json) if getattr(args, "serial_bridge_json", None) is not None else None,
            "primitive_correctness_json": str(args.primitive_correctness_json) if getattr(args, "primitive_correctness_json", None) is not None else None,
        },
        expected_kv_policy={
            "kv_storage": str(getattr(args, "kv_storage", "auto")),
            "kv_scale_dtype": str(getattr(args, "kv_scale_dtype", "fp16")),
            "kv_scale_granularity": str(getattr(args, "kv_scale_granularity", "per_token_head")),
        },
        expected_sampler={
            "batch_sample_mode": str(getattr(args, "batch_sample_mode", "serial_lm_head")),
            "batch_sample_norm_path": str(getattr(args, "batch_sample_norm_path", "batch")),
            "batch_sample_cast_path": str(getattr(args, "batch_sample_cast_path", "auto")),
            "batch_sample_argmax_mode": str(getattr(args, "batch_sample_argmax_mode", "batch")),
            "batch_sample_argmax_audit": bool(getattr(args, "batch_sample_argmax_audit", False)),
            "batch_sample_lm_head_audit": bool(getattr(args, "batch_sample_lm_head_audit", False)),
            "batch_sample_lm_head_kernel_fence": bool(getattr(args, "batch_sample_lm_head_kernel_fence", False)),
            "batch_sample_final_norm_audit": bool(getattr(args, "batch_sample_final_norm_audit", False)),
            "batch_sample_final_norm_kernel_fence": bool(getattr(args, "batch_sample_final_norm_kernel_fence", False)),
            "batch_sample_final_rmsnorm_kernel_fence": bool(getattr(args, "batch_sample_final_rmsnorm_kernel_fence", False)),
            "batch_sample_final_rmsnorm_temp_fence": bool(getattr(args, "batch_sample_final_rmsnorm_temp_fence", False)),
            "batch_sample_final_cast_temp_fence": bool(getattr(args, "batch_sample_final_cast_temp_fence", False)),
            "batch_sample_final_cast_tiny_fence": bool(getattr(args, "batch_sample_final_cast_tiny_fence", False)),
            "batch_sample_final_cast_elems_fence": max(0, int(getattr(args, "batch_sample_final_cast_elems_fence", 0) or 0)),
            "batch_sample_stabilize_cast_elems": max(0, int(getattr(args, "batch_sample_stabilize_cast_elems", 0) or 0)),
            "batch_sample_sync_fence": bool(getattr(args, "batch_sample_sync_fence", False)),
            "batch_sample_suffix_fence": bool(getattr(args, "batch_sample_suffix_fence", False)),
            "batch_sample_suffix_kernel_fence": bool(getattr(args, "batch_sample_suffix_kernel_fence", False)),
            "batch_sample_eq_ok": bool(getattr(args, "batch_sample_eq_ok", False)),
            "batch_sample_eq_artifact": str(getattr(args, "batch_sample_eq_artifact", "") or ""),
            "batch_sample_eq_rows": getattr(args, "batch_sample_eq_rows", None),
        },
    )
    profiler_blockers.extend(_profiler_synthesized_fields_blockers(profiler))
    profiler_blockers.extend(_profiler_kernel_evidence_blockers(profiler))
    profiler_blockers.extend(_profiler_cpu_side_bottleneck_blockers(profiler))
    equality_structure_blockers = _generated_token_equality_blockers(
        equality,
        expected_concurrency=args.batch_size,
        expected_decode_tokens=args.decode_tokens,
        expected_warmup_decode_tokens=args.warmup_decode_tokens,
    )
    output_artifact_blockers = _retained_output_artifact_blockers(getattr(args, "json", None))
    primitive_path_blockers = _retained_json_artifact_path_blockers(
        "primitive_correctness_json",
        primitive_correctness_path,
    )
    scaling_path_blockers = [
        *_retained_json_artifact_path_blockers(
            "c1_baseline_json",
            getattr(args, "c1_baseline_json", None),
        ),
        *_retained_json_artifact_path_blockers(
            "serial_bridge_json",
            getattr(args, "serial_bridge_json", None),
        ),
    ]
    profiler_path_blockers = _retained_json_artifact_path_blockers(
        "profiler_json",
        getattr(args, "profiler_json", None),
    )
    cached_build_blockers = _cached_build_provenance_blockers(args)
    command_provenance_blockers = _retained_command_provenance_blockers(args, argv)
    batch_execution = _batch_execution_with_satisfied_correctness_gates(
        bench["batch_execution"],
        equality_passed=bool(equality.get("passed")),
        kv_storage_dtype=kv_policy.storage_dtype.value,
    )
    throughput_claim_eligible = bool(batch_execution.get("throughput_claim_eligible"))
    native_caware_decode = bool(batch_execution.get("native_caware_decode"))
    batch_execution_blockers = _batch_execution_blockers(
        batch_execution,
        expected_max_layers=args.max_layers,
        expected_concurrency=args.batch_size,
        expected_prompt_length=args.prompt_length,
    )
    projection_dispatch_candidates = bench.get("projection_dispatch_candidates")
    projection_blockers = _projection_dispatch_blockers(
        batch_execution,
        concurrency=args.batch_size,
        candidates=projection_dispatch_candidates,
    )
    projection_blockers.extend(_projection_dispatch_profiler_blockers(batch_execution, profiler))
    sampler_blockers = _sampler_execution_blockers(batch_execution, expected_concurrency=args.batch_size)
    sampler_blockers.extend(_sampler_execution_profiler_blockers(batch_execution, profiler))
    memory = _retained_memory_payload(args, kv_policy, bench)
    memory_blockers = _memory_evidence_blockers(
        memory,
        expected_batch_size=args.batch_size,
        expected_sequence_length=args.prompt_length + args.warmup_decode_tokens + args.decode_tokens + 1,
        expected_kv_policy=kv_policy_json(kv_policy),
        expected_kv_storage_dtype=kv_policy.storage_dtype.value,
    )
    raw_per_request_observability = bench.get("request_observability")
    per_request_observability = (
        dict(raw_per_request_observability)
        if isinstance(raw_per_request_observability, Mapping)
        else raw_per_request_observability
    )
    decode_shape_key = scheduler_metadata.get("decode_shape_key")
    expected_mode = None
    expected_context_bucket = None
    expected_active_mask = None
    expected_top_k = None
    expected_experts_per_token = None
    expected_replay_steps = None
    expected_draft_depth = None
    expected_context_buckets: set[int] = set()
    if isinstance(decode_shape_key, Mapping):
        mode = decode_shape_key.get("mode")
        if isinstance(mode, str) and mode:
            expected_mode = mode
        context_bucket = decode_shape_key.get("context_bucket")
        if isinstance(context_bucket, int) and not isinstance(context_bucket, bool):
            expected_context_bucket = int(context_bucket)
            expected_context_buckets.add(int(context_bucket))
        active_mask = decode_shape_key.get("active_mask")
        if isinstance(active_mask, list) and all(isinstance(active, bool) for active in active_mask):
            expected_active_mask = "".join("1" if active else "0" for active in active_mask)
        top_k = decode_shape_key.get("top_k")
        if isinstance(top_k, int) and not isinstance(top_k, bool):
            expected_top_k = int(top_k)
        experts_per_token = decode_shape_key.get("experts_per_token")
        if isinstance(experts_per_token, int) and not isinstance(experts_per_token, bool):
            expected_experts_per_token = int(experts_per_token)
        replay_steps = decode_shape_key.get("replay_steps")
        if isinstance(replay_steps, int) and not isinstance(replay_steps, bool):
            expected_replay_steps = int(replay_steps)
        draft_depth = decode_shape_key.get("draft_depth")
        if isinstance(draft_depth, int) and not isinstance(draft_depth, bool):
            expected_draft_depth = int(draft_depth)
    observed_decode_shape_keys = scheduler_metadata.get("decode_shape_keys_observed")
    if isinstance(observed_decode_shape_keys, list):
        for observed_shape_key in observed_decode_shape_keys:
            if not isinstance(observed_shape_key, Mapping):
                continue
            observed_context_bucket = observed_shape_key.get("context_bucket")
            if isinstance(observed_context_bucket, int) and not isinstance(observed_context_bucket, bool):
                expected_context_buckets.add(int(observed_context_bucket))
    observability_blockers = _request_observability_blockers(
        per_request_observability,
        expected_concurrency=args.batch_size,
        expected_mode=expected_mode,
        expected_context_bucket=expected_context_bucket,
        expected_context_buckets=expected_context_buckets,
        expected_active_mask=expected_active_mask,
        expected_kv_storage_dtype=kv_policy.storage_dtype.value,
        expected_layer_plan=f"max_layers={int(args.max_layers)}",
        expected_top_k=expected_top_k,
        expected_experts_per_token=expected_experts_per_token,
        expected_replay_steps=expected_replay_steps,
        expected_draft_depth=expected_draft_depth,
    )
    token_evidence_blockers = _execution_token_evidence_blockers(
        bench.get("seed_tokens"),
        bench.get("generated_tokens"),
        equality,
        expected_concurrency=args.batch_size,
        expected_decode_tokens=args.decode_tokens,
    )
    measurement_blockers = _timing_measurement_blockers(
        bench,
        expected_decode_tokens=args.decode_tokens,
        expected_warmup_decode_tokens=args.warmup_decode_tokens,
    )
    completed_blockers = _completed_execution_blockers(
        bench.get("completed"),
        expected_concurrency=args.batch_size,
        expected_prompt_lengths=prompt_lengths,
        expected_decode_tokens=args.decode_tokens,
        expected_warmup_decode_tokens=args.warmup_decode_tokens,
        generated_tokens=bench.get("generated_tokens"),
        per_request_observability=per_request_observability,
    )
    graph_bucket_blockers = _decode_shape_key_blockers(
        scheduler_metadata,
        concurrency=args.batch_size,
        prompt_length=args.prompt_length,
        kv_storage_dtype=kv_policy.storage_dtype.value,
        layer_plan=f"max_layers={int(args.max_layers)}",
    )
    graph_bucket_blockers.extend(_graph_bucket_evidence_blockers(scheduler_metadata))
    graph_bucket_blockers.extend(_graph_replay_profiler_evidence_blockers(scheduler_metadata, profiler))
    equality_passed = bool(equality.get("passed"))
    protocol_shape = args.max_layers == 40 and args.prompt_length >= 512 and args.decode_tokens >= 128
    scaling_complete = bool(scaling["complete"])
    scaling_performance_blockers = _scaling_performance_blockers(scaling) if scaling_complete else []
    primitive_passed = bool(primitive_correctness["passed"])
    int8_kv_primitive_passed = not int8_kv_primitive_blockers
    accepted = bool(
        bench["finite_logits"]
        and throughput_claim_eligible
        and equality_passed
        and not equality_structure_blockers
        and not output_artifact_blockers
        and not primitive_path_blockers
        and not scaling_path_blockers
        and not profiler_path_blockers
        and not cached_build_blockers
        and not command_provenance_blockers
        and primitive_passed
        and protocol_shape
        and scaling_complete
        and not scaling_performance_blockers
        and profiler_captured
        and not profiler_blockers
        and not batch_execution_blockers
        and not projection_blockers
        and not sampler_blockers
        and not memory_blockers
        and not observability_blockers
        and not token_evidence_blockers
        and not measurement_blockers
        and not completed_blockers
        and not graph_bucket_blockers
        and not int8_kv_primitive_blockers
    )
    primitive_loaded = primitive_correctness.get("status") == "loaded"
    correctness_rejected = bool(bench["finite_logits"] and (not equality_passed or (primitive_loaded and not primitive_passed)))
    status = "accepted" if accepted else ("rejected_correctness" if correctness_rejected else "blocked")
    blocked_reasons: list[str] = []
    if not throughput_claim_eligible:
        blocked_reasons.append("batch_execution.throughput_claim_eligible=false")
    if not equality_passed:
        blocked_reasons.append("generated-token equality vs independent c=1 did not pass")
    blocked_reasons.extend(equality_structure_blockers)
    blocked_reasons.extend(output_artifact_blockers)
    blocked_reasons.extend(primitive_path_blockers)
    blocked_reasons.extend(scaling_path_blockers)
    blocked_reasons.extend(profiler_path_blockers)
    blocked_reasons.extend(cached_build_blockers)
    blocked_reasons.extend(command_provenance_blockers)
    if not primitive_passed:
        blocked_reasons.append(f"primitive c>N correctness gate did not pass: {primitive_correctness.get('reason')}")
    if args.prompt_length < 512 or args.decode_tokens < 128:
        blocked_reasons.append("workload is a reduced diagnostic shape, not the docs/BENCHMARK.md c=N 512/128 protocol")
    if args.max_layers != 40:
        blocked_reasons.append("max_layers is not the full 40-layer Qwen3.5/PARO model")
    if not scaling_complete:
        blocked_reasons.append("scaling comparison vs c=1 and serial bridge baselines is incomplete")
    blocked_reasons.extend(scaling_performance_blockers)
    if not profiler_captured:
        blocked_reasons.append("profiler trace was not captured with expected kernels present")
    blocked_reasons.extend(profiler_blockers)
    if not bench["finite_logits"]:
        blocked_reasons.append("non-finite seed or decode logits")
    blocked_reasons.extend(batch_execution_blockers)
    blocked_reasons.extend(projection_blockers)
    blocked_reasons.extend(sampler_blockers)
    blocked_reasons.extend(memory_blockers)
    blocked_reasons.extend(observability_blockers)
    blocked_reasons.extend(token_evidence_blockers)
    blocked_reasons.extend(measurement_blockers)
    blocked_reasons.extend(completed_blockers)
    blocked_reasons.extend(graph_bucket_blockers)
    blocked_reasons.extend(int8_kv_primitive_blockers)
    if not isinstance(per_request_observability, Mapping):
        per_request_observability = {}
    admission_timestamps = {
        request_id: row.get("admitted_timestamp")
        for request_id, row in per_request_observability.items()
        if isinstance(row, dict)
    }
    completion_timestamps = {
        request_id: row.get("completion_timestamp")
        for request_id, row in per_request_observability.items()
        if isinstance(row, dict)
    }
    request_latencies: list[float] = []
    for request_id in sorted(per_request_observability, key=_request_id_sort_key):
        row = per_request_observability[request_id]
        if not isinstance(row, dict):
            continue
        admitted_timestamp = row.get("admitted_timestamp")
        completion_timestamp = row.get("completion_timestamp")
        if _is_finite_nonnegative_number(admitted_timestamp) and _is_finite_nonnegative_number(completion_timestamp):
            latency = float(completion_timestamp) - float(admitted_timestamp)
            if latency > 0.0:
                request_latencies.append(latency)
    latency_summary = _summarize_samples(request_latencies)
    payload = {
        "schema": 3,
        "mode": _ACCEPTED_MODE,
        "status": status,
        "rows": args.batch_size,
        "artifact_path": str(args.json) if args.json is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": f"qwen35-paro-c{args.batch_size}-native-retained",
        "summary": _ACCEPTED_SUMMARY,
        "performance_claim": accepted,
        "hardware": _hardware_context(),
        "software": _software_context(),
        "workload": {
            "shape": f"c={args.batch_size} prompt={args.prompt_length} decode={args.decode_tokens}",
            "model": "Qwen3.5/3.6-35B-A3B-PARO",
            "model_path": str(Path(args.model)),
            "fixture_path": str(Path(args.fixture)) if getattr(args, "fixture", None) is not None else "",
            "quant": "w4_paro",
            "prompt_tokens_per_request": args.prompt_length,
            "prompt_tokens_aggregate": aggregate_prefill_tokens,
            "gen_tokens_per_request": args.decode_tokens,
            "gen_tokens_aggregate": aggregate_decode_tokens,
            "warmup_decode_tokens": args.warmup_decode_tokens,
            "concurrency": args.batch_size,
            "prompt_lengths": prompt_lengths,
            "max_layers": args.max_layers,
            "kv_policy": kv_policy_json(kv_policy),
            "kv_storage_dtype": kv_policy.storage_dtype.value,
            "scheduler_path": "scheduler_native_compact_batch",
            "native_compact_prefill": True,
            "native_caware_decode": native_caware_decode,
            "batch_prefill_linear_path": str(getattr(args, "batch_prefill_linear_path", "packed_segments")),
            "batch_decode_moe_path": _resolved_batch_decode_moe_path(args),
            "batch_decode_linear_path": str(getattr(args, "batch_decode_linear_path", "batch_segments")),
            "batch_decode_linear_projection_path": _resolved_batch_decode_linear_projection_path(args),
            "batch_decode_linear_state_path": str(getattr(args, "batch_decode_linear_state_path", "batch_segments")),
            "batch_decode_linear_moe_path": str(getattr(args, "batch_decode_linear_moe_path", "grouped_compact")),
            "batch_decode_linear_output_path": str(getattr(args, "batch_decode_linear_output_path", "batch_gemv")),
            "batch_decode_full_attention_path": _resolved_batch_decode_full_attn_path(args),
            "batch_decode_full_attention_row_chunk_size": _resolved_batch_decode_full_attn_row_chunk_size(args),
            "batch_decode_full_attention_row_chunk_layers": _resolved_batch_decode_full_attn_row_chunk_layers(args),
            "batch_decode_attention_input_path": str(getattr(args, "batch_decode_attn_input_path", "batch")),
            "batch_decode_attention_qkv_path": str(getattr(args, "batch_decode_attn_qkv_path", "batch")),
            "batch_decode_attention_scratch_path": str(getattr(args, "batch_decode_attn_scratch_path", "batch")),
            "batch_decode_attention_context_path": _resolved_batch_decode_attn_context_path(args),
            "batch_decode_attention_dense_context_batch_gate_layers": _resolved_batch_decode_attn_dense_context_batch_gate_layers(args),
            "batch_decode_attention_gate_path": str(getattr(args, "batch_decode_attn_gate_path", "batch")),
            "batch_decode_full_attention_kv_append_path": str(getattr(args, "batch_decode_full_attn_kv_append_path", "batch")),
            "batch_decode_attention_append_context_order": str(getattr(args, "batch_decode_attn_append_context_order", "phased")),
            "batch_decode_attention_suffix_order": str(getattr(args, "batch_decode_attn_suffix_order", "phased")),
            "batch_decode_full_attention_output_path": _resolved_batch_decode_full_attn_output_path(args),
            "batch_decode_full_attention_layer_copy": str(getattr(args, "batch_decode_full_attn_layer_copy", "batch")),
            "batch_decode_full_attention_moe_path": str(getattr(args, "batch_decode_full_attn_moe_path", "grouped_compact")),
            "batch_decode_post_attention_path": str(getattr(args, "batch_decode_post_attn_path", "batch")),
            "batch_sample_mode": str(getattr(args, "batch_sample_mode", "serial_lm_head")),
            "batch_sample_norm_path": str(getattr(args, "batch_sample_norm_path", "batch")),
            "batch_sample_cast_path": str(getattr(args, "batch_sample_cast_path", "auto")),
            "batch_sample_argmax_mode": str(getattr(args, "batch_sample_argmax_mode", "batch")),
            "batch_sample_argmax_audit": bool(getattr(args, "batch_sample_argmax_audit", False)),
            "batch_sample_lm_head_audit": bool(getattr(args, "batch_sample_lm_head_audit", False)),
            "batch_sample_lm_head_kernel_fence": bool(getattr(args, "batch_sample_lm_head_kernel_fence", False)),
            "batch_sample_final_norm_audit": bool(getattr(args, "batch_sample_final_norm_audit", False)),
            "batch_sample_final_norm_kernel_fence": bool(getattr(args, "batch_sample_final_norm_kernel_fence", False)),
            "batch_sample_final_rmsnorm_kernel_fence": bool(getattr(args, "batch_sample_final_rmsnorm_kernel_fence", False)),
            "batch_sample_final_rmsnorm_temp_fence": bool(getattr(args, "batch_sample_final_rmsnorm_temp_fence", False)),
            "batch_sample_final_cast_temp_fence": bool(getattr(args, "batch_sample_final_cast_temp_fence", False)),
            "batch_sample_final_cast_tiny_fence": bool(getattr(args, "batch_sample_final_cast_tiny_fence", False)),
            "batch_sample_final_cast_elems_fence": max(0, int(getattr(args, "batch_sample_final_cast_elems_fence", 0) or 0)),
            "batch_sample_stabilize_cast_elems": max(0, int(getattr(args, "batch_sample_stabilize_cast_elems", 0) or 0)),
            "batch_sample_sync_fence": bool(getattr(args, "batch_sample_sync_fence", False)),
            "batch_sample_suffix_fence": bool(getattr(args, "batch_sample_suffix_fence", False)),
            "batch_sample_suffix_kernel_fence": bool(getattr(args, "batch_sample_suffix_kernel_fence", False)),
            "batch_sample_eq_ok": bool(getattr(args, "batch_sample_eq_ok", False)),
            "batch_sample_eq_artifact": (
                str(getattr(args, "batch_sample_eq_artifact", "") or "")
            ),
            "batch_sample_eq_rows": getattr(args, "batch_sample_eq_rows", None),
        },
        "benchmark_rollup": {
            "artifact_path": str(args.json) if args.json is not None else None,
            "source_artifact_path": str(args.json) if args.json is not None else None,
            "readme_path": "benchmarks/README.md",
            "changelog_path": "benchmarks/CHANGELOG.md",
        },
        "commands": {
            "environment": [
                "rocminfo | grep -E 'Name:|gfx' | head -4",
                "rocm-smi --showmeminfo vram --showuse --showtemp",
                "hipcc --version",
                "git rev-parse HEAD",
                "git diff --quiet",
            ],
            "correctness_reference": f"inline generated-token equality vs independent c=1 plus {correctness_reference_command}",
            "benchmark": _command(argv),
            "profiler": profiled_command,
        },
        "correctness": {
            "passed": bool(bench["finite_logits"] and equality_passed and primitive_passed and int8_kv_primitive_passed),
            "oracle": "generated-token ids equal independent c=1 resident runs through the same native packed prefill/decode path plus scripts/qwen35_batch_correctness.py primitive GPU correctness for the same c>N row count",
            "finite_logits": bool(bench["finite_logits"]),
            "generated_token_equality": equality,
            "primitive_batch_correctness": primitive_correctness,
            "kl_mean": None,
            "top1_agreement": None,
        },
        "execution": {
            "batch_execution": batch_execution,
            "scheduler_metadata": scheduler_metadata,
            "completed": bench["completed"],
            "seed_tokens": bench["seed_tokens"],
            "generated_tokens": bench["generated_tokens"],
        },
        "observability": {
            "admission_timestamps": admission_timestamps,
            "completion_timestamps": completion_timestamps,
            "request_latency_seconds": {
                "p50": latency_summary["median"],
                "p95": latency_summary["p95"],
                "samples": latency_summary["samples"],
            },
            "per_request": per_request_observability,
        },
        "measurements": {
            "load_seconds": bench["load_seconds"],
            "prefill_seconds": bench["prefill_seconds"],
            "warmup_decode_seconds": bench["warmup_seconds"],
            "decode_seconds": bench["decode_seconds"],
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s_aggregate": decode_tok_s,
            "decode_tok_s_per_request": decode_tok_s_per_request,
            "decode_step_seconds": _summarize_samples(bench["decode_step_seconds"]),
            "warmup_step_seconds": _summarize_samples(bench["warmup_step_seconds"]),
        },
        "scaling": scaling,
        "memory": memory,
        "profiler": profiler,
        "decision": {
            "accepted": accepted,
            "reason": _ACCEPTED_DECISION_REASON if accepted else "; ".join(blocked_reasons),
        },
        "notes": list(_ACCEPTED_NOTES),
    }
    if int8_kv_primitive_layer_accuracy is not None:
        payload["correctness"]["int8_kv_primitive_layer_accuracy"] = int8_kv_primitive_layer_accuracy
    if isinstance(projection_dispatch_candidates, list):
        payload["projection_dispatch_candidates"] = projection_dispatch_candidates
    validate_cn_diagnostic_artifact_payload(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--skip-generated-equality", action="store_true")
    parser.add_argument(
        "--device-resident-decode",
        action="store_true",
        help=(
            "Drive native c>N decode through the device-resident step "
            "(C3.0b pieces A+B+C: token feedback via batch_lm_out_index, device "
            "batched LM-head argmax) instead of host-fed step_batch_native. "
            "Used to gate the capture-ready decode path against independent c1."
        ),
    )
    parser.add_argument(
        "--batch-prefill-linear-path",
        choices=("packed_segments", "per_segment"),
        default="packed_segments",
        help="Diagnostic linear-attention packed-prefill path; per_segment forces per-request c=1-style linear prefill and blocks retained claims.",
    )
    parser.add_argument(
        "--batch-decode-moe-path",
        choices=("auto", "grouped_compact", "selected_c1"),
        default="auto",
        help="Global MoE path for c>N batch decode; auto selects grouped_compact for the retained-claim correctness frontier, while selected_c1 remains an explicit speed/diagnostic path.",
    )
    parser.add_argument(
        "--batch-decode-linear-path",
        choices=("batch_segments", "per_row"),
        default="batch_segments",
        help="Linear-attention decode path for c>N batch decode; batch_segments is the correctness-first default when paired with selected-QKV/Z projection and batched selected-c1 MoE diagnostics, while per_row remains available as a broader row replay fallback.",
    )
    parser.add_argument(
        "--batch-decode-linear-projection-path",
        choices=("auto", "batch", "batch_gemv", "selected_c1", "selected_qkv_z", "selected_qkv_z_input", "selected_qkv", "selected_z", "selected_ab", "batch_gemv_selected_ab"),
        default="auto",
        help="Diagnostic linear-attention projection path for c>N batch decode; auto uses selected-QKV/Z with native A/B for c<=8 and full selected-c1 replay for larger unproven row counts.",
    )
    parser.add_argument(
        "--batch-decode-linear-state-path",
        choices=("batch_segments", "selected_c1"),
        default="batch_segments",
        help="Diagnostic linear-attention conv/GDN/state path for c>N batch decode; batch_segments is the correctness-first default, while selected_c1 forces token-1 state kernels per row.",
    )
    parser.add_argument(
        "--batch-decode-linear-moe-path",
        choices=("grouped_compact", "per_row_c1"),
        default="grouped_compact",
        help="Diagnostic MoE path for linear-attention c>N batch decode when --batch-decode-moe-path=grouped_compact; per_row_c1 replays true token-1 MoE kernels per row.",
    )
    parser.add_argument(
        "--batch-decode-linear-output-path",
        choices=("auto", "batch", "batch_gemv", "selected_c1"),
        default="batch_gemv",
        help="Diagnostic linear-attention output projection path for c>N batch decode; batch_gemv is the correctness-first default with native segmented state and uses the row-aware Marlin/GEMV path when available, while selected_c1 remains the per-row token-1 output replay fallback.",
    )
    parser.add_argument(
        "--batch-decode-full-attn-path",
        choices=("auto", "native_batch", "per_row"),
        default="auto",
        help="Full-attention decode path for c>N batch decode; auto keeps native_batch for c=2 and uses native row-chunk2 diagnostics for covered c=3..c=8 correctness while full native grouping remains blocked there.",
    )
    parser.add_argument(
        "--batch-decode-full-attn-row-chunk-size",
        type=int,
        default=0,
        help="Diagnostic full-attention native row chunk size for c>N batch decode; a positive value below batch size runs native full-attention in row sub-batches and blocks retained claims.",
    )
    parser.add_argument(
        "--batch-decode-full-attn-row-chunk-layers",
        default="",
        help="Comma-separated full-attention layer ids that should use the row-chunk diagnostic when --batch-decode-full-attn-row-chunk-size is positive; empty applies row chunks to every full-attention layer.",
    )
    parser.add_argument(
        "--batch-decode-attn-input-path",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic full-attention input RMSNorm path for c>N batch decode; per_row forces token-1 row kernels and blocks retained claims.",
    )
    parser.add_argument(
        "--batch-decode-attn-qkv-path",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic full-attention QKV prep path for c>N batch decode; per_row uses independent token-1 scratch and blocks retained claims.",
    )
    parser.add_argument(
        "--batch-decode-attn-scratch-path",
        choices=("batch", "per_row", "per_row_batch_scratch", "per_row_attn_batch_moe", "per_row_attn_batch_post_moe", "per_row_attn_batch_o_post_moe", "per_row_preqkv_append_batch_context_o_post_moe", "per_row_preqkv_append_context_batch_gate_o_post_moe", "per_row_preqkv_append_context_gate_batch_o_post_moe", "persistent_c1", "persistent_c1_no_batch_setup"),
        default="batch",
        help="Diagnostic full-attention scratch path for c>N batch decode; per_row runs each row on an independent token-1 attention scratch, per_row_batch_scratch replays each row through c1 full-attention kernels using row views of the batch scratch, per_row_attn_batch_moe replays attention/post per row with batch scratch row views before grouped batch MoE, per_row_attn_batch_post_moe replays attention/O per row with batch scratch row views before batch post-attention and grouped batch MoE, per_row_attn_batch_o_post_moe replays pre-O attention per row with batch scratch row views before batch O projection, batch post-attention, and grouped batch MoE, per_row_preqkv_append_batch_context_o_post_moe replays pre-QKV and KV append per row before batch context/O/post-attention/grouped MoE, per_row_preqkv_append_context_batch_gate_o_post_moe replays pre-QKV, KV append, and context per row before batch gate/O/post-attention/grouped MoE, per_row_preqkv_append_context_gate_batch_o_post_moe replays pre-QKV, KV append, context, and gate per row before batch O/post-attention/grouped MoE, persistent_c1 reuses the session token-1 c1 scratch, persistent_c1_no_batch_setup also skips native batch span/scratch setup, and all non-batch choices block retained claims.",
    )
    parser.add_argument(
        "--batch-decode-attn-context-path",
        choices=(
            "batch",
            "per_row",
            "per_row_context_only",
            "per_row_dense_context_only",
            "per_row_dense_context_batch_gate",
            "per_row_paged_context_only",
            "batch_temp_output",
            "batch_compact_cache",
        ),
        default="batch",
        help="Diagnostic full-attention context/gate path for c>N batch decode; per_row forces token-1 row context+gate kernels, per_row_context_only keeps the current row-count split, per_row_dense_context_only forces row-local dense context before row-local gate, per_row_dense_context_batch_gate forces row-local dense context before the batch gate, per_row_paged_context_only forces row-local paged context before row-local gate, batch_temp_output writes native batch context into a fresh FP32 buffer before copying into the normal context scratch, batch_compact_cache runs the native batch context kernel on compact copied row caches, and all non-batch modes block retained claims.",
    )
    parser.add_argument(
        "--batch-decode-attn-dense-context-batch-gate-layers",
        default="",
        help="Optional comma/range list of full-attention layer ids that should use the row-local dense context diagnostic while keeping the normal batch gate; auto defaults use rowchunk2 for c>=3 correctness and leave this diagnostic unset.",
    )
    parser.add_argument(
        "--batch-decode-attn-gate-path",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic full-attention gate path for c>N batch decode; per_row keeps the native batch context kernel but applies the sigmoid gate row-by-row and blocks retained claims.",
    )
    parser.add_argument(
        "--batch-decode-full-attn-kv-append-path",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic full-attention KV append path for c>N batch decode; per_row writes each token-1 K/V row separately and blocks retained claims.",
    )
    parser.add_argument(
        "--batch-decode-attn-append-context-order",
        choices=("phased", "interleaved"),
        default="phased",
        help="Diagnostic full-attention per-row KV append/context ordering; interleaved requires per-row KV append and context diagnostics.",
    )
    parser.add_argument(
        "--batch-decode-attn-suffix-order",
        choices=("phased", "interleaved"),
        default="phased",
        help="Diagnostic full-attention post-context suffix ordering; interleaved requires per-row context, O, post-attention, and MoE diagnostics.",
    )
    parser.add_argument(
        "--batch-decode-full-attn-output-path",
        choices=("batch", "batch_gemv", "per_row", "native", "native_row_chunk"),
        default="batch",
        help="Diagnostic full-attention O projection path for c>N batch decode; batch is the correctness-first default (rows=2 auto-select row-aware GEMV O; rowchunked rows keep their tail GEMV repair), batch_gemv forces row-aware GEMV for every chunk, native forces native O for rows=2 diagnostics, native_row_chunk bypasses the automatic rowchunk GEMV repair for native O diagnostics, and per_row remains a token-1 output replay fallback.",
    )
    parser.add_argument(
        "--batch-decode-full-attn-layer-copy",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic full-attention layer-output handoff for c>N batch decode; batch is the correctness-first default with per-row output projection, while per_row remains an explicit row-copy diagnostic.",
    )
    parser.add_argument(
        "--batch-decode-full-attn-moe-path",
        choices=("grouped_compact", "per_row_c1"),
        default="grouped_compact",
        help="Diagnostic MoE path for full-attention c>N batch decode when --batch-decode-moe-path=grouped_compact; per_row_c1 replays true token-1 MoE kernels per row.",
    )
    parser.add_argument(
        "--batch-decode-post-attn-path",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic post-attention add/RMSNorm path for c>N batch decode; batch is the correctness-first default after the per-row full-attention output/MoE boundary, while per_row remains available as a diagnostic.",
    )
    parser.add_argument(
        "--batch-sample-mode",
        choices=("serial_lm_head", "batched_lm_head"),
        default="serial_lm_head",
        help="Sampler/LM-head path for native c>N decode; when omitted for c=4/8 the bench uses the repo-retained row-aware batched_lm_head equality artifact if it validates, while c=2 stays serial by default until the c2 batched-sampler flakes are fixed. Explicit batched_lm_head still requires generated-token equality evidence via --batch-sample-eq-*."
    )
    parser.add_argument(
        "--batch-sample-norm-path",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic final RMSNorm path when --batch-sample-mode=batched_lm_head; per_row normalizes each row into the batched LM-head input buffer and blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-cast-path",
        choices=("auto", "batch", "per_row"),
        default="auto",
        help="Diagnostic final FP16-to-BF16 cast path when --batch-sample-mode=batched_lm_head; auto follows --batch-sample-norm-path for backward-compatible combined norm/cast diagnostics, and per_row blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-argmax-mode",
        choices=("batch", "serial_per_row"),
        default="batch",
        help="Diagnostic argmax path when --batch-sample-mode=batched_lm_head; serial_per_row keeps the batched LM-head projection but resolves row argmax with the serial per-row kernel and blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-argmax-audit",
        action="store_true",
        help="Diagnostic-only parity audit for batched_lm_head + batch argmax: runs serial per-row argmax over the same batched logits, records batch-vs-serial mismatches, and blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-lm-head-audit",
        action="store_true",
        help="Diagnostic-only parity audit for batched_lm_head + batch argmax: reruns each row through the serial c=1 LM-head projection from the same normalized BF16 row, records batch-vs-serial projection+argmax mismatches, and blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-lm-head-kernel-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns each normalized BF16 row through serial c=1 LM-head projection and argmax kernels without serial host readback or output comparison; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-final-norm-audit",
        action="store_true",
        help="Diagnostic-only parity audit for batched_lm_head + batch argmax: reruns each row through serial c=1 final RMSNorm, FP16-to-BF16 cast, LM-head projection, and argmax from the same hidden row, records mismatches, and blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-final-norm-kernel-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns each row through serial c=1 final RMSNorm and FP16-to-BF16 cast kernels without LM-head, argmax, host readback, or output comparison; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-final-rmsnorm-kernel-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns each row through the serial c=1 final RMSNorm kernel only, without cast, LM-head, argmax, host readback, or output comparison; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-final-rmsnorm-temp-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns each row through the serial c=1 final RMSNorm kernel into a dedicated temp buffer, without clobbering sampler norm scratch, cast, LM-head, argmax, host readback, or output comparison; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-final-cast-temp-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns each normalized FP16 row through the final FP16-to-BF16 cast kernel into a dedicated temp buffer, without clobbering sampler BF16 scratch, LM-head, argmax, host readback, or output comparison; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-final-cast-tiny-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns one FP16 element per row through the final FP16-to-BF16 cast kernel into a dedicated temp buffer, without clobbering sampler BF16 scratch, LM-head, argmax, host readback, or output comparison; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-final-cast-elems-fence",
        type=int,
        default=0,
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns the first N FP16 elements of each normalized row through the final FP16-to-BF16 cast kernel into a dedicated temp buffer, without clobbering sampler BF16 scratch, LM-head, argmax, host readback, or output comparison; blocks retained sampler claims when N > 0.",
    )
    parser.add_argument(
        "--batch-sample-stabilize-cast-elems",
        type=int,
        default=0,
        help="Correctness-first opt-in stabilization fence for batched_lm_head + batch argmax: reruns the first N FP16 elements of each normalized row through the final FP16-to-BF16 cast kernel into a dedicated temp buffer after sampler host readback; records sampler metadata but does not mark the sampler as diagnostic-blocked.",
    )
    parser.add_argument(
        "--batch-sample-sync-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: performs extra device_synchronize calls without launching kernels, host readback, or output comparison; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-suffix-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns each row through serial c=1 final RMSNorm/cast/LM-head/argmax with host reads but does not compare outputs; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-suffix-kernel-fence",
        action="store_true",
        help="Diagnostic-only timing fence for batched_lm_head + batch argmax: reruns each row through serial c=1 final RMSNorm/cast/LM-head/argmax kernels without serial host readback or output comparison; blocks retained sampler claims.",
    )
    parser.add_argument(
        "--batch-sample-eq-ok",
        action="store_true",
        help="Mark the supplied --batch-sample-eq-artifact as green generated-token equality evidence for enabling batched_lm_head.",
    )
    parser.add_argument(
        "--batch-sample-eq-artifact",
        type=Path,
        help="benchmarks/results JSON artifact with generated-token equality rows for enabling batched_lm_head.",
    )
    parser.add_argument(
        "--batch-sample-eq-rows",
        type=int,
        help="Row count covered by --batch-sample-eq-artifact; must match --batch-size when --batch-sample-mode=batched_lm_head.",
    )
    parser.add_argument(_RETAINED_GATE_FLAGS[0], type=Path, help="c=1 baseline artifact used for retained scaling ratios")
    parser.add_argument(_RETAINED_GATE_FLAGS[1], type=Path, help="scheduler serial-bridge artifact for retained scaling ratios")
    parser.add_argument(_RETAINED_GATE_FLAGS[2], type=Path, help="scripts/qwen35_batch_correctness.py JSON for this c>N row count")
    parser.add_argument(_RETAINED_GATE_FLAGS[3], type=Path, help="Captured rocprofv3 summary JSON to attach to retained evidence")
    parser.add_argument(
        "--int8-kv-primitive-cpu-json",
        type=Path,
        help="scripts/qwen35_kv_int8_accuracy.py --device cpu JSON required before retained int8_per_token_head c>N promotion",
    )
    parser.add_argument(
        "--int8-kv-primitive-hip-json",
        type=Path,
        help="scripts/qwen35_kv_int8_accuracy.py --device hip --require-int8-hip JSON required before retained int8_per_token_head c>N promotion",
    )
    parser.add_argument(
        "--projection-dispatch-artifact",
        type=Path,
        help="Optional benchmarks/results JSON carrying projection_dispatch_candidates for runtime c-aware projection metadata",
    )
    parser.add_argument("--profiler-command", help="Exact rocprofv3 --kernel-trace command that produced --profiler-json")
    add_kv_policy_args(parser, help_prefix="Resident KV storage for retained native c>N benchmark")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    _apply_default_batch_sample_evidence(args, raw_argv)

    if args.batch_size <= 1:
        raise ValueError("--batch-size must be greater than 1 for retained c>N")
    if args.decode_tokens <= 0 or args.warmup_decode_tokens < 0:
        raise ValueError("decode token counts must be positive/non-negative")
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive")
    if args.batch_sample_eq_rows is not None and args.batch_sample_eq_rows <= 0:
        raise ValueError("--batch-sample-eq-rows must be positive")

    prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    compiler_version = _compiler_version(args.compiler_version_file)
    _apply_runtime_env_args(args)
    projection_dispatch_candidates = _projection_dispatch_candidates_for_payload(args)
    bench = _run_native_bench(
        runner,
        prompts,
        max_layers=args.max_layers,
        warmup_decode_tokens=args.warmup_decode_tokens,
        decode_tokens=args.decode_tokens,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        kv_policy=kv_policy,
        device_resident=bool(getattr(args, "device_resident_decode", False)),
    )
    if projection_dispatch_candidates is not None:
        bench["projection_dispatch_candidates"] = projection_dispatch_candidates

    request_ids = list(range(args.batch_size))
    equality_tokens_per_sequence = 1 + args.warmup_decode_tokens + args.decode_tokens
    equality_shape_metadata = {
        "comparison": "native_batch_vs_independent_c1",
        "rows": args.batch_size,
        "warmup_decode_tokens": args.warmup_decode_tokens,
        "gen_tokens_per_request": args.decode_tokens,
        "tokens_per_sequence": equality_tokens_per_sequence,
    }
    batch_sequences = _generated_sequences_from_bench(bench, request_ids)
    if args.skip_generated_equality:
        equality = {
            "passed": False,
            "skipped": True,
            **equality_shape_metadata,
            "reason": "--skip-generated-equality was provided",
            "batch_sequences": batch_sequences,
            "c1_sequences": None,
        }
    else:
        c1_sequences = _run_c1_reference_tokens(
            runner,
            prompts,
            total_decode_tokens=args.warmup_decode_tokens + args.decode_tokens,
            max_layers=args.max_layers,
            max_sequence_length=args.prompt_length + args.warmup_decode_tokens + args.decode_tokens + 1,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            kv_policy=kv_policy,
        )
        equality_prefix_summary = _generated_token_equality_prefix_summary(batch_sequences, c1_sequences)
        equality = {
            "passed": batch_sequences == c1_sequences,
            "skipped": False,
            **equality_shape_metadata,
            **equality_prefix_summary,
            "batch_sequences": batch_sequences,
            "c1_sequences": c1_sequences,
            "mismatch_summaries": _generated_token_equality_mismatch_summaries(batch_sequences, c1_sequences),
            "mismatches": [
                {"row": row, "batch": batch_sequences[row], "c1": c1_sequences[row]}
                for row in range(args.batch_size)
                if batch_sequences[row] != c1_sequences[row]
            ],
        }

    payload = _build_payload(args, argv, bench, [len(prompt) for prompt in prompts], equality)
    text = _payload_json(payload)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if payload["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
