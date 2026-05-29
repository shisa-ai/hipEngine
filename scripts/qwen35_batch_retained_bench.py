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
    ProjectionDispatchEvidence,
    batch_sampler_equality_payload_blockers,
    projection_dispatch_evidence_payload_blockers,
)
from hipengine.generation import GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS, GeneratedToken, GraphBucketCache, ResidentBatchScheduler
from hipengine.kvcache import ResolvedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_artifact_schema import (
    DECODE_EXECUTION_DIAGNOSTIC_TRACE_FIELDS,
    validate_cn_diagnostic_artifact_payload,
)
from scripts.qwen35_batch_constants import (
    PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS,
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT,
    RETAINED_ARTIFACT_PROFILER_TRACE_DURATION_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_END_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_KERNEL_NAME_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_START_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_SYNTHESIZED_FIELDS,
    RETAINED_ARTIFACT_RETAINED_GATE_FLAGS,
    RETAINED_ARTIFACT_RETAINED_GATE_LABELS,
    RETAINED_ARTIFACT_RETAINED_KV_POLICY_FLAGS,
    RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS,
    RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS,
    RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES,
    RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES,
    RETAINED_ARTIFACT_REQUIRED_SCALING_BASELINES,
    RETAINED_ARTIFACT_REQUIRED_SCALING_RATIOS,
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
_RETAINED_GATE_FLAGS = RETAINED_ARTIFACT_RETAINED_GATE_FLAGS
_RETAINED_GATE_LABELS = RETAINED_ARTIFACT_RETAINED_GATE_LABELS
_RETAINED_KV_POLICY_FLAGS = RETAINED_ARTIFACT_RETAINED_KV_POLICY_FLAGS
_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS = RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS
_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS = RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS
_DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS = PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS
_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS
_REQUIRED_PRIMITIVE_CORRECTNESS_SEED = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED
_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT

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
    fixture = json.loads(path.read_text())
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
        "top_k": key.top_k,
        "experts_per_token": key.experts_per_token,
        "replay_steps": key.replay_steps,
        "draft_depth": key.draft_depth,
        "tree_shape": list(key.tree_shape),
    }


def _record_decode_graph_bucket_metadata(scheduler: ResidentBatchScheduler, scheduler_metadata: dict[str, Any]) -> None:
    key = scheduler.shape_key(mode="decode", top_k=0, experts_per_token=0, replay_steps=1)
    scheduler_metadata["decode_shape_key"] = _shape_key_payload(key)
    scheduler.graph_buckets.get_or_create(key, lambda bucket: _shape_key_payload(bucket))
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
    return {str(bucket): int(count) for bucket, count in histogram.items()} or None


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
    updated_stats["kernel_time_histogram_ns"] = merged_histogram
    scheduler_metadata["graph_bucket_stats"] = updated_stats


def _decode_shape_key_blockers(scheduler_metadata: Mapping[str, Any], *, concurrency: int, prompt_length: int) -> list[str]:
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


def _graph_replay_profiler_evidence_blockers(
    scheduler_metadata: Mapping[str, Any], profiler: Mapping[str, Any]
) -> list[str]:
    graph_stats = scheduler_metadata.get("graph_bucket_stats")
    hits = graph_stats.get("hits") if isinstance(graph_stats, Mapping) else None
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


def _all_finite(rows: Iterable[dict[str, Any]]) -> bool:
    return all(math.isfinite(float(row["logit"])) for row in rows)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not (_is_number(numerator) and _is_number(denominator)):
        return None
    denom = float(denominator)
    if denom <= 0.0:
        return None
    return float(numerator) / denom


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


def _scaling_reference(path: Path | None, *, default_workload_concurrency: int | None = None) -> dict[str, Any]:
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
        payload = json.loads(path.read_text())
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
    aggregate, per_request = _extract_decode_rates(payload)
    throughput_missing = aggregate is None or per_request is None
    if reasons:
        aggregate = None
        per_request = None
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
    if throughput_missing:
        reasons.append("decode throughput fields missing")
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
        payload = json.loads(path.read_text())
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
        "attn_batch_vs_c1_max_abs": attn_vs_c1,
        "attn_batch_vs_numpy_max_abs": payload.get("attn_batch_vs_numpy_max_abs"),
        "reason": None if not reasons else "; ".join(reasons),
    }


def _is_finite_nonnegative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_finite_positive_number(value: Any) -> bool:
    return _is_finite_nonnegative_number(value) and float(value) > 0.0


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
    return any(isinstance(name, str) and "batch" in name.lower() for name in kernel_names)


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
        payload = json.loads(path.read_text())
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
            output_format = _command_arg_value(profiler_command, "--output-format")
            if output_format is not None:
                result["output_format"] = output_format
                synthesized_fields.add("output_format")
        if "trace_dir" not in result:
            trace_dir = _command_arg_value(profiler_command, "-d")
            if trace_dir is not None:
                result["trace_dir"] = trace_dir
                synthesized_fields.add("trace_dir")
    if "synthesized_fields" not in result:
        result["synthesized_fields"] = [field for field in _PROFILER_SYNTHESIZED_FIELDS if field in synthesized_fields]
    result.setdefault("artifact_path", str(path))
    result.setdefault("status", "loaded")
    return result


def _profiled_command(args: argparse.Namespace, argv: Sequence[str] | None) -> str | None:
    explicit = getattr(args, "profiler_command", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    if getattr(args, "profiler_json", None) is None:
        return None
    return f"rocprofv3 --kernel-trace --output-format csv -d <profile-dir> -- {_command(argv)}"


def _build_scaling_comparison(
    args: argparse.Namespace,
    *,
    native_decode_tok_s_aggregate: float | None,
    native_decode_tok_s_per_request: float | None,
) -> dict[str, Any]:
    c1 = _scaling_reference(getattr(args, "c1_baseline_json", None), default_workload_concurrency=1)
    serial = _scaling_reference(getattr(args, "serial_bridge_json", None))
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
            if "num_splits_per_row" in layer:
                blockers.append(f"{label}.num_splits_per_row must be absent for native retained decode")
            if "full_attention_input_decode_path" in layer:
                blockers.append(f"{label}.full_attention_input_decode_path must be absent for native retained decode")
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
            if linear_output_path not in {None, "native_batch"}:
                blockers.append(f"{label}.linear_attention_output_path must be native_batch or absent")
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


def _load_retained_json_artifact(value: str) -> tuple[Mapping[str, Any] | None, str | None]:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "artifact_path must point to an existing JSON artifact"
    except json.JSONDecodeError as exc:
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
) -> list[str]:
    blockers: list[str] = []
    command_parts = _split_command_parts(command)
    if command_parts.count("--") != 1:
        blockers.append("profiler command must include exactly one rocprof separator")
    rocprof_prefix = _rocprof_command_prefix(command)
    rocprof_prefix_command = _join_command_parts(rocprof_prefix)
    if not rocprof_prefix or Path(rocprof_prefix[0]).name != "rocprofv3":
        blockers.append("profiler command must start with rocprofv3")
    if not any(Path(part).name == "rocprofv3" for part in rocprof_prefix):
        blockers.append("profiler command must include rocprofv3")
    for flag in ("--kernel-trace", "--output-format", "-d"):
        if _command_flag_count(rocprof_prefix, flag) > 1:
            blockers.append(f"profiler command {flag} must be unique before rocprof separator")
    if "--kernel-trace" not in rocprof_prefix:
        blockers.append("profiler command must include --kernel-trace")
    if "scripts/qwen35_batch_retained_bench.py" not in command:
        blockers.append("profiler command must target scripts/qwen35_batch_retained_bench.py")
    profiled_segment = _profiled_command_segment(command)
    retained_command = command
    if profiled_segment is None:
        blockers.append("profiler command must include rocprof -- separator")
    else:
        retained_command = _join_command_parts(profiled_segment)
        for flag in _RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS:
            if _command_flag_count(profiled_segment, flag) > 1:
                blockers.append(f"profiler command {flag} must be unique after rocprof separator")
        for flag in _RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS:
            if _command_has_flag(retained_command, flag):
                blockers.append(f"profiler command must not include {flag}")
        if (
            len(profiled_segment) < 2
            or not Path(profiled_segment[0]).name.startswith("python")
            or profiled_segment[1] != "scripts/qwen35_batch_retained_bench.py"
        ):
            blockers.append("profiler command must launch retained bench after rocprof separator")
    if _command_arg_value(rocprof_prefix_command, "--output-format") != "csv":
        blockers.append("profiler command must include --output-format csv")
    if trace_dir is not None and _command_arg_value(rocprof_prefix_command, "-d") != trace_dir:
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
    if profiler.get("output_format") != "csv":
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
    elif not all(isinstance(name, str) and name for name in trace_kernel_names):
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
    elif not all(isinstance(name, str) and name for name in expected_kernel_names):
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
        if not isinstance(kernel_name, str) or not kernel_name:
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
        duration_key_set = {key for key in kernel_durations if isinstance(key, str) and key}
        share_key_set = {key for key in kernel_duration_shares if isinstance(key, str) and key}
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
        trace_name_set = {name for name in trace_kernel_names if isinstance(name, str) and name}
        missing_trace_names = [name for name in kernel_durations if isinstance(name, str) and name and name not in trace_name_set]
        if missing_trace_names:
            blockers.append("profiler.trace_kernel_names must include profiler.kernel_durations_ns keys")
    if isinstance(expected_kernel_names, list):
        trace_name_set = set()
        if isinstance(trace_kernel_names, list):
            trace_name_set = {name for name in trace_kernel_names if isinstance(name, str) and name}
        missing_expected_trace_name = False
        for kernel_name in expected_kernel_names:
            if isinstance(kernel_name, str) and kernel_name:
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "equality_artifact must point to an existing JSON artifact"
    except json.JSONDecodeError as exc:
        return None, f"equality_artifact must point to a valid JSON artifact: {exc}"
    if not isinstance(payload, Mapping):
        return None, "equality_artifact must point to a JSON object artifact"
    return payload, None


def _sampler_profiler_name_matches(kernel_name: str) -> bool:
    lowered = kernel_name.lower()
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


def _memory_evidence_blockers(memory: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not _is_finite_nonnegative_number(memory.get("allocator_reserved_peak_bytes")):
        blockers.append("memory.allocator_reserved_peak_bytes is unavailable or non-finite")
    dynamic_pool = memory.get("dynamic_pool")
    if not isinstance(dynamic_pool, Mapping):
        blockers.append("memory.dynamic_pool evidence is missing")
    else:
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
                if not _is_finite_nonnegative_number(pool_counters.get(field)):
                    blockers.append(f"memory.dynamic_pool.pool_counters.{field} is unavailable or non-finite")
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
        if not isinstance(prefix_sharing.get("enabled"), bool):
            blockers.append("memory.prefix_sharing.enabled is not bool")
        if not _is_finite_nonnegative_number(prefix_sharing.get("savings_bytes")):
            blockers.append("memory.prefix_sharing.savings_bytes is unavailable or non-finite")
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


def _hardware_context() -> dict[str, Any]:
    return {
        "gpu": "AMD Radeon Pro W7900",
        "arch": "gfx1100",
        "default_hardware": True,
        "rocminfo": _run_capture(["bash", "-lc", "rocminfo | grep -E 'Name:|gfx' | head -4"], timeout=10.0),
        "rocm_smi": _run_capture(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"], timeout=10.0),
    }


def _command(argv: Sequence[str] | None) -> str:
    parts = ["python3", "scripts/qwen35_batch_retained_bench.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _primitive_correctness_command(path: Path | None, *, rows: int, seed: int = 1234) -> str:
    parts = ["python3", "scripts/qwen35_batch_correctness.py", "--rows", str(rows), "--seed", str(seed), "--json"]
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
) -> tuple[int, bool]:
    work = scheduler.next_decode_work()
    if work is None:
        raise RuntimeError("scheduler did not emit decode work")
    request_ids = tuple(request_id for request_id in work.request_ids if request_id in next_token_by_request)
    slots = [scheduler.active_batch.slot_for(request_id) for request_id in request_ids]
    if tuple(slots) != tuple(range(len(slots))):
        raise RuntimeError(f"native retained benchmark requires compact slots, got {slots!r}")
    results = session.step_batch_native(
        [next_token_by_request[request_id] for request_id in request_ids],
        positions=[scheduler.active_batch.requests[request_id].context_len for request_id in request_ids],
        slots=slots,
        sample=True,
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

        _record_decode_graph_bucket_metadata(scheduler, scheduler_metadata)
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
            active_rows=args.batch_size,
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
    profiled_command = _profiled_command(args, argv)
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
    )
    profiler_blockers.extend(_profiler_synthesized_fields_blockers(profiler))
    profiler_blockers.extend(_profiler_kernel_evidence_blockers(profiler))
    profiler_blockers.extend(_profiler_cpu_side_bottleneck_blockers(profiler))
    batch_execution = dict(bench["batch_execution"])
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
    memory_blockers = _memory_evidence_blockers(memory)
    graph_bucket_blockers = _decode_shape_key_blockers(scheduler_metadata, concurrency=args.batch_size, prompt_length=args.prompt_length)
    graph_bucket_blockers.extend(_graph_bucket_evidence_blockers(scheduler_metadata))
    graph_bucket_blockers.extend(_graph_replay_profiler_evidence_blockers(scheduler_metadata, profiler))
    equality_passed = bool(equality.get("passed"))
    protocol_shape = args.max_layers == 40 and args.prompt_length >= 512 and args.decode_tokens >= 128
    scaling_complete = bool(scaling["complete"])
    primitive_passed = bool(primitive_correctness["passed"])
    accepted = bool(
        bench["finite_logits"]
        and throughput_claim_eligible
        and equality_passed
        and primitive_passed
        and protocol_shape
        and scaling_complete
        and profiler_captured
        and not profiler_blockers
        and not batch_execution_blockers
        and not projection_blockers
        and not sampler_blockers
        and not memory_blockers
        and not graph_bucket_blockers
    )
    primitive_loaded = primitive_correctness.get("status") == "loaded"
    correctness_rejected = bool(bench["finite_logits"] and (not equality_passed or (primitive_loaded and not primitive_passed)))
    status = "accepted" if accepted else ("rejected_correctness" if correctness_rejected else "blocked")
    blocked_reasons: list[str] = []
    if not throughput_claim_eligible:
        blocked_reasons.append("batch_execution.throughput_claim_eligible=false")
    if not equality_passed:
        blocked_reasons.append("generated-token equality vs independent c=1 did not pass")
    if not primitive_passed:
        blocked_reasons.append(f"primitive c>N correctness gate did not pass: {primitive_correctness.get('reason')}")
    if args.prompt_length < 512 or args.decode_tokens < 128:
        blocked_reasons.append("workload is a reduced diagnostic shape, not the docs/BENCHMARK.md c=N 512/128 protocol")
    if args.max_layers != 40:
        blocked_reasons.append("max_layers is not the full 40-layer Qwen3.5/PARO model")
    if not scaling_complete:
        blocked_reasons.append("scaling comparison vs c=1 and serial bridge baselines is incomplete")
    if not profiler_captured:
        blocked_reasons.append("profiler trace was not captured with expected kernels present")
    blocked_reasons.extend(profiler_blockers)
    if not bench["finite_logits"]:
        blocked_reasons.append("non-finite seed or decode logits")
    blocked_reasons.extend(batch_execution_blockers)
    blocked_reasons.extend(projection_blockers)
    blocked_reasons.extend(sampler_blockers)
    blocked_reasons.extend(memory_blockers)
    blocked_reasons.extend(graph_bucket_blockers)
    per_request_observability = dict(bench.get("request_observability", {}))
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
    request_latencies = [
        float(row["completion_timestamp"]) - float(row["submitted_timestamp"])
        for row in per_request_observability.values()
        if isinstance(row, dict)
        and row.get("completion_timestamp") is not None
        and row.get("submitted_timestamp") is not None
    ]
    latency_summary = _summarize_samples(request_latencies)
    payload = {
        "schema": 3,
        "status": status,
        "artifact_path": str(args.json) if args.json is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": f"qwen35-paro-c{args.batch_size}-native-retained",
        "summary": "Qwen3.5/PARO scheduler compact native c>N benchmark",
        "performance_claim": accepted,
        "hardware": _hardware_context(),
        "software": _software_context(),
        "workload": {
            "shape": f"c={args.batch_size} prompt={args.prompt_length} decode={args.decode_tokens}",
            "model": "Qwen3.5/3.6-35B-A3B-PARO",
            "model_path": str(Path(args.model)),
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
            "passed": bool(bench["finite_logits"] and equality_passed and primitive_passed),
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
            "reason": "correctness/protocol passed" if accepted else "; ".join(blocked_reasons),
        },
        "notes": [
            "Native retained c>N path uses packed prompt slabs and step_batch_native for decode.",
            "Batch split-K decode remains out of scope; this accepted protocol keeps context < 1024.",
        ],
    }
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
    parser.add_argument(_RETAINED_GATE_FLAGS[0], type=Path, help="c=1 baseline artifact used for retained scaling ratios")
    parser.add_argument(_RETAINED_GATE_FLAGS[1], type=Path, help="scheduler serial-bridge artifact for retained scaling ratios")
    parser.add_argument(_RETAINED_GATE_FLAGS[2], type=Path, help="scripts/qwen35_batch_correctness.py JSON for this c>N row count")
    parser.add_argument(_RETAINED_GATE_FLAGS[3], type=Path, help="Captured rocprofv3 summary JSON to attach to retained evidence")
    parser.add_argument("--profiler-command", help="Exact rocprofv3 --kernel-trace command that produced --profiler-json")
    add_kv_policy_args(parser, help_prefix="Resident KV storage for retained native c>N benchmark")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    if args.batch_size <= 1:
        raise ValueError("--batch-size must be greater than 1 for retained c>N")
    if args.decode_tokens <= 0 or args.warmup_decode_tokens < 0:
        raise ValueError("decode token counts must be positive/non-negative")
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive")

    prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    compiler_version = _compiler_version(args.compiler_version_file)
    os.environ["HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"] = "1"
    bench = _run_native_bench(
        runner,
        prompts,
        max_layers=args.max_layers,
        warmup_decode_tokens=args.warmup_decode_tokens,
        decode_tokens=args.decode_tokens,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        kv_policy=kv_policy,
    )

    request_ids = list(range(args.batch_size))
    batch_sequences = _generated_sequences_from_bench(bench, request_ids)
    if args.skip_generated_equality:
        equality = {
            "passed": False,
            "skipped": True,
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
        equality = {
            "passed": batch_sequences == c1_sequences,
            "skipped": False,
            "batch_sequences": batch_sequences,
            "c1_sequences": c1_sequences,
            "mismatches": [
                {"row": row, "batch": batch_sequences[row], "c1": c1_sequences[row]}
                for row in range(args.batch_size)
                if batch_sequences[row] != c1_sequences[row]
            ],
        }

    payload = _build_payload(args, argv, bench, [len(prompt) for prompt in prompts], equality)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if payload["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
