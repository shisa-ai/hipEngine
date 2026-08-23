#!/usr/bin/env python3
"""Matched gfx11 F1 HTTP concurrency benchmark for hipEngine and llama.cpp.

The cross-engine primary metric is deliberately limited to the timing boundary
all servers expose: exact returned completion tokens divided by client wall from
simultaneous release through the last completed response.  Backend-native decode
timings are retained as diagnostics and are never substituted for that wall.

Each engine first generates independent c1 token-ID oracles for the four prompt
rows used by an arbitrary logical c1-c32 sweep. Strict evidence binds every
warmup, measured burst, and live-admission row to that trajectory. Production
evidence keeps c1/cN equality diagnostic, requires a complete external
numerical/task/control bundle, and binds exact serving control plus same-schedule
determinism. hipEngine's resident TTFT/ITL summaries and route/fallback counters
are scraped separately; llama.cpp does not expose equivalent non-streaming
percentile summaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import file_sha256, token_ids_sha256  # noqa: E402
from hipengine.benchmark.provenance import collect_model_identity  # noqa: E402
from hipengine.util.amdgpu_vram import VramSampler, select_card  # noqa: E402

SCHEMA_VERSION = 2
ENGINE_CHOICES = ("hipengine", "llamacpp-hip", "llamacpp-vulkan")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_LLAMA_HIP_REPO = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_LLAMA_VULKAN_REPO = Path("/home/lhl/llama.cpp/llama.cpp-vulkan")
DEFAULT_VULKAN_ICD = "/usr/share/vulkan/icd.d/radeon_icd.json"
_CORRECTNESS_PROFILES = ("strict", "production")
_PRODUCTION_CORRECTNESS_BUNDLE_KIND = "hipengine_server_production_correctness_bundle"
_PRODUCTION_NUMERICAL_LIMITS = {
    "kl_mean": 1e-3,
    "kl_p95": 5e-3,
    "kl_p99": 2e-2,
    "kl_max": 5e-2,
    "top1_agreement": 0.99,
}
_CORRECTNESS_RUNTIME_PATHS = (
    "hipengine",
    "kernels",
    "scripts/server_f1_concurrency_bench.py",
    "scripts/execution_profile_gguf_fp16_state_gate.py",
    "scripts/execution_profile_gguf_fp16_state_batch_gate.py",
    "scripts/gguf_arbitrary_c_lifecycle.py",
)
_EFFECTIVE_SERVER_ENV_KEYS = (
    "HIPENGINE_BACKEND",
    "HIPENGINE_HIP_ARCH",
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIPENGINE_EXECUTION_PROFILE",
    "HIPENGINE_SUBMISSION_TRANSPORT",
    "HIPENGINE_GGUF_FP16_RECURRENT_STATE",
    "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS",
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN",
    "HIPENGINE_GGUF_GDN_PREFILL_MODE",
    "HIPENGINE_PREFILL_DECODE_POLICY",
    "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS",
    "HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS",
    "HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE",
    "HIPENGINE_GGUF_AR_PACKED_DECODE",
    "HIPENGINE_GPU_MAX_HW_QUEUES_POLICY",
    "HIPENGINE_PROCESS_ENV_REPORT_PATH",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_MAX_HW_QUEUES",
    "HSA_OVERRIDE_GFX_VERSION",
)
_PROMETHEUS_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)(?:\s+\d+)?$"
)
_PROMETHEUS_LABEL_RE = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>"(?:\\.|[^"\\])*")')


class BenchError(RuntimeError):
    """Raised when a benchmark contract cannot be satisfied."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must contain exact generated token IDs")
    result: list[int] = []
    for index, token in enumerate(value):
        if not isinstance(token, int) or isinstance(token, bool):
            raise ValueError(f"{label}[{index}] must be an integer token ID")
        result.append(int(token))
    return result


def extract_response(engine: str, response: Mapping[str, Any], *, prompt_tokens: int) -> dict[str, Any]:
    """Normalize exact output and native diagnostics from one server response."""

    normalized_engine = str(engine)
    if normalized_engine == "hipengine":
        choices = response.get("choices")
        if not isinstance(choices, Sequence) or not choices or not isinstance(choices[0], Mapping):
            raise ValueError("hipEngine response is missing choices[0]")
        choice = choices[0]
        choice_hip = choice.get("hipengine")
        choice_hip = choice_hip if isinstance(choice_hip, Mapping) else {}
        root_hip = response.get("hipengine")
        root_hip = root_hip if isinstance(root_hip, Mapping) else {}
        accounting = root_hip.get("token_accounting")
        accounting = accounting if isinstance(accounting, Mapping) else {}
        generated_rows = accounting.get("choice_generated_token_ids")
        raw_tokens: Any = choice_hip.get("generated_token_ids")
        if (
            isinstance(generated_rows, Sequence)
            and generated_rows
            and not isinstance(generated_rows, (str, bytes, bytearray))
        ):
            raw_tokens = generated_rows[0]
        generated = _int_list(raw_tokens, label="hipEngine response")
        usage = response.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        timing = choice_hip.get("timing")
        timing = timing if isinstance(timing, Mapping) else {}
        decode_state = choice_hip.get("decode_state")
        decode_state = decode_state if isinstance(decode_state, Mapping) else {}
        generation_shape = root_hip.get("generation_shape")
        generation_shape = dict(generation_shape) if isinstance(generation_shape, Mapping) else None
        backend_timing = {
            str(key): float(value)
            for key, value in timing.items()
            if _is_number(value)
        }
        return {
            "generated_token_ids": generated,
            "text": str(choice.get("text") or ""),
            "prompt_tokens": int(usage.get("prompt_tokens") or prompt_tokens),
            "completion_tokens": len(generated),
            "finish_reason": choice.get("finish_reason"),
            "backend_timing_ms": backend_timing,
            "backend_decode_tok_s": (
                1000.0 * len(generated) / float(backend_timing["decode_batch_ms"])
                if backend_timing.get("decode_batch_ms", 0.0) > 0.0
                else None
            ),
            "execution_path": decode_state.get("execution_path"),
            "serial_decode_fallback": decode_state.get("serial_decode_fallback"),
            "native_caware_decode": decode_state.get("native_caware_decode"),
            "sampler_mode": decode_state.get("sampler_mode"),
            "diagnostics": (
                dict(choice_hip["diagnostics"])
                if isinstance(choice_hip.get("diagnostics"), Mapping)
                else None
            ),
            "generation_shape": generation_shape,
        }

    if normalized_engine not in {"llamacpp-hip", "llamacpp-vulkan"}:
        raise ValueError(f"unknown engine: {normalized_engine!r}")
    generated = _int_list(response.get("tokens"), label="llama.cpp response")
    timings = response.get("timings")
    timings = timings if isinstance(timings, Mapping) else {}
    prompt_n = _number(timings.get("prompt_n"))
    if prompt_n is None:
        prompt_n = _number(response.get("tokens_evaluated"))
    prompt_ms = _number(timings.get("prompt_ms"))
    predicted_ms = _number(timings.get("predicted_ms"))
    backend_timing = {
        key: value
        for key, value in (("prompt_ms", prompt_ms), ("predicted_ms", predicted_ms))
        if value is not None
    }
    predicted_per_second = _number(timings.get("predicted_per_second"))
    if predicted_per_second is None and predicted_ms is not None and predicted_ms > 0.0:
        predicted_per_second = 1000.0 * len(generated) / predicted_ms
    return {
        "generated_token_ids": generated,
        "text": str(response.get("content") or ""),
        "prompt_tokens": int(prompt_n if prompt_n is not None else prompt_tokens),
        "completion_tokens": len(generated),
        "finish_reason": response.get("stop_type") or ("stop" if response.get("stop") else None),
        "backend_timing_ms": backend_timing,
        "backend_decode_tok_s": predicted_per_second,
        "execution_path": "llama-server /completion",
        "serial_decode_fallback": None,
        "native_caware_decode": None,
        "sampler_mode": "greedy_top_k_1",
        "generation_shape": None,
    }


def _int_tokens_or_none(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    tokens: list[int] = []
    for token in value:
        if not isinstance(token, int) or isinstance(token, bool):
            return None
        tokens.append(int(token))
    return tokens


def _append_stream_ids(accumulated: list[int], candidate: Sequence[int]) -> int:
    row = [int(token) for token in candidate]
    if len(row) > 1 and len(row) >= len(accumulated) and row[: len(accumulated)] == accumulated:
        delta = len(row) - len(accumulated)
        accumulated[:] = row
        return delta
    accumulated.extend(row)
    return len(row)


def extract_stream_response(
    engine: str,
    events: Sequence[tuple[float, Mapping[str, Any] | str]],
    *,
    started_at: float,
    completed_at: float,
    prompt_tokens: int,
) -> dict[str, Any]:
    """Normalize one SSE response and client-observed token timing."""

    normalized_engine = str(engine)
    if normalized_engine not in ENGINE_CHOICES:
        raise ValueError(f"unknown engine: {normalized_engine!r}")
    text_parts: list[str] = []
    generated_ids: list[int] = []
    token_times: list[float] = []
    usage: Mapping[str, Any] = {}
    timings: Mapping[str, Any] = {}
    finish_reason: str | None = None
    done_sentinel = False
    streamed_total = 0
    last_decode_state: Mapping[str, Any] = {}
    last_diagnostics: Mapping[str, Any] = {}
    for observed_at, payload in events:
        if payload == "[DONE]":
            done_sentinel = True
            continue
        if not isinstance(payload, Mapping):
            continue
        if isinstance(payload.get("usage"), Mapping):
            usage = payload["usage"]
        if normalized_engine == "hipengine":
            root_hip = payload.get("hipengine")
            root_hip = root_hip if isinstance(root_hip, Mapping) else {}
            if isinstance(root_hip.get("usage"), Mapping):
                usage = root_hip["usage"]
            choices = payload.get("choices")
            if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)):
                continue
            for raw_choice in choices:
                if not isinstance(raw_choice, Mapping):
                    continue
                choice_hip = raw_choice.get("hipengine")
                choice_hip = choice_hip if isinstance(choice_hip, Mapping) else {}
                decode_state = choice_hip.get("decode_state")
                if isinstance(decode_state, Mapping):
                    last_decode_state = decode_state
                diagnostics = choice_hip.get("diagnostics")
                if isinstance(diagnostics, Mapping):
                    last_diagnostics = diagnostics
                raw_ids = _int_tokens_or_none(choice_hip.get("generated_token_ids"))
                if raw_ids is not None:
                    _append_stream_ids(generated_ids, raw_ids)
                raw_finish = raw_choice.get("finish_reason")
                if raw_finish is not None:
                    finish_reason = str(raw_finish)
                    continue
                text = raw_choice.get("text")
                text = text if isinstance(text, str) else ""
                if text:
                    text_parts.append(text)
                token_payload = choice_hip.get("tokens")
                token_payload = token_payload if isinstance(token_payload, Mapping) else {}
                delta = _number(token_payload.get("delta_tokens"))
                delta_count = max(0, int(delta)) if delta is not None else (1 if text else 0)
                if delta_count:
                    token_times.extend([float(observed_at)] * delta_count)
                    streamed_total += delta_count
            continue

        content = payload.get("content")
        content = content if isinstance(content, str) else ""
        if content:
            text_parts.append(content)
        raw_ids = _int_tokens_or_none(payload.get("tokens"))
        id_delta = 0 if raw_ids is None else _append_stream_ids(generated_ids, raw_ids)
        token_count = id_delta if id_delta else (1 if content else 0)
        if token_count:
            token_times.extend([float(observed_at)] * token_count)
            streamed_total += token_count
        if bool(payload.get("stop")):
            done_sentinel = True
            raw_finish = payload.get("stop_type")
            finish_reason = str(raw_finish or "stop")
        if isinstance(payload.get("timings"), Mapping):
            timings = payload["timings"]

    completion = _number(usage.get("completion_tokens"))
    if completion is None:
        completion = _number(timings.get("predicted_n"))
    completion_tokens = int(completion) if completion is not None else max(
        len(generated_ids), streamed_total
    )
    prompt_count = _number(usage.get("prompt_tokens"))
    if prompt_count is None:
        prompt_count = _number(timings.get("prompt_n"))
    if prompt_count is None:
        prompt_count = _number(timings.get("tokens_evaluated"))
    backend_timing = {
        str(key): float(value)
        for key, value in timings.items()
        if _is_number(value) and str(key).endswith("_ms")
    }
    ttft = None if not token_times else float(token_times[0] - started_at)
    inter_token = [
        float(current - previous)
        for previous, current in zip(token_times, token_times[1:])
    ]
    return {
        "generated_token_ids": generated_ids or None,
        "text": "".join(text_parts),
        "prompt_tokens": int(prompt_count if prompt_count is not None else prompt_tokens),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "backend_timing_ms": backend_timing,
        "execution_path": (
            last_decode_state.get("execution_path")
            if normalized_engine == "hipengine"
            else "llama-server /completion SSE"
        ),
        "serial_decode_fallback": last_decode_state.get("serial_decode_fallback"),
        "native_caware_decode": last_decode_state.get("native_caware_decode"),
        "diagnostics": dict(last_diagnostics) if last_diagnostics else None,
        "client_ttft_seconds": ttft,
        "client_inter_token_seconds": inter_token,
        "wall_seconds": float(completed_at - started_at),
        "token_event_count": len(token_times),
        "done_sentinel": done_sentinel,
    }


def generation_shape_proves_native_group(
    records: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
) -> dict[str, Any]:
    """Validate complete queue groups and their declared physical backend rows."""

    expected = int(concurrency)
    group_ids: list[str] = []
    queue_counts: list[int | None] = []
    queue_prompt_rows: list[int | None] = []
    backend_input_rows: list[int | None] = []
    backend_actual_rows: list[list[int]] = []
    backend_max_rows: list[int | None] = []
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]]] = {}
    row_passes: list[bool] = []
    for record in records:
        shape = record.get("generation_shape")
        shape = shape if isinstance(shape, Mapping) else {}
        queue = shape.get("queue_group")
        queue = queue if isinstance(queue, Mapping) else {}
        raw_groups = shape.get("backend_groups")
        groups = (
            [group for group in raw_groups if isinstance(group, Mapping)]
            if isinstance(raw_groups, Sequence)
            and not isinstance(raw_groups, (str, bytes, bytearray))
            else []
        )
        group_id = str(queue.get("id") or "")
        queue_count = int(queue["request_count"]) if _is_number(queue.get("request_count")) else None
        prompt_rows = int(queue["prompt_rows"]) if _is_number(queue.get("prompt_rows")) else None
        normalized_groups: list[tuple[int, list[int], int]] = []
        for backend in groups:
            input_rows = int(backend["input_rows"]) if _is_number(backend.get("input_rows")) else -1
            raw_actual = backend.get("actual_group_rows")
            actual = (
                [int(value) for value in raw_actual if _is_number(value)]
                if isinstance(raw_actual, Sequence)
                and not isinstance(raw_actual, (str, bytes, bytearray))
                else []
            )
            max_rows = (
                int(backend["max_actual_group_rows"])
                if _is_number(backend.get("max_actual_group_rows"))
                else -1
            )
            normalized_groups.append((input_rows, actual, max_rows))
        first_input = normalized_groups[0][0] if len(normalized_groups) == 1 else None
        first_actual = normalized_groups[0][1] if len(normalized_groups) == 1 else []
        first_max = normalized_groups[0][2] if len(normalized_groups) == 1 else None
        group_ids.append(group_id)
        queue_counts.append(queue_count)
        queue_prompt_rows.append(prompt_rows)
        backend_input_rows.append(first_input)
        backend_actual_rows.append(first_actual)
        backend_max_rows.append(first_max)
        backend_valid = bool(normalized_groups) and all(
            input_rows > 0
            and bool(actual)
            and all(rows > 0 for rows in actual)
            and sum(actual) == input_rows
            and max_rows == max(actual)
            for input_rows, actual, max_rows in normalized_groups
        )
        row_passes.append(
            bool(group_id)
            and queue_count is not None
            and queue_count > 0
            and prompt_rows is not None
            and prompt_rows > 0
            and backend_valid
        )
        if group_id:
            grouped.setdefault(group_id, []).append((queue, shape, groups))

    group_passes: list[bool] = []
    queue_group_request_counts: list[int] = []
    queue_group_prompt_rows: list[int] = []
    flattened_backend_rows: list[int] = []
    native_false_records_expected = 0
    for group_records in grouped.values():
        first_queue, _first_shape, first_groups = group_records[0]
        request_count = int(first_queue.get("request_count", -1))
        prompt_rows = int(first_queue.get("prompt_rows", -1))
        first_backend = [
            (
                int(group.get("input_rows", -1)),
                tuple(int(value) for value in group.get("actual_group_rows", ())),
                int(group.get("max_actual_group_rows", -1)),
            )
            for group in first_groups
        ]
        invariant = all(
            int(queue.get("request_count", -1)) == request_count
            and int(queue.get("prompt_rows", -1)) == prompt_rows
            and [
                (
                    int(group.get("input_rows", -1)),
                    tuple(int(value) for value in group.get("actual_group_rows", ())),
                    int(group.get("max_actual_group_rows", -1)),
                )
                for group in groups
            ]
            == first_backend
            for queue, _shape, groups in group_records
        )
        item_indices = [queue.get("item_index") for queue, _shape, _groups in group_records]
        item_indices_valid = all(value is None for value in item_indices) or (
            all(_is_number(value) for value in item_indices)
            and sorted(int(value) for value in item_indices) == list(range(request_count))
        )
        slices = [
            (queue.get("item_prompt_offset"), queue.get("item_prompt_rows"))
            for queue, _shape, _groups in group_records
        ]
        slices_valid = all(offset is None and rows is None for offset, rows in slices)
        if not slices_valid and all(_is_number(offset) and _is_number(rows) for offset, rows in slices):
            cursor = 0
            slices_valid = True
            for offset, rows in sorted((int(offset), int(rows)) for offset, rows in slices):
                if offset != cursor or rows <= 0:
                    slices_valid = False
                    break
                cursor += rows
            slices_valid = slices_valid and cursor == prompt_rows
        group_passes.append(
            request_count > 0
            and prompt_rows > 0
            and len(group_records) == request_count
            and invariant
            and item_indices_valid
            and slices_valid
        )
        queue_group_request_counts.append(request_count)
        queue_group_prompt_rows.append(prompt_rows)
        group_backend_rows = [rows for _input, actual, _maximum in first_backend for rows in actual]
        flattened_backend_rows.extend(group_backend_rows)
        if group_backend_rows and max(group_backend_rows) == 1:
            native_false_records_expected += request_count

    nonempty_group_ids = [group_id for group_id in group_ids if group_id]
    shared_group = len(nonempty_group_ids) == len(records) and len(set(nonempty_group_ids)) == 1
    passed = bool(
        len(records) == expected
        and len(nonempty_group_ids) == expected
        and all(row_passes)
        and all(group_passes)
        and sum(queue_group_request_counts) == expected
        and sum(queue_group_prompt_rows) == expected
        and bool(flattened_backend_rows)
    )
    return {
        "passed": passed,
        "expected_rows": expected,
        "record_count": len(records),
        "shared_queue_group": shared_group,
        "queue_group_count": len(grouped),
        "queue_group_ids": group_ids,
        "queue_request_counts": queue_counts,
        "queue_prompt_rows": queue_prompt_rows,
        "queue_group_request_counts": queue_group_request_counts,
        "queue_group_prompt_rows": queue_group_prompt_rows,
        "backend_input_rows": backend_input_rows,
        "backend_actual_group_rows": backend_actual_rows,
        "backend_max_actual_group_rows": backend_max_rows,
        "backend_group_rows": flattened_backend_rows,
        "max_backend_group_rows": max(flattened_backend_rows) if flattened_backend_rows else None,
        "native_false_records_expected": native_false_records_expected,
    }


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    """Parse the subset of Prometheus text exposition used by the F1 gate."""

    samples: list[dict[str, Any]] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_LINE_RE.match(line)
        if match is None:
            continue
        labels: dict[str, str] = {}
        label_text = match.group("labels") or ""
        for label_match in _PROMETHEUS_LABEL_RE.finditer(label_text):
            try:
                labels[label_match.group("key")] = str(json.loads(label_match.group("value")))
            except json.JSONDecodeError:
                labels[label_match.group("key")] = label_match.group("value").strip('"')
        value = _number(match.group("value"))
        if value is None:
            continue
        samples.append({"name": match.group("name"), "labels": labels, "value": value})
    return samples


def prometheus_sample(
    samples: Sequence[Mapping[str, Any]],
    name: str,
    **labels: str,
) -> Mapping[str, Any] | None:
    for sample in samples:
        if sample.get("name") != name:
            continue
        sample_labels = sample.get("labels")
        if not isinstance(sample_labels, Mapping):
            continue
        if all(str(sample_labels.get(key)) == str(value) for key, value in labels.items()):
            return sample
    return None


def prometheus_value(
    samples: Sequence[Mapping[str, Any]],
    name: str,
    **labels: str,
) -> float | None:
    sample = prometheus_sample(samples, name, **labels)
    return None if sample is None else float(sample["value"])


def hipengine_latency_snapshot(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    base = "hipengine_resident_request_latency_seconds"
    for kind in ("queue", "time_to_first_token", "inter_token", "service", "completion"):
        row = {
            "count": prometheus_value(samples, f"{base}_count", kind=kind),
            "sum": prometheus_value(samples, f"{base}_sum", kind=kind),
            "max": prometheus_value(samples, "hipengine_resident_request_latency_max_seconds", kind=kind),
            "p50": prometheus_value(samples, base, kind=kind, quantile="0.5"),
            "p95": prometheus_value(samples, base, kind=kind, quantile="0.95"),
        }
        rows[kind] = {key: float(value or 0.0) for key, value in row.items()}
    return rows


def metric_summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": [],
            "median": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "stdev": None,
            "stdev_pct_of_median": None,
        }
    ordered = sorted(samples)
    median = float(statistics.median(samples))
    stdev = float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    p99 = ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]
    return {
        "samples": samples,
        "median": median,
        "p95": float(p95),
        "p99": float(p99),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "stdev": stdev,
        "stdev_pct_of_median": None if median == 0.0 else 100.0 * stdev / median,
    }


def _first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def correctness_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    oracle: Mapping[str, Sequence[int]],
    expected_tokens: int,
    profile: str = "strict",
) -> dict[str, Any]:
    normalized_profile = str(profile)
    if normalized_profile not in _CORRECTNESS_PROFILES:
        raise ValueError(f"unsupported correctness profile: {normalized_profile}")
    control_mismatches: list[dict[str, Any]] = []
    generated_id_mismatches: list[dict[str, Any]] = []
    exact_rows = 0
    for index, record in enumerate(records):
        prompt_hash = str(record.get("prompt_token_ids_sha256") or "")
        expected = oracle.get(prompt_hash)
        raw_generated = record.get("generated_token_ids")
        generated_valid = bool(
            isinstance(raw_generated, Sequence)
            and not isinstance(raw_generated, (str, bytes, bytearray))
            and all(isinstance(token, int) and not isinstance(token, bool) for token in raw_generated)
        )
        generated = [int(token) for token in raw_generated] if generated_valid else []
        control_reasons: list[str] = []
        if expected is None:
            control_reasons.append("prompt_not_in_declared_oracle")
        if not generated_valid:
            control_reasons.append("generated_token_ids_not_integer_sequence")
        if len(generated) != int(expected_tokens):
            control_reasons.append("completion_token_count_mismatch")
        if control_reasons:
            control_mismatches.append(
                {
                    "record_index": index,
                    "request_index": record.get("request_index"),
                    "prompt_token_ids_sha256": prompt_hash,
                    "reasons": control_reasons,
                    "expected_oracle_present": expected is not None,
                    "expected_tokens": int(expected_tokens),
                    "observed_tokens": len(generated),
                }
            )
        first = None if expected is None else _first_mismatch(generated, expected)
        if expected is not None and len(generated) == int(expected_tokens) and first is None:
            exact_rows += 1
        elif expected is not None and generated_valid:
            generated_id_mismatches.append(
                {
                    "record_index": index,
                    "request_index": record.get("request_index"),
                    "prompt_token_ids_sha256": prompt_hash,
                    "expected_tokens": int(expected_tokens),
                    "observed_tokens": len(generated),
                    "first_mismatch_index": first,
                    "expected_token": (
                        int(expected[first])
                        if first is not None and first < len(expected)
                        else None
                    ),
                    "observed_token": (
                        int(generated[first])
                        if first is not None and first < len(generated)
                        else None
                    ),
                }
            )
    generated_id_binding = normalized_profile == "strict"
    binding_mismatches = list(control_mismatches)
    if generated_id_binding:
        binding_mismatches.extend(generated_id_mismatches)
    control_passed = bool(records) and not control_mismatches
    generated_id_equality_passed = bool(records) and not generated_id_mismatches
    return {
        "profile": normalized_profile,
        "passed": control_passed and (
            generated_id_equality_passed if generated_id_binding else True
        ),
        "control_passed": control_passed,
        "generated_id_equality_binding": generated_id_binding,
        "generated_id_equality_passed": generated_id_equality_passed,
        "rows": len(records),
        "exact_rows": exact_rows,
        "mismatch_count": len(binding_mismatches),
        "mismatches": binding_mismatches,
        "control_mismatch_count": len(control_mismatches),
        "control_mismatches": control_mismatches,
        "generated_id_mismatch_count": len(generated_id_mismatches),
        "generated_id_mismatches": generated_id_mismatches,
    }


def _record_output_signature(record: Mapping[str, Any]) -> str:
    raw_ids = record.get("generated_token_ids")
    if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
        payload: Any = [int(token) for token in raw_ids]
    else:
        payload = {
            "text": str(record.get("text") or ""),
            "completion_tokens": int(record.get("completion_tokens") or 0),
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def repeat_determinism_summary(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runs: list[dict[tuple[int, str], str]] = []
    run_sha256: list[str] = []
    for sample in samples:
        rows: dict[tuple[int, str], str] = {}
        for record in sample.get("records", ()):
            if not isinstance(record, Mapping):
                continue
            key = (
                int(record.get("request_index", -1)),
                str(record.get("prompt_token_ids_sha256") or ""),
            )
            rows[key] = _record_output_signature(record)
        runs.append(rows)
        encoded = json.dumps(
            sorted((index, prompt_hash, value) for (index, prompt_hash), value in rows.items()),
            separators=(",", ":"),
        ).encode("utf-8")
        run_sha256.append(hashlib.sha256(encoded).hexdigest())
    mismatches: list[dict[str, Any]] = []
    if runs:
        expected = runs[0]
        for run_index, observed in enumerate(runs[1:], start=1):
            for key in sorted(set(expected) | set(observed)):
                if expected.get(key) != observed.get(key):
                    mismatches.append(
                        {
                            "run_index": run_index,
                            "request_index": key[0],
                            "prompt_token_ids_sha256": key[1],
                            "expected_signature": expected.get(key),
                            "observed_signature": observed.get(key),
                        }
                    )
    return {
        "passed": len(runs) >= 3 and bool(runs[0]) and not mismatches,
        "runs": len(runs),
        "required_runs": 3,
        "run_sha256": run_sha256,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _prompt_rows(*, rows: int, prompt_length: int, token_id: int) -> list[list[int]]:
    prompts: list[list[int]] = []
    for row in range(int(rows)):
        prompt = [int(token_id)] * int(prompt_length)
        prompt[-1] = int(token_id) + (row % 4)
        prompts.append(prompt)
    return prompts


def _parse_concurrencies(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("concurrencies must be unique positive integers")
    return values


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _parse_gpu_max_hw_queues(raw: str) -> int | None:
    normalized = str(raw).strip().lower()
    if normalized == "unset":
        return None
    try:
        value = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "gpu-max-hw-queues must be one of 1,2,4,8,unset"
        ) from exc
    if value not in {1, 2, 4, 8}:
        raise argparse.ArgumentTypeError(
            "gpu-max-hw-queues must be one of 1,2,4,8,unset"
        )
    return value


def _gpu_max_hw_queues_label(value: int | None) -> str:
    return "unset" if value is None else str(int(value))


def _validate_concurrency_plan(
    concurrencies: Sequence[int],
    *,
    live_concurrency: int,
    require_c1: bool = True,
) -> list[int]:
    values = [int(value) for value in concurrencies]
    allowed = set(range(1, 33))
    if require_c1 and 1 not in values:
        raise ValueError("concurrencies must include c1 unless focused-width repair is explicit")
    if int(live_concurrency) not in values:
        raise ValueError("live-concurrency must appear in concurrencies")
    if any(value not in allowed for value in values):
        raise ValueError("the matched server packet is limited to logical c1-c32")
    return values


def _post_json(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            parsed = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise BenchError(f"HTTP {exc.code} from {url}: {body}") from exc
    if not isinstance(parsed, dict):
        raise BenchError(f"expected JSON object from {url}, got {type(parsed).__name__}")
    return parsed


def _get_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _request_payload(args: argparse.Namespace, engine: str, prompt: Sequence[int], request_index: int) -> dict[str, Any]:
    if engine == "hipengine":
        return {
            "model": str(args.served_model_name),
            "prompt": [int(token) for token in prompt],
            "max_tokens": int(args.decode_tokens),
            "temperature": 0.0,
            "top_k": int(args.hipengine_top_k),
            "top_p": 1.0,
            "seed": int(args.seed),
            "ignore_eos": True,
            "stream": False,
        }
    return {
        "prompt": [int(token) for token in prompt],
        "n_predict": int(args.decode_tokens),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": int(args.seed),
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
        "return_tokens": True,
    }


def _completion_url(engine: str, base_url: str) -> str:
    return f"{base_url}/v1/completions" if engine == "hipengine" else f"{base_url}/completion"


def _one_request(
    args: argparse.Namespace,
    *,
    engine: str,
    base_url: str,
    prompt: Sequence[int],
    index: int,
    release: threading.Event,
    epoch: float | Sequence[float],
) -> dict[str, Any]:
    release.wait()
    release_epoch = float(epoch[0]) if isinstance(epoch, Sequence) else float(epoch)
    started = time.perf_counter()
    response = _post_json(
        _completion_url(engine, base_url),
        _request_payload(args, engine, prompt, index),
        float(args.request_timeout),
    )
    completed = time.perf_counter()
    record = extract_response(engine, response, prompt_tokens=len(prompt))
    record.update(
        {
            "request_index": int(index),
            "prompt_token_ids_sha256": token_ids_sha256(prompt),
            "wall_seconds": completed - started,
            "started_offset_seconds": started - release_epoch,
            "completed_offset_seconds": completed - release_epoch,
            "generated_token_ids_sha256": token_ids_sha256(record["generated_token_ids"]),
            "first_generated_token_ids": list(record["generated_token_ids"][:8]),
            "last_generated_token_ids": list(record["generated_token_ids"][-8:]),
        }
    )
    return record


def _parse_sse_data_line(raw_line: bytes | str) -> Mapping[str, Any] | str | None:
    text = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
    stripped = text.strip()
    if not stripped.startswith("data:"):
        return None
    payload = stripped[5:].strip()
    if not payload:
        return None
    if payload == "[DONE]":
        return payload
    parsed = json.loads(payload)
    if not isinstance(parsed, Mapping):
        raise BenchError(f"SSE data payload must be an object, got {type(parsed).__name__}")
    return parsed


def _stream_request_payload(
    args: argparse.Namespace,
    engine: str,
    prompt: Sequence[int],
    request_index: int,
) -> dict[str, Any]:
    payload = _request_payload(args, engine, prompt, request_index)
    payload["stream"] = True
    if engine == "hipengine":
        payload["stream_options"] = {
            "include_hipengine": True,
            "include_usage": True,
        }
    return payload


def _one_stream_request(
    args: argparse.Namespace,
    *,
    engine: str,
    base_url: str,
    prompt: Sequence[int],
    index: int,
    release: threading.Event,
    epoch: float | Sequence[float],
) -> dict[str, Any]:
    release.wait()
    release_epoch = float(epoch[0]) if isinstance(epoch, Sequence) else float(epoch)
    parsed_url = urllib.parse.urlparse(_completion_url(engine, base_url))
    if parsed_url.scheme != "http" or parsed_url.hostname is None:
        raise BenchError(f"stream benchmark requires a localhost HTTP URL, got {parsed_url.geturl()!r}")
    started = time.perf_counter()
    connection = http.client.HTTPConnection(
        parsed_url.hostname,
        parsed_url.port,
        timeout=float(args.request_timeout),
    )
    events: list[tuple[float, Mapping[str, Any] | str]] = []
    status = 0
    try:
        connection.request(
            "POST",
            parsed_url.path,
            body=json.dumps(
                _stream_request_payload(args, engine, prompt, index),
                separators=(",", ":"),
            ),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        status = int(response.status)
        if status != 200:
            body = response.read().decode("utf-8", errors="replace")
            raise BenchError(f"HTTP {status} from {parsed_url.geturl()}: {body}")
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            observed_at = time.perf_counter()
            payload = _parse_sse_data_line(raw_line)
            if payload is not None:
                events.append((observed_at, payload))
    finally:
        connection.close()
    completed = time.perf_counter()
    record = extract_stream_response(
        engine,
        events,
        started_at=started,
        completed_at=completed,
        prompt_tokens=len(prompt),
    )
    record.update(
        {
            "status_code": status,
            "request_index": int(index),
            "prompt_token_ids_sha256": token_ids_sha256(prompt),
            "started_offset_seconds": started - release_epoch,
            "completed_offset_seconds": completed - release_epoch,
        }
    )
    return record


def _batch_summary(records: Sequence[Mapping[str, Any]], *, batch_wall_seconds: float) -> dict[str, Any]:
    total = sum(int(record["completion_tokens"]) for record in records)
    decode_ms = [
        float(record["backend_timing_ms"]["decode_batch_ms"])
        for record in records
        if isinstance(record.get("backend_timing_ms"), Mapping)
        and _is_number(record["backend_timing_ms"].get("decode_batch_ms"))
    ]
    if not decode_ms:
        decode_ms = [
            float(record["backend_timing_ms"]["predicted_ms"])
            for record in records
            if isinstance(record.get("backend_timing_ms"), Mapping)
            and _is_number(record["backend_timing_ms"].get("predicted_ms"))
        ]
    max_backend_decode_ms = max(decode_ms) if decode_ms else None
    return {
        "batch_wall_seconds": float(batch_wall_seconds),
        "total_completion_tokens": int(total),
        "http_wall_tok_s_aggregate": total / batch_wall_seconds if batch_wall_seconds > 0.0 else None,
        "http_wall_tok_s_per_request": (
            total / batch_wall_seconds / len(records) if batch_wall_seconds > 0.0 and records else None
        ),
        "max_request_wall_seconds": max(float(record["wall_seconds"]) for record in records),
        "max_backend_decode_ms": max_backend_decode_ms,
        "backend_native_decode_tok_s_aggregate_diagnostic": (
            1000.0 * total / max_backend_decode_ms
            if max_backend_decode_ms is not None and max_backend_decode_ms > 0.0
            else None
        ),
        "records": sorted((dict(record) for record in records), key=lambda row: int(row["request_index"])),
    }


def _stream_batch_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_wall_seconds: float,
    ttft_p95_limit: float,
    itl_p99_limit: float,
    e2e_p95_limit: float,
) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    total_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    ttft = [
        float(row["client_ttft_seconds"])
        for row in rows
        if _is_number(row.get("client_ttft_seconds"))
    ]
    itl = [
        float(value)
        for row in rows
        for value in row.get("client_inter_token_seconds") or ()
        if _is_number(value)
    ]
    e2e = [float(row["wall_seconds"]) for row in rows if _is_number(row.get("wall_seconds"))]
    qualifying = [
        row
        for row in rows
        if row.get("stream_correctness_passed", row.get("stream_exact")) is True
        and _is_number(row.get("client_ttft_seconds"))
        and float(row["client_ttft_seconds"]) <= float(ttft_p95_limit)
        and all(
            float(value) <= float(itl_p99_limit)
            for value in row.get("client_inter_token_seconds") or ()
        )
        and _is_number(row.get("wall_seconds"))
        and float(row["wall_seconds"]) <= float(e2e_p95_limit)
    ]
    qualifying_tokens = sum(int(row.get("completion_tokens") or 0) for row in qualifying)
    ttft_summary = metric_summary(ttft)
    itl_summary = metric_summary(itl)
    e2e_summary = metric_summary(e2e)
    exact = bool(rows) and all(
        row.get("stream_exact") is True
        and row.get("stream_protocol_complete") is True
        for row in rows
    )
    correctness_qualified = bool(rows) and all(
        row.get("stream_correctness_passed", row.get("stream_exact")) is True
        and row.get("stream_protocol_complete") is True
        for row in rows
    )
    wall = float(batch_wall_seconds)
    return {
        "passed": correctness_qualified,
        "batch_wall_seconds": wall,
        "total_completion_tokens": total_tokens,
        "exact_generated_tok_s_aggregate": (
            total_tokens / wall if exact and wall > 0.0 else 0.0
        ),
        "correctness_qualified_tok_s_aggregate": (
            total_tokens / wall if correctness_qualified and wall > 0.0 else 0.0
        ),
        "generated_id_equality_passed": exact,
        "slo_goodput_tok_s_aggregate": qualifying_tokens / wall if wall > 0.0 else 0.0,
        "latency_seconds": {
            "ttft": ttft_summary,
            "itl": itl_summary,
            "end_to_end": e2e_summary,
        },
        "slo": {
            "thresholds": {
                "ttft_p95_seconds": float(ttft_p95_limit),
                "itl_p99_seconds": float(itl_p99_limit),
                "end_to_end_p95_seconds": float(e2e_p95_limit),
            },
            "checks": {
                "ttft_p95": bool(
                    ttft_summary["p95"] is not None
                    and float(ttft_summary["p95"]) <= float(ttft_p95_limit)
                ),
                "itl_p99": bool(
                    itl_summary["p99"] is not None
                    and float(itl_summary["p99"]) <= float(itl_p99_limit)
                ),
                "end_to_end_p95": bool(
                    e2e_summary["p95"] is not None
                    and float(e2e_summary["p95"]) <= float(e2e_p95_limit)
                ),
            },
            "qualifying_requests": len(qualifying),
            "qualifying_completion_tokens": qualifying_tokens,
            "passed": len(qualifying) == len(rows),
        },
        "records": sorted(rows, key=lambda row: int(row.get("request_index", 0))),
    }


def _stream_route_summary(
    engine: str,
    *,
    concurrency: int,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = [
        record
        for sample in samples
        for record in sample.get("records", ())
        if isinstance(record, Mapping)
    ]
    paths = sorted({str(record.get("execution_path")) for record in records})
    serial = [record.get("serial_decode_fallback") for record in records]
    native = [record.get("native_caware_decode") for record in records]
    if str(engine) != "hipengine":
        return {"passed": bool(records), "paths": paths}
    native_expected = False if int(concurrency) == 1 else True
    paro_native_path = "paro_resident_native_width_decode"
    paro_serial_path = "paro_resident_serial_decode"
    laguna_scheduler_path = "laguna_resident_scheduler_c1"
    if paths == [laguna_scheduler_path]:
        return {
            "passed": bool(records)
            and all(value is False for value in serial)
            and all(value is False for value in native),
            "route_policy": "scheduler_native_model_c1",
            "paths": paths,
            "serial_decode_fallback_values": sorted(
                {value for value in serial if isinstance(value, bool)}
            ),
            "native_caware_decode_values": sorted(
                {value for value in native if isinstance(value, bool)}
            ),
            "native_caware_decode_expected": False,
        }
    if paths and set(paths).issubset({paro_native_path, paro_serial_path}):
        if int(concurrency) == 1:
            records_consistent = all(
                record.get("execution_path") == paro_serial_path
                and record.get("serial_decode_fallback") is True
                and record.get("native_caware_decode") is False
                for record in records
            )
            topology_valid = paths == [paro_serial_path]
        else:
            records_consistent = all(
                isinstance(record.get("serial_decode_fallback"), bool)
                and record.get("native_caware_decode") is True
                for record in records
            )
            topology_valid = paro_native_path in paths
        return {
            "passed": bool(records) and records_consistent and topology_valid,
            "route_policy": "paro_occupancy_adaptive",
            "paths": paths,
            "serial_decode_fallback_values": sorted(
                {value for value in serial if isinstance(value, bool)}
            ),
            "native_caware_decode_values": sorted(
                {value for value in native if isinstance(value, bool)}
            ),
            "native_caware_decode_expected": native_expected,
        }
    return {
        "passed": bool(records)
        and all(value is False for value in serial)
        and all(value is native_expected for value in native)
        and all(path == "gguf_packed_ar_server_decode" for path in paths),
        "paths": paths,
        "serial_decode_fallback_values": sorted(
            {value for value in serial if isinstance(value, bool)}
        ),
        "native_caware_decode_values": sorted(
            {value for value in native if isinstance(value, bool)}
        ),
        "native_caware_decode_expected": native_expected,
    }


def _run_burst(
    args: argparse.Namespace,
    *,
    engine: str,
    base_url: str,
    prompts: Sequence[Sequence[int]],
) -> dict[str, Any]:
    release = threading.Event()
    epoch = [time.perf_counter()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(
                _one_request,
                args,
                engine=engine,
                base_url=base_url,
                prompt=prompt,
                index=index,
                release=release,
                epoch=epoch,
            )
            for index, prompt in enumerate(prompts)
        ]
        epoch[0] = time.perf_counter()
        release.set()
        records = [future.result() for future in futures]
        completed = time.perf_counter()
    return _batch_summary(records, batch_wall_seconds=completed - epoch[0])


def _run_stream_burst(
    args: argparse.Namespace,
    *,
    engine: str,
    base_url: str,
    prompts: Sequence[Sequence[int]],
    oracle: Mapping[str, Sequence[int]],
    oracle_text: Mapping[str, str],
) -> dict[str, Any]:
    release = threading.Event()
    epoch = [time.perf_counter()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(
                _one_stream_request,
                args,
                engine=engine,
                base_url=base_url,
                prompt=prompt,
                index=index,
                release=release,
                epoch=epoch,
            )
            for index, prompt in enumerate(prompts)
        ]
        epoch[0] = time.perf_counter()
        release.set()
        records = [future.result() for future in futures]
        completed = time.perf_counter()
    for record in records:
        prompt_hash = str(record["prompt_token_ids_sha256"])
        expected_ids = [int(token) for token in oracle.get(prompt_hash, ())]
        expected_text = oracle_text.get(prompt_hash)
        observed_ids = record.get("generated_token_ids")
        if isinstance(observed_ids, Sequence) and not isinstance(
            observed_ids, (str, bytes, bytearray)
        ):
            output_exact = [int(token) for token in observed_ids] == expected_ids
            oracle_mode = "exact_generated_token_ids"
        else:
            output_exact = (
                expected_text is not None
                and str(record.get("text") or "") == str(expected_text)
                and int(record.get("completion_tokens") or 0) == int(args.decode_tokens)
            )
            oracle_mode = "exact_blocking_text_plus_completion_count"
        protocol_complete = bool(
            int(record.get("status_code") or 0) == 200
            and record.get("done_sentinel") is True
            and int(record.get("completion_tokens") or 0) == int(args.decode_tokens)
            and int(record.get("token_event_count") or 0) == int(args.decode_tokens)
            and record.get("finish_reason") is not None
        )
        correctness_profile = str(args.correctness_profile)
        generated_id_binding = correctness_profile == "strict"
        record["stream_oracle_mode"] = oracle_mode
        record["stream_output_exact"] = output_exact
        record["stream_generated_id_equality_binding"] = generated_id_binding
        record["stream_protocol_complete"] = protocol_complete
        record["stream_exact"] = bool(output_exact and protocol_complete)
        record["stream_correctness_passed"] = bool(
            protocol_complete and (output_exact if generated_id_binding else True)
        )
        record["generated_token_ids_sha256"] = (
            None
            if observed_ids is None
            else token_ids_sha256([int(token) for token in observed_ids])
        )
    return _stream_batch_summary(
        records,
        batch_wall_seconds=completed - epoch[0],
        ttft_p95_limit=float(args.slo_ttft_p95_seconds),
        itl_p99_limit=float(args.slo_itl_p99_seconds),
        e2e_p95_limit=float(args.slo_end_to_end_p95_seconds),
    )


def _metrics_state(base_url: str, *, timeout: float = 10.0) -> tuple[list[dict[str, Any]], str]:
    text = _get_text(f"{base_url}/metrics", timeout=timeout)
    return parse_prometheus(text), text


def _compact_poll_state(samples: Sequence[Mapping[str, Any]], *, at_seconds: float) -> dict[str, Any]:
    bucket = prometheus_sample(samples, "hipengine_resident_bucket_info")
    labels = bucket.get("labels") if isinstance(bucket, Mapping) else {}
    labels = labels if isinstance(labels, Mapping) else {}
    return {
        "at_seconds": round(float(at_seconds), 6),
        "active_rows": prometheus_value(samples, "hipengine_resident_bucket_active_rows"),
        "occupied_slots": prometheus_value(samples, "hipengine_resident_bucket_occupied_slots"),
        "generation_requests_active": prometheus_value(samples, "hipengine_generation_requests_active"),
        "decode_work_total": prometheus_value(samples, "hipengine_resident_work_decode_total"),
        "prefill_work_total": prometheus_value(samples, "hipengine_resident_work_prefill_total"),
        "kv_int8_payload_bytes": prometheus_value(
            samples, "hipengine_resident_kv_int8_payload_bytes"
        ),
        "kv_bf16_payload_bytes": prometheus_value(
            samples, "hipengine_resident_kv_bf16_payload_bytes"
        ),
        "kv_scale_bytes": prometheus_value(
            samples, "hipengine_resident_kv_scale_bytes"
        ),
        "kv_bf16_mirror_bytes": prometheus_value(
            samples, "hipengine_resident_kv_bf16_mirror_bytes"
        ),
        "kv_total_bytes": prometheus_value(
            samples, "hipengine_resident_kv_total_bytes"
        ),
        "active_mask": labels.get("active_mask"),
        "last_work_kind": labels.get("last_work_kind"),
    }


def _hipengine_route_expectation_passes(
    *,
    concurrency: int,
    expectation: str,
    serial_values: Sequence[Any],
    native_values: Sequence[Any],
    shape_passed: bool,
    resident_capacity: float | None,
    execution_paths: Sequence[str] = (),
    native_false_records_expected: int = 0,
) -> bool:
    rows = int(concurrency)
    if resident_capacity != float(rows):
        return False
    if (
        not serial_values
        or len(serial_values) != len(native_values)
        or len(serial_values) % rows != 0
    ):
        return False
    if str(expectation) == "serial":
        return all(value is True for value in serial_values) and all(
            value is False for value in native_values
        )
    if str(expectation) == "serial-c1-per-row":
        serial_route_observed = (
            all(value is False for value in serial_values)
            if rows == 1
            else (
                any(value is True for value in serial_values)
                and all(isinstance(value, bool) for value in serial_values)
            )
        )
        return (
            set(str(path) for path in execution_paths)
            == {"gguf_packed_ar_server_decode"}
            and serial_route_observed
            and all(value is False for value in native_values)
        )
    if str(expectation) == "scheduler-c1":
        return (
            set(str(path) for path in execution_paths)
            == {"laguna_resident_scheduler_c1"}
            and all(value is False for value in serial_values)
            and all(value is False for value in native_values)
        )
    if str(expectation) != "native" or not bool(shape_passed):
        return False
    if rows == 1:
        return all(value is False for value in native_values)
    return (
        all(value is False for value in serial_values)
        and all(isinstance(value, bool) for value in native_values)
        and any(value is True for value in native_values)
    )


def _oracle_join_delay_seconds(
    oracle_records: Sequence[Mapping[str, Any]],
    *,
    join_after_tokens: int,
    expected_tokens: int,
) -> float:
    delays: list[float] = []
    for record in oracle_records:
        timing = record.get("backend_timing_ms")
        if not isinstance(timing, Mapping):
            continue
        prompt_ms = _number(timing.get("prefill_ms"))
        decode_ms = _number(timing.get("decode_batch_ms"))
        decode_steps = max(1, int(expected_tokens) - 1)
        if prompt_ms is None:
            prompt_ms = _number(timing.get("prompt_ms"))
        if decode_ms is None and prompt_ms is not None:
            request_total_ms = _number(timing.get("request_total_ms"))
            if request_total_ms is not None and request_total_ms > prompt_ms:
                decode_ms = request_total_ms - prompt_ms
        if decode_ms is None:
            decode_ms = _number(timing.get("predicted_ms"))
            decode_steps = max(1, int(expected_tokens))
        if prompt_ms is None or decode_ms is None:
            request_wall = _number(record.get("wall_seconds"))
            if request_wall is not None and request_wall > 0.0:
                # PARO's resident telemetry exposes authoritative request total
                # but not a separate prefill/decode split. Keep the metrics poll
                # alive through 90% of the observed c1 HTTP wall so it can join
                # on an actual decode transition without racing completion.
                delays.append(0.9 * request_wall)
            continue
        delays.append((prompt_ms + int(join_after_tokens) * decode_ms / decode_steps) / 1000.0)
    if not delays:
        raise BenchError("oracle records do not expose enough timing to schedule live admission")
    return max(0.001, float(statistics.median(delays)))


def _live_request_function(args: argparse.Namespace):
    return _one_stream_request if bool(args.streaming_primary) else _one_request


def _live_admission_passes(
    engine: str,
    args: argparse.Namespace,
    live: Mapping[str, Any],
) -> bool:
    if not bool(live.get("admission_during_first_request")):
        return False
    if engine != "hipengine" or not bool(args.streaming_primary):
        return True
    return bool(
        live.get("request_protocol") == "streaming_sse"
        and live.get("join_during_observed_first_stream_decode") is True
        and live.get("resident_overlap_before_first_completion") is True
    )


def _run_live_admission(
    args: argparse.Namespace,
    *,
    engine: str,
    base_url: str,
    prompts: Sequence[Sequence[int]],
    oracle_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(prompts) < 2:
        raise ValueError("live admission requires at least two prompts")
    first_release = threading.Event()
    join_release = threading.Event()
    epoch = [time.perf_counter()]
    poll_events: list[dict[str, Any]] = []
    trigger: dict[str, Any] | None = None
    strategy = (
        "observed_resident_decode_or_c1_timed_decode_join"
        if engine == "hipengine"
        else "c1_oracle_prefill_plus_decode_offset"
    )
    fallback_delay = _oracle_join_delay_seconds(
        oracle_records,
        join_after_tokens=int(args.live_join_after_tokens),
        expected_tokens=int(args.decode_tokens),
    )
    request_function = _live_request_function(args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(
                request_function,
                args,
                engine=engine,
                base_url=base_url,
                prompt=prompt,
                index=index,
                release=first_release if index == 0 else join_release,
                epoch=epoch,
            )
            for index, prompt in enumerate(prompts)
        ]
        epoch[0] = time.perf_counter()
        first_release.set()
        if engine == "hipengine":
            planned_join_at = epoch[0] + fallback_delay
            poll_seconds = max(0.005, float(args.metrics_poll_ms) / 1000.0)
            metrics_timeout = max(0.1, poll_seconds)
            last_signature: tuple[Any, ...] | None = None
            while time.perf_counter() < planned_join_at and not futures[0].done():
                remaining = max(0.001, planned_join_at - time.perf_counter())
                try:
                    samples, _ = _metrics_state(
                        base_url,
                        timeout=min(metrics_timeout, remaining),
                    )
                except Exception:
                    continue
                state = _compact_poll_state(samples, at_seconds=time.perf_counter() - epoch[0])
                state["phase"] = "before_join"
                state["first_request_done"] = futures[0].done()
                signature = tuple(state.get(key) for key in state if key != "at_seconds")
                if signature != last_signature:
                    poll_events.append(state)
                    last_signature = signature
                if (
                    state.get("last_work_kind") == "decode"
                    and float(state.get("active_rows") or 0.0) == 1.0
                    and float(state.get("decode_work_total") or 0.0) > 0.0
                ):
                    trigger = dict(state)
                    trigger["source"] = "observed_resident_decode"
                    trigger["first_request_done_before_join"] = futures[0].done()
                    break
            if trigger is None:
                remaining = planned_join_at - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                trigger = {
                    "at_seconds": time.perf_counter() - epoch[0],
                    "source": "c1_oracle_prefill_plus_decode_offset",
                    "first_request_done_before_join": futures[0].done(),
                }
        else:
            time.sleep(fallback_delay)
            trigger = {
                "at_seconds": time.perf_counter() - epoch[0],
                "source": "c1_oracle_prefill_plus_decode_offset",
                "first_request_done_before_join": futures[0].done(),
            }
        join_release.set()
        if engine == "hipengine" and bool(args.streaming_primary):
            while not all(future.done() for future in futures):
                try:
                    samples, _ = _metrics_state(base_url, timeout=metrics_timeout)
                except Exception:
                    continue
                state = _compact_poll_state(samples, at_seconds=time.perf_counter() - epoch[0])
                state["phase"] = "after_join"
                state["first_request_done"] = futures[0].done()
                signature = tuple(state.get(key) for key in state if key != "at_seconds")
                if signature != last_signature:
                    poll_events.append(state)
                    last_signature = signature
                time.sleep(poll_seconds)
        records = [future.result() for future in futures]
    wall = max(float(record["completed_offset_seconds"]) for record in records)
    result = _batch_summary(records, batch_wall_seconds=wall)
    first_record = min(records, key=lambda record: int(record["request_index"]))
    first_ttft = first_record.get("client_ttft_seconds")
    join_at = trigger.get("at_seconds")
    join_during_observed_stream_decode = bool(
        request_function is _one_stream_request
        and _is_number(first_ttft)
        and _is_number(join_at)
        and float(first_ttft) <= float(join_at) < float(first_record["completed_offset_seconds"])
    )
    overlap_observed = any(
        _is_number(state.get("active_rows"))
        and float(state["active_rows"]) >= 2.0
        and state.get("first_request_done") is False
        for state in poll_events
    )
    result.update(
        {
            "strategy": strategy,
            "request_protocol": (
                "streaming_sse" if request_function is _one_stream_request else "blocking_http"
            ),
            "oracle_timing_fallback_seconds": fallback_delay,
            "join_after_decode_tokens_target": int(args.live_join_after_tokens),
            "join_trigger": trigger,
            "observed_decode_trigger": trigger.get("source") == "observed_resident_decode",
            "join_during_observed_first_stream_decode": join_during_observed_stream_decode,
            "resident_overlap_before_first_completion": overlap_observed,
            "observed_active_rows": sorted(
                {
                    int(float(state["active_rows"]))
                    for state in poll_events
                    if _is_number(state.get("active_rows"))
                }
            ),
            "poll_events": poll_events,
            "admission_during_first_request": not bool(trigger.get("first_request_done_before_join")),
        }
    )
    return result


def _capture(command: Sequence[str], *, cwd: Path | None = None, timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"command": list(command), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": list(command),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _sampled_file_fingerprint(path: Path, *, sample_bytes: int = 1 << 20) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    size = resolved.stat().st_size
    offsets = sorted({0, max(0, size // 2 - sample_bytes // 2), max(0, size - sample_bytes)})
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(handle.read(sample_bytes))
    return {
        "path": str(resolved),
        "size_bytes": size,
        "algorithm": "sha256-sampled-v1",
        "sampled_bytes_per_offset": int(sample_bytes),
        "sample_offsets": offsets,
        "value": digest.hexdigest(),
    }


def _model_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return {**_sampled_file_fingerprint(resolved), "path_type": "file", "revision": None}
    identity = collect_model_identity(resolved)
    fingerprint = identity.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ValueError(f"model fingerprint is unavailable: {resolved}")
    return {
        "path": str(identity.get("path") or resolved),
        "revision": identity.get("revision"),
        **dict(fingerprint),
    }


def _full_file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def parse_ldd_local_paths(text: str, *, root: Path) -> list[Path]:
    """Return existing absolute ldd targets located under one build tree."""

    resolved_root = root.expanduser().resolve()
    paths: list[Path] = []
    for raw_line in str(text).splitlines():
        fields = raw_line.strip().split()
        candidate = None
        if len(fields) >= 3 and fields[1] == "=>" and fields[2].startswith("/"):
            candidate = Path(fields[2])
        elif fields and fields[0].startswith("/"):
            candidate = Path(fields[0])
        if candidate is None or not candidate.exists():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            continue
        if resolved not in paths:
            paths.append(resolved)
    return sorted(paths)


def _server_paths(args: argparse.Namespace, engine: str) -> tuple[Path | None, Path | None]:
    if engine == "llamacpp-hip":
        return args.llamacpp_hip_repo, args.llamacpp_hip_server_bin
    if engine == "llamacpp-vulkan":
        return args.llamacpp_vulkan_repo, args.llamacpp_vulkan_server_bin
    return REPO_ROOT, None


def _process_env_report_path(args: argparse.Namespace, *, port: int) -> Path:
    return args.work_dir / f"hipengine-process-env-{int(port)}.json"


def _server_command_and_env(
    args: argparse.Namespace,
    *,
    engine: str,
    concurrency: int,
    port: int,
) -> tuple[list[str], dict[str, str], Path]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpu_max_hw_queues is None:
        env.pop("GPU_MAX_HW_QUEUES", None)
        env["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] = "runtime_default"
    else:
        env["GPU_MAX_HW_QUEUES"] = str(int(args.gpu_max_hw_queues))
        env["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] = "explicit"
    if args.compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if engine == "hipengine":
        # hipEngine returns from this branch before the llama.cpp device-env
        # setup below. HIP_VISIBLE_DEVICES alone is intentional: applying the
        # same numeric selector again through ROCR_VISIBLE_DEVICES can filter
        # the already-remapped one-device view down to zero devices.
        env["HIP_VISIBLE_DEVICES"] = str(args.gpu)
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env["HIPENGINE_PROCESS_ENV_REPORT_PATH"] = str(
            _process_env_report_path(args, port=port)
        )
        env["HIPENGINE_PREFILL_DECODE_POLICY"] = str(
            args.hipengine_prefill_decode_policy
        )
        if args.hipengine_prefill_chunk_tokens is not None:
            env["HIPENGINE_MAX_PREFILL_CHUNK_TOKENS"] = str(
                args.hipengine_prefill_chunk_tokens
            )
        env["HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS"] = "1"
        native_batch = args.hipengine_route_expectation == "native"
        env["HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"] = (
            "1" if native_batch else "0"
        )
        env["HIPENGINE_GGUF_AR_PACKED_DECODE"] = "1" if native_batch else "0"
        command = [
            str(args.hipengine_python),
            "-m",
            "hipengine.server",
            "--model",
            str(args.model),
            "--backend",
            str(args.backend),
            "--quant",
            str(args.quant),
            "--served-model-name",
            str(args.served_model_name),
            "--max-context-tokens",
            str(args.ctx_per_seq),
            "--kv-storage",
            str(args.hipengine_kv_storage),
            "--kv-scale-dtype",
            str(args.hipengine_kv_scale_dtype),
            "--kv-scale-granularity",
            str(args.hipengine_kv_scale_granularity),
            "--generation-batch-window-ms",
            str(args.batch_window_ms),
            "--max-active-requests",
            str(concurrency),
            "--metrics",
            "prometheus",
            "--speculative-mtp-serving",
            "off",
            "--prefix-cache",
            "off",
            "--no-startup-chat-smoke",
            "--no-startup-scratch-probe",
            "--shutdown-grace-seconds",
            "10",
            "--host",
            str(args.host),
            "--port",
            str(port),
            "--log-level",
            str(args.server_log_level),
        ]
        return command, env, REPO_ROOT

    repo, server_bin = _server_paths(args, engine)
    assert repo is not None and server_bin is not None
    command = [
        str(server_bin),
        "-m",
        str(args.model),
        "-ngl",
        str(args.n_gpu_layers),
        "-fa",
        "on",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "-c",
        str(int(args.ctx_per_seq) * int(concurrency)),
        "-np",
        str(concurrency),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "--host",
        str(args.host),
        "--port",
        str(port),
        "--no-webui",
        "--no-cache-prompt",
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--metrics",
        "--no-warmup",
    ]
    if engine == "llamacpp-vulkan":
        env["DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"] = "1"
        env["VK_DRIVER_FILES"] = str(args.vk_driver_files)
        env["GGML_VK_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        env["HIP_VISIBLE_DEVICES"] = str(args.gpu)
        env["ROCR_VISIBLE_DEVICES"] = str(args.gpu)
    return command, env, repo


def _effective_server_environment(
    args: argparse.Namespace,
    *,
    engine: str,
) -> dict[str, str | None]:
    """Return the selected environment axes exactly as the server receives them."""

    _, environment, _ = _server_command_and_env(
        args,
        engine=engine,
        concurrency=int(args.concurrencies[0]),
        port=int(args.port_base),
    )
    return {key: environment.get(key) for key in _EFFECTIVE_SERVER_ENV_KEYS}


def _resolve_correctness_contract(
    args: argparse.Namespace,
    *,
    engine: str,
) -> dict[str, Any]:
    profile = str(args.correctness_profile)
    artifact_path = args.production_correctness_artifact
    effective_environment = _effective_server_environment(args, engine=engine)
    if profile == "strict":
        if artifact_path is not None:
            raise ValueError(
                "--production-correctness-artifact is valid only with "
                "--correctness-profile production"
            )
        return {
            "profile": "strict",
            "arithmetic_binding": "exact same-engine c1 generated trajectory",
            "generated_id_equality_binding": True,
            "public_profile_qualification_claim": False,
            "runtime_execution_profile": effective_environment.get(
                "HIPENGINE_EXECUTION_PROFILE"
            ),
            "bundle": None,
        }
    if profile != "production":
        raise ValueError(f"unsupported correctness profile: {profile}")
    if engine != "hipengine":
        raise ValueError("production correctness profile is supported only for hipEngine")
    if artifact_path is None:
        raise ValueError(
            "--correctness-profile production requires "
            "--production-correctness-artifact"
        )
    if int(args.measured_runs) < 3:
        raise ValueError("production correctness requires at least three measured runs")
    if bool(args.streaming_primary) and int(args.stream_measured_runs) < 3:
        raise ValueError("production streaming correctness requires at least three measured runs")
    if effective_environment.get("HIPENGINE_GGUF_FP16_RECURRENT_STATE") != "1":
        raise ValueError(
            "production correctness requires explicit "
            "HIPENGINE_GGUF_FP16_RECURRENT_STATE=1"
        )
    if effective_environment.get("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS") is not None:
        raise ValueError("production correctness forbids a physical-width environment override")

    requested_bundle = Path(artifact_path).expanduser()
    if requested_bundle.is_symlink():
        raise ValueError(
            f"production correctness bundle must not be a symlink: {requested_bundle}"
        )
    resolved = requested_bundle.resolve()
    if not resolved.is_file():
        raise ValueError(f"production correctness bundle must be a regular file: {resolved}")
    bundle = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(bundle, Mapping):
        raise ValueError("production correctness bundle root must be an object")
    if bundle.get("schema_version") != 1:
        raise ValueError("production correctness bundle schema_version must be 1")
    if bundle.get("kind") != _PRODUCTION_CORRECTNESS_BUNDLE_KIND:
        raise ValueError("unexpected production correctness bundle kind")
    if bundle.get("status") != "passed" or bundle.get("correctness_profile") != "production":
        raise ValueError("production correctness bundle must have passed production status")
    runtime_scope = str(bundle.get("runtime_scope") or "")
    public_claim = bundle.get("profile_qualification_claim")
    if not (
        (runtime_scope == "named_production" and public_claim is True)
        or (
            runtime_scope == "scoped_legacy_default_candidate"
            and public_claim is False
        )
    ):
        raise ValueError("production correctness bundle has an invalid runtime/profile claim")
    if bundle.get("generated_id_equality_binding") is not False:
        raise ValueError("production generated-ID equality must be diagnostic")

    head = _capture(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    current_commit = str(head.get("stdout") or "").strip()
    bundle_source_commit = str(bundle.get("source_commit") or "")
    if head.get("returncode") != 0 or not bundle_source_commit:
        raise ValueError("production correctness bundle source_commit is unavailable")
    ancestor = _capture(
        ["git", "merge-base", "--is-ancestor", bundle_source_commit, current_commit],
        cwd=REPO_ROOT,
    )
    runtime_diff = _capture(
        [
            "git",
            "diff",
            "--quiet",
            f"{bundle_source_commit}..{current_commit}",
            "--",
            *_CORRECTNESS_RUNTIME_PATHS,
        ],
        cwd=REPO_ROOT,
    )
    if ancestor.get("returncode") != 0 or runtime_diff.get("returncode") != 0:
        raise ValueError(
            "production correctness bundle source_commit is not a runtime-equivalent ancestor"
        )
    host = bundle.get("host")
    if not isinstance(host, Mapping) or host.get("physical_host") != socket.gethostname():
        raise ValueError("production correctness bundle physical host does not match")

    configuration = bundle.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("production correctness bundle configuration is missing")
    model_identity = collect_model_identity(args.model)
    model_fingerprint = model_identity.get("fingerprint")
    if not isinstance(model_fingerprint, Mapping):
        raise ValueError("model fingerprint is unavailable")
    expected_fingerprint = configuration.get("model_fingerprint")
    if not isinstance(expected_fingerprint, Mapping) or any(
        expected_fingerprint.get(key) != model_fingerprint.get(key)
        for key in ("algorithm", "value", "size_bytes")
    ):
        raise ValueError("production correctness bundle model fingerprint does not match")
    expected_configuration = {
        "backend": str(args.backend),
        "quant": str(args.quant),
        "kv_storage": str(args.hipengine_kv_storage),
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(f"production correctness bundle {key} does not match")
    candidate_environment = configuration.get("candidate_environment")
    if not isinstance(candidate_environment, Mapping) or candidate_environment.get(
        "HIPENGINE_GGUF_FP16_RECURRENT_STATE"
    ) != "1":
        raise ValueError("production correctness bundle does not identify FP16 state")

    gates = bundle.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("production correctness bundle gates are missing")
    required_gates = (
        "numerical",
        "repeat_determinism",
        "isolation",
        "control_ownership",
        "lifecycle",
        "bf16_relative",
        "task_quality",
        "strict_fallback",
    )
    for name in required_gates:
        gate = gates.get(name)
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            raise ValueError(f"production correctness gate failed or missing: {name}")
    if gates["repeat_determinism"].get("runs", 0) < 3:
        raise ValueError("production repeat_determinism requires at least three runs")
    if gates["strict_fallback"].get("registered") is not True:
        raise ValueError("production strict_fallback must remain registered")

    numerical = gates["numerical"]
    summary = numerical.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("production numerical summary is missing")
    for metric, limit in _PRODUCTION_NUMERICAL_LIMITS.items():
        value = _number(summary.get(metric))
        if value is None:
            raise ValueError(f"production numerical metric is missing: {metric}")
        if metric == "top1_agreement":
            passed = value >= limit
        else:
            passed = value <= limit
        if not passed:
            raise ValueError(f"production numerical metric failed: {metric}")
    if list(numerical.get("scope_failures") or ()):
        raise ValueError("production numerical scope failures are present")
    if numerical.get("requires_outlier_review") is True:
        review = numerical.get("manual_review")
        if not isinstance(review, Mapping) or review.get("passed") is not True:
            raise ValueError("production numerical outlier review is missing or failed")

    source_artifacts = bundle.get("source_artifacts")
    if not isinstance(source_artifacts, Sequence) or isinstance(
        source_artifacts, (str, bytes, bytearray)
    ) or not source_artifacts:
        raise ValueError("production correctness source_artifacts are missing")
    verified_sources: list[dict[str, str]] = []
    for index, source in enumerate(source_artifacts):
        if not isinstance(source, Mapping):
            raise ValueError(f"production correctness source_artifacts[{index}] is invalid")
        requested_source = Path(str(source.get("path") or "")).expanduser()
        if requested_source.is_symlink():
            raise ValueError(
                f"production correctness source artifact must not be a symlink: {requested_source}"
            )
        source_path = requested_source.resolve()
        if not source_path.is_file():
            raise ValueError(f"production correctness source artifact is unavailable: {source_path}")
        observed_sha256 = file_sha256(source_path)
        if source.get("sha256") != observed_sha256:
            raise ValueError(f"production correctness source artifact hash mismatch: {source_path}")
        verified_sources.append({"path": str(source_path), "sha256": observed_sha256})

    return {
        "profile": "production",
        "arithmetic_binding": (
            "external same-model production numerical/task bundle plus exact "
            "serving control and schedule-local determinism"
        ),
        "generated_id_equality_binding": False,
        "public_profile_qualification_claim": bool(public_claim),
        "runtime_scope": runtime_scope,
        "runtime_execution_profile": effective_environment.get(
            "HIPENGINE_EXECUTION_PROFILE"
        ),
        "bundle": str(resolved),
        "bundle_sha256": file_sha256(resolved),
        "source_commit": bundle_source_commit,
        "current_commit": current_commit,
        "model_fingerprint": dict(model_fingerprint),
        "numerical_summary": {
            key: float(summary[key]) for key in _PRODUCTION_NUMERICAL_LIMITS
        },
        "gates": {name: True for name in required_gates},
        "source_artifacts": verified_sources,
    }


def _wait_ready(base_url: str, process: subprocess.Popen[str], log_path: Path, timeout: float) -> float:
    started = time.perf_counter()
    deadline = started + float(timeout)
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:] if log_path.exists() else ""
            raise BenchError(f"server exited with {process.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return time.perf_counter() - started
        except Exception:
            pass
        time.sleep(0.25)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:] if log_path.exists() else ""
    raise BenchError(f"server did not become ready within {timeout}s\n{tail}")


def _stop_server(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30.0)
    log_handle = getattr(process, "_f1_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def _start_server(
    args: argparse.Namespace,
    *,
    engine: str,
    concurrency: int,
    label: str,
) -> tuple[subprocess.Popen[str], str, list[str], Path, float]:
    port = int(args.port_base) + int(concurrency)
    base_url = f"http://{args.host}:{port}"
    command, env, cwd = _server_command_and_env(
        args,
        engine=engine,
        concurrency=concurrency,
        port=port,
    )
    log_path = args.work_dir / f"{engine}-{label}.server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process._f1_log_handle = log_handle  # type: ignore[attr-defined]
    try:
        startup_seconds = _wait_ready(base_url, process, log_path, float(args.server_ready_timeout))
    except Exception:
        _stop_server(process)
        raise
    return process, base_url, command, log_path, startup_seconds


def _memory_sampler(args: argparse.Namespace) -> VramSampler | None:
    if args.no_memory_sampling:
        return None
    card = select_card(index=int(args.drm_card_index))
    return VramSampler(
        card,
        interval_ms=float(args.memory_poll_ms),
        keep_samples=False,
        memory_domain=str(args.memory_domain),
    )


def _server_log_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    payload = _full_file_fingerprint(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    payload["tail"] = text[-4000:]
    return payload


def _process_env_report_record(args: argparse.Namespace, *, port: int) -> dict[str, Any]:
    path = _process_env_report_path(args, port=port)
    if not path.is_file():
        return {
            "path": str(path),
            "missing": True,
            "runtime_queue_ids": None,
            "runtime_queue_count": None,
            "runtime_observation": "requires rocprof queue trace",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise BenchError(f"process environment report root must be an object: {path}")
    queue = payload.get("gpu_max_hw_queues")
    if not isinstance(queue, Mapping):
        raise BenchError(f"process environment report queue record is missing: {path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "missing": False,
        "configured": dict(queue),
        "runtime_queue_ids": queue.get("runtime_queue_ids"),
        "runtime_queue_count": queue.get("runtime_queue_count"),
        "runtime_observation": queue.get("runtime_observation"),
    }


def _selected_metrics(samples: Sequence[Mapping[str, Any]], raw_path: Path) -> dict[str, Any]:
    scalar_names = (
        "hipengine_requests_total",
        "hipengine_request_completed_total",
        "hipengine_request_failed_total",
        "hipengine_request_rejected_total",
        "hipengine_request_cancelled_total",
        "hipengine_prompt_tokens_total",
        "hipengine_completion_tokens_total",
        "hipengine_resident_requests_admitted_total",
        "hipengine_resident_requests_reclaimed_total",
        "hipengine_resident_bucket_capacity",
        "hipengine_resident_bucket_active_rows",
        "hipengine_resident_bucket_occupied_slots",
        "hipengine_resident_bucket_free_slots",
        "hipengine_resident_work_prefill_total",
        "hipengine_resident_work_decode_total",
        "hipengine_resident_work_reclaim_total",
        "hipengine_resident_packed_workspace_current_bytes",
        "hipengine_resident_packed_workspace_release_events_total",
        "hipengine_resident_packed_workspace_released_bytes_total",
        "hipengine_resident_kv_int8_payload_bytes",
        "hipengine_resident_kv_bf16_payload_bytes",
        "hipengine_resident_kv_scale_bytes",
        "hipengine_resident_kv_bf16_mirror_bytes",
        "hipengine_resident_kv_total_bytes",
        "hipengine_kv_pool_current_bytes",
        "hipengine_kv_pool_high_water_observed_bytes",
        "hipengine_kv_pool_current_pages",
        "hipengine_kv_pool_high_water_observed_pages",
        "hipengine_kv_pool_refcounted_pages",
        "hipengine_kv_pool_pinned_pages",
        "hipengine_graph_bucket_captures_total",
        "hipengine_graph_bucket_replays_total",
        "hipengine_graph_bucket_invalidations_total",
    )
    bucket = prometheus_sample(samples, "hipengine_resident_bucket_info")
    return {
        "raw": _full_file_fingerprint(raw_path),
        "scalars": {name: prometheus_value(samples, name) for name in scalar_names},
        "latency_seconds": hipengine_latency_snapshot(samples),
        "bucket_info": None if bucket is None else dict(bucket),
        "routes": [dict(sample) for sample in samples if sample.get("name") == "hipengine_resident_route_total"],
        "fallbacks": [
            dict(sample) for sample in samples if sample.get("name") == "hipengine_resident_fallback_total"
        ],
        "route_manifest": next(
            (
                dict(sample)
                for sample in samples
                if sample.get("name") == "hipengine_resident_route_manifest_info"
            ),
            None,
        ),
        "graph_buckets": [
            dict(sample)
            for sample in samples
            if str(sample.get("name") or "").startswith("hipengine_graph_bucket_")
            and bool((sample.get("labels") or {}).get("bucket"))
        ],
    }


def _request_oracle_rows(
    args: argparse.Namespace,
    *,
    engine: str,
    base_url: str,
    prompts: Sequence[Sequence[int]],
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    oracle: dict[str, list[int]] = {}
    for index, prompt in enumerate(prompts):
        release = threading.Event()
        epoch = time.perf_counter()
        release.set()
        record = _one_request(
            args,
            engine=engine,
            base_url=base_url,
            prompt=prompt,
            index=index,
            release=release,
            epoch=epoch,
        )
        if len(record["generated_token_ids"]) != int(args.decode_tokens):
            raise BenchError(
                f"oracle row {index} returned {len(record['generated_token_ids'])} tokens; "
                f"expected {args.decode_tokens}"
            )
        prompt_hash = str(record["prompt_token_ids_sha256"])
        oracle[prompt_hash] = list(record["generated_token_ids"])
        records.append(record)
    return oracle, records


def _run_oracle(
    args: argparse.Namespace,
    *,
    engine: str,
    prompts: Sequence[Sequence[int]],
) -> tuple[dict[str, list[int]], list[dict[str, Any]], dict[str, Any]]:
    sampler = _memory_sampler(args)
    process: subprocess.Popen[str] | None = None
    server_metadata: dict[str, Any] = {}
    if sampler is not None:
        sampler.start()
    try:
        process, base_url, command, log_path, startup_seconds = _start_server(
            args,
            engine=engine,
            concurrency=1,
            label="oracle-c1",
        )
        oracle, records = _request_oracle_rows(
            args,
            engine=engine,
            base_url=base_url,
            prompts=prompts,
        )
        server_metadata.update(
            {
                "server_command": command,
                "server_command_shell": shlex.join(command),
                "server_startup_seconds": startup_seconds,
                "server_log": _server_log_record(log_path),
                "runtime_queue_observation": (
                    _process_env_report_record(
                        args,
                        port=int(args.port_base) + 1,
                    )
                    if engine == "hipengine"
                    else None
                ),
            }
        )
        return oracle, records, server_metadata
    finally:
        if (
            sampler is not None
            and process is not None
            and bool(args.memory_sample_through_shutdown)
        ):
            _stop_server(process)
            process = None
            time.sleep(max(0.05, 2.0 * float(args.memory_poll_ms) / 1000.0))
        if sampler is not None:
            sampler.stop()
            server_metadata["memory"] = sampler.result().to_dict()
        if process is not None:
            _stop_server(process)


def _run_width(
    args: argparse.Namespace,
    *,
    engine: str,
    concurrency: int,
    oracle: Mapping[str, Sequence[int]],
    oracle_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prompts = _prompt_rows(
        rows=concurrency,
        prompt_length=int(args.prompt_length),
        token_id=int(args.prompt_token_id),
    )
    oracle_scope = "separate_c1_server"
    width_oracle_records = list(oracle_records)
    sampler = _memory_sampler(args)
    process: subprocess.Popen[str] | None = None
    if sampler is not None:
        sampler.start()
    memory: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    try:
        process, base_url, command, log_path, startup_seconds = _start_server(
            args,
            engine=engine,
            concurrency=concurrency,
            label=f"c{concurrency}",
        )
        if bool(args.same_server_oracle):
            oracle, width_oracle_records = _request_oracle_rows(
                args,
                engine=engine,
                base_url=base_url,
                prompts=_prompt_rows(
                    rows=int(args.oracle_rows),
                    prompt_length=int(args.prompt_length),
                    token_id=int(args.prompt_token_id),
                ),
            )
            oracle_scope = "same_loaded_server_serial_c1"
        oracle_text = {
            str(record.get("prompt_token_ids_sha256") or ""): str(record.get("text") or "")
            for record in width_oracle_records
        }
        warmups = [
            _run_burst(args, engine=engine, base_url=base_url, prompts=prompts)
            for _ in range(int(args.warmup_runs))
        ]
        measured: list[dict[str, Any]] = []
        for repetition in range(int(args.measured_runs)):
            sample = _run_burst(args, engine=engine, base_url=base_url, prompts=prompts)
            sample["repetition"] = repetition
            measured.append(sample)
            print(
                f"{engine} c{concurrency} rep{repetition}: "
                f"http={sample['http_wall_tok_s_aggregate']:.3f} tok/s "
                f"wall={sample['batch_wall_seconds']:.3f}s",
                flush=True,
            )
            time.sleep(float(args.inter_rep_seconds))
        streaming: dict[str, Any] | None = None
        if bool(args.streaming_primary):
            stream_warmups = [
                _run_stream_burst(
                    args,
                    engine=engine,
                    base_url=base_url,
                    prompts=prompts,
                    oracle=oracle,
                    oracle_text=oracle_text,
                )
                for _ in range(int(args.stream_warmup_runs))
            ]
            stream_measured: list[dict[str, Any]] = []
            for repetition in range(int(args.stream_measured_runs)):
                sample = _run_stream_burst(
                    args,
                    engine=engine,
                    base_url=base_url,
                    prompts=prompts,
                    oracle=oracle,
                    oracle_text=oracle_text,
                )
                sample["repetition"] = repetition
                stream_measured.append(sample)
                print(
                    f"{engine} c{concurrency} stream rep{repetition}: "
                    f"qualified={sample['correctness_qualified_tok_s_aggregate']:.3f} tok/s "
                    f"exact_diag={sample['exact_generated_tok_s_aggregate']:.3f} tok/s "
                    f"goodput={sample['slo_goodput_tok_s_aggregate']:.3f} tok/s "
                    f"wall={sample['batch_wall_seconds']:.3f}s",
                    flush=True,
                )
                time.sleep(float(args.inter_rep_seconds))
            streaming = {
                "warmup_runs": stream_warmups,
                "measured_runs": stream_measured,
                "summary": {
                    "correctness_qualified_tok_s_aggregate": metric_summary(
                        [
                            float(sample["correctness_qualified_tok_s_aggregate"])
                            for sample in stream_measured
                        ]
                    ),
                    "exact_generated_tok_s_aggregate": metric_summary(
                        [float(sample["exact_generated_tok_s_aggregate"]) for sample in stream_measured]
                    ),
                    "slo_goodput_tok_s_aggregate": metric_summary(
                        [float(sample["slo_goodput_tok_s_aggregate"]) for sample in stream_measured]
                    ),
                    "ttft_p95_seconds": metric_summary(
                        [
                            float(sample["latency_seconds"]["ttft"]["p95"])
                            for sample in stream_measured
                            if _is_number(sample["latency_seconds"]["ttft"]["p95"])
                        ]
                    ),
                    "itl_p99_seconds": metric_summary(
                        [
                            float(sample["latency_seconds"]["itl"]["p99"])
                            for sample in stream_measured
                            if _is_number(sample["latency_seconds"]["itl"]["p99"])
                        ]
                    ),
                    "end_to_end_p95_seconds": metric_summary(
                        [
                            float(sample["latency_seconds"]["end_to_end"]["p95"])
                            for sample in stream_measured
                            if _is_number(sample["latency_seconds"]["end_to_end"]["p95"])
                        ]
                    ),
                    "slo_passed_runs": sum(
                        1 for sample in stream_measured if sample["slo"]["passed"] is True
                    ),
                },
                "route": _stream_route_summary(
                    engine,
                    concurrency=int(concurrency),
                    samples=stream_measured,
                ),
                "passed": bool(stream_measured)
                and all(sample["passed"] is True for sample in stream_measured),
            }
        burst_metrics: dict[str, Any] | None = None
        if engine == "hipengine":
            samples, text = _metrics_state(base_url)
            metrics_path = args.work_dir / f"{engine}-c{concurrency}.burst.metrics"
            metrics_path.write_text(text, encoding="utf-8")
            burst_metrics = _selected_metrics(samples, metrics_path)
        live = None
        live_metrics = None
        if int(concurrency) == int(args.live_concurrency):
            live = _run_live_admission(
                args,
                engine=engine,
                base_url=base_url,
                prompts=prompts,
                oracle_records=width_oracle_records,
            )
            print(
                f"{engine} c{concurrency} live: http={live['http_wall_tok_s_aggregate']:.3f} tok/s "
                f"admission_during_first={live['admission_during_first_request']}",
                flush=True,
            )
            if engine == "hipengine":
                samples, text = _metrics_state(base_url)
                metrics_path = args.work_dir / f"{engine}-c{concurrency}.live.metrics"
                metrics_path.write_text(text, encoding="utf-8")
                live_metrics = _selected_metrics(samples, metrics_path)
        measured_records = [record for sample in measured for record in sample["records"]]
        measured_correctness = correctness_summary(
            measured_records,
            oracle=oracle,
            expected_tokens=int(args.decode_tokens),
            profile=str(args.correctness_profile),
        )
        repeat_determinism = repeat_determinism_summary(measured)
        warmup_records = [
            record for sample in warmups for record in sample["records"]
        ]
        warmup_correctness = (
            correctness_summary(
                warmup_records,
                oracle=oracle,
                expected_tokens=int(args.decode_tokens),
                profile=str(args.correctness_profile),
            )
            if warmup_records
            else {
                "passed": True,
                "skipped": True,
                "rows": 0,
                "exact_rows": 0,
                "mismatch_count": 0,
                "mismatches": [],
            }
        )
        live_correctness = (
            None
            if live is None
            else correctness_summary(
                live["records"],
                oracle=oracle,
                expected_tokens=int(args.decode_tokens),
                profile=str(args.correctness_profile),
            )
        )
        rates = [float(sample["http_wall_tok_s_aggregate"]) for sample in measured]
        walls = [float(sample["batch_wall_seconds"]) for sample in measured]
        request_walls = [float(record["wall_seconds"]) for record in measured_records]
        route_paths = sorted({str(record.get("execution_path")) for record in measured_records})
        serial_values = [record.get("serial_decode_fallback") for record in measured_records]
        native_values = [record.get("native_caware_decode") for record in measured_records]
        shape_evidence = None
        resident_capacity = None
        route_ok = True
        if engine == "hipengine":
            shape_runs = [
                generation_shape_proves_native_group(
                    sample["records"],
                    concurrency=int(concurrency),
                )
                for sample in measured
            ]
            shape_evidence = {
                "passed": all(bool(run["passed"]) for run in shape_runs),
                "runs": shape_runs,
            }
            resident_capacity = (
                None
                if burst_metrics is None
                else burst_metrics["scalars"].get("hipengine_resident_bucket_capacity")
            )
            route_ok = _hipengine_route_expectation_passes(
                concurrency=int(concurrency),
                expectation=str(args.hipengine_route_expectation),
                serial_values=serial_values,
                native_values=native_values,
                shape_passed=bool(shape_evidence["passed"]),
                resident_capacity=resident_capacity,
                execution_paths=route_paths,
                native_false_records_expected=sum(
                    int(run["native_false_records_expected"])
                    for run in shape_runs
                ),
            )
        result = {
            "concurrency": int(concurrency),
            "prompt_rows": {
                "count": len(prompts),
                "length": int(args.prompt_length),
                "token_ids_sha256": [token_ids_sha256(prompt) for prompt in prompts],
            },
            "server": {
                "command": command,
                "command_shell": shlex.join(command),
                "startup_seconds": startup_seconds,
                "log": _server_log_record(log_path),
                "runtime_queue_observation": (
                    _process_env_report_record(
                        args,
                        port=int(args.port_base) + int(concurrency),
                    )
                    if engine == "hipengine"
                    else None
                ),
            },
            "warmup_runs": warmups,
            "measured_runs": measured,
            "summary": {
                "http_wall_tok_s_aggregate": metric_summary(rates),
                "http_wall_tok_s_per_request": metric_summary(
                    [rate / int(concurrency) for rate in rates]
                ),
                "batch_wall_seconds": metric_summary(walls),
                "request_wall_seconds": metric_summary(request_walls),
            },
            "correctness": {
                "profile": str(args.correctness_profile),
                "generated_id_equality_binding": str(args.correctness_profile) == "strict",
                "oracle_scope": oracle_scope,
                "oracle_generated_rows": [
                    list(oracle[token_ids_sha256(prompt)])
                    for prompt in _prompt_rows(
                        rows=int(args.oracle_rows),
                        prompt_length=int(args.prompt_length),
                        token_id=int(args.prompt_token_id),
                    )
                ],
                "oracle_records": width_oracle_records,
                "warmups": warmup_correctness,
                "measured": measured_correctness,
                "repeat_determinism": repeat_determinism,
                "live_admission": live_correctness,
            },
            "execution": {
                "paths": route_paths,
                "route_ok": route_ok,
                "route_expectation": (
                    str(args.hipengine_route_expectation)
                    if engine == "hipengine"
                    else None
                ),
                "serial_decode_fallback_values": sorted(
                    {value for value in serial_values if isinstance(value, bool)}
                ),
                "native_caware_decode_values": sorted(
                    {value for value in native_values if isinstance(value, bool)}
                ),
                "generation_shape": shape_evidence,
                "resident_capacity": resident_capacity,
            },
            "burst_metrics": burst_metrics,
            "streaming": streaming,
            "live_admission": live,
            "live_metrics": live_metrics,
        }
        return result
    finally:
        if (
            sampler is not None
            and process is not None
            and bool(args.memory_sample_through_shutdown)
        ):
            _stop_server(process)
            process = None
            time.sleep(max(0.05, 2.0 * float(args.memory_poll_ms) / 1000.0))
        if sampler is not None:
            sampler.stop()
            memory = sampler.result().to_dict()
            if result is not None:
                result["memory"] = memory
        if process is not None:
            _stop_server(process)


def _hardware_capture(args: argparse.Namespace, *, engine: str) -> dict[str, Any]:
    return {
        "kernel_cmdline": Path("/proc/cmdline").read_text(encoding="utf-8").strip(),
        "uname": _capture(["uname", "-a"]),
        "rocminfo": _capture(["bash", "-lc", "rocminfo | grep -E 'Name:|gfx' | head -8"]),
        "rocm_smi": _capture(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"]),
        "hipcc": _capture(["hipcc", "--version"]),
        "tuned_profile": _capture(["tuned-adm", "active"]),
        "gpu_max_hw_queues": (
            None if args.gpu_max_hw_queues is None else int(args.gpu_max_hw_queues)
        ),
        "gpu_max_hw_queues_requested_policy": _gpu_max_hw_queues_label(
            args.gpu_max_hw_queues
        ),
        "memory_domain": str(args.memory_domain),
        "effective_server_environment": _effective_server_environment(
            args,
            engine=engine,
        ),
    }


def _source_provenance(args: argparse.Namespace, engine: str) -> dict[str, Any]:
    repo, server_bin = _server_paths(args, engine)
    payload: dict[str, Any] = {
        "hipengine": {
            "head": _capture(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
            "status": _capture(["git", "status", "-sb", "--untracked-files=no"], cwd=REPO_ROOT),
        },
        "model": _model_fingerprint(args.model),
    }
    if args.compiler_version_file is not None and args.compiler_version_file.exists():
        payload["compiler_version_file"] = _full_file_fingerprint(args.compiler_version_file)
    if engine.startswith("llamacpp") and repo is not None and server_bin is not None:
        ldd = _capture(["ldd", str(server_bin)])
        linked_paths = parse_ldd_local_paths(str(ldd.get("stdout") or ""), root=repo)
        payload["llamacpp"] = {
            "repo": str(repo),
            "head": _capture(["git", "rev-parse", "HEAD"], cwd=repo),
            "status": _capture(["git", "status", "-sb", "--untracked-files=no"], cwd=repo),
            "version": _capture([str(server_bin), "--version"], cwd=repo),
            "server_binary": _full_file_fingerprint(server_bin),
            "ldd": ldd,
            "local_linked_libraries": [_full_file_fingerprint(path) for path in linked_paths],
        }
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    engine = str(args.engine)
    concurrencies = _validate_concurrency_plan(
        args.concurrencies,
        live_concurrency=int(args.live_concurrency),
        require_c1=not bool(args.focused_width_repair),
    )
    if not args.model.exists():
        raise ValueError(f"model does not exist: {args.model}")
    correctness_contract = _resolve_correctness_contract(args, engine=engine)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    invocation = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]]
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "kind": "gfx1151_f1_matched_server_concurrency_backend",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "passed": False,
        "performance_claim": False,
        "engine": engine,
        "protocol": {
            "primary_metric": (
                "correctness-profile-qualified SLO completion tokens / barrier-to-last-SSE-completion wall"
                if bool(args.streaming_primary)
                else "C*correctness-profile-qualified completion tokens / barrier-to-last-response HTTP wall"
            ),
            "primary_metric_scope": "prefill + decode + server scheduling + localhost HTTP/SSE",
            "blocking_control_metric": "C*returned_completion_tokens / barrier-to-last-response HTTP wall",
            "native_decode_metric_scope": "engine-specific diagnostic only; not cross-engine comparable",
            "model": str(args.model.resolve()),
            "quant": str(args.quant),
            "kv_storage": str(args.hipengine_kv_storage) if engine == "hipengine" else "f16",
            "prompt_length": int(args.prompt_length),
            "decode_tokens": int(args.decode_tokens),
            "prompt_row_construction": (
                f"{args.prompt_length}-token rows filled with {args.prompt_token_id}; final token cycles "
                f"{args.prompt_token_id}..{args.prompt_token_id + 3}"
            ),
            "sampling": (
                f"temperature=0, top_k={args.hipengine_top_k if engine == 'hipengine' else 1}, "
                "top_p=1, ignore_eos=true, MTP disabled"
            ),
            "concurrencies": concurrencies,
            "gpu_max_hw_queues_requested_policy": _gpu_max_hw_queues_label(
                args.gpu_max_hw_queues
            ),
            "gpu_max_hw_queues_runtime_observation": (
                "per-server process report records configured value/source; actual "
                "runtime queue IDs/count require the cache-only rocprof queue trace"
            ),
            "warmup_runs_per_width": int(args.warmup_runs),
            "measured_runs_per_width": int(args.measured_runs),
            "streaming_primary": bool(args.streaming_primary),
            "stream_warmup_runs_per_width": int(args.stream_warmup_runs),
            "stream_measured_runs_per_width": int(args.stream_measured_runs),
            "stream_client_slo_seconds": {
                "ttft_p95": float(args.slo_ttft_p95_seconds),
                "itl_p99": float(args.slo_itl_p99_seconds),
                "end_to_end_p95": float(args.slo_end_to_end_p95_seconds),
            },
            "fresh_server_per_width": True,
            "server_capacity_matches_logical_concurrency": True,
            "context_tokens_per_sequence": int(args.ctx_per_seq),
            "hipengine_batch_window_ms": (
                float(args.batch_window_ms) if engine == "hipengine" else None
            ),
            "hipengine_generation_batch_window_ms": (
                float(args.batch_window_ms) if engine == "hipengine" else None
            ),
            "hipengine_prefill_decode_policy": (
                str(args.hipengine_prefill_decode_policy)
                if engine == "hipengine"
                else None
            ),
            "hipengine_prefill_chunk_tokens": (
                int(args.hipengine_prefill_chunk_tokens)
                if (
                    engine == "hipengine"
                    and args.hipengine_prefill_chunk_tokens is not None
                )
                else None
            ),
            "llamacpp_prompt_cache": False,
            "correctness_profile": str(args.correctness_profile),
            "generated_id_equality_binding": bool(
                correctness_contract["generated_id_equality_binding"]
            ),
            "output_contract": (
                "exact same-engine c1 generated trajectory"
                if str(args.correctness_profile) == "strict"
                else (
                    "exact request/control ownership and schedule-local determinism; "
                    "c1/cN generated-ID equality is diagnostic and arithmetic is bound "
                    "to the production correctness bundle"
                )
            ),
            "live_admission_concurrency": int(args.live_concurrency),
        },
        "command": invocation,
        "command_shell": shlex.join(invocation),
        "environment": _hardware_capture(args, engine=engine),
        "provenance": _source_provenance(args, engine),
        "correctness_contract": correctness_contract,
        "public_profile_qualification_claim": bool(
            correctness_contract["public_profile_qualification_claim"]
        ),
        "oracle": None,
        "rows": {},
        "limitations": [
            "TTFT/ITL/end-to-end curves are matched client-observed SSE timings; engine-resident timing fields are not cross-engine comparable.",
            (
                "hipEngine SSE does not expose generated token IDs: strict uses exact blocking-c1 text plus completion count; production records that equality diagnostically and binds protocol completion plus its external correctness bundle."
            ),
            "hipEngine and llama.cpp backend-native decode timings have different ownership boundaries and are diagnostic only.",
            "gfx1151 is UMA: whole-card GTT, not the 512 MiB visible-VRAM aperture, is the relevant external memory domain.",
        ],
    }
    _write_json(args.json, payload)
    oracle_prompts = _prompt_rows(
        rows=int(args.oracle_rows),
        prompt_length=int(args.prompt_length),
        token_id=int(args.prompt_token_id),
    )
    if bool(args.same_server_oracle):
        oracle: dict[str, list[int]] = {}
        oracle_records: list[dict[str, Any]] = []
        payload["oracle"] = {
            "scope": "same_loaded_server_serial_c1_per_width",
            "prompt_rows": oracle_prompts,
            "prompt_token_ids_sha256": [
                token_ids_sha256(prompt) for prompt in oracle_prompts
            ],
            "generated_rows": None,
            "generated_token_ids_sha256": None,
            "records": None,
            "server": None,
            "passed": True,
        }
    else:
        oracle, oracle_records, oracle_server = _run_oracle(
            args,
            engine=engine,
            prompts=oracle_prompts,
        )
        payload["oracle"] = {
            "scope": "separate_c1_server",
            "prompt_rows": oracle_prompts,
            "prompt_token_ids_sha256": [token_ids_sha256(prompt) for prompt in oracle_prompts],
            "generated_rows": [list(oracle[token_ids_sha256(prompt)]) for prompt in oracle_prompts],
            "generated_token_ids_sha256": [
                token_ids_sha256(oracle[token_ids_sha256(prompt)]) for prompt in oracle_prompts
            ],
            "records": oracle_records,
            "server": oracle_server,
            "passed": len(oracle) == len(oracle_prompts),
        }
    _write_json(args.json, payload)
    for concurrency in concurrencies:
        row = _run_width(
            args,
            engine=engine,
            concurrency=int(concurrency),
            oracle=oracle,
            oracle_records=oracle_records,
        )
        payload["rows"][str(concurrency)] = row
        _write_json(args.json, payload)
        time.sleep(float(args.inter_server_seconds))
    rows = list(payload["rows"].values())
    passed = bool(payload["oracle"]["passed"]) and all(
        row["correctness"]["warmups"]["passed"]
        and row["correctness"]["measured"]["passed"]
        and row["correctness"]["repeat_determinism"]["passed"]
        and (row["correctness"]["live_admission"] is None or row["correctness"]["live_admission"]["passed"])
        and row["execution"]["route_ok"]
        and (
            row["streaming"] is None
            or (
                row["streaming"]["passed"] is True
                and row["streaming"]["route"]["passed"] is True
            )
        )
        for row in rows
    )
    live_row = payload["rows"][str(args.live_concurrency)]
    passed = passed and _live_admission_passes(
        engine,
        args,
        live_row["live_admission"],
    )
    payload["passed"] = passed
    payload["performance_claim"] = passed
    payload["status"] = "accepted_backend_packet" if passed else "failed_gate"
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.json, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINE_CHOICES, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--correctness-profile",
        choices=_CORRECTNESS_PROFILES,
        default="strict",
        help=(
            "Serving evidence contract only; this does not select runtime dispatch. "
            "Strict binds same-engine c1 generated equality. Production requires a "
            "matching complete correctness bundle and records c1/cN equality diagnostically."
        ),
    )
    parser.add_argument(
        "--production-correctness-artifact",
        type=Path,
        help="matching fail-closed production numerical/task/control bundle",
    )
    parser.add_argument("--served-model-name", default="qwen36-35b-q4km")
    parser.add_argument("--hipengine-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--hipengine-route-expectation",
        choices=("native", "serial", "serial-c1-per-row", "scheduler-c1"),
        default="native",
        help=(
            "Expected hipEngine model route; serial-c1-per-row is Qwen GGUF "
            "resident ownership with exact serial physical-c1 model transitions, "
            "while scheduler-c1 is the corresponding Laguna route"
        ),
    )
    parser.add_argument(
        "--hipengine-top-k",
        type=int,
        choices=(0, 1),
        default=1,
        help="hipEngine request top_k; Laguna exact greedy serving requires 0",
    )
    parser.add_argument(
        "--hipengine-prefill-decode-policy",
        choices=("protect_decode", "protect_ttft", "fair", "token_budget"),
        default="protect_ttft",
        help="Explicit hipEngine resident scheduling policy; retained F1 uses protect_ttft",
    )
    parser.add_argument(
        "--hipengine-kv-storage",
        choices=("auto", "bf16", "int8_per_token_head", "tail4_hadamard_group32"),
        default="bf16",
        help="hipEngine server KV policy; non-hipEngine packets continue to use f16",
    )
    parser.add_argument(
        "--hipengine-kv-scale-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
    )
    parser.add_argument(
        "--hipengine-kv-scale-granularity",
        choices=("per_token_head", "hadamard_group32"),
        default="per_token_head",
    )
    parser.add_argument("--llamacpp-hip-repo", type=Path, default=DEFAULT_LLAMA_HIP_REPO)
    parser.add_argument(
        "--llamacpp-hip-server-bin",
        type=Path,
        default=DEFAULT_LLAMA_HIP_REPO / "build/bin/llama-server",
    )
    parser.add_argument("--llamacpp-vulkan-repo", type=Path, default=DEFAULT_LLAMA_VULKAN_REPO)
    parser.add_argument(
        "--llamacpp-vulkan-server-bin",
        type=Path,
        default=DEFAULT_LLAMA_VULKAN_REPO / "build/bin/llama-server",
    )
    parser.add_argument("--vk-driver-files", default=DEFAULT_VULKAN_ICD)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--gpu-max-hw-queues",
        type=_parse_gpu_max_hw_queues,
        default=1,
        metavar="{1,2,4,8,unset}",
        help=(
            "Explicit ROCm queue limit, or unset to suppress both the harness "
            "value and gfx1151 backend package default for the ROCm runtime default"
        ),
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--concurrencies", type=_parse_concurrencies, default=[1, 2, 4, 8, 13])
    parser.add_argument(
        "--focused-width-repair",
        action="store_true",
        help=(
            "Permit a width list without c1 after an earlier broad packet established "
            "the independent oracle/c1 result; the oracle server still runs"
        ),
    )
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--oracle-rows", type=int, default=4)
    parser.add_argument(
        "--same-server-oracle",
        action="store_true",
        help=(
            "Generate independent serial c1 trajectories in each width's loaded "
            "server before its burst; use when process-to-process model startup "
            "arithmetic is not deterministic"
        ),
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--streaming-primary", action="store_true")
    parser.add_argument("--stream-warmup-runs", type=int, default=0)
    parser.add_argument("--stream-measured-runs", type=int, default=3)
    parser.add_argument("--slo-ttft-p95-seconds", type=float, default=10.0)
    parser.add_argument("--slo-itl-p99-seconds", type=float, default=0.5)
    parser.add_argument("--slo-end-to-end-p95-seconds", type=float, default=30.0)
    parser.add_argument(
        "--generation-batch-window-ms",
        "--batch-window-ms",
        dest="batch_window_ms",
        type=float,
        default=5.0,
        help=(
            "Milliseconds to coalesce compatible HTTP requests before generation; "
            "this is independent of the resident prefill chunk size"
        ),
    )
    parser.add_argument(
        "--hipengine-prefill-chunk-tokens",
        type=_positive_int,
        default=None,
        help=(
            "Explicit HIPENGINE_MAX_PREFILL_CHUNK_TOKENS override; unset uses the "
            "registry/package policy"
        ),
    )
    parser.add_argument("--ctx-per-seq", type=int, default=1024)
    parser.add_argument("--live-concurrency", type=int, default=13)
    parser.add_argument("--live-join-after-tokens", type=int, default=8)
    parser.add_argument("--metrics-poll-ms", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-gpu-layers", default="99")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-base", type=int, default=19100)
    parser.add_argument("--server-log-level", default="warning")
    parser.add_argument("--server-ready-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--inter-rep-seconds", type=float, default=0.5)
    parser.add_argument("--inter-server-seconds", type=float, default=2.0)
    parser.add_argument("--memory-domain", choices=("vram", "gtt"), default="gtt")
    parser.add_argument("--drm-card-index", type=int, default=0)
    parser.add_argument("--memory-poll-ms", type=float, default=10.0)
    parser.add_argument(
        "--memory-sample-through-shutdown",
        action="store_true",
        help="Keep the UMA sampler active through child-server teardown for lifecycle evidence",
    )
    parser.add_argument("--no-memory-sampling", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/hipengine-gfx1151-f1-server"))
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except Exception as exc:
        failure = {
            "schema": SCHEMA_VERSION,
            "kind": "gfx1151_f1_matched_server_concurrency_backend",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed_exception",
            "passed": False,
            "performance_claim": False,
            "engine": str(args.engine),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        if args.json.exists():
            try:
                existing = json.loads(args.json.read_text(encoding="utf-8"))
            except Exception:
                existing = None
            if isinstance(existing, dict):
                existing.update(failure)
                failure = existing
        _write_json(args.json, failure)
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        f"FINAL {payload['engine']}: status={payload['status']} passed={payload['passed']} artifact={args.json}",
        flush=True,
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
