#!/usr/bin/env python3
"""Matched gfx1151 F1 HTTP concurrency benchmark for hipEngine and llama.cpp.

The cross-engine primary metric is deliberately limited to the timing boundary
all servers expose: exact returned completion tokens divided by client wall from
simultaneous release through the last completed response.  Backend-native decode
timings are retained as diagnostics and are never substituted for that wall.

Each engine first generates independent c1 token-ID oracles for the four prompt
rows used by the c1/c2/c4/c8 sweep.  Every warmup, measured burst, and live-
admission row must return exactly the c1 trajectory for its prompt.  hipEngine's
resident TTFT/ITL summaries and route/fallback counters are scraped separately;
llama.cpp does not expose equivalent non-streaming percentile summaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import file_sha256, token_ids_sha256  # noqa: E402
from hipengine.util.amdgpu_vram import VramSampler, select_card  # noqa: E402

SCHEMA_VERSION = 1
ENGINE_CHOICES = ("hipengine", "llamacpp-hip", "llamacpp-vulkan")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_LLAMA_HIP_REPO = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_LLAMA_VULKAN_REPO = Path("/home/lhl/llama.cpp/llama.cpp-vulkan")
DEFAULT_VULKAN_ICD = "/usr/share/vulkan/icd.d/radeon_icd.json"
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


def generation_shape_proves_native_group(
    records: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
) -> dict[str, Any]:
    expected = int(concurrency)
    group_ids: list[str] = []
    queue_counts: list[int | None] = []
    queue_prompt_rows: list[int | None] = []
    backend_input_rows: list[int | None] = []
    backend_actual_rows: list[list[int]] = []
    backend_max_rows: list[int | None] = []
    row_passes: list[bool] = []
    for record in records:
        shape = record.get("generation_shape")
        shape = shape if isinstance(shape, Mapping) else {}
        queue = shape.get("queue_group")
        queue = queue if isinstance(queue, Mapping) else {}
        groups = shape.get("backend_groups")
        groups = (
            list(groups)
            if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes, bytearray))
            else []
        )
        backend = groups[0] if len(groups) == 1 and isinstance(groups[0], Mapping) else {}
        queue_count = int(queue["request_count"]) if _is_number(queue.get("request_count")) else None
        prompt_rows = int(queue["prompt_rows"]) if _is_number(queue.get("prompt_rows")) else None
        input_rows = int(backend["input_rows"]) if _is_number(backend.get("input_rows")) else None
        max_rows = (
            int(backend["max_actual_group_rows"])
            if _is_number(backend.get("max_actual_group_rows"))
            else None
        )
        raw_actual = backend.get("actual_group_rows")
        actual = (
            [int(value) for value in raw_actual if _is_number(value)]
            if isinstance(raw_actual, Sequence) and not isinstance(raw_actual, (str, bytes, bytearray))
            else []
        )
        group_ids.append(str(queue.get("id") or ""))
        queue_counts.append(queue_count)
        queue_prompt_rows.append(prompt_rows)
        backend_input_rows.append(input_rows)
        backend_actual_rows.append(actual)
        backend_max_rows.append(max_rows)
        row_passes.append(
            queue_count == expected
            and prompt_rows == expected
            and input_rows == expected
            and actual == [expected]
            and max_rows == expected
        )
    nonempty_group_ids = [group_id for group_id in group_ids if group_id]
    shared_group = len(nonempty_group_ids) == len(records) and len(set(nonempty_group_ids)) == 1
    return {
        "passed": len(records) == expected and all(row_passes) and shared_group,
        "expected_rows": expected,
        "record_count": len(records),
        "shared_queue_group": shared_group,
        "queue_group_ids": group_ids,
        "queue_request_counts": queue_counts,
        "queue_prompt_rows": queue_prompt_rows,
        "backend_input_rows": backend_input_rows,
        "backend_actual_group_rows": backend_actual_rows,
        "backend_max_actual_group_rows": backend_max_rows,
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
            "min": None,
            "max": None,
            "stdev": None,
            "stdev_pct_of_median": None,
        }
    ordered = sorted(samples)
    median = float(statistics.median(samples))
    stdev = float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "samples": samples,
        "median": median,
        "p95": float(p95),
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
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    exact_rows = 0
    for index, record in enumerate(records):
        prompt_hash = str(record.get("prompt_token_ids_sha256") or "")
        generated = [int(token) for token in record.get("generated_token_ids") or ()]
        expected = oracle.get(prompt_hash)
        first = None if expected is None else _first_mismatch(generated, expected)
        if expected is not None and len(generated) == int(expected_tokens) and first is None:
            exact_rows += 1
            continue
        mismatches.append(
            {
                "record_index": index,
                "prompt_token_ids_sha256": prompt_hash,
                "expected_oracle_present": expected is not None,
                "expected_tokens": int(expected_tokens),
                "observed_tokens": len(generated),
                "first_mismatch_index": first,
                "expected_token": (
                    int(expected[first])
                    if expected is not None and first is not None and first < len(expected)
                    else None
                ),
                "observed_token": int(generated[first]) if first is not None and first < len(generated) else None,
            }
        )
    return {
        "passed": bool(records) and not mismatches,
        "rows": len(records),
        "exact_rows": exact_rows,
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
            "top_k": 1,
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
        "active_mask": labels.get("active_mask"),
        "last_work_kind": labels.get("last_work_kind"),
    }


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
            continue
        delays.append((prompt_ms + int(join_after_tokens) * decode_ms / decode_steps) / 1000.0)
    if not delays:
        raise BenchError("oracle records do not expose enough timing to schedule live admission")
    return max(0.001, float(statistics.median(delays)))


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(
                _one_request,
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
            last_signature: tuple[Any, ...] | None = None
            while time.perf_counter() < planned_join_at and not futures[0].done():
                remaining = max(0.001, planned_join_at - time.perf_counter())
                try:
                    samples, _ = _metrics_state(
                        base_url,
                        timeout=min(poll_seconds, remaining),
                    )
                except Exception:
                    continue
                state = _compact_poll_state(samples, at_seconds=time.perf_counter() - epoch[0])
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
        records = [future.result() for future in futures]
    wall = max(float(record["completed_offset_seconds"]) for record in records)
    result = _batch_summary(records, batch_wall_seconds=wall)
    result.update(
        {
            "strategy": strategy,
            "oracle_timing_fallback_seconds": fallback_delay,
            "join_after_decode_tokens_target": int(args.live_join_after_tokens),
            "join_trigger": trigger,
            "observed_decode_trigger": trigger.get("source") == "observed_resident_decode",
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


def _server_command_and_env(
    args: argparse.Namespace,
    *,
    engine: str,
    concurrency: int,
    port: int,
) -> tuple[list[str], dict[str, str], Path]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GPU_MAX_HW_QUEUES"] = str(args.gpu_max_hw_queues)
    if args.compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if engine == "hipengine":
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
            "bf16",
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
        "graph_buckets": [
            dict(sample)
            for sample in samples
            if str(sample.get("name") or "").startswith("hipengine_graph_bucket_")
            and bool((sample.get("labels") or {}).get("bucket"))
        ],
    }


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
        server_metadata.update(
            {
                "server_command": command,
                "server_command_shell": shlex.join(command),
                "server_startup_seconds": startup_seconds,
                "server_log": _server_log_record(log_path),
            }
        )
        return oracle, records, server_metadata
    finally:
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
                oracle_records=oracle_records,
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
        )
        warmup_correctness = correctness_summary(
            [record for sample in warmups for record in sample["records"]],
            oracle=oracle,
            expected_tokens=int(args.decode_tokens),
        )
        live_correctness = (
            None
            if live is None
            else correctness_summary(
                live["records"],
                oracle=oracle,
                expected_tokens=int(args.decode_tokens),
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
            route_ok = (
                all(value is False for value in serial_values)
                and (int(concurrency) == 1 or all(value is True for value in native_values))
                and bool(shape_evidence["passed"])
                and resident_capacity == float(concurrency)
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
                "warmups": warmup_correctness,
                "measured": measured_correctness,
                "live_admission": live_correctness,
            },
            "execution": {
                "paths": route_paths,
                "route_ok": route_ok,
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
            "live_admission": live,
            "live_metrics": live_metrics,
        }
        return result
    finally:
        if sampler is not None:
            sampler.stop()
            memory = sampler.result().to_dict()
            if result is not None:
                result["memory"] = memory
        if process is not None:
            _stop_server(process)


def _hardware_capture(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "kernel_cmdline": Path("/proc/cmdline").read_text(encoding="utf-8").strip(),
        "uname": _capture(["uname", "-a"]),
        "rocminfo": _capture(["bash", "-lc", "rocminfo | grep -E 'Name:|gfx' | head -8"]),
        "rocm_smi": _capture(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"]),
        "hipcc": _capture(["hipcc", "--version"]),
        "tuned_profile": _capture(["tuned-adm", "active"]),
        "gpu_max_hw_queues": int(args.gpu_max_hw_queues),
        "memory_domain": str(args.memory_domain),
    }


def _source_provenance(args: argparse.Namespace, engine: str) -> dict[str, Any]:
    repo, server_bin = _server_paths(args, engine)
    payload: dict[str, Any] = {
        "hipengine": {
            "head": _capture(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
            "status": _capture(["git", "status", "-sb", "--untracked-files=no"], cwd=REPO_ROOT),
        },
        "model": _sampled_file_fingerprint(args.model),
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
    concurrencies = list(args.concurrencies)
    if 1 not in concurrencies:
        raise ValueError("concurrencies must include c1")
    if int(args.live_concurrency) not in concurrencies:
        raise ValueError("live-concurrency must appear in concurrencies")
    if max(concurrencies) > 8:
        raise ValueError("the F1 native packet is limited to c1/c2/c4/c8")
    if not args.model.is_file():
        raise ValueError(f"model does not exist: {args.model}")
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
            "primary_metric": "C*returned_completion_tokens / barrier_to_last_response_http_wall_seconds",
            "primary_metric_scope": "prefill + decode + server scheduling + localhost HTTP",
            "native_decode_metric_scope": "engine-specific diagnostic only; not cross-engine comparable",
            "model": str(args.model.resolve()),
            "quant": "GGUF Q4_K_M",
            "kv_storage": "bf16" if engine == "hipengine" else "f16",
            "prompt_length": int(args.prompt_length),
            "decode_tokens": int(args.decode_tokens),
            "prompt_row_construction": (
                f"{args.prompt_length}-token rows filled with {args.prompt_token_id}; final token cycles "
                f"{args.prompt_token_id}..{args.prompt_token_id + 3}"
            ),
            "sampling": "temperature=0, top_k=1, top_p=1, ignore_eos=true, MTP disabled",
            "concurrencies": concurrencies,
            "warmup_runs_per_width": int(args.warmup_runs),
            "measured_runs_per_width": int(args.measured_runs),
            "fresh_server_per_width": True,
            "server_capacity_matches_logical_concurrency": True,
            "context_tokens_per_sequence": int(args.ctx_per_seq),
            "hipengine_batch_window_ms": float(args.batch_window_ms),
            "llamacpp_prompt_cache": False,
            "exact_output_contract": "every server row equals an independent same-engine c1 token-ID oracle",
            "live_admission_concurrency": int(args.live_concurrency),
        },
        "command": invocation,
        "command_shell": shlex.join(invocation),
        "environment": _hardware_capture(args),
        "provenance": _source_provenance(args, engine),
        "oracle": None,
        "rows": {},
        "limitations": [
            "llama.cpp does not expose matched non-streaming resident TTFT/ITL percentile summaries; those fields remain unavailable.",
            "hipEngine and llama.cpp backend-native decode timings have different ownership boundaries and are diagnostic only.",
            "gfx1151 is UMA: whole-card GTT, not the 512 MiB visible VRAM aperture, is the relevant external memory domain.",
        ],
    }
    _write_json(args.json, payload)
    oracle_prompts = _prompt_rows(
        rows=int(args.oracle_rows),
        prompt_length=int(args.prompt_length),
        token_id=int(args.prompt_token_id),
    )
    oracle, oracle_records, oracle_server = _run_oracle(
        args,
        engine=engine,
        prompts=oracle_prompts,
    )
    payload["oracle"] = {
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
        and (row["correctness"]["live_admission"] is None or row["correctness"]["live_admission"]["passed"])
        and row["execution"]["route_ok"]
        for row in rows
    )
    live_row = payload["rows"][str(args.live_concurrency)]
    passed = passed and bool(live_row["live_admission"]["admission_during_first_request"])
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
    parser.add_argument("--served-model-name", default="qwen36-35b-q4km")
    parser.add_argument("--hipengine-python", type=Path, default=Path(sys.executable))
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
    parser.add_argument("--gpu-max-hw-queues", type=int, default=1)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--concurrencies", type=_parse_concurrencies, default=[1, 2, 4, 8])
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--oracle-rows", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--batch-window-ms", type=float, default=5.0)
    parser.add_argument("--ctx-per-seq", type=int, default=1024)
    parser.add_argument("--live-concurrency", type=int, default=8)
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
