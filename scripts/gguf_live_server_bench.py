#!/usr/bin/env python3
"""Measure real GGUF OpenAI streaming burst and live-admission concurrency.

This harness owns one prepared GGUF model and the real FastAPI generation
batcher.  It sends concurrent ``/v1/completions`` SSE requests whose text
round-trips to frozen raw token-ID prompts, records authoritative generated IDs
at resident-runner reclaim, and slices scheduler latency/occupancy/route metrics
per sample.  Packed eager execution is reported honestly as ``exact_hybrid``;
the same-loop packed-off route is the serial bridge.  Fully native graph evidence
is joined separately from ``scripts/gguf_packed_ar_bench.py`` and the profiler.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence

from fastapi.testclient import TestClient

from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.server import ServerConfig, create_app


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_PROVENANCE_ENV_KEYS = (
    "HIPENGINE_BACKEND",
    "HIPENGINE_HIP_ARCH",
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_MAX_HW_QUEUES",
)
_EXACT_ENV = {
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
    "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
    "HIPENGINE_PREFILL_DECODE_POLICY": "protect_ttft",
    "HIPENGINE_GGUF_AR_STREAM_DECODE": "0",
}


@dataclass(frozen=True)
class ServerConfiguration:
    name: str
    logical_rows: int
    packed_decode: bool
    execution_class: str


CONFIGURATIONS: dict[str, ServerConfiguration] = {
    "c1": ServerConfiguration("c1", 1, True, "occupancy_adaptive_c1"),
    "packed_c8": ServerConfiguration("packed_c8", 8, True, "exact_hybrid"),
    "packed_c9": ServerConfiguration("packed_c9", 9, True, "grouped_exact_hybrid"),
    "packed_c13": ServerConfiguration("packed_c13", 13, True, "grouped_exact_hybrid"),
    "serial_c13": ServerConfiguration("serial_c13", 13, False, "serial_bridge"),
}
_CANONICAL_CONFIGURATIONS = tuple(CONFIGURATIONS)
_SUPPORTED_BACKENDS = ("hip_gfx1100", "hip_gfx1151")
_NATIVE_EXECUTION_PATHS = frozenset(
    {"packed_native", "native_c1", "native_c1_eager", "native_c1_graph"}
)


def _artifact_backend_scope(resolved_backend: str, target_arch: str) -> str:
    backend_scope = str(resolved_backend).removeprefix("hip_")
    arch_scope = str(target_arch).strip()
    if backend_scope not in {"gfx1100", "gfx1151"} or arch_scope != backend_scope:
        raise RuntimeError(
            "live-server artifact backend/target mismatch: "
            f"backend={resolved_backend!r}, target_arch={target_arch!r}"
        )
    return arch_scope


@dataclass
class _ReclaimedRow:
    request_id: int
    prompt_ids: list[int]
    generated_ids: list[int]
    finish_reason: str
    finish_details: dict[str, Any]
    observability: dict[str, Any]
    block_ids: list[int]
    completion_time: float


@dataclass(frozen=True)
class _ReferenceRun:
    generated_tokens: list[int]
    prefill_seconds: float
    decode_step_seconds: list[float]


@dataclass
class _HTTPTrace:
    row_index: int
    status_code: int
    request_id: int | None
    started_at: float
    first_delta_at: float | None
    completed_at: float
    delta_times: list[float]
    delta_text: str
    finish_reason: str | None
    finish_details: dict[str, Any] | None
    usage: dict[str, Any] | None
    done_sentinel: bool
    error: str | None


def _parse_configurations(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("configurations must not be empty")
    unknown = sorted(set(names) - set(CONFIGURATIONS))
    if unknown:
        raise ValueError(f"unknown server configurations: {unknown!r}")
    if len(set(names)) != len(names):
        raise ValueError("configurations must be unique")
    if set(names) == set(_CANONICAL_CONFIGURATIONS) and names != _CANONICAL_CONFIGURATIONS:
        raise ValueError(
            "the complete packet must use canonical c1,packed_c8,packed_c9,packed_c13,serial_c13 order"
        )
    return names


def _stats(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": [],
            "count": 0,
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
        "count": len(samples),
        "median": median,
        "p95": float(p95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "stdev": stdev,
        "stdev_pct_of_median": None if median == 0.0 else 100.0 * stdev / median,
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _owned_physical_plans(
    timeline: Sequence[Mapping[str, Any]],
    observed_request_ids: Sequence[int],
) -> tuple[list[dict[str, Any]], int]:
    """Keep plans owned by this HTTP sample, not a prior poll's last plan."""

    owned = {int(request_id) for request_id in observed_request_ids}
    selected: list[dict[str, Any]] = []
    foreign = 0
    for item in timeline:
        for raw_plan in item.get("physical_group_plans", ()):
            if not isinstance(raw_plan, dict) or not raw_plan:
                continue
            plan = copy.deepcopy(raw_plan)
            plan_request_ids = {
                int(request_id)
                for group in plan.get("groups", ())
                for request_id in group.get("request_ids", ())
            }
            if plan_request_ids and plan_request_ids <= owned:
                selected.append(plan)
            else:
                foreign += 1
    return selected, foreign


def _logical_shape_covers(
    shape: tuple[int, tuple[int, ...], tuple[str, ...]],
    *,
    logical_c: int,
    group_count: int,
) -> bool:
    observed_c, widths, masks = shape
    return bool(
        observed_c == int(logical_c)
        and len(widths) == int(group_count)
        and len(masks) == int(group_count)
        and all(width in {1, 2, 4, 8} for width in widths)
        and all(
            len(mask) == width and set(mask) <= {"0", "1"}
            for width, mask in zip(widths, masks, strict=True)
        )
        and sum(mask.count("1") for mask in masks) == int(logical_c)
    )


def _parse_sse_data_line(line: str) -> dict[str, Any] | str | None:
    text = str(line).strip()
    if not text or not text.startswith("data:"):
        return None
    payload = text[5:].strip()
    if payload == "[DONE]":
        return payload
    return json.loads(payload)


def _reference_c1_summary(
    reference_runs: Mapping[int, _ReferenceRun],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prefill = [float(run.prefill_seconds) for run in reference_runs.values()]
    decode_steps = [
        float(seconds)
        for run in reference_runs.values()
        for seconds in run.decode_step_seconds
    ]
    prefill_stats = _stats(prefill)
    decode_stats = _stats(decode_steps)
    direct_itl = decode_stats["median"]
    c1_summary = summaries.get("c1")
    occupancy_itl = (
        None
        if c1_summary is None
        else c1_summary["scheduler_latency_seconds"]["inter_token"]["median"]
    )
    direct_rate = (
        None if direct_itl is None or float(direct_itl) <= 0.0 else 1.0 / float(direct_itl)
    )
    occupancy_rate = (
        None
        if occupancy_itl is None or float(occupancy_itl) <= 0.0
        else 1.0 / float(occupancy_itl)
    )
    ratio = (
        None
        if direct_rate is None or occupancy_rate is None
        else float(occupancy_rate) / float(direct_rate)
    )
    return {
        "timing_scope": "same-process synchronized eager c1 transition versus scheduler ITL",
        "prompt_count": len(reference_runs),
        "prefill_seconds": prefill_stats,
        "decode_step_seconds": decode_stats,
        "same_process_direct_c1_decode_tok_s": direct_rate,
        "occupancy_one_transition_tok_s": occupancy_rate,
        "occupancy_one_vs_direct_c1": ratio,
        "within_five_percent": bool(ratio is not None and float(ratio) >= 0.95),
    }


def _latency_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for kind, after_row in after.items():
        before_row = before.get(kind, {})
        before_samples = [float(value) for value in before_row.get("samples", ())]
        after_samples = [float(value) for value in after_row.get("samples", ())]
        remaining = Counter(before_samples)
        added: list[float] = []
        for value in after_samples:
            if remaining[value] > 0:
                remaining[value] -= 1
            else:
                added.append(value)
        if any(count > 0 for count in remaining.values()):
            raise RuntimeError(f"latency sample history changed for {kind}")
        result[str(kind)] = added
    return result


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {
        str(key): int(after.get(key, 0)) - int(before.get(key, 0))
        for key in keys
    }


def _prompt_rows(
    tokenizer: Any,
    *,
    rows: int,
    prompt_length: int,
    prompt_token_id: int,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for row_index in range(int(rows)):
        token_id = int(prompt_token_id) + (row_index % 4)
        token_ids = tuple([token_id] * int(prompt_length))
        text = str(tokenizer.decode(token_ids))
        roundtrip = tuple(int(token) for token in tokenizer.encode(text))
        if roundtrip != token_ids:
            raise RuntimeError(
                f"prompt row {row_index} failed exact tokenizer roundtrip: "
                f"expected={len(token_ids)} observed={len(roundtrip)}"
            )
        digest = hashlib.sha256()
        for token in token_ids:
            digest.update(int(token).to_bytes(8, "little", signed=True))
        result.append(
            {
                "row_index": row_index,
                "token_id": token_id,
                "token_ids": token_ids,
                "text": text,
                "token_count": len(token_ids),
                "token_ids_sha256": digest.hexdigest(),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "roundtrip_exact": True,
            }
        )
    return tuple(result)


def _memory_snapshot(label: str, runner: Any) -> dict[str, Any]:
    from hipengine.core.memory import memory_stats

    runtime = runner._shared_runner.runtime
    free_bytes, total_bytes = runtime.mem_get_info()
    return {
        "label": str(label),
        "tracked": memory_stats(),
        "hip_free_bytes": int(free_bytes),
        "hip_total_bytes": int(total_bytes),
        "hip_used_bytes": int(total_bytes - free_bytes),
        "kv_pool": copy.deepcopy(runner.kv_pool_memory_snapshot()),
    }


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    text = resolved.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"compiler-version file is empty: {resolved}")
    return text


@contextmanager
def _temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    prior = {str(key): os.environ.get(str(key)) for key in updates}
    os.environ.update({str(key): str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_reference(session: Any, prompt: Sequence[int], max_tokens: int) -> _ReferenceRun:
    session.reset()
    prefill_started = time.perf_counter()
    first = session.prefill(
        tuple(int(token) for token in prompt),
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=False,
    )
    prefill_seconds = time.perf_counter() - prefill_started
    generated = [int(first.token_id)]
    decode_step_seconds: list[float] = []
    while len(generated) < int(max_tokens):
        decode_started = time.perf_counter()
        generated.append(int(session.step(generated[-1], return_logits=False).token_id))
        decode_step_seconds.append(time.perf_counter() - decode_started)
    return _ReferenceRun(
        generated_tokens=generated,
        prefill_seconds=prefill_seconds,
        decode_step_seconds=decode_step_seconds,
    )


def _stream_completion(
    client: TestClient,
    *,
    served_model_name: str,
    row: Mapping[str, Any],
    max_tokens: int,
    barrier: threading.Barrier | None,
) -> _HTTPTrace:
    if barrier is not None:
        barrier.wait(timeout=30.0)
    started_at = time.perf_counter()
    request_id: int | None = None
    delta_times: list[float] = []
    delta_text: list[str] = []
    finish_reason: str | None = None
    finish_details: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    done = False
    status_code = 0
    error: str | None = None
    try:
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": str(served_model_name),
                "prompt": str(row["text"]),
                "max_tokens": int(max_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {
                    "include_hipengine": True,
                    "include_usage": True,
                },
            },
        ) as response:
            status_code = int(response.status_code)
            if status_code != 200:
                error = response.read().decode("utf-8", errors="replace")
            else:
                for raw_line in response.iter_lines():
                    observed_at = time.perf_counter()
                    payload = _parse_sse_data_line(raw_line)
                    if payload is None:
                        continue
                    if payload == "[DONE]":
                        done = True
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if isinstance(payload.get("usage"), dict):
                        usage = copy.deepcopy(payload["usage"])
                    choices = payload.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    hipengine = choice.get("hipengine")
                    if isinstance(hipengine, dict):
                        decode_state = hipengine.get("decode_state")
                        if isinstance(decode_state, dict) and decode_state.get("request_id") is not None:
                            observed_id = int(decode_state["request_id"])
                            if request_id is not None and request_id != observed_id:
                                raise RuntimeError(
                                    f"HTTP stream request id changed {request_id}->{observed_id}"
                                )
                            request_id = observed_id
                    if choice.get("finish_reason") is None:
                        delta_times.append(observed_at)
                        delta_text.append(str(choice.get("text", "")))
                    else:
                        finish_reason = str(choice.get("finish_reason"))
                        raw_details = choice.get("finish_details")
                        finish_details = (
                            copy.deepcopy(raw_details)
                            if isinstance(raw_details, dict)
                            else None
                        )
    except Exception as exc:  # pragma: no cover - hardware/error-path diagnostic
        error = f"{type(exc).__name__}: {exc}"
    completed_at = time.perf_counter()
    return _HTTPTrace(
        row_index=int(row["row_index"]),
        status_code=status_code,
        request_id=request_id,
        started_at=started_at,
        first_delta_at=delta_times[0] if delta_times else None,
        completed_at=completed_at,
        delta_times=delta_times,
        delta_text="".join(delta_text),
        finish_reason=finish_reason,
        finish_details=finish_details,
        usage=usage,
        done_sentinel=done,
        error=error,
    )


def _wait_for_live_admission_trigger(
    llm: LLM,
    timeline: Sequence[Mapping[str, Any]],
    *,
    timeline_start: int,
    initial_rows: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        snapshot = llm.live_loop_snapshot()
        matching = [
            item
            for item in timeline[int(timeline_start) :]
            if any(
                int(plan.get("logical_c", 0)) == int(initial_rows)
                for plan in item.get("physical_group_plans", ())
            )
        ]
        if int(snapshot["loop"]["requests"]["active"]) >= int(initial_rows) and matching:
            return {
                "triggered_at": time.perf_counter(),
                "snapshot": copy.deepcopy(snapshot),
                "timeline_event": copy.deepcopy(matching[-1]),
            }
        time.sleep(0.001)
    raise TimeoutError(
        f"live-admission trigger did not observe logical C={initial_rows} before timeout"
    )


def _run_http_sample(
    *,
    client: TestClient,
    llm: LLM,
    runner: Any,
    config: ServerConfiguration,
    prompt_rows: Sequence[Mapping[str, Any]],
    reference_tokens: Mapping[int, Sequence[int]],
    max_tokens: int,
    measured: bool,
    run_index: int,
    reclaimed: Mapping[int, _ReclaimedRow],
    timeline: list[dict[str, Any]],
    capture_state: dict[str, Any],
    live_initial_rows: int | None = None,
    live_tail_rows: int = 0,
    trigger_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    rows = int(config.logical_rows)
    if rows > len(prompt_rows):
        raise ValueError(f"configuration {config.name} requires {rows} prompt rows")
    before_snapshot = llm.live_loop_snapshot()
    before_routes = before_snapshot["runner"]["routes"]["counts"]
    before_fallbacks = before_snapshot["runner"]["routes"]["fallback_reasons"]
    before_latency = before_snapshot["loop"]["latency_seconds"]
    timeline_start = len(timeline)
    reclaimed_before = set(reclaimed)
    memory_before = _memory_snapshot(f"before_{config.name}", runner)
    capture_state["label"] = f"{config.name}:{run_index}"
    route_env = {
        "HIPENGINE_GGUF_AR_PACKED_DECODE": "1" if config.packed_decode else "0",
        "HIPENGINE_GGUF_AR_STREAM_DECODE": "0",
    }
    traces: list[_HTTPTrace] = []
    live_trigger: dict[str, Any] | None = None
    with _temporary_environment(route_env):
        with concurrent.futures.ThreadPoolExecutor(max_workers=rows) as executor:
            if live_initial_rows is None:
                barrier = threading.Barrier(rows + 1)
                futures = [
                    executor.submit(
                        _stream_completion,
                        client,
                        served_model_name="qwen35-concurrency",
                        row=prompt_rows[row_index],
                        max_tokens=int(max_tokens),
                        barrier=barrier,
                    )
                    for row_index in range(rows)
                ]
                barrier.wait(timeout=30.0)
            else:
                initial = int(live_initial_rows)
                tail = int(live_tail_rows)
                if initial + tail != rows:
                    raise ValueError("live initial+tail rows must equal configuration rows")
                initial_barrier = threading.Barrier(initial + 1)
                futures = [
                    executor.submit(
                        _stream_completion,
                        client,
                        served_model_name="qwen35-concurrency",
                        row=prompt_rows[row_index],
                        max_tokens=int(max_tokens),
                        barrier=initial_barrier,
                    )
                    for row_index in range(initial)
                ]
                initial_barrier.wait(timeout=30.0)
                live_trigger = _wait_for_live_admission_trigger(
                    llm,
                    timeline,
                    timeline_start=timeline_start,
                    initial_rows=initial,
                    timeout_seconds=float(trigger_timeout_seconds),
                )
                tail_barrier = threading.Barrier(tail + 1)
                futures.extend(
                    executor.submit(
                        _stream_completion,
                        client,
                        served_model_name="qwen35-concurrency",
                        row=prompt_rows[row_index],
                        max_tokens=int(max_tokens),
                        barrier=tail_barrier,
                    )
                    for row_index in range(initial, rows)
                )
                tail_barrier.wait(timeout=30.0)
            traces = [future.result() for future in futures]
    capture_state["label"] = None
    after_snapshot = llm.live_loop_snapshot()
    memory_after = _memory_snapshot(f"after_{config.name}", runner)
    observed_ids = [trace.request_id for trace in traces if trace.request_id is not None]
    new_reclaimed_ids = sorted(set(reclaimed) - reclaimed_before)
    row_reclaims = {
        int(request_id): reclaimed[int(request_id)]
        for request_id in observed_ids
        if int(request_id) in reclaimed
    }
    exact_rows: list[dict[str, Any]] = []
    for trace in sorted(traces, key=lambda item: item.row_index):
        reclaimed_row = (
            None if trace.request_id is None else row_reclaims.get(int(trace.request_id))
        )
        expected = list(reference_tokens[int(prompt_rows[trace.row_index]["token_id"])])[
            : int(max_tokens)
        ]
        expected_prompt_ids = [
            int(token) for token in prompt_rows[trace.row_index]["token_ids"]
        ]
        generated = [] if reclaimed_row is None else list(reclaimed_row.generated_ids)
        actual_prompt_ids = (
            [] if reclaimed_row is None else list(reclaimed_row.prompt_ids)
        )
        prompt_exact = actual_prompt_ids == expected_prompt_ids
        generated_exact = generated == expected
        exact_rows.append(
            {
                "row_index": int(trace.row_index),
                "request_id": trace.request_id,
                "prompt_token_id": int(prompt_rows[trace.row_index]["token_id"]),
                "actual_prompt_ids": actual_prompt_ids,
                "expected_prompt_token_ids_sha256": str(
                    prompt_rows[trace.row_index]["token_ids_sha256"]
                ),
                "prompt_exact": prompt_exact,
                "generated_ids": generated,
                "expected_ids": expected,
                "generated_exact": generated_exact,
                "exact": prompt_exact and generated_exact,
                "generated_count": len(generated),
                "finish_reason": (
                    trace.finish_reason
                    if reclaimed_row is None
                    else reclaimed_row.finish_reason
                ),
                "finish_details": (
                    trace.finish_details
                    if reclaimed_row is None
                    else reclaimed_row.finish_details
                ),
                "observability": (
                    None if reclaimed_row is None else reclaimed_row.observability
                ),
                "block_ids": [] if reclaimed_row is None else reclaimed_row.block_ids,
            }
        )
    sample_timeline = copy.deepcopy(timeline[timeline_start:])
    plans, foreign_plan_count = _owned_physical_plans(
        sample_timeline,
        [int(request_id) for request_id in observed_ids],
    )
    declared_widths_only = all(
        int(group["physical_rows"]) in {1, 2, 4, 8}
        for plan in plans
        for group in plan.get("groups", ())
    )
    packed_plans_only = all(
        group.get("execution_path") in _NATIVE_EXECUTION_PATHS
        for plan in plans
        for group in plan.get("groups", ())
    )
    logical_shapes = sorted(
        {
            (
                int(plan.get("logical_c", 0)),
                tuple(int(group["physical_rows"]) for group in plan.get("groups", ())),
                tuple(
                    "".join("1" if value else "0" for value in group.get("active_mask", ()))
                    for group in plan.get("groups", ())
                ),
            )
            for plan in plans
        }
    )
    latency_samples = _latency_delta(
        before_latency,
        after_snapshot["loop"]["latency_seconds"],
    )
    route_delta = _counter_delta(
        before_routes,
        after_snapshot["runner"]["routes"]["counts"],
    )
    fallback_delta = _counter_delta(
        before_fallbacks,
        after_snapshot["runner"]["routes"]["fallback_reasons"],
    )
    starts = [trace.started_at for trace in traces]
    ends = [trace.completed_at for trace in traces]
    wall_seconds = max(ends) - min(starts) if starts and ends else 0.0
    generated_tokens = sum(row["generated_count"] for row in exact_rows)
    client_ttft = [
        float(trace.first_delta_at - trace.started_at)
        for trace in traces
        if trace.first_delta_at is not None
    ]
    client_itl = [
        float(current - previous)
        for trace in traces
        for previous, current in zip(trace.delta_times, trace.delta_times[1:])
    ]
    expected_serial = config.execution_class == "serial_bridge"
    routes_ok = bool(
        route_delta.get("serial_decode_fallback_steps", 0) > 0
        if expected_serial
        else route_delta.get("serial_decode_fallback_steps", 0) == 0
        and route_delta.get("resident_fallback_requests", 0) == 0
    )
    http_ok = all(
        trace.status_code == 200
        and trace.error is None
        and trace.done_sentinel
        and trace.request_id is not None
        and len(trace.delta_times) == int(max_tokens)
        and isinstance(trace.usage, dict)
        and int(trace.usage.get("prompt_tokens", -1))
        == int(prompt_rows[trace.row_index]["token_count"])
        and int(trace.usage.get("completion_tokens", -1)) == int(max_tokens)
        for trace in traces
    )
    ownership_ok = bool(
        len(observed_ids) == rows
        and len(set(observed_ids)) == rows
        and set(observed_ids) == set(new_reclaimed_ids)
        and int(after_snapshot["loop"]["requests"]["active"]) == 0
        and int(after_snapshot["loop"]["requests"]["pending"]) == 0
        and int(after_snapshot["runner"]["model_runner"]["active_requests"]) == 0
    )
    expected_group_shape_seen = True
    if config.name == "packed_c8":
        expected_group_shape_seen = any(
            logical_c == 8 and widths == (8,) and masks == ("11111111",)
            for logical_c, widths, masks in logical_shapes
        )
    elif config.name == "packed_c9":
        expected_group_shape_seen = any(
            _logical_shape_covers(
                (logical_c, widths, masks),
                logical_c=9,
                group_count=2,
            )
            and widths[0] == 8
            for logical_c, widths, masks in logical_shapes
        )
    elif config.name in {"packed_c13", "serial_c13"}:
        expected_group_shape_seen = any(
            _logical_shape_covers(
                (logical_c, widths, masks),
                logical_c=13,
                group_count=2,
            )
            and widths == (8, 8)
            for logical_c, widths, masks in logical_shapes
        )
    live_shape_seen = bool(
        live_initial_rows is None
        or (
            live_trigger is not None
            and any(logical_c == int(live_initial_rows) for logical_c, _widths, _masks in logical_shapes)
            and any(logical_c == rows for logical_c, _widths, _masks in logical_shapes)
        )
    )
    passed = bool(
        http_ok
        and ownership_ok
        and all(row["exact"] for row in exact_rows)
        and declared_widths_only
        and routes_ok
        and expected_group_shape_seen
        and live_shape_seen
        and (packed_plans_only if not expected_serial else True)
        and generated_tokens == rows * int(max_tokens)
    )
    return {
        "configuration": config.name,
        "run_index": int(run_index),
        "measured": bool(measured),
        "passed": passed,
        "route": {
            **asdict(config),
            "claim_level": (
                "serial_bridge"
                if expected_serial
                else ("native_c1" if config.name == "c1" else "exact_hybrid")
            ),
            "route_counts_delta": route_delta,
            "fallback_reasons_delta": fallback_delta,
            "declared_widths_only": declared_widths_only,
            "packed_plans_only": packed_plans_only,
            "expected_group_shape_seen": expected_group_shape_seen,
        },
        "workload": {
            "prompt_tokens_per_request": int(prompt_rows[0]["token_count"]),
            "generated_tokens_per_request": int(max_tokens),
            "logical_rows": rows,
            "live_initial_rows": live_initial_rows,
            "live_tail_rows": int(live_tail_rows),
        },
        "accounting": {
            "generated_tokens": generated_tokens,
            "expected_generated_tokens": rows * int(max_tokens),
            "authoritative_source": "resident runner reclaim generated_ids",
        },
        "throughput": {
            "wall_seconds": wall_seconds,
            "aggregate_generated_tok_s": (
                generated_tokens / wall_seconds if wall_seconds > 0.0 else None
            ),
            "per_request_generated_tok_s": (
                generated_tokens / wall_seconds / rows
                if wall_seconds > 0.0
                else None
            ),
        },
        "latency": {
            "scheduler": {
                kind: _stats(values) for kind, values in latency_samples.items()
            },
            "client_ttft_seconds": _stats(client_ttft),
            "client_inter_delta_seconds": _stats(client_itl),
            "client_timing_note": (
                "TestClient may buffer ASGI body chunks; scheduler latency is the retained TTFT/ITL source."
            ),
        },
        "http": [
            {
                **asdict(trace),
                "delta_times": [float(value) for value in trace.delta_times],
            }
            for trace in sorted(traces, key=lambda item: item.row_index)
        ],
        "exact_rows": exact_rows,
        "ownership": {
            "observed_request_ids": observed_ids,
            "new_reclaimed_request_ids": new_reclaimed_ids,
            "passed": ownership_ok,
            "final_requests": copy.deepcopy(after_snapshot["loop"]["requests"]),
            "final_model_runner": copy.deepcopy(
                after_snapshot["runner"]["model_runner"]
            ),
        },
        "occupancy": {
            "timeline": sample_timeline,
            "logical_physical_shapes": [
                {
                    "logical_c": logical_c,
                    "physical_widths": list(widths),
                    "active_masks": list(masks),
                }
                for logical_c, widths, masks in logical_shapes
            ],
            "sample_owned_plan_count": len(plans),
            "foreign_plan_count": foreign_plan_count,
            "plan_scope": "all group request_ids belong to this sample's HTTP request IDs",
            "live_trigger": live_trigger,
            "live_shape_seen": live_shape_seen,
        },
        "memory": {
            "before": memory_before,
            "after": memory_after,
        },
    }


def _summarize_configuration(
    config: ServerConfiguration,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    measured = [sample for sample in samples if sample.get("measured") is True]
    rates = [
        float(sample["throughput"]["aggregate_generated_tok_s"])
        for sample in measured
    ]
    per_request = [
        float(sample["throughput"]["per_request_generated_tok_s"])
        for sample in measured
    ]
    walls = [float(sample["throughput"]["wall_seconds"]) for sample in measured]
    latency_kinds = (
        "queue",
        "time_to_first_token",
        "inter_token",
        "service",
        "completion",
    )
    return {
        "configuration": config.name,
        "route": asdict(config),
        "sample_count": len(measured),
        "passed": bool(measured and all(sample.get("passed") is True for sample in measured)),
        "rates": {
            "aggregate_generated_tok_s": _stats(rates),
            "per_request_generated_tok_s": _stats(per_request),
        },
        "wall_seconds": _stats(walls),
        "scheduler_latency_seconds": {
            kind: _stats(
                [
                    float(value)
                    for sample in measured
                    for value in sample["latency"]["scheduler"].get(kind, {}).get(
                        "samples", ()
                    )
                ]
            )
            for kind in latency_kinds
        },
        "variance_guard": {
            "limit_pct": 5.0,
            "aggregate_rate_stdev_pct_of_median": _stats(rates)[
                "stdev_pct_of_median"
            ],
            "passed": bool(
                rates
                and _stats(rates)["stdev_pct_of_median"] is not None
                and float(_stats(rates)["stdev_pct_of_median"]) <= 5.0
            ),
        },
        "samples": [copy.deepcopy(sample) for sample in samples],
    }


def _scaling_summary(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def rate(name: str) -> float | None:
        value = (
            summaries.get(name, {})
            .get("rates", {})
            .get("aggregate_generated_tok_s", {})
            .get("median")
        )
        return float(value) if isinstance(value, (int, float)) else None

    c1 = rate("c1")
    c8 = rate("packed_c8")
    c9 = rate("packed_c9")
    c13 = rate("packed_c13")
    serial13 = rate("serial_c13")
    return {
        "c1_aggregate_generated_tok_s": c1,
        "packed_c8_aggregate_generated_tok_s": c8,
        "packed_c9_aggregate_generated_tok_s": c9,
        "packed_c13_aggregate_generated_tok_s": c13,
        "serial_c13_aggregate_generated_tok_s": serial13,
        "ratios": {
            "packed_c8_vs_c1": _safe_ratio(c8, c1),
            "packed_c9_vs_c1": _safe_ratio(c9, c1),
            "packed_c13_vs_c1": _safe_ratio(c13, c1),
            "packed_c13_vs_serial_c13": _safe_ratio(c13, serial13),
        },
        "grouped_c13_scaling_gate_passed": bool(
            _safe_ratio(c13, c1) is not None
            and _safe_ratio(c13, serial13) is not None
            and float(_safe_ratio(c13, c1)) > 1.0
            and float(_safe_ratio(c13, serial13)) > 1.0
        ),
        "policy": (
            "C>8 lowers to multiple declared physical buckets; no wider native width is claimed."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    names = _parse_configurations(args.configurations)
    if int(args.prompt_length) <= 0:
        raise ValueError("prompt-length must be positive")
    if int(args.decode_tokens) <= 0 or int(args.live_decode_tokens) <= 0:
        raise ValueError("decode token counts must be positive")
    if int(args.warmup_runs) < 0 or int(args.measured_runs) <= 0:
        raise ValueError("warmup-runs must be non-negative and measured-runs positive")
    if float(args.batch_window_ms) < 0.0:
        raise ValueError("batch-window-ms must be non-negative")
    max_rows = max(
        [CONFIGURATIONS[name].logical_rows for name in names]
        + ([int(args.live_initial_rows) + int(args.live_tail_rows)] if not args.skip_live else [])
    )
    if max_rows > 13:
        raise ValueError("the gfx1100 F1 harness currently caps logical rows at 13")
    if int(args.live_initial_rows) <= 0 or int(args.live_tail_rows) <= 0:
        raise ValueError("live initial/tail rows must be positive")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.require_cached_build and compiler_version is None:
        raise ValueError("require-cached-build requires compiler-version-file")
    max_tokens = max(int(args.decode_tokens), int(args.live_decode_tokens))
    max_sequence_length = int(args.prompt_length) + max_tokens + 2
    env = {
        **_EXACT_ENV,
        "HIPENGINE_MAX_ACTIVE_REQUESTS": str(max_rows),
        "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS": str(int(args.prefill_chunk_size)),
        "HIPENGINE_KV_POOL_INITIAL_PAGES": "3",
        "HIPENGINE_KV_POOL_LOW_WATER_PAGES": "3",
        "HIPENGINE_KV_POOL_HIGH_WATER_PAGES": str(max_rows * 3),
        "HIPENGINE_KV_POOL_CHUNK_PAGES": "3",
        "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS": "0",
    }
    started = time.perf_counter()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    source_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    samples_by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    live_sample: dict[str, Any] | None = None
    with _temporary_environment(env):
        llm = LLM(model, backend=str(args.backend))
        try:
            adapter = llm._get_text_generator()
            llm.prepare(
                max_sequence_length=max_sequence_length,
                sampling_params=SamplingParams(max_tokens=max_tokens),
            )
            runner = adapter._runner
            prompt_rows = _prompt_rows(
                runner.generator.tokenizer,
                rows=max_rows,
                prompt_length=int(args.prompt_length),
                prompt_token_id=int(args.prompt_token_id),
            )
            from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

            reference_session = Qwen35GGUFResidentSession(
                model,
                backend=str(args.backend),
                runtime=runner._shared_runner.runtime,
                shared_runner=runner._shared_runner,
                max_sequence_length=max_sequence_length,
                use_wmma_prefill=True,
                use_gemv_decode=True,
                compiler_version=compiler_version,
                require_cached_build=bool(args.require_cached_build),
            )
            try:
                reference_runs = {
                    int(token_id): _run_reference(
                        reference_session,
                        tuple([int(token_id)] * int(args.prompt_length)),
                        max_tokens,
                    )
                    for token_id in sorted(
                        {int(row["token_id"]) for row in prompt_rows}
                    )
                }
                reference_tokens = {
                    token_id: run.generated_tokens
                    for token_id, run in reference_runs.items()
                }
            finally:
                reference_session.close()

            reclaimed: dict[int, _ReclaimedRow] = {}
            original_reclaim = runner.reclaim

            def capture_reclaim(self, completed):
                request_id = int(completed.request_id)
                row = self._rows.get(request_id)
                if row is not None and row.lease is not None and row.slot is not None:
                    self._flush_row_owner(row)
                    reclaimed[request_id] = _ReclaimedRow(
                        request_id=request_id,
                        prompt_ids=[int(token) for token in completed.prompt_tokens],
                        generated_ids=list(row.slot.generated_ids),
                        finish_reason=str(completed.finish_reason),
                        finish_details=completed.finish_details.to_json_dict(),
                        observability=completed.observability.to_json_dict(),
                        block_ids=(
                            []
                            if row.kv_allocation is None
                            else [int(block) for block in row.kv_allocation.block_ids]
                        ),
                        completion_time=time.perf_counter(),
                    )
                return original_reclaim(completed)

            runner.reclaim = MethodType(capture_reclaim, runner)
            timeline: list[dict[str, Any]] = []
            capture_state: dict[str, Any] = {"label": None}
            original_poll = adapter.poll

            def capture_poll(self, *, max_ticks=1):
                events = tuple(original_poll(max_ticks=max_ticks))
                plans: list[dict[str, Any]] = []
                for event in events:
                    if event.kind == "work" and event.work_kind is not None and event.work_kind.value == "decode":
                        plan = runner._last_physical_group_plan
                        if isinstance(plan, dict) and plan:
                            plans.append(copy.deepcopy(plan))
                snapshot = adapter.live_loop_snapshot()
                timeline.append(
                    {
                        "label": capture_state["label"],
                        "observed_at": time.perf_counter(),
                        "admitted": [int(event.request_id) for event in events if event.kind == "admitted"],
                        "tokens": [int(event.request_id) for event in events if event.kind == "token"],
                        "completed": [int(event.request_id) for event in events if event.kind == "completed"],
                        "work": [
                            {
                                "kind": event.work_kind.value,
                                "request_ids": [int(value) for value in event.request_ids],
                            }
                            for event in events
                            if event.kind == "work" and event.work_kind is not None
                        ],
                        "physical_group_plans": plans,
                        "physical_bucket": copy.deepcopy(snapshot["loop"]["physical_bucket"]),
                        "request_counts": copy.deepcopy(snapshot["loop"]["requests"]),
                    }
                )
                return events

            adapter.poll = MethodType(capture_poll, adapter)
            app = create_app(
                ServerConfig(
                    model=str(model),
                    backend=str(args.backend),
                    quant=str(args.quant),
                    served_model_name="qwen35-concurrency",
                    eager_load=False,
                    metrics="prometheus",
                    generation_batch_window_ms=float(args.batch_window_ms),
                    max_context_tokens=max_sequence_length,
                    max_active_requests=max_rows,
                    stream_queue_max_chunks=max_tokens + 8,
                    shutdown_grace_seconds=5.0,
                ),
                llm=llm,
            )
            with TestClient(app) as client:
                for name in names:
                    config = CONFIGURATIONS[name]
                    for raw_index in range(int(args.warmup_runs) + int(args.measured_runs)):
                        measured = raw_index >= int(args.warmup_runs)
                        run_index = (
                            raw_index - int(args.warmup_runs) + 1
                            if measured
                            else raw_index + 1
                        )
                        sample = _run_http_sample(
                            client=client,
                            llm=llm,
                            runner=runner,
                            config=config,
                            prompt_rows=prompt_rows,
                            reference_tokens=reference_tokens,
                            max_tokens=int(args.decode_tokens),
                            measured=measured,
                            run_index=run_index,
                            reclaimed=reclaimed,
                            timeline=timeline,
                            capture_state=capture_state,
                            trigger_timeout_seconds=float(args.trigger_timeout_seconds),
                        )
                        samples_by_name[name].append(sample)
                        print(
                            f"{name} {'measured' if measured else 'warmup'} {run_index}: "
                            f"aggregate={sample['throughput']['aggregate_generated_tok_s']:.6f} "
                            f"wall={sample['throughput']['wall_seconds']:.6f}",
                            file=sys.stderr,
                            flush=True,
                        )
                if not args.skip_live:
                    live_rows = int(args.live_initial_rows) + int(args.live_tail_rows)
                    live_config = ServerConfiguration(
                        "live_c8_to_c13",
                        live_rows,
                        True,
                        "live_admission_exact_hybrid",
                    )
                    live_sample = _run_http_sample(
                        client=client,
                        llm=llm,
                        runner=runner,
                        config=live_config,
                        prompt_rows=prompt_rows,
                        reference_tokens=reference_tokens,
                        max_tokens=int(args.live_decode_tokens),
                        measured=True,
                        run_index=1,
                        reclaimed=reclaimed,
                        timeline=timeline,
                        capture_state=capture_state,
                        live_initial_rows=int(args.live_initial_rows),
                        live_tail_rows=int(args.live_tail_rows),
                        trigger_timeout_seconds=float(args.trigger_timeout_seconds),
                    )
            final_snapshot = llm.live_loop_snapshot()
            final_memory = _memory_snapshot("final", runner)
            resolved_backend = str(runner.generator.backend)
            target_arch = str(runner._shared_runner.target_arch)
        finally:
            llm.close()

    summaries = {
        name: _summarize_configuration(CONFIGURATIONS[name], samples_by_name[name])
        for name in names
    }
    scaling = _scaling_summary(summaries)
    reference_c1 = _reference_c1_summary(reference_runs, summaries)
    complete_packet = tuple(names) == _CANONICAL_CONFIGURATIONS
    static_passed = all(
        summary["passed"] is True and summary["variance_guard"]["passed"] is True
        for summary in summaries.values()
    )
    if "c1" in names:
        static_passed = static_passed and bool(reference_c1["within_five_percent"])
    passed = bool(
        static_passed
        and (live_sample is None or live_sample["passed"] is True)
        and int(final_snapshot["loop"]["requests"]["active"]) == 0
        and int(final_snapshot["loop"]["requests"]["pending"]) == 0
    )
    command = [sys.executable, "scripts/gguf_live_server_bench.py", *sys.argv[1:]]
    artifact_scope = _artifact_backend_scope(resolved_backend, target_arch)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=model,
        quant=str(args.quant),
        kv_dtype="bf16",
        command=command,
        environment={
            **{key: os.environ.get(key) for key in _PROVENANCE_ENV_KEYS},
            **env,
        },
        build_profile=(
            f"{artifact_scope}_gguf_openai_live_concurrency_exact_hybrid_and_serial"
        ),
        timing_protocol=(
            "one prepared model; exact text/raw-token roundtrip prompts; concurrent OpenAI SSE; "
            "scheduler latency samples; resident reclaim token accounting"
        ),
        warmups=int(args.warmup_runs),
        repetitions=int(args.measured_runs),
        profiler={"used": False, "reason": "real server-wall and scheduler-latency packet"},
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "kind": f"{artifact_scope}_gguf_live_server_concurrency_packet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "measurement_complete" if passed else "failed",
        "passed": passed,
        "complete_packet": complete_packet,
        "performance_claim": False,
        "source": {
            "commit": source_commit,
            "dirty": source_dirty,
        },
        "provenance": provenance,
        "workload": {
            "model": str(model),
            "backend": str(args.backend),
            "quant": str(args.quant),
            "kv_dtype": "bf16",
            "prompt_length": int(args.prompt_length),
            "decode_tokens": int(args.decode_tokens),
            "live_decode_tokens": int(args.live_decode_tokens),
            "warmup_runs": int(args.warmup_runs),
            "measured_runs": int(args.measured_runs),
            "configurations": list(names),
            "live_initial_rows": int(args.live_initial_rows),
            "live_tail_rows": int(args.live_tail_rows),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "generation_batch_window_ms": float(args.batch_window_ms),
            "sampling": "greedy_top1_ignore_eos",
            "speculative_decode": False,
        },
        "prompt_manifest": [
            {key: value for key, value in row.items() if key not in {"text", "token_ids"}}
            for row in prompt_rows
        ],
        "reference_tokens": {
            str(key): list(value) for key, value in sorted(reference_tokens.items())
        },
        "reference_c1": reference_c1,
        "summaries": summaries,
        "scaling": scaling,
        "live_admission": live_sample,
        "final_ownership": {
            "loop": copy.deepcopy(final_snapshot["loop"]),
            "runner": copy.deepcopy(final_snapshot["runner"]),
        },
        "final_memory": final_memory,
        "command": shlex.join(command),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Occupancy-adaptive c1 uses the exact eager GEMV route unless the backend graph break-even is met.",
            "Packed c2/c4/c8 groups are eager in this server packet; graph and profiler claims remain separate.",
            "The same-loop packed-off route is an explicit serial bridge and is expected to report fallback reasons.",
            "TestClient may buffer ASGI chunks; scheduler-owned latency samples are authoritative for TTFT/ITL.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=_SUPPORTED_BACKENDS, default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--configurations",
        default=",".join(_CANONICAL_CONFIGURATIONS),
    )
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--live-initial-rows", type=int, default=8)
    parser.add_argument("--live-tail-rows", type=int, default=5)
    parser.add_argument("--live-decode-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument("--trigger-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
