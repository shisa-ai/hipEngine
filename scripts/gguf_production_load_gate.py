#!/usr/bin/env python3
"""Run the gfx11 GGUF OpenAI production workload and SLO gate.

The gate owns one prepared GGUF model and serves it over a real localhost
Uvicorn socket.  It measures static and ragged bursts, deterministic fixed and
Poisson arrivals, cancellation/disconnect, overload/backpressure, idle
recovery, and a sustained soak.  Generated token IDs come from resident-runner
reclaim and are compared with independent c1 sessions; decoded text is never a
throughput denominator.

Before the retained workload, a bounded policy/chunk sweep runs the same mixed
continuous shape.  The selected configuration is the highest exact generated-
token goodput among candidates that satisfy every declared latency SLO.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import http.client
import json
import math
import os
import random
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

import uvicorn

from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.server import ServerConfig, create_app
from scripts.gguf_live_server_bench import (
    _artifact_backend_scope,
    _memory_snapshot,
    _read_compiler_version,
    _run_reference,
    _temporary_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_SUPPORTED_BACKENDS = ("hip_gfx1100", "hip_gfx1151")
_NATIVE_EXECUTION_PATHS = frozenset(
    {"packed_native", "native_c1", "native_c1_eager", "native_c1_graph"}
)
_CANONICAL_WORKLOADS = (
    "static_c1",
    "static_c8",
    "ragged_burst",
    "continuous_fixed",
    "continuous_poisson",
    "cancellation_disconnect",
    "overload",
    "idle_recovery",
    "soak",
)
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
    "HIPENGINE_GGUF_AR_STREAM_DECODE": "0",
    "HIPENGINE_GGUF_AR_PACKED_DECODE": "1",
}
_COUNTER_METRICS = frozenset(
    {
        "hipengine_requests_total",
        "hipengine_request_completed_total",
        "hipengine_request_failed_total",
        "hipengine_request_rejected_total",
        "hipengine_request_cancelled_total",
        "hipengine_prompt_tokens_total",
        "hipengine_completion_tokens_total",
        "hipengine_resident_requests_admitted_total",
        "hipengine_resident_requests_reclaimed_total",
        "hipengine_resident_work_prefill_total",
        "hipengine_resident_work_decode_total",
        "hipengine_resident_work_reclaim_total",
        "hipengine_kv_pool_grow_events_total",
        "hipengine_kv_pool_grow_failures_total",
        "hipengine_kv_pool_shrink_events_total",
        "hipengine_graph_bucket_hits_total",
        "hipengine_graph_bucket_misses_total",
        "hipengine_graph_bucket_captures_total",
        "hipengine_graph_bucket_replays_total",
        "hipengine_graph_bucket_invalidations_total",
    }
)
_SELECTED_GAUGES = frozenset(
    {
        "hipengine_resident_requests_pending",
        "hipengine_resident_requests_active",
        "hipengine_resident_bucket_active_rows",
        "hipengine_resident_bucket_occupancy_ratio",
        "hipengine_kv_pool_current_bytes",
        "hipengine_kv_pool_high_water_observed_bytes",
        "hipengine_kv_pool_current_pages",
        "hipengine_kv_pool_high_water_observed_pages",
        "hipengine_kv_pool_free_pages",
        "hipengine_kv_pool_refcounted_pages",
        "hipengine_kv_pool_pinned_pages",
        "hipengine_generation_queue_depth",
        "hipengine_generation_queue_max_depth",
        "hipengine_generation_requests_active",
        "hipengine_generation_requests_max_active",
    }
)


@dataclass(frozen=True)
class WorkloadRequest:
    label: str
    token_id: int
    prompt_length: int
    max_tokens: int
    arrival_offset_seconds: float = 0.0
    action: str = "complete"
    disconnect_after_tokens: int | None = None
    timeout_ms: float | None = None
    read_delay_seconds: float = 0.0

    @property
    def oracle_key(self) -> str:
        return f"token={int(self.token_id)}:prompt={int(self.prompt_length)}"


@dataclass(frozen=True)
class SLOThresholds:
    queue_p99_seconds: float
    ttft_p95_seconds: float
    itl_p99_seconds: float
    end_to_end_p95_seconds: float


@dataclass(frozen=True)
class RequestResult:
    label: str
    action: str
    outcome: str
    status_code: int
    error_code: str | None
    request_id: int | None
    generated_count: int
    exact: bool
    queue_seconds: float | None
    ttft_seconds: float | None
    inter_token_seconds: tuple[float, ...]
    end_to_end_seconds: float | None
    finish_reason: str | None
    prompt_exact: bool = True
    http_protocol_exact: bool = True
    disconnect_triggered: bool = False
    done_sentinel: bool = True
    error_status_code: int | None = None
    generated_ids: tuple[int, ...] = ()
    expected_ids: tuple[int, ...] = ()
    client_ttft_seconds: float | None = None
    client_inter_delta_seconds: tuple[float, ...] = ()
    scheduler_completion_seconds: float | None = None
    usage: Mapping[str, Any] | None = None
    finish_details: Mapping[str, Any] | None = None
    http_error: str | None = None


@dataclass(frozen=True)
class TuningConfiguration:
    candidate_id: str
    prefill_decode_policy: str
    prefill_chunk_tokens: int
    fair_prefill_burst_chunks: int
    batch_window_ms: float


@dataclass(frozen=True)
class TuningCandidate:
    prefill_decode_policy: str
    prefill_chunk_tokens: int
    goodput_generated_tokens_per_second: float
    ttft_p95_seconds: float
    itl_p99_seconds: float
    passed: bool
    candidate_id: str = ""
    fair_prefill_burst_chunks: int = 1
    batch_window_ms: float = 0.0
    end_to_end_p95_seconds: float = 0.0
    measurement_repetitions: int = 1


@dataclass
class _ReclaimedRow:
    request_id: int
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    finish_reason: str
    finish_details: dict[str, Any]
    observability: dict[str, Any]
    block_ids: tuple[int, ...]
    completed_at: float
    pointers: tuple[int, ...] = ()


@dataclass
class _HTTPTrace:
    spec: WorkloadRequest
    status_code: int
    request_id: int | None
    started_at: float
    first_delta_at: float | None
    completed_at: float
    delta_times: list[float]
    finish_reason: str | None
    finish_details: dict[str, Any] | None
    usage: dict[str, Any] | None
    done_sentinel: bool
    error_code: str | None
    error_status_code: int | None
    error_message: str | None
    disconnected_by_client: bool
    error_payload: Mapping[str, Any] | None = None


class _LocalUvicorn:
    """Run one ASGI app on a pre-bound localhost socket in a thread."""

    def __init__(self, app: Any) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(256)
        self.port = int(self._socket.getsockname()[1])
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [self._socket]},
            name="hipengine-production-load-uvicorn",
            daemon=True,
        )

    def __enter__(self) -> "_LocalUvicorn":
        self.thread.start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.server.started:
                return self
            if not self.thread.is_alive():
                raise RuntimeError("Uvicorn exited before startup")
            time.sleep(0.01)
        raise TimeoutError("Uvicorn did not start within 30 seconds")

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=30.0)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(timeout=10.0)
        self._socket.close()
        if self.thread.is_alive():
            raise RuntimeError("Uvicorn did not stop")


class _LiveSampler:
    """Poll queue depth and scheduler occupancy while one workload runs."""

    def __init__(self, llm: LLM, batcher: Any, *, interval_seconds: float = 0.01) -> None:
        self.llm = llm
        self.batcher = batcher
        self.interval_seconds = max(0.001, float(interval_seconds))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        self._thread.join(timeout=10.0)
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = self.llm.live_loop_snapshot() or {}
                loop = snapshot.get("loop", {})
                bucket = loop.get("physical_bucket", {})
                requests = loop.get("requests", {})
                depths = tuple(int(value) for value in self.batcher.stream_queue_depths())
                self.samples.append(
                    {
                        "observed_at": time.perf_counter(),
                        "active": int(requests.get("active", 0)),
                        "pending": int(requests.get("pending", 0)),
                        "occupancy_ratio": float(bucket.get("occupancy_ratio", 0.0)),
                        "generation_queue_depth": int(self.batcher.queue_depth()),
                        "generation_requests_active": int(self.batcher.active_requests()),
                        "stream_queue_depths": list(depths),
                        "stream_queue_max_depth": max(depths, default=0),
                    }
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                self.samples.append(
                    {
                        "observed_at": time.perf_counter(),
                        "sampling_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            self._stop.wait(self.interval_seconds)


def _nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, math.ceil(float(quantile) * len(ordered)))
    return float(ordered[min(len(ordered), rank) - 1])


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values if math.isfinite(float(value))]
    if not samples:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "mean": None,
            "stdev": None,
        }
    return {
        "count": len(samples),
        "p50": _nearest_rank(samples, 0.50),
        "p95": _nearest_rank(samples, 0.95),
        "p99": _nearest_rank(samples, 0.99),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _poisson_arrival_offsets(*, count: int, rate_per_second: float, seed: int) -> tuple[float, ...]:
    if int(count) <= 0:
        raise ValueError("count must be positive")
    if float(rate_per_second) <= 0.0:
        raise ValueError("rate_per_second must be positive")
    rng = random.Random(int(seed))
    offsets = [0.0]
    for _ in range(1, int(count)):
        offsets.append(offsets[-1] + rng.expovariate(float(rate_per_second)))
    return tuple(offsets)


def _base_shapes() -> tuple[tuple[int, int, int], ...]:
    return (
        (9707, 64, 16),
        (9708, 128, 24),
        (9709, 256, 32),
        (9710, 512, 48),
        (9708, 64, 48),
        (9709, 128, 32),
        (9710, 256, 24),
        (9707, 512, 16),
    )


def _series_specs(
    prefix: str,
    *,
    count: int,
    offsets: Sequence[float],
    max_tokens_cap: int | None = None,
) -> tuple[WorkloadRequest, ...]:
    shapes = _base_shapes()
    result: list[WorkloadRequest] = []
    for index in range(int(count)):
        token_id, prompt_length, max_tokens = shapes[index % len(shapes)]
        if max_tokens_cap is not None:
            max_tokens = min(int(max_tokens), int(max_tokens_cap))
        result.append(
            WorkloadRequest(
                label=f"{prefix}-{index:04d}",
                token_id=token_id,
                prompt_length=prompt_length,
                max_tokens=max_tokens,
                arrival_offset_seconds=float(offsets[index]),
                read_delay_seconds=0.05 if index == 1 and prefix == "cancel" else 0.0,
            )
        )
    return tuple(result)


def _build_workload_specs(
    *,
    fixed_rate_per_second: float,
    poisson_rate_per_second: float,
    poisson_seed: int,
    soak_seconds: float,
    soak_rate_per_second: float,
) -> dict[str, tuple[WorkloadRequest, ...]]:
    for label, value in (
        ("fixed_rate_per_second", fixed_rate_per_second),
        ("poisson_rate_per_second", poisson_rate_per_second),
        ("soak_seconds", soak_seconds),
        ("soak_rate_per_second", soak_rate_per_second),
    ):
        if float(value) <= 0.0:
            raise ValueError(f"{label} must be positive")
    fixed_count = 12
    fixed_offsets = tuple(index / float(fixed_rate_per_second) for index in range(fixed_count))
    poisson_offsets = _poisson_arrival_offsets(
        count=12,
        rate_per_second=float(poisson_rate_per_second),
        seed=int(poisson_seed),
    )
    soak_count = max(8, int(math.ceil(float(soak_seconds) * float(soak_rate_per_second))))
    soak_offsets = tuple(index / float(soak_rate_per_second) for index in range(soak_count))
    ragged = _series_specs("ragged", count=8, offsets=(0.0,) * 8)
    cancellation = list(_series_specs("cancel", count=8, offsets=(0.0,) * 8))
    cancellation[2] = replace(
        cancellation[2],
        action="disconnect",
        disconnect_after_tokens=1,
        max_tokens=48,
    )
    cancellation[5] = replace(
        cancellation[5],
        action="timeout",
        timeout_ms=250.0,
        max_tokens=48,
    )
    overload = tuple(
        WorkloadRequest(
            label=f"overload-{index:04d}",
            token_id=9707 + (index % 4),
            prompt_length=64,
            max_tokens=8,
        )
        for index in range(32)
    )
    return {
        "static_c1": (
            WorkloadRequest("static-c1-0000", 9707, 256, 32),
        ),
        "static_c8": tuple(
            WorkloadRequest(f"static-c8-{index:04d}", 9707 + (index % 4), 256, 32)
            for index in range(8)
        ),
        "ragged_burst": ragged,
        "continuous_fixed": _series_specs(
            "fixed",
            count=fixed_count,
            offsets=fixed_offsets,
        ),
        "continuous_poisson": _series_specs(
            "poisson",
            count=12,
            offsets=poisson_offsets,
        ),
        "cancellation_disconnect": tuple(cancellation),
        "overload": overload,
        "idle_recovery": (
            WorkloadRequest("idle-recovery-0000", 9710, 128, 16),
        ),
        "soak": _series_specs(
            "soak",
            count=soak_count,
            offsets=soak_offsets,
            max_tokens_cap=24,
        ),
    }


def _request_meets_slo(row: RequestResult, slos: SLOThresholds) -> bool:
    if row.outcome != "completed" or not row.exact or not row.prompt_exact:
        return False
    if row.queue_seconds is None or row.queue_seconds > float(slos.queue_p99_seconds):
        return False
    if row.ttft_seconds is None or row.ttft_seconds > float(slos.ttft_p95_seconds):
        return False
    if row.end_to_end_seconds is None or row.end_to_end_seconds > float(slos.end_to_end_p95_seconds):
        return False
    if row.inter_token_seconds and max(row.inter_token_seconds) > float(slos.itl_p99_seconds):
        return False
    return True


def _evaluate_workload(
    name: str,
    rows: Sequence[RequestResult],
    *,
    wall_seconds: float,
    slos: SLOThresholds,
    require_rejects: bool = False,
) -> dict[str, Any]:
    outcomes = Counter(str(row.outcome) for row in rows)
    completed = [row for row in rows if row.outcome == "completed"]
    service_rows = [row for row in completed if row.exact and row.prompt_exact]
    queue = [float(row.queue_seconds) for row in service_rows if row.queue_seconds is not None]
    ttft = [float(row.ttft_seconds) for row in service_rows if row.ttft_seconds is not None]
    itl = [float(value) for row in service_rows for value in row.inter_token_seconds]
    end_to_end = [
        float(row.end_to_end_seconds)
        for row in service_rows
        if row.end_to_end_seconds is not None
    ]
    latency = {
        "queue": _distribution(queue),
        "ttft": _distribution(ttft),
        "itl": _distribution(itl),
        "end_to_end": _distribution(end_to_end),
    }
    exact_generated = sum(int(row.generated_count) for row in rows if row.exact)
    observed_generated = sum(int(row.generated_count) for row in rows)
    qualifying = sum(
        int(row.generated_count) for row in rows if _request_meets_slo(row, slos)
    )
    correctness_passed = all(
        row.exact and row.prompt_exact and row.http_protocol_exact
        for row in rows
    ) and all(
        row.outcome in ({"completed", "rejected"} if require_rejects else {"completed"})
        for row in rows
        if row.action == "complete"
    ) and all(
        row.outcome in {"disconnected", "cancelled"}
        for row in rows
        if row.action == "disconnect"
    ) and all(
        row.outcome == "timeout"
        for row in rows
        if row.action == "timeout"
    )
    slo_checks = {
        "queue_p99": bool(
            latency["queue"]["p99"] is not None
            and float(latency["queue"]["p99"]) <= float(slos.queue_p99_seconds)
        ),
        "ttft_p95": bool(
            latency["ttft"]["p95"] is not None
            and float(latency["ttft"]["p95"]) <= float(slos.ttft_p95_seconds)
        ),
        "itl_p99": bool(
            latency["itl"]["p99"] is not None
            and float(latency["itl"]["p99"]) <= float(slos.itl_p99_seconds)
        ),
        "end_to_end_p95": bool(
            latency["end_to_end"]["p95"] is not None
            and float(latency["end_to_end"]["p95"]) <= float(slos.end_to_end_p95_seconds)
        ),
    }
    slo_passed = bool(service_rows and all(slo_checks.values()))
    rejects = [row for row in rows if row.outcome == "rejected"]
    overload_passed = bool(
        (not require_rejects)
        or (
            completed
            and rejects
            and all(row.error_code == "engine_busy" for row in rejects)
            and all(row.exact and row.prompt_exact for row in completed)
        )
    )
    reasons: list[str] = []
    if not correctness_passed:
        if any(not row.exact or not row.prompt_exact for row in rows if row.outcome != "rejected"):
            reasons.append("generated_token_mismatch")
        if any(not row.http_protocol_exact for row in rows):
            reasons.append("http_sse_protocol_failed")
        allowed_complete_outcomes = {"completed", "rejected"} if require_rejects else {"completed"}
        if any(
            row.action == "complete" and row.outcome not in allowed_complete_outcomes
            for row in rows
        ):
            reasons.append("unexpected_request_outcome")
        if any(row.action == "disconnect" and row.outcome not in {"disconnected", "cancelled"} for row in rows):
            reasons.append("disconnect_not_observed")
        if any(row.action == "timeout" and row.outcome != "timeout" for row in rows):
            reasons.append("timeout_not_observed")
    if not slo_passed:
        reasons.extend(f"slo_{key}_failed" for key, passed in slo_checks.items() if not passed)
    if not overload_passed:
        reasons.append("overload_accept_reject_contract_failed")
    passed = bool(correctness_passed and slo_passed and overload_passed)
    return {
        "name": str(name),
        "passed": passed,
        "wall_seconds": float(wall_seconds),
        "request_count": len(rows),
        "outcomes": dict(sorted(outcomes.items())),
        "accounting": {
            "observed_generated_tokens": observed_generated,
            "exact_generated_tokens": exact_generated,
            "authoritative_source": "resident runner reclaim generated_ids",
        },
        "throughput": {
            "exact_generated_tokens_per_second": (
                exact_generated / float(wall_seconds) if float(wall_seconds) > 0.0 else None
            ),
        },
        "goodput": {
            "definition": (
                "exact completed generated tokens from requests whose queue, TTFT, every ITL, "
                "and end-to-end latency meet the declared SLO"
            ),
            "qualifying_generated_tokens": qualifying,
            "generated_tokens_per_second": (
                qualifying / float(wall_seconds) if float(wall_seconds) > 0.0 else None
            ),
        },
        "latency_seconds": latency,
        "slo": {
            "thresholds": asdict(slos),
            "checks": slo_checks,
            "passed": slo_passed,
        },
        "correctness": {
            "passed": correctness_passed,
            "exact_rows": sum(
                1
                for row in rows
                if row.exact and row.prompt_exact and row.http_protocol_exact
            ),
            "eligible_rows": sum(1 for row in rows if row.outcome != "rejected"),
        },
        "overload": {
            "required": bool(require_rejects),
            "accepted": len(completed),
            "rejected": len(rejects),
            "engine_busy_rejected": sum(row.error_code == "engine_busy" for row in rejects),
            "passed": overload_passed,
        },
        "failure_reasons": sorted(set(reasons)),
        "requests": [asdict(row) for row in rows],
    }


def _load_tuning_protocol(
    path: str | Path,
) -> tuple[tuple[TuningConfiguration, ...], dict[str, Any]]:
    protocol_path = Path(path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "gfx1100_agentic_a4_predeclared_protocol":
        raise ValueError("tuning protocol kind is unsupported")
    if payload.get("status") != "predeclared_ready":
        raise ValueError("tuning protocol is not ready")
    if payload.get("performance_claim") is not False or payload.get("timing_claim") is not False:
        raise ValueError("tuning protocol must be a no-timing predeclaration")
    try:
        rows = payload["candidate_funnel"]["stage_1_mixed_arrival_screen"]["candidates"]
    except (KeyError, TypeError) as exc:
        raise ValueError("tuning protocol candidates are missing") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("tuning protocol candidates must be a non-empty list")
    configurations: list[TuningConfiguration] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("tuning protocol candidate must be an object")
        candidate_id = str(row.get("id", ""))
        policy = str(row.get("prefill_decode_policy", ""))
        chunk = int(row.get("prefill_chunk_tokens", 0))
        burst = int(row.get("fair_prefill_burst_chunks", 0))
        window = float(row.get("batch_window_ms", -1.0))
        if not candidate_id:
            raise ValueError("tuning protocol candidate id must not be empty")
        if policy not in {"protect_decode", "protect_ttft", "fair"}:
            raise ValueError(f"unsupported tuning policy {policy!r}")
        if chunk <= 0 or burst <= 0 or window < 0.0:
            raise ValueError("tuning protocol chunk/burst/window values are invalid")
        configurations.append(
            TuningConfiguration(
                candidate_id=candidate_id,
                prefill_decode_policy=policy,
                prefill_chunk_tokens=chunk,
                fair_prefill_burst_chunks=burst,
                batch_window_ms=window,
            )
        )
    ids = [configuration.candidate_id for configuration in configurations]
    if len(ids) != len(set(ids)):
        raise ValueError("tuning protocol candidate ids must be unique")
    return tuple(configurations), {
        "path": str(protocol_path),
        "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "kind": str(payload["kind"]),
        "source_revision": str(payload.get("source_revision", "")),
        "candidate_count": len(configurations),
    }


def _rotated_tuning_plan(
    configurations: Sequence[TuningConfiguration],
    *,
    repetitions: int,
) -> tuple[tuple[TuningConfiguration, ...], ...]:
    if not configurations:
        raise ValueError("tuning configurations must not be empty")
    if int(repetitions) <= 0:
        raise ValueError("tuning repetitions must be positive")
    rows = tuple(configurations)
    step = max(1, math.ceil(len(rows) / int(repetitions)))
    result = []
    for repetition in range(int(repetitions)):
        offset = (repetition * step) % len(rows)
        result.append(rows[offset:] + rows[:offset])
    return tuple(result)


def _tuning_configuration_from_row(row: Mapping[str, Any]) -> TuningConfiguration:
    configuration = row.get("configuration")
    if isinstance(configuration, TuningConfiguration):
        return configuration
    if not isinstance(configuration, Mapping):
        raise ValueError("tuning run configuration is missing")
    return TuningConfiguration(**dict(configuration))


def _tuning_candidate_from_row(row: Mapping[str, Any]) -> TuningCandidate:
    candidate = row.get("candidate")
    if isinstance(candidate, TuningCandidate):
        return candidate
    if not isinstance(candidate, Mapping):
        raise ValueError("tuning run candidate is missing")
    return TuningCandidate(**dict(candidate))


def _aggregate_tuning_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    configurations: Sequence[TuningConfiguration],
    expected_repetitions: int,
) -> tuple[list[dict[str, Any]], tuple[TuningCandidate, ...]]:
    if int(expected_repetitions) <= 0:
        raise ValueError("expected tuning repetitions must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = {
        configuration.candidate_id: [] for configuration in configurations
    }
    for row in runs:
        configuration = _tuning_configuration_from_row(row)
        if configuration.candidate_id not in grouped:
            raise ValueError(f"undeclared tuning candidate {configuration.candidate_id!r}")
        grouped[configuration.candidate_id].append(row)
    aggregates: list[dict[str, Any]] = []
    candidates: list[TuningCandidate] = []
    expected_indices = set(range(int(expected_repetitions)))
    for configuration in configurations:
        rows = grouped[configuration.candidate_id]
        observed_indices = {int(row.get("repetition", -1)) for row in rows}
        if len(rows) != int(expected_repetitions) or observed_indices != expected_indices:
            raise ValueError(
                f"tuning candidate {configuration.candidate_id!r} is incomplete: "
                f"expected repetitions {sorted(expected_indices)!r}, observed {sorted(observed_indices)!r}"
            )
        samples = [_tuning_candidate_from_row(row) for row in rows]
        if any(
            _tuning_configuration_from_row(row) != configuration
            or sample.candidate_id != configuration.candidate_id
            for row, sample in zip(rows, samples)
        ):
            raise ValueError(f"tuning candidate {configuration.candidate_id!r} configuration drifted")
        aggregate = TuningCandidate(
            prefill_decode_policy=configuration.prefill_decode_policy,
            prefill_chunk_tokens=configuration.prefill_chunk_tokens,
            goodput_generated_tokens_per_second=float(
                statistics.median(
                    sample.goodput_generated_tokens_per_second for sample in samples
                )
            ),
            ttft_p95_seconds=float(
                statistics.median(sample.ttft_p95_seconds for sample in samples)
            ),
            itl_p99_seconds=float(
                statistics.median(sample.itl_p99_seconds for sample in samples)
            ),
            passed=all(sample.passed for sample in samples),
            candidate_id=configuration.candidate_id,
            fair_prefill_burst_chunks=configuration.fair_prefill_burst_chunks,
            batch_window_ms=configuration.batch_window_ms,
            end_to_end_p95_seconds=float(
                statistics.median(sample.end_to_end_p95_seconds for sample in samples)
            ),
            measurement_repetitions=int(expected_repetitions),
        )
        candidates.append(aggregate)
        aggregates.append(
            {
                "configuration": asdict(configuration),
                "complete": True,
                "all_repetitions_passed": aggregate.passed,
                "aggregate": asdict(aggregate),
                "samples": [asdict(sample) for sample in samples],
            }
        )
    return aggregates, tuple(candidates)


def _select_tuning_candidate(candidates: Sequence[TuningCandidate]) -> TuningCandidate:
    passing = [candidate for candidate in candidates if candidate.passed]
    if not passing:
        raise ValueError("no SLO-passing tuning candidate")
    return max(
        passing,
        key=lambda item: (
            float(item.goodput_generated_tokens_per_second),
            -float(item.ttft_p95_seconds),
            -float(item.itl_p99_seconds),
            -float(item.end_to_end_p95_seconds),
            -int(item.prefill_chunk_tokens),
            -float(item.batch_window_ms),
            item.prefill_decode_policy,
        ),
    )


def _prompt_manifest(tokenizer: Any, specs: Sequence[WorkloadRequest]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for spec in specs:
        key = spec.oracle_key
        if key in result:
            continue
        token_ids = tuple([int(spec.token_id)] * int(spec.prompt_length))
        text = str(tokenizer.decode(token_ids))
        roundtrip = tuple(int(token) for token in tokenizer.encode(text))
        if roundtrip != token_ids:
            raise RuntimeError(
                f"prompt {key} failed exact tokenizer roundtrip: expected={len(token_ids)} "
                f"observed={len(roundtrip)}"
            )
        digest = hashlib.sha256()
        for token in token_ids:
            digest.update(int(token).to_bytes(8, "little", signed=True))
        result[key] = {
            "oracle_key": key,
            "token_id": int(spec.token_id),
            "token_ids": token_ids,
            "text": text,
            "prompt_tokens": len(token_ids),
            "token_ids_sha256": digest.hexdigest(),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "roundtrip_exact": True,
        }
    return result


def _parse_sse_line(raw_line: bytes) -> dict[str, Any] | str | None:
    text = raw_line.decode("utf-8", errors="replace").strip()
    if not text or not text.startswith("data:"):
        return None
    payload = text[5:].strip()
    if payload == "[DONE]":
        return payload
    return json.loads(payload)


def _http_json(host: str, port: int, method: str, path: str, *, timeout: float = 30.0) -> Any:
    connection = http.client.HTTPConnection(host, int(port), timeout=float(timeout))
    try:
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"{method} {path} returned {response.status}: {body[:500]!r}")
        content_type = str(response.getheader("Content-Type") or "")
        if "json" in content_type:
            return json.loads(body)
        return body.decode("utf-8", errors="replace")
    finally:
        connection.close()


def _openai_error_fields(
    payload: Mapping[str, Any],
    *,
    fallback_status: int | None = None,
) -> tuple[str | None, int | None, str]:
    extension = payload.get("hipengine")
    extension = extension if isinstance(extension, Mapping) else {}
    raw_code = payload.get("code", extension.get("code"))
    raw_status = payload.get("status_code", extension.get("status_code", fallback_status))
    return (
        None if raw_code is None else str(raw_code),
        None if raw_status is None else int(raw_status),
        str(payload.get("message", "")),
    )


def _stream_request(
    host: str,
    port: int,
    *,
    spec: WorkloadRequest,
    prompt: Mapping[str, Any],
    start_event: threading.Event,
    workload_start: float,
    served_model_name: str,
    request_timeout_seconds: float,
) -> _HTTPTrace:
    start_event.wait(timeout=30.0)
    target = float(workload_start) + float(spec.arrival_offset_seconds)
    remaining = target - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)
    started_at = time.perf_counter()
    payload: dict[str, Any] = {
        "model": str(served_model_name),
        "prompt": str(prompt["text"]),
        "max_tokens": int(spec.max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {
            "include_hipengine": True,
            "include_usage": True,
        },
    }
    if spec.timeout_ms is not None:
        payload["timeout_ms"] = float(spec.timeout_ms)
    connection = http.client.HTTPConnection(host, int(port), timeout=float(request_timeout_seconds))
    status_code = 0
    request_id: int | None = None
    delta_times: list[float] = []
    finish_reason: str | None = None
    finish_details: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    done = False
    error_code: str | None = None
    error_status: int | None = None
    error_message: str | None = None
    error_payload: dict[str, Any] | None = None
    disconnected = False
    try:
        body = json.dumps(payload, separators=(",", ":"))
        connection.request(
            "POST",
            "/v1/completions",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        status_code = int(response.status)
        if status_code != 200:
            raw = response.read()
            try:
                raw_error_payload = json.loads(raw).get("error", {})
                error_payload = (
                    copy.deepcopy(raw_error_payload)
                    if isinstance(raw_error_payload, dict)
                    else None
                )
                error_code, error_status, error_message = _openai_error_fields(
                    raw_error_payload,
                    fallback_status=status_code,
                )
            except Exception:
                error_message = raw.decode("utf-8", errors="replace")
        elif (
            spec.action == "disconnect"
            and spec.disconnect_after_tokens is not None
            and int(spec.disconnect_after_tokens) == 0
        ):
            disconnected = True
            response.close()
            connection.close()
        else:
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                observed_at = time.perf_counter()
                item = _parse_sse_line(raw_line)
                if item is None:
                    continue
                if item == "[DONE]":
                    done = True
                    continue
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("usage"), dict):
                    usage = copy.deepcopy(item["usage"])
                raw_error = item.get("error")
                if isinstance(raw_error, dict):
                    error_payload = copy.deepcopy(raw_error)
                    error_code, error_status, error_message = _openai_error_fields(
                        raw_error
                    )
                choices = item.get("choices")
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
                                f"request id changed during stream {request_id}->{observed_id}"
                            )
                        request_id = observed_id
                if choice.get("finish_reason") is None:
                    delta_times.append(observed_at)
                    if float(spec.read_delay_seconds) > 0.0:
                        time.sleep(float(spec.read_delay_seconds))
                    if (
                        spec.action == "disconnect"
                        and spec.disconnect_after_tokens is not None
                        and len(delta_times) >= int(spec.disconnect_after_tokens)
                    ):
                        disconnected = True
                        response.close()
                        connection.close()
                        break
                else:
                    finish_reason = str(choice.get("finish_reason"))
                    raw_details = choice.get("finish_details")
                    finish_details = copy.deepcopy(raw_details) if isinstance(raw_details, dict) else None
                if disconnected:
                    break
    except Exception as exc:  # pragma: no cover - hardware/socket diagnostics
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
    return _HTTPTrace(
        spec=spec,
        status_code=status_code,
        request_id=request_id,
        started_at=started_at,
        first_delta_at=delta_times[0] if delta_times else None,
        completed_at=time.perf_counter(),
        delta_times=delta_times,
        finish_reason=finish_reason,
        finish_details=finish_details,
        usage=usage,
        done_sentinel=done,
        error_code=error_code,
        error_status_code=error_status,
        error_message=error_message,
        disconnected_by_client=disconnected,
        error_payload=error_payload,
    )


def _prometheus_values(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name_with_labels, raw_value = line.rsplit(None, 1)
        name = name_with_labels.split("{", 1)[0]
        if name not in _COUNTER_METRICS and name not in _SELECTED_GAUGES:
            continue
        values[name] = values.get(name, 0.0) + float(raw_value)
    return values


def _metrics_snapshot(host: str, port: int) -> dict[str, float]:
    return _prometheus_values(_http_json(host, port, "GET", "/metrics"))


def _counter_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {
        name: float(after.get(name, 0.0)) - float(before.get(name, 0.0))
        for name in sorted(_COUNTER_METRICS)
    }


def _wait_for_idle(llm: LLM, batcher: Any, *, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_seconds)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snapshot = llm.live_loop_snapshot() or {}
        loop = snapshot.get("loop", {})
        requests = loop.get("requests", {})
        runner = snapshot.get("runner", {}).get("model_runner", {})
        last = {
            "snapshot": snapshot,
            "generation_queue_depth": int(batcher.queue_depth()),
            "generation_active_requests": int(batcher.active_requests()),
            "generation_batcher_active": bool(batcher.active()),
        }
        if (
            int(requests.get("active", 0)) == 0
            and int(requests.get("pending", 0)) == 0
            and int(runner.get("active_requests", 0)) == 0
            and int(batcher.queue_depth()) == 0
            and int(batcher.active_requests()) == 0
        ):
            return last
        time.sleep(0.01)
    raise TimeoutError(f"serving owner did not drain: {last}")


def _occupancy_summary(
    samples: Sequence[Mapping[str, Any]],
    timeline: Sequence[Mapping[str, Any]],
    *,
    stream_queue_limit: int,
) -> dict[str, Any]:
    active = [float(item.get("active", 0)) for item in samples if "sampling_error" not in item]
    pending = [float(item.get("pending", 0)) for item in samples if "sampling_error" not in item]
    occupancy = [
        float(item.get("occupancy_ratio", 0.0))
        for item in samples
        if "sampling_error" not in item
    ]
    generation_queue = [
        float(item.get("generation_queue_depth", 0))
        for item in samples
        if "sampling_error" not in item
    ]
    stream_depths = [
        int(item.get("stream_queue_max_depth", 0))
        for item in samples
        if "sampling_error" not in item
    ]
    plans = [
        copy.deepcopy(plan)
        for item in timeline
        for plan in item.get("physical_group_plans", ())
        if isinstance(plan, dict) and plan
    ]
    execution_paths = sorted(
        {
            str(group.get("execution_path"))
            for plan in plans
            for group in plan.get("groups", ())
            if group.get("execution_path") is not None
        }
    )
    logical_shapes = sorted(
        {
            (
                int(plan.get("logical_c", 0)),
                tuple(int(group.get("physical_rows", 0)) for group in plan.get("groups", ())),
                tuple(
                    "".join("1" if bit else "0" for bit in group.get("active_mask", ()))
                    for group in plan.get("groups", ())
                ),
            )
            for plan in plans
        }
    )
    routes_passed = bool(
        plans
        and execution_paths
        and set(execution_paths) <= _NATIVE_EXECUTION_PATHS
        and all(width in {1, 2, 4, 8} for _c, widths, _masks in logical_shapes for width in widths)
    )
    max_stream_depth = max(stream_depths, default=0)
    return {
        "sample_count": len(samples),
        "active_rows": _distribution(active),
        "pending_requests": _distribution(pending),
        "occupancy_ratio": _distribution(occupancy),
        "generation_queue_depth": _distribution(generation_queue),
        "stream_queue": {
            "configured_max_chunks": int(stream_queue_limit),
            "max_observed_depth": int(max_stream_depth),
            "bounded": bool(max_stream_depth <= int(stream_queue_limit)),
        },
        "logical_physical_shapes": [
            {
                "logical_c": logical_c,
                "physical_widths": list(widths),
                "active_masks": list(masks),
            }
            for logical_c, widths, masks in logical_shapes
        ],
        "execution_paths": execution_paths,
        "route_passed": routes_passed,
    }


def _trace_outcome(trace: _HTTPTrace, reclaimed: _ReclaimedRow | None) -> str:
    if trace.error_code == "engine_busy" or trace.error_status_code == 429 or trace.status_code == 429:
        return "rejected"
    if trace.error_code == "deadline_exceeded" or (
        reclaimed is not None and reclaimed.finish_reason == "timeout"
    ):
        return "timeout"
    if trace.disconnected_by_client:
        return "disconnected"
    if trace.error_code == "cancelled" or (
        reclaimed is not None and reclaimed.finish_reason in {"cancel", "disconnect"}
    ):
        return "cancelled"
    if trace.finish_reason not in {None, "error"} or (
        reclaimed is not None and reclaimed.finish_reason in {"length", "stop"}
    ):
        return "completed"
    return "failed"


def _reclaimed_row_matches_action(row: _ReclaimedRow, action: str) -> bool:
    if action == "timeout":
        return row.finish_reason == "timeout"
    if action == "disconnect":
        return row.finish_reason in {"cancel", "disconnect"}
    return row.finish_reason in {"length", "stop"}


def _reclaimed_overrides(
    traces: Sequence[_HTTPTrace],
    *,
    prompt_manifest: Mapping[str, Mapping[str, Any]],
    reclaimed_rows: Mapping[int, _ReclaimedRow],
    new_reclaimed_ids: Sequence[int],
) -> dict[str, _ReclaimedRow]:
    claimed = {
        int(trace.request_id)
        for trace in traces
        if trace.request_id is not None
    }
    available = {
        int(request_id): reclaimed_rows[int(request_id)]
        for request_id in new_reclaimed_ids
        if int(request_id) not in claimed
    }
    overrides: dict[str, _ReclaimedRow] = {}
    for trace in traces:
        if trace.request_id is not None:
            continue
        expected_prompt = tuple(
            int(token)
            for token in prompt_manifest[trace.spec.oracle_key]["token_ids"]
        )
        matches = [
            row
            for row in available.values()
            if row.prompt_ids == expected_prompt
            and _reclaimed_row_matches_action(row, trace.spec.action)
        ]
        if len(matches) == 1:
            matched = matches[0]
            overrides[trace.spec.label] = matched
            available.pop(int(matched.request_id), None)
    return overrides


def _join_trace(
    trace: _HTTPTrace,
    *,
    prompt: Mapping[str, Any],
    expected: Sequence[int],
    reclaimed_rows: Mapping[int, _ReclaimedRow],
    reclaimed_override: _ReclaimedRow | None = None,
) -> RequestResult:
    reclaimed = reclaimed_override
    if reclaimed is None and trace.request_id is not None:
        reclaimed = reclaimed_rows.get(int(trace.request_id))
    outcome = _trace_outcome(trace, reclaimed)
    actual_prompt = () if reclaimed is None else reclaimed.prompt_ids
    generated = () if reclaimed is None else reclaimed.generated_ids
    expected_ids = tuple(int(token) for token in expected[: len(generated)])
    prompt_exact = bool(
        outcome in {"rejected", "timeout", "disconnected"} and reclaimed is None
        or actual_prompt == tuple(int(token) for token in prompt["token_ids"])
    )
    generated_exact = bool(
        outcome == "rejected"
        or (
            generated == expected_ids
            and (
                outcome != "completed"
                or len(generated) == int(trace.spec.max_tokens)
            )
        )
    )
    effective_request_id = trace.request_id if reclaimed is None else int(reclaimed.request_id)
    usage_exact = bool(
        isinstance(trace.usage, Mapping)
        and int(trace.usage.get("prompt_tokens", -1)) == int(prompt["prompt_tokens"])
        and int(trace.usage.get("completion_tokens", -1)) == int(trace.spec.max_tokens)
    )
    if outcome == "completed":
        http_protocol_exact = bool(
            trace.status_code == 200
            and trace.error_code is None
            and trace.done_sentinel
            and usage_exact
        )
    elif outcome == "rejected":
        http_protocol_exact = bool(
            trace.error_code == "engine_busy"
            and (trace.status_code == 429 or trace.error_status_code == 429)
        )
    elif outcome == "timeout":
        http_protocol_exact = bool(
            trace.error_code == "deadline_exceeded"
            and trace.error_status_code == 408
            and trace.done_sentinel
        )
    elif outcome in {"cancelled", "disconnected"}:
        http_protocol_exact = bool(
            trace.spec.action == "disconnect"
            and trace.disconnected_by_client
            and trace.status_code == 200
        )
    else:
        http_protocol_exact = False
    observability = {} if reclaimed is None else reclaimed.observability
    scheduler_ttft = observability.get("time_to_first_token_seconds")
    scheduler_itl = tuple(float(value) for value in observability.get("inter_token_seconds", ()))
    client_ttft = (
        None
        if trace.first_delta_at is None
        else float(trace.first_delta_at - trace.started_at)
    )
    client_itl = tuple(
        float(current - previous)
        for previous, current in zip(trace.delta_times, trace.delta_times[1:])
    )
    return RequestResult(
        label=trace.spec.label,
        action=trace.spec.action,
        outcome=outcome,
        status_code=int(trace.status_code),
        error_code=trace.error_code,
        request_id=effective_request_id,
        generated_count=len(generated),
        exact=generated_exact,
        queue_seconds=(
            None if observability.get("queue_seconds") is None else float(observability["queue_seconds"])
        ),
        ttft_seconds=(
            None if scheduler_ttft is None else float(scheduler_ttft)
        ),
        inter_token_seconds=scheduler_itl,
        end_to_end_seconds=float(trace.completed_at - trace.started_at),
        finish_reason=(
            trace.finish_reason if reclaimed is None else reclaimed.finish_reason
        ),
        prompt_exact=prompt_exact,
        http_protocol_exact=http_protocol_exact,
        disconnect_triggered=trace.disconnected_by_client,
        done_sentinel=trace.done_sentinel,
        error_status_code=trace.error_status_code,
        generated_ids=generated,
        expected_ids=expected_ids,
        client_ttft_seconds=client_ttft,
        client_inter_delta_seconds=client_itl,
        scheduler_completion_seconds=(
            None
            if observability.get("completion_seconds") is None
            else float(observability["completion_seconds"])
        ),
        usage=trace.usage,
        finish_details=(
            trace.finish_details if reclaimed is None else reclaimed.finish_details
        ),
        http_error=trace.error_message,
    )


def _execute_workload(
    name: str,
    specs: Sequence[WorkloadRequest],
    *,
    host: str,
    port: int,
    llm: LLM,
    runner: Any,
    batcher: Any,
    prompt_manifest: Mapping[str, Mapping[str, Any]],
    reference_tokens: Mapping[str, Sequence[int]],
    reclaimed: Mapping[int, _ReclaimedRow],
    timeline: list[dict[str, Any]],
    capture_state: dict[str, Any],
    slos: SLOThresholds,
    stream_queue_limit: int,
    idle_timeout_seconds: float,
    request_timeout_seconds: float,
    require_rejects: bool = False,
) -> dict[str, Any]:
    if not specs:
        raise ValueError(f"workload {name} has no requests")
    before_ids = set(reclaimed)
    timeline_start = len(timeline)
    metrics_before = _metrics_snapshot(host, port)
    memory_before = _memory_snapshot(f"before_{name}", runner)
    capture_state["label"] = str(name)
    sampler = _LiveSampler(llm, batcher)
    sampler.start()
    start_event = threading.Event()
    workload_start = time.perf_counter() + 0.02
    traces: list[_HTTPTrace] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(64, len(specs)),
            thread_name_prefix=f"load-{name}",
        ) as executor:
            futures = [
                executor.submit(
                    _stream_request,
                    host,
                    port,
                    spec=spec,
                    prompt=prompt_manifest[spec.oracle_key],
                    start_event=start_event,
                    workload_start=workload_start,
                    served_model_name="qwen35-production-load",
                    request_timeout_seconds=float(request_timeout_seconds),
                )
                for spec in specs
            ]
            start_event.set()
            traces = [future.result() for future in futures]
        idle = _wait_for_idle(llm, batcher, timeout_seconds=float(idle_timeout_seconds))
    finally:
        live_samples = sampler.stop()
        capture_state["label"] = None
    workload_end = max((trace.completed_at for trace in traces), default=time.perf_counter())
    wall_seconds = max(0.0, workload_end - workload_start)
    metrics_after = _metrics_snapshot(host, port)
    memory_after = _memory_snapshot(f"after_{name}", runner)
    new_reclaimed_ids = sorted(set(reclaimed) - before_ids)
    overrides = _reclaimed_overrides(
        traces,
        prompt_manifest=prompt_manifest,
        reclaimed_rows=reclaimed,
        new_reclaimed_ids=new_reclaimed_ids,
    )
    rows = [
        _join_trace(
            trace,
            prompt=prompt_manifest[trace.spec.oracle_key],
            expected=reference_tokens[trace.spec.oracle_key][: int(trace.spec.max_tokens)],
            reclaimed_rows=reclaimed,
            reclaimed_override=overrides.get(trace.spec.label),
        )
        for trace in sorted(traces, key=lambda item: item.spec.label)
    ]
    summary = _evaluate_workload(
        name,
        rows,
        wall_seconds=wall_seconds,
        slos=slos,
        require_rejects=require_rejects,
    )
    sample_timeline = copy.deepcopy(timeline[timeline_start:])
    occupancy = _occupancy_summary(
        live_samples,
        sample_timeline,
        stream_queue_limit=int(stream_queue_limit),
    )
    metrics_delta = _counter_delta(metrics_before, metrics_after)
    completed_rows = [row for row in rows if row.outcome == "completed"]
    metrics_accounting_passed = bool(
        metrics_delta.get("hipengine_requests_total", -1.0) == len(rows)
        and metrics_delta.get("hipengine_request_completed_total", -1.0)
        == len(completed_rows)
        and metrics_delta.get("hipengine_completion_tokens_total", -1.0)
        == sum(row.generated_count for row in completed_rows)
    )
    final_snapshot = llm.live_loop_snapshot() or {}
    route_counts = final_snapshot.get("runner", {}).get("routes", {}).get("counts", {})
    # Workload-local route counts are reconstructed from the owned poll timeline;
    # cumulative counters remain attached for independent auditing.
    summary.update(
        {
            "passed": bool(
                summary["passed"]
                and occupancy["route_passed"]
                and occupancy["stream_queue"]["bounded"]
                and metrics_accounting_passed
            ),
            "occupancy": occupancy,
            "metrics": {
                "before": metrics_before,
                "after": metrics_after,
                "counter_delta": metrics_delta,
                "accounting_passed": metrics_accounting_passed,
            },
            "memory": {
                "before": memory_before,
                "after": memory_after,
            },
            "ownership": {
                "new_reclaimed_request_ids": new_reclaimed_ids,
                "idle": idle,
                "passed": True,
            },
            "route": {
                "cumulative_counts_after": copy.deepcopy(route_counts),
                "sample_execution_paths": occupancy["execution_paths"],
                "passed": occupancy["route_passed"],
            },
            "timeline": sample_timeline,
        }
    )
    if not occupancy["route_passed"]:
        summary["failure_reasons"] = sorted(
            set([*summary["failure_reasons"], "native_route_evidence_failed"])
        )
    if not occupancy["stream_queue"]["bounded"]:
        summary["failure_reasons"] = sorted(
            set([*summary["failure_reasons"], "stream_queue_bound_exceeded"])
        )
    if not metrics_accounting_passed:
        summary["failure_reasons"] = sorted(
            set([*summary["failure_reasons"], "server_counter_accounting_failed"])
        )
    return summary


def _parse_workload_names(
    raw: str,
    *,
    available: Sequence[str] = _CANONICAL_WORKLOADS,
) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("workloads must not be empty")
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown workloads: {unknown!r}")
    if len(set(names)) != len(names):
        raise ValueError("workloads must be unique")
    return names


def _parse_tuning_candidates(raw: str) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for item in str(raw).split(","):
        value = item.strip()
        if not value:
            continue
        try:
            policy, chunk = value.split(":", 1)
        except ValueError as exc:
            raise ValueError("tuning candidates must use policy:chunk") from exc
        if policy not in {"protect_decode", "protect_ttft", "fair"}:
            raise ValueError(f"unsupported tuning policy {policy!r}")
        parsed_chunk = int(chunk)
        if parsed_chunk <= 0:
            raise ValueError("tuning chunk sizes must be positive")
        candidate = (policy, parsed_chunk)
        if candidate in result:
            raise ValueError("tuning candidates must be unique")
        result.append(candidate)
    if not result:
        raise ValueError("at least one tuning candidate is required")
    return tuple(result)


def _memory_recovery_gate(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    tracked_tolerance_bytes: int,
) -> dict[str, Any]:
    baseline_tracked = int(baseline.get("tracked", {}).get("current_bytes", 0))
    final_tracked = int(final.get("tracked", {}).get("current_bytes", 0))
    pool = final.get("kv_pool", {}).get("dynamic_pool") or {}
    refcounted = int(pool.get("refcounted_pages", 0))
    pinned = int(pool.get("pinned_pages", 0))
    current_pages = int(pool.get("current_pages", 0))
    free_pages = int(pool.get("free_pages", 0))
    tracked_delta = final_tracked - baseline_tracked
    passed = bool(
        tracked_delta <= int(tracked_tolerance_bytes)
        and refcounted == 0
        and pinned == 0
        and current_pages == free_pages
    )
    return {
        "passed": passed,
        "baseline_tracked_current_bytes": baseline_tracked,
        "final_tracked_current_bytes": final_tracked,
        "tracked_delta_bytes": tracked_delta,
        "tracked_tolerance_bytes": int(tracked_tolerance_bytes),
        "kv_pool_current_pages": current_pages,
        "kv_pool_free_pages": free_pages,
        "kv_pool_refcounted_pages": refcounted,
        "kv_pool_pinned_pages": pinned,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    if int(args.max_active_requests) <= 0 or int(args.max_queued_requests) <= 0:
        raise ValueError("active and queued request limits must be positive")
    if int(args.max_pending_requests) < int(args.max_queued_requests):
        raise ValueError("max-pending-requests must cover max-queued-requests")
    if int(args.stream_queue_max_chunks) < 2:
        raise ValueError("stream-queue-max-chunks must be at least two")
    if int(args.initial_fair_prefill_burst_chunks) <= 0:
        raise ValueError("initial-fair-prefill-burst-chunks must be positive")
    if int(args.tuning_repetitions) <= 0:
        raise ValueError("tuning-repetitions must be positive")
    parsed_candidates = _parse_tuning_candidates(args.tuning_candidates)
    tuning_protocol_metadata: dict[str, Any] | None = None
    if args.tuning_protocol is not None:
        tuning_configurations, tuning_protocol_metadata = _load_tuning_protocol(
            args.tuning_protocol
        )
    else:
        tuning_configurations = tuple(
            TuningConfiguration(
                candidate_id=f"{policy}_{chunk}",
                prefill_decode_policy=policy,
                prefill_chunk_tokens=chunk,
                fair_prefill_burst_chunks=int(args.initial_fair_prefill_burst_chunks),
                batch_window_ms=float(args.batch_window_ms),
            )
            for policy, chunk in parsed_candidates
        )
    tuning_plans = _rotated_tuning_plan(
        tuning_configurations,
        repetitions=int(args.tuning_repetitions),
    )
    slos = SLOThresholds(
        queue_p99_seconds=float(args.slo_queue_p99_seconds),
        ttft_p95_seconds=float(args.slo_ttft_p95_seconds),
        itl_p99_seconds=float(args.slo_itl_p99_seconds),
        end_to_end_p95_seconds=float(args.slo_end_to_end_p95_seconds),
    )
    all_workloads = _build_workload_specs(
        fixed_rate_per_second=float(args.fixed_rate_per_second),
        poisson_rate_per_second=float(args.poisson_rate_per_second),
        poisson_seed=int(args.poisson_seed),
        soak_seconds=float(args.soak_seconds),
        soak_rate_per_second=float(args.soak_rate_per_second),
    )
    workload_names = _parse_workload_names(
        args.workloads,
        available=tuple(all_workloads),
    )
    workloads = {name: all_workloads[name] for name in workload_names}
    complete_packet = bool(
        workload_names == _CANONICAL_WORKLOADS and not args.skip_tuning
    )
    tuning_specs = all_workloads["continuous_fixed"]
    all_specs = [spec for rows in workloads.values() for spec in rows]
    if not args.skip_tuning:
        all_specs.extend(tuning_specs)
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.require_cached_build and compiler_version is None:
        raise ValueError("require-cached-build requires compiler-version-file")
    max_prompt = max(spec.prompt_length for spec in all_specs)
    max_output = max(spec.max_tokens for spec in all_specs)
    max_sequence_length = max_prompt + max_output + 2
    max_pages_per_request = max(1, math.ceil(max_sequence_length / 256))
    low_water_pages = max_pages_per_request
    high_water_pages = int(args.max_active_requests) * max_pages_per_request
    env = {
        **_EXACT_ENV,
        "HIPENGINE_PREFILL_DECODE_POLICY": str(args.initial_policy),
        "HIPENGINE_MAX_ACTIVE_REQUESTS": str(int(args.max_active_requests)),
        "HIPENGINE_MAX_PENDING_REQUESTS": str(int(args.max_pending_requests)),
        "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS": str(int(args.initial_prefill_chunk_tokens)),
        "HIPENGINE_FAIR_PREFILL_BURST_CHUNKS": str(
            int(args.initial_fair_prefill_burst_chunks)
        ),
        "HIPENGINE_KV_POOL_INITIAL_PAGES": str(low_water_pages),
        "HIPENGINE_KV_POOL_LOW_WATER_PAGES": str(low_water_pages),
        "HIPENGINE_KV_POOL_HIGH_WATER_PAGES": str(high_water_pages),
        "HIPENGINE_KV_POOL_CHUNK_PAGES": str(max_pages_per_request),
        "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS": "0",
    }
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    source_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    started_at = time.perf_counter()
    tuning_runs: list[dict[str, Any]] = []
    tuning_aggregates: list[dict[str, Any]] = []
    selected: TuningCandidate | None = None
    workload_results: dict[str, dict[str, Any]] = {}
    selection_error: str | None = None
    with _temporary_environment(env):
        llm = LLM(
            model,
            backend=str(args.backend),
            max_active_requests=int(args.max_active_requests),
        )
        try:
            adapter = llm._get_text_generator()
            llm.prepare(
                max_sequence_length=max_sequence_length,
                sampling_params=SamplingParams(max_tokens=max_output),
            )
            runner = adapter._runner
            prompt_rows = _prompt_manifest(runner.generator.tokenizer, all_specs)
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
                    key: _run_reference(
                        reference_session,
                        row["token_ids"],
                        max(
                            spec.max_tokens
                            for spec in all_specs
                            if spec.oracle_key == key
                        ),
                    )
                    for key, row in sorted(prompt_rows.items())
                }
            finally:
                reference_session.close()
            reference_tokens = {
                key: tuple(int(token) for token in run.generated_tokens)
                for key, run in reference_runs.items()
            }
            reclaimed: dict[int, _ReclaimedRow] = {}
            reclaimed_lock = threading.Lock()
            original_reclaim = runner.reclaim

            def capture_reclaim(self, completed):
                request_id = int(completed.request_id)
                row = self._rows.get(request_id)
                if row is not None:
                    self._flush_row_owner(row)
                    captured = _ReclaimedRow(
                        request_id=request_id,
                        prompt_ids=tuple(int(token) for token in completed.prompt_tokens),
                        generated_ids=(
                            ()
                            if row.slot is None
                            else tuple(int(token) for token in row.slot.generated_ids)
                        ),
                        finish_reason=str(completed.finish_reason),
                        finish_details=completed.finish_details.to_json_dict(),
                        observability=completed.observability.to_json_dict(),
                        block_ids=(
                            ()
                            if row.kv_allocation is None
                            else tuple(int(block) for block in row.kv_allocation.block_ids)
                        ),
                        completed_at=time.perf_counter(),
                        pointers=(
                            ()
                            if row.kv_allocation is None
                            else tuple(int(pointer) for pointer in row.kv_allocation.pointers)
                        ),
                    )
                    with reclaimed_lock:
                        reclaimed[request_id] = captured
                return original_reclaim(completed)

            runner.reclaim = MethodType(capture_reclaim, runner)
            timeline: list[dict[str, Any]] = []
            capture_state: dict[str, Any] = {"label": None}
            original_poll = adapter.poll

            def capture_poll(self, *, max_ticks=1):
                events = tuple(original_poll(max_ticks=max_ticks))
                plans: list[dict[str, Any]] = []
                for event in events:
                    if (
                        event.kind == "work"
                        and event.work_kind is not None
                        and event.work_kind.value == "decode"
                    ):
                        plan = runner._last_physical_group_plan
                        if isinstance(plan, dict) and plan:
                            plans.append(copy.deepcopy(plan))
                snapshot = adapter.live_loop_snapshot()
                timeline.append(
                    {
                        "label": capture_state["label"],
                        "observed_at": time.perf_counter(),
                        "admitted": [
                            int(event.request_id)
                            for event in events
                            if event.kind == "admitted" and event.request_id is not None
                        ],
                        "tokens": [
                            int(event.request_id)
                            for event in events
                            if event.kind == "token" and event.request_id is not None
                        ],
                        "completed": [
                            int(event.request_id)
                            for event in events
                            if event.kind == "completed" and event.request_id is not None
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
                    served_model_name="qwen35-production-load",
                    eager_load=False,
                    startup_chat_smoke=False,
                    startup_scratch_probe=False,
                    metrics="prometheus",
                    generation_batch_window_ms=float(args.batch_window_ms),
                    max_context_tokens=max_sequence_length,
                    max_active_requests=int(args.max_active_requests),
                    max_queued_requests=int(args.max_queued_requests),
                    queue_retry_after_seconds=int(args.queue_retry_after_seconds),
                    stream_queue_max_chunks=int(args.stream_queue_max_chunks),
                    shutdown_grace_seconds=10.0,
                ),
                llm=llm,
            )
            batcher = app.state.hipengine_generation_batcher
            baseline_memory = _memory_snapshot("prepared_baseline", runner)
            with _LocalUvicorn(app) as server:
                ready = _http_json("127.0.0.1", server.port, "GET", "/ready")
                if not bool(ready.get("ready")):
                    raise RuntimeError(f"server readiness failed: {ready}")
                if args.skip_tuning:
                    selected = TuningCandidate(
                        prefill_decode_policy=str(args.initial_policy),
                        prefill_chunk_tokens=int(args.initial_prefill_chunk_tokens),
                        goodput_generated_tokens_per_second=0.0,
                        ttft_p95_seconds=0.0,
                        itl_p99_seconds=0.0,
                        passed=True,
                        candidate_id="focused_initial",
                        fair_prefill_burst_chunks=int(
                            args.initial_fair_prefill_burst_chunks
                        ),
                        batch_window_ms=float(args.batch_window_ms),
                        end_to_end_p95_seconds=0.0,
                    )
                for repetition, plan in (() if args.skip_tuning else enumerate(tuning_plans)):
                    for order_index, configuration in enumerate(plan):
                        adapter._loop.prefill_decode_policy = str(
                            configuration.prefill_decode_policy
                        )
                        adapter._loop.prefill_chunk_size = int(
                            configuration.prefill_chunk_tokens
                        )
                        adapter._loop.fair_prefill_burst_chunks = int(
                            configuration.fair_prefill_burst_chunks
                        )
                        batcher._batch_window_seconds = (
                            float(configuration.batch_window_ms) / 1000.0
                        )
                        name = (
                            f"tuning_r{repetition}_{order_index}_"
                            f"{configuration.candidate_id}"
                        )
                        summary = _execute_workload(
                            name,
                            tuning_specs,
                            host="127.0.0.1",
                            port=server.port,
                            llm=llm,
                            runner=runner,
                            batcher=batcher,
                            prompt_manifest=prompt_rows,
                            reference_tokens=reference_tokens,
                            reclaimed=reclaimed,
                            timeline=timeline,
                            capture_state=capture_state,
                            slos=slos,
                            stream_queue_limit=int(args.stream_queue_max_chunks),
                            idle_timeout_seconds=float(args.idle_timeout_seconds),
                            request_timeout_seconds=float(args.request_timeout_seconds),
                        )
                        goodput = float(
                            summary["goodput"]["generated_tokens_per_second"] or 0.0
                        )
                        ttft_p95 = float(
                            summary["latency_seconds"]["ttft"]["p95"] or math.inf
                        )
                        itl_p99 = float(
                            summary["latency_seconds"]["itl"]["p99"] or math.inf
                        )
                        end_to_end_p95 = float(
                            summary["latency_seconds"]["end_to_end"]["p95"]
                            or math.inf
                        )
                        candidate = TuningCandidate(
                            prefill_decode_policy=configuration.prefill_decode_policy,
                            prefill_chunk_tokens=configuration.prefill_chunk_tokens,
                            goodput_generated_tokens_per_second=goodput,
                            ttft_p95_seconds=ttft_p95,
                            itl_p99_seconds=itl_p99,
                            passed=bool(summary["passed"]),
                            candidate_id=configuration.candidate_id,
                            fair_prefill_burst_chunks=(
                                configuration.fair_prefill_burst_chunks
                            ),
                            batch_window_ms=configuration.batch_window_ms,
                            end_to_end_p95_seconds=end_to_end_p95,
                        )
                        tuning_runs.append(
                            {
                                "repetition": int(repetition),
                                "order_index": int(order_index),
                                "configuration": asdict(configuration),
                                "candidate": asdict(candidate),
                                "workload": summary,
                            }
                        )
                        print(
                            f"{name}: passed={summary['passed']} goodput={goodput:.6f} "
                            f"ttft_p95={ttft_p95:.6f} itl_p99={itl_p99:.6f} "
                            f"e2e_p95={end_to_end_p95:.6f}",
                            file=sys.stderr,
                            flush=True,
                        )
                if not args.skip_tuning:
                    try:
                        tuning_aggregates, aggregate_candidates = _aggregate_tuning_runs(
                            tuning_runs,
                            configurations=tuning_configurations,
                            expected_repetitions=int(args.tuning_repetitions),
                        )
                        selected = _select_tuning_candidate(aggregate_candidates)
                    except ValueError as exc:
                        selection_error = str(exc)
                if selected is not None:
                    adapter._loop.prefill_decode_policy = selected.prefill_decode_policy
                    adapter._loop.prefill_chunk_size = selected.prefill_chunk_tokens
                    adapter._loop.fair_prefill_burst_chunks = int(
                        selected.fair_prefill_burst_chunks
                    )
                    batcher._batch_window_seconds = (
                        float(selected.batch_window_ms) / 1000.0
                    )
                    for name, specs in workloads.items():
                        if name == "idle_recovery":
                            time.sleep(float(args.idle_recovery_seconds))
                        summary = _execute_workload(
                            name,
                            specs,
                            host="127.0.0.1",
                            port=server.port,
                            llm=llm,
                            runner=runner,
                            batcher=batcher,
                            prompt_manifest=prompt_rows,
                            reference_tokens=reference_tokens,
                            reclaimed=reclaimed,
                            timeline=timeline,
                            capture_state=capture_state,
                            slos=slos,
                            stream_queue_limit=int(args.stream_queue_max_chunks),
                            idle_timeout_seconds=float(args.idle_timeout_seconds),
                            request_timeout_seconds=float(args.request_timeout_seconds),
                            require_rejects=(name == "overload"),
                        )
                        workload_results[name] = summary
                        print(
                            f"{name}: passed={summary['passed']} "
                            f"goodput={float(summary['goodput']['generated_tokens_per_second'] or 0.0):.6f} "
                            f"outcomes={summary['outcomes']}",
                            file=sys.stderr,
                            flush=True,
                        )
                final_idle = _wait_for_idle(
                    llm,
                    batcher,
                    timeout_seconds=float(args.idle_timeout_seconds),
                )
                final_metrics = _metrics_snapshot("127.0.0.1", server.port)
            final_memory = _memory_snapshot("final", runner)
            resolved_backend = str(runner.generator.backend)
            target_arch = str(runner._shared_runner.target_arch)
        finally:
            llm.close()
    memory_recovery = _memory_recovery_gate(
        baseline_memory,
        final_memory,
        tracked_tolerance_bytes=int(args.tracked_memory_tolerance_mib) * 1024 * 1024,
    )
    workload_passed = bool(
        workload_results
        and set(workload_results) == set(workloads)
        and all(row["passed"] is True for row in workload_results.values())
    )
    overload_metrics_ok = bool(
        "overload" not in workload_results
        or (
            workload_results.get("overload", {})
            .get("metrics", {})
            .get("counter_delta", {})
            .get("hipengine_request_rejected_total", 0.0)
            == workload_results.get("overload", {})
            .get("overload", {})
            .get("rejected", -1)
        )
    )
    final_ownership_passed = bool(
        int(final_idle["snapshot"]["loop"]["requests"]["active"]) == 0
        and int(final_idle["snapshot"]["loop"]["requests"]["pending"]) == 0
        and int(final_idle["snapshot"]["runner"]["model_runner"]["active_requests"]) == 0
        and int(final_idle["generation_queue_depth"]) == 0
        and int(final_idle["generation_active_requests"]) == 0
    )
    clean_source_passed = not source_dirty
    passed = bool(
        selected is not None
        and workload_passed
        and overload_metrics_ok
        and final_ownership_passed
        and memory_recovery["passed"]
        and clean_source_passed
    )
    command = [sys.executable, "scripts/gguf_production_load_gate.py", *sys.argv[1:]]
    scope = _artifact_backend_scope(resolved_backend, target_arch)
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
        build_profile=f"{scope}_gguf_production_openai_load_slo",
        timing_protocol=(
            "one prepared model; real localhost Uvicorn SSE; independent c1 generated-ID oracle; "
            "scheduler queue/TTFT/ITL plus client end-to-end; static/ragged/fixed/Poisson/cancel/"
            "disconnect/overload/recovery/soak"
        ),
        warmups=0,
        repetitions=1,
        profiler={"used": False, "reason": "production server-wall and scheduler-SLO gate"},
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "kind": f"{scope}_gguf_production_serving_load_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "accepted"
            if passed and complete_packet
            else "measurement_complete"
            if passed
            else "failed"
        ),
        "passed": passed,
        "complete_packet": complete_packet,
        "performance_claim": bool(passed and complete_packet),
        "source": {"commit": source_commit, "dirty": source_dirty},
        "provenance": provenance,
        "configuration": {
            "model": str(model),
            "backend": str(args.backend),
            "quant": str(args.quant),
            "kv_dtype": "bf16",
            "max_sequence_length": max_sequence_length,
            "max_active_requests": int(args.max_active_requests),
            "max_pending_requests": int(args.max_pending_requests),
            "max_queued_requests": int(args.max_queued_requests),
            "stream_queue_max_chunks": int(args.stream_queue_max_chunks),
            "generation_batch_window_ms": float(args.batch_window_ms),
            "selected_generation_batch_window_ms": (
                None if selected is None else float(selected.batch_window_ms)
            ),
            "selected_fair_prefill_burst_chunks": (
                None if selected is None else int(selected.fair_prefill_burst_chunks)
            ),
            "sampling": "greedy_top1_ignore_eos",
            "speculative_decode": False,
            "slo_thresholds": asdict(slos),
        },
        "tuning": {
            "method": (
                "configured focused validation; no selection claim"
                if args.skip_tuning
                else "maximize median exact generated-token SLO goodput across complete "
                "balanced repetitions; tie-break lower TTFT p95, ITL p99, end-to-end p95, "
                "smaller prefill chunk, then smaller batch window"
            ),
            "protocol": tuning_protocol_metadata,
            "repetitions": int(args.tuning_repetitions),
            "candidates": tuning_runs,
            "candidate_runs": tuning_runs,
            "candidate_aggregates": tuning_aggregates,
            "selected": None if selected is None else asdict(selected),
            "selection_error": selection_error,
        },
        "prompt_manifest": [
            {key: value for key, value in row.items() if key not in {"text", "token_ids"}}
            for _oracle_key, row in sorted(prompt_rows.items())
        ],
        "reference_tokens": {
            key: list(value) for key, value in sorted(reference_tokens.items())
        },
        "workloads": workload_results,
        "gates": {
            "workloads_passed": workload_passed,
            "overload_metrics_exact": overload_metrics_ok,
            "final_ownership_passed": final_ownership_passed,
            "memory_recovery": memory_recovery,
            "clean_source_passed": clean_source_passed,
        },
        "final_metrics": final_metrics,
        "final_ownership": final_idle,
        "baseline_memory": baseline_memory,
        "final_memory": final_memory,
        "command": shlex.join(command),
        "elapsed_seconds": time.perf_counter() - started_at,
        "limitations": [
            "Greedy exact Q4_K_M/BF16-KV is the retained production path; sampled serving is an F5 gate.",
            "The policy sweep is bounded to the declared mixed workload and does not imply universal optimality.",
            "SSE network timing is localhost Uvicorn; scheduler-owned queue/TTFT/ITL and client end-to-end are reported separately.",
            "The soak duration is explicit and configurable; this artifact does not claim multi-day reliability.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=_SUPPORTED_BACKENDS, default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--workloads",
        default=",".join(_CANONICAL_WORKLOADS),
        help="Comma-separated workload subset; subsets are diagnostic and not performance claims.",
    )
    parser.add_argument("--max-active-requests", type=int, default=8)
    parser.add_argument("--max-pending-requests", type=int, default=16)
    parser.add_argument("--max-queued-requests", type=int, default=16)
    parser.add_argument("--stream-queue-max-chunks", type=int, default=16)
    parser.add_argument("--queue-retry-after-seconds", type=int, default=1)
    parser.add_argument("--batch-window-ms", type=float, default=100.0)
    parser.add_argument("--initial-policy", choices=("protect_decode", "protect_ttft", "fair"), default="fair")
    parser.add_argument("--initial-prefill-chunk-tokens", type=int, default=128)
    parser.add_argument("--initial-fair-prefill-burst-chunks", type=int, default=1)
    parser.add_argument(
        "--tuning-candidates",
        default="protect_decode:128,protect_ttft:128,fair:128,fair:256",
    )
    parser.add_argument(
        "--tuning-protocol",
        type=Path,
        help="Optional no-timing A4 protocol whose frozen candidate list replaces --tuning-candidates.",
    )
    parser.add_argument(
        "--tuning-repetitions",
        type=int,
        default=1,
        help="Balanced independently reset repetitions per tuning candidate.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Use the initial policy/chunk for a focused diagnostic workload subset.",
    )
    parser.add_argument("--fixed-rate-per-second", type=float, default=2.0)
    parser.add_argument("--poisson-rate-per-second", type=float, default=2.0)
    parser.add_argument("--poisson-seed", type=int, default=1234)
    parser.add_argument("--soak-seconds", type=float, default=60.0)
    parser.add_argument("--soak-rate-per-second", type=float, default=2.0)
    parser.add_argument("--idle-recovery-seconds", type=float, default=1.0)
    parser.add_argument("--idle-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--slo-queue-p99-seconds", type=float, default=10.0)
    parser.add_argument("--slo-ttft-p95-seconds", type=float, default=10.0)
    parser.add_argument("--slo-itl-p99-seconds", type=float, default=0.5)
    parser.add_argument("--slo-end-to-end-p95-seconds", type=float, default=30.0)
    parser.add_argument("--tracked-memory-tolerance-mib", type=int, default=64)
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
