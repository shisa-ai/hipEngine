"""Readiness and health response composition.

The application collects live engine, service, queue and session state. This
owner receives a normalized snapshot of the values it consumes and decides the
readiness status, diagnostics, public payload shape and HTTP status code. It
does not import FastAPI, an engine, the server application or a generation
scheduler.

Every mapping in :class:`ReadinessSnapshot` is supplied by the application after
it detaches the consumed values from live engine/app state. Frozen dataclasses
alone do not freeze a borrowed map; the collector is responsible for copying it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReadinessStartup:
    """Startup state consumed by the readiness payload."""

    ready: bool
    status: str
    eager_load: bool
    model_loaded: bool
    warmup_complete: bool
    startup_error: Mapping[str, Any] | None
    last_startup_timings: Mapping[str, float | None]
    last_startup_memory: Mapping[str, Any]
    last_startup_checks: Mapping[str, Any]


@dataclass(frozen=True)
class ReadinessConfig:
    """Config values the readiness payload publishes."""

    chat_default_max_tokens: int
    kv_storage: str
    kv_scale_dtype: str
    kv_scale_granularity: str
    dms_requested_metadata_path: str | None
    generation_batch_window_ms: float


@dataclass(frozen=True)
class ReadinessQueue:
    """Read-only generation-batcher queue values."""

    depth: int
    max_depth: int
    worker_active: bool
    active_requests: int
    max_active_requests: int


@dataclass(frozen=True)
class ReadinessSnapshot:
    """Normalized readiness inputs.

    The application supplies values, never an engine, ``app.state``, a batcher or
    a live engine/service mapping. Mappings are detached before construction.
    """

    model_identity: Mapping[str, str]
    startup: ReadinessStartup
    configured_max_context_tokens: int | None
    effective_max_context_tokens: int | None
    loaded_model_count: int
    kv_capability: Mapping[str, Any]
    service_health: Mapping[str, Any] | None
    service_ready: bool
    service_diagnostics: tuple[str, ...]
    engine_service: Mapping[str, Any] | None
    graph: Mapping[str, Any]
    pool: Mapping[str, Any]
    prefix_cache: Mapping[str, Any]
    kv_capacity_estimate: Mapping[str, Any] | None
    selected_device: Mapping[str, Any]
    dms_policy: Any
    scheduler_fairness: Mapping[str, Any]
    queue: ReadinessQueue
    sessions: Mapping[str, Any]
    config: ReadinessConfig


def build_readiness_payload(snapshot: ReadinessSnapshot) -> dict[str, Any]:
    """Assemble the public ``/ready`` payload from normalized values."""

    startup = snapshot.startup
    diagnostics: list[str] = []
    if not startup.ready:
        diagnostics.append("server startup is not ready; check startup.error and server logs")
        if startup.startup_error:
            diagnostics.append(
                f"{startup.startup_error.get('message')}: "
                f"{startup.startup_error.get('guidance')}"
            )
    scratch_check = startup.last_startup_checks.get("scratch_probe")
    if isinstance(scratch_check, Mapping) and str(scratch_check.get("status")) == "reduced":
        diagnostics.append(
            "startup scratch probe ran at width "
            f"{scratch_check.get('max_batch_size')} instead of "
            f"{scratch_check.get('requested_max_batch_size')}: "
            f"{scratch_check.get('width_reduction_detail')}"
        )
    runtime_status = startup.status
    runtime_ready = bool(startup.ready)
    engine_service = snapshot.engine_service
    if engine_service is not None and not bool(engine_service.get("serving", True)):
        unhealthy = engine_service.get("unhealthy")
        if unhealthy:
            runtime_status = "unhealthy"
            diagnostics.append(
                "resident engine service stopped serving after a fatal execution "
                f"failure ({unhealthy.get('exception_type')}: {unhealthy.get('message')})"
                + (
                    f" at {unhealthy.get('location')}"
                    if unhealthy.get("location")
                    else ""
                )
                + "; restart the server to restore serving"
            )
        else:
            runtime_status = "closed"
            diagnostics.append(
                "resident engine service is closed; restart the server to restore serving"
            )
        runtime_ready = False
    elif snapshot.service_health is not None and not snapshot.service_ready:
        # A service that is neither serving nor closed still fails its lifetime
        # checks; report why instead of a bare status change.
        runtime_status = "unavailable"
        runtime_ready = False
        diagnostics.extend(snapshot.service_diagnostics)
    payload: dict[str, Any] = {
        "object": "hipengine.readiness",
        "status": runtime_status,
        "ready": runtime_ready,
        "diagnostics": diagnostics,
        "engine_service": (
            dict(engine_service) if engine_service is not None else None
        ),
        "model": {
            **dict(snapshot.model_identity),
            "loaded": bool(startup.model_loaded),
            "loaded_model_count": int(snapshot.loaded_model_count),
            "kv_capability": dict(snapshot.kv_capability),
        },
        "startup": {
            "eager_load": bool(startup.eager_load),
            "warmup_complete": bool(startup.warmup_complete),
            "last_timings_s": dict(startup.last_startup_timings),
            "checks": dict(startup.last_startup_checks),
            "memory": dict(startup.last_startup_memory),
            "error": (
                None
                if startup.startup_error is None
                else dict(startup.startup_error)
            ),
        },
        "context": {
            "configured_max_context_tokens": snapshot.configured_max_context_tokens,
            "effective_max_context_tokens": (
                None
                if snapshot.effective_max_context_tokens is None
                else int(snapshot.effective_max_context_tokens)
            ),
            "chat_default_max_tokens": snapshot.config.chat_default_max_tokens,
        },
        "kv_capacity": {
            "storage": snapshot.config.kv_storage,
            "scale_dtype": snapshot.config.kv_scale_dtype,
            "scale_granularity": snapshot.config.kv_scale_granularity,
            "estimate": (
                None
                if snapshot.kv_capacity_estimate is None
                else dict(snapshot.kv_capacity_estimate)
            ),
            "pool": dict(snapshot.pool),
        },
        "prefix_cache": dict(snapshot.prefix_cache),
        "dms": {
            "requested_metadata_path": snapshot.config.dms_requested_metadata_path,
            "policy": snapshot.dms_policy,
        },
        "graph_cache": {
            "entries": snapshot.graph["entries"],
            "hits": snapshot.graph["hits"],
            "misses": snapshot.graph["misses"],
            "replay_hit_rate": snapshot.graph["replay_hit_rate"],
        },
        "device": dict(snapshot.selected_device),
        "queue": {
            "depth": snapshot.queue.depth,
            "max_depth": snapshot.queue.max_depth,
            "worker_active": snapshot.queue.worker_active,
            "active_requests": snapshot.queue.active_requests,
            "max_active_requests": snapshot.queue.max_active_requests,
            "scheduler_fairness": dict(snapshot.scheduler_fairness),
            "batch_window_ms": float(snapshot.config.generation_batch_window_ms),
        },
        "sessions": dict(snapshot.sessions),
    }
    if snapshot.service_health is not None:
        # The documented lifetime quartet, kept apart from the service's own
        # ``status``/``serving`` fields published under ``engine_service``.
        payload["service"] = {
            "accepting_requests": bool(snapshot.service_health.get("accepting_requests")),
            "driver_alive": bool(snapshot.service_health.get("driver_alive")),
            "closing": bool(snapshot.service_health.get("closing")),
            "closed": bool(snapshot.service_health.get("closed")),
            "failure": snapshot.service_health.get("failure"),
        }
    return payload


def build_health_payload(model_id: str) -> dict[str, Any]:
    """Assemble the static ``/health`` payload."""

    return {
        "object": "hipengine.health",
        "status": "ok",
        "model": model_id,
    }


def readiness_http_status(payload: Mapping[str, Any]) -> int:
    """Return the public ``/ready`` status for an already-built payload."""

    return 200 if payload["ready"] else 503
