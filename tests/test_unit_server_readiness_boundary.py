"""Contract tests for the readiness/health composition owner.

The owner receives a normalized snapshot and decides the public payload, status
and diagnostics. The application collector keeps engine/config/batcher reads.
These tests pin both halves:

* the owner is importable with standard-library dependencies only;
* the collector detaches consumed maps and reads startup state after its live
  await rather than snapshotting it early.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from hipengine.server import ServerConfig, create_app
from hipengine.server.readiness import (
    ReadinessConfig,
    ReadinessQueue,
    ReadinessSnapshot,
    ReadinessStartup,
    build_health_payload,
    build_readiness_payload,
    readiness_http_status,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER_PATH = REPO_ROOT / "hipengine" / "server" / "readiness.py"


def _startup(**overrides: Any) -> ReadinessStartup:
    values: dict[str, Any] = {
        "ready": True,
        "status": "ready",
        "eager_load": False,
        "model_loaded": True,
        "warmup_complete": True,
        "startup_error": None,
        "last_startup_timings": {"engine_create_s": None},
        "last_startup_memory": {"stage": "startup_begin"},
        "last_startup_checks": {},
    }
    values.update(overrides)
    return ReadinessStartup(**values)


def _config(**overrides: Any) -> ReadinessConfig:
    values: dict[str, Any] = {
        "chat_default_max_tokens": 4096,
        "kv_storage": "bf16",
        "kv_scale_dtype": "none",
        "kv_scale_granularity": "none",
        "dms_requested_metadata_path": None,
        "generation_batch_window_ms": 2.0,
    }
    values.update(overrides)
    return ReadinessConfig(**values)


def _queue(**overrides: Any) -> ReadinessQueue:
    values: dict[str, Any] = {
        "depth": 0,
        "max_depth": 8,
        "worker_active": False,
        "active_requests": 0,
        "max_active_requests": 2,
    }
    values.update(overrides)
    return ReadinessQueue(**values)


def _snapshot(**overrides: Any) -> ReadinessSnapshot:
    values: dict[str, Any] = {
        "model_identity": {"id": "fake-model", "backend": "test", "quant": "test"},
        "startup": _startup(),
        "configured_max_context_tokens": 512,
        "effective_max_context_tokens": 512,
        "loaded_model_count": 1,
        "kv_capability": {"storage": "bf16"},
        "service_health": None,
        "service_ready": True,
        "service_diagnostics": (),
        "engine_service": None,
        "graph": {"entries": 1, "hits": 2, "misses": 3, "replay_hit_rate": 0.5},
        "pool": {"current_bytes": 10},
        "prefix_cache": {"mode": "off"},
        "kv_capacity_estimate": {"tokens": 512},
        "selected_device": {"device": "gfx1151"},
        "dms_policy": None,
        "scheduler_fairness": {"policy": "fifo"},
        "queue": _queue(),
        "sessions": {"active": 0, "continuations": {"active": 0}},
        "config": _config(),
    }
    values.update(overrides)
    return ReadinessSnapshot(**values)


def test_health_payload_is_exact() -> None:
    assert build_health_payload("fake-model") == {
        "object": "hipengine.health",
        "status": "ok",
        "model": "fake-model",
    }


def test_readiness_http_status_is_200_or_503() -> None:
    assert readiness_http_status({"ready": True}) == 200
    assert readiness_http_status({"ready": False}) == 503


def test_healthy_payload_shape_and_values() -> None:
    payload = build_readiness_payload(_snapshot())

    assert list(payload) == [
        "object",
        "status",
        "ready",
        "diagnostics",
        "engine_service",
        "model",
        "startup",
        "context",
        "kv_capacity",
        "prefix_cache",
        "dms",
        "graph_cache",
        "device",
        "queue",
        "sessions",
    ]
    assert payload == {
        "object": "hipengine.readiness",
        "status": "ready",
        "ready": True,
        "diagnostics": [],
        "engine_service": None,
        "model": {
            "id": "fake-model",
            "backend": "test",
            "quant": "test",
            "loaded": True,
            "loaded_model_count": 1,
            "kv_capability": {"storage": "bf16"},
        },
        "startup": {
            "eager_load": False,
            "warmup_complete": True,
            "last_timings_s": {"engine_create_s": None},
            "checks": {},
            "memory": {"stage": "startup_begin"},
            "error": None,
        },
        "context": {
            "configured_max_context_tokens": 512,
            "effective_max_context_tokens": 512,
            "chat_default_max_tokens": 4096,
        },
        "kv_capacity": {
            "storage": "bf16",
            "scale_dtype": "none",
            "scale_granularity": "none",
            "estimate": {"tokens": 512},
            "pool": {"current_bytes": 10},
        },
        "prefix_cache": {"mode": "off"},
        "dms": {"requested_metadata_path": None, "policy": None},
        "graph_cache": {
            "entries": 1,
            "hits": 2,
            "misses": 3,
            "replay_hit_rate": 0.5,
        },
        "device": {"device": "gfx1151"},
        "queue": {
            "depth": 0,
            "max_depth": 8,
            "worker_active": False,
            "active_requests": 0,
            "max_active_requests": 2,
            "scheduler_fairness": {"policy": "fifo"},
            "batch_window_ms": 2.0,
        },
        "sessions": {"active": 0, "continuations": {"active": 0}},
    }


def test_startup_diagnostics_keep_order_and_text() -> None:
    payload = build_readiness_payload(
        _snapshot(
            startup=_startup(
                ready=False,
                status="error",
                warmup_complete=False,
                startup_error={
                    "message": "startup scratch_probe failed",
                    "guidance": "reduce the requested batch width",
                },
                last_startup_checks={
                    "scratch_probe": {
                        "status": "reduced",
                        "max_batch_size": 2,
                        "requested_max_batch_size": 4,
                        "width_reduction_detail": "scratch allocation missed",
                    }
                },
            )
        )
    )

    assert payload["status"] == "error"
    assert payload["ready"] is False
    assert payload["diagnostics"] == [
        "server startup is not ready; check startup.error and server logs",
        "startup scratch_probe failed: reduce the requested batch width",
        "startup scratch probe ran at width 2 instead of 4: scratch allocation missed",
    ]


def test_unavailable_service_reports_lifetime_diagnostics() -> None:
    diagnostics = (
        "engine service is not accepting requests; check server logs and restart the server",
        "engine service failed: RuntimeError",
    )
    payload = build_readiness_payload(
        _snapshot(
            service_health={
                "accepting_requests": False,
                "driver_alive": False,
                "closing": False,
                "closed": False,
                "failure": "RuntimeError",
            },
            service_ready=False,
            service_diagnostics=diagnostics,
        )
    )

    assert payload["status"] == "unavailable"
    assert payload["ready"] is False
    assert payload["diagnostics"] == list(diagnostics)
    assert payload["service"] == {
        "accepting_requests": False,
        "driver_alive": False,
        "closing": False,
        "closed": False,
        "failure": "RuntimeError",
    }
    assert list(payload)[-1] == "service"


def test_closed_engine_service_reports_closed() -> None:
    payload = build_readiness_payload(
        _snapshot(engine_service={"serving": False})
    )

    assert payload["status"] == "closed"
    assert payload["ready"] is False
    assert payload["diagnostics"] == [
        "resident engine service is closed; restart the server to restore serving"
    ]
    assert payload["engine_service"] == {"serving": False}
    assert "service" not in payload


def test_unhealthy_engine_service_reports_failure_location() -> None:
    payload = build_readiness_payload(
        _snapshot(
            engine_service={
                "serving": False,
                "unhealthy": {
                    "exception_type": "RuntimeError",
                    "message": "boom",
                    "location": "engine_loop.py:1",
                },
            }
        )
    )

    assert payload["status"] == "unhealthy"
    assert payload["ready"] is False
    assert payload["diagnostics"] == [
        "resident engine service stopped serving after a fatal execution "
        "failure (RuntimeError: boom) at engine_loop.py:1; restart the server "
        "to restore serving"
    ]
    assert payload["engine_service"]["unhealthy"] == {
        "exception_type": "RuntimeError",
        "message": "boom",
        "location": "engine_loop.py:1",
    }


def test_missing_service_health_keeps_legacy_shape() -> None:
    payload = build_readiness_payload(_snapshot(service_health=None, service_ready=True))

    assert "service" not in payload
    assert payload["status"] == "ready"


def test_owner_detaches_top_level_maps_from_payload() -> None:
    identity = {"id": "fake-model", "backend": "test", "quant": "test"}
    pool = {"current_bytes": 10}
    sessions = {"active": 0, "continuations": {"active": 0}}
    payload = build_readiness_payload(
        _snapshot(model_identity=identity, pool=pool, sessions=sessions)
    )

    identity["id"] = "mutated"
    pool["current_bytes"] = 99
    sessions["active"] = 9

    assert payload["model"]["id"] == "fake-model"
    assert payload["kv_capacity"]["pool"] == {"current_bytes": 10}
    assert payload["sessions"]["active"] == 0


def test_owner_imports_only_standard_library() -> None:
    tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "collections", "dataclasses", "typing"}


def test_owner_loads_without_the_application_or_engine() -> None:
    script = textwrap.dedent(
        f"""
        import importlib.util
        import sys

        for name in ("hipengine", "hipengine.server.api", "fastapi", "starlette", "torch"):
            assert name not in sys.modules, name
        spec = importlib.util.spec_from_file_location("_readiness_standalone", {str(OWNER_PATH)!r})
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert module.build_health_payload("m")["model"] == "m"
        assert module.readiness_http_status({{"ready": False}}) == 503
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class _DetachingEngine:
    """A resident engine whose service payload is mutated after the response."""

    def __init__(self) -> None:
        self.service = {
            "serving": False,
            "unhealthy": {"exception_type": "RuntimeError", "message": "boom"},
        }
        self.snapshot_calls = 0

    def service_health(self) -> dict[str, Any]:
        return {
            "accepting_requests": True,
            "driver_alive": True,
            "closing": False,
            "closed": False,
            "failure": None,
        }

    def engine_service_health(self) -> dict[str, Any]:
        return self.service

    def live_loop_snapshot(self) -> dict[str, Any]:
        self.snapshot_calls += 1
        return {}


def test_ready_detaches_nested_service_and_startup_maps(monkeypatch) -> None:
    import hipengine.server.api as api

    engine = _DetachingEngine()
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=engine,
    )
    readiness = app.state.hipengine_readiness
    startup_error = {"message": "startup failed", "guidance": "retry"}
    scratch = {
        "status": "reduced",
        "max_batch_size": 2,
        "requested_max_batch_size": 4,
        "width_reduction_detail": "detail",
    }
    readiness.startup_error = startup_error
    readiness.last_startup_checks = {"scratch_probe": scratch}

    original_build = build_readiness_payload
    observed = []

    def inspect_snapshot(snapshot):
        # Mutate live sources before payload building and HTTP serialization.
        # A decoded JSON response alone cannot prove collector detachment.
        engine.service["unhealthy"]["message"] = "mutated"
        startup_error["message"] = "mutated"
        scratch["width_reduction_detail"] = "mutated"
        assert snapshot.engine_service["unhealthy"]["message"] == "boom"
        assert snapshot.startup.startup_error["message"] == "startup failed"
        assert snapshot.startup.last_startup_checks["scratch_probe"]["width_reduction_detail"] == "detail"
        observed.append(snapshot)
        return original_build(snapshot)

    monkeypatch.setattr(api, "build_readiness_payload", inspect_snapshot)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert len(observed) == 1
    payload = response.json()

    assert payload["engine_service"]["unhealthy"]["message"] == "boom"
    assert payload["startup"]["error"]["message"] == "startup failed"
    assert payload["startup"]["checks"]["scratch_probe"]["width_reduction_detail"] == "detail"


class _RacingEngine:
    """Mutate startup state while the readiness live probe is in flight."""

    def __init__(self) -> None:
        self.readiness: Any | None = None

    def service_health(self) -> dict[str, Any]:
        return {
            "accepting_requests": True,
            "driver_alive": True,
            "closing": False,
            "closed": False,
            "failure": None,
        }

    def live_loop_snapshot(self) -> dict[str, Any]:
        assert self.readiness is not None
        self.readiness.ready = False
        self.readiness.status = "draining"
        return {}


def test_ready_reads_startup_state_after_the_live_await() -> None:
    engine = _RacingEngine()
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=engine,
    )
    engine.readiness = app.state.hipengine_readiness

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "draining"
    assert response.json()["ready"] is False
