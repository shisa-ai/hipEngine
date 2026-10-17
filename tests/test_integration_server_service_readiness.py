"""HTTP /ready must reflect a failed, closing, or closed EngineService.

The readiness route historically reported only startup state. Once a resident
engine service owns the model, a startup-ready server can still be unable to
accept work. These tests pin the service-aware contract: unhealthy services
answer 503 without touching the blocking live-loop snapshot, healthy services
keep their live metrics, and legacy engines without a health hook are unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import (
    EngineService,
    GenerationRequest,
    GenerationSubmission,
)
from hipengine.server import ServerConfig, create_app

_PREFIX_CACHE = {
    "mode": "radix",
    "block_size_tokens": 256,
    "snapshot_entries": 2,
    "snapshot_limit": 4,
    "retained_snapshot_entries": 2,
    "snapshot_bytes": 1024,
    "retained_kv_pages": 3,
    "retained_kv_bytes": 12_288,
    "resident_bytes": 13_312,
    "resident_limit_bytes": 65_536,
}


class _FakeServiceHealth:
    """Agreed stand-in for EngineService.service_health()."""

    def __init__(self, **health: Any) -> None:
        self.payload: dict[str, Any] = {
            "accepting_requests": True,
            "driver_alive": True,
            "closing": False,
            "closed": False,
            "failure": None,
        }
        self.payload.update(health)

    def service_health(self) -> dict[str, Any]:
        return dict(self.payload)


class _ServiceEngine:
    """Minimal resident-engine double whose loaded text generator owns a service."""

    def __init__(
        self,
        service: Any | None,
        *,
        snapshot: Any | None = None,
    ) -> None:
        self._text_generator = service
        self._snapshot = snapshot if snapshot is not None else (lambda: {})
        self.snapshot_calls = 0

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: SamplingParams,
    ) -> int:
        return 4096

    def generate(self, prompts: Any, sampling_params: SamplingParams) -> list[str]:
        return [f"warm:{prompt}" for prompt in prompts]

    def prepare_request_scratch(self, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "ok"}

    def live_loop_snapshot(self) -> Mapping[str, Any]:
        self.snapshot_calls += 1
        return self._snapshot()


class _ScratchProbeFailureEngine(_ServiceEngine):
    def prepare_request_scratch(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("scratch probe failed")


class _RaisingServiceHealth:
    def service_health(self) -> dict[str, Any]:
        raise RuntimeError("health probe failed")


class _LazyLoadGuardEngine(_ServiceEngine):
    def _get_text_generator(self) -> Any:
        raise AssertionError("readiness must not lazy-load a model")


def _config(**overrides: Any) -> ServerConfig:
    options: dict[str, Any] = {
        "model": "fake-path",
        "served_model_name": "fake-model",
        "eager_load": False,
        "startup_chat_smoke": False,
    }
    options.update(overrides)
    return ServerConfig(**options)


def _healthy_snapshot() -> Mapping[str, Any]:
    return {"runner": {"prefix_cache": _PREFIX_CACHE}}


def test_ready_reports_service_driver_failure_as_unavailable() -> None:
    service = _FakeServiceHealth()
    engine = _ServiceEngine(service, snapshot=_healthy_snapshot)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)

    healthy = client.get("/ready")
    assert healthy.status_code == 200
    assert healthy.json()["service"] == {
        "accepting_requests": True,
        "driver_alive": True,
        "closing": False,
        "closed": False,
        "failure": None,
    }

    service.payload.update(
        {
            "accepting_requests": False,
            "driver_alive": False,
            "failure": "RuntimeError",
        }
    )
    failed = client.get("/ready")

    assert failed.status_code == 503
    body = failed.json()
    assert body["ready"] is False
    assert body["status"] == "unavailable"
    assert body["service"] == {
        "accepting_requests": False,
        "driver_alive": False,
        "closing": False,
        "closed": False,
        "failure": "RuntimeError",
    }
    assert any("not accepting requests" in item for item in body["diagnostics"])
    assert any("engine service failed: RuntimeError" in item for item in body["diagnostics"])
    # Startup evidence is retained rather than replaced by the service verdict.
    assert body["startup"]["eager_load"] is False
    assert body["startup"]["warmup_complete"] is True
    assert body["startup"]["error"] is None
    assert body["model"]["loaded"] is True


@pytest.mark.parametrize("state", ["closing", "closed"])
def test_ready_reports_service_shutdown_states_as_unavailable(state: str) -> None:
    service = _FakeServiceHealth()
    engine = _ServiceEngine(service, snapshot=_healthy_snapshot)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)

    assert client.get("/ready").status_code == 200

    service.payload.update({"accepting_requests": False, state: True})
    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["status"] == "unavailable"
    assert body["service"][state] is True
    assert any(f"engine service is {state}" in item for item in body["diagnostics"])


def test_ready_keeps_live_snapshot_metrics_for_healthy_service() -> None:
    service = _FakeServiceHealth()
    engine = _ServiceEngine(service, snapshot=_healthy_snapshot)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["status"] == "ready"
    assert body["prefix_cache"] == _PREFIX_CACHE
    assert engine.snapshot_calls == 1


def test_ready_startup_not_ready_stays_503_with_healthy_service() -> None:
    service = _FakeServiceHealth()
    engine = _ScratchProbeFailureEngine(service, snapshot=_healthy_snapshot)
    app = create_app(_config(eager_load=True, eager_load_max_tokens=2), llm=engine)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["status"] == "error"
    assert body["startup"]["error"]["stage"] == "scratch_probe"
    assert body["startup"]["warmup_complete"] is False
    assert body["service"]["accepting_requests"] is True
    assert body["diagnostics"][0] == (
        "server startup is not ready; check startup.error and server logs"
    )


def test_ready_without_service_health_hook_preserves_startup_behavior() -> None:
    engine = _ServiceEngine(None, snapshot=_healthy_snapshot)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["status"] == "ready"
    assert "service" not in body
    assert body["prefix_cache"] == _PREFIX_CACHE
    assert engine.snapshot_calls == 1


def test_ready_unhealthy_service_never_reads_live_loop_snapshot() -> None:
    def _snapshot_must_not_run() -> Mapping[str, Any]:
        raise AssertionError("unhealthy service must not be probed for a live snapshot")

    service = _FakeServiceHealth(
        accepting_requests=False,
        driver_alive=False,
        failure="RuntimeError",
    )
    engine = _ServiceEngine(service, snapshot=_snapshot_must_not_run)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["status"] == "unavailable"
    assert engine.snapshot_calls == 0


def test_ready_treats_unreadable_service_health_probe_as_unavailable() -> None:
    def _snapshot_must_not_run() -> Mapping[str, Any]:
        raise AssertionError("unreadable service health must not be probed for a live snapshot")

    engine = _ServiceEngine(_RaisingServiceHealth(), snapshot=_snapshot_must_not_run)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["status"] == "unavailable"
    assert body["service"]["failure"] == "RuntimeError"
    assert any("engine service failed: RuntimeError" in item for item in body["diagnostics"])
    assert engine.snapshot_calls == 0


def test_ready_does_not_lazy_load_a_text_generator_for_service_health() -> None:
    engine = _LazyLoadGuardEngine(None, snapshot=_healthy_snapshot)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert "service" not in response.json()


class _IdleDriver:
    stream_queue_max_chunks = 4

    def __init__(self) -> None:
        self.closed = False

    def poll(self, *, max_ticks: int = 1) -> tuple[Any, ...]:
        return ()

    def close(self) -> None:
        self.closed = True


class _FailingDriver(_IdleDriver):
    def submit_detailed(self, request: GenerationRequest) -> GenerationSubmission:
        return GenerationSubmission(request_ids=(1,), request=request, max_ticks=4)

    def poll(self, *, max_ticks: int = 1) -> tuple[Any, ...]:
        raise RuntimeError("driver tick failed")


def test_ready_reports_real_engine_service_closed() -> None:
    driver = _IdleDriver()
    service = EngineService(driver, idle_wait_seconds=0.0)
    engine = _ServiceEngine(service, snapshot=_healthy_snapshot)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)
    try:
        healthy = client.get("/ready")
        assert healthy.status_code == 200
        assert healthy.json()["service"] == {
            "accepting_requests": True,
            "driver_alive": True,
            "closing": False,
            "closed": False,
            "failure": None,
        }

        service.close()

        closed = client.get("/ready")
    finally:
        service.close()

    assert closed.status_code == 503
    body = closed.json()
    assert body["ready"] is False
    assert body["status"] == "unavailable"
    assert body["service"]["closed"] is True
    assert body["service"]["accepting_requests"] is False
    assert any("engine service is closed" in item for item in body["diagnostics"])
    assert body["startup"]["warmup_complete"] is True


def test_ready_reports_real_engine_service_driver_failure() -> None:
    driver = _FailingDriver()
    service = EngineService(driver, idle_wait_seconds=0.0)
    engine = _ServiceEngine(service, snapshot=_healthy_snapshot)
    app = create_app(_config(), llm=engine)
    client = TestClient(app)
    try:
        assert client.get("/ready").status_code == 200

        request = GenerationRequest(
            prompts=("hello",),
            max_tokens=2,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
        )
        handle = service.submit_child(request)
        with pytest.raises(RuntimeError, match="driver tick failed"):
            handle.result(timeout=5.0)
        service.close()

        failed = client.get("/ready")
    finally:
        service.close()

    assert failed.status_code == 503
    body = failed.json()
    assert body["ready"] is False
    assert body["status"] == "unavailable"
    assert body["service"]["failure"] == "RuntimeError"
    assert body["service"]["accepting_requests"] is False
    assert any("engine service failed: RuntimeError" in item for item in body["diagnostics"])
