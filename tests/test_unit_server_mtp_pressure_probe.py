"""Probe responsiveness with a real EngineService holding its driver thread.

The acceptance asks that while engine work holds the GPU, a concurrent
``/health`` or ``/ready`` returns within a bounded time.

The existing responsiveness test
(``test_integration_server_api.py::test_busy_snapshot_does_not_block_other_http_requests``)
injects a fake engine whose ``live_loop_snapshot`` blocks on a worker thread.
That proves the event loop stays free, but it bypasses the engine service
entirely: there is no driver thread and no command budget, so it cannot show
what bounds ``/ready`` or how the endpoints behave once the budget expires.

This module drives the **real** ``EngineService`` with a driver whose control
method holds the driver thread, and puts it behind the real ASGI app. That
reproduces the shape of the pressure case -- a long-running piece of engine work
occupying the sole driver thread while probes are polled -- without a GPU.

What it establishes:

- ``/health`` answers while the driver is held, and does not wait for it;
- ``/ready`` is bounded by the engine-service command budget and then degrades
  to a verdict without live snapshot fields, rather than failing or hanging;
- ``engine_service_health`` is a direct field read and never queues on the
  driver, so only the snapshot read is exposed to the pressure.
"""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from hipengine.generation.engine_service import EngineService
from hipengine.server import ServerConfig, create_app

# Short enough to keep the test fast, long enough that a genuine wait is
# distinguishable from scheduling noise.
_BUDGET_SECONDS = 0.6


class _HoldingDriver:
    """A driver that occupies its thread inside the snapshot control method."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def poll(self, *, max_ticks: int = 1):
        del max_ticks
        return ()

    def live_loop_snapshot(self) -> dict[str, object]:
        self.entered.set()
        self.release.wait(timeout=30.0)
        return {
            "runner": {"kv_pool": {"current_bytes": 1024}},
            "engine_service": {
                "sole_driver": True,
                "speculative_routes": {
                    "engine_service_verify_chain": 3,
                    "legacy_prelaunch_fallback": 0,
                },
                "last_speculative_work_kind": "specdec2_verify_chain",
            },
        }


class _ServiceEngine:
    """The engine surface the readiness path uses, backed by a real service."""

    def __init__(self, service: EngineService) -> None:
        self.service = service

    def live_loop_snapshot(self) -> dict[str, object]:
        return self.service.live_loop_snapshot()

    def engine_service_health(self) -> dict[str, object]:
        return self.service.health()


@pytest.fixture()
def held_service():
    driver = _HoldingDriver()
    service = EngineService(
        driver,
        command_timeout_seconds=_BUDGET_SECONDS,
        slow_command_warn_seconds=0.0,
    )
    try:
        yield driver, service
    finally:
        # ``close`` enqueues its own command and waits on the same budget, so
        # the driver must be released before the service is shut down.
        driver.release.set()
        service.close()


def test_health_answers_while_the_driver_thread_is_held(held_service) -> None:
    """``/health`` must not wait for the driver, and must not read the engine."""

    driver, service = held_service
    app = create_app(
        ServerConfig(model="fake", served_model_name="fake-model", eager_load=False),
        llm=_ServiceEngine(service),
    )

    async def run() -> tuple[float, httpx.Response, httpx.Response]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Start the probe that will block on the driver thread.
            pending = asyncio.create_task(client.get("/ready"))
            assert await asyncio.to_thread(driver.entered.wait, 5.0), (
                "the readiness read never reached the driver"
            )
            assert not pending.done(), "/ready finished before the driver was held"

            started = time.monotonic()
            health = await client.get("/health")
            elapsed = time.monotonic() - started

            driver.release.set()
            ready = await pending
            return elapsed, health, ready

    elapsed, health, ready = asyncio.run(run())

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    # The bound is the command budget; /health must not approach it.
    assert elapsed < _BUDGET_SECONDS, (
        f"/health waited on the driver thread for {elapsed:.3f}s"
    )
    assert ready.status_code == 200


def test_ready_is_bounded_by_the_command_budget_and_degrades(held_service) -> None:
    """A driver that never releases times the snapshot read out, not the probe.

    ``/ready`` still answers with its own verdict; it simply carries no live
    snapshot. That is the difference between a slow diagnostic read and a
    failing instance.
    """

    driver, service = held_service
    app = create_app(
        ServerConfig(model="fake", served_model_name="fake-model", eager_load=False),
        llm=_ServiceEngine(service),
    )

    async def run() -> tuple[float, httpx.Response]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            started = time.monotonic()
            response = await client.get("/ready")
            return time.monotonic() - started, response

    # The driver is never released, so the snapshot read can only time out.
    try:
        elapsed, response = asyncio.run(run())
    finally:
        driver.release.set()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ready"] is True
    assert "engine_unavailable" not in response.text
    # Bounded by the budget, not unbounded: it gives up rather than hanging.
    assert elapsed >= _BUDGET_SECONDS, elapsed
    assert elapsed < _BUDGET_SECONDS + 3.0, f"not bounded by the budget: {elapsed:.2f}s"


def test_engine_service_health_never_queues_on_the_driver(held_service) -> None:
    """``engine_service_health`` is a field read, so only the snapshot is exposed.

    This is why the pressure case costs ``/ready`` its diagnostics rather than
    its verdict: the service-health half of the payload is answered from
    ``_unhealthy``/``_closed`` without touching the driver thread.
    """

    driver, service = held_service
    engine = _ServiceEngine(service)

    holder = threading.Thread(target=service.live_loop_snapshot, daemon=True)
    holder.start()
    try:
        assert driver.entered.wait(5.0), "the driver never received the command"

        started = time.monotonic()
        health = engine.engine_service_health()
        elapsed = time.monotonic() - started

        assert health["status"] == "ok"
        assert health["serving"] is True
        assert elapsed < _BUDGET_SECONDS, (
            f"engine_service_health queued on the driver for {elapsed:.3f}s"
        )
    finally:
        driver.release.set()
        holder.join(5.0)
