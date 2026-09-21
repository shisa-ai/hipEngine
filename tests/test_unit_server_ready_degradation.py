"""A driver-thread timeout degrades ``/ready``; it does not make it a 503.

Companion to ``test_unit_server_ready_driver_bound.py``, which pins that
``/ready``'s engine read is bounded by the engine-service command budget. This
module pins what happens when that budget expires.

``readiness_payload`` reads the engine through ``_live_loop_snapshot`` and
``_engine_service_health``, and both swallow every exception and return ``None``.
So a command timeout on a busy driver thread costs the readiness payload its
live diagnostics -- it does **not** turn ``/ready`` into an ``engine_unavailable``
503. That is the right behaviour: an orchestrator polling readiness during a long
generation must not see the instance flap out of rotation for a transient busy
period, and a genuine service fault is reported through the service-health path
rather than through a timed-out probe.

``/health`` is stronger still: it returns a static dict and never touches the
engine, so it answers even when the engine read cannot.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from hipengine.generation.engine_service import EngineCommandTimeout
from hipengine.server import ServerConfig, create_app


class _BusyEngine:
    """An engine whose every readiness read times out on the driver thread."""

    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.health_calls = 0

    def live_loop_snapshot(self) -> dict[str, Any]:
        self.snapshot_calls += 1
        raise EngineCommandTimeout(
            "engine service command timed out: method=live_loop_snapshot "
            "budget_s=300.0 driver_busy=specdec2_verify_chain"
        )

    def engine_service_health(self) -> dict[str, Any]:
        self.health_calls += 1
        raise EngineCommandTimeout(
            "engine service command timed out: method=health budget_s=300.0"
        )


def _app(engine: _BusyEngine):
    return create_app(
        ServerConfig(
            model="fake",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=engine,
    )


@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_readiness_endpoints_answer_while_the_engine_read_times_out(path: str) -> None:
    """Neither endpoint propagates the engine read's timeout.

    ``/ready`` reports its own startup readiness, and a timed-out engine read
    only costs it the live snapshot fields. ``/health`` never calls the engine.
    """

    engine = _BusyEngine()
    app = _app(engine)

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(path)

    response = asyncio.run(run())

    assert response.status_code == 200, response.text
    body = response.json()
    assert "engine_unavailable" not in response.text
    if path == "/health":
        assert body["status"] == "ok"
        assert body["object"] == "hipengine.health"
        assert engine.snapshot_calls == 0, "/health must not read the engine"
    else:
        assert body["ready"] is True


def test_health_does_not_read_the_engine_at_all() -> None:
    """``/health`` is responsive by construction, which is the acceptance's floor.

    Its handler builds a static dict from the config, so no engine state -- and
    therefore no driver-thread contention -- can delay it.
    """

    engine = _BusyEngine()
    app = _app(engine)

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get("/health")

    response = asyncio.run(run())

    assert response.status_code == 200
    assert engine.snapshot_calls == 0
    assert engine.health_calls == 0
