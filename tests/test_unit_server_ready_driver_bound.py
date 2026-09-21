"""``/ready``'s engine read is bounded by the engine-service command budget.

The readiness checklist requires that a concurrent ``/health`` or ``/ready``
returns within a bounded time while engine work holds the GPU.

The two endpoints have very different bounds, and only one of them is tight:

- ``/health`` (``hipengine/server/api.py:7012``) returns a static dict and never
  touches the engine, so it is responsive by construction;
- ``/ready`` calls ``readiness_payload``, which reads the engine through
  ``asyncio.to_thread`` -- so the *event loop* stays responsive, but the request
  itself waits for the engine read.

That engine read is not a plain getter. It goes
``_live_loop_snapshot`` -> ``llm.live_loop_snapshot`` ->
``EngineService.live_loop_snapshot`` -> ``_control("live_loop_snapshot")``,
which **enqueues a command onto the sole driver thread** and waits. The wait is
bounded by ``_command_timeout_seconds``, whose default is
``DEFAULT_COMMAND_TIMEOUT_SECONDS = 300.0`` -- five minutes.

So the bound on ``/ready`` is the command budget, not a probe timeout. This
module pins that relationship so the bound is a known, configurable number
rather than an assumption.

The existing fake-engine responsiveness test
(``test_integration_server_api.py::test_busy_snapshot_does_not_block_other_http_requests``)
does not exercise this: its fake engine's ``live_loop_snapshot`` is called
directly on a worker thread, with no driver thread and no command budget, so it
proves event-loop responsiveness and says nothing about how long ``/ready``
itself can take.
"""

from __future__ import annotations

import threading
import time

import pytest

from hipengine.generation.engine_service import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    EngineCommandTimeout,
    EngineService,
)


class _BlockingDriver:
    """A driver whose control method holds the driver thread until released.

    The hold is an event rather than a bare sleep so a test can release the
    driver before ``close()``: ``close`` enqueues its own command and waits on
    the same budget, so a driver still sleeping would time the shutdown out.
    """

    def __init__(self, *, hold_seconds: float) -> None:
        self.hold_seconds = float(hold_seconds)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def poll(self, *, max_ticks: int = 1):
        del max_ticks
        return ()

    def live_loop_snapshot(self) -> dict[str, object]:
        self.calls += 1
        self.entered.set()
        self.release.wait(timeout=self.hold_seconds)
        return {"runner": {"kv_pool": {"current_bytes": 1024}}}


def _close(service: EngineService, driver: _BlockingDriver) -> None:
    """Release a held driver, then shut the service down."""

    driver.release.set()
    service.close()


def _service(driver, *, budget: float) -> EngineService:
    return EngineService(
        driver,
        command_timeout_seconds=budget,
        slow_command_warn_seconds=0.0,
    )


def test_ready_engine_read_is_bounded_by_the_command_budget() -> None:
    """A driver that outlasts the budget times the readiness read out.

    This is the bound the acceptance asks about. It is not an instant failure
    and not an unbounded wait: the read gives up at the budget.
    """

    driver = _BlockingDriver(hold_seconds=5.0)
    service = _service(driver, budget=0.4)
    try:
        started = time.monotonic()
        with pytest.raises(EngineCommandTimeout) as excinfo:
            service.live_loop_snapshot()
        elapsed = time.monotonic() - started

        assert driver.entered.is_set(), "the driver never received the command"
        assert elapsed >= 0.4, elapsed
        assert elapsed < 3.0, f"the budget did not bound the wait: {elapsed:.2f}s"
        assert "budget_s=0.4" in str(excinfo.value)
    finally:
        _close(service, driver)


def test_ready_engine_read_succeeds_when_the_driver_answers_in_budget() -> None:
    """The same call returns normally when the driver is quick enough."""

    driver = _BlockingDriver(hold_seconds=0.0)
    service = _service(driver, budget=5.0)
    try:
        snapshot = service.live_loop_snapshot()

        assert snapshot["runner"]["kv_pool"]["current_bytes"] == 1024
        assert driver.calls == 1
    finally:
        _close(service, driver)


def test_the_ready_bound_follows_the_configured_budget() -> None:
    """A larger budget waits longer; the bound is the budget, not a constant.

    Stated as its own case because the practical risk is a deployment that
    leaves the default in place: the probe then has a five-minute ceiling, which
    no orchestrator or load balancer treats as a healthy response time.
    """

    driver = _BlockingDriver(hold_seconds=10.0)
    service = _service(driver, budget=0.8)
    try:
        started = time.monotonic()
        with pytest.raises(EngineCommandTimeout):
            service.live_loop_snapshot()
        elapsed = time.monotonic() - started

        assert elapsed >= 0.8, elapsed
        assert elapsed < 4.0, elapsed
    finally:
        _close(service, driver)


def test_default_command_budget_is_five_minutes() -> None:
    """Pin the shipped default, because it is what bounds a readiness probe.

    ``/ready`` blocks up to this long when the driver thread is busy. A change
    to this constant changes a probe's worst-case latency, so it is asserted
    rather than left implicit.
    """

    assert DEFAULT_COMMAND_TIMEOUT_SECONDS == 300.0
