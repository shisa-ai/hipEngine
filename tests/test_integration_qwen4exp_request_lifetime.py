"""Slow native work must retain service request ownership, not time out commands."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pytest
import threading
import time

from hipengine.generation.engine_service import EngineService
from hipengine.generation.deadline import GenerationCancellationToken, GenerationCancelled
from tests.test_unit_qwen4exp_mtp_submission import _driver, _request


@pytest.fixture(autouse=True)
def _request_owned_c1_path(monkeypatch):
    """These cases drive the request-owned c1 path with a stand-in resident.

    The shared Flash-Next decode batch is the default; its own suite is
    ``tests/test_unit_qwen4exp_ar_batch_decode.py``.
    """

    from tests._qwen4_exp_c1_control import select_request_owned_c1
    select_request_owned_c1(monkeypatch)


def test_mtp_slow_work_outlives_administrative_timeout():
    driver, target, _ = _driver()
    target.on_prefill = lambda: time.sleep(0.05)
    service = EngineService(driver, command_timeout_seconds=0.02)
    try:
        output = service.generate_speculative_mtp_detailed(_request())[0]
        assert output.generated_token_ids == tuple(range(10, 17))
        assert not service.closed
        assert service.service_health()["accepting_requests"] is True
    finally:
        service.close()
    assert service.service_health()["closed"] is True
    assert service.service_health()["driver_alive"] is False


def test_mtp_submit_queued_behind_slow_work_retains_ownership():
    driver, target, _ = _driver()
    entered, release = threading.Event(), threading.Event()

    def slow_prefill():
        entered.set()
        assert release.wait(timeout=2)

    target.on_prefill = slow_prefill
    service = EngineService(driver, command_timeout_seconds=0.02)
    pool = ThreadPoolExecutor(2)
    try:
        first = pool.submit(service.generate_speculative_mtp_detailed, _request())
        assert entered.wait(timeout=2)
        second = pool.submit(service.generate_speculative_mtp_detailed, _request())
        # Cross the old administrative deadline while a bounded GPU operation
        # owns the driver. Neither client has a request deadline.
        time.sleep(0.06)
        assert not second.done(), "queued submission escaped with an unowned request"
        release.set()
        assert first.result(timeout=2)[0].generated_token_ids == tuple(range(10, 17))
        assert second.result(timeout=2)[0].generated_token_ids == tuple(range(10, 17))
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        release.set()
        pool.shutdown(wait=True)
        service.close()


@pytest.mark.parametrize("retire", ["cancel", "deadline", "failure", "shutdown"])
def test_queued_work_has_terminal_owner_on_retirement(retire):
    driver, target, _ = _driver()
    entered, release = threading.Event(), threading.Event()

    def blocked_prefill():
        entered.set()
        assert release.wait(timeout=2)
        if retire == "failure":
            raise RuntimeError("fake device failure")

    target.on_prefill = blocked_prefill
    service = EngineService(driver, command_timeout_seconds=0.02)
    pool = ThreadPoolExecutor(3)
    token = GenerationCancellationToken()
    try:
        first = pool.submit(service.generate_speculative_mtp_detailed, _request())
        assert entered.wait(timeout=2)
        request = _request(cancellation_token=token)
        if retire == "deadline":
            request = replace(request, deadline_at=time.perf_counter() + 0.02)
        second = pool.submit(service.generate_speculative_mtp_detailed, request)
        if retire == "cancel":
            token.cancel()
        closer = pool.submit(service.close) if retire == "shutdown" else None
        time.sleep(0.06)
        assert not second.done()
        if closer is not None:
            assert not closer.done(), "shutdown returned while the driver still owned work"
            assert service.service_health()["accepting_requests"] is False
            assert service.service_health()["closing"] is True
        release.set()
        if retire in {"failure", "shutdown"}:
            with pytest.raises((RuntimeError, GenerationCancelled)):
                first.result(timeout=2)
            with pytest.raises((RuntimeError, GenerationCancelled)):
                second.result(timeout=2)
            if closer is not None:
                closer.result(timeout=2)
            if retire == "failure":
                assert service.service_health()["accepting_requests"] is False
                assert service.service_health()["failure"] == "RuntimeError"
        else:
            assert first.result(timeout=2)[0].generated_token_ids == tuple(range(10, 17))
            with pytest.raises(GenerationCancelled) as error:
                second.result(timeout=2)
            assert error.value.finish_details.deadline_exceeded == (retire == "deadline")
            assert target.chunks == [(1, 2), (3, 4)]
            clean = service.generate_speculative_mtp_detailed(_request())[0]
            assert clean.generated_token_ids == tuple(range(10, 17))
    finally:
        release.set()
        pool.shutdown(wait=True)
        service.close()
