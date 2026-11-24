"""Control requests must not wait for unrelated device work or become ghosts."""
from __future__ import annotations

import threading
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest

from hipengine.generation import (
    EngineService, EngineServiceClosed,
    GenerationCancellationToken, GenerationCancelled,
)
from tests.test_integration_generation_engine_service import (
    _BlockingTickDriver, _FakeSoleDriver, _request,
)


class _FrontendDriver(_BlockingTickDriver):
    # Only adapters with immutable CPU-only tokenization may opt in.
    supports_thread_safe_tokenization = True

    def count_tokens(self, text):
        return len(self.tokenize(text))

    def detokenize(self, tokens, **kwargs):
        return str(tuple(tokens))


@pytest.mark.parametrize('method,args,expected', [
    ('tokenize', ('hello',), (5,)),
    ('count_tokens', ('hello',), 1),
    ('detokenize', ((1, 2),), '(1, 2)'),
])
def test_frontend_tokenization_remains_available_during_a_device_tick(method, args, expected):
    driver = _FrontendDriver()
    service = EngineService(driver, command_timeout_seconds=.1)
    try:
        service.submit_child(_request('prompt:4'))
        assert driver.tick_entered.wait(2)
        assert getattr(service, method)(*args) == expected
        assert not driver.tick_release.is_set()
    finally:
        driver.tick_release.set()
        service.close()


def test_handle_cancel_signals_intent_during_tick_before_driver_reclamation():
    class Driver(_FakeSoleDriver):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.saw_intent = threading.Event()
            self.reclaim_allowed = threading.Event()
            self.abort_threads = []

        def poll(self, *, max_ticks=1):
            state = next(iter(self._active.values()))
            token = state.submission.request.cancellation_token
            self.entered.set()
            # A real runtime checks intent at existing kernel/chunk boundaries.
            assert token is not None
            assert token._event.wait(1) or token.cancel_requested
            self.saw_intent.set()
            assert self.reclaim_allowed.wait(2)
            return ()

        def abort_submission(self, submission, *, reason='cancel'):
            self.abort_threads.append(threading.get_ident())
            super().abort_submission(submission, reason=reason)

    driver = Driver()
    service = EngineService(driver, command_timeout_seconds=3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            handle = service.submit_child(_request('prompt:4'))
            assert driver.entered.wait(2)
            cancelling = pool.submit(handle.cancel, reason='disconnect')
            assert driver.saw_intent.wait(2)
            assert not handle.done  # Intent must not publish completion before reclaim.
            driver.reclaim_allowed.set()
            cancelling.result(timeout=2)
            with pytest.raises(GenerationCancelled):
                handle.result(timeout=2)
            assert driver.abort_threads == [service.driver_thread_id]
            # A fresh request can be admitted after the cancelled row retires.
            driver.poll = lambda **kwargs: _FakeSoleDriver.poll(driver, **kwargs)
            assert service.submit_child(_request('retry:1')).result(2).text
        finally:
            driver.reclaim_allowed.set()
            service.close()


@pytest.mark.parametrize('method,args', [
    ('tokenize', ('hello',)), ('count_tokens', ('hello',)), ('detokenize', ((1,),)),
])
def test_frontend_controls_reject_work_after_close(method, args):
    service = EngineService(_FrontendDriver())
    service.close()
    with pytest.raises(EngineServiceClosed):
        getattr(service, method)(*args)


def test_explicit_child_cancel_does_not_cancel_a_shared_parent_token():
    token = GenerationCancellationToken()
    driver = _BlockingTickDriver()
    service = EngineService(driver, command_timeout_seconds=3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            first, second = service.submit_children((
                _request('first:4', cancellation_token=token),
                _request('second:4', cancellation_token=token),
            ))
            assert driver.tick_entered.wait(2)
            cancelling = pool.submit(first.cancel)
            driver.tick_release.set()
            cancelling.result(2)
            with pytest.raises(GenerationCancelled):
                first.result(2)
            assert not token.cancel_requested
            driver.poll = lambda **kwargs: _FakeSoleDriver.poll(driver, **kwargs)
            assert second.result(2).text
        finally:
            driver.tick_release.set()
            service.close()


@pytest.mark.parametrize('batch', [False, True])
def test_partial_binding_failure_reclaims_bound_and_unbound_submissions(monkeypatch, batch):
    class Driver(_FakeSoleDriver):
        def submit_speculative_many_detailed(self, requests):
            return tuple(replace(self.submit_detailed(request), work_kind='verify_chain')
                         for request in requests)

    driver = Driver()
    service = EngineService(driver)
    bind = service._bind_child_submission

    def fail_second(state, submission, **kwargs):
        if submission.request_ids == (2,):
            raise ValueError('injected bind failure')
        return bind(state, submission, **kwargs)

    monkeypatch.setattr(service, '_bind_child_submission', fail_second)
    try:
        submit = service.submit_speculative_children if batch else service.submit_children
        with pytest.raises(ValueError, match='injected bind failure'):
            submit((_request('first:1'), _request('second:1')))
        assert driver.release_order == [1, 2]
        assert not driver._active
        assert not service._states_by_backend_id and not service._states_by_service_id
        assert service.submit_child(_request('retry:1')).result(2).text
    finally:
        service.close()
