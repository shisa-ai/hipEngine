"""Committed Flash-Next MTP output must be visible before the next cycle ends."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from dataclasses import replace

import numpy as np
import pytest

from hipengine.generation.sampling import RowSamplingState
from hipengine.generation.deadline import GenerationCancellationToken

from hipengine.generation.engine_service import EngineService
from tests.test_unit_qwen4exp_mtp_resident_sampling import driver
from tests.test_unit_qwen4exp_mtp_submission import _finish
from tests.test_unit_qwen4exp_sampling import request


def test_logprob_stream_capability_is_resident_owned(monkeypatch):
    api, _, _ = driver(monkeypatch)
    try:
        assert api.supports_stream_token_logprobs
        assert not getattr(api.inner, 'supports_stream_token_logprobs', False)
        monkeypatch.setattr(api._runner, 'supports_stream_token_logprobs', False)
        assert not api.supports_stream_token_logprobs
    finally:
        api.close()


def test_committed_cycle_events_carry_text_and_target_logprobs(monkeypatch):
    api, target, _ = driver(monkeypatch)
    req = request(max_tokens=15)
    try:
        output, events = _finish(api, api.submit_speculative_detailed(req))
        chunks = [e.stream_chunk for e in events if e.kind == 'token']
        assert any(call[0] == 'verify' for call in target.calls)
        assert ''.join(c.text for c in chunks) == output.text
        assert tuple(p for c in chunks for p in c.token_logprobs) == output.token_logprobs
    finally:
        api.close()


@pytest.mark.parametrize('control', ['length', 'stop', 'eos', 'ignore_eos', 'min_tokens'])
def test_service_stream_matches_terminal_text_logs_and_ids(monkeypatch, control):
    api, target, _ = driver(monkeypatch)
    target.constant = np.zeros(4)
    req = request(max_tokens=12, temperature=0)
    if control == 'stop':
        req = replace(req, stop_token_sequences=((0, 0, 0),))
    elif control == 'eos':
        req = replace(req, eos_token_id=0)
    elif control == 'ignore_eos':
        req = replace(req, eos_token_id=0, ignore_eos=True)
    elif control == 'min_tokens':
        req = replace(req, eos_token_id=0, min_tokens=4)
    service = EngineService(api, idle_wait_seconds=0)
    try:
        handle = service.submit_speculative_child(req, streaming=True)
        chunks = list(handle.iter_chunks())
        output = handle.result()
        assert ''.join(c.text for c in chunks) == output.text
        assert tuple(p for c in chunks for p in c.token_logprobs) == output.token_logprobs
        assert chunks[-1].generated_token_ids == output.generated_token_ids
        assert sum(c.finish_details is not None for c in chunks) == 1
    finally:
        service.close()


@pytest.mark.parametrize('max_tokens', [1, 2, 3, 4, 8])
def test_utf8_byte_fallback_across_root_and_committed_cycles(monkeypatch, max_tokens):
    api, target, _ = driver(monkeypatch)
    target.constant = np.zeros(4)
    draws = iter([.1, .3, .6, .9] * 3)
    monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(draws))
    parts = (b'\xe3', b'\x81', b'\x82', b'X')
    monkeypatch.setattr(api.inner.tokenizer, 'decode',
                        lambda ids, **kw: b''.join(parts[i] for i in ids).decode('utf-8', errors='replace'))
    service = EngineService(api, idle_wait_seconds=0)
    try:
        handle = service.submit_speculative_child(request(max_tokens=max_tokens), streaming=True)
        chunks = list(handle.iter_chunks())
        output = handle.result()
        assert ''.join(c.text for c in chunks) == output.text
        assert tuple(p for c in chunks for p in c.token_logprobs) == output.token_logprobs
        if max_tokens >= 3:
            assert '\ufffd' not in ''.join(c.text for c in chunks)
    finally:
        service.close()



@pytest.mark.parametrize('failure', ['disconnect', 'error'])
def test_interrupted_verification_publishes_no_candidates_and_recovers(monkeypatch, failure):
    api, target, draft = driver(monkeypatch)
    token = GenerationCancellationToken()
    def interrupt():
        if failure == 'disconnect':
            token.cancel()
        else:
            raise RuntimeError('injected verifier failure')
    target.on_verify = interrupt
    service = EngineService(api, idle_wait_seconds=0)
    seen = []
    try:
        stream = service.stream_speculative_detailed(request(max_tokens=15, cancellation_token=token))
        if failure == 'error':
            with pytest.raises(RuntimeError, match='injected verifier failure'):
                for chunk in stream:
                    seen.append(chunk)
        else:
            seen = list(stream)
            assert seen[-1].finish_details.cancelled
            assert len(seen[-1].generated_token_ids) == 1
        assert sum(len(c.token_logprobs) for c in seen) <= 1
        assert all(tx.state != 'open' for tx in target.transactions)
        assert target.position == draft.position
        if failure == 'disconnect':
            target.on_verify = lambda: None
            chunks = list(service.stream_speculative_detailed(request(max_tokens=15)))
            assert chunks[-1].finish_details.reason == 'length'
            assert chunks[-1].telemetry.timing['mtp_cycles_count'] > 0
        else:
            assert service.closed  # Unknown driver failures fail closed.
    finally:
        service.close()


def test_service_publishes_root_before_verification_finishes(monkeypatch):
    api, target, _ = driver(monkeypatch)
    release = Event()
    entered = Event()

    def pause_verify():
        entered.set()
        assert release.wait(10), 'test did not release verifier'

    target.on_verify = pause_verify
    service = EngineService(api, idle_wait_seconds=0)
    stream = service.stream_speculative_detailed(request(max_tokens=15))
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        first = pool.submit(next, stream)
        assert entered.wait(5), 'test must reach a real MTP verification cycle'
        chunk = first.result(timeout=1)
        assert chunk.text
        assert chunk.finish_details is None
        release.set()
        rest = list(stream)
        assert rest[-1].finish_details is not None
        assert rest[-1].telemetry.timing['mtp_cycles_count'] > 0
    finally:
        release.set()
        pool.shutdown(wait=True)
        stream.close()
        service.close()
