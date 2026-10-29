"""Timing intervals are not host/device attribution; patches have bounded lifetime."""
import ctypes
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_decode_host_bound_probe as probe


def test_timing_fields_do_not_infer_host_bound():
    for skipped in (False, True):
        fields = probe._timing_fields(20, 0, 30, 2, skipped)
        assert fields['submission_loop_ms_per_step'] == 10
        assert fields['trailing_sync_wait_ms'] == 0
        assert fields['loop_wall_ms_per_step'] == 10
        assert 'host_bound' not in fields
        assert 'host_enqueue_ms_total' not in fields


def test_skip_only_token_buffer_and_restore_after_exception():
    calls = []
    def original(*a, **kw):
        calls.append((a, kw))
    module = SimpleNamespace(copy_device_to_host=original)
    runner = SimpleNamespace(token_id_buffer=SimpleNamespace(ptr=12, nbytes=8))
    host = ctypes.c_int64(99)
    with pytest.raises(RuntimeError):
        with probe._token_readback_scope(runner, True, module=module):
            module.copy_device_to_host(ctypes.addressof(host), runner.token_id_buffer)
            assert host.value == 0
            module.copy_device_to_host(ctypes.addressof(host), SimpleNamespace(ptr=13, nbytes=8))
            module.copy_device_to_host(ctypes.addressof(host), runner.token_id_buffer, 4)
            raise RuntimeError('injected')
    assert module.copy_device_to_host is original
    assert len(calls) == 2


def test_step_requests_compact_device_token_path():
    calls = []
    runner = SimpleNamespace(step=lambda *a, **kw: (calls.append((a, kw)) or SimpleNamespace(token_id=7)))
    assert probe._step(runner, 99) == 7
    assert calls == [((99,), dict(capture_logits=False, capture_target_hidden=False, token_id_resident=True))]


def test_generator_closed_on_prefill_failure():
    closed=[]
    def fail(*a, **kw):
        raise RuntimeError('prefill')
    generator=SimpleNamespace(runner=SimpleNamespace(runtime=object(), prefill=fail), close=lambda:closed.append(True))
    with pytest.raises(RuntimeError, match='prefill'):
        probe._measure_case(generator, [1], SimpleNamespace(), 2)
    assert closed == [True]


@pytest.mark.parametrize('skip', [False, True])
def test_measurement_runs_compact_loops_and_closes(monkeypatch, skip):
    from contextlib import contextmanager
    events=[]
    @contextmanager
    def scope(runner, enabled):
        events.append(('enter', enabled))
        try:
            yield
        finally:
            events.append(('exit', enabled))
    monkeypatch.setattr(probe, '_token_readback_scope', scope)
    ticks=iter([0, 20_000_000, 24_000_000, 30_000_000, 60_000_000])
    monkeypatch.setattr(probe.time, 'perf_counter_ns', lambda: next(ticks))
    def step(token, **kw):
        assert kw == dict(capture_logits=False, capture_target_hidden=False, token_id_resident=True)
        events.append(('step', token))
        return SimpleNamespace(token_id=token+1)
    runner=SimpleNamespace(runtime=SimpleNamespace(device_synchronize=lambda: events.append(('sync',))),
                           prefill=lambda *a, **kw: SimpleNamespace(token_id=1), step=step)
    generator=SimpleNamespace(runner=runner, close=lambda: events.append(('close',)))
    args=SimpleNamespace(warm_steps=1, host_profile=False, skip_token_readback=skip)
    result=probe._measure_case(generator, [1,2], args, 2)
    assert result['loop_wall_ms_per_step'] == 12
    assert result['sync_each_ms_per_step'] == 15
    assert events[-1] == ('close',)
    assert events.count(('sync',)) == 4
    assert ('enter',skip) in events and ('exit',skip) in events
