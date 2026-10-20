"""A suspended layer iterator preserves dispatch context and prompt identity."""
from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from hipengine.runtime.gguf_linear import raw_k_prefill_rowbatch, raw_k_prefill_rowbatch_session


def _runner(monkeypatch):
    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    runner.calls = []

    def steps(tokens, **kwargs):
        with raw_k_prefill_rowbatch_session(32):
            for layer in range(3):
                runner.calls.append((layer, raw_k_prefill_rowbatch()))
                yield None
            return SimpleNamespace(token_id=7)

    runner._prefill_chunked_steps = steps
    clock = iter((0, 1, 2, 3, 4, 5, 6, 7))
    monkeypatch.setattr('hipengine.runtime.qwen4_exp_runner.time.perf_counter', lambda: next(clock))
    return runner


def test_prefill_resume_isolates_dispatch_context_and_returns_only_final_result(monkeypatch):
    runner = _runner(monkeypatch)
    with raw_k_prefill_rowbatch_session(0):
        assert runner.prefill_append_resumable((1, 2)) is None
        assert raw_k_prefill_rowbatch() == 0
        assert runner.prefill_append_resumable((1, 2)) is None
        assert raw_k_prefill_rowbatch() == 0
        assert runner.prefill_append_resumable((1, 2)) is None
        result = runner.prefill_append_resumable((1, 2))
        assert result.token_id == 7
        assert raw_k_prefill_rowbatch() == 0
    assert runner.calls == [(0, 32), (1, 32), (2, 32)]
    assert runner._pending_prefill_append is None


def test_prefill_resume_rejects_a_different_chunk_before_advancing(monkeypatch):
    runner = _runner(monkeypatch)
    assert runner.prefill_append_resumable((1, 2)) is None
    with pytest.raises(ValueError, match='same prompt chunk'):
        runner.prefill_append_resumable((1, 3))
    assert runner.calls == [(0, 32)]
    context, steps, _ = runner._pending_prefill_append
    context.run(steps.close)


@pytest.mark.parametrize('length, expected_segments', [(37, 2), (256, 2), (257, 4), (509, 4)])
def test_default_share_tracks_narrow_and_dense_dispatch_cost(monkeypatch, length, expected_segments):
    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    clock = [0.0]
    calls = []

    def steps(tokens, **kwargs):
        for layer in range(8):
            calls.append(layer)
            clock[0] += 0.04
            yield None
        return SimpleNamespace(token_id=7)

    runner._prefill_chunked_steps = steps
    monkeypatch.setattr('hipengine.runtime.qwen4_exp_runner.time.perf_counter', lambda: clock[0])
    try:
        assert runner.prefill_append_resumable((1,) * length) is None
        assert len(calls) == expected_segments
    finally:
        context, pending, _ = runner._pending_prefill_append
        context.run(pending.close)


@pytest.mark.parametrize('budget', [0, -1, float('inf'), float('nan')])
def test_invalid_time_budget_does_not_start_prompt_work(monkeypatch, budget):
    runner = _runner(monkeypatch)
    with pytest.raises(ValueError, match='finite and positive'):
        runner.prefill_append_resumable((1, 2), time_budget_seconds=budget)
    assert runner.calls == []
    assert getattr(runner, '_pending_prefill_append', None) is None


def test_prefill_finalization_yields_even_with_time_left(monkeypatch):
    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    stages = []

    def steps(tokens, **kwargs):
        stages.append('layers')
        yield 'prefill_layers_complete'
        stages.append('head')
        return SimpleNamespace(token_id=7)

    runner._prefill_chunked_steps = steps
    monkeypatch.setattr('hipengine.runtime.qwen4_exp_runner.time.perf_counter', lambda: 0.0)
    assert runner.prefill_append_resumable((1, 2)) is None
    assert stages == ['layers']
    assert runner.prefill_append_resumable((1, 2)).token_id == 7
    assert stages == ['layers', 'head']


def test_closed_runner_cannot_start_resumable_work(monkeypatch):
    runner = _runner(monkeypatch)
    runner.closed = True
    with pytest.raises(RuntimeError, match='closed'):
        runner.prefill_append_resumable((1, 2))
    assert runner.calls == []
