"""A suspended layer iterator preserves dispatch context and prompt identity."""
import contextvars
from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from hipengine.runtime.gguf_linear import (
    _q8_mmq_prefill_session,
    q8_mmq_lane_selection,
    q8_mmq_lane_session,
    q8_mmq_prefill_session,
    raw_k_prefill_rowbatch,
    raw_k_prefill_rowbatch_session,
    resolve_q8_mmq_prefill_policy,
    wide_f16_activation_session,
    wide_f16_activation_workspace,
    wide_f16_pair_enabled,
    wide_f16_pair_session,
)


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


def _lane_state():
    """What the caller's own context sees for each prefill-owned lane."""

    lane = q8_mmq_lane_selection()
    return (
        _q8_mmq_prefill_session.get() is not None,
        (lane.token64, lane.raw_vector, lane.vec4),
        wide_f16_activation_workspace() is not None,
        wide_f16_pair_enabled(),
    )


def test_a_suspended_prefill_owns_no_dispatch_lane_in_the_caller():
    """The saved Context, not the caller's, holds MMQ/lane/pair while suspended.

    ``_prefill_chunked_steps`` yields from inside ``_q8_mmq_prefill_context`` and
    ``_wide_f16_activation_context``. Without the copied Context each resume
    would leave those ContextVars set for whatever the scheduler runs next,
    pointing another request's Q8_0 launches at this request's workspace and
    token64 policy.
    """

    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    policy = resolve_q8_mmq_prefill_policy("gguf_ud_q4_k_xl")
    inside = []

    def steps(tokens, **kwargs):
        with (
            q8_mmq_prefill_session(
                workspace_ptr=1 << 32, workspace_nbytes=1 << 30,
                risk_count_ptr=2 << 32, risk_count_nbytes=4,
                risk_indices_ptr=3 << 32, risk_indices_nbytes=1 << 30,
                policy=policy,
            ),
            q8_mmq_lane_session(token64=True, raw_vector=True, vec4=True),
            wide_f16_activation_session(
                True, workspace_ptr=4 << 32, workspace_nbytes=1 << 20),
            wide_f16_pair_session(True),
        ):
            for _ in range(2):
                inside.append(_lane_state())
                yield None
            return SimpleNamespace(token_id=7)

    runner._prefill_chunked_steps = steps
    clean = (False, (False, False, False), False, False)
    assert _lane_state() == clean
    assert runner.prefill_append_resumable((1, 2), time_budget_seconds=1e-9) is None
    assert _lane_state() == clean
    context, pending, _ = runner._pending_prefill_append
    assert isinstance(context, contextvars.Context)
    assert runner.prefill_append_resumable((1, 2), time_budget_seconds=1e-9) is None
    assert _lane_state() == clean
    result = runner.prefill_append_resumable((1, 2), time_budget_seconds=1e-9)
    assert result.token_id == 7
    assert _lane_state() == clean
    assert inside == [
        (True, (True, True, True), True, True),
        (True, (True, True, True), True, True),
    ]
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
