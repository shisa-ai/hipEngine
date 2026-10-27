"""MTP commits only through the first stop/EOS boundary, with no extra draws."""
from dataclasses import replace

import numpy as np
import pytest

from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.sampling import RowSamplingState
from tests.test_unit_qwen4exp_mtp_sampling import provider, Target
from tests.test_unit_qwen4exp_sampling import generator, request
from tests.test_unit_qwen4exp_mtp_steps import _drain
from tests.test_unit_qwen4exp_mtp_resident_sampling import driver
from tests.test_unit_qwen4exp_mtp_submission import _finish


@pytest.fixture(autouse=True)
def _request_owned_c1_path(monkeypatch):
    """These cases drive the request-owned c1 path with a stand-in resident.

    The shared Flash-Next decode batch is the default; its own suite is
    ``tests/test_unit_qwen4exp_ar_batch_decode.py``.
    """

    from tests._qwen4_exp_c1_control import select_request_owned_c1
    select_request_owned_c1(monkeypatch)


@pytest.mark.parametrize('at_budget', [False, True])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('candidate', [0, 1])
@pytest.mark.parametrize('prefix,controls,reason', [
    ((1,), {'stop_token_ids': (1,)}, 'stop'),
    ((0, 1), {'stop_token_ids': (1,)}, 'stop'),
    ((0, 0, 1), {'stop_token_sequences': ((0, 1),)}, 'stop'),
    ((0, 0, 0, 0, 1), {'stop_token_sequences': ((0, 1),)}, 'stop'),
    ((0, 0, 1), {'eos_token_id': 1}, 'eos'),
])
def test_provider_stops_at_root_accepted_correction_bonus_or_cross_cycle_boundary(
        monkeypatch, batched, candidate, prefix, controls, reason, at_budget):
    obj, target, draft = provider(batched=batched, constant=np.log([.6, .4]), candidate=candidate)
    draws = iter(.1 if t == 0 else .9 for t in prefix)
    monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(draws))
    steps, output = _drain(obj.iter_generate((1,), request(
        max_tokens=len(prefix) if at_budget else 12, top_logprobs=2, **controls)))
    assert output.generated_token_ids == prefix
    assert output.finish_details.reason == reason
    assert output.finish_details.stop_sequence == (() if reason == 'eos' else controls.get('stop_token_sequences', ((1,),))[0])
    assert tuple(t for s in steps for t in s.token_ids) == prefix
    assert len(output.token_logprobs) == len(prefix)  # raw provider output includes terminal tokens
    assert steps[-1].finished and not any(s.finished for s in steps[:-1])
    assert target.history == [1, *prefix[:-1]]
    assert target.position == draft.position == len(prefix)
    if len(prefix) > 1:
        assert output.telemetry.timing['mtp_generated_draft_tokens'] > 0
    with pytest.raises(StopIteration):
        next(draws)


def test_k0_step_preserves_a_stop_prefix_from_the_previous_cycle(monkeypatch):
    obj, target, draft = provider(constant=np.log([.6, .4]), candidate=0)
    draws = iter([.1, .9])
    monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(draws))
    req = request(max_tokens=12, top_logprobs=2, stop_token_sequences=((0, 1),))
    steps = obj.iter_generate((1,), req)
    assert next(steps).token_ids == (0,)
    terminal = steps.send(0)
    assert terminal.finished and terminal.token_ids == (1,) and terminal.cycle is None
    with pytest.raises(StopIteration) as done:
        next(steps)
    assert done.value.value.finish_details.stop_sequence == (0, 1)
    assert target.history == [1, 0] and target.position == draft.position == 2
    assert done.value.value.telemetry.timing['mtp_cycles_count'] == 0


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('controls', [
    {'eos_token_id': 1, 'min_tokens': 4},
    {'eos_token_id': 1, 'ignore_eos': True},
    {'stop_token_ids': (1,), 'min_tokens': 4, 'eos_token_id': 2},
    {'stop_token_sequences': ((0, 1),), 'min_tokens': 4, 'eos_token_id': 2},
    {'eos_token_id': 1, 'stop_token_ids': (1,), 'ignore_eos': True},
])
def test_minimum_and_ignore_eos_follow_ar_policy(batched, controls):
    scores = [0, 1, -1]
    obj, target, draft = provider(batched=batched, constant=scores, candidate=1)
    req = request(max_tokens=8, top_logprobs=2, **controls)
    expected = generator(Target(constant=scores)).generate_detailed(req)[0]
    actual = obj.generate_detailed(req)[0]
    assert actual.generated_token_ids == expected.generated_token_ids
    assert actual.token_logprobs == expected.token_logprobs
    assert replace(actual.finish_details, sampler_mode=None) == replace(expected.finish_details, sampler_mode=None)
    assert target.history == [1, *actual.generated_token_ids[:-1]]
    assert target.position == draft.position


@pytest.mark.parametrize('prefix,controls,visible,reason', [
    ((1,), {'stop_token_ids': (1,)}, (), 'stop'),
    ((0, 0, 1), {'stop_token_sequences': ((0, 1),)}, (0,), 'stop'),
    ((0, 0, 0, 0, 1), {'stop_token_sequences': ((0, 1),)}, (0, 0, 0), 'stop'),
    ((0, 0, 1), {'eos_token_id': 1}, (0, 0), 'eos'),
])
@pytest.mark.parametrize('candidate', [0, 1])
def test_resident_stops_filter_only_visible_logs_and_preserve_committed_ids(
        monkeypatch, prefix, controls, visible, reason, candidate):
    api, target, draft = driver(monkeypatch)
    target.constant = np.log([.6, .4])
    draft.candidate = candidate
    draws = iter(.1 if t == 0 else .9 for t in prefix)
    monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(draws))
    try:
        output, events = _finish(api, api.submit_speculative_detailed(
            request(max_tokens=12, top_logprobs=2, **controls)))
        assert output.generated_token_ids == prefix
        assert output.text == api.inner.tokenizer.decode(visible)
        assert tuple(t.token_id for t in output.token_logprobs) == visible
        assert tuple(e.token_id for e in events if e.kind == 'token') == prefix
        assert output.finish_details.reason == reason
        assert target.history == [1, *prefix[:-1]]
        assert target.position == draft.position == len(prefix)
        if len(prefix) > 1:
            assert output.telemetry.timing['mtp_generated_draft_tokens'] > 0
        with pytest.raises(StopIteration):
            next(draws)
    finally:
        api.close()


def test_cancellation_after_stop_selection_restores_dfa_and_committed_output(monkeypatch):
    api, target, draft = driver(monkeypatch)
    target.constant = np.log([.6, .4])
    draft.candidate = 0
    cancellation = GenerationCancellationToken()
    draws = iter([.1, .9])
    monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(draws))
    try:
        sub = api.submit_speculative_detailed(request(max_tokens=12, top_logprobs=2,
            stop_token_sequences=((0, 1),), cancellation_token=cancellation))
        api.poll(max_ticks=1)
        api.poll(max_ticks=1)  # Commit root 0, a partial stop sequence.
        row = next(iter(api._runner._rows.values()))
        snapshot = row.sampling_state.clone()
        original = api.inner._controlled_token
        def cancel_after_stop(*args, **kwargs):
            token = original(*args, **kwargs)
            assert token == 1
            cancellation.cancel()
            return token
        monkeypatch.setattr(api.inner, '_controlled_token', cancel_after_stop)
        output, events = _finish(api, sub)
        assert output.finish_details.cancelled
        assert output.generated_token_ids == (0,)
        assert tuple(t.token_id for t in output.token_logprobs) == (0,)
        assert not [e for e in events if e.kind == 'token']
        assert row.sampling_state.stop_suffix_state == snapshot.stop_suffix_state
        assert row.sampling_state.step_index == snapshot.step_index == 1
        assert row.sampling_state.generated_tokens == snapshot.generated_tokens
        assert target.transactions[-1].state == 'rolled_back'
        assert target.history == [1] and target.position == draft.position == 1
    finally:
        api.close()
