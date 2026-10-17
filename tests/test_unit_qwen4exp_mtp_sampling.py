"""Sampled MTP uses target draws, not draft/target argmax acceptance.

The prefix-sensitive CPU model is an exact oracle for host control/ownership;
real production verifier arithmetic has separate numerical/task gates.
"""
from collections import Counter
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.deadline import GenerationCancellationToken, GenerationCancelled
from hipengine.generation.qwen4_exp_mtp import Qwen4ExpMTPTextProvider
from hipengine.generation.sampling import RowSamplingState
from hipengine.speculative.registry import SpeculativeProviderConfig
from tests.test_unit_qwen4exp_sampling import generator, request
from tests.test_unit_qwen4exp_mtp_steps import _drain


class SerialTarget:
    max_sequence_length = 512
    runtime = None

    def __init__(self, *, constant=None):
        self.constant = constant
        self.history = []
        self.calls = []
        self.position = 0
        self.commits = []

    def scores(self):
        if self.constant is not None:
            return np.asarray(self.constant, dtype=np.float32)
        h = sum((i + 1) * token for i, token in enumerate(self.history))
        return np.array([.2 * (h % 5), .7 - .1 * (h % 3), -.3, -.7], dtype=np.float32)

    def result(self, capture_logits):
        scores = self.scores()
        return SimpleNamespace(token_id=int(np.argmax(scores)),
                               logits=scores if capture_logits else None,
                               hidden_seed=np.ones(8, dtype=np.float32),
                               hidden_seeds=np.ones((len(self.history), 8), dtype=np.float32))

    def prefill(self, ids, *, capture_logits=True, **kwargs):
        self.calls.append(('prefill', tuple(ids), capture_logits))
        self.history = list(ids)
        self.position = len(ids)
        return self.result(capture_logits)

    def step(self, token, *, capture_logits=True, **kwargs):
        self.calls.append(('step', token, capture_logits))
        self.history.append(int(token))
        self.position += 1
        return self.result(capture_logits)

    def close(self):
        pass


class Target(SerialTarget):
    def verify_target_block_state_rows(self, inputs, *, capture_logits=False):
        self.calls.append(('verify', tuple(inputs), capture_logits))
        self.pending = list(self.history)
        self.inputs = list(inputs)
        rows = []
        for token in inputs:
            self.history.append(int(token))
            rows.append(self.scores())
        self.position = len(self.history)
        return SimpleNamespace(token_ids=tuple(int(np.argmax(x)) for x in rows),
                               logits=tuple(rows) if capture_logits else (),
                               hidden_seeds=np.ones((len(rows), 8), dtype=np.float32))

    def commit_target_block_rows(self, count):
        self.commits.append(count)
        self.history = self.pending + self.inputs[:count]
        self.position = len(self.history)



class Draft:
    max_sequence_length = 512

    def __init__(self, candidate=1):
        self.candidate = candidate
        self.position = 0
        self.calls = []

    def prime_prompt(self, ids, hidden):
        self.calls.append(('prime', tuple(ids)))
        self.position = len(ids)

    def propose_chain(self, *, draft_n_max, **kwargs):
        self.calls.append(('propose', draft_n_max))
        self.position += draft_n_max
        return tuple(SimpleNamespace(token_id=self.candidate) for _ in range(draft_n_max))

    def forward(self, token, hidden, **kwargs):
        self.calls.append(('forward', token))
        self.position += 1

    def trim(self, position):
        self.calls.append(('trim', position))
        self.position = position


def provider(*, batched=True, constant=None, candidate=1):
    target = (Target if batched else SerialTarget)(constant=constant)
    model = generator(target)
    draft = Draft(candidate)
    obj = Qwen4ExpMTPTextProvider(target_generator=model,
        config=SpeculativeProviderConfig(provider='qwen4_exp_mtp', draft_model='/tmp/fake', candidate_budget=3),
        draft_runner=draft, draft_resident=object())
    return obj, target, draft


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('controls', [
    {}, {'top_k': 2}, {'top_p': .7}, {'min_p': .5},
    {'temperature': .8, 'repetition_penalty': 1.2, 'frequency_penalty': .2, 'presence_penalty': .3},
    {'logit_bias': {2: 1.5}, 'suppress_token_ids': (0,)},
    {'temperature': 0, 'frequency_penalty': .5},
    {'temperature': 0}, {'logprobs': False, 'top_logprobs': 0},
])
def test_sampled_and_processed_mtp_match_same_seed_ar_with_prefix_sensitive_logits(batched, controls):
    obj, target, draft = provider(batched=batched)
    req = request(max_tokens=17, **controls)
    expected = generator(Target()).generate_detailed(req)[0]
    steps, actual = _drain(obj.iter_generate(req.prompts[0], req))
    assert actual.generated_token_ids == expected.generated_token_ids
    assert actual.token_logprobs == expected.token_logprobs
    assert tuple(t for step in steps for t in step.token_logprobs) == actual.token_logprobs
    assert tuple(t for step in steps for t in step.token_ids) == actual.generated_token_ids
    assert target.history == list(req.prompts[0]) + list(actual.generated_token_ids[:-1])
    assert target.position == draft.position == len(target.history)
    assert all(call[2] for call in target.calls)
    assert actual.finish_details.sampler_mode == ('sampled_speculative_mtp' if req.temperature else 'processed_speculative_mtp')
    assert actual.telemetry.diagnostics['sampling_algorithm'] == 'target_draw_match_deterministic_draft'
    assert actual.telemetry.timing['mtp_generated_draft_tokens'] > 0


@pytest.mark.parametrize('candidate', [0, 1, 2])
def test_enumerated_target_distribution_is_independent_of_deterministic_proposal(monkeypatch, candidate):
    # Enumerate a uniform grid of three independent draws, not a flaky Monte
    # Carlo test. Each p=(.5,.3,.2) boundary lies on this grid. Full three-token
    # joint probabilities must match p(x1)p(x2)p(x3) for any draft token.
    counts = Counter()
    p = (.5, .3, .2)
    for grid in product(range(10), repeat=3):
        draws = iter((i + .5) / 10 for i in grid)
        monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(draws))
        obj, target, draft = provider(constant=np.log(p), candidate=candidate)
        req = request(max_tokens=3, top_logprobs=3)
        output = obj.generate_detailed(req)[0]
        with pytest.raises(StopIteration):
            next(draws)
        counts[output.generated_token_ids] += 1
        assert len(output.token_logprobs) == 3
        for token in output.token_logprobs:
            assert np.exp(token.logprob) == pytest.approx(p[token.token_id])
        assert target.position == draft.position == 3
    for ids in product(range(3), repeat=3):
        assert counts[ids] / 1000 == pytest.approx(np.prod([p[i] for i in ids]))


def test_bonus_mismatch_tail_and_k0_consume_exactly_one_draw_per_output(monkeypatch):
    draws = iter([.1, .1, .1, .9, .9, .1, .9])
    monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(draws))
    obj, target, draft = provider(constant=np.log([.6, .4]), candidate=0)
    req = request(max_tokens=7, top_logprobs=2)
    steps = obj.iter_generate(req.prompts[0], req)
    assert next(steps).token_ids == (0,)
    cycle = steps.send(2)
    assert cycle.token_ids == (0, 0, 1)  # accepted drafts plus sampled bonus
    assert cycle.cycle.accepted == 2 and cycle.cycle.mismatch_token is None
    assert steps.send(0).token_ids == (1,)
    tail = steps.send(1)
    assert tail.token_ids == (0, 1) and tail.finished
    with pytest.raises(StopIteration) as result:
        next(steps)
    assert result.value.value.generated_token_ids == (0, 0, 0, 1, 1, 0, 1)
    assert len(result.value.value.token_logprobs) == 7
    with pytest.raises(StopIteration):
        next(draws)
    assert target.position == draft.position == 7


def test_seed_repeat_change_and_explicit_row_seed_ownership():
    obj, target, draft = provider()
    req = request(max_tokens=19)
    first = obj.generate_detailed(req)[0]
    again = obj.generate_detailed(req)[0]
    changed = obj.generate_detailed(replace(req, seed=42))[0]
    assert first.generated_token_ids == again.generated_token_ids
    assert first.token_logprobs == again.token_logprobs
    assert first.generated_token_ids != changed.generated_token_ids
    pair = obj.generate_detailed(replace(req, prompts=((2,), (1,)), row_seeds=(99, 123)))
    single = obj.generate_detailed(replace(req, row_seeds=(123,)))[0]
    assert pair[1].generated_token_ids == single.generated_token_ids
    assert pair[1].token_logprobs == single.token_logprobs
    with pytest.raises(ValueError, match='row_seeds'):
        obj.generate_detailed(replace(req, row_seeds=(1, 2)))


def test_interleaved_and_cancelled_request_does_not_consume_neighbor_rng():
    obj, idle_target, idle_draft = provider()
    req = request(max_tokens=12)
    expected = obj.generate_detailed(req)[0]
    cancellation = GenerationCancellationToken()
    a = obj.iter_generate((1,), replace(req, cancellation_token=cancellation), target=Target(), draft=Draft())
    b = obj.iter_generate((1,), req, target=Target(), draft=Draft())
    a_root = next(a)
    b_root = next(b)
    assert a_root == b_root
    next(a)
    cancellation.cancel()
    with pytest.raises(GenerationCancelled):
        next(a)
    _, output = _drain(b)
    assert output.generated_token_ids == expected.generated_token_ids
    assert output.token_logprobs == expected.token_logprobs


def test_processed_prefill_requires_logits_and_zero_tokens_does_no_work():
    obj, target, draft = provider()
    req = request(max_tokens=0)
    output = obj.generate_detailed(req)[0]
    assert not output.generated_token_ids and not output.token_logprobs
    assert target.calls == draft.calls == []
    req = replace(req, max_tokens=4)
    prefill = target.prefill((1,), capture_logits=False)
    with pytest.raises(ValueError, match='logits'):
        next(obj.iter_generate((1,), req, prefill_result=prefill))
    assert draft.calls == []


def test_buffered_provider_stream_preserves_selected_logprobs():
    obj, _, _ = provider()
    req = request(max_tokens=8)
    expected = obj.generate_detailed(req)[0]
    chunks = list(obj.stream_detailed(req))
    assert ''.join(c.text for c in chunks) == expected.text
    assert tuple(t for c in chunks for t in c.token_logprobs) == expected.token_logprobs


@pytest.mark.parametrize('batched', [False, True])
def test_sampled_eos_stops_and_ignore_eos_continues(batched):
    obj, target, draft = provider(batched=batched, constant=[0, -100, -100, 100], candidate=3)
    obj.target_generator.tokenizer.eos_token_id = 3
    req = request(max_tokens=4)
    output = obj.generate_detailed(req)[0]
    assert output.generated_token_ids == (3,)
    assert output.finish_details.reason == 'eos'
    continued = obj.generate_detailed(replace(req, ignore_eos=True))[0]
    assert continued.generated_token_ids == (3,) * 4
    assert len(continued.token_logprobs) == 4
    assert target.position == draft.position == 4
