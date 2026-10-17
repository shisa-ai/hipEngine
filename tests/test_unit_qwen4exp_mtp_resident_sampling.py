"""Resident sampled MTP owns RNG, history and logprobs inside transactions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.sampling import RowSamplingState

from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_mtp import Qwen4ExpMTPTextProvider
from hipengine.speculative.registry import SpeculativeProviderConfig
from tests.test_unit_qwen4exp_mtp_sampling import Target, Draft
from tests.test_unit_qwen4exp_mtp_submission import _finish
from tests.test_unit_qwen4exp_sampling import generator, request


class ResidentTarget(Target):
    prefill_chunk_size = 2
    config = SimpleNamespace()

    def __init__(self):
        super().__init__()
        self.transactions = []
        self.on_verify = lambda: None
        self.on_step = lambda: None

    def reset(self):
        self.history = []
        self.position = 0

    def prefill_append(self, ids, *, capture_logits=False, **kwargs):
        self.history.extend(ids)
        self.position = len(self.history)
        self.calls.append(('prefill_append', tuple(ids), capture_logits))
        result = self.result(capture_logits)
        result.hidden_seeds = result.hidden_seeds[-len(ids):]
        return result

    def verify_target_block_state_rows(self, *args, **kwargs):
        result = super().verify_target_block_state_rows(*args, **kwargs)
        self.on_verify()
        return result

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        self.on_step()
        return result

    def begin_device_transaction(self, **kwargs):
        tx = SimpleNamespace(position=self.position, history=list(self.history), state='open')
        self.transactions.append(tx)
        return tx

    def commit_device_transaction(self, tx):
        assert tx.state == 'open'
        tx.state = 'committed'

    def rollback_device_transaction(self, tx):
        assert tx.state == 'open'
        tx.state = 'rolled_back'
        self.position, self.history = tx.position, tx.history


class ResidentDraft(Draft):
    def reset(self):
        self.position = 0

    def prime_prompt_chunk(self, ids, hidden, *, check_cancelled=None, **kwargs):
        if check_cancelled:
            check_cancelled()
        self.position += len(ids)

    def snapshot(self):
        return SimpleNamespace(position=self.position)

    def restore(self, snapshot):
        self.position = snapshot.position

    def close(self):
        pass


def driver(monkeypatch, capacity=1):
    target, draft = ResidentTarget(), ResidentDraft()
    model = generator(target)
    model._resident = SimpleNamespace(close=lambda: None)
    model.attach_speculative_provider(Qwen4ExpMTPTextProvider(target_generator=model,
        config=SpeculativeProviderConfig(provider='qwen4_exp_mtp', draft_model='unused', candidate_budget=3),
        draft_runner=draft, draft_resident=object()))
    monkeypatch.setattr('hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner', lambda *a, **k: ResidentTarget())
    monkeypatch.setattr('hipengine.generation.qwen4_exp_serving_mtp.Qwen4ExpGGUFMTPDraftRunner', lambda *a, **k: ResidentDraft())
    api = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=capacity, max_prefill_chunk_tokens=2))
    return api, target, draft


@pytest.mark.parametrize('controls', [{}, {'temperature': 0},
    {'top_k': 2, 'top_p': .8, 'min_p': .1, 'frequency_penalty': .3, 'repetition_penalty': 1.2},
    {'logit_bias': {2: 2}, 'suppress_token_ids': (0,)},
    {'logprobs': False, 'top_logprobs': 0}])
def test_resident_sampled_cycles_match_ar_and_return_committed_metadata(monkeypatch, controls):
    api, target, draft = driver(monkeypatch)
    req = request(prompts=((1, 2, 3, 1),), max_tokens=15, **controls)
    expected = generator(Target()).generate_detailed(req)[0]
    try:
        output, events = _finish(api, api.submit_speculative_detailed(req))
        assert output.generated_token_ids == expected.generated_token_ids
        assert output.token_logprobs == expected.token_logprobs
        assert tuple(e.token_id for e in events if e.kind == 'token') == output.generated_token_ids
        assert output.telemetry.timing['mtp_cycles_count'] > 0
        assert any(call[0] == 'verify' for call in target.calls), 'must execute MTP, not all K0'
        assert all(call[2] for call in target.calls)
        assert target.history == list(req.prompts[0]) + list(output.generated_token_ids[:-1])
        assert target.position == draft.position
        assert output.finish_details.sampler_mode == ('sampled_speculative_mtp' if req.temperature else 'processed_speculative_mtp')
    finally:
        api.close()


@pytest.mark.parametrize('stage', ['root', 'verify', 'k0'])
def test_cancelled_sampled_work_restores_rng_history_and_does_not_publish_logs(monkeypatch, stage):
    api, target, draft = driver(monkeypatch)
    token = GenerationCancellationToken()
    req = request(max_tokens=2 if stage == 'k0' else 9, cancellation_token=token)
    try:
        sub = api.submit_speculative_detailed(req)
        api.poll(max_ticks=1)  # one-token prompt
        if stage != 'root':
            api.poll(max_ticks=1)  # committed root
        row = next(iter(api._runner._rows.values()))
        snapshot = row.sampling_state.clone()
        before_ids, before_logs = tuple(row.generated_ids), tuple(row.token_logprobs)
        if stage == 'root':
            original = api.inner._controlled_token
            def cancel_after_selection(*args, **kwargs):
                result = original(*args, **kwargs)
                token.cancel()
                return result
            monkeypatch.setattr(api.inner, '_controlled_token', cancel_after_selection)
        elif stage == 'verify':
            target.on_verify = token.cancel
        else:
            target.on_step = token.cancel
        output, events = _finish(api, sub)
        assert output.finish_details.cancelled
        assert output.generated_token_ids == before_ids
        assert output.token_logprobs == before_logs
        assert not [e for e in events if e.kind == 'token']
        assert row.sampling_state.generated_tokens == snapshot.generated_tokens
        assert row.sampling_state.step_index == snapshot.step_index
        assert row.sampling_state._rng.bit_generator.state == snapshot._rng.bit_generator.state
        assert target.transactions[-1].state == 'rolled_back'
        assert target.position == draft.position
        target.on_verify = target.on_step = lambda: None
        if stage == 'root':
            monkeypatch.setattr(api.inner, '_controlled_token', original)
        clean = replace(req, cancellation_token=None)
        recovered, _ = _finish(api, api.submit_speculative_detailed(clean))
        expected = generator(Target()).generate_detailed(clean)[0]
        assert recovered.generated_token_ids == expected.generated_token_ids
        assert recovered.token_logprobs == expected.token_logprobs
    finally:
        api.close()


@pytest.mark.parametrize('draws,candidate', [([.9], 0), ([.1, .9], 0),
                                          ([.1, .9], 1), ([.1, .1, .1, .1, .9], 0)])
def test_sampled_eos_metadata_excludes_only_terminal_token(monkeypatch, draws, candidate):
    api, target, draft = driver(monkeypatch)
    target.constant = np.log([.6, .4])
    draft.candidate = candidate
    api.inner.tokenizer.eos_token_id = 1
    values = iter(draws)
    monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(values))
    try:
        req = request(max_tokens=9, top_logprobs=2)
        output, events = _finish(api, api.submit_speculative_detailed(req))
        assert output.generated_token_ids == (0,) * (len(draws) - 1) + (1,)
        assert len(output.token_logprobs) == len(draws) - 1
        assert all(t.token_id == 0 and np.exp(t.logprob) == pytest.approx(.6) for t in output.token_logprobs)
        assert output.finish_details.reason == 'eos'
        assert tuple(e.token_id for e in events if e.kind == 'token') == output.generated_token_ids
        assert target.position == draft.position == len(draws)
        with pytest.raises(StopIteration):
            next(values)
    finally:
        api.close()


def test_resident_row_seeds_and_transactions_are_request_owned(monkeypatch):
    api, _, _ = driver(monkeypatch, capacity=2)
    req = request(max_tokens=14, row_seeds=(123,))
    transactions = []
    original = api._runner.execute_speculative_cycle
    def capture(plan, **kwargs):
        result = original(plan, **kwargs)
        transactions.append(result.transaction)
        return result
    monkeypatch.setattr(api._runner, 'execute_speculative_cycle', capture)
    try:
        a, b = api.submit_speculative_many_detailed((req, replace(req, seed=999)))
        first, _ = _finish(api, a)
        second, _ = _finish(api, b)
        alone, _ = _finish(api, api.submit_speculative_detailed(req))
        assert first.generated_token_ids == second.generated_token_ids == alone.generated_token_ids
        assert first.token_logprobs == second.token_logprobs == alone.token_logprobs
        # This submission API is intentionally one prompt per request; direct
        # multi-prompt seed coverage lives in test_unit_qwen4exp_mtp_sampling.
        with pytest.raises(ValueError, match='exactly one prompt'):
            api.submit_speculative_detailed(replace(req, prompts=((2,), (1,)), row_seeds=(88, 123)))
        assert transactions and all(tx.pre_rng_counters[0] >= 1 for tx in transactions)
        assert len(api._runner._available_drafts) == len(api._runner._available) == 2
    finally:
        api.close()
