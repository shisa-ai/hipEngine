"""Structured MTP must advance only committed grammar/processor prefixes."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.constraints import ToolCallConstraintSpec
from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.grammar import GrammarSession
from hipengine.generation.registry import FinishDetails
from tests.test_unit_grammar_masks import tokenizer
from tests.test_unit_qwen4exp_mtp_sampling import provider, Target
from tests.test_unit_qwen4exp_mtp_resident_sampling import driver
from tests.test_unit_qwen4exp_mtp_submission import _finish
from tests.test_unit_qwen4exp_sampling import generator, request


class ByteTokenizer:
    eos_token_id = 256

    def __init__(self):
        self.grammar = tokenizer()

    def encode(self, text):
        return self.grammar.tokenize_str(text, parse_special=True)

    def decode(self, ids, *, skip_special=False):
        return self.grammar.decode_str([int(i) for i in ids if i != 256])

    def decode_token_bytes(self, token):
        return self.decode((token,)).encode()


def configure(model, target, text, *, invalid_argmax=False):
    model.tokenizer = ByteTokenizer()
    model._grammar_tokenizer = model.tokenizer.grammar
    ids = (*model.tokenizer.encode(text), 256)
    def scores():
        result = np.full(261, -20., dtype=np.float32)
        result[ids[min(len(target.history) - 1, len(ids) - 1)]] = 20
        if invalid_argmax:
            result[ord('!')] = 40  # Raw target argmax violates these grammars.
        return result
    target.scores = scores
    return ids


TOOL = {'type': 'function', 'function': {'name': 'ping', 'parameters': {'type': 'object', 'properties': {}}}}
CALL = '<tool_call>\n<function=ping>\n</function>\n</tool_call>'
SCHEMA = {'format': 'json_schema', 'value': {'const': {'x': 7}}}
CASES = [
    (SCHEMA, '{"x":7}'),
    ({**SCHEMA, 'reasoning_open': True}, 'h</think>{"x":7}'),
    ({'format': 'qwen4exp_tools', 'mode': 'required', 'tools': [TOOL], 'parallel': False}, CALL),
    ({'format': 'qwen4exp_tools', 'mode': 'auto', 'tools': [TOOL], 'answer': SCHEMA}, CALL),
    ({'format': 'qwen4exp_tools', 'mode': 'auto', 'tools': [TOOL], 'answer': SCHEMA}, '{"x":7}'),
]


@pytest.mark.parametrize('route', ['provider_serial', 'provider_batched', 'direct', 'stream', 'resident', 'many'])
@pytest.mark.parametrize('temperature', [0, .8])
@pytest.mark.parametrize('spec,text', CASES)
def test_real_grammar_and_tool_routes_use_mtp_and_preserve_all_visible_tokens(monkeypatch, route, temperature, spec, text):
    if route in ('resident', 'many'):
        api, target, draft = driver(monkeypatch)
        model = api.inner
    else:
        obj, target, draft = provider(batched=route != 'provider_serial', candidate=ord('x'))
        model = obj.target_generator
        model.attach_speculative_provider(obj)
        api = model if route in ('direct', 'stream') else obj
    ids = configure(model, target, text, invalid_argmax=not spec.get('reasoning_open'))
    draft.candidate = ord('x')
    thinking = {'thinking_hard_token_cap': 1, 'thinking_close_token_ids': (260,)} if spec.get('reasoning_open') else {}
    req = request(max_tokens=len(ids) + 4, temperature=temperature, grammar=spec, top_logprobs=2, **thinking)
    ar_target = Target()
    ar = generator(ar_target)
    configure(ar, ar_target, text, invalid_argmax=not spec.get('reasoning_open'))
    expected = ar.generate_detailed(req)[0]
    try:
        if route in ('resident', 'many'):
            sub = (api.submit_speculative_detailed(req) if route == 'resident'
                   else api.submit_speculative_many_detailed((req,))[0])
            actual, events = _finish(api, sub)
            assert tuple(e.token_id for e in events if e.kind == 'token') == actual.generated_token_ids
            assert actual.text == expected.text == text
            visible = expected.token_logprobs[:-1] if expected.finish_details.reason == 'eos' else expected.token_logprobs
            assert actual.token_logprobs == visible
        else:
            if route == 'stream':
                chunks = list(api.stream_speculative_detailed(req))
                assert len(chunks) == 1  # This API is still explicitly buffered.
                actual = chunks[0]
            else:
                actual = (api.generate_speculative_detailed(req) if route == 'direct' else api.generate_detailed(req))[0]
            assert actual.token_logprobs == expected.token_logprobs
        assert actual.generated_token_ids == expected.generated_token_ids
        assert replace(actual.finish_details, sampler_mode=None) == replace(expected.finish_details, sampler_mode=None)
        assert actual.telemetry.timing['mtp_cycles_count'] > 0
        assert target.history == [1, *actual.generated_token_ids[:-1]]
        assert target.position == draft.position
    finally:
        api.close()


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('controls', [
    {'forced_tokens_pending': (0, 2, 1)},
    {'force_sequence_completion_token_sequences': ((0, 2, 1),)},
    {'thinking_close_token_ids': (2, 3), 'thinking_hard_token_cap': 2},
    {'thinking_close_token_ids': (2, 3), 'thinking_hard_token_cap': 0,
     'post_thinking_forced_tokens_pending': (0, 1)},
])
def test_processors_match_ar_ids_logs_and_finish_metadata(monkeypatch, resident, controls):
    if resident:
        api, target, draft = driver(monkeypatch)
    else:
        api, target, draft = provider()
    req = request(max_tokens=12, **controls)
    ar_target = Target()
    for runner in (target, ar_target):
        original_scores = runner.scores
        runner.scores = lambda scores=original_scores: np.pad(scores(), (0, 6), constant_values=-np.inf)
    expected = generator(ar_target).generate_detailed(req)[0]
    try:
        actual = (_finish(api, api.submit_speculative_detailed(req))[0] if resident
                  else api.generate_detailed(req)[0])
        assert actual.generated_token_ids == expected.generated_token_ids
        assert actual.token_logprobs == expected.token_logprobs
        assert replace(actual.finish_details, sampler_mode=None) == replace(expected.finish_details, sampler_mode=None)
        assert actual.telemetry.timing['mtp_cycles_count'] > 0
        assert target.history == [1, *actual.generated_token_ids[:-1]]
        assert target.position == draft.position
    finally:
        api.close()


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('controls,text', [
    ({'json_object_close_forcing': True}, '{"x":7}'),
    ({'tool_call_constraint': ToolCallConstraintSpec(tool_names=('ping',), mode='required')},
     '<tool_call>{"name":"ping","arguments":{}}</tool_call>'),
    ({'tool_call_constraint': ToolCallConstraintSpec(tool_names=('ping',), mode='auto')}, 'plain answer'),
])
def test_token_text_constraints_and_close_forcing_match_ar(monkeypatch, resident, controls, text):
    if resident:
        api, target, draft = driver(monkeypatch)
        model = api.inner
    else:
        api, target, draft = provider()
        model = api.target_generator
    ids = configure(model, target, text)
    ar_target = Target()
    ar = generator(ar_target)
    configure(ar, ar_target, text)
    # No EOS slot: tokenizer-safe closing repairs must fit exactly in budget.
    req = request(max_tokens=len(ids) - 1, top_logprobs=2, **controls)
    expected = ar.generate_detailed(req)[0]
    try:
        actual = (_finish(api, api.submit_speculative_detailed(req))[0] if resident else api.generate_detailed(req)[0])
        assert actual.text == expected.text == text
        assert actual.generated_token_ids == expected.generated_token_ids
        assert actual.token_logprobs == expected.token_logprobs
        assert actual.telemetry.timing['mtp_cycles_count'] > 0
        assert target.position == draft.position
    finally:
        api.close()


def test_forced_root_transaction_counter_counts_draws_not_output(monkeypatch):
    api, target, draft = driver(monkeypatch)
    transactions = []
    original = api._runner._advance_mtp
    def record(*args, **kwargs):
        progress, transaction = original(*args, **kwargs)
        if transaction is not None:
            transactions.append(transaction)
        return progress, transaction
    monkeypatch.setattr(api._runner, '_advance_mtp', record)
    try:
        output, _ = _finish(api, api.submit_speculative_detailed(request(forced_tokens_pending=(0,), max_tokens=8)))
        assert output.generated_token_ids[0] == 0
        assert transactions and transactions[0].pre_rng_counters == (0,)
        assert transactions[0].pre_target_cursors == (1,)
    finally:
        api.close()


def test_grammar_checkpoint_is_independent_even_after_terminal_selection():
    original = GrammarSession(tokenizer(), SCHEMA)
    for token in b'{"x":':
        original.select(np.eye(261, dtype=np.float32)[token] * 20)
    saved = original.clone()
    before = saved.mask_logits(np.zeros(261))
    for token in [ord('7'), ord('}'), 256]:
        original.select(np.eye(261, dtype=np.float32)[token] * 20)
    assert original.done and not saved.done
    np.testing.assert_array_equal(saved.mask_logits(np.zeros(261)), before)
    for token in [ord('7'), ord('}'), 256]:
        assert saved.select(np.eye(261, dtype=np.float32)[token] * 20) == token
    assert saved.done


@pytest.mark.parametrize('stage', ['root', 'verify', 'terminal'])
def test_resident_cancellation_restores_grammar_rng_and_published_prefix(monkeypatch, stage):
    api, target, draft = driver(monkeypatch)
    ids = configure(api.inner, target, '{"x":7}')
    cancellation = GenerationCancellationToken()
    req = request(max_tokens=20, grammar=SCHEMA, cancellation_token=cancellation)
    try:
        sub = api.submit_speculative_detailed(req)
        api.poll(max_ticks=1)  # prefill
        row = next(iter(api._runner._rows.values()))
        if stage != 'root':
            api.poll(max_ticks=1)  # root
        if stage == 'terminal':
            while len(row.generated_ids) < len(ids) - 2:  # schema stops at }, before EOS
                api.poll(max_ticks=1)
        state = row.sampling_state.clone()
        mask = row.grammar.mask_logits(np.zeros(261))
        done = row.grammar.done
        before = tuple(row.generated_ids)
        original = api.inner._controlled_token
        def cancel_after_selection(*args, **kwargs):
            token = original(*args, **kwargs)
            cancellation.cancel()
            return token
        monkeypatch.setattr(api.inner, '_controlled_token', cancel_after_selection)
        actual, events = _finish(api, sub)
        assert actual.finish_details.cancelled
        assert actual.generated_token_ids == before
        assert not [e for e in events if e.kind == 'token']
        np.testing.assert_array_equal(row.grammar.mask_logits(np.zeros(261)), mask)
        assert row.grammar.done == done
        assert row.sampling_state._rng.bit_generator.state == state._rng.bit_generator.state
        assert row.sampling_state.rng_draw_count == state.rng_draw_count
        assert target.transactions[-1].state == 'rolled_back'
        assert target.position == draft.position == max(1, len(before))
    finally:
        api.close()


@pytest.mark.parametrize('control', ['json', 'tool'])
def test_token_text_constraint_cancellation_restores_parser_and_queues(monkeypatch, control):
    api, target, draft = driver(monkeypatch)
    controls = ({'json_object_close_forcing': True} if control == 'json' else
                {'tool_call_constraint': ToolCallConstraintSpec(tool_names=('ping',), mode='required')})
    text = '{"x":7}' if control == 'json' else '<tool_call>{"name":"ping","arguments":{}}</tool_call>'
    configure(api.inner, target, text)
    cancellation = GenerationCancellationToken()
    req = request(max_tokens=80, cancellation_token=cancellation, **controls)
    try:
        sub = api.submit_speculative_detailed(req)
        api.poll(max_ticks=1)
        api.poll(max_ticks=1)
        row = next(iter(api._runner._rows.values()))
        state = row.sampling_state.clone()
        original = api.inner._controlled_token
        def cancel_after_selection(*args, **kwargs):
            token = original(*args, **kwargs)
            cancellation.cancel()
            return token
        monkeypatch.setattr(api.inner, '_controlled_token', cancel_after_selection)
        actual, events = _finish(api, sub)
        assert actual.finish_details.cancelled
        assert actual.generated_token_ids == tuple(state.generated_tokens)
        assert not [e for e in events if e.kind == 'token']
        assert row.sampling_state._rng.bit_generator.state == state._rng.bit_generator.state
        assert row.sampling_state.forced_tokens == state.forced_tokens
        field = '_json_object_constraint' if control == 'json' else '_tool_call_constraint'
        assert getattr(row.sampling_state, field).observed_text == getattr(state, field).observed_text
        assert target.transactions[-1].state == 'rolled_back'
        assert target.position == draft.position == 1
    finally:
        api.close()


def test_mtp_grammar_completion_without_eos_keeps_the_closing_brace():
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpResidentServingRunner
    owner = Qwen4ExpResidentServingRunner.__new__(Qwen4ExpResidentServingRunner)
    row = SimpleNamespace(generated_ids=[123, 125], grammar=SimpleNamespace(done=True))
    finish = FinishDetails(reason='stop', sampler_mode='sampled_speculative_mtp')
    assert owner._visible_ids(row, finish) == (123, 125)
