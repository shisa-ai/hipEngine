"""Provider-declared controls must reach actual resident MTP through HTTP."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hipengine.generation import EngineService
from hipengine.server import create_app
from tests.test_integration_server_sampled_speculation import ResidentLLM
from tests.test_integration_server_speculative_provider import _config
from tests.test_unit_qwen4exp_mtp_resident_sampling import driver
from tests.test_unit_qwen4exp_mtp_structured import ByteTokenizer, TOOL, CALL


def setup(monkeypatch, text, *, invalid_argmax=False):
    api, target, draft = driver(monkeypatch)
    model = api.inner
    model.tokenizer = ByteTokenizer()
    model._grammar_tokenizer = model.tokenizer.grammar
    # Byte pieces must not be decoded separately when generating Japanese.
    model.tokenizer.decode_token_bytes = lambda i: bytes([i]) if i < 256 else model.tokenizer.decode((i,)).encode()
    target.max_sequence_length = draft.max_sequence_length = 4096
    service = EngineService(api)
    llm = ResidentLLM(service)
    llm._text_generator = service
    llm.tokenize, llm.detokenize = model.tokenize, model.detokenize
    llm.count_tokens = model.count_tokens
    from hipengine.chat.qwen4_exp import Qwen4ExpReasoningParser
    model.chat_reasoning_parser = Qwen4ExpReasoningParser(model.tokenizer)
    def render(messages, **kwargs):
        return llm.render_chat_prompt(messages, **kwargs) + ('<think>' if kwargs.get('enable_thinking') else '')
    monkeypatch.setattr(model, 'render_chat_prompt', render)
    original = target.prefill_append
    def prefill(ids, **kwargs):
        target.prompt_length = len(target.history) + len(ids)
        return original(ids, **kwargs)
    monkeypatch.setattr(target, 'prefill_append', prefill)
    ids = (*model.tokenizer.encode(text), 256)
    def scores():
        logits = np.full(261, -20., dtype=np.float32)
        index = max(0, len(target.history) - target.prompt_length)
        logits[ids[min(index, len(ids) - 1)]] = 20.
        if invalid_argmax:
            logits[ord('!')] = 40.
        return logits
    target.scores = scores
    draft.candidate = ord('x')
    cfg = _config(speculative_provider='qwen4_exp_mtp', speculative_candidate_budget=3,
                  speculative_mtp_serving='opt_in', speculative_mtp_thinking='hard')
    return service, llm, target, draft, cfg


def extract(response, stream):
    assert response.status_code == 200, response.text
    if not stream:
        body = response.json()
        choice = body['choices'][0]
        return choice, {**body['hipengine'], **choice.get('hipengine', {})}, [body]
    payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
    assert payloads and not any('error' in p for p in payloads), response.text
    terminal, = [c for p in payloads for c in p.get('choices', []) if c.get('finish_reason')]
    return terminal, terminal['hipengine'], payloads


def prove_mtp(llm, target, draft, extra, spelling):
    assert not llm.ar_calls, 'AR fallback cannot qualify an MTP control'
    assert any(c[0] == 'verify' for c in target.calls)
    assert extra['generation_shape']['route'] == spelling
    assert extra['speculative_mtp']['used'] is True
    assert extra['timing']['mtp_cycles_count'] > 0
    assert extra['timing']['mtp_generated_draft_tokens'] > 0
    assert target.position == draft.position


@pytest.mark.parametrize('spelling', ['speculative', 'speculative_mtp'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('temperature', [0., .8])
@pytest.mark.parametrize('logprobs', [False, True])
@pytest.mark.parametrize('kind,text,controls,visible,finish', [
    ('ascii_stop', 'alphaENDignored', {'stop': 'END'}, 'alpha', 'stop'),
    ('japanese_stop', '回答。終わり尾', {'stop': ['終わり', 'never']}, '回答。', 'stop'),
    ('alternate_eos', 'abcdXtail', {'eos_token_id': ord('X')}, 'abcd', 'stop'),
    ('ignore_eos', 'abc</s>def', {'ignore_eos': True, 'max_tokens': 7}, 'abcdef', 'length'),
    ('min_tokens', 'ab</s>def', {'min_tokens': 5, 'eos_token_id': 256}, None, 'stop'),
])
def test_stops_and_eos_execute_mtp(monkeypatch, spelling, stream, temperature, logprobs, kind, text, controls, visible, finish):
    api, llm, target, draft, cfg = setup(monkeypatch, text)
    body = dict(model='fake-model', prompt='request', max_tokens=50, temperature=temperature,
                seed=41, stream=stream, **{spelling: True})
    if logprobs:
        body['logprobs'] = 2
    body.update(controls)
    if stream:
        body['stream_options'] = dict(include_usage=True, include_hipengine=True)
    try:
        with TestClient(create_app(cfg, llm=llm)) as client:
            response = client.post('/v1/completions', json=body)
        choice, extra, payloads = extract(response, stream)
        prove_mtp(llm, target, draft, extra, spelling)
        output = ''.join(c.get('text', '') for p in payloads for c in p.get('choices', []))
        if visible is not None:
            assert output == visible
        assert choice['finish_reason'] == finish
        if kind == 'min_tokens':
            assert len(extra['generated_token_ids']) >= 5
            assert 256 not in extra['generated_token_ids'][:4]
        if logprobs:
            entries = [c['logprobs'] for p in payloads for c in p.get('choices', []) if c.get('logprobs')]
            raw = b''.join(bytes(piece) for e in entries for piece in e['hipengine']['token_bytes'])
            assert raw.decode(errors='replace') == output
    finally:
        api.close()


def test_queue_does_not_coalesce_hard_and_hint_policies():
    from hipengine import SamplingParams
    from hipengine.server.api import _GenerationBatcher, _QueuedGeneration
    batcher = _GenerationBatcher(engine_factory=lambda: None, batch_window_seconds=0)
    hard = _QueuedGeneration(('prompt',), SamplingParams(), route='speculative_mtp', mtp_thinking_policy='hard')
    hint = _QueuedGeneration(('prompt',), SamplingParams(), route='speculative_mtp', mtp_thinking_policy='hint')
    assert batcher._group_key(hard) != batcher._group_key(hint)


@pytest.mark.parametrize('text', ['', 'Calling tools now.'])
def test_buffered_tool_logprobs_publish_once_across_calls_and_fragments(text):
    from hipengine.server.api import _chat_stream_parsed, _ParsedChatOutput, _ParsedToolCall
    logs = {'content': [{'token': '<tool_call>', 'bytes': list(b'<tool_call>'), 'logprob': 0.0}]}
    calls = tuple(_ParsedToolCall(f'call_{i}', 'ping', json.dumps({'data': 'x' * 3000})) for i in range(2))
    events = list(_chat_stream_parsed('id', 0, 'model', _ParsedChatOutput(text, calls), 'stop', logprobs=logs))
    choices = [c for event in events for c in json.loads(event.removeprefix('data: '))['choices']]
    assert [c['logprobs'] for c in choices if c.get('logprobs')] == [logs]
    assert sum(bool(c.get('delta', {}).get('tool_calls')) for c in choices) > 2


SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'result', 'strict': True, 'schema': {'const': {'x': 7}}}}


@pytest.mark.parametrize('spelling', ['speculative', 'speculative_mtp'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('temperature', [0., .8])
@pytest.mark.parametrize('logprobs', [False, True])
@pytest.mark.parametrize('kind,text,controls', [
    ('schema_bound', '{"x":7}', {'response_format': SCHEMA, 'max_tokens': 7}),
    ('required_tool', CALL, {'tools': [TOOL], 'tool_choice': 'required', 'parallel_tool_calls': False}),
    ('auto_tool', CALL, {'tools': [TOOL], 'tool_choice': 'auto', 'parallel_tool_calls': False}),
    ('auto_answer', '{"x":7}', {'tools': [TOOL], 'tool_choice': 'auto', 'response_format': SCHEMA}),
    ('hard_thinking', 'ab</think>{"x":7}', {'response_format': SCHEMA, 'enable_thinking': True,
                                          'hard_think_cap': 2, 'min_answer_tokens': 7}),
])
def test_structured_chat_controls_execute_mtp(monkeypatch, spelling, stream, temperature, logprobs, kind, text, controls):
    api, llm, target, draft, cfg = setup(monkeypatch, text, invalid_argmax=kind not in ('hard_thinking', 'auto_tool'))
    body = dict(model='fake-model', messages=[dict(role='user', content='request')], max_tokens=80,
                temperature=temperature, seed=41,
                enable_thinking=False, stream=stream, **{spelling: True})
    if logprobs:
        body.update(logprobs=True, top_logprobs=2)
    body.update(controls)
    if stream:
        body['stream_options'] = dict(include_usage=True, include_hipengine=True)
    try:
        with TestClient(create_app(cfg, llm=llm)) as client:
            response = client.post('/v1/chat/completions', json=body)
        choice, extra, payloads = extract(response, stream)
        prove_mtp(llm, target, draft, extra, spelling)
        messages = [c.get('delta', c.get('message', {})) for p in payloads for c in p.get('choices', [])]
        if kind in ('required_tool', 'auto_tool'):
            calls = [call for m in messages for call in m.get('tool_calls', [])]
            assert ''.join(call['function'].get('name', '') for call in calls) == 'ping'
            assert json.loads(''.join(call['function'].get('arguments', '') for call in calls)) == {}
            assert choice['finish_reason'] == 'tool_calls'
            if logprobs:
                entries = [token for p in payloads for c in p.get('choices', [])
                           for token in (c.get('logprobs') or {}).get('content', [])]
                assert entries, 'tool SSE must not discard requested raw model logprobs'
                raw = bytes(b for token in entries for b in token['bytes']).decode()
                assert raw == CALL
        else:
            assert json.loads(''.join(m.get('content') or '' for m in messages)) == {'x': 7}
            assert choice['finish_reason'] == 'stop'
        if kind == 'hard_thinking':
            assert ''.join(m.get('reasoning_content') or '' for m in messages) == 'ab'
            details = extra['finish_details']
            assert details['reasoning_tokens'] <= 3
            assert details['phase'] == 'answer'
    finally:
        api.close()
