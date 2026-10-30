"""Literal response stops are suppressed before SSE publication."""
import json

import pytest
from fastapi.testclient import TestClient
from hipengine.server import ServerConfig, create_app
from tests.support.server_api_fakes import FakeLLM
from hipengine.generation.registry import FinishDetails, GenerationStreamChunk, TokenLogprob


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('chunks,stop,expected', [
    (['alpha<', 'stop>', 'tail'], '<stop>', 'alpha'),
    (['alpha<stop>tail'], '<stop>', 'alpha'),
    (['alpha<st'], '<stop>', 'alpha<st'),
    (['alpha', 'beta'], 'unseen', 'alphabeta'),
    (['ab', 'aba', 'tail'], ['bab', 'aba'], ''),
    (['日', '本語', 'tail'], '日本語', ''),
])
def test_literal_stop_is_not_published(chat, chunks, stop, expected):
    fake = FakeLLM(stream_chunks=chunks)
    client = TestClient(create_app(ServerConfig(model='fake', served_model_name='fake'), llm=fake))
    payload = {'model': 'fake', 'stream': True, 'stop': stop, 'max_tokens': 32}
    payload.update({'messages': [{'role': 'user', 'content': 'x'}]} if chat else {'prompt': 'x'})
    response = client.post('/v1/chat/completions' if chat else '/v1/completions', json=payload)
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    assert not any('error' in e for e in events)
    text = ''.join(c.get('delta', {}).get('content', '') if chat else c.get('text', '')
                   for e in events for c in e.get('choices', []))
    assert text == expected


def test_literal_stop_is_applied_before_reasoning_split():
    fake = FakeLLM(stream_chunks=['<think>reason<', 'stop>hidden</think>answer'])
    client = TestClient(create_app(ServerConfig(model='fake', served_model_name='fake'), llm=fake))
    response = client.post('/v1/chat/completions', json={
        'model': 'fake', 'messages': [{'role': 'user', 'content': 'x'}],
        'stream': True, 'stop': '<stop>', 'max_tokens': 32,
    })
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    deltas = [c.get('delta', {}) for e in events for c in e.get('choices', [])]
    assert ''.join(d.get('reasoning_content', '') for d in deltas) == 'reason'
    assert ''.join(d.get('content', '') for d in deltas) == ''


@pytest.mark.parametrize('chat', [False, True])
def test_literal_stop_preserves_terminal_ids_and_raw_usage(chat):
    fake = FakeLLM(stream_chunks=[
        GenerationStreamChunk(text='alpha<'), GenerationStreamChunk(text='stop>tail'),
        GenerationStreamChunk(text='', generated_token_ids=(10, 11, 12),
                              finish_details=FinishDetails(reason='length', length_limit=3)),
    ])
    client = TestClient(create_app(ServerConfig(model='fake', served_model_name='fake'), llm=fake))
    payload = {'model': 'fake', 'stream': True, 'stop': '<stop>', 'max_tokens': 32,
               'stream_options': {'include_usage': True, 'include_hipengine': True}}
    payload.update({'messages': [{'role': 'user', 'content': 'x'}]} if chat else {'prompt': 'x'})
    response = client.post('/v1/chat/completions' if chat else '/v1/completions', json=payload)
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    done = next(c for e in events for c in e.get('choices', []) if c.get('finish_reason'))
    assert done['finish_reason'] == 'stop'
    assert done['finish_details']['reason'] == 'stop'
    assert done['hipengine']['generated_token_ids'] == [10, 11, 12]
    assert next(e['usage'] for e in events if 'usage' in e)['completion_tokens'] == 3


def test_chat_literal_stop_inside_logprob_token_reports_fragment_omission():
    fake = FakeLLM(stream_chunks=[GenerationStreamChunk(
        text='alphaSTOPtail', token_logprobs=(TokenLogprob(token_id=42, token_text='alphaSTOPtail', logprob=-0.5),),
        generated_token_ids=(42,), finish_details=FinishDetails(reason='length', length_limit=1),
    )])
    fake.supports_stream_logprobs = True
    client = TestClient(create_app(ServerConfig(model='fake', served_model_name='fake'), llm=fake))
    response = client.post('/v1/chat/completions', json={
        'model': 'fake', 'messages': [{'role': 'user', 'content': 'x'}],
        'stream': True, 'stop': 'STOP', 'max_tokens': 32, 'logprobs': True,
    })
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    assert not any('error' in e for e in events)
    choices = [c for e in events for c in e.get('choices', [])]
    assert ''.join(c.get('delta', {}).get('content', '') for c in choices) == 'alpha'
    record = next(c['logprobs']['content'][0] for c in choices if c.get('logprobs'))
    assert record['token'] == 'alpha' and record['logprob'] is None
