"""OpenAI logprob windows use decoded bytes, not replacement-token lengths."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from hipengine.generation.registry import GenerationOutput, GenerationStreamChunk, TokenLogprob
from hipengine.server.api import (
    ServerConfig, create_app, _chat_logprobs, _chat_visible_content_logprobs,
    _completion_logprobs, _coerce_token_logprobs, _stream_token_logprobs_for_text_span,
    _trim_token_logprobs, _strip_chat_terminal_markers,
)
from tests.test_integration_server_api import FakeLLM, _sse_payloads
from tests.test_unit_token_logprob_bytes import fragments


def test_trimming_retains_both_byte_fragments_and_punctuation():
    tokens = fragments()
    assert _trim_token_logprobs(tokens, '舐。') == tokens
    assert _trim_token_logprobs(tokens, '舐') == tokens[:2]
    assert _trim_token_logprobs(tokens, '') == ()
    chunk = GenerationStreamChunk(text='舐。', token_logprobs=tokens)
    assert _stream_token_logprobs_for_text_span(chunk, 0, 1) == tokens[:2]
    assert _stream_token_logprobs_for_text_span(chunk, 1, 2) == tokens[2:]
    payload = _completion_logprobs(GenerationOutput(text=chunk.text, token_logprobs=tokens), chunk.text, echo_text='Hi ')
    assert payload['text_offset'] == [0, 3, 3, 4]
    assert payload['token_logprobs'] == [None, -.25, -.25, -.25]


def test_incomplete_final_codepoint_keeps_its_real_bytes_and_probability():
    first = fragments()[0]
    assert _trim_token_logprobs((first,), '�') == (first,)
    assert _chat_logprobs(GenerationOutput(text='�', token_logprobs=(first,)), '�')['content'][0]['bytes'] == [232, 136]


def test_token_crossing_visible_boundary_is_not_given_a_partial_probability():
    t = TokenLogprob(0, 'A�', -.5, token_bytes=tuple(b'A\xe8\x88'))
    assert _trim_token_logprobs((t,), 'A') == ()


@pytest.mark.parametrize('initially_open', [False, True])
def test_reasoning_removal_preserves_fragment_group(initially_open):
    prefix = 'hidden</think>' if initially_open else '<think>hidden</think>'
    tokens = (TokenLogprob(99, prefix, -.9, token_bytes=tuple(prefix.encode())),) + fragments()
    raw = prefix + '舐。'
    result = _chat_visible_content_logprobs(GenerationOutput(text=raw, token_logprobs=tokens), raw, initially_open=initially_open)
    assert len(result['content']) == 3
    assert bytes(b for t in result['content'] for b in t['bytes']).decode() == '舐。'
    assert all(t['logprob'] == -.25 for t in result['content'])


@pytest.mark.parametrize('coerce', [lambda t: asdict(t), lambda t: SimpleNamespace(**asdict(t))])
def test_mapping_and_object_coercion_preserve_bytes(coerce):
    assert _coerce_token_logprobs([coerce(t) for t in fragments()]) == fragments()


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('stream', [False, True])
def test_http_byte_fragments_have_all_probabilities(chat, stream):
    tokens = fragments()
    fake = FakeLLM(detailed_outputs=[GenerationOutput(text='舐。', token_logprobs=tokens, generated_token_ids=(0, 1, 2))])
    client = TestClient(create_app(ServerConfig(model='fake-path', served_model_name='fake-model'), llm=fake))
    body = dict(model='fake-model', max_tokens=3, stream=stream)
    if chat:
        body.update(messages=[dict(role='user', content='hello')], logprobs=True, top_logprobs=1)
    else:
        body.update(prompt='hello', logprobs=1)
    response = client.post('/v1/chat/completions' if chat else '/v1/completions', json=body)
    assert response.status_code == 200
    payloads = _sse_payloads(response.text) if stream else [response.json()]
    rows = [c['logprobs'] for p in payloads for c in p.get('choices', []) if c.get('logprobs')]
    if chat:
        content = [t for row in rows for t in row['content']]
        assert bytes(b for t in content for b in t['bytes']).decode() == '舐。'
        assert [t['logprob'] for t in content] == [-.25] * 3
        assert [t['top_logprobs'][0]['bytes'] for t in content] == [list(t.token_bytes) for t in tokens]
    else:
        assert [p for row in rows for p in row['token_logprobs']] == [-.25] * 3
        raw = [b for row in rows for piece in row['hipengine']['token_bytes'] for b in piece]
        assert bytes(raw).decode() == '舐。'


@pytest.mark.parametrize('prefix', ['', '<think>hidden</think>'])
def test_blocking_and_buffered_chat_preserve_generated_trailing_newline(prefix):
    visible = '舐。\n'
    tokens = fragments() + (TokenLogprob(3, '\n', -.5, token_bytes=(10,)),)
    if prefix:
        tokens = (TokenLogprob(99, prefix, -.9, token_bytes=tuple(prefix.encode())),) + tokens
    fake = FakeLLM(detailed_outputs=[GenerationOutput(text=prefix + visible, token_logprobs=tokens)])
    # Flash-Next's model-owned parser leaves plain text untouched without tools.
    fake.chat_tool_parser = SimpleNamespace(parse=lambda *_: None, requires_declared_tools=True)
    client = TestClient(create_app(ServerConfig(model='fake-path', served_model_name='fake-model'), llm=fake))
    body = dict(model='fake-model', messages=[dict(role='user', content='hello')], logprobs=True, top_logprobs=1)
    blocking = client.post('/v1/chat/completions', json=body).json()['choices'][0]
    events = _sse_payloads(client.post('/v1/chat/completions', json=body | {'stream': True}).text)
    chunks = [c for event in events for c in event.get('choices', [])]
    streamed = ''.join(c.get('delta', {}).get('content', '') for c in chunks)
    assert blocking['message']['content'] == streamed == visible
    for logs in [blocking['logprobs']['content'], [t for c in chunks if c.get('logprobs') for t in c['logprobs']['content']]]:
        assert bytes(b for t in logs for b in t['bytes']).decode() == visible
        assert len(logs) == 4


@pytest.mark.parametrize('text', ['answer\n', 'answer\n<|im_end|>', 'answer\n<|im_end|> \n'])
def test_terminal_marker_cleanup_preserves_whitespace_generated_before_eos(text):
    assert _strip_chat_terminal_markers(text) == 'answer\n'


def test_completion_preserves_top_candidates_with_identical_replacement_text():
    token = TokenLogprob(0, '�', -.2, ((0, '�', -.2), (1, '�', -1.7)),
                        token_bytes=(232, 136), top_token_bytes=((232, 136), (232, 137)))
    payload = _completion_logprobs(GenerationOutput(text='�', token_logprobs=(token,)), '�')
    top = payload['hipengine']['top_logprob_tokens'][0]
    assert [(t['token_id'], t['bytes'], t['logprob']) for t in top] == [
        (0, [232, 136], -.2), (1, [232, 137], -1.7)]
