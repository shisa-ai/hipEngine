"""The HTTP gate rejects missing probabilities/bytes and preserves error bodies."""
import asyncio
import json
import math
from types import SimpleNamespace

import httpx
import pytest

from scripts.qwen4exp_sampling_http_gate import parse_response, unpack, validate, validate_rejection


def token(raw):
    return dict(token='�', bytes=list(raw), logprob=-math.log(2), top_logprobs=[
        dict(token='�', bytes=list(raw), logprob=-math.log(2)),
        dict(token='other', bytes=[65], logprob=-math.log(2))])


def test_gate_accepts_split_bytes_not_individual_text_alignment():
    validate('舐', [token(b'\xe8\x88'), token(b'\x90')], full_mass=True)


@pytest.mark.parametrize('mutation', ['missing_bytes', 'missing_tail', 'bad_probability', 'wrong_candidate_bytes'])
def test_gate_rejects_corrupt_metadata(mutation):
    tokens = [token(b'\xe8\x88'), token(b'\x90')]
    if mutation == 'missing_bytes':
        tokens[0]['bytes'] = None
    elif mutation == 'missing_tail':
        tokens.pop()
    elif mutation == 'bad_probability':
        tokens[0]['logprob'] = float('nan')
    else:
        tokens[0]['top_logprobs'][0]['bytes'] = [66]
    with pytest.raises(AssertionError):
        validate('舐', tokens, full_mass=True)


def test_completion_unpack_preserves_same_text_top_candidates():
    candidate = token(b'\xe8\x88')
    choice = dict(text='�', logprobs=dict(tokens=['�'], token_logprobs=[candidate['logprob']],
                  hipengine=dict(token_bytes=[candidate['bytes']], top_logprob_tokens=[candidate['top_logprobs']])))
    text, tokens = unpack(choice, chat=False)
    validate(text, tokens, full_mass=True)


def test_focused_thinking_probe_uses_public_field_and_only_two_requests(monkeypatch, tmp_path):
    import scripts.qwen4exp_sampling_http_gate as gate
    requests = []

    def respond(request):
        if request.url.path == '/ready':
            return httpx.Response(200, json={'ready': True})
        body = json.loads(request.content)
        requests.append(body)
        assert body['hard_think_cap'] == 4
        assert 'thinking_hard_token_cap' not in body
        content = [token(b'\xe8\x88'), token(b'\x90')]
        choice = dict(logprobs={'content': content}, finish_reason='length')
        if body['stream']:
            choice['delta'] = {'content': '舐'}
            event = dict(choices=[choice], usage={'completion_tokens': 2})
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  text='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n')
        choice['message'] = {'content': '舐'}
        return httpx.Response(200, json={'choices': [choice]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url='http://test')
    monkeypatch.setattr(gate.httpx, 'AsyncClient', lambda **kw: client)
    report = asyncio.run(gate.run(SimpleNamespace(output=tmp_path, baseline_report=None,
                        source_commit='test', base_url='http://test', model='test', case=['hard_thinking'])))
    assert report['passed'] and len(report['rows']) == len(requests) == 2


def test_sse_request_returning_http400_json_keeps_error():
    body = dict(error=dict(code='unsupported_parameter', param='speculative', hipengine=dict(
        speculative=dict(compatibility_guard='raw_greedy_bf16_c1', blockers=['temperature']))))
    response = httpx.Response(400, json=body)
    assert parse_response(response, stream=True) == body
    validate_rejection(body, 400)
    validate_rejection([body], 200)
    with pytest.raises(AssertionError):
        validate_rejection(dict(choices=[dict(text='silently generated')]), 200)
