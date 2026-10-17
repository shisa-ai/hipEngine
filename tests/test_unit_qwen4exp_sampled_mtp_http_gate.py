"""The sampled-MTP audit cannot pass on AR fallback or incomplete metadata."""
from copy import deepcopy
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts.qwen4exp_sampled_mtp_http_gate import extract, fingerprint, load_prompts, validate_route


def payload():
    token = dict(token='A', bytes=[65], logprob=0., top_logprobs=[dict(token='A', bytes=[65], logprob=0.)])
    return dict(choices=[dict(message=dict(content='AA'), logprobs=dict(content=[token, token]),
        finish_reason='length', hipengine=dict(generated_token_ids=[1, 1],
            diagnostics=dict(sampling_algorithm='target_draw_match_deterministic_draft'),
            decode_state=dict(execution_path='qwen4exp_mtp_batched_state_rows_verify'),
            finish_details=dict(sampler_mode='sampled_speculative_mtp')))],
        usage=dict(completion_tokens=2), hipengine=dict(speculative_mtp=dict(
            used=True, draft_tokens=1, accepted_draft_tokens=1, rejected_draft_tokens=0, draft_cycles=1)))


def test_both_prompt_files_cover_every_category_without_duplicates():
    rows = load_prompts()
    assert len(rows) == 18
    assert len({row['suite'] for row in rows}) == 2


def test_gate_normalizes_blocking_and_sse_without_losing_terminal_metadata():
    body = payload()
    result = extract(body, chat=True)
    validate_route(result, mtp=True)
    stream = deepcopy(body)
    stream['choices'][0]['delta'] = stream['choices'][0].pop('message')
    stream['choices'][0]['hipengine'].update(stream.pop('hipengine'))
    streamed = extract([stream], chat=True)
    validate_route(streamed, mtp=True)
    assert fingerprint(streamed) == fingerprint(result)


@pytest.mark.parametrize('mutation', ['ar', 'no_cycles', 'bad_count', 'bad_usage', 'wrong_algorithm', 'wrong_path', 'missing_bytes'])
def test_gate_rejects_incomplete_or_mislabeled_execution(mutation):
    result = extract(payload(), chat=True)
    if mutation == 'ar':
        result['mtp']['used'] = False
    elif mutation == 'no_cycles':
        result['mtp']['draft_cycles'] = 0
    elif mutation == 'bad_count':
        result['mtp']['accepted_draft_tokens'] = 2
    elif mutation == 'bad_usage':
        result['usage']['completion_tokens'] = 3
    elif mutation == 'wrong_algorithm':
        result['sampling_algorithm'] = 'raw_argmax_match'
    elif mutation == 'wrong_path':
        result['execution_path'] = 'ar'
    else:
        result['tokens'][0]['bytes'] = None
    with pytest.raises(AssertionError):
        validate_route(result, mtp=True)


def test_gate_keeps_errors_as_failures_even_after_sse_headers():
    with pytest.raises(AssertionError):
        extract([{'error': {'code': 'generation_failed'}}], chat=True)


def test_replay_only_failed_sse_requests_opt_in_execution_metadata(monkeypatch, tmp_path):
    import scripts.qwen4exp_sampled_mtp_http_gate as gate
    original = dict(name='prompt-generic', passed=True, result=extract(payload(), chat=True))
    failed = dict(name='prompt-sse', passed=False, endpoint='/v1/chat/completions',
                  request=dict(stream=True, logprobs=True, stream_options=dict(include_usage=True)))
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps(dict(rows=[original, failed], comparisons=[dict(
        name='prompt-blocking-sse', left=original['name'], right=failed['name'])])))
    requests = []
    def respond(request):
        if request.url.path == '/ready':
            return httpx.Response(200, json={'ready': True})
        if request.url.path.endswith('/capabilities'):
            caps = dict(processed_target_verification=True, default_enabled=False,
                        supported_sampling_fields=['temperature'])
            return httpx.Response(200, json=dict(sampling=dict(speculative=caps, speculative_mtp=caps)))
        body = json.loads(request.content)
        requests.append(body)
        assert body['stream_options'] == dict(include_usage=True, include_hipengine=True)
        event = payload()
        event['choices'][0]['delta'] = event['choices'][0].pop('message')
        event['choices'][0]['hipengine'].update(event.pop('hipengine'))
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n')
    client = httpx.AsyncClient(base_url='http://test', transport=httpx.MockTransport(respond))
    monkeypatch.setattr(gate.httpx, 'AsyncClient', lambda **kw: client)
    report = asyncio.run(gate.run(SimpleNamespace(output=tmp_path / 'replay',
        replay_report=baseline, case=None, source_commit='test', base_url='http://test')))
    assert report['passed'] and len(requests) == len(report['rows']) == 1
    assert report['comparisons'][0]['passed']
