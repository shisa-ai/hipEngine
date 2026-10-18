"""HTTP routing proof with a real resident MTP owner and analytic fake logits."""
import json

import pytest

from hipengine.llm import LLM, _generation_request
from hipengine.server import create_app
from fastapi.testclient import TestClient
from tests.test_integration_server_speculative_provider import _SpeculativeFakeLLM, _config
from tests.test_unit_qwen4exp_mtp_resident_sampling import driver


class ResidentLLM(_SpeculativeFakeLLM):
    supports_speculative_mtp = True

    def __init__(self, api):
        super().__init__()
        from hipengine.generation.engine_service import EngineService
        self.api = EngineService(api, idle_wait_seconds=0)
        self._text_generator = self.api
        self.speculative_provider = 'qwen4_exp_mtp'

    def _get_text_generator(self):
        return self.api

    @property
    def speculative_capabilities(self):
        return LLM.speculative_capabilities.fget(self)

    def generate_detailed(self, prompts, sampling_params):
        self.ar_calls.append((prompts, sampling_params))
        return self.api.generate_detailed(_generation_request(prompts, sampling_params))

    def generate_speculative_detailed(self, prompts, sampling_params):
        self.speculative_calls.append((prompts, sampling_params))
        return self.api.generate_speculative_detailed(_generation_request(prompts, sampling_params))

    generate_speculative_mtp_detailed = generate_speculative_detailed

    def stream_speculative_detailed(self, prompt, sampling_params):
        yield from self.api.stream_speculative_detailed(_generation_request((prompt,), sampling_params))

    stream_speculative_mtp_detailed = stream_speculative_detailed


@pytest.mark.parametrize('include_hipengine', [False, True])
@pytest.mark.parametrize('endpoint', ['completions', 'chat/completions'])
@pytest.mark.parametrize('spelling', ['speculative', 'speculative_mtp'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('logprobs', [False, True])
def test_http_sampled_requests_execute_mtp_not_ar(monkeypatch, endpoint, spelling, stream, logprobs, include_hipengine):
    api, target, _ = driver(monkeypatch)
    llm = ResidentLLM(api)
    counted_texts = []
    original_count = llm.count_tokens
    def count(text):
        counted_texts.append(text)
        return original_count(text)
    monkeypatch.setattr(llm, 'count_tokens', count)
    monkeypatch.setattr(api.inner, 'render_chat_prompt', llm.render_chat_prompt)
    cfg = _config(speculative_provider='qwen4_exp_mtp', speculative_candidate_budget=3,
                  speculative_mtp_serving='opt_in', speculative_mtp_thinking='hard')
    body = dict(model='fake-model', max_tokens=12, temperature=.8, top_k=2,
                seed=41, stream=stream, **{spelling: True})
    if stream:
        body['stream_options'] = dict(include_usage=True, include_hipengine=include_hipengine)
    if endpoint == 'completions':
        body.update(prompt='one')
        if logprobs:
            body['logprobs'] = 3
    else:
        body.update(messages=[dict(role='user', content='one')], enable_thinking=False)
        if logprobs:
            body.update(logprobs=True, top_logprobs=3)
    try:
        with TestClient(create_app(cfg, llm=llm)) as client:
            response = client.post('/v1/' + endpoint, json=body)
            assert response.status_code == 200, response.text
            if stream:
                payloads = [json.loads(line[6:]) for line in response.text.splitlines()
                            if line.startswith('data: {')]
                assert payloads and not any('error' in p for p in payloads), response.text
                terminal, = [c for p in payloads for c in p.get('choices', []) if c.get('finish_reason')]
                if include_hipengine:
                    extra = terminal['hipengine']
                    assert extra['generation_shape']['route'] == spelling
                    summary = extra['speculative_mtp']
                    assert summary['used'] is True
                    assert summary['draft_tokens'] == extra['timing']['mtp_generated_draft_tokens'] > 0
                    assert summary['accepted_draft_tokens'] == extra['timing']['mtp_accepted_draft_tokens']
                    assert summary['draft_cycles'] == extra['timing']['mtp_cycles_count'] > 0
                    usage, = [p['usage'] for p in payloads if p.get('usage')]
                    assert usage['completion_tokens_details']['accepted_prediction_tokens'] == summary['accepted_draft_tokens']
                    assert usage['completion_tokens_details']['rejected_prediction_tokens'] == summary['rejected_draft_tokens']
                else:
                    assert all('hipengine' not in p for p in payloads)
                    assert all('hipengine' not in c for p in payloads for c in p.get('choices', []))
            else:
                payloads = [response.json()]
            assert not llm.ar_calls, 'selected speculation must never run the direct AR logprob shortcut'
            assert any(call[0] == 'verify' for call in target.calls), response.text
            assert all(call[2] for call in target.calls)
            assert llm.speculative_calls or stream
            if stream and include_hipengine:
                assert not any(text in ('A', 'B', 'C', 'D') for text in counted_texts), (
                    'committed chunk accounting must not enqueue per-token tokenizer controls'
                )
            if logprobs:
                entries = [p['choices'][0]['logprobs'] for p in payloads
                           if p.get('choices') and p['choices'][0].get('logprobs')]
                assert entries, response.text
                if stream:
                    assert len(entries) > 1, 'logprobs must not force full-response buffering'
                if endpoint == 'completions':
                    assert sum(len(e['tokens']) for e in entries) == 12
                else:
                    assert sum(len(e['content']) for e in entries) == 12
    finally:
        llm.api.close()
        api.close()


@pytest.mark.parametrize('endpoint', ['completions', 'chat/completions'])
@pytest.mark.parametrize('max_tokens', [1, 2, 3, 4, 7])
def test_live_logprobs_preserve_utf8_fragments(monkeypatch, endpoint, max_tokens):
    import numpy as np
    from hipengine.generation.sampling import RowSamplingState
    from scripts.qwen4exp_sampled_mtp_http_gate import extract, fingerprint

    api, target, _ = driver(monkeypatch)
    target.constant = np.zeros(4)
    parts = (b'\xe3', b'\x81', b'\x82', b'X')
    monkeypatch.setattr(api.inner.tokenizer, 'decode',
                        lambda ids, **kw: b''.join(parts[i] for i in ids).decode('utf-8', errors='replace'))
    monkeypatch.setattr(api.inner.tokenizer, 'decode_token_bytes', lambda i: parts[i], raising=False)
    llm = ResidentLLM(api)
    monkeypatch.setattr(api.inner, 'render_chat_prompt', llm.render_chat_prompt)
    cfg = _config(speculative_provider='qwen4_exp_mtp', speculative_candidate_budget=3,
                  speculative_mtp_serving='opt_in', speculative_mtp_thinking='hard')
    chat = endpoint == 'chat/completions'
    body = dict(model='fake-model', max_tokens=max_tokens, temperature=1, speculative=True)
    body.update(dict(messages=[dict(role='user', content='one')], enable_thinking=False,
                     logprobs=True, top_logprobs=4) if chat else dict(prompt='one', logprobs=4))
    results = []
    try:
        with TestClient(create_app(cfg, llm=llm)) as client:
            for stream in (False, True):
                draws = iter([.1, .3, .6, .9] * 4)
                monkeypatch.setattr(RowSamplingState, 'random_unit', lambda self: next(draws))
                req = body | dict(stream=stream)
                if stream:
                    req['stream_options'] = dict(include_usage=True, include_hipengine=True)
                response = client.post('/v1/' + endpoint, json=req)
                assert response.status_code == 200, response.text
                payload = ([json.loads(line[6:]) for line in response.text.splitlines()
                            if line.startswith('data: {')] if stream else response.json())
                if stream:
                    assert not any('error' in p for p in payload), response.text
                result = fingerprint(extract(payload, chat=chat))
                result['finish'] = {k: v for k, v in result['finish'].items()
                                    if k not in ('continuation_id', 'continuation_eligible')}
                results.append(result)
        assert results[0] == results[1]
    finally:
        llm.api.close()


def test_logprob_requests_preserve_operator_mtp_rollback(monkeypatch):
    api, target, _ = driver(monkeypatch)
    llm = ResidentLLM(api)
    cfg = _config(speculative_provider='qwen4_exp_mtp', speculative_candidate_budget=3,
                  speculative_mtp_serving='opt_in', speculative_mtp_thinking='hard')
    try:
        with TestClient(create_app(cfg, llm=llm)) as client:
            assert client.post('/v1/hipengine/speculative_mtp/rollback').status_code == 200
            # An explicit ``speculative_mtp`` request is refused with the typed
            # rollback reason rather than silently served by the AR path; only
            # an automatic request keeps the reported AR fallback.
            response = client.post('/v1/completions', json=dict(
                model='fake-model', prompt='one', max_tokens=12, temperature=.8,
                top_k=2, logprobs=3, seed=41, speculative_mtp=True))
            assert response.status_code == 503, response.text
            detail = response.json()['error']['hipengine']['speculative_mtp']
            assert detail['reason'] == 'mtp_operator_rollback'
            assert not llm.ar_calls and not llm.speculative_calls
            assert not any(call[0] == 'verify' for call in target.calls)
    finally:
        llm.api.close()
        api.close()
