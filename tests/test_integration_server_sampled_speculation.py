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
        self.api = api
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


@pytest.mark.parametrize('endpoint', ['completions', 'chat/completions'])
@pytest.mark.parametrize('spelling', ['speculative', 'speculative_mtp'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('logprobs', [False, True])
def test_http_sampled_requests_execute_mtp_not_ar(monkeypatch, endpoint, spelling, stream, logprobs):
    api, target, _ = driver(monkeypatch)
    llm = ResidentLLM(api)
    monkeypatch.setattr(api.inner, 'render_chat_prompt', llm.render_chat_prompt)
    cfg = _config(speculative_provider='qwen4_exp_mtp', speculative_candidate_budget=3,
                  speculative_mtp_serving='opt_in', speculative_mtp_thinking='hard')
    body = dict(model='fake-model', max_tokens=12, temperature=.8, top_k=2,
                seed=41, stream=stream, **{spelling: True})
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
            else:
                payloads = [response.json()]
            assert not llm.ar_calls, 'selected speculation must never run the direct AR logprob shortcut'
            assert any(call[0] == 'verify' for call in target.calls), response.text
            assert all(call[2] for call in target.calls)
            assert llm.speculative_calls or stream and not logprobs
            if logprobs:
                entries = [p['choices'][0]['logprobs'] for p in payloads
                           if p.get('choices') and p['choices'][0].get('logprobs')]
                assert entries, response.text
                if endpoint == 'completions':
                    assert sum(len(e['tokens']) for e in entries) == 12
                else:
                    assert sum(len(e['content']) for e in entries) == 12
    finally:
        api.close()


def test_logprob_requests_preserve_operator_mtp_rollback(monkeypatch):
    api, target, _ = driver(monkeypatch)
    llm = ResidentLLM(api)
    cfg = _config(speculative_provider='qwen4_exp_mtp', speculative_candidate_budget=3,
                  speculative_mtp_serving='opt_in', speculative_mtp_thinking='hard')
    try:
        with TestClient(create_app(cfg, llm=llm)) as client:
            assert client.post('/v1/hipengine/speculative_mtp/rollback').status_code == 200
            response = client.post('/v1/completions', json=dict(
                model='fake-model', prompt='one', max_tokens=12, temperature=.8,
                top_k=2, logprobs=3, seed=41, speculative_mtp=True))
            assert response.status_code == 200, response.text
            assert llm.ar_calls and not llm.speculative_calls
            assert not any(call[0] == 'verify' for call in target.calls)
            assert response.json()['hipengine']['speculative_mtp']['decision_reason'] == 'mtp_operator_rollback'
    finally:
        api.close()
