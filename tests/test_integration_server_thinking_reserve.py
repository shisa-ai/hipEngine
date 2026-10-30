"""Lower real close-token cost before promising an answer reserve."""
import json
import pytest
from fastapi.testclient import TestClient

from hipengine.server.api import (
    OpenAIHTTPError, ServerConfig,
    _thinking_budget_sampling_kwargs, create_app,
)
from hipengine.server.schemas import ChatCompletionRequest
from tests.support.server_api_fakes import FakeLLM


def lower(**fields):
    config = ServerConfig(model='fake', served_model_name='fake-model')
    engine = FakeLLM(token_map={'</think>': [42, 43, 44]})
    req = ChatCompletionRequest(model='fake-model', messages=[{'role':'user','content':'Think then answer.'}],
                                enable_thinking=True, **fields)
    return _thinking_budget_sampling_kwargs(config, req, engine=engine, max_context_tokens=None)


def test_max_think_tokens_alone_sets_an_enforced_cap():
    assert lower(max_think_tokens=4, max_tokens=20)['thinking_hard_token_cap'] == 4


def test_answer_reserve_includes_complete_tokenized_close():
    lowered = lower(hard_think_cap=20, min_answer_tokens=4, max_tokens=12)
    assert lowered['thinking_hard_token_cap'] == 5
    assert lowered['thinking_close_token_ids'] == (42, 43, 44)


def test_answer_reserve_alone_enables_close_control():
    assert lower(min_answer_tokens=4, max_tokens=12)['thinking_hard_token_cap'] == 5


def test_unbounded_effort_still_preserves_answer_reserve():
    assert lower(reasoning_effort='low', thinking={'allow_unbounded': True},
                 max_tokens=12)['thinking_hard_token_cap'] == 3


def test_zero_cap_keeps_zero_but_reserves_close_space():
    assert lower(hard_think_cap=0, min_answer_tokens=2, max_tokens=8)['thinking_hard_token_cap'] == 0


def test_zero_cap_without_answer_override_is_accepted():
    assert lower(hard_think_cap=0, max_tokens=8)['thinking_hard_token_cap'] == 0


def test_unbounded_without_reserve_does_not_add_a_cap():
    assert lower(thinking={'allow_unbounded': True}, min_answer_tokens=0, max_tokens=12) == {}


def test_closure_cannot_silently_consume_entire_answer_reserve():
    with pytest.raises(OpenAIHTTPError) as error:
        lower(hard_think_cap=0, min_answer_tokens=2, max_tokens=4)
    assert error.value.status_code == 400


@pytest.mark.parametrize('stream', [False, True])
def test_impossible_close_is_rejected_before_generation(stream):
    fake = FakeLLM(token_map={'</think>': [42, 43, 44]})
    client = TestClient(create_app(ServerConfig(model='fake', served_model_name='fake-model'), llm=fake))
    result = client.post('/v1/chat/completions', json={
        'model':'fake-model', 'messages':[{'role':'user','content':'Think.'}],
        'hard_think_cap':0, 'min_answer_tokens':2, 'max_tokens':4, 'stream':stream,
    })
    if stream:
        assert result.status_code == 200  # SSE headers precede async admission.
        events = [json.loads(line[6:]) for line in result.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        error = next(event['error'] for event in events if 'error' in event)
        assert error['code'] == 'invalid_request'
        assert error['param'] == 'max_tokens'
    else:
        assert result.status_code == 400
    assert fake.calls == []
    assert fake.stream_calls == []
