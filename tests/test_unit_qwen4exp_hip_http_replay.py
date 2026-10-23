"""Ordered depth adaptation must reuse exact prompt checkpoints and native counts."""
from types import SimpleNamespace

import pytest

from scripts import qwen4exp_hip_http_replay as replay


class Response:
    def __init__(self, body):
        self.body = body
    def raise_for_status(self):
        pass
    def json(self):
        return self.body


class Client:
    def __init__(self, rendered):
        self.rendered = iter(rendered)
    def post(self, url, json):
        if url.endswith('count_tokens'):
            self.current = next(self.rendered)
            assert json['enable_thinking'] is False
            return Response({'text': 'rendered', 'token_count': len(self.current)})
        assert url.endswith('tokenize')
        return Response({'token_ids': self.current, 'token_count': len(self.current)})


def result(ids, count, mtp, cached=0):
    return {'status': 200, 'prompt_tokens': len(ids), 'completion_tokens': count,
            'speculative_requested': mtp, 'speculative_used': mtp,
            'text': 'reply', 'message': None,
            'token_accounting': {'choice_generated_token_ids': [[7] * count]},
            'diagnostics': {'proposed_draft_tokens': 2 if mtp else 0, 'accepted_draft_tokens': 1 if mtp else 0},
            'phase_metrics': {'counts': {'prompt_tokens': len(ids), 'cached_prompt_tokens': cached,
                'new_prompt_tokens': len(ids) - cached, 'generated_tokens': count, 'decode_transitions': count - 1},
                'times_ms': {'prefill_wall_ms': 20.0, 'decode_wall_ms': 10.0}}}


@pytest.mark.parametrize('mtp', [False, True])
def test_prepares_only_rendered_history_prefix_before_measured_request(monkeypatch, tmp_path, mtp):
    calls = []
    def one(client, url, model, row, flag, count, **kw):
        calls.append((row, flag, count, kw))
        ids = row['prompt_token_ids']
        return result(ids, count, flag, cached=3 if count == 128 else 0)
    monkeypatch.setattr(replay, '_one', one)
    session = replay.DepthSession(Client([[1,2,3,4,5], [1,2,3,9]]), 'model', mtp, tmp_path/'journal')
    messages = [{'role':'user','content':'old'}, {'role':'assistant','content':'reply'},
                {'role':'user','content':'new'}]
    observed = session.request('url', 'new', 128, messages=messages)
    assert calls[0][0]['prompt_token_ids'] == [1,2,3] and calls[0][2] == 1
    assert calls[1][0]['prompt_token_ids'] == [1,2,3,4,5] and calls[1][2] == 128
    assert all(c[1] is mtp and c[3]['ignore_eos'] is False for c in calls)
    assert observed.cached_prompt_tokens == 3 and observed.prefill_tokens == 2
    assert observed.completion_tokens == 128
    assert observed.public()['checkpoint_tokens'] == 3
    assert observed.draft_tokens == (2 if mtp else 0)
    assert (tmp_path/'journal').read_text().count('\n') == 2


def test_cache_miss_is_not_a_depth_measurement(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, '_one', lambda c,u,m,row,flag,n,**kw: result(row['prompt_token_ids'], n, flag))
    session = replay.DepthSession(Client([[1,2,3,4], [1,2,9]]), 'm', False, tmp_path/'journal')
    with pytest.raises(ValueError, match='checkpoint'):
        session.request('url', 'new', 128, messages=[{'role':'user','content':'x'},
            {'role':'assistant','content':'y'}, {'role':'user','content':'new'}])


def test_short_prefix_reply_is_retained_but_measurement_still_requires_full_count(monkeypatch, tmp_path):
    calls = []
    def one(c,u,m,row,flag,n,**kw):
        calls.append(kw)
        return result(row['prompt_token_ids'], 2, flag)
    monkeypatch.setattr(replay, '_one', one)
    session = replay.DepthSession(Client([[1,2,3]]), 'm', False, tmp_path/'journal')
    assert session.chat_text('url', [{'role':'user','content':'prefix'}], 8) == 'reply'
    assert calls == [{'ignore_eos': False, 'require_max_tokens': False}]


def test_route_failure_is_recorded_not_labelled_mtp(monkeypatch, tmp_path):
    def one(c,u,m,row,flag,n,**kw):
        return {**result(row['prompt_token_ids'], n, flag), 'speculative_used': False}
    monkeypatch.setattr(replay, '_one', one)
    session = replay.DepthSession(Client([[1,2]]), 'm', True, tmp_path/'journal')
    with pytest.raises(ValueError, match='route'):
        session.request('url', 'prompt', 128)
    assert 'failure' in (tmp_path/'journal').read_text()


@pytest.mark.parametrize('full,history', [([1,2],[3,4]), ([1,2],[1,2,3])])
def test_checkpoint_cannot_be_empty_or_include_entire_measured_prompt(full, history):
    with pytest.raises(ValueError, match='prefix'):
        replay.checkpoint_prefix(full, history)


def test_missing_phase_data_cannot_make_an_observation(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, '_one', lambda *a, **kw: {
        'status':200,'speculative_requested':False,'speculative_used':False,'phase_metrics':None})
    session = replay.DepthSession(Client([[1,2]]), 'm', False, tmp_path/'journal')
    with pytest.raises(ValueError, match='phase'):
        session.request('url', 'prompt', 128)
