"""CPU-only request bounds and transcription preservation regressions."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.runtime import vibevoice_qwen2 as q
from scripts.vibevoice_asr_e2e import parse_transcript


def runner(monkeypatch, capacity=2):
    r = SimpleNamespace(
        max_context=capacity, spec=SimpleNamespace(hidden_size=1), reset=Mock(),
        prefill_host_rows=Mock(), prefill_rows=Mock(), runtime=SimpleNamespace(memcpy=Mock()),
        _hidden=SimpleNamespace(ptr=1), logits_argmax=Mock(return_value=(None, 1)),
        embed_row=Mock(return_value=np.array([1.])), push_token=Mock(), forward_layers=Mock(),
    )
    monkeypatch.setattr(q, '_upload', Mock(return_value=SimpleNamespace(ptr=1)))
    monkeypatch.setattr(q, 'free', Mock())
    return r


def test_final_token_does_not_consume_cache_slot(monkeypatch):
    r = runner(monkeypatch)
    assert q.greedy_generate(r, [np.array([1.])] * 2, max_new_tokens=1) == [1]
    r.forward_layers.assert_not_called()


@pytest.mark.parametrize('budget', [-1, 1.5, True, 3])
def test_invalid_budget_rejected_before_device_work(monkeypatch, budget):
    r = runner(monkeypatch)
    with pytest.raises(ValueError):
        q.greedy_generate(r, [np.array([1.])] * 2, max_new_tokens=budget)
    q._upload.assert_not_called()
    r.reset.assert_not_called()


def test_zero_budget_has_no_device_work(monkeypatch):
    r = runner(monkeypatch)
    assert q.greedy_generate(r, [np.array([1.])], max_new_tokens=0) == []
    q._upload.assert_not_called()


def test_decode_only_advances_for_needed_tokens(monkeypatch):
    r = runner(monkeypatch, capacity=3)
    assert q.greedy_generate(r, [np.array([1.])] * 2, max_new_tokens=2) == [1, 1]
    r.forward_layers.assert_called_once_with(2)


@pytest.mark.parametrize('position', [-1, 2, 1.5, True])
@pytest.mark.parametrize('method', ['push_token', 'forward_layers'])
def test_direct_decode_rejects_invalid_positions(position, method):
    r = q.VibevoiceQwen2Runtime.__new__(q.VibevoiceQwen2Runtime)
    r.max_context = 2
    with pytest.raises(ValueError, match='position'):
        if method == 'push_token':
            r.push_token(np.array([1.]), position)
        else:
            r.forward_layers(position)


def test_transcript_preserves_assistant_in_content():
    text = '[{"Start":0,"End":1,"Speaker":0,"Content":"my assistant called"}]'
    assert parse_transcript(text)[0]['Content'] == 'my assistant called'


@pytest.mark.parametrize('text', [
    '[1]', '[{}]', '[{"Start":2,"End":1,"Speaker":0,"Content":"x"}]',
    '[{"Start":0,"End":1,"Speaker":0,"Content":"x"}] trailing',
    '[{"Start":0,"End":NaN,"Speaker":0,"Content":"x"}]',
])
def test_malformed_output_is_not_success(text):
    assert parse_transcript(text) is None
