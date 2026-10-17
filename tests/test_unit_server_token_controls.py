"""Token control lowering uses model metadata, not artifact identity."""
from types import SimpleNamespace

import pytest

from hipengine.server import api


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('eos', [0, 99])
def test_min_tokens_resolves_model_eos(wrapped, eos):
    target = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=eos))
    engine = SimpleNamespace(_text_generator=target) if wrapped else target
    request = api.CompletionRequest(prompt='x', min_tokens=2)
    assert api._request_eos_token_id(request, engine) == eos


@pytest.mark.parametrize('min_tokens,explicit,expected', [(0, None, None), (2, 7, 7)])
def test_eos_resolution_preserves_override_and_fast_path(min_tokens, explicit, expected):
    def forbidden():
        raise AssertionError('must not initialize tokenizer')
    engine = SimpleNamespace(_get_text_generator=forbidden)
    request = api.CompletionRequest(prompt='x', min_tokens=min_tokens, eos_token_id=explicit)
    assert api._request_eos_token_id(request, engine) == expected


def test_min_tokens_initializes_lazy_metadata():
    target = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=19))
    engine = SimpleNamespace(_get_text_generator=lambda: target)
    request = api.CompletionRequest(prompt='x', min_tokens=2)
    assert api._request_eos_token_id(request, engine) == 19


def test_min_tokens_without_any_eos_fails_with_named_parameter():
    with pytest.raises(api.OpenAIHTTPError) as exc:
        api._request_eos_token_id(api.CompletionRequest(prompt='x', min_tokens=2), object())
    assert exc.value.param == 'min_tokens'
    assert 'eos_token_id' in str(exc.value)


def test_stop_lowering_covers_space_and_newline_contexts():
    # A BPE token for a word is not its space-prefixed token. Likewise a
    # multi-token spelling can merge differently after leading whitespace.
    tokens = {'that': (10,), ' that': (11,), '\nthat': (12, 10),
              'ensures your': (20, 21, 22), ' ensures your': (23, 22),
              '\nensures your': (12, 20, 21, 22)}
    engine = SimpleNamespace(tokenize=tokens.__getitem__)
    ids, sequences = api._stop_tokens_from_stop(['that', 'ensures your'], engine)
    assert ids == (10, 11)
    assert sequences == ((12, 10), (20, 21, 22), (23, 22), (12, 20, 21, 22))


def test_stop_lowering_deduplicates_and_keeps_partial_tokenizers():
    tokens = {'that': (10,), ' that': (10,)}
    engine = SimpleNamespace(tokenize=tokens.__getitem__)
    assert api._stop_tokens_from_stop(['that', 'that', '', 'unknown'], engine) == ((10,), ())


def test_context_variants_are_not_added_to_explicit_whitespace_stops():
    calls = []
    def tokenize(text):
        calls.append(text)
        return (1, 2)
    assert api._stop_tokens_from_stop('\nEND', SimpleNamespace(tokenize=tokenize)) == ((), ((1, 2),))
    assert calls == ['\nEND']
