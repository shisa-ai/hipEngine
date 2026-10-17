"""Raw token bytes survive UTF-8 splits without inventing probabilities."""
from dataclasses import asdict

import pytest

from hipengine.generation.registry import TokenLogprob
from hipengine.tokenization.gguf import Qwen4ExpGGUFTokenizer, bytes_to_unicode
from tests.test_unit_qwen4exp_sampling import generator, request


def tokenizer():
    mapping = bytes_to_unicode()
    pieces = (b'\xe8\x88', b'\x90', '。'.encode(), b'A', b'<eos>')
    return Qwen4ExpGGUFTokenizer(
        tokens=tuple(''.join(mapping[b] for b in piece) for piece in pieces),
        merges=(), token_types=(1, 1, 1, 1, 3), eos_token_id=4,
    )


def fragments():
    return tuple(TokenLogprob(
        token_id=i, token_text=piece.decode('utf-8', errors='replace'),
        logprob=-.25, top_logprobs=((i, piece.decode('utf-8', errors='replace'), -.25),),
        token_bytes=tuple(piece), top_token_bytes=(tuple(piece),),
    ) for i, piece in enumerate((b'\xe8\x88', b'\x90', '。'.encode())))


def test_tokenizer_exposes_original_bytes_not_replacement_encoding():
    tok = tokenizer()
    assert tok.decode([0, 1, 2]) == '舐。'
    assert [tok.decode_token_bytes(i) for i in range(3)] == [b'\xe8\x88', b'\x90', '。'.encode()]
    assert tok.decode_token_bytes(4) == b'<eos>'
    for bad in (-1, 5):
        with pytest.raises(ValueError):
            tok.decode_token_bytes(bad)


def test_metadata_is_immutable_json_compatible_and_keeps_top_bytes():
    t = fragments()[0]
    assert TokenLogprob(**asdict(t)) == t
    assert t.token_bytes == (232, 136)
    assert t.top_token_bytes == ((232, 136),)
    with pytest.raises(ValueError):
        TokenLogprob(0, '', token_bytes=(256,))
    with pytest.raises(ValueError):
        TokenLogprob(0, '', top_token_bytes=((0,),))


@pytest.mark.parametrize('pieces,expected', [
    ((b'\xe8\x88', b'\x90', b'A'), ((0, 1), (0, 1), (1, 2))),
    ((b'\xe8', b'A'), ((0, 1), (1, 2))),
    ((b'\xe8', b'\xff', b'A'), ((0, 1), (1, 2), (2, 3))),
    ((b'\xe0', b'\x80'), ((0, 1), (1, 2))),
    ((b'\xef\xbf', b'\xbd'), ((0, 1), (0, 1))),
])
def test_fragment_spans_include_shared_chars_but_not_invalid_previous_prefix(pieces, expected):
    from hipengine.generation.logprobs import logprob_text_spans
    tokens = tuple(TokenLogprob(i, '', token_bytes=tuple(piece)) for i, piece in enumerate(pieces))
    assert tuple(logprob_text_spans(tokens)) == expected


@pytest.mark.parametrize('resident', [False, True])
def test_flash_sampling_captures_selected_and_all_top_bytes(resident):
    from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
    model = generator()
    model.tokenizer = tokenizer()
    api = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1)) if resident else model
    try:
        result = api.generate_detailed(request(max_tokens=3, temperature=0))[0]
        for t in result.token_logprobs:
            assert t.token_bytes == tuple(model.tokenizer.decode_token_bytes(t.token_id))
            assert t.top_token_bytes == tuple(tuple(model.tokenizer.decode_token_bytes(i)) for i, _, _ in t.top_logprobs)
    finally:
        api.close()
