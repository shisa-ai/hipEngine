"""Request-local stop buffering preserves backend terminal ownership."""
import asyncio

from hipengine.generation.registry import FinishDetails, GenerationStreamChunk, TokenLogprob
from hipengine.server.api import _LiteralStopStream, _iterate_with_literal_stops, _literal_stop_terminal_chunk


def collect(chunks, stop):
    async def source():
        for chunk in chunks:
            yield chunk
    state = _LiteralStopStream(stop)
    async def run():
        return [chunk async for chunk in _iterate_with_literal_stops(source(), None, state)]
    return asyncio.run(run()), state


def test_filter_keeps_terminal_ids_and_finish_after_stop():
    finish = FinishDetails(reason='length', length_limit=8)
    terminal = GenerationStreamChunk(text='', generated_token_ids=(1, 2, 3), finish_details=finish)
    chunks, state = collect([GenerationStreamChunk(text='prefix<'),
                            GenerationStreamChunk(text='stop>tail'), terminal], '<stop>')
    assert ''.join(c.text for c in chunks) == 'prefix'
    assert state.raw_text == 'prefix<stop>tail'
    assert chunks[-1].generated_token_ids == (1, 2, 3)
    final = _literal_stop_terminal_chunk(chunks[-1], state.matched)
    assert final.finish_details.reason == 'stop'
    assert final.finish_details.length_limit is None
    assert final.generated_token_ids == (1, 2, 3)


def test_no_stop_preserves_chunk_objects_and_does_not_copy_raw_text():
    chunks = [GenerationStreamChunk(text='a'), GenerationStreamChunk(text='b')]
    result, state = collect(chunks, None)
    assert all(a is b for a, b in zip(result, chunks, strict=True))
    assert state.raw_parts == []


def test_pending_prefix_flushes_without_replacing_terminal_metadata():
    terminal = GenerationStreamChunk(text='', generated_token_ids=(1, 2),
                                    finish_details=FinishDetails(reason='length', length_limit=2))
    result, state = collect([GenerationStreamChunk(text='alpha<st'), terminal], '<stop>')
    assert ''.join(c.text for c in result) == 'alpha<st'
    assert result[-1].generated_token_ids == (1, 2)
    assert not state.matched


def test_request_stop_state_is_not_shared():
    a = _LiteralStopStream('<stop>')
    b = _LiteralStopStream('<stop>')
    assert a.feed(GenerationStreamChunk(text='<st')) == []
    assert ''.join(c.text for c in b.feed(GenerationStreamChunk(text='op>'))) == 'op>'
    assert a.matched is b.matched is False


def test_partial_token_logprob_is_explicitly_omitted_not_invented():
    token = TokenLogprob(token_id=42, token_text='alphaSTOPtail', logprob=-0.5)
    result, state = collect([GenerationStreamChunk(text=token.token_text, token_logprobs=(token,))], 'STOP')
    visible = [c for c in result if c.text]
    assert len(visible) == 1
    assert visible[0].text == 'alpha'
    record = visible[0].token_logprobs[0]
    assert record.token_id == 42 and record.token_text == 'alpha'
    assert record.logprob is None and record.top_logprobs == ()
    assert state.matched
