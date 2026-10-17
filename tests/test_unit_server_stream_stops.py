"""Text stops must suppress publication without buffering safe stream prefixes."""
import asyncio

import pytest

pytest.importorskip('fastapi')
from hipengine.generation import GenerationStreamChunk, FinishDetails, TokenLogprob
from hipengine.server.api import _stop_filtered_stream


@pytest.mark.parametrize('parts,stops,expected', [
    (['alpha', 'E', 'N', 'D', 'ignored'], ['END'], 'alpha'),
    (['a', 'b', 'x', 'ab'], ['abc'], 'abxab'),
    (['回答。', '終', 'わ', 'り', '尾'], ['終わり'], '回答。'),
    (['alphaENDignored'], ['END', 'ND'], 'alpha'),
    (['ab', 'ab', 'x'], ['abab'], ''),
    (['a', 'b', 'c', 'd'], ['abcd', 'bc'], ''),
    (['a', 'b', 'c', 'x'], ['abcd', 'bc'], 'a'),
    (['a', 'b', 'c'], ['abcd', 'bc'], 'a'),
    (['a', 'b'], [], 'ab'),
])
def test_stop_stream_holds_only_possible_stop_prefix(parts, stops, expected):
    async def run():
        consumed = []
        async def source():
            for i, text in enumerate(parts):
                consumed.append(text)
                yield GenerationStreamChunk(text=text, generated_token_ids=tuple(range(i + 1)))
            yield GenerationStreamChunk(text='', generated_token_ids=tuple(range(len(parts))),
                                        finish_details=FinishDetails(reason='length'))
        output = []
        async for chunk in _stop_filtered_stream(source(), stops):
            output.append(chunk)
            if chunk.text == 'alpha' and len(parts) > 1:
                assert consumed == ['alpha'], 'safe prefix must not wait for generation completion'
        assert ''.join(c.text for c in output) == expected
        assert output[-1].generated_token_ids == tuple(range(len(parts)))
        matched = expected != ''.join(parts)
        assert output[-1].finish_details.reason == ('stop' if matched else 'length')
    asyncio.run(run())


@pytest.mark.parametrize('tail,expected', [('x', 'alphaEx'), ('NDtail', 'alpha')])
def test_stop_inside_token_does_not_publish_partial_token_probability(tail, expected):
    from hipengine.server.api import _completion_stream_logprobs
    async def run():
        async def source():
            for index, text in enumerate(('alphaE', tail)):
                yield GenerationStreamChunk(text=text, token_logprobs=(TokenLogprob(index, text, -.25),))
        chunks = [c async for c in _stop_filtered_stream(source(), 'END')]
        assert ''.join(c.text for c in chunks) == expected
        logs = [t for c in chunks if c.text for t in _completion_stream_logprobs(c)['tokens']]
        assert logs == (['alphaE', tail] if tail == 'x' else [])
    asyncio.run(run())


def test_stop_filter_closes_source_on_disconnect_and_propagates_error():
    async def run():
        closed = []
        async def source():
            try:
                yield GenerationStreamChunk(text='safe')
                raise RuntimeError('driver failed')
            finally:
                closed.append(True)
        stream = _stop_filtered_stream(source(), 'END')
        assert (await anext(stream)).text == 'safe'
        await stream.aclose()
        assert closed == [True]
        stream = _stop_filtered_stream(source(), 'END')
        await anext(stream)
        with pytest.raises(RuntimeError, match='driver failed'):
            await anext(stream)
        assert closed == [True, True]
    asyncio.run(run())
