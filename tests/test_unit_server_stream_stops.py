"""Literal stops must publish safe prefixes early and close the backend source."""
import asyncio

import pytest

pytest.importorskip('fastapi')
from hipengine.generation import GenerationStreamChunk, FinishDetails
from hipengine.server.api import _LiteralStopStream, _iterate_with_literal_stops


def _stream(source, stop):
    state = _LiteralStopStream(stop)
    return _iterate_with_literal_stops(source(), None, state), state


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

        stream, state = _stream(source, stops)
        output = []
        async for chunk in stream:
            output.append(chunk)
            if chunk.text == 'alpha' and len(parts) > 1:
                assert consumed == ['alpha'], 'safe prefix must not wait for generation completion'
        assert ''.join(c.text for c in output) == expected
        assert output[-1].generated_token_ids == tuple(range(len(parts)))
        assert state.matched == (expected != ''.join(parts))
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

        stream, _ = _stream(source, 'END')
        assert (await anext(stream)).text == 'safe'
        await stream.aclose()
        assert closed == [True]

        stream, _ = _stream(source, 'END')
        await anext(stream)
        with pytest.raises(RuntimeError, match='driver failed'):
            await anext(stream)
        assert closed == [True, True]
    asyncio.run(run())


def test_stop_filter_without_stops_forwards_every_chunk_unbuffered():
    async def run():
        async def source():
            for text in ('a', 'b', 'c'):
                yield GenerationStreamChunk(text=text)

        stream, state = _stream(source, None)
        assert [c.text async for c in stream] == ['a', 'b', 'c']
        assert state.stops == ()
    asyncio.run(run())
