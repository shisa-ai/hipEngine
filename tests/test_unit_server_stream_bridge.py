"""Ownership and backpressure for synchronous stream producers."""
from __future__ import annotations

import asyncio
import threading

import pytest

from hipengine.server.stream_bridge import StreamBridgePool


def test_first_ready_item_does_not_wait_for_next_item():
    async def check():
        release = threading.Event()
        entered = threading.Event()
        closed = threading.Event()

        def source():
            try:
                yield "first"
                entered.set()
                assert release.wait(5)
                yield "second"
            finally:
                closed.set()

        stream = StreamBridgePool(max_workers=1, buffer_size=1).stream(source)
        try:
            assert await asyncio.wait_for(anext(stream), 1) == "first"
        finally:
            release.set()
            await stream.aclose()
        assert await asyncio.to_thread(closed.wait, 1)

    asyncio.run(check())


def test_order_and_producer_failure_are_preserved():
    async def check():
        def source():
            yield 1
            yield 2
            raise ValueError("producer failed")

        seen = []
        with pytest.raises(ValueError, match="producer failed"):
            async for item in StreamBridgePool(max_workers=1).stream(source):
                seen.append(item)
        assert seen == [1, 2]

    asyncio.run(check())


def test_cancelled_blocked_producer_retains_capacity_and_owns_close():
    async def check():
        entered = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        second_started = threading.Event()
        threads = []

        def source():
            threads.append(threading.get_ident())
            try:
                entered.set()
                assert release.wait(5)
                yield 1
            finally:
                threads.append(threading.get_ident())
                closed.set()

        def second():
            second_started.set()
            yield 2

        pool = StreamBridgePool(max_workers=1, buffer_size=1)
        stream = pool.stream(source)
        pending = asyncio.create_task(anext(stream))
        assert await asyncio.to_thread(entered.wait, 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        other = pool.stream(second)
        other_pending = asyncio.create_task(anext(other))
        await asyncio.sleep(0)
        assert not second_started.is_set()
        assert not closed.is_set()
        release.set()
        assert await asyncio.wait_for(other_pending, 1) == 2
        await other.aclose()
        assert closed.is_set()
        assert len(threads) == 2 and threads[0] == threads[1]

    asyncio.run(check())


def test_buffer_bounds_production_and_consumer_disconnect_releases_slot():
    async def check():
        third = threading.Event()
        closed = threading.Event()

        def source():
            try:
                yield 0
                yield 1
                third.set()
                yield 2
                raise AssertionError("producer advanced past bounded pending item")
            finally:
                closed.set()

        pool = StreamBridgePool(max_workers=1, buffer_size=1)
        stream = pool.stream(source)
        assert await asyncio.wait_for(anext(stream), 1) == 0
        assert await asyncio.to_thread(third.wait, 1)
        await stream.aclose()
        # A buffered item plus one producer-held item is the upper bound.
        assert await asyncio.to_thread(closed.wait, 1)
        assert [x async for x in pool.stream(lambda: iter((9,)))] == [9]

    asyncio.run(check())


def test_none_items_and_factory_failure_release_capacity():
    async def check():
        pool = StreamBridgePool(max_workers=1, buffer_size=1)
        assert [x async for x in pool.stream(lambda: iter((None, 3)))] == [None, 3]

        def broken():
            raise LookupError("factory")

        with pytest.raises(LookupError, match="factory"):
            async for _ in pool.stream(broken):
                pass
        assert [x async for x in pool.stream(lambda: iter((4,)))] == [4]

    asyncio.run(check())


def test_waiting_consumer_cancellation_does_not_consume_a_slot():
    async def check():
        entered = threading.Event()
        release = threading.Event()
        pool = StreamBridgePool(max_workers=1, buffer_size=1)

        def source():
            entered.set()
            assert release.wait(5)
            yield 1

        first = pool.stream(source)
        pending = asyncio.create_task(anext(first))
        assert await asyncio.to_thread(entered.wait, 1)
        waiting = pool.stream(lambda: iter((2,)))
        cancelled = asyncio.create_task(anext(waiting))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        assert await asyncio.wait_for(pending, 1) == 1
        await first.aclose()
        assert [x async for x in pool.stream(lambda: iter((3,)))] == [3]

    asyncio.run(check())


def test_close_failure_propagates_and_releases_capacity():
    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            raise ValueError("close failed")

    async def check():
        pool = StreamBridgePool(max_workers=1)
        with pytest.raises(ValueError, match="close failed"):
            async for _ in pool.stream(Source):
                pass
        assert [x async for x in pool.stream(lambda: iter((1,)))] == [1]

    asyncio.run(check())


@pytest.mark.parametrize("many", [False, True])
def test_server_stream_uses_bounded_bridge_and_finalizes_on_disconnect(many, monkeypatch):
    from types import SimpleNamespace
    from hipengine.generation import GenerationStreamChunk
    from hipengine import SamplingParams
    from hipengine.server import api

    async def check():
        release = threading.Event()
        closed = threading.Event()

        def source(*_args):
            try:
                yield GenerationStreamChunk(text="ready")
                assert release.wait(5)
                yield GenerationStreamChunk(text="later")
            finally:
                closed.set()

        monkeypatch.setattr(api, "_STREAM_BRIDGE_POOL", StreamBridgePool(max_workers=1, buffer_size=1))
        if many:
            monkeypatch.setattr(api, "_engine_stream_many_callable", lambda _: source)
            stream = api._stream_engine_many(object(), ("a", "b"), SamplingParams())
        else:
            stream = api._stream_engine_text(SimpleNamespace(stream_detailed=source), "a", SamplingParams())
        try:
            assert (await asyncio.wait_for(anext(stream), 1)).text == "ready"
        finally:
            release.set()
            await stream.aclose()
        assert await asyncio.to_thread(closed.wait, 1)

    asyncio.run(check())


@pytest.mark.parametrize("workers,buffer", [(0, 1), (1, 0), (-1, 2)])
def test_pool_rejects_unbounded_geometry(workers, buffer):
    with pytest.raises(ValueError):
        StreamBridgePool(max_workers=workers, buffer_size=buffer)
