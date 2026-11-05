"""Bounded worker ownership for synchronous producers consumed asynchronously."""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
import threading
from typing import Any


def _wake(loop: asyncio.AbstractEventLoop, event: asyncio.Event) -> None:
    try:
        loop.call_soon_threadsafe(event.set)
    except RuntimeError:
        # A disconnected consumer may already have closed its event loop.
        pass


class StreamBridgePool:
    """Limit dedicated producer threads without using the shared threadpool.

    Each stream buffers at most ``buffer_size`` items plus one producer-held
    item. Cancellation stops buffering immediately. Only the producer closes
    its iterator, after any in-flight ``next()`` returns. A blocked iterator
    therefore retains its worker slot; Python cannot safely terminate it.
    """

    def __init__(self, *, max_workers: int = 8, buffer_size: int = 8) -> None:
        if max_workers <= 0 or buffer_size <= 0:
            raise ValueError("stream worker and buffer limits must be positive")
        self._slots = threading.BoundedSemaphore(max_workers)
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    async def _acquire(self) -> None:
        loop = asyncio.get_running_loop()
        ready = asyncio.Event()
        waiter = (loop, ready)
        with self._lock:
            self._waiters.append(waiter)
        try:
            while True:
                ready.clear()
                if self._slots.acquire(blocking=False):
                    return
                await ready.wait()
        finally:
            with self._lock:
                self._waiters.remove(waiter)

    def _release(self) -> None:
        self._slots.release()
        with self._lock:
            for loop, ready in self._waiters:
                _wake(loop, ready)

    async def stream(self, factory: Callable[[], Iterator[Any]]) -> AsyncIterator[Any]:
        """Construct, advance and close the iterator on one owned worker."""
        await self._acquire()
        loop = asyncio.get_running_loop()
        ready = asyncio.Event()
        condition = threading.Condition()
        items: deque[Any] = deque()
        stopped = False
        done = False
        error: BaseException | None = None

        def produce() -> None:
            nonlocal done, error
            iterator = None
            try:
                iterator = iter(factory())
                while True:
                    with condition:
                        if stopped:
                            break
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    with condition:
                        condition.wait_for(lambda: stopped or len(items) < self._buffer_size)
                        if stopped:
                            break
                        items.append(item)
                        _wake(loop, ready)
            except BaseException as exc:
                error = exc
            finally:
                try:
                    closer = getattr(iterator, "close", None)
                    if callable(closer):
                        closer()
                except BaseException as exc:
                    if error is None:
                        error = exc
                finally:
                    with condition:
                        done = True
                        _wake(loop, ready)
                    self._release()

        worker = threading.Thread(target=produce, name="hipengine-stream", daemon=True)
        try:
            worker.start()
        except BaseException:
            self._release()
            raise
        try:
            while True:
                with condition:
                    ready.clear()
                    waiting = not items and not done
                    if items:
                        item = items.popleft()
                        condition.notify()
                    elif done:
                        if error is not None:
                            raise error
                        return
                if waiting:
                    await ready.wait()
                else:
                    yield item
        finally:
            with condition:
                stopped = True
                items.clear()
                condition.notify_all()
