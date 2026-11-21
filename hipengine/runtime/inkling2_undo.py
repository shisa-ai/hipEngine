"""Undo records for provisional target KV and convolution writes."""

from contextlib import ExitStack

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc


class Inkling2UndoJournal:
    def __init__(self):
        self.records = []
        self.allocations = ExitStack()

    def before(self, position, restore, *args):
        self.records.append((position, restore, args))

    def host(self, position, array):
        self.before(position, self._restore_array, array, array.copy())

    @staticmethod
    def _restore_array(array, saved):
        array[...] = saved

    def device(self, position, runtime, dst, size):
        saved = malloc(size, runtime=runtime)
        self.allocations.callback(free, saved, runtime=runtime)
        runtime.memcpy(saved.ptr, dst, size, HipMemcpyKind.DEVICE_TO_DEVICE)
        self.before(position, runtime.memcpy, dst, saved.ptr, size, HipMemcpyKind.DEVICE_TO_DEVICE)

    def finish(self, first_rejected):
        try:
            for position, restore, args in reversed(self.records):
                if position >= first_rejected:
                    restore(*args)
        finally:
            self.records.clear()
            self.allocations.close()
