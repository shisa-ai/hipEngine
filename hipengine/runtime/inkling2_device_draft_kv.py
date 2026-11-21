"""Committed HIP draft KV rings. Proposal rows only exist in scratch."""

from contextlib import ExitStack

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc


class Inkling2DeviceDraftKV:
    def __init__(self, capacity, cfg, *, runtime, attend):
        if capacity <= 0:
            raise ValueError("Inkling 2 DSpark KV capacity must be positive")
        self.capacity = capacity
        self.rt, self._attend = runtime, attend
        self.shape = cfg["num_key_value_heads"], cfg["head_dim"]
        self.row_bytes = self.shape[0] * self.shape[1] * 4
        local = set(cfg["local_layer_ids"])
        self.lengths = tuple(min(capacity, cfg["sliding_window_size"]) if i in local else capacity
                             for i in range(cfg["num_hidden_layers"]))
        self.positions = [0 for _ in self.lengths]
        self.closed = False
        with ExitStack() as owned:
            def allocate(length):
                b = malloc(length * self.row_bytes, runtime=runtime)
                owned.callback(free, b, runtime=runtime)
                return b.ptr

            self.banks = tuple((allocate(n), allocate(n)) for n in self.lengths)
            self._owned = owned.pop_all()

    def append(self, layer, position, key, value):
        if self.closed or position != self.positions[layer] or position >= self.capacity:
            raise ValueError("Inkling 2 DSpark KV append is outside the live cursor")
        if key.shape != self.shape or value.shape != self.shape:
            raise ValueError("invalid Inkling 2 DSpark KV row geometry")
        offset = (position % self.lengths[layer]) * self.row_bytes
        for ptr, data in zip(self.banks[layer], (key, value)):
            data = np.ascontiguousarray(data, dtype=np.float32)
            self.rt.memcpy(ptr + offset, data.ctypes.data, self.row_bytes, HipMemcpyKind.HOST_TO_DEVICE)
        self.positions[layer] += 1

    def attend(self, layer, start, end, q, k, v):
        if self.closed or end != self.positions[layer] or not 0 <= start <= end or end - start > self.lengths[layer]:
            raise ValueError("Inkling 2 DSpark KV read is outside the live prefix")
        if k.shape[1:] != self.shape:
            raise ValueError("invalid Inkling 2 DSpark proposal KV geometry")
        n = self.lengths[layer]
        slot = start % n
        first = min(end - start, n - slot)
        kp, vp = self.banks[layer]
        offset = slot * self.row_bytes
        prefix = []
        if first:
            prefix.append((kp + offset, vp + offset, first))
        if first < end - start:
            prefix.append((kp, vp, end - start - first))
        return self._attend(q, k, v, prefix=prefix)

    def close(self):
        self.closed = True
        self._owned.close()
