"""Per-visit device KV state for Inkling 2 target attention."""

from contextlib import ExitStack

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc
from hipengine.core.tensor import Tensor
from hipengine.kvcache.spans import KVLiveSpans


class Inkling2DeviceAttention:
    def __init__(self, *, runtime, launch, geometry, device=None):
        self.rt, self.launch = runtime, launch
        self.device = Device("hip") if device is None else device
        self.geometry = tuple(geometry)  # (capacity, query heads, KV heads, head width, bias extent) by state slot
        self.banks = {}
        self.positions = {}
        if not self.geometry or any(cap <= 0 or nq <= 0 or nk <= 0 or nq % nk or not 0 < d <= 128 or e <= 0
                                    for cap, nq, nk, d, e in self.geometry):
            raise ValueError("unsupported gfx1100 Inkling 2 target attention geometry")
        self.closed = False
        with ExitStack() as owner:
            self.owner = owner
            self.q = self._allocate(4 * max(nq * d for _, nq, _, d, _ in self.geometry))
            self.bias = self._allocate(4 * max(nq * e for _, nq, _, _, e in self.geometry))
            self.sink = self._allocate(4 * max(nq for _, nq, _, _, _ in self.geometry))
            self.out = self._allocate(4 * max(nq * d for _, nq, _, d, _ in self.geometry))
            self.meta = self._allocate(8)
            self.owner = owner.pop_all()

    def _allocate(self, size):
        buf = malloc(size, runtime=self.rt)
        self.owner.callback(free, buf, runtime=self.rt)
        return buf.ptr

    def _upload(self, ptr, value, dtype):
        arr = np.ascontiguousarray(value, dtype=dtype)
        self.rt.memcpy(ptr, arr.ctypes.data, arr.nbytes, HipMemcpyKind.HOST_TO_DEVICE)

    def _make_bank(self, cap, nk, d):
        keys, values = (self._allocate(cap * nk * d * 4) for _ in range(2))
        offsets = self._allocate(cap * 4)
        positions = self._allocate(cap * 4)
        evicted = self._allocate(cap)
        self._upload(offsets, np.arange(cap), np.int32)
        self.rt.memset(evicted, 0, cap)
        return keys, values, KVLiveSpans(
            base_offsets=Tensor(offsets, (cap,), DType.INT32, self.device),
            live_counts=Tensor(self.meta, (1,), DType.INT32, self.device),
            max_live_count=cap,
            token_positions=Tensor(positions, (cap,), DType.INT32, self.device),
            evict_mask=Tensor(evicted, (cap,), DType.BOOL, self.device),
            row_positions=Tensor(self.meta + 4, (1,), DType.INT32, self.device),
            storage_dtype=DType.FP32,
        )

    def step(self, visit, position, q, key, value, bias, sink, *, undo=None):
        if self.closed:
            raise RuntimeError("Inkling 2 attention state is closed")
        if position != self.positions.get(visit, 0):
            raise ValueError("Inkling 2 attention position must advance in order")
        cap, nq, nk, d, extent = self.geometry[visit]
        expected = ((nq, d), (nk, d), (nk, d), (nq, extent), (nq,))
        if tuple(a.shape for a in (q, key, value, bias, sink)) != expected:
            raise ValueError("invalid Inkling 2 attention input shapes")
        if visit not in self.banks:
            self.banks[visit] = self._make_bank(cap, nk, d)
        keys, values, spans = self.banks[visit]
        slot = position % cap
        if undo is not None:
            undo.before(position, self.positions.__setitem__, visit, position)
            if position >= cap:
                for ptr in (keys, values):
                    undo.device(position, self.rt, ptr + slot * nk * d * 4, nk * d * 4)
                undo.device(position, self.rt, spans.token_positions.ptr + slot * 4, 4)
        self._upload(keys + slot * nk * d * 4, key, np.float32)
        self._upload(values + slot * nk * d * 4, value, np.float32)
        self._upload(spans.token_positions.ptr + slot * 4, [position], np.int32)
        self._upload(self.meta, [min(position + 1, cap), position], np.int32)
        self._upload(self.q, q, np.float32)
        self._upload(self.bias, bias, np.float32)
        self._upload(self.sink, sink, np.float32)
        self.launch(self.q, keys, values, self.bias, self.sink, self.out, spans, nq, nk, d, extent)
        out = np.empty((nq, d), dtype=np.float32)
        self.rt.memcpy(out.ctypes.data, self.out, out.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        self.positions[visit] = position + 1
        return out

    def close(self):
        self.closed = True
        self.owner.close()
        self.banks.clear()
        self.positions.clear()
