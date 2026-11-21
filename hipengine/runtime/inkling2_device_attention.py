"""Per-visit device KV state for Inkling 2 target attention."""

from contextlib import ExitStack

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc
from hipengine.core.tensor import Tensor
from hipengine.kvcache.spans import KVLiveSpans
from hipengine.loading.inkling2_activation_transfer import Inkling2ActivationTransfer


# A second launch starts paying for itself at 16 live positions on gfx1100.
_SPLIT_MIN_LIVE = 16


class Inkling2DeviceAttention:
    def __init__(self, *, runtime, launch, write, geometry, split=None, gate=None, device=None):
        self.rt, self.launch, self.write = runtime, launch, write
        self.split, self.gate = split, gate
        self.scores = 0
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
            self.transfer = Inkling2ActivationTransfer(runtime)
            owner.callback(self.transfer.close)
            q_copies = 1 if gate is None else 2
            scratch_values = 2 + max(nq * (q_copies * d + e + 1) + 2 * nk * d for _, nq, nk, d, e in self.geometry)
            self.staging = np.empty(scratch_values, dtype=np.float32)
            self.scratch = self._allocate(self.staging.nbytes)
            self.out = self._allocate(4 * max(nq * d for _, nq, _, d, _ in self.geometry))
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
            live_counts=Tensor(self.scratch, (1,), DType.INT32, self.device),
            max_live_count=cap,
            token_positions=Tensor(positions, (cap,), DType.INT32, self.device),
            evict_mask=Tensor(evicted, (cap,), DType.BOOL, self.device),
            row_positions=Tensor(self.scratch + 4, (1,), DType.INT32, self.device),
            storage_dtype=DType.FP32,
        )

    def step(self, visit, position, q, key, value, bias, sink, *, undo=None, gain=None, projection=None):
        if self.closed:
            raise RuntimeError("Inkling 2 attention state is closed")
        if position != self.positions.get(visit, 0):
            raise ValueError("Inkling 2 attention position must advance in order")
        cap, nq, nk, d, extent = self.geometry[visit]
        live = min(position + 1, cap)
        use_split = self.split is not None and live >= _SPLIT_MIN_LIVE
        expected = ((nq, d), (nk, d), (nk, d), (nq, extent), (nq,))
        if tuple(a.shape for a in (q, key, value, bias, sink)) != expected:
            raise ValueError("invalid Inkling 2 attention input shapes")
        if gain is not None and (self.gate is None or gain.shape != (nq * d,)):
            raise ValueError("invalid Inkling 2 attention gain or unavailable device gate")
        if use_split and not self.scores:
            self.scores = self._allocate(4 * max(capacity * heads for capacity, heads, *_ in self.geometry))
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
        self.staging[:2].view(np.int32)[:] = (live, position)
        ptrs = []
        cursor = 2
        for array in (q, key, value, bias, sink):
            ptrs.append(self.scratch + cursor * 4)
            end = cursor + array.size
            self.staging[cursor:end].reshape(array.shape)[...] = array
            cursor = end
        q_ptr, key_ptr, value_ptr, bias_ptr, sink_ptr = ptrs
        if gain is not None:
            gain_ptr = self.scratch + cursor * 4
            self.staging[cursor:cursor + gain.size] = gain
            cursor += gain.size
        args = (q_ptr, keys, values, bias_ptr, sink_ptr, self.out, spans, nq, nk, d, extent)
        if projection is None:
            dst, out, consume = self.out, np.empty((nq, d), dtype=np.float32), None
        else:
            dst, out, consume = projection
        with self.transfer.exchange(self.scratch, dst, self.staging[:cursor], out, pinned=True):
            self.write(key_ptr, value_ptr, keys, values, spans, nk * d)
            if use_split:
                self.split(*args, self.scores, live)
            else:
                self.launch(*args)
            if gain is not None:
                self.gate(self.out, gain_ptr, gain.size)
            if consume is not None:
                consume(self.out)
        self.positions[visit] = position + 1
        return out

    def close(self):
        self.closed = True
        self.owner.close()
        self.banks.clear()
        self.positions.clear()
