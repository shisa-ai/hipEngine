"""Per-visit device KV state for Inkling 2 target attention."""

from contextlib import ExitStack
from dataclasses import replace

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
    def __init__(self, *, runtime, launch, write, geometry, split=None, block=None, device=None):
        self.rt, self.launch, self.write = runtime, launch, write
        self.split, self.block = split, block
        self._block_scratch = None
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
            scratch_values = 2 + max(nq * (2 * d + e + 1) + 2 * nk * d for _, nq, nk, d, e in self.geometry)
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
        kv_bytes = cap * nk * d * 4
        # Preserve vector alignment for both FP32 arrays, including odd geometries.
        values_offset = (kv_bytes + 15) & ~15
        offsets_offset = values_offset + kv_bytes
        positions_offset = offsets_offset + cap * 4
        evicted_offset = positions_offset + cap * 4
        keys = self._allocate(evicted_offset + cap)
        values = keys + values_offset
        offsets = keys + offsets_offset
        positions = keys + positions_offset
        evicted = keys + evicted_offset
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

    def reserve_capacity(self):
        """Allocate the banks and depth-sized scores without advancing a visit."""
        if self.closed:
            raise RuntimeError("Inkling 2 attention state is closed")
        for visit, (cap, _, kv_heads, dim, _) in enumerate(self.geometry):
            if visit not in self.banks:
                self.banks[visit] = self._make_bank(cap, kv_heads, dim)
        if self.split is not None and any(cap >= _SPLIT_MIN_LIVE for cap, *_ in self.geometry):
            self._reserve_scores()

    def _reserve_scores(self):
        if not self.scores:
            self.scores = self._allocate(4 * max(capacity * heads for capacity, heads, *_ in self.geometry))

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
        if gain is not None and gain.shape != (nq * d,):
            raise ValueError("invalid Inkling 2 attention gain shape")
        if use_split:
            self._reserve_scores()
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
        gain_ptr = 0
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
                self.split(*args, self.scores, live, gain=gain_ptr)
            else:
                self.launch(*args, gain=gain_ptr)
            if consume is not None:
                consume(self.out)
        self.positions[visit] = position + 1
        return out

    def step_block(self, visit, position, q, key, value, bias, sink, *, undo=None):
        if self.closed:
            raise RuntimeError("Inkling 2 attention state is closed")
        if position != self.positions.get(visit, 0):
            raise ValueError("Inkling 2 attention position must advance in order")
        cap, nq, nk, d, extent = self.geometry[visit]
        rows = len(q)
        expected = ((rows, nq, d), (rows, nk, d), (rows, nk, d), (rows, nq, extent), (rows, nq))
        arrays = q, key, value, bias, sink
        if not 0 < rows <= 65535 or tuple(a.shape for a in arrays) != expected:
            raise ValueError("invalid Inkling 2 attention block input shapes")
        if self.block is None or undo is not None:
            return np.stack([self.step(visit, position + i, *a, undo=undo)
                             for i, a in enumerate(zip(*arrays, strict=True))])
        if visit not in self.banks:
            self.banks[visit] = self._make_bank(cap, nk, d)
        keys, values, spans = self.banks[visit]
        # The farthest lag in this block cannot exceed either the final row's
        # absolute position or the live ring length minus one.
        live = min(position + rows, cap)
        extent = min(extent, live)
        arrays = q, key, value, bias[..., :extent], sink
        host = np.empty(2 + sum(a.size for a in arrays), dtype=np.float32)
        host[:2].view(np.int32)[:] = (live, position)
        result = np.empty((rows, nq, d), dtype=np.float32)
        need = host.nbytes + result.nbytes
        scratch = self._block_scratch
        if scratch is None or scratch.nbytes < need:
            self._close_block_scratch()
            self._block_scratch = scratch = malloc(1 << (need - 1).bit_length(), runtime=self.rt)
        inp, out = scratch.ptr, scratch.ptr + host.nbytes
        ptrs, cursor = [], 2
        for a in arrays:
            ptrs.append(inp + cursor * host.itemsize)
            end = cursor + a.size
            host[cursor:end].reshape(a.shape)[...] = a
            cursor = end
        query, block_k, block_v, rel, sinks = ptrs
        spans = replace(spans, live_counts=Tensor(inp, (1,), DType.INT32, self.device),
                        row_positions=Tensor(inp + 4, (1,), DType.INT32, self.device))
        with self.transfer.exchange(inp, out, host, result, pinned=True):
            self.block(query, keys, values, rel, sinks, out, spans, nq, nk, d, extent, block_k, block_v, rows)
        self.positions[visit] = position + rows
        return result

    def _close_block_scratch(self):
        if self._block_scratch is not None:
            free(self._block_scratch, runtime=self.rt)
            self._block_scratch = None

    def close(self):
        self.closed = True
        try:
            self._close_block_scratch()
        finally:
            self.owner.close()
            self.banks.clear()
            self.positions.clear()
