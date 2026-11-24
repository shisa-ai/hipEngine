"""Per-visit device KV state for Inkling 2 target attention."""

from contextlib import ExitStack, contextmanager
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
_CHUNKED_MIN_LIVE = 513
# One FP32 value vector, maximum and denominator per 64 slots and head.
_VALUE_CHUNK = 64


class Inkling2DeviceAttention:
    def __init__(self, *, runtime, launch, write, geometry, split=None, split_chunked=None, block=None,
                 block_sigmoid=None, device=None, block_table=None):
        self.rt, self.launch, self.write = runtime, launch, write
        self.split, self.split_chunked, self.block = split, split_chunked, block
        self.block_sigmoid = block_sigmoid
        self.block_table = block_table
        self._block_scratch = None
        self.scores = self.value_partials = 0
        self.device = Device("hip") if device is None else device
        self.geometry = tuple(geometry)  # (capacity, query heads, KV heads, head width, bias extent) by state slot
        self.banks = {}
        self.positions = {}
        if not self.geometry or any(cap <= 0 or nq <= 0 or nk <= 0 or nq % nk or not 0 < d <= 128 or e <= 0
                                    for cap, nq, nk, d, e in self.geometry):
            raise ValueError("unsupported gfx1100 Inkling 2 target attention geometry")
        self.closed = False
        self._borrowed_output = False
        with ExitStack() as owner:
            self.owner = owner
            self.transfer = Inkling2ActivationTransfer(runtime)
            owner.callback(self.transfer.close)
            scratch_values = 2 + max(nq * (2 * d + e + 1) + 2 * nk * d for _, nq, nk, d, e in self.geometry)
            self.staging = np.empty(scratch_values, dtype=np.float32)
            self.scratch = self._allocate(self.staging.nbytes)
            self.out = self._allocate(4 * max(nq * d for _, nq, _, d, _ in self.geometry))
            # Live count and first row of one resident block launch.
            self._block_header = self._allocate(8)
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
        if self.split_chunked is not None and any(cap >= _CHUNKED_MIN_LIVE for cap, *_ in self.geometry):
            self._reserve_value_partials()

    def _reserve_scores(self):
        if not self.scores:
            self.scores = self._allocate(4 * max(capacity * heads for capacity, heads, *_ in self.geometry))

    def _reserve_value_partials(self):
        if not self.value_partials:
            cells = max(heads * ((cap + _VALUE_CHUNK - 1) // _VALUE_CHUNK) * (dim + 2)
                        for cap, heads, _, dim, _ in self.geometry)
            self.value_partials = self._allocate(4 * cells)

    @contextmanager
    def resident_step(self, visit, position, q, key, value, bias, sink, *, gain=None):
        """Borrow a queued output. The consumer must fence stream zero before returning."""
        if self._borrowed_output:
            raise RuntimeError("Inkling 2 attention output is already borrowed")
        try:
            result = self.step(visit, position, q, key, value, bias, sink, gain=gain, _resident=True)
            self._borrowed_output = True
            yield result
        except BaseException:
            self.rt.stream_synchronize(0)
            raise
        finally:
            self._borrowed_output = False

    def step(self, visit, position, q, key, value, bias, sink, *, undo=None, gain=None, projection=None, _resident=False):
        if self.closed or self._borrowed_output:
            raise RuntimeError("Inkling 2 attention state is closed or still has a borrowed output")
        if _resident and (undo is not None or projection is not None):
            raise ValueError("Inkling 2 resident attention requires a committed step without a projection consumer")
        if position != self.positions.get(visit, 0):
            raise ValueError("Inkling 2 attention position must advance in order")
        cap, nq, nk, d, extent = self.geometry[visit]
        live = min(position + 1, cap)
        use_split = self.split is not None and live >= _SPLIT_MIN_LIVE
        use_chunked = use_split and self.split_chunked is not None and live >= _CHUNKED_MIN_LIVE
        expected = ((nq, d), (nk, d), (nk, d), (nq, extent), (nq,))
        if tuple(a.shape for a in (q, key, value, bias, sink)) != expected:
            raise ValueError("invalid Inkling 2 attention input shapes")
        if gain is not None and gain.shape != (nq * d,):
            raise ValueError("invalid Inkling 2 attention gain shape")
        if use_split:
            self._reserve_scores()
        if use_chunked:
            self._reserve_value_partials()
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
        # No live key is farther away than live - 1 in this ring.
        extent = min(extent, live)
        ptrs = []
        cursor = 2
        for array in (q, key, value, bias[:, :extent], sink):
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
        def launch():
            self.write(key_ptr, value_ptr, keys, values, spans, nk * d)
            if use_chunked:
                self.split_chunked(*args, self.scores, live, partial=self.value_partials, gain=gain_ptr)
            elif use_split:
                self.split(*args, self.scores, live, gain=gain_ptr)
            else:
                self.launch(*args, gain=gain_ptr)

        if _resident:
            self.transfer.enqueue_upload(self.scratch, self.staging[:cursor])
            launch()
            self.positions[visit] = position + 1
            return self.out
        if projection is None:
            dst, out, consume = self.out, np.empty((nq, d), dtype=np.float32), None
        else:
            dst, out, consume = projection
        with self.transfer.exchange(self.scratch, dst, self.staging[:cursor], out, pinned=True):
            launch()
            if consume is not None:
                consume(self.out)
        self.positions[visit] = position + 1
        return out

    def step_block(self, visit, position, q, key, value, bias, sink, *, undo=None, gain=None, gate_logits=None,
                   projection=None):
        if self.closed or self._borrowed_output:
            raise RuntimeError("Inkling 2 attention state is closed or still has a borrowed output")
        if position != self.positions.get(visit, 0):
            raise ValueError("Inkling 2 attention position must advance in order")
        cap, nq, nk, d, extent = self.geometry[visit]
        rows = len(q)
        expected = ((rows, nq, d), (rows, nk, d), (rows, nk, d), (rows, nq, extent), (rows, nq))
        arrays = q, key, value, bias, sink
        if not 0 < rows <= 65535 or tuple(a.shape for a in arrays) != expected:
            raise ValueError("invalid Inkling 2 attention block input shapes")
        block = self.block
        if gate_logits is not None:
            if gain is not None or undo is not None or self.block_sigmoid is None:
                raise ValueError("Inkling 2 raw gate logits require committed sigmoid block dispatch")
            gain, block = gate_logits, self.block_sigmoid
        if gain is not None and gain.shape != (rows, nq * d):
            raise ValueError("invalid Inkling 2 attention block gain shape")
        if block is None or undo is not None:
            if projection is not None:
                raise ValueError("Inkling 2 projected block attention requires committed block dispatch")
            return np.stack([self.step(visit, position + i, *a, undo=undo, gain=None if gain is None else gain[i])
                             for i, a in enumerate(zip(*arrays, strict=True))])
        if visit not in self.banks:
            self.banks[visit] = self._make_bank(cap, nk, d)
        keys, values, spans = self.banks[visit]
        # The farthest lag in this block cannot exceed either the final row's
        # absolute position or the live ring length minus one.
        live = min(position + rows, cap)
        extent = min(extent, live)
        arrays = q, key, value, bias[..., :extent], sink
        if gain is not None:
            arrays += (gain,)
        host = np.empty(2 + sum(a.size for a in arrays), dtype=np.float32)
        host[:2].view(np.int32)[:] = (live, position)
        need = host.nbytes + rows * nq * d * host.itemsize
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
        query, block_k, block_v, rel, sinks = ptrs[:5]
        gain_ptr = ptrs[5] if gain is not None else 0
        spans = replace(spans, live_counts=Tensor(inp, (1,), DType.INT32, self.device),
                        row_positions=Tensor(inp + 4, (1,), DType.INT32, self.device))
        if projection is None:
            dst, result, consume = out, np.empty((rows, nq, d), dtype=np.float32), None
        else:
            dst, result, consume = projection
        with self.transfer.exchange(inp, dst, host, result, pinned=True):
            block(query, keys, values, rel, sinks, out, spans, nq, nk, d, extent, block_k, block_v, rows,
                  gain=gain_ptr)
            if consume is not None:
                consume(out)
        self.positions[visit] = position + rows
        return result

    def step_block_resident(self, visit, position, query, key, value, rel, sink, gate, out, table, rel_width, rows):
        """Block attention over device-resident inputs, with the relative bias read from a lag table.

        The caller owns the stream: nothing here fences, so later launches on stream zero
        may consume `out` before the host reads anything.
        """
        if self.closed or self._borrowed_output:
            raise RuntimeError("Inkling 2 attention state is closed or still has a borrowed output")
        if self.block_table is None:
            raise RuntimeError("Inkling 2 resident block attention requires the lag-table kernel")
        if position != self.positions.get(visit, 0):
            raise ValueError("Inkling 2 attention position must advance in order")
        cap, nq, nk, d, extent = self.geometry[visit]
        if not 0 < rows <= 65535:
            raise ValueError("invalid Inkling 2 attention block row count")
        if visit not in self.banks:
            self.banks[visit] = self._make_bank(cap, nk, d)
        keys, values, spans = self.banks[visit]
        live = min(position + rows, cap)
        self.staging[:2].view(np.int32)[:] = (live, position)
        self.transfer.enqueue_upload(self._block_header, self.staging[:2])
        spans = replace(spans, live_counts=Tensor(self._block_header, (1,), DType.INT32, self.device),
                        row_positions=Tensor(self._block_header + 4, (1,), DType.INT32, self.device))
        self.block_table(query, keys, values, rel, sink, out, spans, nq, nk, d, min(extent, live), key, value, rows,
                         gain=gate, table=table, rel_width=rel_width)
        self.positions[visit] = position + rows

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
