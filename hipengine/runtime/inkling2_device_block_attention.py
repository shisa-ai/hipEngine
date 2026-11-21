"""Synchronous HIP scratch for dense vision and draft attention blocks."""

from threading import Lock

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc


class Inkling2DeviceBlockAttention:
    def __init__(self, *, runtime, launch):
        self._rt, self._launch = runtime, launch
        self._buffers = {}
        self._lock = Lock()
        self._closed = False

    def _reserve(self, slot, size):
        current = self._buffers.get(slot)
        if current is None or current.nbytes < size:
            if current is not None:
                free(current, runtime=self._rt)
                del self._buffers[slot]
            current = self._buffers[slot] = malloc(size, runtime=self._rt)
        return current.ptr

    def __call__(self, q, k, v, *, prefix=()):
        if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
            raise ValueError("Inkling 2 block attention requires [tokens, heads, width] arrays")
        queries, nq, dim = q.shape
        keys, nk, kd = k.shape
        if not (0 < queries <= 65535 and keys > 0 and nq > 0 and nk > 0 and nq % nk == 0 and 0 < kd == dim <= 128):
            raise ValueError("unsupported Inkling 2 block attention geometry")
        prefix_rows = sum(n for _, _, n in prefix)
        row_bytes = nk * dim * 4
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 block attention is closed")
            copy = self._rt.memcpy
            ptrs = []
            for i, a in enumerate((q, k, v)):
                a = np.ascontiguousarray(a, dtype=np.float32)
                prefix_bytes = prefix_rows * row_bytes if i else 0
                ptr = self._reserve(i, prefix_bytes + a.nbytes)
                offset = 0
                if i:
                    for kp, vp, rows in prefix:
                        size = rows * row_bytes
                        copy(ptr + offset, kp if i == 1 else vp, size, HipMemcpyKind.DEVICE_TO_DEVICE)
                        offset += size
                copy(ptr + offset, a.ctypes.data, a.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
                ptrs.append(ptr)
            out = np.empty(q.shape, dtype=np.float32)
            dst = self._reserve(3, out.nbytes)
            self._launch(*ptrs, dst, queries, prefix_rows + keys, nq, nk, dim)
            copy(out.ctypes.data, dst, out.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            return out

    def close(self):
        with self._lock:
            self._closed = True
            for slot in tuple(self._buffers):
                free(self._buffers[slot], runtime=self._rt)
                del self._buffers[slot]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
