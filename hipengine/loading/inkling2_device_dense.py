"""Resident dense projections for the Inkling 2 HIP bring-up."""

from __future__ import annotations

from threading import Lock

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc


class Inkling2DeviceDense:
    """Own lazily uploaded FP32 matrices and synchronous projection scratch."""

    def __init__(self, weights, *, runtime, blas, lock=None):
        self._weights = weights
        self._runtime = runtime
        self._blas = blas  # Borrowed; the runner owns the rocBLAS handle.
        self._lock = Lock() if lock is None else lock
        self._matrices = {}
        self._input = self._output = None
        self._closed = False

    def _matrix(self, name, limit):
        key = name, limit
        if key not in self._matrices:
            host = np.ascontiguousarray(self._weights.dense(name, slice(limit)), dtype=np.float32)
            if host.ndim != 2 or not all(host.shape):
                raise ValueError(f"Inkling 2 dense weight {name} is not a nonempty matrix")
            buf = malloc(host.nbytes, runtime=self._runtime)
            try:
                self._runtime.memcpy(buf.ptr, host.ctypes.data, host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            except BaseException:
                free(buf, runtime=self._runtime)
                raise
            self._matrices[key] = buf, host.shape
        return self._matrices[key]

    def _scratch(self, name, nbytes):
        old = getattr(self, name)
        if old is not None and old.nbytes >= nbytes:
            return old
        if old is not None:
            free(old, runtime=self._runtime)
            setattr(self, name, None)
        buf = malloc(nbytes, runtime=self._runtime)
        setattr(self, name, buf)
        return buf

    def project(self, name: str, hidden: np.ndarray, *, limit: int | None = None) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 dense residency is closed")
            if hidden.ndim not in (1, 2) or not all(hidden.shape):
                raise ValueError("Inkling 2 dense projection expects a vector or row matrix")
            weight, (outputs, inputs) = self._matrix(name, limit)
            if hidden.shape[-1] != inputs:
                raise ValueError(f"Inkling 2 dense projection {name} expects {inputs} input values per row")
            x = np.ascontiguousarray(hidden, dtype=np.float32)
            y = np.empty((*hidden.shape[:-1], outputs), dtype=np.float32)
            inp = self._scratch("_input", x.nbytes)
            out = self._scratch("_output", y.nbytes)
            self._runtime.memcpy(inp.ptr, x.ctypes.data, x.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            if x.ndim == 1:
                self._blas.sgemv_rowmajor_nt(inp.ptr, weight.ptr, out.ptr, in_features=inputs, out_features=outputs)
            else:
                self._blas.sgemm_rowmajor_nt(inp.ptr, weight.ptr, out.ptr, rows=x.shape[0],
                                            in_features=inputs, out_features=outputs)
            self._runtime.memcpy(y.ctypes.data, out.ptr, y.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            return y

    def project_many(self, names, hidden: np.ndarray):
        """Share transfers while preserving each FP32 BLAS call's geometry."""
        names = tuple(names)
        if not names:
            raise ValueError("Inkling 2 projection group must not be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 dense residency is closed")
            if hidden.ndim not in (1, 2) or not all(hidden.shape):
                raise ValueError("Inkling 2 dense projection expects a vector or row matrix")
            matrices = [self._matrix(name, None) for name in names]
            if any(hidden.shape[-1] != n for _, (_, n) in matrices):
                raise ValueError("Inkling 2 grouped dense projection input width differs from its matrices")
            x = np.ascontiguousarray(hidden, dtype=np.float32)
            widths = [m for _, (m, _) in matrices]
            rows = 1 if x.ndim == 1 else x.shape[0]
            y = np.empty(rows * sum(widths), dtype=np.float32)
            inp = self._scratch("_input", x.nbytes)
            out = self._scratch("_output", y.nbytes)
            copy = self._runtime.memcpy
            copy(inp.ptr, x.ctypes.data, x.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            offset = 0
            for weight, (m, n) in matrices:
                dst = out.ptr + offset * y.itemsize
                if x.ndim == 1:
                    self._blas.sgemv_rowmajor_nt(inp.ptr, weight.ptr, dst, in_features=n, out_features=m)
                else:
                    self._blas.sgemm_rowmajor_nt(inp.ptr, weight.ptr, dst, rows=rows, in_features=n, out_features=m)
                offset += rows * m
            copy(y.ctypes.data, out.ptr, y.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            results = []
            offset = 0
            for width in widths:
                end = offset + rows * width
                results.append(y[offset:end].reshape(*x.shape[:-1], width))
                offset = end
            return tuple(results)

    def close(self):
        with self._lock:
            self._closed = True
            for key in tuple(self._matrices):
                free(self._matrices[key][0], runtime=self._runtime)
                del self._matrices[key]
            for name in ("_output", "_input"):
                buf = getattr(self, name)
                if buf is not None:
                    free(buf, runtime=self._runtime)
                    setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
