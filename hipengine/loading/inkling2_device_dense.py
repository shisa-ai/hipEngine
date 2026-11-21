"""Resident dense projections for the Inkling 2 HIP bring-up."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc
from hipengine.loading.inkling2_activation_transfer import Inkling2ActivationTransfer


class Inkling2DeviceDense:
    """Own lazily uploaded dense matrices and synchronous FP32 activation scratch."""

    def __init__(self, weights, *, runtime, blas, lock=None):
        self._weights = weights
        self._runtime = runtime
        self._blas = blas  # Borrowed; the runner owns the rocBLAS handle.
        self._lock = Lock() if lock is None else lock
        self._matrices = {}
        self._input = self._output = None
        self._transfer = Inkling2ActivationTransfer(runtime)
        self._closed = False

    def _matrix(self, name, limit, *, preserve_bf16=False):
        key = name, limit, preserve_bf16
        if key not in self._matrices:
            selector = slice(limit)
            bits = self._weights.native_bf16(name, selector) if preserve_bf16 else None
            host = (np.ascontiguousarray(bits) if bits is not None else
                    np.ascontiguousarray(self._weights.dense(name, selector), dtype=np.float32))
            if host.ndim != 2 or not all(host.shape):
                raise ValueError(f"Inkling 2 dense weight {name} is not a nonempty matrix")
            buf = malloc(host.nbytes, runtime=self._runtime)
            try:
                self._runtime.memcpy(buf.ptr, host.ctypes.data, host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            except BaseException:
                free(buf, runtime=self._runtime)
                raise
            self._matrices[key] = buf, host.shape, bits is not None
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

    def project(self, name: str, hidden: np.ndarray, *, limit: int | None = None, native_bf16=None,
                pinned=False, transform_output=None) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 dense residency is closed")
            if hidden.ndim not in (1, 2) or not all(hidden.shape):
                raise ValueError("Inkling 2 dense projection expects a vector or row matrix")
            weight, (outputs, inputs), bf16 = self._matrix(name, limit, preserve_bf16=native_bf16 is not None)
            if hidden.shape[-1] != inputs:
                raise ValueError(f"Inkling 2 dense projection {name} expects {inputs} input values per row")
            x = np.ascontiguousarray(hidden, dtype=np.float32)
            y = np.empty((*hidden.shape[:-1], outputs), dtype=np.float32)
            inp = self._scratch("_input", x.nbytes)
            out = self._scratch("_output", y.nbytes)
            with self._transfer.exchange(inp.ptr, out.ptr, x, y, pinned=pinned):
                self._launch_projection(inp.ptr, out.ptr, 1 if x.ndim == 1 else x.shape[0], x.ndim,
                                        (weight, (outputs, inputs), bf16), native_bf16)
                if transform_output is not None:
                    transform_output(out.ptr, y.shape)
            return y

    @contextmanager
    def project_device(self, name, width, *, native_bf16=None, transform_output=None):
        """Borrow projection scratch until the input owner downloads the result."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 dense residency is closed")
            matrix = self._matrix(name, None, preserve_bf16=native_bf16 is not None)
            outputs, inputs = matrix[1]
            if width != inputs:
                raise ValueError(f"Inkling 2 dense projection {name} expects {inputs} input values per row")
            result = np.empty(outputs, dtype=np.float32)
            out = self._scratch("_output", result.nbytes)

            def consume(ptr):
                self._launch_projection(ptr, out.ptr, 1, 1, matrix, native_bf16)
                if transform_output is not None:
                    transform_output(out.ptr, result.shape)

            yield out.ptr, result, consume

    def project_many(self, names, hidden: np.ndarray, *, native_bf16=None, pinned=False):
        """Share transfers for projections that consume the same FP32 input."""
        return self.project_groups(((names, hidden),), native_bf16=native_bf16, pinned=pinned)[0]

    def project_groups(self, groups, *, native_bf16=None, pinned=False):
        """Exchange independent inputs and results together, preserving projection order."""
        return self._project_groups(((None, names, x) for names, x in groups), native_bf16, pinned)

    def project_chains(self, chains, *, native_bf16=None, pinned=False):
        """Feed each first projection directly into the remaining projections."""
        return self._project_groups(chains, native_bf16, pinned)

    def _project_groups(self, chains, native_bf16, pinned):
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 dense residency is closed")
            plans = []
            output_count = intermediate_count = 0
            for first, names, hidden in chains:
                if hidden.ndim not in (1, 2) or not all(hidden.shape):
                    raise ValueError("Inkling 2 dense projection expects a vector or row matrix")
                before = None
                width = hidden.shape[-1]
                rows = 1 if hidden.ndim == 1 else hidden.shape[0]
                if first is not None:
                    before = self._matrix(first, None, preserve_bf16=native_bf16 is not None)
                    width, n = before[1]
                    if n != hidden.shape[-1]:
                        raise ValueError("Inkling 2 first projection input width differs from its matrix")
                    intermediate_count = max(intermediate_count, rows * width)
                matrices = [self._matrix(name, None, preserve_bf16=native_bf16 is not None) for name in names]
                if not matrices:
                    raise ValueError("Inkling 2 projection group must not be empty")
                if any(width != n for _, (_, n), _ in matrices):
                    raise ValueError("Inkling 2 grouped dense projection input width differs from its matrices")
                x = np.ascontiguousarray(hidden, dtype=np.float32)
                plans.append((x, rows, before, matrices))
                output_count += rows * sum(m for _, (m, _), _ in matrices)
            if not plans:
                raise ValueError("Inkling 2 projection groups must not be empty")
            inputs = plans[0][0] if len(plans) == 1 else np.concatenate([x.ravel() for x, *_ in plans])
            output = np.empty(output_count, dtype=np.float32)
            inp = self._scratch("_input", inputs.nbytes)
            out = self._scratch("_output", output.nbytes + intermediate_count * output.itemsize)
            intermediate = out.ptr + output.nbytes
            input_offset = output_offset = 0
            results = []
            with self._transfer.exchange(inp.ptr, out.ptr, inputs, output, pinned=pinned):
                for x, rows, before, matrices in plans:
                    src = inp.ptr + input_offset
                    if before is not None:
                        self._launch_projection(src, intermediate, rows, x.ndim, before, native_bf16)
                        src = intermediate
                    group = []
                    for matrix in matrices:
                        m, _ = matrix[1]
                        dst = out.ptr + output_offset * output.itemsize
                        self._launch_projection(src, dst, rows, x.ndim, matrix, native_bf16)
                        end = output_offset + rows * m
                        group.append(output[output_offset:end].reshape(*x.shape[:-1], m))
                        output_offset = end
                    results.append(tuple(group))
                    input_offset += x.nbytes
            return tuple(results)

    def _launch_projection(self, src, dst, rows, ndim, matrix, native_bf16):
        weight, (m, n), bf16 = matrix
        if bf16:
            native_bf16(src, weight.ptr, dst, rows, n, m, runtime=self._runtime)
        elif ndim == 1:
            self._blas.sgemv_rowmajor_nt(src, weight.ptr, dst, in_features=n, out_features=m)
        else:
            self._blas.sgemm_rowmajor_nt(src, weight.ptr, dst, rows=rows, in_features=n, out_features=m)

    def close(self):
        with self._lock:
            self._closed = True
            for key in tuple(self._matrices):
                free(self._matrices[key][0], runtime=self._runtime)
                del self._matrices[key]
            self._transfer.close()
            for name in ("_output", "_input"):
                buf = getattr(self, name)
                if buf is not None:
                    free(buf, runtime=self._runtime)
                    setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
