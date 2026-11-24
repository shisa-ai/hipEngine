"""Resident dense projections for the Inkling 2 HIP bring-up."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, RLock

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
        # Reentrant: one resident block holds the arena across calls that take the lock again.
        self._lock = RLock() if lock is None else lock
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
    def project_device(self, name, width, *, rows=None, native_bf16=None, transform_output=None):
        """Borrow projection scratch until the input owner downloads the result."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 dense residency is closed")
            matrix = self._matrix(name, None, preserve_bf16=native_bf16 is not None)
            outputs, inputs = matrix[1]
            if width != inputs:
                raise ValueError(f"Inkling 2 dense projection {name} expects {inputs} input values per row")
            if rows is not None and rows <= 0:
                raise ValueError("Inkling 2 device projection requires positive row count")
            result = np.empty(outputs if rows is None else (rows, outputs), dtype=np.float32)
            out = self._scratch("_output", result.nbytes)

            def consume(ptr):
                self._launch_projection(ptr, out.ptr, 1 if rows is None else rows, result.ndim, matrix, native_bf16)
                if transform_output is not None:
                    transform_output(out.ptr, result.shape)

            yield out.ptr, result, consume

    def project_many(self, names, hidden: np.ndarray, *, native_bf16=None, pinned=False, transform_outputs=None):
        """Share transfers and optionally consume each resident output before download."""
        names = tuple(names)
        if transform_outputs is not None and len(transform_outputs) != len(names):
            raise ValueError("Inkling 2 projection requires one transform slot per output")
        return self._project_groups(((None, names, hidden),), native_bf16, pinned, transform_outputs)[0]

    def project_groups(self, groups, *, native_bf16=None, pinned=False, transform_groups=None):
        """Exchange independent inputs and optionally consume each resident output group."""
        return self._project_groups(((None, names, x) for names, x in groups), native_bf16, pinned,
                                    transform_groups=transform_groups)

    def project_chains(self, chains, *, native_bf16=None, pinned=False):
        """Feed each first projection directly into the remaining projections."""
        return self._project_groups(chains, native_bf16, pinned)

    def _project_groups(self, chains, native_bf16, pinned, transform_outputs=None, *, transform_groups=None,
                        transform_batch=None):
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
            if transform_groups is not None and len(transform_groups) != len(plans):
                raise ValueError("Inkling 2 projection requires one transform slot per group")
            inputs = plans[0][0] if len(plans) == 1 else np.concatenate([x.ravel() for x, *_ in plans])
            output = np.empty(output_count, dtype=np.float32)
            inp = self._scratch("_input", inputs.nbytes)
            out = self._scratch("_output", output.nbytes + intermediate_count * output.itemsize)
            intermediate = out.ptr + output.nbytes
            input_offset = output_offset = 0
            results = []
            resident_batch = [] if transform_batch is not None else None
            with self._transfer.exchange(inp.ptr, out.ptr, inputs, output, pinned=pinned):
                for group_index, (x, rows, before, matrices) in enumerate(plans):
                    consume_group = None if transform_groups is None else transform_groups[group_index]
                    resident = [] if consume_group is not None or resident_batch is not None else None
                    src = inp.ptr + input_offset
                    if before is not None:
                        self._launch_projection(src, intermediate, rows, x.ndim, before, native_bf16)
                        src = intermediate
                    group = []
                    for index, matrix in enumerate(matrices):
                        m, _ = matrix[1]
                        dst = out.ptr + output_offset * output.itemsize
                        self._launch_projection(src, dst, rows, x.ndim, matrix, native_bf16)
                        end = output_offset + rows * m
                        view = output[output_offset:end].reshape(*x.shape[:-1], m)
                        if transform_outputs is not None and (consume := transform_outputs[index]) is not None:
                            consume(dst, view.shape)
                        group.append(view)
                        if resident is not None:
                            resident.append((dst, view.shape))
                        output_offset = end
                    if consume_group is not None:
                        consume_group(*resident)
                    if resident_batch is not None:
                        resident_batch.append(tuple(resident))
                    results.append(tuple(group))
                    input_offset += x.nbytes
                if transform_batch is not None:
                    transform_batch(resident_batch)
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
