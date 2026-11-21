"""Owned HIP projections over packed Inkling 2 Mini expert banks."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.loading.inkling2_mini import Inkling2ExpertBank


@dataclass(frozen=True, slots=True)
class _ResidentBank:
    weights: DeviceBuffer
    scales: DeviceBuffer
    expert_count: int
    rows: int
    columns: int
    outer_scales: np.ndarray | None


class Inkling2DeviceExperts:
    """Own packed banks and synchronized FP32 projection scratch on one device."""

    def __init__(self, weights, layers, *, runtime, launch):
        self._runtime = runtime
        self._launch = launch
        self._lock = Lock()
        self._allocations: list[DeviceBuffer] = []
        self._banks: dict[tuple[int, str], _ResidentBank] = {}
        self._closed = False
        try:
            layout = {(layer, stem): weights.expert_bank(layer, stem)
                      for layer in dict.fromkeys(layers) for stem in ("w13", "w2")}
            if not layout:
                raise ValueError("Inkling 2 expert residency requires at least one layer")
            self._input = self._allocate(4 * max(bank.columns for bank in layout.values()))
            self._output = self._allocate(4 * max(bank.values.shape[1] for bank in layout.values()))
            for key, bank in layout.items():
                self._banks[key] = self._upload(bank)
        except BaseException:
            self.close()
            raise

    def _allocate(self, nbytes):
        buffer = malloc(nbytes, runtime=self._runtime)
        self._allocations.append(buffer)
        return buffer

    def _upload(self, bank: Inkling2ExpertBank):
        def upload(array):
            buffer = self._allocate(array.nbytes)
            self._runtime.memcpy(buffer.ptr, array.ctypes.data, array.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            return buffer

        weights = upload(bank.values)
        scales = upload(bank.scales)
        exp, rows, _ = bank.values.shape
        outer = None if bank.amax is None else bank.amax / np.float32(bank.denominator)
        return _ResidentBank(weights, scales, exp, rows, bank.columns, outer)

    def project(self, layer, projection, expert, hidden):
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 expert residency is closed")
            bank = self._banks[layer, projection]
            if not 0 <= expert < bank.expert_count:
                raise IndexError(f"Inkling 2 expert index out of range: {expert}")
            if hidden.shape != (bank.columns,):
                raise ValueError(f"Inkling 2 expert projection expected {bank.columns} input values")
            x = np.ascontiguousarray(hidden, dtype=np.float32)
            y = np.empty(bank.rows, dtype=np.float32)
            self._runtime.memcpy(self._input.ptr, x.ctypes.data, x.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            outer = bank.outer_scales
            self._launch(bank.weights.ptr + expert * (bank.weights.nbytes // bank.expert_count),
                         bank.scales.ptr + expert * (bank.scales.nbytes // bank.expert_count),
                         self._input.ptr, self._output.ptr, bank.rows, bank.columns,
                         nvfp4=outer is not None, outer_scale=1.0 if outer is None else float(outer[expert]))
            self._runtime.memcpy(y.ctypes.data, self._output.ptr, y.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            return y

    def close(self):
        with self._lock:
            self._closed = True
            self._banks.clear()
            while self._allocations:
                free(self._allocations[-1], runtime=self._runtime)
                self._allocations.pop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
