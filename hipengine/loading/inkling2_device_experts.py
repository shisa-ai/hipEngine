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
    device_outer: int = 0


class Inkling2DeviceExperts:
    """Own packed banks and synchronized FP32 projection scratch on one device."""

    def __init__(self, weights, layers, *, runtime, launch, grouped=None, max_selected=1, pointwise=None):
        self._pointwise = pointwise
        self._runtime = runtime
        self._launch = launch
        self._grouped = grouped
        self._max_selected = max_selected
        self._lock = Lock()
        self._allocations: list[DeviceBuffer] = []
        self._banks: dict[tuple[int, str], _ResidentBank] = {}
        self._closed = False
        try:
            layout = {(layer, stem): weights.expert_bank(layer, stem)
                      for layer in dict.fromkeys(layers) for stem in ("w13", "w2")}
            if not layout:
                raise ValueError("Inkling 2 expert residency requires at least one layer")
            if not 1 <= max_selected <= 65535:
                raise ValueError("Inkling 2 selected expert scratch width is invalid")
            input_values = max_selected * max(bank.columns for bank in layout.values())
            if grouped is not None:
                input_values += max_selected * (2 if pointwise is not None else 1)
            self._staging = np.empty(input_values, dtype=np.float32) if grouped is not None else None
            self._input = self._allocate(4 * input_values)
            self._output = self._allocate(4 * max_selected * max(bank.values.shape[1] for bank in layout.values()))
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
        device_outer = upload(outer).ptr if outer is not None and self._grouped is not None else 0
        return _ResidentBank(weights, scales, exp, rows, bank.columns, outer, device_outer)

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

    def project_group(self, layer, projection, experts, hidden):
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 expert residency is closed")
            bank = self._banks[layer, projection]
            if self._grouped is None:
                raise RuntimeError("Inkling 2 grouped expert projection is unavailable")
            ids = np.ascontiguousarray(experts, dtype=np.int32)
            count = ids.size
            if ids.ndim != 1 or not 1 <= count <= self._max_selected or np.any(ids < 0) or np.any(ids >= bank.expert_count):
                raise ValueError("invalid Inkling 2 selected expert IDs")
            if hidden.shape not in ((bank.columns,), (count, bank.columns)):
                raise ValueError("invalid Inkling 2 grouped expert input geometry")
            y = np.empty((count, bank.rows), dtype=np.float32)
            staging = self._staging[:count + hidden.size]
            staging[:count].view(np.int32)[:] = ids
            staging[count:].reshape(hidden.shape)[...] = hidden
            copy = self._runtime.memcpy
            copy(self._input.ptr, staging.ctypes.data, staging.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            self._grouped(bank.weights.ptr, bank.scales.ptr, self._input.ptr + ids.nbytes, self._output.ptr,
                          self._input.ptr, bank.device_outer, count, bank.rows, bank.columns,
                          bank.columns if hidden.ndim == 2 else 0,
                          nvfp4=bank.outer_scales is not None)
            copy(y.ctypes.data, self._output.ptr, y.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            return y

    def feed_forward(self, layer, experts, mix, hidden, gate_cap, up_cap):
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 expert residency is closed")
            if self._grouped is None or self._pointwise is None:
                raise RuntimeError("Inkling 2 device expert feed-forward is unavailable")
            gate, down = self._banks[layer, "w13"], self._banks[layer, "w2"]
            ids = np.ascontiguousarray(experts, dtype=np.int32)
            count = ids.size
            if (ids.ndim != 1 or not 1 <= count <= self._max_selected or np.any(ids < 0)
                    or np.any(ids >= min(gate.expert_count, down.expert_count)) or mix.shape != (count,)):
                raise ValueError("invalid Inkling 2 device expert routing")
            if hidden.shape != (gate.columns,) or gate.rows != 2 * down.columns or down.rows != gate.columns:
                raise ValueError("invalid Inkling 2 device expert feed-forward geometry")
            staging = self._staging[:2 * count + hidden.size]
            staging[:count].view(np.int32)[:] = ids
            staging[count:2 * count] = mix
            staging[2 * count:] = hidden
            base, output = self._input.ptr, self._output.ptr
            mix_ptr = base + ids.nbytes
            inputs = mix_ptr + ids.nbytes
            copy = self._runtime.memcpy
            copy(base, staging.ctypes.data, staging.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            self._grouped(gate.weights.ptr, gate.scales.ptr, inputs, output, base, gate.device_outer,
                          count, gate.rows, gate.columns, 0, nvfp4=gate.outer_scales is not None)
            self._pointwise(output, 0, inputs, count, down.columns, gate_cap, up_cap)
            self._grouped(down.weights.ptr, down.scales.ptr, inputs, output, base, down.device_outer,
                          count, down.rows, down.columns, down.columns, nvfp4=down.outer_scales is not None)
            self._pointwise(output, mix_ptr, inputs, count, down.rows)
            result = np.empty(down.rows, dtype=np.float32)
            copy(result.ctypes.data, inputs, result.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            return result

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
