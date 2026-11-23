"""Owned HIP projections over packed Inkling 2 Mini expert banks."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.loading.inkling2_activation_transfer import Inkling2ActivationTransfer
from hipengine.loading.inkling2_mini import Inkling2ExpertBank
from hipengine.quant.inkling2_experts import widen_nvfp4_scales


@dataclass(frozen=True, slots=True)
class _ResidentBank:
    weights: DeviceBuffer
    scales: DeviceBuffer
    expert_count: int
    rows: int
    columns: int
    outer_scales: np.ndarray | None
    device_outer: int = 0
    expanded_scales: bool = False


def build_expert_route_tiles(ids, width):
    """Build fixed-width records of same-expert routes, padded with -1."""
    flat = ids.ravel()
    # Decode-sized route lists are cheaper to group without NumPy temporaries.
    if 0 < flat.size <= 32 and width in (2, 4) and flat.dtype == np.int32:
        groups = {}
        for route, expert in enumerate(flat.tolist()):
            groups.setdefault(expert, []).append(route)
        records = []
        for expert, routes in sorted(groups.items()):
            for start in range(0, len(routes), width):
                tile = routes[start:start + width]
                records.append([expert, *tile, *([-1] * (width - len(tile)))])
        return np.asarray(records, dtype=np.int32)
    ordered = np.argsort(flat, kind="stable").astype(np.int32)
    experts = flat[ordered]
    i = np.arange(flat.size, dtype=np.int32)
    starts = np.maximum.accumulate(np.where(np.r_[True, experts[1:] != experts[:-1]], i, 0))
    first = i[(i - starts) % width == 0]
    locations = first[:, None] + np.arange(width, dtype=np.int32)
    safe = np.minimum(locations, len(i) - 1)
    routes = np.where((locations == safe) & (experts[safe] == experts[first, None]), ordered[safe], -1)
    return np.column_stack((experts[first], routes))


class Inkling2DeviceExperts:
    """Own packed banks and synchronized FP32 projection scratch on one device."""

    def __init__(self, weights, layers, *, runtime, launch, grouped=None, max_selected=1, pointwise=None,
                 expand_nvfp4_scales=False, tiled=None, tile_width=4):
        if tile_width not in (2, 4):
            raise ValueError("Inkling 2 expert tile width must be 2 or 4")
        self._expand_nvfp4_scales = expand_nvfp4_scales
        self._pointwise = pointwise
        self._tiled, self._tile_width = tiled, tile_width
        self._runtime = runtime
        self._launch = launch
        self._grouped = grouped
        self._max_selected = max_selected
        self._lock = Lock()
        self._allocations: list[DeviceBuffer] = []
        self._banks: dict[tuple[int, str], _ResidentBank] = {}
        self._block_scratch = None
        self._transfer = Inkling2ActivationTransfer(runtime)
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
        expanded = self._expand_nvfp4_scales and bank.amax is not None
        scales = upload(widen_nvfp4_scales(bank.scales) if expanded else bank.scales)
        exp, rows, _ = bank.values.shape
        outer = None if bank.amax is None else bank.amax / np.float32(bank.denominator)
        device_outer = upload(outer).ptr if outer is not None and self._grouped is not None else 0
        return _ResidentBank(weights, scales, exp, rows, bank.columns, outer, device_outer, expanded)

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
                         nvfp4=outer is not None, outer_scale=1.0 if outer is None else float(outer[expert]),
                         expanded_scales=bank.expanded_scales)
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
                          nvfp4=bank.outer_scales is not None, expanded_scales=bank.expanded_scales)
            copy(y.ctypes.data, self._output.ptr, y.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            return y

    def _feed_scratch(self, tokens):
        if tokens == 1:
            return self._staging, self._input, self._output
        block = self._block_scratch
        if block is None or tokens > block[0]:
            # Geometric growth bounds the temporary allocation count when block widths vary.
            capacity = min(65535, 1 << (tokens - 1).bit_length())
            extra = 5 * self._max_selected if self._tiled is not None else 0
            staging = np.empty((self._staging.size + extra) * capacity, dtype=np.float32)
            active = self._allocate(staging.nbytes)
            projection = self._allocate(self._output.nbytes * capacity)
            if block is not None:
                for buf in block[2:]:
                    free(buf, runtime=self._runtime)
                    self._allocations.remove(buf)
            block = self._block_scratch = capacity, staging, active, projection
        return block[1:]

    def feed_forward(self, layer, experts, mix, hidden, gate_cap, up_cap, *, transform_output=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Inkling 2 expert residency is closed")
            if self._grouped is None or self._pointwise is None:
                raise RuntimeError("Inkling 2 device expert feed-forward is unavailable")
            gate, down = self._banks[layer, "w13"], self._banks[layer, "w2"]
            ids = np.ascontiguousarray(experts, dtype=np.int32)
            if (hidden.ndim not in (1, 2) or ids.ndim != hidden.ndim or mix.shape != ids.shape
                    or not 1 <= ids.shape[-1] <= self._max_selected or np.any(ids < 0)
                    or np.any(ids >= min(gate.expert_count, down.expert_count))):
                raise ValueError("invalid Inkling 2 device expert routing")
            tokens = 1 if hidden.ndim == 1 else hidden.shape[0]
            if (not 1 <= tokens <= 65535 or hidden.shape[:-1] != ids.shape[:-1]
                    or hidden.shape[-1] != gate.columns or gate.rows != 2 * down.columns or down.rows != gate.columns):
                raise ValueError("invalid Inkling 2 device expert feed-forward geometry")
            host, device_input, device_output = self._feed_scratch(tokens)
            count = ids.shape[-1]
            # Reserve weight-sharing tiles for prompt blocks; short verification is unchanged.
            tile_width = self._tile_width
            tiles = (build_expert_route_tiles(ids, tile_width)
                     if self._tiled is not None and tokens >= 16 and ids.size <= 65535 else None)
            header = 2 * ids.size
            if tiles is not None:
                host[header:header + tiles.size].view(np.int32)[:] = tiles.ravel()
            input_start = header + (0 if tiles is None else tiles.size)
            stage = host[:input_start + hidden.size]
            stage[:ids.size].view(np.int32)[:] = ids.ravel()
            stage[ids.size:header].reshape(mix.shape)[...] = mix
            stage[input_start:].reshape(hidden.shape)[...] = hidden
            base, output = device_input.ptr, device_output.ptr
            mix_ptr = base + ids.nbytes
            inputs = base + input_start * host.itemsize
            result = np.empty(hidden.shape, dtype=np.float32)

            def launch(bank, stride):
                args = (bank.weights.ptr, bank.scales.ptr, inputs, output)
                geometry = (bank.device_outer, count, bank.rows, bank.columns, stride)
                modes = dict(nvfp4=bank.outer_scales is not None, expanded_scales=bank.expanded_scales)
                if tiles is None:
                    self._grouped(*args, base, *geometry, tokens=tokens, **modes)
                else:
                    self._tiled(*args, base + 2 * ids.nbytes, *geometry, tiles=len(tiles), tile_width=tile_width, **modes)

            with self._transfer.exchange(base, inputs, stage, result, pinned=True):
                launch(gate, 0)
                self._pointwise(output, 0, inputs, count, down.columns, gate_cap, up_cap, tokens=tokens)
                launch(down, down.columns)
                self._pointwise(output, mix_ptr, inputs, count, down.rows, tokens=tokens)
                if transform_output is not None:
                    transform_output(inputs, result.shape)
            return result

    def close(self):
        with self._lock:
            self._closed = True
            self._banks.clear()
            self._transfer.close()
            while self._allocations:
                free(self._allocations[-1], runtime=self._runtime)
                self._allocations.pop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
