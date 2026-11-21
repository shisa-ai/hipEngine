"""Per-sequence device ownership for Inkling 2 short convolutions."""

from contextlib import ExitStack
from math import sqrt

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc
from hipengine.loading.inkling2_activation_transfer import Inkling2ActivationTransfer


class Inkling2DeviceConvolution:
    def __init__(self, *, runtime, launch, channels, width, launch_block=None):
        if channels <= 0 or width <= 0:
            raise ValueError("Inkling 2 convolution geometry must be positive")
        self.rt, self.launch, self.launch_block = runtime, launch, launch_block
        self.channels, self.width = channels, width
        self.histories = {}
        self.static_taps = {}
        self.closed = False
        with ExitStack() as owner:
            self.transfer = Inkling2ActivationTransfer(runtime)
            owner.callback(self.transfer.close)

            def allocate(size):
                buf = malloc(size, runtime=runtime)
                owner.callback(free, buf, runtime=runtime)
                return buf.ptr

            self.staging = np.empty(2 * channels * (1 + width), dtype=np.float32)
            self.input = allocate(self.staging.nbytes)
            self.out = allocate(2 * channels * 4)
            self.owner = owner.pop_all()

    def _history(self, visit, name, shape, taps):
        if self.closed:
            raise RuntimeError("Inkling 2 convolution state is closed")
        if len(shape) != 1 or not 0 < shape[0] <= self.channels or taps.shape != (shape[0], self.width):
            raise ValueError("Inkling 2 convolution input and taps do not match session geometry")
        channels, = shape
        key = visit, name
        if key not in self.histories:
            buf = malloc(channels * self.width * 4, runtime=self.rt)
            self.owner.callback(free, buf, runtime=self.rt)
            self.rt.memset(buf.ptr, 0, buf.nbytes)
            self.histories[key] = buf
        history = self.histories[key]
        if history.nbytes != channels * self.width * 4:
            raise ValueError("Inkling 2 convolution history geometry changed")
        return history

    def _snapshot(self, history, position, undo):
        if undo is not None:
            undo.device(position, self.rt, history.ptr, history.nbytes)

    def _pack(self, x, taps, offset=0):
        channels = x.size
        end = offset + channels * (1 + self.width)
        self.staging[offset:offset + channels] = x
        self.staging[offset + channels:end].reshape(channels, self.width)[...] = taps
        return end

    def _scale(self, dynamic):
        return 1 / sqrt((2 if dynamic else 1) * self.width + 1)

    def step(self, visit, name, x, taps, *, dynamic=False, position=0, undo=None):
        history = self._history(visit, name, x.shape, taps)
        self._snapshot(history, position, undo)
        out = np.empty(x.size, dtype=np.float32)
        used = self._pack(x, taps)
        with self.transfer.exchange(self.input, self.out, self.staging[:used], out, pinned=True):
            self.launch(history.ptr, self.input, self.input + out.nbytes, self.out,
                        x.size, self.width, self._scale(dynamic))
        return out

    def step_kv(self, visit, k, v, kt, vt, *, dynamic=False, position=0, undo=None):
        kh = self._history(visit, "k", k.shape, kt)
        vh = self._history(visit, "v", v.shape, vt)
        self._snapshot(kh, position, undo)
        self._snapshot(vh, position, undo)
        end_k = self._pack(k, kt)
        end_v = self._pack(v, vt, end_k)
        out = np.empty(k.size + v.size, dtype=np.float32)
        kbytes = k.size * out.itemsize
        vptr = self.input + end_k * out.itemsize
        scale = self._scale(dynamic)
        with self.transfer.exchange(self.input, self.out, self.staging[:end_v], out, pinned=True):
            self.launch(kh.ptr, self.input, self.input + kbytes, self.out, k.size, self.width, scale)
            self.launch(vh.ptr, vptr, vptr + v.size * out.itemsize, self.out + kbytes, v.size, self.width, scale)
        return out[:k.size], out[k.size:]

    def apply_static(self, visit, name, taps, ptr, shape, *, position=0, undo=None):
        """Convolve a device buffer in place using immutable taps for this visit."""
        if len(shape) not in (1, 2) or not all(shape):
            raise ValueError("Inkling 2 static convolution expects a vector or row matrix")
        channels = shape[-1]
        history = self._history(visit, name, (channels,), taps)
        key = visit, name
        if key not in self.static_taps:
            host = np.ascontiguousarray(taps, dtype=np.float32)
            buf = malloc(host.nbytes, runtime=self.rt)
            self.owner.callback(free, buf, runtime=self.rt)
            self.rt.memcpy(buf.ptr, host.ctypes.data, host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            self.static_taps[key] = buf.ptr
        tap_ptr = self.static_taps[key]
        scale = self._scale(False)
        rows = 1 if len(shape) == 1 else shape[0]
        if rows > 1 and undo is None and self.launch_block is not None:
            self.launch_block(history.ptr, ptr, tap_ptr, ptr, channels, self.width, scale, rows)
            return
        for i in range(rows):
            self._snapshot(history, position + i, undo)
            row = ptr + i * channels * 4
            self.launch(history.ptr, row, tap_ptr, row, channels, self.width, scale)

    def close(self):
        self.closed = True
        self.owner.close()
        self.histories.clear()
        self.static_taps.clear()
