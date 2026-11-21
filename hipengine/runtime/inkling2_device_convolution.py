"""Per-sequence device ownership for Inkling 2 short convolutions."""

from contextlib import ExitStack
from math import sqrt

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import free, malloc


class Inkling2DeviceConvolution:
    def __init__(self, *, runtime, launch, channels, width):
        if channels <= 0 or width <= 0:
            raise ValueError("Inkling 2 convolution geometry must be positive")
        self.rt, self.launch = runtime, launch
        self.channels, self.width = channels, width
        self.histories = {}
        self.closed = False
        with ExitStack() as owner:
            def allocate(size):
                buf = malloc(size, runtime=runtime)
                owner.callback(free, buf, runtime=runtime)
                return buf.ptr

            self.x = allocate(channels * 4)
            self.taps = allocate(channels * width * 4)
            self.out = allocate(channels * 4)
            self.owner = owner.pop_all()

    def step(self, visit, name, x, taps, *, dynamic=False):
        if self.closed:
            raise RuntimeError("Inkling 2 convolution state is closed")
        channels = x.size
        if x.shape != (channels,) or not 0 < channels <= self.channels or taps.shape != (channels, self.width):
            raise ValueError("Inkling 2 convolution input and taps do not match session geometry")
        key = visit, name
        if key not in self.histories:
            shape = channels, self.width
            buf = malloc(channels * self.width * 4, runtime=self.rt)
            self.owner.callback(free, buf, runtime=self.rt)
            self.rt.memset(buf.ptr, 0, buf.nbytes)
            self.histories[key] = buf
        history = self.histories[key]
        x = np.ascontiguousarray(x, dtype=np.float32)
        taps = np.ascontiguousarray(taps, dtype=np.float32)
        out = np.empty_like(x)
        if history.nbytes != taps.nbytes:
            raise ValueError("Inkling 2 convolution history geometry changed")
        copy = self.rt.memcpy
        copy(self.x, x.ctypes.data, x.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        copy(self.taps, taps.ctypes.data, taps.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        scale = 1 / sqrt((2 if dynamic else 1) * self.width + 1)
        self.launch(history.ptr, self.x, self.taps, self.out, channels, self.width, scale)
        copy(out.ctypes.data, self.out, out.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        return out

    def close(self):
        self.closed = True
        self.owner.close()
        self.histories.clear()
