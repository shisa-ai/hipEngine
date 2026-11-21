"""Per-sequence device ownership for Inkling 2 short convolutions."""

from contextlib import ExitStack
from math import sqrt

import numpy as np

from hipengine.core.memory import free, malloc
from hipengine.loading.inkling2_activation_transfer import Inkling2ActivationTransfer


class Inkling2DeviceConvolution:
    def __init__(self, *, runtime, launch, channels, width):
        if channels <= 0 or width <= 0:
            raise ValueError("Inkling 2 convolution geometry must be positive")
        self.rt, self.launch = runtime, launch
        self.channels, self.width = channels, width
        self.histories = {}
        self.closed = False
        with ExitStack() as owner:
            self.transfer = Inkling2ActivationTransfer(runtime)
            owner.callback(self.transfer.close)

            def allocate(size):
                buf = malloc(size, runtime=runtime)
                owner.callback(free, buf, runtime=runtime)
                return buf.ptr

            self.staging = np.empty(channels * (1 + width), dtype=np.float32)
            self.input = allocate(self.staging.nbytes)
            self.out = allocate(channels * 4)
            self.owner = owner.pop_all()

    def step(self, visit, name, x, taps, *, dynamic=False, position=0, undo=None):
        if self.closed:
            raise RuntimeError("Inkling 2 convolution state is closed")
        channels = x.size
        if x.shape != (channels,) or not 0 < channels <= self.channels or taps.shape != (channels, self.width):
            raise ValueError("Inkling 2 convolution input and taps do not match session geometry")
        key = visit, name
        if key not in self.histories:
            buf = malloc(channels * self.width * 4, runtime=self.rt)
            self.owner.callback(free, buf, runtime=self.rt)
            self.rt.memset(buf.ptr, 0, buf.nbytes)
            self.histories[key] = buf
        history = self.histories[key]
        out = np.empty(channels, dtype=np.float32)
        if history.nbytes != channels * self.width * out.itemsize:
            raise ValueError("Inkling 2 convolution history geometry changed")
        if undo is not None:
            undo.device(position, self.rt, history.ptr, history.nbytes)
        used = channels * (1 + self.width)
        self.staging[:channels] = x
        self.staging[channels:used].reshape(channels, self.width)[...] = taps
        scale = 1 / sqrt((2 if dynamic else 1) * self.width + 1)
        with self.transfer.exchange(self.input, self.out, self.staging[:used], out, pinned=True):
            self.launch(history.ptr, self.input, self.input + out.nbytes, self.out, channels, self.width, scale)
        return out

    def close(self):
        self.closed = True
        self.owner.close()
        self.histories.clear()
