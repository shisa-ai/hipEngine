"""Reusable page-locked staging for synchronous Inkling activation exchanges."""

import ctypes
import mmap
from contextlib import contextmanager

from hipengine.core.hip import HipError, HipMemcpyKind

# HIP can reject registration when page locking is unavailable or exhausted.
_HOST_REGISTRATION_UNAVAILABLE = frozenset((2, 304, 801))


class _PinnedScratch:
    def __init__(self, runtime, nbytes):
        self.nbytes = ((nbytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        self.mapping = mmap.mmap(-1, self.nbytes)
        self.ptr = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        try:
            runtime.host_register(self.ptr, self.nbytes)
        except BaseException:
            self.mapping.close()
            raise

    def close(self, runtime):
        runtime.host_unregister(self.ptr)
        self.mapping.close()


class Inkling2ActivationTransfer:
    """Order copies and kernels on stream zero; wait before exposing results."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.host_input = self.host_output = None
        self.pin_unavailable = False

    def _scratch(self, name, nbytes):
        old = getattr(self, name)
        if old is not None and old.nbytes >= nbytes:
            return old
        if old is not None:
            old.close(self.runtime)
            setattr(self, name, None)
        buf = _PinnedScratch(self.runtime, nbytes)
        setattr(self, name, buf)
        return buf

    def _try_pin(self, name, nbytes):
        if self.pin_unavailable:
            return None
        try:
            return self._scratch(name, nbytes)
        except HipError as exc:
            if exc.code not in _HOST_REGISTRATION_UNAVAILABLE:
                raise
        except OSError:
            pass
        self.pin_unavailable = True
        return None

    def enqueue_upload(self, inp, x):
        """Submit a copy on stream zero. The caller fences before reusing scratch."""
        pinned = self._try_pin("host_input", x.nbytes)
        kind = HipMemcpyKind.HOST_TO_DEVICE
        if pinned is None:
            self.runtime.memcpy(inp, x.ctypes.data, x.nbytes, kind)
        else:
            ctypes.memmove(pinned.ptr, x.ctypes.data, x.nbytes)
            self.runtime.memcpy_async(inp, pinned.ptr, x.nbytes, kind, 0)

    def download(self, out, y):
        """Read grouped results and fence any preceding launches on stream zero."""
        pinned = self._try_pin("host_output", y.nbytes)
        kind = HipMemcpyKind.DEVICE_TO_HOST
        if pinned is None:
            self.runtime.memcpy(y.ctypes.data, out, y.nbytes, kind)
        else:
            try:
                self.runtime.memcpy_async(pinned.ptr, out, y.nbytes, kind, 0)
            finally:
                self.runtime.stream_synchronize(0)
            ctypes.memmove(y.ctypes.data, pinned.ptr, y.nbytes)

    @contextmanager
    def exchange(self, inp, out, x, y, *, pinned):
        runtime = self.runtime
        up, down = HipMemcpyKind.HOST_TO_DEVICE, HipMemcpyKind.DEVICE_TO_HOST
        host_input = self._try_pin("host_input", x.nbytes) if pinned else None
        host_output = self._try_pin("host_output", y.nbytes) if host_input is not None else None
        if host_output is not None:
            ctypes.memmove(host_input.ptr, x.ctypes.data, x.nbytes)
            try:
                runtime.memcpy_async(inp, host_input.ptr, x.nbytes, up, 0)
                yield
                runtime.memcpy_async(host_output.ptr, out, y.nbytes, down, 0)
            finally:
                runtime.stream_synchronize(0)
            ctypes.memmove(y.ctypes.data, host_output.ptr, y.nbytes)
            return
        runtime.memcpy(inp, x.ctypes.data, x.nbytes, up)
        yield
        runtime.memcpy(y.ctypes.data, out, y.nbytes, down)

    def close(self):
        for name in ("host_output", "host_input"):
            buf = getattr(self, name)
            if buf is not None:
                buf.close(self.runtime)
                setattr(self, name, None)
