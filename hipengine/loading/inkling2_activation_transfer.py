"""Reusable page-locked staging for Inkling activation exchanges."""

import ctypes
import mmap
from contextlib import contextmanager

import numpy as np

from hipengine.core.hip import HipError, HipMemcpyKind, capture_stream

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


class _UploadSlot:
    """Pinned staging buffer whose last async copy is tracked by an event."""

    def __init__(self, runtime, nbytes):
        self.scratch = _PinnedScratch(runtime, nbytes)
        self.event = runtime.event_create()
        self.inflight = False

    @property
    def nbytes(self):
        return self.scratch.nbytes

    @property
    def ptr(self):
        return self.scratch.ptr

    def close(self, runtime):
        runtime.event_destroy(self.event)
        self.scratch.close(runtime)



# Pinned buffers backing upload nodes baked into a CUDA/HIP graph. Each captured
# upload gets its own pin because the graph may replay long after the call; the
# pool is intentionally never released (a graph may be replayed for its whole
# lifetime) and is capped so a runaway capture fails loudly instead of leaking.
_CAPTURE_PINS = []
_CAPTURE_PIN_BUDGET = 256 * 1024 * 1024
_capture_pin_bytes = 0


def _captured_upload(runtime, dst, host):
    global _capture_pin_bytes
    raw = np.ascontiguousarray(host)
    size = max(mmap.PAGESIZE, (raw.nbytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE)
    if _capture_pin_bytes + size > _CAPTURE_PIN_BUDGET:
        raise RuntimeError("HIP graph capture pin budget exceeded; refusing to bake more uploads")
    pinned = mmap.mmap(-1, size)
    try:
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(pinned))
        runtime.host_register(ptr, size)
    except BaseException:
        pinned.close()
        raise
    ctypes.memmove(ptr, raw.ctypes.data, raw.nbytes)
    _CAPTURE_PINS.append((pinned, ptr, size))
    _capture_pin_bytes += size
    runtime.memcpy_async(dst, ptr, raw.nbytes, HipMemcpyKind.HOST_TO_DEVICE, capture_stream())

class Inkling2ActivationTransfer:
    """Order copies and kernels on stream zero; wait before exposing results.

    ``enqueue_upload`` overlaps the host copy with device work, so an upload
    returns before the GPU has staged its bytes. Each upload therefore gets a
    pinned buffer of its own from a small pool, and a buffer is handed out
    again only once the event recorded after its copy reports completion.
    Reusing one shared buffer let a later payload overwrite bytes the GPU had
    not read yet.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.host_input = self.host_output = None
        self.pin_unavailable = False
        self.upload_slots = []

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

    def _checkout_upload(self, nbytes):
        if self.pin_unavailable:
            return None
        for slot in self.upload_slots:
            if slot.nbytes < nbytes:
                continue
            if slot.inflight and not self.runtime.event_query(slot.event):
                continue
            slot.inflight = False
            return slot
        try:
            slot = _UploadSlot(self.runtime, nbytes)
        except HipError as exc:
            if exc.code not in _HOST_REGISTRATION_UNAVAILABLE:
                raise
            self.pin_unavailable = True
            return None
        except OSError:
            self.pin_unavailable = True
            return None
        self.upload_slots.append(slot)
        return slot

    def _pending_uploads(self):
        """Release pool buffers whose copies finished; handy for tests and teardown."""
        for slot in self.upload_slots:
            if slot.inflight and self.runtime.event_query(slot.event):
                slot.inflight = False

    def enqueue_upload(self, inp, x):
        """Submit a copy on stream zero; the source stays valid until its event completes.

        Inside a graph capture the copy is baked into the graph from a dedicated
        pinned buffer instead, because stream-zero copies are not captured and
        would silently replay stale data.
        """
        host = np.ascontiguousarray(x)
        if capture_stream():
            _captured_upload(self.runtime, inp, host)
            return
        kind = HipMemcpyKind.HOST_TO_DEVICE
        slot = self._checkout_upload(host.nbytes)
        if slot is None:
            self.runtime.memcpy(inp, host.ctypes.data, host.nbytes, kind)
            return
        ctypes.memmove(slot.ptr, host.ctypes.data, host.nbytes)
        self.runtime.memcpy_async(inp, slot.ptr, host.nbytes, kind, 0)
        self.runtime.event_record(slot.event, 0)
        slot.inflight = True

    def download(self, out, y):
        """Read grouped results and fence any preceding launches on stream zero."""
        if capture_stream():
            raise RuntimeError("download inside a HIP graph capture would read stale data; downloads must happen after replay")
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
        for slot in self.upload_slots:
            slot.close(self.runtime)
        self.upload_slots = []
        for name in ("host_output", "host_input"):
            buf = getattr(self, name)
            if buf is not None:
                buf.close(self.runtime)
                setattr(self, name, None)
