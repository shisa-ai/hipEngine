"""Unit tests for the Inkling activation-transfer pinned-upload pool.

Regression: every asynchronous upload used to reuse one shared pinned buffer,
so a later payload could overwrite bytes the device had not read yet. The
pool hands each in-flight upload its own buffer and reclaims a buffer only
once the event recorded after its copy reports completion.
"""

import numpy as np
import pytest

from hipengine.core.hip import HipError, HipMemcpyKind
from hipengine.loading.inkling2_activation_transfer import Inkling2ActivationTransfer


class FakeRuntime:
    def __init__(self, *, complete=False):
        self.registered = []
        self.copies = []          # (dst, src, nbytes) recorded in order
        self.events = []
        self._complete = complete

    # pin
    def host_register(self, ptr, nbytes):
        self.registered.append((ptr, nbytes))

    def host_unregister(self, ptr):
        pass

    # copies
    def memcpy(self, dst, src, nbytes, kind):
        self.copies.append(("sync", dst, src, nbytes, kind))

    def memcpy_async(self, dst, src, nbytes, kind, stream):
        self.copies.append(("async", dst, src, nbytes, kind))

    # events
    def event_create(self):
        self.events.append(False)
        return len(self.events) - 1

    def event_record(self, event, stream):
        pass

    def event_query(self, event):
        return self._complete

    def event_destroy(self, event):
        pass

    def stream_synchronize(self, stream):
        pass


def test_consecutive_uploads_use_distinct_pinned_buffers():
    runtime = FakeRuntime()
    transfer = Inkling2ActivationTransfer(runtime)
    a = np.full(1024, 1.0, dtype=np.float32)
    b = np.full(1024, 2.0, dtype=np.float32)
    transfer.enqueue_upload(0x1000, a)
    transfer.enqueue_upload(0x2000, b)
    async_copies = [c for c in runtime.copies if c[0] == "async"]
    assert len(async_copies) == 2
    assert async_copies[0][2] != async_copies[1][2], (
        "back-to-back uploads must not stage through one pinned buffer")


def test_completed_buffer_is_recycled():
    runtime = FakeRuntime(complete=True)
    transfer = Inkling2ActivationTransfer(runtime)
    for i in range(4):
        payload = np.full(64, float(i + 1), dtype=np.float32)
        transfer.enqueue_upload(0x1000 + i * 0x1000, payload)
    assert len(transfer.upload_slots) == 1
    assert len(runtime.copies) == 4


def test_pending_buffers_are_not_recycled():
    runtime = FakeRuntime()  # event_query always False: copies never retire
    transfer = Inkling2ActivationTransfer(runtime)
    for i in range(3):
        transfer.enqueue_upload(0x1000 * (i + 1), np.full(64, float(i + 1), dtype=np.float32))
    srcs = [c[2] for c in runtime.copies]
    assert len(set(srcs)) == 3
    assert len(transfer.upload_slots) == 3


def test_pin_failure_falls_back_to_synchronous_copy():
    class RejectingRuntime(FakeRuntime):
        def host_register(self, ptr, nbytes):
            raise HipError(801, "host pinning unavailable")

    runtime = RejectingRuntime()
    transfer = Inkling2ActivationTransfer(runtime)
    transfer.enqueue_upload(0x1000, np.ones(16, dtype=np.float32))
    assert transfer.pin_unavailable
    assert [c[0] for c in runtime.copies] == ["sync"]
    assert runtime.copies[0][4] == HipMemcpyKind.HOST_TO_DEVICE


def test_close_releases_slots():
    runtime = FakeRuntime()
    transfer = Inkling2ActivationTransfer(runtime)
    transfer.enqueue_upload(0x1000, np.ones(64, dtype=np.float32))
    transfer.close()
    assert transfer.upload_slots == []
