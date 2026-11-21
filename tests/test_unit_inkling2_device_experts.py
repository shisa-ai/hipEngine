import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.inkling2_device_experts import Inkling2DeviceExperts
from hipengine.loading.inkling2_mini import Inkling2ExpertBank


class Memory:
    def __init__(self):
        self.buffers = {}
        self.copies = 0

    def malloc(self, size):
        buffer = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(buffer)
        self.buffers[ptr] = buffer
        return ptr

    def free(self, ptr):
        del self.buffers[ptr]

    def memcpy(self, dst, src, size, kind):
        self.copies += 1
        ctypes.memmove(dst, src, size)


def _weights():
    values = np.arange(32, dtype=np.uint8).reshape(2, 2, 8)
    scales = np.array([10, 20, 30, 40], dtype=np.uint8).reshape(2, 2, 1)
    bank = Inkling2ExpertBank(values, scales, np.array([1, 2], dtype=np.float32), 4)
    return SimpleNamespace(expert_bank=lambda *_: bank)


def test_resident_projection_uses_expert_offsets_and_respects_lifetime():
    memory = Memory()
    calls = []

    def launch(w, s, x, y, rows, columns, *, nvfp4, outer_scale, expanded_scales):
        assert not expanded_scales
        calls.append((ctypes.string_at(w, 1)[0], ctypes.string_at(s, 1)[0], columns, nvfp4, outer_scale))
        target = (ctypes.c_float * rows).from_address(y)
        target[:] = [1.0, 3.0]

    with Inkling2DeviceExperts(_weights(), [0, 0], runtime=memory, launch=launch) as owner:
        assert len(memory.buffers) == 6
        assert memory.copies == 4
        np.testing.assert_array_equal(owner.project(0, "w13", 1, np.ones(16)), [1, 3])
        assert calls == [(16, 30, 16, True, 0.5)]
        assert memory.copies == 6
    assert not memory.buffers
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.project(0, "w13", 1, np.ones(16))


def test_partial_upload_failure_releases_device_memory():
    class FailingMemory(Memory):
        def memcpy(self, *args):
            if self.copies == 2:
                raise RuntimeError("copy failed")
            super().memcpy(*args)

    memory = FailingMemory()
    with pytest.raises(RuntimeError, match="copy failed"):
        Inkling2DeviceExperts(_weights(), [0], runtime=memory, launch=None)
    assert not memory.buffers
