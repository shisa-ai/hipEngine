"""Request-time uploads retain sources without a device-wide queue drain."""

import ctypes
from types import SimpleNamespace

import numpy as np

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import surya


def test_upload_converts_strided_source_without_device_synchronize(monkeypatch):
    copied = []
    synchronizations = []

    def copy(buffer, ptr, nbytes):
        assert nbytes == 16
        copied.extend((ctypes.c_float * 4).from_address(ptr))

    monkeypatch.setattr(surya, "copy_host_to_device", copy)
    runner = SimpleNamespace(runtime=SimpleNamespace(
        device_synchronize=lambda: synchronizations.append(True),
    ))
    surya.SuryaGpuRunner._upload(runner, DeviceBuffer(1, 16), np.arange(8)[::2])
    assert copied == [0.0, 2.0, 4.0, 6.0]
    assert synchronizations == []
