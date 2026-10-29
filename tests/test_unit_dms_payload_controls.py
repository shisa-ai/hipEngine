"""Explicit DMS payload ownership and host-staged overflow status checks."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kvcache import dms, dms_device
from tests.test_unit_kvcache_dms import _retrofit


def _backend(**kwargs):
    return dms.DMSCompactBackend(
        retrofit=_retrofit(), codec="bf16", slots_per_layer=64,
        max_request_rows=2, max_pack_rows=16, **kwargs,
    )


@pytest.mark.parametrize("value", ["0", "1", "true"])
def test_retired_payload_environment_does_not_select_device(monkeypatch, value):
    monkeypatch.setenv("HIPENGINE_DMS_DEVICE_PAYLOADS", value)

    def unexpected_store(**_kwargs):
        raise AssertionError("host construction must not read the retired selector")

    monkeypatch.setattr(dms, "DMSDevicePayloadStore", unexpected_store)
    assert not _backend().device_payloads_enabled
    assert not _backend(device_payloads=False).device_payloads_enabled


def test_explicit_device_request_keeps_its_owner_despite_retired_environment(monkeypatch):
    monkeypatch.setenv("HIPENGINE_DMS_DEVICE_PAYLOADS", "0")
    stores = []

    def store(**kwargs):
        owner = SimpleNamespace(options=kwargs)
        stores.append(owner)
        return owner

    monkeypatch.setattr(dms, "DMSDevicePayloadStore", store)
    backend = _backend(device_payloads=True)
    assert backend.device_payloads_enabled
    assert backend._device_store is stores[0]
    assert stores[0].options["codec"] == "bf16"


def test_explicit_device_capability_miss_is_not_silently_host_backed(monkeypatch):
    def unavailable(**_kwargs):
        raise dms_device.DMSDeviceUnavailable("fixture: no append kernel")

    monkeypatch.setattr(dms, "DMSDevicePayloadStore", unavailable)
    with pytest.raises(dms_device.DMSDeviceUnavailable, match="no append kernel"):
        _backend(device_payloads=True)


@pytest.mark.parametrize("status", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize("old_tripwire", [None, "0", "1"])
def test_host_staged_append_always_checks_current_overflow_status(
    monkeypatch, status, old_tripwire,
):
    if old_tripwire is None:
        monkeypatch.delenv("HIPENGINE_DMS_DEVICE_TRIPWIRE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_DMS_DEVICE_TRIPWIRE", old_tripwire)
    store = dms_device.DMSDevicePayloadStore.__new__(dms_device.DMSDevicePayloadStore)
    store._heads, store._dim = 2, 3
    store._stg = {
        name: DeviceBuffer(0x1000 + index * 256, 64)
        for index, name in enumerate(("append_k", "append_v", "append_evict", "status"))
    }
    events = []
    device_status = np.full(2, 99, dtype=np.int32)
    store._check_closed = lambda: None
    store._ensure_layer = lambda layer: None
    store.configure_layer = lambda *args, **kwargs: None

    def upload(name, values):
        if name == "status":
            device_status[:] = values
            events.append("clear")

    def append(*args, **kwargs):
        np.testing.assert_array_equal(device_status, [0, 0])
        device_status[:] = status
        events.append("append")

    def readback(ptr, buffer, nbytes):
        assert buffer is store._stg["status"]
        assert nbytes == device_status.nbytes
        events.append("read")
        ctypes.memmove(ptr, device_status.ctypes.data, nbytes)

    store._upload = upload
    store.append_layer_device = append
    monkeypatch.setattr(dms_device, "copy_device_to_host", readback)
    bits = np.zeros((2, 3), dtype=np.uint16)
    metadata = np.ones(2, dtype=np.int32)

    def run():
        store.append_layer(0, bits, bits, np.zeros(2, np.uint8), 17,
                           metadata, metadata, metadata)

    if any(status):
        with pytest.raises(RuntimeError, match="overflow"):
            run()
    else:
        run()
    assert events == ["clear", "append", "read"]


@pytest.mark.parametrize("old_tripwire", ["0", "1"])
def test_direct_device_append_adds_no_status_upload_or_readback(monkeypatch, old_tripwire):
    monkeypatch.setenv("HIPENGINE_DMS_DEVICE_TRIPWIRE", old_tripwire)
    store = dms_device.DMSDevicePayloadStore.__new__(dms_device.DMSDevicePayloadStore)
    store._heads, store._dim, store._layers, store._window = 2, 3, 1, 7
    store._last_row_position = 17
    store._runtime, store._library = object(), object()
    store._stg = {
        name: DeviceBuffer(0x1000 + index * 256, 64)
        for index, name in enumerate(("row_positions", "status"))
    }
    for index, name in enumerate(("_base_meta", "_capacity_meta", "_live_meta",
                                "_k_slot", "_v_slot", "_positions", "_slot_evict")):
        setattr(store, name, [DeviceBuffer(0x2000 + index * 256, 64)])
    store._check_closed = lambda: None
    store._ensure_layer = lambda layer: None
    store._scale_kwargs = lambda layer: {}
    calls = []
    store._append_fn = lambda *args, **kwargs: calls.append((args, kwargs))

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("direct append must not stage or read overflow status")

    store._upload = unexpected_copy
    monkeypatch.setattr(dms_device, "copy_device_to_host", unexpected_copy)
    store.append_layer_device(0, k_ptr=0x4000, v_ptr=0x5000,
                              evict_ptr=0x6000, row_position=17, stream=23)
    assert len(calls) == 1
    assert calls[0][0][:3] == (0x4000, 0x5000, 0x6000)
    assert calls[0][1]["stream"] == 23
