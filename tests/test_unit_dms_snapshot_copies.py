"""DMS snapshots coalesce adjacent owned ranges without reading gaps."""
import ctypes

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kvcache import dms_device as device


@pytest.mark.parametrize("dtype", [np.uint16, np.int8])
@pytest.mark.parametrize("starts,groups", [([1, 3, 5], 1), ([1, 5, 9], 3), ([5, 1, 3], 2)])
def test_snapshot_owned_ranges_and_restore(monkeypatch, dtype, starts, groups):
    store = device.DMSDevicePayloadStore.__new__(device.DMSDevicePayloadStore)
    store._closed = False
    store._layers, store._heads, store._slots, store._dim = 1, 3, 12, 4
    store._payload_dtype = np.dtype(dtype)
    arrays = {
        '_k_slot': np.arange(48, dtype=dtype).reshape(12, 4),
        '_v_slot': np.arange(48, dtype=dtype).reshape(12, 4) + 2,
        '_positions': np.arange(12, dtype=np.int32),
        '_slot_evict': np.arange(12, dtype=np.uint8) % 2,
        '_live_meta': np.array([2, 2, 2], dtype=np.int32),
    }
    if dtype == np.int8:
        arrays.update(_k_scales=np.arange(12, dtype=np.float32) + .5,
                      _v_scales=np.arange(12, dtype=np.float32) + 1.5)
    for name, array in arrays.items():
        setattr(store, name, [DeviceBuffer(array.ctypes.data, array.nbytes)])
    if dtype != np.int8:
        store._k_scales, store._v_scales = [], []
    monkeypatch.setattr(store, '_ensure_layer', lambda layer: None)
    reads = []
    def read(dst, src, size):
        reads.append((src.ptr, size))
        ctypes.memmove(dst, src.ptr, size)
    monkeypatch.setattr(device, 'copy_device_to_host', read)
    monkeypatch.setattr(device, 'copy_host_to_device',
                        lambda dst, src, size: ctypes.memmove(dst.ptr, src, size))
    originals = {name: array.copy() for name, array in arrays.items()}
    snapshot = store.snapshot(np.array([starts]), np.array([[2, 2, 2]]))
    planes = 6 if dtype == np.int8 else 4
    assert len(reads) == groups * planes + 1
    fields = {'k_bits': '_k_slot', 'v_bits': '_v_slot', 'positions': '_positions',
              'evict': '_slot_evict'}
    if dtype == np.int8:
        fields.update(k_scales='_k_scales', v_scales='_v_scales')
    for head, extent in enumerate(snapshot.extents):
        assert (extent.head, extent.start, extent.length) == (head, starts[head], 2)
        for field, name in fields.items():
            np.testing.assert_array_equal(getattr(extent, field), originals[name][starts[head]:starts[head]+2])
    # No copy may cover slots outside this request's owned ranges.
    owned = {slot for start in starts for slot in range(start, start + 2)}
    for ptr, size in reads:
        for name, array in arrays.items():
            if array.ctypes.data <= ptr < array.ctypes.data + array.nbytes:
                if name != '_live_meta':
                    stride = array.strides[0]
                    assert set(range((ptr-array.ctypes.data)//stride,
                                     (ptr-array.ctypes.data+size)//stride)) <= owned
                break
        else:
            pytest.fail('read outside payload planes')
    for array in arrays.values():
        array.fill(0)
    store.restore(snapshot)
    for name, array in arrays.items():
        if name == '_live_meta':
            np.testing.assert_array_equal(array, originals[name])
        else:
            for slot in range(12):
                np.testing.assert_array_equal(array[slot], originals[name][slot] if slot in owned else np.zeros_like(array[slot]))
