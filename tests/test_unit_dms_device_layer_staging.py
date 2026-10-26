"""Temporary host staging for already packed compact DMS device layers.

A full-window layer-outer prefill kept every compact payload resident while the
bulk hidden/scratch workspace and the shared oracle were still live, and the
store failed allocating a later layer's K payload. The store now hands one
finished layer's device planes back as host bytes and releases their device
buffers, and the next touch of that layer restores it byte for byte.

CPU contract (fake device memory):
- staging copies K/V, INT8 scales, positions, evict flags and the
  base/capacity/live metadata exactly, then frees only that layer's buffers and
  drops them from the store's allocation ledger;
- the next ``_ensure_layer`` restores every plane byte for byte, for BF16 and
  INT8 codecs;
- a failed stage (read failure) leaves the live device buffers untouched;
- a failed restore (allocation or copy) frees its partial allocation, leaks
  nothing, and keeps the recoverable host copy;
- staging an unallocated layer stages nothing, and an already staged layer
  returns the existing host copy;
- ``close`` stays idempotent and drops the staged host copies.
"""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kvcache import dms_device as device

_PLANES = {
    "k_slot": "_k_slot",
    "v_slot": "_v_slot",
    "positions": "_positions",
    "evict": "_slot_evict",
    "base": "_base_meta",
    "capacity": "_capacity_meta",
    "live": "_live_meta",
    "k_scales": "_k_scales",
    "v_scales": "_v_scales",
}


class _FakeDevice:
    """Real host bytes behind fake device pointers, with injectable failures."""

    def __init__(self) -> None:
        self.backing: dict[int, np.ndarray] = {}
        self.freed: list[DeviceBuffer] = []
        self._next_ptr = 0x10000000
        self.malloc_count = 0
        self.copy_count = {"host_to_device": 0, "device_to_host": 0}
        self.fail_malloc_at: int | None = None
        self.fail_copy_at: dict[str, int] = {
            "host_to_device": None,
            "device_to_host": None,
        }

    def _addr(self, ptr: int) -> int:
        for base, array in self.backing.items():
            if base <= ptr < base + array.nbytes:
                return int(array.ctypes.data) + (ptr - base)
        raise AssertionError(f"copy touched unowned device pointer {ptr:#x}")

    def malloc(self, nbytes: int, *, runtime=None) -> DeviceBuffer:
        del runtime
        self.malloc_count += 1
        if self.fail_malloc_at is not None and self.malloc_count == self.fail_malloc_at:
            raise MemoryError("fake device malloc failed")
        size = int(nbytes)
        array = np.zeros(size, dtype=np.uint8)
        ptr = self._next_ptr
        self._next_ptr += max(256, (size + 255) // 256 * 256)
        self.backing[ptr] = array
        return DeviceBuffer(ptr, size)

    def free(self, buffer: DeviceBuffer, *, runtime=None) -> None:
        del runtime
        self.freed.append(buffer)
        self.backing.pop(int(buffer.ptr), None)

    def _check_copy(self, kind: str) -> None:
        self.copy_count[kind] += 1
        if self.copy_count[kind] == self.fail_copy_at[kind]:
            raise RuntimeError(f"fake {kind} copy failed")

    def copy_host_array_to_device(self, buffer, array, nbytes=None, *, runtime=None):
        del runtime
        count = array.nbytes if nbytes is None else int(nbytes)
        self._check_copy("host_to_device")
        ctypes.memmove(self._addr(int(buffer.ptr)), array.ctypes.data, count)

    def copy_device_to_host(self, host_ptr, buffer, nbytes=None, *, runtime=None):
        del runtime
        count = buffer.nbytes if nbytes is None else int(nbytes)
        self._check_copy("device_to_host")
        ctypes.memmove(host_ptr, self._addr(int(buffer.ptr)), count)

    def plane_bytes(self, buffer: DeviceBuffer) -> bytes:
        address = self._addr(int(buffer.ptr))
        return ctypes.string_at(address, int(buffer.nbytes))


def _plane_sizes(
    codec: str, *, heads: int, dim: int, slots: int
) -> dict[str, int]:
    payload = slots * dim * (2 if codec == "bf16" else 1)
    sizes = {
        "k_slot": payload,
        "v_slot": payload,
        "positions": slots * 4,
        "evict": slots,
        "base": heads * 4,
        "capacity": heads * 4,
        "live": heads * 4,
    }
    if codec != "bf16":
        sizes["k_scales"] = slots * 4
        sizes["v_scales"] = slots * 4
    return sizes


def _store(
    monkeypatch,
    *,
    codec: str = "bf16",
    layers: int = 2,
    heads: int = 2,
    dim: int = 4,
    slots: int = 8,
):
    fake = _FakeDevice()
    monkeypatch.setattr(device, "malloc", fake.malloc)
    monkeypatch.setattr(device, "free", fake.free)
    monkeypatch.setattr(device, "copy_host_array_to_device", fake.copy_host_array_to_device)
    monkeypatch.setattr(device, "copy_device_to_host", fake.copy_device_to_host)
    store = device.DMSDevicePayloadStore.__new__(device.DMSDevicePayloadStore)
    store.codec = codec
    store._payload_dtype = np.dtype(np.uint16 if codec == "bf16" else np.int8)
    store._runtime = SimpleNamespace(device_synchronize=lambda: None)
    store._layers = layers
    store._heads = heads
    store._q_heads = heads
    store._dim = dim
    store._window = slots
    store._slots = slots
    store._max_rows = 4
    store._buffers = []
    store._k_slot = [None] * layers
    store._v_slot = [None] * layers
    store._positions = [None] * layers
    store._slot_evict = [None] * layers
    store._base_meta = [None] * layers
    store._capacity_meta = [None] * layers
    store._live_meta = [None] * layers
    store._k_scales = [None] * layers if codec != "bf16" else []
    store._v_scales = [None] * layers if codec != "bf16" else []
    store._staged = [None] * layers
    store._closed = False
    return store, fake


def _fill_planes(store, fake, layer: int, *, seed: int = 0) -> None:
    for index, name in enumerate(_plane_sizes(
        store.codec, heads=store._heads, dim=store._dim, slots=store._slots
    )):
        buffer = getattr(store, _PLANES[name])[layer]
        values = fake.backing[int(buffer.ptr)]
        values[:] = (np.arange(buffer.nbytes) + seed + 17 * index) % 251


def _plane_bytes(store, fake, layer: int) -> dict[str, bytes]:
    return {
        name: fake.plane_bytes(getattr(store, _PLANES[name])[layer])
        for name in _plane_sizes(
            store.codec, heads=store._heads, dim=store._dim, slots=store._slots
        )
    }


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_stage_and_restore_round_trip_is_byte_exact(monkeypatch, codec) -> None:
    store, fake = _store(monkeypatch, codec=codec)
    store._ensure_layer(0)
    store._ensure_layer(1)
    _fill_planes(store, fake, 0, seed=3)
    _fill_planes(store, fake, 1, seed=41)
    expected = _plane_bytes(store, fake, 0)
    other = _plane_bytes(store, fake, 1)
    sizes = _plane_sizes(codec, heads=2, dim=4, slots=8)
    resident = store.resident_bytes
    layer0_buffers = [getattr(store, _PLANES[name])[0] for name in sizes]

    staged = store.stage_layer(0)

    assert staged.layer == 0
    assert staged.allocated is True
    assert staged.nbytes == sum(sizes.values())
    for name, raw in expected.items():
        np.testing.assert_array_equal(getattr(staged, name), np.frombuffer(raw, np.uint8))
    # Device payload released, ledger row dropped, nothing else touched.
    assert store.resident_bytes == resident - sum(sizes.values())
    assert store.staged_layers == (0,)
    assert store.staged_host_bytes == sum(sizes.values())
    assert sorted(buffer.ptr for buffer in fake.freed) == sorted(
        buffer.ptr for buffer in layer0_buffers
    )
    for name in sizes:
        assert getattr(store, _PLANES[name])[0] is None
    assert {buffer.ptr for buffer in store._buffers}.isdisjoint(
        {buffer.ptr for buffer in layer0_buffers}
    )
    assert _plane_bytes(store, fake, 1) == other

    # Staging again is a no-op that returns the same recoverable copy.
    assert store.stage_layer(0) is staged
    assert len(fake.freed) == len(sizes)

    store._ensure_layer(0)

    assert store.staged_layers == ()
    assert store.staged_host_bytes == 0
    assert store.resident_bytes == resident
    assert _plane_bytes(store, fake, 0) == expected
    assert _plane_bytes(store, fake, 1) == other
    # The restored planes are live device buffers again, not the freed ones.
    for name in sizes:
        assert getattr(store, _PLANES[name])[0].ptr not in {
            buffer.ptr for buffer in layer0_buffers
        }

    # The store's own readback agrees with the bytes, plane by plane.
    view = store.layer_view(0)
    payload_dtype = np.uint16 if codec == "bf16" else np.int8
    for field, name in (
        ("k_bits", "k_slot"),
        ("v_bits", "v_slot"),
    ):
        np.testing.assert_array_equal(
            getattr(view, field),
            np.frombuffer(expected[name], payload_dtype).reshape(8, 4),
        )
    np.testing.assert_array_equal(
        view.positions, np.frombuffer(expected["positions"], np.int32)
    )
    np.testing.assert_array_equal(
        view.evict, np.frombuffer(expected["evict"], np.uint8)
    )
    if codec == "bf16":
        assert view.k_scales is None and view.v_scales is None
    else:
        np.testing.assert_array_equal(
            view.k_scales, np.frombuffer(expected["k_scales"], np.float32)
        )
        np.testing.assert_array_equal(
            view.v_scales, np.frombuffer(expected["v_scales"], np.float32)
        )


def test_unallocated_layer_stages_nothing_and_allocates_fresh(monkeypatch) -> None:
    store, fake = _store(monkeypatch)
    sizes = _plane_sizes("bf16", heads=2, dim=4, slots=8)

    staged = store.stage_layer(0)

    assert staged.allocated is False
    assert staged.nbytes == 0
    assert store.staged_layers == (0,)
    assert fake.freed == []
    assert store._buffers == []

    store._ensure_layer(0)

    assert store.staged_layers == ()
    assert len(store._buffers) == len(sizes)
    assert store.resident_bytes == sum(sizes.values())
    # Idempotent second touch allocates nothing further.
    store._ensure_layer(0)
    assert len(store._buffers) == len(sizes)


def test_stage_read_failure_leaves_live_buffers_untouched(monkeypatch) -> None:
    store, fake = _store(monkeypatch, codec="int8_per_token_head")
    store._ensure_layer(0)
    _fill_planes(store, fake, 0, seed=9)
    expected = _plane_bytes(store, fake, 0)
    resident = store.resident_bytes
    buffers = list(store._buffers)
    fake.fail_copy_at["device_to_host"] = 4

    with pytest.raises(RuntimeError, match="device_to_host copy failed"):
        store.stage_layer(0)

    assert fake.freed == []
    assert store._buffers == buffers
    assert store.resident_bytes == resident
    assert store.staged_layers == ()
    assert _plane_bytes(store, fake, 0) == expected
    # A retry after the failure stages the layer normally.
    fake.fail_copy_at["device_to_host"] = None
    staged = store.stage_layer(0)
    assert staged.allocated is True
    assert store.staged_layers == (0,)


def test_restore_allocation_failure_keeps_host_copy_and_leaks_nothing(
    monkeypatch,
) -> None:
    store, fake = _store(monkeypatch, codec="int8_per_token_head")
    store._ensure_layer(0)
    _fill_planes(store, fake, 0, seed=5)
    expected = _plane_bytes(store, fake, 0)
    sizes = _plane_sizes("int8_per_token_head", heads=2, dim=4, slots=8)
    staged = store.stage_layer(0)
    freed_after_stage = len(fake.freed)
    resident = store.resident_bytes
    fake.fail_malloc_at = fake.malloc_count + 3

    with pytest.raises(MemoryError, match="fake device malloc failed"):
        store._ensure_layer(0)

    # Nothing of the failed attempt survives, and the host copy is intact.
    assert len(store._buffers) == 0
    assert store.resident_bytes == resident
    assert len(fake.freed) == freed_after_stage + 2
    assert store.staged_layers == (0,)
    assert store.staged_host_bytes == staged.nbytes
    assert store._k_slot[0] is None
    for name in sizes:
        assert getattr(store, _PLANES[name])[0] is None
    for name, raw in expected.items():
        np.testing.assert_array_equal(getattr(staged, name), np.frombuffer(raw, np.uint8))

    fake.fail_malloc_at = None
    store._ensure_layer(0)
    assert store.staged_layers == ()
    assert _plane_bytes(store, fake, 0) == expected


def test_restore_copy_failure_keeps_host_copy_and_leaks_nothing(monkeypatch) -> None:
    store, fake = _store(monkeypatch, codec="int8_per_token_head")
    store._ensure_layer(0)
    _fill_planes(store, fake, 0, seed=13)
    expected = _plane_bytes(store, fake, 0)
    sizes = _plane_sizes("int8_per_token_head", heads=2, dim=4, slots=8)
    staged = store.stage_layer(0)
    freed_after_stage = len(fake.freed)
    fake.fail_copy_at["host_to_device"] = 2

    with pytest.raises(RuntimeError, match="host_to_device copy failed"):
        store._ensure_layer(0)

    assert store._buffers == []
    assert len(fake.freed) == freed_after_stage + len(sizes)
    assert store.staged_layers == (0,)
    assert store.staged_host_bytes == staged.nbytes
    for name in sizes:
        assert getattr(store, _PLANES[name])[0] is None

    fake.fail_copy_at["host_to_device"] = None
    store._ensure_layer(0)
    assert store.staged_layers == ()
    assert _plane_bytes(store, fake, 0) == expected


def test_malformed_staged_layer_is_rejected_without_allocation(monkeypatch) -> None:
    store, fake = _store(monkeypatch)
    store._staged[0] = device.DMSDeviceStagedLayer(
        layer=0, k_slot=np.zeros(4, dtype=np.uint8)
    )

    with pytest.raises(ValueError, match="k_slot plane size mismatch"):
        store._ensure_layer(0)

    assert store._buffers == []
    assert fake.freed == []
    assert store.staged_layers == (0,)


def test_close_is_idempotent_and_drops_staged_copies(monkeypatch) -> None:
    store, fake = _store(monkeypatch, codec="int8_per_token_head")
    store._ensure_layer(0)
    store._ensure_layer(1)
    staged = store.stage_layer(0)
    freed_by_stage = [buffer.ptr for buffer in fake.freed]
    live = [buffer.ptr for buffer in store._buffers]
    assert freed_by_stage and live

    store.close()
    store.close()

    assert store._buffers == []
    assert store.staged_layers == ()
    assert store.staged_host_bytes == 0
    assert staged.nbytes > 0
    # Each remaining device buffer is freed exactly once across both closes.
    assert sorted(buffer.ptr for buffer in fake.freed) == sorted(
        freed_by_stage + live
    )
    with pytest.raises(RuntimeError, match="closed"):
        store.stage_layer(1)


def test_backend_forwards_stage_and_validates_layer() -> None:
    from hipengine.kvcache.dms import DMSCompactBackend

    backend = object.__new__(DMSCompactBackend)
    backend.retrofit = SimpleNamespace(num_layers=2)
    backend._device_store = None
    with pytest.raises(ValueError, match="device payloads are not enabled"):
        backend.stage_device_prefill_layer(0)

    calls: list[int] = []

    def stage(layer: int) -> str:
        calls.append(layer)
        return "staged"

    backend._device_store = SimpleNamespace(stage_layer=stage)
    assert backend.stage_device_prefill_layer(1) == "staged"
    with pytest.raises(ValueError, match="compact DMS layer index is out of range"):
        backend.stage_device_prefill_layer(2)
    with pytest.raises(ValueError, match="compact DMS layer index is out of range"):
        backend.stage_device_prefill_layer(-1)
    assert calls == [1]
