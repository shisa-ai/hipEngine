"""Bounded append-tail snapshot/restore for the compact DMS device store.

A verify cycle appends its candidate rows into the window every head will write
next, and a rejected row must be undone. Undoing it with the full-extent
``snapshot``/``restore`` pair reads and rewrites every owned slot of the
request; when no existing token can expire over the candidate horizon the only
bytes a rollback has to recover are that append window. These helpers bound
both directions to ``rows`` slots per head.

CPU contract (fake device memory):
- ``snapshot_append_tail`` reads exactly ``rows`` slots per (layer, head) from
  ``base + live``, for both codecs, and copies the caller's live counts;
- ``restore_append_tail(snapshot, accepted_rows=k)`` rewrites only the suffix
  ``[start + k, start + rows)`` of each head's window, leaves the accepted
  prefix bytes untouched, and publishes ``live_counts + k``;
- ``k == rows`` publishes the live counts and copies no payload at all;
- one snapshot serves every row checkpoint: restores never mutate its host
  arrays, and the live-count plane it publishes is a fresh array;
- the helpers allocate nothing and copy nothing outside the bounded intervals;
- invalid metadata, codec, shape, dtype, bound, or accepted count is rejected
  before any device byte moves;
- a device copy failure propagates and leaves the live counts unpublished.
"""
from __future__ import annotations

import ctypes
from dataclasses import replace
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

# Varied per-head windows: some heads are adjacent, some are not, and the live
# counts differ per head so the tail never starts at the request base.
_LAYERS = 2
_HEADS = 3
_DIM = 4
_SLOTS = 16
_ROWS = 3
_BASES = np.array([[1, 5, 9], [2, 6, 11]], dtype=np.int32)
_LIVE = np.array([[2, 1, 0], [0, 3, 1]], dtype=np.int32)
_STARTS = _BASES + _LIVE  # [[3, 6, 9], [2, 9, 12]]


class _FakeDevice:
    """Real host bytes behind fake device pointers, with injectable failures."""

    def __init__(self) -> None:
        self.backing: dict[int, np.ndarray] = {}
        self._next_ptr = 0x10000000
        self.malloc_count = 0
        self.malloc_bytes = 0
        self.copies: list[tuple[str, int]] = []
        self.copy_targets: list[tuple[str, int, int]] = []
        self.copy_count = {"host_to_device": 0, "device_to_host": 0}
        self.fail_copy_at: dict[str, int | None] = {
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
        self.malloc_bytes += int(nbytes)
        size = int(nbytes)
        array = np.zeros(size, dtype=np.uint8)
        ptr = self._next_ptr
        self._next_ptr += max(256, (size + 255) // 256 * 256)
        self.backing[ptr] = array
        return DeviceBuffer(ptr, size)

    def free(self, buffer: DeviceBuffer, *, runtime=None) -> None:
        del runtime
        self.backing.pop(int(buffer.ptr), None)

    def _check_copy(self, kind: str) -> None:
        self.copy_count[kind] += 1
        if self.copy_count[kind] == self.fail_copy_at[kind]:
            raise RuntimeError(f"fake {kind} copy failed")

    def copy_host_array_to_device(self, buffer, array, nbytes=None, *, runtime=None):
        del runtime
        count = array.nbytes if nbytes is None else int(nbytes)
        self._check_copy("host_to_device")
        self.copies.append(("host_to_device", count))
        self.copy_targets.append(("host_to_device", int(buffer.ptr), count))
        ctypes.memmove(self._addr(int(buffer.ptr)), array.ctypes.data, count)

    def copy_device_to_host(self, host_ptr, buffer, nbytes=None, *, runtime=None):
        del runtime
        count = buffer.nbytes if nbytes is None else int(nbytes)
        self._check_copy("device_to_host")
        self.copies.append(("device_to_host", count))
        self.copy_targets.append(("device_to_host", int(buffer.ptr), count))
        ctypes.memmove(host_ptr, self._addr(int(buffer.ptr)), count)

    def reset_copies(self) -> None:
        self.copies.clear()
        self.copy_targets.clear()


def _plane_sizes(codec: str, *, heads: int, dim: int, slots: int) -> dict[str, int]:
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


def _store(monkeypatch, *, codec: str = "bf16"):
    fake = _FakeDevice()
    monkeypatch.setattr(device, "malloc", fake.malloc)
    monkeypatch.setattr(device, "free", fake.free)
    monkeypatch.setattr(device, "copy_host_array_to_device", fake.copy_host_array_to_device)
    monkeypatch.setattr(device, "copy_device_to_host", fake.copy_device_to_host)
    store = device.DMSDevicePayloadStore.__new__(device.DMSDevicePayloadStore)
    store.codec = codec
    store._payload_dtype = np.dtype(np.uint16 if codec == "bf16" else np.int8)
    store._runtime = SimpleNamespace(device_synchronize=lambda: None)
    store._layers = _LAYERS
    store._heads = _HEADS
    store._q_heads = _HEADS
    store._dim = _DIM
    store._window = _SLOTS
    store._slots = _SLOTS
    store._max_rows = 4
    store._buffers = []
    store._k_slot = [None] * _LAYERS
    store._v_slot = [None] * _LAYERS
    store._positions = [None] * _LAYERS
    store._slot_evict = [None] * _LAYERS
    store._base_meta = [None] * _LAYERS
    store._capacity_meta = [None] * _LAYERS
    store._live_meta = [None] * _LAYERS
    store._k_scales = [None] * _LAYERS if codec != "bf16" else []
    store._v_scales = [None] * _LAYERS if codec != "bf16" else []
    store._staged = [None] * _LAYERS
    store._closed = False
    return store, fake


def _codec_planes(codec: str) -> tuple[str, ...]:
    if codec == "bf16":
        return ("k_slot", "v_slot", "positions", "evict")
    return ("k_slot", "v_slot", "positions", "evict", "k_scales", "v_scales")


def _slot_bytes(store, name: str) -> int:
    if name in {"k_slot", "v_slot"}:
        return store._dim * store._payload_dtype.itemsize
    if name in {"positions", "k_scales", "v_scales"}:
        return 4
    if name == "evict":
        return 1
    raise AssertionError(f"no per-slot stride for plane {name}")


def _fill_planes(store, fake, layer: int, *, seed: int = 0) -> None:
    for index, name in enumerate(
        _plane_sizes(
            store.codec, heads=store._heads, dim=store._dim, slots=store._slots
        )
    ):
        buffer = getattr(store, _PLANES[name])[layer]
        values = fake.backing[int(buffer.ptr)]
        values[:] = (np.arange(buffer.nbytes) + seed + 17 * index) % 251


def _plane_bytes(store, fake, layer: int) -> dict[str, bytes]:
    return {
        name: fake.backing[int(getattr(store, _PLANES[name])[layer].ptr)].tobytes()
        for name in _codec_planes(store.codec)
    }


def _all_plane_bytes(store, fake) -> dict[int, dict[str, bytes]]:
    return {layer: _plane_bytes(store, fake, layer) for layer in range(store._layers)}


def _extent_bytes(extent) -> dict[str, bytes]:
    planes = {
        "k_slot": extent.k_bits,
        "v_slot": extent.v_bits,
        "positions": extent.positions,
        "evict": extent.evict,
    }
    if extent.k_scales is not None:
        planes["k_scales"] = extent.k_scales
        planes["v_scales"] = extent.v_scales
    return {name: np.ascontiguousarray(values).tobytes() for name, values in planes.items()}


def _scribble(store, fake, layer: int, start: int, length: int, *, seed: int) -> None:
    """Write recognisable non-original bytes into one head's append window."""

    for name in _codec_planes(store.codec):
        buffer = getattr(store, _PLANES[name])[layer]
        values = fake.backing[int(buffer.ptr)]
        stride = _slot_bytes(store, name)
        lo, hi = start * stride, (start + length) * stride
        values[lo:hi] = (np.arange(hi - lo) * 37 + 101 + seed) % 251


def _scribble_live(store, fake, layer: int, value: int) -> None:
    buffer = getattr(store, _PLANES["live"])[layer]
    values = fake.backing[int(buffer.ptr)]
    values[:] = np.full(store._heads, value, dtype=np.int32).view(np.uint8)


def _locate(store, ptr: int) -> tuple[int, str, int]:
    for name, attribute in _PLANES.items():
        buffers = getattr(store, attribute)
        if not buffers:
            continue
        for layer, buffer in enumerate(buffers):
            if buffer is None:
                continue
            base = int(buffer.ptr)
            if base <= ptr < base + int(buffer.nbytes):
                return layer, name, ptr - base
    raise AssertionError(f"copy touched unowned device pointer {ptr:#x}")


def _tail_slots(layer: int, accepted: int) -> set[int]:
    slots: set[int] = set()
    for head in range(_HEADS):
        start = int(_STARTS[layer, head])
        slots.update(range(start + accepted, start + _ROWS))
    return slots


def _live_bytes(store, fake, layer: int) -> bytes:
    buffer = getattr(store, _PLANES["live"])[layer]
    return fake.backing[int(buffer.ptr)].tobytes()


def _prepared(monkeypatch, codec: str):
    store, fake = _store(monkeypatch, codec=codec)
    for layer in range(store._layers):
        store._ensure_layer(layer)
        _fill_planes(store, fake, layer, seed=11 * (layer + 1))
    return store, fake


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_snapshot_append_tail_reads_only_tail_windows(monkeypatch, codec) -> None:
    store, fake = _prepared(monkeypatch, codec)
    original = _all_plane_bytes(store, fake)
    allocations = fake.malloc_count
    fake.reset_copies()
    live = _LIVE.copy()

    snapshot = store.snapshot_append_tail(_BASES, live, _ROWS)

    # Nothing allocated, and every read lands inside one head's tail window.
    assert fake.malloc_count == allocations
    planes = _codec_planes(codec)
    assert len(fake.copies) == _LAYERS * _HEADS * len(planes)
    for kind, ptr, nbytes in fake.copy_targets:
        assert kind == "device_to_host"
        layer, name, offset = _locate(store, ptr)
        stride = _slot_bytes(store, name)
        assert offset % stride == 0 and nbytes % stride == 0
        slots = set(range(offset // stride, (offset + nbytes) // stride))
        assert len(slots) == _ROWS
        assert slots <= _tail_slots(layer, 0)
    expected_read_bytes = _LAYERS * _HEADS * _ROWS * sum(
        _slot_bytes(store, name) for name in planes
    )
    assert sum(nbytes for _kind, nbytes in fake.copies) == expected_read_bytes

    # Per-head extents start at base + live and carry exactly the tail bytes.
    assert len(snapshot.extents) == _LAYERS * _HEADS
    for extent in snapshot.extents:
        layer, head = extent.layer, extent.head
        start = int(_STARTS[layer, head])
        assert extent.start == start
        assert extent.length == _ROWS
        for name, raw in _extent_bytes(extent).items():
            stride = _slot_bytes(store, name)
            assert raw == original[layer][name][start * stride:(start + _ROWS) * stride]

    # The live-count plane is a copy of the caller's, not a view of it.
    assert snapshot.live_counts.dtype == np.int32
    np.testing.assert_array_equal(snapshot.live_counts, _LIVE)
    assert snapshot.live_counts is not live
    live[...] = 99
    np.testing.assert_array_equal(snapshot.live_counts, _LIVE)


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
@pytest.mark.parametrize("accepted", [0, 1, _ROWS - 1])
def test_restore_append_tail_restores_suffix_only(monkeypatch, codec, accepted) -> None:
    store, fake = _prepared(monkeypatch, codec)
    snapshot = store.snapshot_append_tail(_BASES, _LIVE, _ROWS)
    snapshot_bytes = [_extent_bytes(extent) for extent in snapshot.extents]
    # Simulate the candidate appends: junk in every window plus horizon counts.
    for layer in range(_LAYERS):
        for head in range(_HEADS):
            _scribble(store, fake, layer, int(_STARTS[layer, head]), _ROWS, seed=head)
        _scribble_live(store, fake, layer, 99)
    scribbled = _all_plane_bytes(store, fake)
    allocations = fake.malloc_count
    fake.reset_copies()

    store.restore_append_tail(snapshot, accepted_rows=accepted)

    assert fake.malloc_count == allocations
    planes = _codec_planes(codec)
    suffix_rows = _ROWS - accepted
    expected_bytes = _LAYERS * _HEADS * suffix_rows * sum(
        _slot_bytes(store, name) for name in planes
    ) + _LAYERS * _HEADS * 4
    assert len(fake.copies) == _LAYERS * _HEADS * len(planes) + _LAYERS
    assert sum(nbytes for _kind, nbytes in fake.copies) == expected_bytes
    for kind, ptr, nbytes in fake.copy_targets:
        assert kind == "host_to_device"
        layer, name, offset = _locate(store, ptr)
        if name == "live":
            assert (offset, nbytes) == (0, _HEADS * 4)
            continue
        stride = _slot_bytes(store, name)
        slots = set(range(offset // stride, (offset + nbytes) // stride))
        assert len(slots) == suffix_rows
        assert slots <= _tail_slots(layer, accepted)

    # Accepted prefix untouched, suffix byte-exact, live counts advanced.
    for extent, expected in zip(snapshot.extents, snapshot_bytes):
        layer, start = extent.layer, extent.start
        current = _plane_bytes(store, fake, layer)
        for name, raw in expected.items():
            stride = _slot_bytes(store, name)
            assert current[name][start * stride:(start + accepted) * stride] == (
                scribbled[layer][name][start * stride:(start + accepted) * stride]
            )
            assert current[name][(start + accepted) * stride:(start + _ROWS) * stride] == (
                raw[accepted * stride:]
            )
    for layer in range(_LAYERS):
        np.testing.assert_array_equal(store.live_counts(layer), _LIVE[layer] + accepted)
    # The snapshot itself still holds its original tail bytes.
    assert [_extent_bytes(extent) for extent in snapshot.extents] == snapshot_bytes


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_one_snapshot_serves_every_row_checkpoint(monkeypatch, codec) -> None:
    store, fake = _prepared(monkeypatch, codec)
    snapshot = store.snapshot_append_tail(_BASES, _LIVE, _ROWS)
    snapshot_bytes = [_extent_bytes(extent) for extent in snapshot.extents]

    # A verifier appends its candidate rows and restores from the same
    # checkpoint at several acceptance depths; each cycle must recover the
    # identical suffix from the one shared snapshot.
    for accepted in (1, _ROWS - 1, 0, _ROWS - 1, 2):
        for layer in range(_LAYERS):
            for head in range(_HEADS):
                _scribble(
                    store, fake, layer, int(_STARTS[layer, head]), _ROWS, seed=head
                )
            _scribble_live(store, fake, layer, 99)
        scribbled = _all_plane_bytes(store, fake)
        store.restore_append_tail(snapshot, accepted_rows=accepted)
        for extent, expected in zip(snapshot.extents, snapshot_bytes):
            layer, start = extent.layer, extent.start
            current = _plane_bytes(store, fake, layer)
            for name, raw in expected.items():
                stride = _slot_bytes(store, name)
                assert current[name][start * stride:(start + accepted) * stride] == (
                    scribbled[layer][name][start * stride:(start + accepted) * stride]
                )
                assert current[name][
                    (start + accepted) * stride:(start + _ROWS) * stride
                ] == raw[accepted * stride:]
        for layer in range(_LAYERS):
            np.testing.assert_array_equal(
                store.live_counts(layer), _LIVE[layer] + accepted
            )
    assert [_extent_bytes(extent) for extent in snapshot.extents] == snapshot_bytes
    np.testing.assert_array_equal(snapshot.live_counts, _LIVE)


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_full_horizon_acceptance_publishes_live_without_payload_copies(
    monkeypatch, codec
) -> None:
    store, fake = _prepared(monkeypatch, codec)
    snapshot = store.snapshot_append_tail(_BASES, _LIVE, _ROWS)
    for layer in range(_LAYERS):
        for head in range(_HEADS):
            _scribble(store, fake, layer, int(_STARTS[layer, head]), _ROWS, seed=head)
        _scribble_live(store, fake, layer, 99)
    scribbled = _all_plane_bytes(store, fake)
    allocations = fake.malloc_count
    fake.reset_copies()

    store.restore_append_tail(snapshot, accepted_rows=_ROWS)

    # The accepted prefix is the whole window: no payload byte is rewritten.
    assert fake.malloc_count == allocations
    assert [kind for kind, _nbytes in fake.copies] == ["host_to_device"] * _LAYERS
    assert [nbytes for _kind, nbytes in fake.copies] == [_HEADS * 4] * _LAYERS
    assert _all_plane_bytes(store, fake) == scribbled
    for layer in range(_LAYERS):
        np.testing.assert_array_equal(store.live_counts(layer), _LIVE[layer] + _ROWS)


@pytest.mark.parametrize(
    "base_offsets, live_counts, rows, error",
    [
        (_BASES[:1], _LIVE, _ROWS, ValueError),
        (_BASES, _LIVE[:1], _ROWS, ValueError),
        (_BASES.astype(np.int64), _LIVE, _ROWS, ValueError),
        (_BASES, _LIVE.astype(np.int64), _ROWS, ValueError),
        (-_BASES, _LIVE, _ROWS, ValueError),
        (_BASES, -_LIVE, _ROWS, ValueError),
        (_BASES, _LIVE, 0, ValueError),
        (_BASES, _LIVE, -2, ValueError),
        (_BASES, _LIVE, 2.0, TypeError),
        (_BASES, _LIVE, _SLOTS, ValueError),  # base 11 + live 1 + 16 slots
    ],
)
def test_snapshot_append_tail_rejects_invalid_input_before_reads(
    monkeypatch, base_offsets, live_counts, rows, error
) -> None:
    store, fake = _prepared(monkeypatch, "bf16")
    original = _all_plane_bytes(store, fake)
    allocations = fake.malloc_count
    fake.reset_copies()

    with pytest.raises(error):
        store.snapshot_append_tail(base_offsets, live_counts, rows)

    assert fake.copies == []
    assert fake.malloc_count == allocations
    assert _all_plane_bytes(store, fake) == original


def _invalid_restores(codec: str):
    """(label, snapshot, accepted_rows, error) cases, all rejected pre-copy."""

    payload_dtype = np.uint16 if codec == "bf16" else np.int8
    store_snapshot = device.DMSDevicePayloadSnapshot(
        extents=tuple(
            device.DMSDeviceExtentSnapshot(
                layer=layer,
                head=head,
                start=int(_STARTS[layer, head]),
                length=_ROWS,
                k_bits=np.zeros((_ROWS, _DIM), dtype=payload_dtype),
                v_bits=np.zeros((_ROWS, _DIM), dtype=payload_dtype),
                positions=np.zeros(_ROWS, dtype=np.int32),
                evict=np.zeros(_ROWS, dtype=np.uint8),
                k_scales=None if codec == "bf16" else np.zeros(_ROWS, dtype=np.float32),
                v_scales=None if codec == "bf16" else np.zeros(_ROWS, dtype=np.float32),
            )
            for layer in range(_LAYERS)
            for head in range(_HEADS)
        ),
        live_counts=_LIVE.copy(),
    )
    extents = store_snapshot.extents
    yield "not-a-snapshot", {}, 0, TypeError
    yield "negative-accepted", store_snapshot, -1, ValueError
    yield "oversized-accepted", store_snapshot, _ROWS + 1, ValueError
    yield "float-accepted", store_snapshot, 1.0, TypeError
    yield "short-extents", replace(store_snapshot, extents=extents[:-1]), 0, ValueError
    yield (
        "duplicate-extent",
        replace(store_snapshot, extents=extents[:-1] + (extents[0],)),
        0,
        ValueError,
    )
    yield (
        "mixed-horizon",
        replace(store_snapshot, extents=(replace(extents[0], length=_ROWS - 1),) + extents[1:]),
        0,
        ValueError,
    )
    yield (
        "zero-length",
        replace(store_snapshot, extents=(replace(extents[0], length=0),) + extents[1:]),
        0,
        ValueError,
    )
    yield (
        "out-of-range-window",
        replace(
            store_snapshot,
            extents=(replace(extents[0], start=_SLOTS - 1),) + extents[1:],
        ),
        0,
        ValueError,
    )
    yield (
        "negative-start",
        replace(store_snapshot, extents=(replace(extents[0], start=-1),) + extents[1:]),
        0,
        ValueError,
    )
    yield (
        "bad-layer",
        replace(store_snapshot, extents=(replace(extents[0], layer=_LAYERS),) + extents[1:]),
        0,
        ValueError,
    )
    yield (
        "k-dtype",
        replace(
            store_snapshot,
            extents=(replace(extents[0], k_bits=np.zeros((_ROWS, _DIM), np.float32)),)
            + extents[1:],
        ),
        0,
        ValueError,
    )
    yield (
        "v-shape",
        replace(
            store_snapshot,
            extents=(replace(extents[0], v_bits=np.zeros((_ROWS, _DIM + 1), np.uint16)),)
            + extents[1:],
        ),
        0,
        ValueError,
    )
    yield (
        "positions-dtype",
        replace(
            store_snapshot,
            extents=(replace(extents[0], positions=np.zeros(_ROWS, np.int64)),) + extents[1:],
        ),
        0,
        ValueError,
    )
    yield (
        "evict-shape",
        replace(
            store_snapshot,
            extents=(replace(extents[0], evict=np.zeros(_ROWS + 1, np.uint8)),) + extents[1:],
        ),
        0,
        ValueError,
    )
    yield (
        "live-shape",
        replace(store_snapshot, live_counts=np.zeros((_LAYERS, _HEADS + 1), np.int32)),
        0,
        ValueError,
    )
    yield (
        "live-dtype",
        replace(store_snapshot, live_counts=_LIVE.astype(np.int64)),
        0,
        ValueError,
    )
    scales = np.zeros(_ROWS, dtype=np.float32)
    if codec == "bf16":
        yield (
            "scales-in-bf16",
            replace(
                store_snapshot,
                extents=(
                    replace(extents[0], k_scales=scales, v_scales=scales),
                )
                + extents[1:],
            ),
            0,
            ValueError,
        )
    else:
        yield (
            "missing-scales",
            replace(store_snapshot, extents=(replace(extents[0], k_scales=None),) + extents[1:]),
            0,
            ValueError,
        )
        yield (
            "scale-dtype",
            replace(
                store_snapshot,
                extents=(
                    replace(
                        extents[0],
                        k_scales=scales.astype(np.float64),
                        v_scales=scales,
                    ),
                )
                + extents[1:],
            ),
            0,
            ValueError,
        )


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_restore_append_tail_rejects_invalid_input_before_copies(
    monkeypatch, codec
) -> None:
    store, fake = _prepared(monkeypatch, codec)
    original = _all_plane_bytes(store, fake)
    allocations = fake.malloc_count
    seen = 0
    for label, snapshot, accepted, error in _invalid_restores(codec):
        seen += 1
        fake.reset_copies()
        with pytest.raises(error, match=None):
            store.restore_append_tail(snapshot, accepted_rows=accepted)
        assert fake.copies == [], label
        assert fake.malloc_count == allocations, label
        assert _all_plane_bytes(store, fake) == original, label
    assert seen >= 18


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_append_tail_helpers_require_live_device_planes(monkeypatch, codec) -> None:
    store, fake = _store(monkeypatch, codec=codec)

    with pytest.raises(RuntimeError, match="requires live device planes"):
        store.snapshot_append_tail(_BASES, _LIVE, _ROWS)
    assert fake.malloc_count == 0

    store._ensure_layer(0)
    store._ensure_layer(1)
    snapshot = store.snapshot_append_tail(_BASES, _LIVE, _ROWS)
    store.stage_layer(0)

    with pytest.raises(RuntimeError, match="requires live device planes"):
        store.restore_append_tail(snapshot, accepted_rows=1)
    assert store.staged_layers == (0,)

    store._ensure_layer(0)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.snapshot_append_tail(_BASES, _LIVE, _ROWS)
    with pytest.raises(RuntimeError, match="closed"):
        store.restore_append_tail(snapshot)


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_restore_append_tail_propagates_copy_failure(monkeypatch, codec) -> None:
    store, fake = _prepared(monkeypatch, codec)
    snapshot = store.snapshot_append_tail(_BASES, _LIVE, _ROWS)
    snapshot_bytes = [_extent_bytes(extent) for extent in snapshot.extents]
    for layer in range(_LAYERS):
        for head in range(_HEADS):
            _scribble(store, fake, layer, int(_STARTS[layer, head]), _ROWS, seed=head)
        _scribble_live(store, fake, layer, 99)
    horizon_live = [_live_bytes(store, fake, layer) for layer in range(_LAYERS)]
    fake.fail_copy_at["host_to_device"] = 2

    with pytest.raises(RuntimeError, match="host_to_device copy failed"):
        store.restore_append_tail(snapshot, accepted_rows=1)

    # The live-count plane is published last, so a payload failure leaves the
    # device counts at their post-append value and the snapshot intact.
    assert [_live_bytes(store, fake, layer) for layer in range(_LAYERS)] == horizon_live
    assert [_extent_bytes(extent) for extent in snapshot.extents] == snapshot_bytes

    fake.fail_copy_at["host_to_device"] = None
    store.restore_append_tail(snapshot, accepted_rows=1)
    for extent, expected in zip(snapshot.extents, snapshot_bytes):
        layer, start = extent.layer, extent.start
        current = _plane_bytes(store, fake, layer)
        for name, raw in expected.items():
            stride = _slot_bytes(store, name)
            assert current[name][(start + 1) * stride:(start + _ROWS) * stride] == raw[stride:]
    for layer in range(_LAYERS):
        np.testing.assert_array_equal(store.live_counts(layer), _LIVE[layer] + 1)


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_snapshot_append_tail_propagates_read_failure(monkeypatch, codec) -> None:
    store, fake = _prepared(monkeypatch, codec)
    original = _all_plane_bytes(store, fake)
    fake.fail_copy_at["device_to_host"] = 3

    with pytest.raises(RuntimeError, match="device_to_host copy failed"):
        store.snapshot_append_tail(_BASES, _LIVE, _ROWS)

    # A failed read mutates nothing and a retry succeeds.
    assert _all_plane_bytes(store, fake) == original
    fake.fail_copy_at["device_to_host"] = None
    snapshot = store.snapshot_append_tail(_BASES, _LIVE, _ROWS)
    assert len(snapshot.extents) == _LAYERS * _HEADS
