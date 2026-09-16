"""CPU tests for batched (multi-row) staged exchange and shard-group rows.

No device: a fake runtime models device buffers and the registered pinned host
regions, so the Python transport's real numpy reduction runs over simulated
data. The compiled transport is driven through the same fake-driver ABI the
existing compiled-transport tests use. These tests pin the batched contract:
capacity-sized slots, active-row staging, the inactive tail zeroed (never
reduced as valid), row-value validation, and the unchanged single-row route.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.runtime import MemcpyKind
from hipengine.distributed.shard_group import MlpShardGroup, ShardGroupError
from hipengine.distributed.staged import StagedExchangeTransport
from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport
from hipengine.distributed.transport import TransportError, TransportStateError


class FakeRuntime:
    """Models device memory and the registered pinned host regions."""

    def __init__(self):
        self.device = 0
        self.dev_bufs: dict[int, bytearray] = {}
        self.host_regions: list[tuple[int, int]] = []
        self.next_ptr = 0x10_0000
        self.syncs = 0
        self.unregistered: list[int] = []

    # -- device selection --
    def get_device(self):
        return self.device

    def set_device(self, device):
        self.device = int(device)

    # -- allocation --
    def malloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += int(nbytes) + 64
        self.dev_bufs[ptr] = bytearray(int(nbytes))
        return ptr

    def free(self, ptr):
        self.dev_bufs.pop(int(ptr), None)

    # -- pinned host registration --
    def host_register(self, ptr, nbytes):
        self.host_regions.append((int(ptr), int(nbytes)))

    def host_unregister(self, ptr):
        self.unregistered.append(int(ptr))
        self.host_regions = [r for r in self.host_regions if r[0] != int(ptr)]

    # -- copies --
    def _region_view(self, addr, nbytes):
        addr = int(addr)
        for start, size in self.host_regions:
            if start <= addr and addr + int(nbytes) <= start + size:
                buf = (ctypes.c_ubyte * int(nbytes)).from_address(addr)
                return np.frombuffer(buf, dtype=np.uint8)
        raise AssertionError(f"address 0x{addr:x} is not inside a registered host region")

    def memcpy_async(self, dst, src, nbytes, kind, stream):
        nbytes = int(nbytes)
        if kind == MemcpyKind.DEVICE_TO_HOST:
            self._region_view(dst, nbytes)[:] = np.frombuffer(
                self.dev_bufs[int(src)][:nbytes], dtype=np.uint8
            )
        elif kind == MemcpyKind.HOST_TO_DEVICE:
            self.dev_bufs[int(dst)][:nbytes] = bytes(self._region_view(src, nbytes))
        else:  # pragma: no cover - the transports never stage device-to-device
            raise AssertionError(f"unexpected memcpy kind {kind}")

    def stream_synchronize(self, stream):
        self.syncs += 1

    # -- test helpers --
    def write_device(self, ptr, values: np.ndarray):
        self.dev_bufs[int(ptr)][: values.nbytes] = np.ascontiguousarray(values).tobytes()

    def read_device(self, ptr, nbytes, dtype):
        return np.frombuffer(bytes(self.dev_bufs[int(ptr)][: int(nbytes)]), dtype=dtype)


def _python_transport(runtime, *, hidden=4, rows=1, dtype="f32"):
    return StagedExchangeTransport(
        runtime, devices=(0, 1), streams={0: 10, 1: 11},
        hidden=hidden, staging_dtype=dtype, rows=rows,
    )


def test_single_row_reduction_is_unchanged():
    runtime = FakeRuntime()
    transport = _python_transport(runtime, hidden=4, rows=1)
    p0 = runtime.malloc(4 * 4)
    p1 = runtime.malloc(4 * 4)
    runtime.write_device(p0, np.arange(4, dtype="<f4"))
    runtime.write_device(p1, np.full(4, 10, dtype="<f4"))
    transport.reduce({0: p0, 1: p1})
    out = runtime.read_device(transport.reduced_ptr(0), 4 * 4, "<f4")
    assert np.array_equal(out, np.arange(4, dtype="<f4") + 10)
    transport.close()


def test_batched_reduction_sums_active_rows_and_zeroes_the_tail():
    runtime = FakeRuntime()
    transport = _python_transport(runtime, hidden=4, rows=3)
    # rank 0 rows: [0..11], rank 1 rows: 100 each; active 2 rows -> sum 2 rows.
    p0 = runtime.malloc(3 * 4 * 4)
    p1 = runtime.malloc(3 * 4 * 4)
    runtime.write_device(p0, np.arange(12, dtype="<f4"))
    runtime.write_device(p1, np.full(12, 100, dtype="<f4"))
    # Pre-fill the reduced buffer with a sentinel so a stale tail is detectable.
    sentinel = np.full(12, -7, dtype="<f4")
    runtime.write_device(transport.reduced_ptr(0), sentinel)
    transport.reduce({0: p0, 1: p1}, rows=2)
    out = runtime.read_device(transport.reduced_ptr(0), 3 * 4 * 4, "<f4")
    assert np.array_equal(out[:8], np.arange(8, dtype="<f4") + 100)
    assert np.array_equal(out[8:], np.zeros(4, dtype="<f4")), (
        "the inactive tail must be zeroed, never the stale sentinel"
    )
    transport.close()


def test_batched_reduction_defaults_to_capacity_rows():
    runtime = FakeRuntime()
    transport = _python_transport(runtime, hidden=4, rows=3)
    p0 = runtime.malloc(3 * 4 * 4)
    p1 = runtime.malloc(3 * 4 * 4)
    runtime.write_device(p0, np.ones(12, dtype="<f4"))
    runtime.write_device(p1, np.ones(12, dtype="<f4"))
    transport.reduce({0: p0, 1: p1})
    out = runtime.read_device(transport.reduced_ptr(0), 3 * 4 * 4, "<f4")
    assert np.array_equal(out, np.full(12, 2, dtype="<f4"))
    transport.close()


def test_batched_bf16_staging_widens_and_sums():
    runtime = FakeRuntime()
    transport = _python_transport(runtime, hidden=4, rows=2, dtype="bf16")
    p0 = runtime.malloc(2 * 4 * 2)
    p1 = runtime.malloc(2 * 4 * 2)
    a = np.arange(8, dtype="<f4")
    b = np.full(8, 2.0, dtype="<f4")
    runtime.write_device(p0, (a.astype("<u4") >> 16).astype("<u2"))
    runtime.write_device(p1, (b.astype("<u4") >> 16).astype("<u2"))
    transport.reduce({0: p0, 1: p1}, rows=2)
    out = runtime.read_device(transport.reduced_ptr(0), 2 * 4 * 4, "<f4")
    expected = ((a.astype("<u4") >> 16) << 16).view("<f4") + (
        (b.astype("<u4") >> 16) << 16
    ).view("<f4")
    assert np.array_equal(out, expected)
    transport.close()


def test_rows_values_must_be_real_positive_ints():
    runtime = FakeRuntime()
    transport = _python_transport(runtime, hidden=4, rows=3)
    p0 = runtime.malloc(3 * 4 * 4)
    p1 = runtime.malloc(3 * 4 * 4)
    for bad in (True, 1.5, "2"):
        with pytest.raises(ValueError, match="integer"):
            transport.reduce({0: p0, 1: p1}, rows=bad)
    with pytest.raises(ValueError, match="positive"):
        transport.reduce({0: p0, 1: p1}, rows=0)
    with pytest.raises(ValueError, match="capacity"):
        transport.reduce({0: p0, 1: p1}, rows=4)
    with pytest.raises(ValueError, match="integer"):
        _python_transport(runtime, hidden=4, rows=True)
    transport.close()


def test_a_missing_partial_poisons_the_batched_transport():
    runtime = FakeRuntime()
    transport = _python_transport(runtime, hidden=4, rows=2)
    with pytest.raises(TransportStateError, match="no partial given"):
        transport.reduce({0: runtime.malloc(32)}, rows=2)
    assert transport.poisoned is True
    with pytest.raises(TransportError, match="poisoned"):
        transport.reduce({0: 1, 1: 2}, rows=2)
    transport.close()


# -- the compiled transport's batched ABI selection ---------------------------

_CREATE_HANDLE = 0xBEEF
_PAYLOAD_PTR = 0x7000_0000


class FakeDriver:
    def __init__(self):
        self.create_calls = []
        self.reduce_calls = []
        self.reduce_rows_calls = []
        self.reduce_at_rows_calls = []
        self._message = b""

    def bind(self):
        driver = self

        def tp2_staged_create(devices, world, streams, hidden, dtype, slots, rows, err):
            ctypes.cast(err, ctypes.POINTER(ctypes.c_int32))[0] = 0
            driver.create_calls.append(
                (tuple(devices), int(world), int(hidden), int(dtype), int(slots), int(rows))
            )
            return _CREATE_HANDLE

        def tp2_staged_reduce(handle, partials, out_payload):
            array = ctypes.cast(partials, ctypes.POINTER(ctypes.c_void_p))
            driver.reduce_calls.append([int(array[i]) for i in range(2)])
            ctypes.cast(out_payload, ctypes.POINTER(ctypes.c_uint64))[0] = _PAYLOAD_PTR
            return 0

        def tp2_staged_reduce_rows(handle, partials, rows, out_payload):
            array = ctypes.cast(partials, ctypes.POINTER(ctypes.c_void_p))
            driver.reduce_rows_calls.append((int(rows), [int(array[i]) for i in range(2)]))
            ctypes.cast(out_payload, ctypes.POINTER(ctypes.c_uint64))[0] = _PAYLOAD_PTR
            return 0

        def tp2_staged_reduce_at_rows(handle, partials, slot, rows, out_payload):
            array = ctypes.cast(partials, ctypes.POINTER(ctypes.c_void_p))
            driver.reduce_at_rows_calls.append(
                (int(slot), int(rows), [int(array[i]) for i in range(2)])
            )
            ctypes.cast(out_payload, ctypes.POINTER(ctypes.c_uint64))[0] = (
                _PAYLOAD_PTR + int(slot) * 16
            )
            return 0

        def tp2_staged_reduce_at(handle, partials, slot, out_payload):
            ctypes.cast(out_payload, ctypes.POINTER(ctypes.c_uint64))[0] = _PAYLOAD_PTR
            return 0

        self.tp2_staged_create = tp2_staged_create
        self.tp2_staged_reduce = tp2_staged_reduce
        self.tp2_staged_reduce_rows = tp2_staged_reduce_rows
        self.tp2_staged_reduce_at = tp2_staged_reduce_at
        self.tp2_staged_reduce_at_rows = tp2_staged_reduce_at_rows
        self.tp2_staged_payload_base = lambda handle: _PAYLOAD_PTR
        self.tp2_staged_slot_stride = lambda handle: 16
        self.tp2_staged_last_error = lambda handle: self._message
        self.tp2_staged_destroy = lambda handle: None
        return self


def _compiled(runtime, driver, *, hidden=4, rows=1):
    return CompiledStagedExchangeTransport(
        runtime, devices=(0, 1), streams={0: 10, 1: 11},
        hidden=hidden, staging_dtype="f32", library=driver.bind(), rows=rows,
    )


def test_compiled_single_row_keeps_the_original_abi():
    runtime = FakeRuntime()
    driver = FakeDriver()
    transport = _compiled(runtime, driver, rows=1)
    transport.reduce({0: 0x1000, 1: 0x2000})
    assert driver.create_calls == [((0, 1), 2, 4, 0, 2, 1)]
    assert driver.reduce_calls == [[0x1000, 0x2000]]
    assert driver.reduce_rows_calls == []
    transport.close()


def test_compiled_batched_calls_the_rows_abi_with_active_rows():
    runtime = FakeRuntime()
    driver = FakeDriver()
    transport = _compiled(runtime, driver, hidden=4, rows=3)
    assert driver.create_calls == [((0, 1), 2, 4, 0, 2, 3)]
    transport.reduce({0: 0x1000, 1: 0x2000}, rows=2)
    assert driver.reduce_rows_calls == [(2, [0x1000, 0x2000])]
    assert driver.reduce_calls == []
    transport.reduce({0: 0x1000, 1: 0x2000}, rows=3, slot=1)
    assert driver.reduce_at_rows_calls == [(1, 3, [0x1000, 0x2000])]
    transport.close()


def test_compiled_batched_validates_rows():
    runtime = FakeRuntime()
    driver = FakeDriver()
    transport = _compiled(runtime, driver, hidden=4, rows=2)
    for bad in (True, 1.5, 0, 3):
        with pytest.raises(ValueError):
            transport.reduce({0: 0x1000, 1: 0x2000}, rows=bad)
    assert driver.reduce_rows_calls == []
    transport.close()


# -- shard group rows propagation ---------------------------------------------


def test_group_rows_are_threaded_to_ranks_and_transport(monkeypatch):
    import hipengine.distributed.shard_exec as shard_exec_module
    import hipengine.distributed.staged_compiled as staged_compiled_module
    import hipengine.kernels.hip_gfx1100.convert as convert_module
    import hipengine.runtime.gguf_linear as gguf_linear
    from hipengine.distributed.shard_exec import MlpShardRank
    from tests.test_unit_distributed_staged_and_shard import FakeHipRuntime, _shard_weights

    runtime = FakeHipRuntime()
    driver = FakeDriver()
    monkeypatch.setattr(
        staged_compiled_module, "build_tp2_staged_exchange", lambda **_kw: driver.bind()
    )
    monkeypatch.setattr(MlpShardRank, "_require_batched_route", lambda self, rows: None)
    monkeypatch.setattr(
        gguf_linear, "launch_gguf_linear",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        shard_exec_module, "silu_mul_separate_out_bf16", lambda *a, **k: None
    )
    monkeypatch.setattr(convert_module, "f32_to_bf16", lambda *a, **k: None)

    weights = {0: {d: _shard_weights(runtime, d) for d in (0, 1)}}
    group = MlpShardGroup(
        runtime, devices=(0, 1), streams={0: 10, 1: 11}, hidden=8,
        per_rank_ffn=4, weights=weights, driver="compiled", rows=2,
    )
    assert group.rows == 2
    assert all(rank.rows == 2 for rank in group._ranks.values())
    assert group._transport.rows == 2

    inputs = {0: runtime.malloc(2 * 8 * 2), 1: runtime.malloc(2 * 8 * 2)}
    group.forward(0, inputs, rows=2)
    assert driver.reduce_rows_calls[-1][0] == 2
    group.forward(0, inputs, rows=1)
    assert driver.reduce_rows_calls[-1][0] == 1
    with pytest.raises(ValueError, match="capacity"):
        group.forward(0, inputs, rows=3)
    group.close()


def test_group_rejects_an_empty_layer_set():
    runtime = FakeRuntime()
    with pytest.raises(ShardGroupError, match="at least one layer"):
        MlpShardGroup(
            runtime, devices=(0, 1), streams={0: 10, 1: 11},
            hidden=8, per_rank_ffn=4, weights={}, rows=2,
        )
