"""HIP-guarded hardware tests for the compiled staged-exchange transport.

These run the real hipcc-built host driver against two devices and pin what
the CPU fakes cannot: bit-parity of the reduced f32 payload against both the
Python route and a host oracle, the mapped payload surviving slot
alternation, and the zero-copy consumer contract - a device kernel
(``f32_to_bf16``, the boundary cast the shard group runs) reading the mapped
host row on the second device's own stream.

Skipped without ROCm or a second visible device, like every HIP test here.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

pytest.importorskip("numpy")

try:  # HIP availability guard: no-ROCm runners skip instead of failing.
    import ctypes as _ctypes

    _ctypes.CDLL("libamdhip64.so")
    HAVE_HIP = True
except OSError:  # pragma: no cover - CI/publish runners
    HAVE_HIP = False

pytestmark = pytest.mark.skipif(not HAVE_HIP, reason="HIP runtime unavailable")


def _two_devices(rt) -> bool:
    return rt.device_count() >= 2


def _bf16_bits_from_f32(values: np.ndarray) -> np.ndarray:
    """Round f32 to bf16 bits the way the device contract (RNE) does."""

    f = np.ascontiguousarray(values, dtype="<f4")
    u = f.view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def test_compiled_exchange_reduces_bit_identically_to_the_python_route() -> None:
    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.staged import StagedExchangeTransport
    from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden = 4096
    rng = np.random.default_rng(11)
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()

    for staging_dtype in ("f32", "bf16"):
        itemsize = 4 if staging_dtype == "f32" else 2
        if staging_dtype == "f32":
            rows = {
                0: rng.standard_normal(hidden).astype("<f4"),
                1: rng.standard_normal(hidden).astype("<f4"),
            }
        else:
            rows = {
                0: _bf16_bits_from_f32(rng.standard_normal(hidden)),
                1: _bf16_bits_from_f32(rng.standard_normal(hidden)),
            }
        dev_ptrs = {}
        for device in (0, 1):
            payload = np.ascontiguousarray(rows[device].view(np.uint8).reshape(-1))
            with scoped_current_device(rt, device):
                ptr = int(rt.malloc(hidden * itemsize))
            rt.memcpy(ptr, payload.ctypes.data, hidden * itemsize, 1)  # H2D
            dev_ptrs[device] = ptr

        python_route = StagedExchangeTransport(
            rt, devices=(0, 1), streams=streams, hidden=hidden, staging_dtype=staging_dtype
        )
        compiled = CompiledStagedExchangeTransport(
            rt, devices=(0, 1), streams=streams, hidden=hidden, staging_dtype=staging_dtype
        )
        if staging_dtype == "f32":
            expected = (rows[0] + rows[1]).astype("<f4")
        else:
            wide0 = (rows[0].astype(np.uint32) << 16).view(np.float32)
            wide1 = (rows[1].astype(np.uint32) << 16).view(np.float32)
            expected = (wide0 + wide1).astype("<f4")

        readback = np.empty(hidden, dtype="<f4")
        for _call in range(2):  # both slot sets must agree
            py_ptrs = python_route.reduce(dev_ptrs)
            compiled_ptrs = compiled.reduce(dev_ptrs)
            assert compiled_ptrs[0] == compiled_ptrs[1], (
                "both ranks consume the same mapped payload row"
            )
            with scoped_current_device(rt, 0):
                rt.memcpy(readback.ctypes.data, py_ptrs[0], hidden * 4, 2)  # D2H
            assert np.array_equal(readback.view("<u4"), expected.view("<u4"))
            got = np.frombuffer(ctypes.string_at(compiled_ptrs[0], hidden * 4), dtype="<f4")
            assert np.array_equal(got.view("<u4"), expected.view("<u4")), (
                "the compiled driver's payload is bit-identical to the host oracle"
            )
        python_route.close()
        compiled.close()

    for device in (0, 1):
        rt.stream_destroy(streams[device])


def test_the_mapped_payload_feeds_the_zero_copy_boundary_cast() -> None:
    """The production consumer contract on the second device's own stream."""

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport
    from hipengine.kernels.hip_gfx1100.convert import f32_to_bf16

    rt = get_hip_runtime()
    if not _two_devices(rt):
        pytest.skip("this test needs two visible devices")
    hidden = 4096
    rng = np.random.default_rng(5)
    streams = {}
    for device in (0, 1):
        with scoped_current_device(rt, device):
            streams[device] = rt.stream_create()
    rows = {
        0: rng.standard_normal(hidden).astype("<f4"),
        1: rng.standard_normal(hidden).astype("<f4"),
    }
    dev_ptrs = {}
    for device in (0, 1):
        payload = np.ascontiguousarray(rows[device])
        with scoped_current_device(rt, device):
            ptr = int(rt.malloc(hidden * 4))
        rt.memcpy(ptr, payload.ctypes.data, hidden * 4, 1)  # H2D
        dev_ptrs[device] = ptr

    compiled = CompiledStagedExchangeTransport(
        rt, devices=(0, 1), streams=streams, hidden=hidden, staging_dtype="f32"
    )
    mapped = compiled.reduce(dev_ptrs)[1]

    # Device 1 casts the mapped host row (zero-copy over PCIe) into its own
    # device bf16 buffer on its own stream - exactly what MlpShardGroup's
    # boundary cast does after a compiled reduction.
    expected_sum = rows[0] + rows[1]
    exp_bits = (expected_sum.view("<u4") + 0x7FFF + ((expected_sum.view("<u4") >> 16) & 1)) >> 16
    got = np.empty(hidden, dtype="<u2")
    with scoped_current_device(rt, 1):
        out = int(rt.malloc(hidden * 2))
        f32_to_bf16(mapped, out, hidden, stream=streams[1], runtime=rt)
        rt.stream_synchronize(streams[1])
        rt.memcpy(got.ctypes.data, out, hidden * 2, 2)  # D2H
    assert np.array_equal(got, exp_bits.astype("<u2")), (
        "the boundary cast over the mapped payload produces the same bf16 bits"
    )
    compiled.close()
    for device in (0, 1):
        rt.stream_destroy(streams[device])
