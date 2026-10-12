"""The dense projections' pre-converted prefill operand is a memory rule, not a flag.

``Gemma4DeviceWeights.ensure_dense_f16w`` holds an f16 copy of every dense Q8_0
projection's prefill operand when the device keeps a reserve free afterwards,
registers each copy against the T16 tiles it was built from, and unregisters and
frees them with the weights. Byte equality of the launch that reads them is a
GPU contract (``tests/test_gpu_gguf_q8_0_t16_wmma_prefill.py``); this file pins
the ownership and the rule on a fake device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.core.hip as hip
from hipengine.core import memory as device_memory
from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_t16_prefill as q8_t16
from hipengine.runtime.gemma4 import Gemma4DeviceWeights

GIB = 1 << 30


def _weight(tiles_ptr: int, in_features: int, out_features: int, *, quant: str = "gguf_q8_0",
            tiles: bool = True):
    allocation = SimpleNamespace(
        buffer=SimpleNamespace(ptr=tiles_ptr),
        tensor=SimpleNamespace(shape=(out_features // 16, in_features // 32, 544)),
    )
    return SimpleNamespace(
        spec=SimpleNamespace(quant_key=quant),
        has_allocation=lambda name: tiles and name == "tiles",
        allocation=lambda name=None: allocation,
        free=lambda: None,
    )


class _Device:
    def __init__(self, free_bytes: int) -> None:
        self.free_bytes = free_bytes
        self.allocated: list[DeviceBuffer] = []
        self.freed: list[int] = []
        self.converted: list[tuple[int, int, int, int]] = []
        self.synchronized = 0

    def mem_get_info(self) -> tuple[int, int]:
        return self.free_bytes, 48 * GIB

    def device_synchronize(self) -> None:
        self.synchronized += 1

    def malloc(self, nbytes: int, **kwargs) -> DeviceBuffer:
        buffer = DeviceBuffer(ptr=0x100000 * (len(self.allocated) + 1), nbytes=nbytes)
        self.allocated.append(buffer)
        return buffer

    def free(self, buffer: DeviceBuffer, **kwargs) -> None:
        self.freed.append(buffer.ptr)


def _weights(monkeypatch: pytest.MonkeyPatch, owned, *, free_bytes: int):
    device = _Device(free_bytes)
    monkeypatch.setattr(hip, "get_hip_runtime", lambda: device)
    monkeypatch.setattr(device_memory, "malloc", device.malloc)
    monkeypatch.setattr(device_memory, "free", device.free)
    monkeypatch.setattr(
        q8_t16, "gguf_q8_0_t16_tiles_to_f16w",
        lambda tiles_ptr, f16w_ptr, in_features, out_features, **kwargs: device.converted.append(
            (tiles_ptr, f16w_ptr, in_features, out_features)
        ),
    )
    monkeypatch.setattr(q8_t16, "_F16W_TILES", {})
    weights = object.__new__(Gemma4DeviceWeights)
    weights._owned = list(owned)
    weights._freed = False
    return weights, device


def test_dense_projections_are_held_and_released_with_the_weights(monkeypatch) -> None:
    dense = [_weight(0xA000, 2816, 4224), _weight(0xB000, 2112, 2816)]
    others = [
        _weight(0xC000, 2816, 1408, quant="gguf_q4_k"),   # a tiled expert stack
        _weight(0xD000, 2816, 4096, tiles=False),         # Q8_0 with raw residency only
        SimpleNamespace(free=lambda: None),               # a plain scalar allocation
    ]
    need = (2816 * 4224 + 2112 * 2816) * 2
    weights, device = _weights(
        monkeypatch, dense + others,
        free_bytes=need + Gemma4DeviceWeights.DENSE_F16W_RESERVE_BYTES,
    )

    status = weights.ensure_dense_f16w()

    assert status == f"resident: 2 projections, {need} bytes"
    assert [buffer.nbytes for buffer in device.allocated] == [2816 * 4224 * 2, 2112 * 2816 * 2]
    assert device.converted == [
        (0xA000, device.allocated[0].ptr, 2816, 4224),
        (0xB000, device.allocated[1].ptr, 2112, 2816),
    ]
    # The conversions are drained before any launch can be routed to them.
    assert device.synchronized == 1
    assert q8_t16.gguf_q8_0_t16_f16w_for(0xA000, 2816, 4224) == device.allocated[0].ptr
    assert q8_t16.gguf_q8_0_t16_f16w_for(0xB000, 2112, 2816) == device.allocated[1].ptr
    assert q8_t16.gguf_q8_0_t16_f16w_for(0xC000, 2816, 1408) is None
    assert q8_t16.gguf_q8_0_t16_f16w_for(0xD000, 2816, 4096) is None

    # Idempotent: a second runner on the same weights allocates nothing more.
    assert weights.ensure_dense_f16w() == status
    assert len(device.allocated) == 2 and len(device.converted) == 2

    weights.free()
    assert q8_t16._F16W_TILES == {}
    assert device.freed == [buffer.ptr for buffer in device.allocated]


@pytest.mark.parametrize("shortfall", [1, GIB])
def test_the_reserve_is_what_decides(monkeypatch, shortfall: int) -> None:
    """One byte under the reserve skips the whole set; nothing is allocated."""

    need = 2816 * 4224 * 2
    weights, device = _weights(
        monkeypatch, [_weight(0xA000, 2816, 4224)],
        free_bytes=need + Gemma4DeviceWeights.DENSE_F16W_RESERVE_BYTES - shortfall,
    )

    status = weights.ensure_dense_f16w()

    assert status.startswith("skipped: insufficient memory")
    assert str(need) in status
    assert not device.allocated and not device.converted
    assert q8_t16._F16W_TILES == {}
    weights.free()
    assert not device.freed


def test_weights_without_tiled_dense_projections_touch_no_device(monkeypatch) -> None:
    weights, device = _weights(
        monkeypatch, [_weight(0xC000, 2816, 1408, quant="gguf_q4_k")], free_bytes=0,
    )
    monkeypatch.setattr(hip, "get_hip_runtime", lambda: pytest.fail("no device query expected"))

    assert weights.ensure_dense_f16w().startswith("none")
    assert not device.allocated


def test_a_failed_allocation_leaves_nothing_registered(monkeypatch) -> None:
    need = (2816 * 4224 + 2112 * 2816) * 2
    weights, device = _weights(
        monkeypatch, [_weight(0xA000, 2816, 4224), _weight(0xB000, 2112, 2816)],
        free_bytes=need + Gemma4DeviceWeights.DENSE_F16W_RESERVE_BYTES,
    )
    first = device.malloc

    def failing(nbytes: int, **kwargs):
        if device.allocated:
            raise hip.HipError(2, "out of memory")
        return first(nbytes, **kwargs)

    monkeypatch.setattr(device_memory, "malloc", failing)

    status = weights.ensure_dense_f16w()

    assert status.startswith("skipped: allocation failed")
    assert device.freed == [device.allocated[0].ptr]
    assert q8_t16._F16W_TILES == {}
