"""The dense projections' pre-converted prefill operand is a memory rule, not a flag.

``Gemma4DeviceWeights.ensure_dense_f16w`` holds an f16 copy of every dense Q8_0
projection's prefill operand when the device keeps a reserve free afterwards,
attaches each copy to the weight it was built from as an ``f16w`` sidecar, and
that weight's own ``free()`` releases it. Byte equality of the launch that reads
it is a GPU contract (``tests/test_gpu_gguf_q8_0_t16_wmma_prefill.py``); this
file pins the ownership and the rule on a fake device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.core.hip as hip
import hipengine.loading.materialize as materialize
from hipengine.core import memory as device_memory
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_t16_prefill as q8_t16
from hipengine.loading.gemma4_gguf_device import Gemma4GGUFDeviceWeight
from hipengine.loading.materialize import DeviceTensorAllocation, index_virtual_path
from hipengine.loading.safetensors import TensorInfo
from hipengine.runtime.gemma4 import Gemma4DeviceWeights

GIB = 1 << 30


def _allocation(name: str, ptr: int, shape: tuple[int, ...], dtype: DType) -> DeviceTensorAllocation:
    nbytes = dtype.itemsize
    for dim in shape:
        nbytes *= dim
    return DeviceTensorAllocation(
        name=name,
        source=TensorInfo(
            name=name, shard_path=index_virtual_path(name), dtype="I8", shape=shape
        ),
        buffer=DeviceBuffer(ptr=ptr, nbytes=nbytes),
        tensor=Tensor.from_handle(ptr, shape, dtype, Device("hip", 0)),
    )


def _weight(
    tiles_ptr: int,
    in_features: int,
    out_features: int,
    *,
    quant: str = "gguf_q8_0",
    tiles: bool = True,
) -> Gemma4GGUFDeviceWeight:
    allocations = {}
    if tiles:
        allocations["tiles"] = _allocation(
            f"{quant}.tiles",
            tiles_ptr,
            (out_features // 16, in_features // 32, 544),
            DType.INT8,
        )
    return Gemma4GGUFDeviceWeight(
        spec=SimpleNamespace(
            slot_path=f"layers.0.ffn_down.{quant}",
            quant_key=quant,
            source=SimpleNamespace(name=f"{quant}.weight"),
            allocation_names=("tiles",) if tiles else ("raw",),
        ),
        allocations=allocations,
        backend="hip_gfx1100",
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
    # ``DeviceTensorAllocation.free`` binds this module's own ``free``.
    monkeypatch.setattr(materialize, "free", device.free)
    monkeypatch.setattr(
        q8_t16, "gguf_q8_0_t16_tiles_to_f16w",
        lambda tiles_ptr, f16w_ptr, in_features, out_features, **kwargs: device.converted.append(
            (tiles_ptr, f16w_ptr, in_features, out_features)
        ),
    )
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
    # The conversions are drained before the operand becomes reachable.
    assert device.synchronized == 1
    # The operand is the weight's own second allocation, so every consumer that
    # asks the weight sees it and nothing has to remember a pointer table.
    for item, (in_features, out_features) in zip(dense, ((2816, 4224), (2112, 2816))):
        assert item.has_allocation("f16w")
        operand = item.allocation("f16w")
        assert operand.tensor.shape == (out_features, in_features)
        assert operand.tensor.dtype is DType.FP16
        assert operand.buffer.nbytes == in_features * out_features * 2
        assert item.nbytes == sum(
            allocation.buffer.nbytes for allocation in item.allocations.values()
        ) + in_features * out_features * 2
    for item in others[:2]:
        assert not item.has_allocation("f16w")
        with pytest.raises(KeyError):
            item.allocation("f16w")

    # Idempotent: a second runner on the same weights allocates nothing more.
    assert weights.ensure_dense_f16w() == status
    assert len(device.allocated) == 2 and len(device.converted) == 2

    weights.free()
    # Each weight releases its operand before the tiles it was built from, and
    # each buffer exactly once.
    assert device.freed == [
        device.allocated[0].ptr, 0xA000,
        device.allocated[1].ptr, 0xB000,
        0xC000,
    ]


def test_the_operand_is_attached_only_after_the_conversions_are_drained(monkeypatch) -> None:
    weight = _weight(0xA000, 2816, 4224)
    weights, device = _weights(
        monkeypatch, [weight],
        free_bytes=2816 * 4224 * 2 + Gemma4DeviceWeights.DENSE_F16W_RESERVE_BYTES,
    )
    observed: list[bool] = []

    def synchronize() -> None:
        device.synchronized += 1
        observed.append(weight.has_allocation("f16w"))

    device.device_synchronize = synchronize

    weights.ensure_dense_f16w()

    assert observed == [False]
    assert weight.has_allocation("f16w")


@pytest.mark.parametrize("shortfall", [1, GIB])
def test_the_reserve_is_what_decides(monkeypatch, shortfall: int) -> None:
    """One byte under the reserve skips the whole set; nothing is allocated."""

    need = 2816 * 4224 * 2
    weight = _weight(0xA000, 2816, 4224)
    weights, device = _weights(
        monkeypatch, [weight],
        free_bytes=need + Gemma4DeviceWeights.DENSE_F16W_RESERVE_BYTES - shortfall,
    )

    status = weights.ensure_dense_f16w()

    assert status.startswith("skipped: insufficient memory")
    assert str(need) in status
    assert not device.allocated and not device.converted
    assert not weight.has_allocation("f16w")
    weights.free()
    # Only the planned tiles are released; no operand was ever allocated.
    assert device.freed == [0xA000]


def test_weights_without_tiled_dense_projections_touch_no_device(monkeypatch) -> None:
    weights, device = _weights(
        monkeypatch, [_weight(0xC000, 2816, 1408, quant="gguf_q4_k")], free_bytes=0,
    )
    monkeypatch.setattr(hip, "get_hip_runtime", lambda: pytest.fail("no device query expected"))

    assert weights.ensure_dense_f16w().startswith("none")
    assert not device.allocated


def test_a_failed_allocation_leaves_nothing_registered(monkeypatch) -> None:
    dense = [_weight(0xA000, 2816, 4224), _weight(0xB000, 2112, 2816)]
    need = (2816 * 4224 + 2112 * 2816) * 2
    weights, device = _weights(
        monkeypatch, dense,
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
    for item in dense:
        assert not item.has_allocation("f16w")
    # The weights still own nothing but their planned tiles.
    weights.free()
    assert device.freed == [device.allocated[0].ptr, 0xA000, 0xB000]


def test_a_rejected_sidecar_releases_every_copy_and_publishes_none(monkeypatch) -> None:
    """A wrapper that refuses a copy must not leave the set half attached.

    The copies are built before any is published, so a rejection releases all
    of them and no weight keeps an operand: an optional memory trade must not
    turn into a partial publication or an exception on the load path.
    """

    dense = [_weight(0xA000, 2816, 4224), _weight(0xB000, 2112, 2816)]
    need = (2816 * 4224 + 2112 * 2816) * 2
    weights, device = _weights(
        monkeypatch, dense,
        free_bytes=need + Gemma4DeviceWeights.DENSE_F16W_RESERVE_BYTES,
    )
    calls = {"n": 0}
    real = materialize.device_allocation_from_buffer

    def rejecting(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("buffer does not hold the declared shape")
        return real(*args, **kwargs)

    monkeypatch.setattr(materialize, "device_allocation_from_buffer", rejecting)

    status = weights.ensure_dense_f16w()

    assert status.startswith("skipped: sidecar rejected")
    assert "does not hold the declared shape" in status
    assert len(device.converted) == 2
    # Every copy is released, including the one whose wrapper was never built.
    assert device.freed == [buffer.ptr for buffer in device.allocated]
    for item in dense:
        assert not item.has_allocation("f16w")
    # The rejection is the weight set's answer, not something a retry re-runs.
    assert weights.ensure_dense_f16w() == status
    assert len(device.allocated) == 2
    weights.free()
    assert device.freed == [buffer.ptr for buffer in device.allocated] + [0xA000, 0xB000]


def test_a_weight_that_cannot_take_a_sidecar_is_not_a_target(monkeypatch) -> None:
    """Selection asks for a sidecar map, so publishing cannot fail per weight."""

    dense = _weight(0xA000, 2816, 4224)
    foreign = SimpleNamespace(
        spec=SimpleNamespace(
            quant_key="gguf_q8_0", source=SimpleNamespace(name="foreign.weight")
        ),
        has_allocation=lambda name: name == "tiles",
        allocation=lambda name=None: SimpleNamespace(
            tensor=SimpleNamespace(ptr=0xE000, shape=(4224 // 16, 2816 // 32, 544))
        ),
        free=lambda: None,
    )
    weights, device = _weights(
        monkeypatch, [dense, foreign],
        free_bytes=2816 * 4224 * 2 + Gemma4DeviceWeights.DENSE_F16W_RESERVE_BYTES,
    )

    status = weights.ensure_dense_f16w()

    assert status == f"resident: 1 projections, {2816 * 4224 * 2} bytes"
    assert device.converted == [(0xA000, device.allocated[0].ptr, 2816, 4224)]
    assert dense.has_allocation("f16w")
    assert not hasattr(foreign, "sidecars")
