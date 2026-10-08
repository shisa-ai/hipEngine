"""The T16 tile-residency binding stages: attach, accounting, and detach.

Design: docs/QWEN4EXP-T16-RESIDENCY.md. The stage is additive (raw stays
until an explicit detach), the accounting grows and shrinks with the stage
bytes, and a detach without a t16 stage is refused.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, copy_host_to_device, host_array_ptr
from hipengine.loading.materialize import load_host_array_to_device_as_dtype
from hipengine.loading.qwen4_exp_materialize import (
    Qwen4ExpDeviceWeight,
    Qwen4ExpGGUFWeightSpec,
    attach_t16_allocation,
    detach_raw_allocation,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


class _FakeTensorRef:
    def __init__(self) -> None:
        self.tensor = None


def _synthetic_weight(nbytes: int = 64):
    spec = Qwen4ExpGGUFWeightSpec(
        slot_path="layers.7.expert_gate",
        source_ref=_FakeTensorRef(),
        quant_key="gguf_q4_k",
        layout="raw",
        allocation_names=("raw",),
        device_resident=True,
        device_nbytes=nbytes,
    )
    raw_array = np.full(nbytes, 0x3C, dtype=np.uint8)
    raw = load_host_array_to_device_as_dtype(
        "layers.7.expert_gate", raw_array, "int8", source_dtype="int8"
    )
    weight = Qwen4ExpDeviceWeight(
        spec=spec, backend="hip_gfx1151", allocations={"raw": raw}
    )
    return weight, raw_array


def _t16_allocation(tiles: np.ndarray):
    runtime = get_hip_runtime()
    dev = malloc(int(tiles.nbytes), runtime=runtime)
    copy_host_to_device(dev, host_array_ptr(tiles), int(tiles.nbytes), runtime=runtime)
    from hipengine.core.tensor import Tensor
    from hipengine.core.device import Device
    from hipengine.loading.materialize import DeviceTensorAllocation

    return DeviceTensorAllocation(
        name="layers.7.expert_gate.t16",
        source=None,
        buffer=dev,
        tensor=Tensor.from_handle(
            dev.ptr, tiles.shape, "int8", Device(kind="hip", index=0)
        ),
        owns_buffer=True,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_attach_t16_stage_is_additive_and_accounted():
    weight, _ = _synthetic_weight()
    tiles = np.full((4, 8, 384), 0x5A, dtype=np.uint8)
    t16 = _t16_allocation(tiles)

    grown = attach_t16_allocation(weight, t16)

    # The stage resolves by name; the raw default is unchanged.
    assert grown.allocation("t16").buffer.nbytes == tiles.nbytes
    assert grown.allocation().name == weight.allocation().name
    assert set(grown.allocations) == {"raw", "t16"}
    # The accounting sees the transient dual residency.
    assert grown.spec.device_nbytes == weight.spec.device_nbytes + tiles.nbytes
    assert grown.spec.allocation_names == ("raw", "t16")
    # The original weight is untouched (frozen replacement).
    assert set(weight.allocations) == {"raw"}

    # Double attach is refused.
    with pytest.raises(ValueError):
        attach_t16_allocation(grown, t16)

    grown.free()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_detach_raw_frees_raw_and_leaves_t16_default():
    weight, _ = _synthetic_weight()
    tiles = np.full((2, 4, 96), 0x2D, dtype=np.uint8)
    grown = attach_t16_allocation(weight, _t16_allocation(tiles))

    detached = detach_raw_allocation(grown)

    # The raw stage is gone; the t16 default resolves without a name.
    with pytest.raises(KeyError):
        detached.allocation("raw")
    assert detached.allocation("t16").buffer.nbytes == tiles.nbytes
    assert detached.allocation().buffer.nbytes == tiles.nbytes
    assert set(detached.allocations) == {"t16"}
    # The accounting shrank back by the raw bytes.
    assert (
        detached.spec.device_nbytes
        == grown.spec.device_nbytes - weight.allocation("raw").buffer.nbytes
    )
    # Freeing the detached weight frees only the t16 buffer (raw is already
    # freed by the detach; no double free).
    detached.free()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_detach_refused_without_t16_stage():
    weight, _ = _synthetic_weight()
    with pytest.raises(ValueError):
        detach_raw_allocation(weight)
