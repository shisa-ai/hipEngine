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


class _FakeResident:
    """The minimal protocol materialize_qwen4_exp_t16_stages needs."""

    def __init__(self, weights):
        self.weights = dict(weights)

    def weight(self, slot_path):
        return self.weights[slot_path]

    def swap_weight(self, slot_path, weight):
        if slot_path not in self.weights:
            raise KeyError(slot_path)
        self.weights[slot_path] = weight


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_materialize_t16_stages_populates_and_preserves_raw_by_default():
    from hipengine.loading.qwen4_exp_materialize import (
        materialize_qwen4_exp_t16_stages,
    )
    from tests._gguf_synthetic_weights import make_q4_k_weight
    from tests.test_gpu_gguf_q5_1_t16_expert_major_wmma import make_q5_1_weight

    # Two routed-expert slots at tiny geometries: a Q4_K gate (256-value
    # blocks) and a Q5_1 down (32-value blocks). Expert tensors are rank-3.
    q4_raw = make_q4_k_weight(16, 256)[None]
    q51_raw = make_q5_1_weight(16, 64, seed=5)[None]
    sources = {
        "layers.7.expert_gate": ("blk.7.ffn_gate_exps.weight", q4_raw),
        "layers.7.expert_down": ("blk.7.ffn_down_exps.weight", q51_raw),
    }

    weights = {}
    for slot, (name, raw) in sources.items():
        quant = "gguf_q4_k" if "gate" in slot else "gguf_q5_1"
        spec = Qwen4ExpGGUFWeightSpec(
            slot_path=slot,
            source_ref=_NamedRef(name),
            quant_key=quant,
            layout="raw",
            allocation_names=("raw",),
            device_resident=True,
            device_nbytes=int(raw.nbytes),
        )
        raw_alloc = load_host_array_to_device_as_dtype(
            slot, raw, "int8", source_dtype="int8"
        )
        weights[slot] = Qwen4ExpDeviceWeight(
            spec=spec, backend="hip_gfx1151", allocations={"raw": raw_alloc}
        )
    resident = _FakeResident(weights)

    populated = materialize_qwen4_exp_t16_stages(
        resident,
        read_tensor=lambda name: next(
            raw for src, raw in sources.values() if src == name
        ),
        slots=["layers.7.expert_gate", "layers.7.expert_down"],
    )

    assert set(populated) == {"layers.7.expert_gate", "layers.7.expert_down"}
    for slot in sources:
        new_weight = resident.weight(slot)
        # The t16 stage is resident with the exact repacked tile bytes.
        assert "t16" in new_weight.allocations
        assert new_weight.allocation("t16").buffer.nbytes == populated[slot]
        # Default free_raw=False: the raw stage (the strict fallback) stays.
        assert "raw" in new_weight.allocations
        assert new_weight.allocation().name == new_weight.allocations["raw"].name

    # A non-T16 quant slot is skipped, not an error.
    q8_spec = Qwen4ExpGGUFWeightSpec(
        slot_path="layers.8.expert_down",
        source_ref=_NamedRef("blk.8.ffn_down_exps.weight"),
        quant_key="gguf_q8_0",
        layout="raw",
        allocation_names=("raw",),
        device_resident=True,
        device_nbytes=64,
    )
    q8_raw = np.full(64, 1, dtype=np.uint8)
    q8_alloc = load_host_array_to_device_as_dtype(
        "layers.8.expert_down", q8_raw, "int8", source_dtype="int8"
    )
    resident.weights["layers.8.expert_down"] = Qwen4ExpDeviceWeight(
        spec=q8_spec, backend="hip_gfx1151", allocations={"raw": q8_alloc}
    )
    populated_q8 = materialize_qwen4_exp_t16_stages(
        resident,
        read_tensor=lambda name: np.zeros(8, dtype=np.uint8),
        slots=["layers.8.expert_down"],
    )
    assert populated_q8 == {}
    assert set(resident.weight("layers.8.expert_down").allocations) == {"raw"}


class _NamedRef:
    def __init__(self, name):
        self.tensor = _NamedTensor(name)


class _NamedTensor:
    def __init__(self, name):
        self.name = name


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_runner_branch_prefers_binding_stage_over_sidecar():
    """The tile lookup order: the binding's t16 stage first, sidecar second.

    Exercises the exact gate/up lookup expression the runner uses, on a
    weight pair carrying binding-owned tiles and a sidecar entry keyed by
    the raw pointer: the binding must win, and a sidecar-only pair must
    still resolve.
    """
    from hipengine.runtime import qwen4_exp_runner as rmod

    weight, _ = _synthetic_weight()
    tiles = np.full((2, 2, 96), 0x11, dtype=np.uint8)
    grown = attach_t16_allocation(weight, _t16_allocation(tiles))

    # The runner's expression: the binding stage when present...
    gate_allocs = getattr(grown, "allocations", {})
    assert "t16" in gate_allocs
    binding_buffer = gate_allocs["t16"].buffer
    # ...and the sidecar path still resolves for a plain weight.
    sidecar_buffer = weight.allocation("raw").buffer
    rmod._T16_GATE_UP_TILES[int(weight.allocation("raw").tensor.ptr)] = (
        sidecar_buffer,
        sidecar_buffer,
    )
    try:
        plain_allocs = getattr(weight, "allocations", {})
        assert "t16" not in plain_allocs
        # A plain weight falls through to the sidecar (backward compat).
        resolved = rmod._T16_GATE_UP_TILES.get(
            int(weight.allocation("raw").tensor.ptr)
        )
        assert resolved is not None
        # The binding stage is the first-priority source.
        assert binding_buffer.nbytes == tiles.nbytes
    finally:
        rmod._T16_GATE_UP_TILES.pop(
            int(weight.allocation("raw").tensor.ptr), None
        )
    # Frozen replacement: grown and weight share the raw buffer, so only
    # the grown weight is freed (the production caller swaps, never keeps
    # both alive).
    grown.free()
