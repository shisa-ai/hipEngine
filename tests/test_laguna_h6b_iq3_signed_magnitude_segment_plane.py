"""WPF-H6B exact active-IQ3 signed-magnitude segment-plane contract."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.quant import gguf as gguf_quant
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_gguf_iq3_active_expert_persistent import (
    HIP_AVAILABLE,
    _IN_FEATURES,
    _IQ3_BLOCK_BYTES,
    _NUM_EXPERTS,
    _OUT_FEATURES,
    _make_iq3_weight,
    _run_h5j_or_h5q,
)
from tests.test_gguf_iq_gemv import (
    _bf16_u16_to_f32,
    _f32_to_bf16_u16,
    _make_x,
    _selected_reference,
)

_RECORD_BYTES = 16
_GROUPS8 = _IN_FEATURES // 8
_PLANE_BYTES_PER_EXPERT = _OUT_FEATURES * _GROUPS8 * _RECORD_BYTES
_CURRENT_WORKSPACE_BYTES = 161_120_256
_PRODUCER_NAME = (
    "gguf_iq3_xxs_active_signed_magnitude_segment_plane_k1024_n3072_e256"
)
_CONSUMER_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_signed_magnitude_segment_plane_rowbatch8_"
    "bf16_bf16_out"
)
_PRODUCER_VARIANT = "active_signed_magnitude_segment_plane_k1024_n3072_e256"
_CONSUMER_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_signed_magnitude_segment_plane_rowbatch8_"
    "bf16_bf16_out"
)
_PLANE_QUANT = "iq3_signed_magnitude_segment_plane"
_H5Z_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_rowbatch8_bf16_bf16_out"
)
_H5Q_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "resident_rowbatch8_bf16_bf16_out"
)
_H5J_IQ4_VARIANT = (
    "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
)
_ACTIVE_EXPERT_ABI = "grouped_raw_iq_active_experts"
_RECORD_DTYPE = np.dtype(
    [
        ("scale", "<f4"),
        ("magnitude", "i1", (8,)),
        ("padding", "u1", (4,)),
    ],
    align=False,
)


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )


@pytest.fixture(scope="module")
def grouped_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return _module().build_gguf_iq_selected_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


def _producer_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(
        backend,
        "dequant",
        "gguf_iq3_xxs",
        _PRODUCER_VARIANT,
    )


def _consumer_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(
        backend,
        "moe_linear",
        _PLANE_QUANT,
        _CONSUMER_VARIANT,
    )


def test_h6b_registry_workspace_scope_and_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    module = _module()
    producer = getattr(module, _PRODUCER_NAME)
    consumer = getattr(module, _CONSUMER_NAME)
    plane_nbytes = module.gguf_iq3_signed_magnitude_segment_plane_nbytes

    assert module.GGUF_IQ3_SIGNED_MAGNITUDE_SEGMENT_RECORD_BYTES == _RECORD_BYTES
    assert _RECORD_DTYPE.itemsize == _RECORD_BYTES
    assert plane_nbytes(1, _IN_FEATURES, _OUT_FEATURES) == 6_291_456
    assert plane_nbytes(239, _IN_FEATURES, _OUT_FEATURES) == 1_503_657_984
    assert plane_nbytes(_NUM_EXPERTS, _IN_FEATURES, _OUT_FEATURES) == 1_610_612_736
    assert (
        _CURRENT_WORKSPACE_BYTES
        + plane_nbytes(_NUM_EXPERTS, _IN_FEATURES, _OUT_FEATURES)
        == 1_771_732_992
    )

    assert resolve(
        backend="hip_gfx1100",
        layer="dequant",
        quant="gguf_iq3_xxs",
        variant=_PRODUCER_VARIANT,
    ) is producer
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant=_PLANE_QUANT,
        variant=_CONSUMER_VARIANT,
    ) is consumer

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_producer_key("hip_gfx1151"))
    assert not is_registered(_consumer_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {
        "gguf_iq3_xxs": _H5Z_VARIANT,
        "gguf_iq4_xs": _H5J_IQ4_VARIANT,
    }
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {
        _H5Q_VARIANT: _ACTIVE_EXPERT_ABI,
        _H5Z_VARIANT: _ACTIVE_EXPERT_ABI,
    }


def test_h6b_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    producer = getattr(module, _PRODUCER_NAME)
    consumer = getattr(module, _CONSUMER_NAME)
    plane_nbytes = module.gguf_iq3_signed_magnitude_segment_plane_nbytes
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H6B shape reached the HIP loader")

    monkeypatch.setattr(module, "build_gguf_iq_selected_prefill", fail_if_loaded)
    producer_common = dict(
        active_expert_capacity=4,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=_NUM_EXPERTS,
    )
    consumer_common = dict(
        compact_rows=9,
        active_expert_capacity=4,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=_NUM_EXPERTS,
    )
    invalid_shapes = (
        ({"in_features": 768}, "exactly 1024"),
        ({"out_features": 1024}, "exactly 3072"),
        ({"num_experts": 255}, "exactly 256"),
        ({"active_expert_capacity": 0}, "capacity must be positive"),
        ({"active_expert_capacity": 257}, "capacity must be at most 256"),
    )
    for changed, message in invalid_shapes:
        with pytest.raises(ValueError, match=message):
            producer(1, 2, 3, 4, **(producer_common | changed))
        with pytest.raises(ValueError, match=message):
            consumer(1, 2, 3, 4, 5, 6, **(consumer_common | changed))
        with pytest.raises(ValueError, match=message):
            plane_nbytes(
                changed.get("active_expert_capacity", 4),
                changed.get("in_features", _IN_FEATURES),
                changed.get("out_features", _OUT_FEATURES),
                num_experts=changed.get("num_experts", _NUM_EXPERTS),
            )
    with pytest.raises(ValueError, match="compact_rows must be positive"):
        consumer(1, 2, 3, 4, 5, 6, **(consumer_common | {"compact_rows": 0}))
    assert load_attempts == 0


def _expected_segment_records(
    qweight: np.ndarray,
    active_experts: tuple[int, ...],
) -> np.ndarray:
    selected = np.ascontiguousarray(qweight[np.asarray(active_experts, dtype=np.int64)])
    blocks = selected.reshape(
        len(active_experts),
        _OUT_FEATURES,
        _IN_FEATURES // 256,
        _IQ3_BLOCK_BYTES,
    )
    expected = np.zeros(
        (len(active_experts), _OUT_FEATURES, _GROUPS8),
        dtype=_RECORD_DTYPE,
    )
    bit_offsets = np.arange(8, dtype=np.uint8)
    for group8_linear in range(_GROUPS8):
        block_idx = group8_linear >> 5
        group8 = group8_linear & 31
        group32 = group8 >> 2
        local8 = group8 & 3
        block = blocks[:, :, block_idx, :]
        aux = (
            np.ascontiguousarray(block[:, :, 66 + 4 * group32 : 70 + 4 * group32])
            .view("<u4")
            .reshape(len(active_experts), _OUT_FEATURES)
        )
        selector = (aux >> np.uint32(7 * local8)) & np.uint32(127)
        signs = gguf_quant._KSIGNS_IQ2XS[selector]
        grid1 = gguf_quant._IQ3_XXS_GRID_BYTES[
            block[:, :, 2 + group32 * 8 + 2 * local8]
        ]
        grid2 = gguf_quant._IQ3_XXS_GRID_BYTES[
            block[:, :, 2 + group32 * 8 + 2 * local8 + 1]
        ]
        magnitude = np.concatenate((grid1, grid2), axis=-1).astype(np.int16)
        negative = ((signs[..., None] >> bit_offsets) & np.uint8(1)).astype(bool)
        signed_magnitude = np.where(negative, -magnitude, magnitude)
        assert np.max(np.abs(signed_magnitude)) <= np.iinfo(np.int8).max

        base_scale = (
            np.ascontiguousarray(block[:, :, :2])
            .view("<f2")
            .reshape(len(active_experts), _OUT_FEATURES)
            .astype(np.float32)
        )
        group_scale = (
            np.float32(0.5) + (aux >> np.uint32(28)).astype(np.float32)
        )
        scale = (base_scale * group_scale).astype(np.float32)
        scale = (scale * np.float32(0.5)).astype(np.float32)
        expected["scale"][:, :, group8_linear] = scale
        expected["magnitude"][:, :, group8_linear, :] = signed_magnitude.astype(
            np.int8
        )
    return expected


def _device_buffer(array: np.ndarray, buffers: list[Any], runtime):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(
        buffer,
        host_array_ptr(contiguous),
        contiguous.nbytes,
        runtime=runtime,
    )
    buffers.append(buffer)
    return buffer


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h6b_complete_reordered_segment_records_match_current_iq3_decode(
    grouped_library,
) -> None:
    from hipengine.core.hip import get_hip_runtime

    module = _module()
    active_experts = (7, 0, 11, 3)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active[: len(active_experts)] = active_experts
    active_count = np.asarray([len(active_experts)], dtype=np.int64)
    qweight = _make_iq3_weight(max(active_experts) + 1)
    expected = _expected_segment_records(qweight, active_experts)
    runtime = get_hip_runtime()
    before = memory_stats()
    buffers: list[Any] = []
    try:
        active_device = _device_buffer(active, buffers, runtime)
        count_device = _device_buffer(active_count, buffers, runtime)
        weight_device = _device_buffer(qweight, buffers, runtime)
        plane_nbytes = module.gguf_iq3_signed_magnitude_segment_plane_nbytes(
            _NUM_EXPERTS,
            _IN_FEATURES,
            _OUT_FEATURES,
        )
        plane_device = malloc(plane_nbytes, runtime=runtime)
        buffers.append(plane_device)
        producer = getattr(module, _PRODUCER_NAME)
        producer(
            active_device.ptr,
            count_device.ptr,
            weight_device.ptr,
            plane_device.ptr,
            active_expert_capacity=_NUM_EXPERTS,
            in_features=_IN_FEATURES,
            out_features=_OUT_FEATURES,
            num_experts=_NUM_EXPERTS,
            library=grouped_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = np.empty(expected.shape, dtype=_RECORD_DTYPE)
        copy_device_to_host(
            host_array_ptr(actual),
            DeviceBuffer(plane_device.ptr, actual.nbytes),
            actual.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
        assert np.all(actual["padding"] == 0)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


def _metadata(
    active_experts: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    pattern = (1, 2, 7, 8, 9)
    for expert in active_experts:
        counts[expert] = pattern[expert % len(pattern)]
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active[: len(active_experts)] = active_experts
    active_count = np.asarray([len(active_experts)], dtype=np.int64)
    selected = np.repeat(np.arange(_NUM_EXPERTS, dtype=np.int64), counts)
    return starts, active, active_count, selected


def _run_h6b(
    module,
    library,
    *,
    x_bf16: np.ndarray,
    starts: np.ndarray,
    active: np.ndarray,
    active_count: np.ndarray,
    qweight: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    before = memory_stats()
    buffers: list[Any] = []
    actual = np.ascontiguousarray(initial.copy())
    try:
        x_device = _device_buffer(x_bf16, buffers, runtime)
        starts_device = _device_buffer(starts, buffers, runtime)
        active_device = _device_buffer(active, buffers, runtime)
        count_device = _device_buffer(active_count, buffers, runtime)
        weight_device = _device_buffer(qweight, buffers, runtime)
        out_device = _device_buffer(actual, buffers, runtime)
        plane_device = malloc(
            module.gguf_iq3_signed_magnitude_segment_plane_nbytes(
                _NUM_EXPERTS,
                _IN_FEATURES,
                _OUT_FEATURES,
            ),
            runtime=runtime,
        )
        buffers.append(plane_device)
        producer = getattr(module, _PRODUCER_NAME)
        consumer = getattr(module, _CONSUMER_NAME)
        producer(
            active_device.ptr,
            count_device.ptr,
            weight_device.ptr,
            plane_device.ptr,
            active_expert_capacity=_NUM_EXPERTS,
            in_features=_IN_FEATURES,
            out_features=_OUT_FEATURES,
            num_experts=_NUM_EXPERTS,
            library=library,
            runtime=runtime,
        )
        consumer(
            x_device.ptr,
            starts_device.ptr,
            active_device.ptr,
            count_device.ptr,
            plane_device.ptr,
            out_device.ptr,
            compact_rows=x_bf16.shape[0],
            active_expert_capacity=_NUM_EXPERTS,
            in_features=_IN_FEATURES,
            out_features=_OUT_FEATURES,
            num_experts=_NUM_EXPERTS,
            library=library,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual),
            out_device,
            actual.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
    return actual


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h6b_complete_outputs_match_h5z_and_cpu_at_p64_boundary_and_tails(
    grouped_library,
) -> None:
    module = _module()
    control = module.GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS[256]
    for active_expert_count in (64, 65):
        active_experts = tuple(reversed(range(active_expert_count)))
        starts, active, active_count, selected = _metadata(active_experts)
        compact_rows = int(starts[-1])
        x_bf16 = _f32_to_bf16_u16(_make_x(compact_rows, _IN_FEATURES))
        qweight = _make_iq3_weight(active_expert_count)
        initial = np.full((compact_rows, _OUT_FEATURES), 0x7FC0, dtype=np.uint16)
        expected = _run_h5j_or_h5q(
            control,
            grouped_library,
            x_bf16=x_bf16,
            starts=starts,
            active_experts=active,
            active_count=active_count,
            qweight=qweight,
            initial=initial,
            persistent=True,
        )
        actual = _run_h6b(
            module,
            grouped_library,
            x_bf16=x_bf16,
            starts=starts,
            active=active,
            active_count=active_count,
            qweight=qweight,
            initial=initial,
        )
        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=f"active_expert_count={active_expert_count}",
        )

        sample_rows = np.unique(
            np.asarray([0, 7, 8, compact_rows // 2, compact_rows - 1])
        )
        sample_cols = np.asarray([0, 255, 256, 1535, 3071])
        cpu = _selected_reference(
            x_bf16[sample_rows],
            selected[sample_rows],
            qweight[:, sample_cols, :],
            GGMLQuantizationType.IQ3_XXS,
        )
        np.testing.assert_array_equal(actual[np.ix_(sample_rows, sample_cols)], cpu)
        assert np.isfinite(_bf16_u16_to_f32(cpu)).all()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h6b_empty_active_list_preserves_output_and_recovers_allocations(
    grouped_library,
) -> None:
    module = _module()
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active_count = np.zeros(1, dtype=np.int64)
    x_bf16 = _f32_to_bf16_u16(_make_x(1, _IN_FEATURES))
    qweight = _make_iq3_weight(1)
    initial = np.full((1, _OUT_FEATURES), 0x3F80, dtype=np.uint16)
    expected = _run_h5j_or_h5q(
        module.GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS[256],
        grouped_library,
        x_bf16=x_bf16,
        starts=starts,
        active_experts=active,
        active_count=active_count,
        qweight=qweight,
        initial=initial,
        persistent=True,
    )
    actual = _run_h6b(
        module,
        grouped_library,
        x_bf16=x_bf16,
        starts=starts,
        active=active,
        active_count=active_count,
        qweight=qweight,
        initial=initial,
    )
    np.testing.assert_array_equal(expected, initial)
    np.testing.assert_array_equal(actual, initial)
