"""RED contracts for WPF-H5Y exact tile-K-row BF16 activation planes."""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q5_k_f32_rocblas_prefill as q5_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _device,
    _edge_q5_weight,
)

_CANDIDATES = (
    (8, 4, "bf16", "tile_k_col"),
    (8, 12, "bf16", "row_major"),
    (16, 5, "bf16", "tile_k_col"),
    (12, 8, "bf16", "row_major"),
    (16, 5, "f32", "tile_k_col"),
    (8, 10, "f32", "tile_k_col"),
)
_ACTUAL_ROLES = (
    (3_072, 1_024, 4),
    (3_072, 12_288, 12),
    (6_144, 3_072, 5),
    (9_216, 3_072, 8),
    (3_072, 6_144, 5),
    (3_072, 9_216, 10),
)
_Q5_PRODUCTION_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch4"
    ),
    ("bf16", 3072, 12288): (
        "weight_major_row_major_activation_tile_k_row_coltile8_rowbatch12"
    ),
    ("bf16", 6144, 3072): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("bf16", 9216, 3072): (
        "weight_major_row_major_activation_tile_k_row_coltile12_rowbatch8"
    ),
    ("f32", 3072, 48): "coltile12_rowbatch4",
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 6144): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("f32", 3072, 9216): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch10"
    ),
}


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _suffix(col_tile: int, row_batch: int, output_dtype: str) -> str:
    return (
        f"coltile{col_tile}_rowbatch{row_batch}_"
        f"bf16_{output_dtype}_out"
    )


def _candidate_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
) -> tuple[str, str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    return (
        f"gguf_bf16_activation_pack_tile_k_row_{suffix}",
        "gguf_q5_k_f32_weight_ordered_weight_major_"
        f"{weight_layout}_activation_tile_k_row_{suffix}",
        "gguf_q5_k_f32_ordered_weight_major_"
        f"{weight_layout}_activation_tile_k_row_{suffix}",
    )


def _control_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    layout = "tile_k_col_" if weight_layout == "tile_k_col" else ""
    return (
        "gguf_q5_k_f32_weight_ordered_weight_major_"
        f"{layout}{suffix}",
        "gguf_q5_k_f32_ordered_weight_major_"
        f"{layout}{suffix}",
    )


def _activation_plane_shape(
    rows: int,
    in_features: int,
    row_batch: int,
) -> tuple[int, int, int]:
    return (math.ceil(rows / row_batch), in_features, math.ceil(row_batch / 8) * 8)


def _expected_activation_plane(
    x_bf16: np.ndarray,
    row_batch: int,
) -> np.ndarray:
    rows, in_features = x_bf16.shape
    row_groups, _, padded_rows = _activation_plane_shape(
        rows,
        in_features,
        row_batch,
    )
    expected = np.zeros(
        (row_groups, in_features, padded_rows),
        dtype=np.uint16,
    )
    for row_group in range(row_groups):
        row_base = row_group * row_batch
        live_rows = min(row_batch, rows - row_base)
        expected[row_group, :, :live_rows] = x_bf16[
            row_base : row_base + live_rows
        ].T
    return expected


def test_h5y_registry_workspace_scope_and_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    assert q5_f32._Q5_ACTIVATION_TILE_K_ROW_GEOMETRIES == _CANDIDATES
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == (
        _Q5_PRODUCTION_POLICY
    )

    actual_plane_nbytes = []
    for in_features, out_features, row_batch in _ACTUAL_ROLES:
        expected = (
            math.ceil(512 / row_batch)
            * in_features
            * (math.ceil(row_batch / 8) * 8)
            * 2
        )
        actual = q5_f32.q5_k_f32_activation_tile_k_row_nbytes(
            512,
            in_features,
            row_batch,
        )
        assert actual == expected
        assert q5_f32.q5_k_f32_activation_tile_k_row_workspace_nbytes(
            512,
            in_features,
            out_features,
            row_batch,
        ) == expected + in_features * out_features * 4
        actual_plane_nbytes.append(actual)
    assert max(actual_plane_nbytes) == 10_125_312
    assert 150_994_944 + max(actual_plane_nbytes) == 161_120_256

    for col_tile, row_batch, output_dtype, weight_layout in _CANDIDATES:
        producer_name, primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        control_primitive_name, control_composite_name = _control_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        producer = getattr(q5_f32, producer_name)
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        assert primitive is not getattr(q5_f32, control_primitive_name)
        assert composite is not getattr(q5_f32, control_composite_name)
        suffix = _suffix(col_tile, row_batch, output_dtype)
        entries = (
            (
                KernelKey(
                    "hip_gfx1100",
                    "activation_pack",
                    "bf16",
                    f"tile_k_row_{suffix}",
                ),
                producer,
            ),
            (
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "f32_weight",
                    "ordered_weight_major_"
                    f"{weight_layout}_activation_tile_k_row_{suffix}",
                ),
                primitive,
            ),
            (
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "gguf_q5_k",
                    "f32_ordered_weight_major_"
                    f"{weight_layout}_activation_tile_k_row_{suffix}",
                ),
                composite,
            ),
        )
        for key, function in entries:
            assert resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            ) is function
            assert not is_registered(
                KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
            )


def test_h5y_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, composite_name = _candidate_names(8, 4, "bf16", "tile_k_col")
    candidate = getattr(q5_f32, composite_name)

    def _unexpected_build(**_kwargs):
        raise AssertionError("invalid H5Y shape reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    with pytest.raises(ValueError, match="rows must be positive"):
        candidate(1, 2, 3, 4, 5, 0, 512, 48)
    with pytest.raises(ValueError, match="multiple of 256"):
        candidate(1, 2, 3, 4, 5, 17, 384, 48)
    with pytest.raises(ValueError, match="divisible by 8"):
        candidate(1, 2, 3, 4, 5, 17, 512, 50)


@pytest.mark.parametrize("rows", [17, 33, 512])
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h5y_activation_plane_and_outputs_match_current_bytes(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    in_features, out_features = 512, 48
    rng = np.random.default_rng(20260731 + 37 * rows)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    qweight = _edge_q5_weight(out_features, in_features)
    runtime = get_hip_runtime()
    library = q5_f32.build_gguf_q5_k_f32_rocblas_prefill(load=True)
    weight_plane_nbytes = q5_f32.q5_k_f32_ordered_workspace_nbytes(
        in_features,
        out_features,
    )
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        weight_f32_dev = malloc(weight_plane_nbytes, runtime=runtime)
        buffers.extend((x_dev, weight_dev, weight_f32_dev))

        for col_tile, row_batch, output_dtype, weight_layout in _CANDIDATES:
            host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
            expected = np.empty((rows, out_features), dtype=host_dtype)
            actual = np.empty_like(expected)
            expected_dev = malloc(expected.nbytes, runtime=runtime)
            actual_dev = malloc(actual.nbytes, runtime=runtime)
            plane_shape = _activation_plane_shape(
                rows,
                in_features,
                row_batch,
            )
            plane = np.empty(plane_shape, dtype=np.uint16)
            activation_dev = malloc(plane.nbytes, runtime=runtime)
            buffers.extend((expected_dev, actual_dev, activation_dev))

            _, control_name = _control_names(
                col_tile,
                row_batch,
                output_dtype,
                weight_layout,
            )
            control = getattr(q5_f32, control_name)
            producer_name, _, composite_name = _candidate_names(
                col_tile,
                row_batch,
                output_dtype,
                weight_layout,
            )
            control(
                x_dev.ptr,
                weight_dev.ptr,
                expected_dev.ptr,
                weight_f32_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            producer = getattr(q5_f32, producer_name)
            candidate = getattr(q5_f32, composite_name)
            candidate(
                x_dev.ptr,
                weight_dev.ptr,
                actual_dev.ptr,
                weight_f32_dev.ptr,
                activation_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            producer(
                x_dev.ptr,
                activation_dev.ptr,
                rows,
                in_features,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(expected),
                expected_dev,
                expected.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(actual),
                actual_dev,
                actual.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(plane),
                activation_dev,
                plane.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                plane,
                _expected_activation_plane(x_bf16, row_batch),
            )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
